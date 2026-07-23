"""Validate notebook exports against the generated dashboard cache.

Run from the repository root or ``visualization/``:

    python visualization/validate_consistency.py

The checks deliberately use ``Source_RowNumber`` as the customer grain and
the executed notebook metric ledger as the aggregate source of truth.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / "dashboard_data"


def _parse_frozenset(value: str) -> frozenset[str]:
    text = str(value).strip()
    if text.startswith("frozenset(") and text.endswith(")"):
        text = text[len("frozenset("):-1]
    return frozenset(ast.literal_eval(text))


def _ledger_value(ledger: pd.DataFrame, phase: str, metric: str) -> float:
    rows = ledger[(ledger["Phase"] == phase) & (ledger["Metric"] == metric)]
    assert len(rows) == 1, f"Expected one ledger row for {phase} / {metric}"
    return float(rows.iloc[0]["Value"])


def validate_consistency(root: Path = ROOT) -> int:
    cache = root / "visualization" / "dashboard_data"
    clustered = pd.read_csv(root / "data/processed/churn_clustered.csv")
    anomaly = pd.read_csv(root / "outputs/ph4_anomaly_report.csv")
    high_confidence = pd.read_csv(
        root / "outputs/ph4_high_confidence_anomalies.csv"
    )
    top_rules = pd.read_csv(root / "outputs/ph3_top_association_rules.csv")
    all_rules = pd.read_csv(root / "outputs/ph3_all_association_rules.csv")
    ledger = pd.read_csv(root / "outputs/evaluation_metrics.csv")
    records = pd.read_csv(cache / "records.csv")
    metrics = json.loads((cache / "metrics.json").read_text(encoding="utf-8"))
    rule_cache = json.loads((cache / "rules.json").read_text(encoding="utf-8"))

    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        assert bool(condition), message
        checks += 1

    expected_rows = int(_ledger_value(ledger, "Phase 1", "Raw rows audited"))
    check(
        len(clustered) == len(anomaly) == len(records) == expected_rows,
        "Customer row counts disagree",
    )

    for name, frame in (
        ("clustered", clustered),
        ("anomaly", anomaly),
        ("records", records),
    ):
        check(frame["Source_RowNumber"].notna().all(), f"{name} has null source keys")
        check(frame["Source_RowNumber"].is_unique, f"{name} source keys are not unique")

    check(
        clustered["Source_RowNumber"].equals(anomaly["Source_RowNumber"])
        and clustered["Source_RowNumber"].equals(records["Source_RowNumber"]),
        "Customer artifacts are not in the same source-key order",
    )

    pd.testing.assert_frame_equal(
        clustered,
        records[clustered.columns],
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    checks += 1

    anomaly_columns = [
        "IQR_flag", "ZScore_flag", "IF_flag", "MAHA_flag", "LOF_flag",
        "DBSCAN_flag", "UNI_flag", "MV_flag", "Composite_Anomaly_Score",
        "UniMV_Segment", "Anomaly_Class", "Anomaly_Evidence",
        "Recommended_Action", "Max_Abs_Z", "HighBal_Churn_flag",
    ]
    pd.testing.assert_frame_equal(
        anomaly[anomaly_columns],
        records[anomaly_columns],
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    checks += 1
    check(
        np.allclose(
            records["IF_score"],
            anomaly["IF_anomaly_score"].round(4),
            rtol=0,
            atol=5e-5,
        ),
        "Rounded Isolation Forest scores drifted",
    )
    check(
        np.allclose(
            records["MAHA_dist2"],
            anomaly["MAHA_dist2"].round(3),
            rtol=0,
            atol=5e-4,
        ),
        "Rounded Mahalanobis distances drifted",
    )
    check(
        np.array_equal(
            (records["DBSCAN_Label"] == -1).astype(int),
            records["DBSCAN_flag"].astype(int),
        ),
        "Dashboard DBSCAN labels disagree with Phase 4 flags",
    )

    expected_high_confidence = anomaly[
        anomaly["Composite_Anomaly_Score"] >= 2
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        expected_high_confidence,
        high_confidence.reset_index(drop=True),
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    checks += 1

    age_band = pd.cut(
        clustered["Age"],
        [0, 30, 45, 60, 100],
        labels=["Age 18-30", "Age 31-45", "Age 46-60", "Age 61+"],
    )
    balance_band = pd.cut(
        clustered["Balance"],
        [-1, 0, 100_000, clustered["Balance"].max()],
        labels=["Zero balance", "Balance 0-100K", "Balance above 100K"],
    )
    check(
        age_band.astype(object).equals(records["Age_Band"].astype(object)),
        "Age bands drifted from Phase 1/3 boundaries",
    )
    check(
        balance_band.astype(object).equals(records["Balance_Band"].astype(object)),
        "Balance bands drifted from Phase 1/3 boundaries",
    )
    check(
        clustered["IsActiveMember"].map({1: "Active", 0: "Inactive"}).equals(
            records["Active_Status"]
        ),
        "Active status labels drifted",
    )
    check(
        clustered["Exited"].map({1: "Churned", 0: "Retained"}).equals(
            records["Churn_Status"]
        ),
        "Churn status labels drifted",
    )

    kpi = metrics["kpi"]
    check(kpi["n_customers"] == expected_rows, "Dashboard customer KPI drifted")
    check(kpi["n_churned"] == int(clustered["Exited"].sum()), "Churn count drifted")
    check(
        np.isclose(kpi["churn_rate"], clustered["Exited"].mean() * 100, atol=0.005),
        "Churn rate drifted",
    )
    check(
        kpi["n_clusters"] == int(_ledger_value(ledger, "Phase 2", "Selected K")),
        "Selected K drifted",
    )
    check(
        kpi["dbscan_noise"]
        == int(_ledger_value(ledger, "Phase 2", "DBSCAN noise records")),
        "DBSCAN count drifted",
    )
    check(
        kpi["n_documented_churn_rules"]
        == int(_ledger_value(ledger, "Phase 3", "Documented churn rules")),
        "Documented-rule count drifted",
    )
    check(
        kpi["n_rules_generated"]
        == int(_ledger_value(ledger, "Phase 3", "Rules generated")),
        "Generated-rule count drifted",
    )
    check(
        kpi["n_rules_confidence"]
        == int(_ledger_value(ledger, "Phase 3", "Rules after confidence filter")),
        "Confidence-filter count drifted",
    )
    check(
        kpi["n_rules_filtered"]
        == int(_ledger_value(
            ledger, "Phase 3", "Rules after confidence and lift filters"
        )),
        "Final filtered-rule count drifted",
    )
    check(
        kpi["n_nonredundant_churn_rules"]
        == int(_ledger_value(
            ledger, "Phase 3", "Non-redundant churn rules retained"
        )),
        "Non-redundant rule count drifted",
    )
    check(
        kpi["n_flagged"]
        == int(_ledger_value(
            ledger, "Phase 4", "Anomaly candidates before corroboration"
        )),
        "Anomaly candidate count drifted",
    )

    for cluster_id, group in clustered.groupby("Cluster"):
        profile = metrics["clusters"][str(int(cluster_id))]
        check(profile["n"] == len(group), f"C{cluster_id} size drifted")
        check(
            profile["name"] == group["Cluster_Name"].iloc[0],
            f"C{cluster_id} systematic name drifted",
        )
        check(
            profile["alias"] == group["Persona_Alias"].iloc[0],
            f"C{cluster_id} persona alias drifted",
        )
        check(
            np.isclose(
                profile["churn"],
                round(group["Exited"].mean() * 100, 1),
            ),
            f"C{cluster_id} churn profile drifted",
        )

    check(rule_cache["n_total_rules"] == len(all_rules), "Rule cache total drifted")
    check(
        len(rule_cache["top10"]) == len(top_rules),
        "Documented rule cache count drifted",
    )
    for index, (cached, (_, exported)) in enumerate(
        zip(rule_cache["top10"], top_rules.iterrows()), start=1
    ):
        antecedent = sorted(_parse_frozenset(exported["antecedents"]))
        check(cached["rank"] == index, f"Rule {index} rank drifted")
        check(cached["if_raw"] == antecedent, f"Rule {index} antecedent drifted")
        check(
            np.isclose(cached["support_pct"], exported["Support (%)"]),
            f"Rule {index} support drifted",
        )
        check(
            np.isclose(cached["confidence_pct"], exported["Confidence (%)"]),
            f"Rule {index} confidence drifted",
        )
        check(np.isclose(cached["lift"], exported["Lift"]), f"Rule {index} lift drifted")
        check(
            cached["customers"] == int(round(exported["support"] * expected_rows)),
            f"Rule {index} customer count drifted",
        )
        check(
            cached["commentary"] == exported["Business Commentary"],
            f"Rule {index} commentary drifted",
        )

    all_rules = all_rules.copy()
    all_rules["A"] = all_rules["antecedents"].map(_parse_frozenset)
    all_rules["C"] = all_rules["consequents"].map(_parse_frozenset)
    single_churn = all_rules[
        (all_rules["C"] == frozenset({"Churn_Status_Churned"}))
        & (~all_rules["A"].map(lambda items: "Churn_Status_Churned" in items))
    ]
    check(
        rule_cache["n_single_churn_consequent_rules"] == len(single_churn),
        "Single-churn-consequent rule count drifted",
    )

    check(
        sum(row["n"] for row in metrics["anomaly_classes"])
        == int((anomaly["Composite_Anomaly_Score"] >= 1).sum()),
        "Anomaly-class totals drifted",
    )
    check(
        metrics["cross_ref"]["if_dbscan_overlap"]
        == int(((anomaly["IF_flag"] == 1) & (anomaly["DBSCAN_flag"] == 1)).sum()),
        "IF/DBSCAN overlap drifted",
    )

    print(
        f"PASS: {checks} notebook-to-visualization checks "
        f"({expected_rows:,} customers, {len(top_rules)} documented rules, "
        f"{kpi['n_flagged']:,} anomaly candidates)."
    )
    return checks


if __name__ == "__main__":
    validate_consistency()
