"""Shared runtime configuration and artifact paths for the KDD pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "churn.csv"
CLEAN_PATH = PROCESSED_DIR / "churn_clean.csv"
CLUSTER_MATRIX = PROCESSED_DIR / "churn_clustering_matrix.csv"
TRANSACTIONS_PATH = PROCESSED_DIR / "churn_transactions.csv"
OHE_TRANSACTIONS_PATH = PROCESSED_DIR / "churn_ohe_transactions.csv"
CLUSTERED_PATH = PROCESSED_DIR / "churn_clustered.csv"
DBSCAN_OUTLIERS_PATH = PROCESSED_DIR / "dbscan_outlier_indices.npy"

TOP_RULES_PATH = OUTPUTS_DIR / "ph3_top_association_rules.csv"
ALL_RULES_PATH = OUTPUTS_DIR / "ph3_all_association_rules.csv"
ANOMALY_REPORT_PATH = OUTPUTS_DIR / "ph4_anomaly_report.csv"
HIGH_CONFIDENCE_ANOMALIES_PATH = (
    OUTPUTS_DIR / "ph4_high_confidence_anomalies.csv"
)
EVALUATION_METRICS_PATH = OUTPUTS_DIR / "evaluation_metrics.csv"

EVALUATION_METRIC_COLUMNS = [
    "Phase",
    "Metric",
    "Value",
    "Unit",
    "Definition",
    "Source",
]


try:
    from IPython.display import display
except ImportError:

    def display(value: object) -> None:
        """Print tables when IPython's rich display is unavailable."""

        print(value)


def configure_runtime() -> None:
    """Prepare output directories and UTF-8 console streams."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def require_files(paths: Iterable[Path], phase_name: str) -> None:
    """Fail early with a useful prerequisite message for standalone phases."""

    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"{phase_name} is missing prerequisite artifacts:\n{formatted}\n"
            "Run the required earlier phase(s), or run `python -m src.pipeline`."
        )


def upsert_evaluation_metrics(rows: Iterable[dict]) -> "object":
    """Persist deterministic, source-labelled metrics shared by all phases.

    Existing rows with the same ``(Phase, Metric)`` key are replaced so each
    notebook can be rerun independently without duplicating appendix entries.
    The returned object is a pandas DataFrame; pandas is imported lazily to keep
    this path helper lightweight for callers that do not write metrics.
    """

    import pandas as pd

    new_rows = pd.DataFrame(list(rows))
    missing_columns = [
        column
        for column in EVALUATION_METRIC_COLUMNS
        if column not in new_rows.columns
    ]
    if missing_columns:
        raise ValueError(
            "Evaluation metric rows are missing columns: "
            + ", ".join(missing_columns)
        )

    new_rows = new_rows[EVALUATION_METRIC_COLUMNS].copy()
    if EVALUATION_METRICS_PATH.is_file():
        existing = pd.read_csv(EVALUATION_METRICS_PATH)
        for column in EVALUATION_METRIC_COLUMNS:
            if column not in existing.columns:
                existing[column] = pd.NA
        existing = existing[EVALUATION_METRIC_COLUMNS]

        replacement_keys = set(
            zip(new_rows["Phase"], new_rows["Metric"], strict=False)
        )
        keep_mask = [
            (phase, metric) not in replacement_keys
            for phase, metric in zip(
                existing["Phase"], existing["Metric"], strict=False
            )
        ]
        combined = pd.concat(
            [existing.loc[keep_mask], new_rows], ignore_index=True
        )
    else:
        combined = new_rows

    combined = (
        combined.drop_duplicates(["Phase", "Metric"], keep="last")
        .assign(
            _phase_order=lambda frame: frame["Phase"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .fillna("999")
            .astype(int)
        )
        .sort_values(["_phase_order", "Metric"], kind="stable")
        .drop(columns="_phase_order")
        .reset_index(drop=True)
    )
    EVALUATION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(EVALUATION_METRICS_PATH, index=False)
    return combined
