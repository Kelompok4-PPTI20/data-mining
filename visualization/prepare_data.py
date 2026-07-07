"""
Phase 5 — Dashboard data preparation
====================================
Assembles every number the dashboard shows into a small cache folder
(`visualization/dashboard_data/`) so the Dash app itself does ZERO mining at
runtime and every interaction stays well under the 100 ms rubric budget.

Inputs (produced by notebooks/notebook.ipynb, Phases 1-4):
    data/raw/churn.csv                       raw dataset (10,000 x 14)
    data/processed/churn_clustered.csv       K-Means K=3 labels + persona names
    data/processed/churn_clustering_matrix.csv  Path-A scaled matrix
    data/processed/dbscan_outlier_indices.npy   Phase-2 DBSCAN noise indices
    outputs/ph3_top_association_rules.csv    Phase-3 deliverable rule table
    outputs/ph3_all_association_rules.csv    all 520 filtered rules
    outputs/ph4_anomaly_report.csv           per-record anomaly flags + classes

Outputs:
    visualization/dashboard_data/records.csv   one tidy row per customer
    visualization/dashboard_data/metrics.json  every aggregate the app displays
    visualization/dashboard_data/rules.json    top rules + business commentary

Run once (from the visualization/ folder):  python prepare_data.py
Heavy steps are checkpointed, so if the run is interrupted just run it again.
"""

import ast
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (adjusted_rand_score, cohen_kappa_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "dashboard_data"
OUT.mkdir(exist_ok=True)
CK = OUT / "_checkpoints"          # heavy steps are checkpointed so the script
CK.mkdir(exist_ok=True)            # can be interrupted and simply re-run

print("Loading Phase 1-4 artifacts ...", flush=True)
df = pd.read_csv(ROOT / "data/processed/churn_clustered.csv")          # base + Cluster
anom = pd.read_csv(ROOT / "outputs/ph4_anomaly_report.csv")            # per-record flags
matrix = pd.read_csv(ROOT / "data/processed/churn_clustering_matrix.csv")
dbscan_noise_idx = np.load(ROOT / "data/processed/dbscan_outlier_indices.npy")
top_rules = pd.read_csv(ROOT / "outputs/ph3_top_association_rules.csv")
all_rules = pd.read_csv(ROOT / "outputs/ph3_all_association_rules.csv")

assert len(df) == len(anom) == len(matrix) == 10_000, "row alignment broken"

X_cl = matrix.drop(columns=["Geography", "Gender", "Exited"])
baseline = df["Exited"].mean()

# ---------------------------------------------------------------------------
# 1. Recompute the pieces the notebook did not persist
# ---------------------------------------------------------------------------
print("PCA projection of the clustering space ...", flush=True)
if (CK / "pca.npz").exists():
    z = np.load(CK / "pca.npz", allow_pickle=True)
    coords, evr = z["coords"], z["evr"]
    load = pd.DataFrame(z["load"], index=X_cl.columns, columns=["PC1", "PC2"])
else:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_cl)
    evr = pca.explained_variance_ratio_ * 100
    load = pd.DataFrame(pca.components_.T, index=X_cl.columns, columns=["PC1", "PC2"])
    # Fix sign convention so the axes read like the notebook figure
    # (PC1: + = product depth / - = balance ; PC2: + = salary/tenure mix)
    if load.loc["NumOfProducts", "PC1"] < 0:
        coords[:, 0] *= -1
        load["PC1"] *= -1
    if load.loc["EstimatedSalary", "PC2"] < 0:
        coords[:, 1] *= -1
        load["PC2"] *= -1
    np.savez(CK / "pca.npz", coords=coords, evr=evr, load=load.values)

print("DBSCAN (eps=1.25, min_samples=10) ...", flush=True)
if (CK / "dbscan.npy").exists():
    db_labels = np.load(CK / "dbscan.npy")
else:
    db = DBSCAN(eps=1.25, min_samples=10, metric="euclidean", n_jobs=-1)
    db_labels = db.fit_predict(X_cl)
    np.save(CK / "dbscan.npy", db_labels)
n_noise = int((db_labels == -1).sum())
print(f"  noise points: {n_noise} (notebook: 554)")
saved_noise = set(dbscan_noise_idx.tolist())
recomputed_noise = set(np.where(db_labels == -1)[0].tolist())
print(f"  overlap with saved noise indices: {len(saved_noise & recomputed_noise)}/{len(saved_noise)}", flush=True)

print("Ward hierarchical (K=3) ...", flush=True)
if (CK / "ward.npy").exists():
    ward_labels = np.load(CK / "ward.npy")
else:
    try:
        ward = AgglomerativeClustering(n_clusters=3, linkage="ward")
        ward_labels = ward.fit_predict(X_cl)
    except MemoryError:
        print("  ! MemoryError - Ward skipped, dashboard hides that panel")
        ward_labels = np.full(len(X_cl), -99)
    np.save(CK / "ward.npy", ward_labels)

print("Elbow + silhouette sweep (K=2..12) ...", flush=True)
K_RANGE = list(range(2, 13))
sweep_path = CK / "sweep.json"
sweep = json.loads(sweep_path.read_text()) if sweep_path.exists() else {}
for k in K_RANGE:
    if str(k) in sweep:
        continue
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
    km.fit(X_cl)
    sweep[str(k)] = [float(km.inertia_), float(silhouette_score(X_cl, km.labels_))]
    sweep_path.write_text(json.dumps(sweep))
    print(f"  K={k:2d}  inertia={sweep[str(k)][0]:,.0f}  sil={sweep[str(k)][1]:.4f}", flush=True)
inertias = [sweep[str(k)][0] for k in K_RANGE]
sils = [sweep[str(k)][1] for k in K_RANGE]

sil_ward = float(silhouette_score(X_cl, ward_labels)) if ward_labels[0] != -99 else None
ari = float(adjusted_rand_score(df["Cluster"], ward_labels)) if sil_ward else None
nmi = float(normalized_mutual_info_score(df["Cluster"], ward_labels)) if sil_ward else None
print(f"  Ward sil={sil_ward}  ARI={ari}  NMI={nmi}  (notebook: 0.1279 / 0.7461 / 0.7014)", flush=True)

# ---------------------------------------------------------------------------
# 2. Tidy per-customer table for the app
# ---------------------------------------------------------------------------
print("Building records.csv ...", flush=True)
rec = df.copy()
rec["PC1"], rec["PC2"] = coords[:, 0].round(4), coords[:, 1].round(4)
rec["DBSCAN_Label"] = db_labels
rec["Ward_Label"] = ward_labels
for c in ["IQR_flag", "ZScore_flag", "IF_flag", "MAHA_flag", "LOF_flag",
          "DBSCAN_flag", "UNI_flag", "MV_flag", "Composite_Anomaly_Score",
          "UniMV_Segment", "Anomaly_Class", "Max_Abs_Z", "HighBal_Churn_flag"]:
    rec[c] = anom[c]
rec["IF_score"] = anom["IF_anomaly_score"].round(4)
rec["MAHA_dist2"] = anom["MAHA_dist2"].round(3)

# Phase-1 / Phase-3 bands (same boundaries as the notebook Path-B binning)
rec["Age_Band"] = pd.cut(rec["Age"], [0, 30, 45, 60, 100],
                         labels=["Young adult (18-30)", "Middle-aged (31-45)",
                                 "Senior (46-60)", "Elderly (60+)"])
# Balance bands anchored on the EUR 100,000 EU deposit-guarantee ceiling
# (Directive 2014/49/EU) — same boundaries as the notebook Path-B binning:
#   https://eur-lex.europa.eu/legal-content/EN/LSU/?uri=celex:32014L0049
rec["Balance_Band"] = pd.cut(rec["Balance"], [-1, 0, 100_000, rec["Balance"].max()],
                             labels=["Zero balance", "Insured (0-100K)",
                                     "Above ceiling (>100K)"])
rec["Active_Status"] = rec["IsActiveMember"].map({1: "Active", 0: "Inactive"})
rec["Churn_Status"] = rec["Exited"].map({1: "Churned", 0: "Retained"})
rec.to_csv(OUT / "records.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Aggregates -> metrics.json
# ---------------------------------------------------------------------------
print("Computing aggregates ...", flush=True)
M = {}

M["kpi"] = {
    "n_customers": 10_000,
    "n_features_raw": 14,
    "churn_rate": round(baseline * 100, 2),
    "n_churned": int(df["Exited"].sum()),
    "n_clusters": 3,
    "n_churn_rules": 17,   # under the DGS-anchored balance bands (was 13 with 50K/125K)
    "n_rules_total": int(len(all_rules)),
    "top_rule_lift": round(float(top_rules["Lift"].max()), 2),
    "top_rule_conf": round(float(top_rules["Confidence (%)"].max()), 1),
    "n_flagged": int((anom["Composite_Anomaly_Score"] >= 1).sum()),
    "n_risk_signals": int(anom["Anomaly_Class"].str.startswith("C").sum()),
    "dbscan_noise": n_noise,
    "dbscan_noise_churn": round(float(df.loc[db_labels == -1, "Exited"].mean() * 100), 1),
}

def churn_by(col, order=None):
    g = rec.groupby(col, observed=True)["Exited"].agg(["mean", "sum", "count"])
    if order is not None:
        g = g.reindex(order)
    return {
        "labels": [str(i) for i in g.index],
        "churn_pct": (g["mean"] * 100).round(2).tolist(),
        "churned": g["sum"].astype(int).tolist(),
        "total": g["count"].astype(int).tolist(),
    }

M["churn_by"] = {
    "Geography": churn_by("Geography", ["France", "Germany", "Spain"]),
    "Gender": churn_by("Gender"),
    "Age_Band": churn_by("Age_Band"),
    "Active_Status": churn_by("Active_Status", ["Active", "Inactive"]),
    "NumOfProducts": churn_by("NumOfProducts"),
    "Balance_Band": churn_by("Balance_Band"),
    "Tenure_Band": churn_by(pd.cut(rec["Tenure"], [-1, 2, 5, 10],
                                   labels=["New (0-2y)", "Established (3-5y)", "Loyal (6-10y)"]).rename("Tenure_Band")),
}

# --- Phase 1: feature selection, both lenses --------------------------------
print("Feature selection lenses (correlation + entropy) ...", flush=True)
base_cols = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
             "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
enc = rec[base_cols].copy()
for c in ["Geography", "Gender"]:
    enc[c] = LabelEncoder().fit_transform(enc[c])
mi = mutual_info_classif(enc, rec["Exited"], random_state=RANDOM_STATE)
mi_map = dict(zip(base_cols, mi))

numeric = rec[[c for c in base_cols if c not in ("Geography", "Gender")] + ["Exited"]]
corr_map = numeric.corr()["Exited"].drop("Exited").abs().to_dict()

def shannon_entropy(s):
    p = s.value_counts(normalize=True).values
    return float(-(p * np.log2(p)).sum())

H_y = shannon_entropy(rec["Exited"])
ig_map = {}
for c in base_cols:
    if rec[c].dtype == object or rec[c].nunique() <= 10:
        x = rec[c].astype(str)
    else:
        x = pd.qcut(rec[c], q=5, duplicates="drop").astype(str)
    H_cond = sum(len(g) / len(rec) * shannon_entropy(g) for _, g in rec.groupby(x)["Exited"])
    ig_map[c] = H_y - H_cond

M["feature_selection"] = {
    "features": base_cols,
    "pearson_abs": [round(corr_map[c], 4) if c in corr_map else None for c in base_cols],
    "mutual_info": [round(mi_map[c], 4) for c in base_cols],
    "info_gain": [round(ig_map[c], 4) for c in base_cols],
    "target_entropy": round(H_y, 4),
    "selected": [c for c in base_cols if mi_map[c] > 0.005],
}

# --- Phase 2: clusters -------------------------------------------------------
print("Cluster profiles ...", flush=True)
clus_names = rec.groupby("Cluster")["Cluster_Name"].first().to_dict()
prof = {}
for k, sub in rec.groupby("Cluster"):
    geo = sub["Geography"].value_counts(normalize=True).mul(100).round(1)
    pmix = sub["NumOfProducts"].value_counts(normalize=True).mul(100).round(1).sort_index()
    prof[int(k)] = {
        "name": clus_names[k],
        "n": int(len(sub)),
        "share": round(len(sub) / len(rec) * 100, 1),
        "churn": round(sub["Exited"].mean() * 100, 1),
        "lift": round(sub["Exited"].mean() / baseline, 2),
        "balance_mean": round(float(sub["Balance"].mean()), 0),
        "zero_balance_pct": round(float((sub["Balance"] == 0).mean() * 100), 1),
        "products_mean": round(float(sub["NumOfProducts"].mean()), 2),
        "age_mean": round(float(sub["Age"].mean()), 1),
        "active_pct": round(float(sub["IsActiveMember"].mean() * 100), 1),
        "geo_mix": {g: float(geo.get(g, 0)) for g in ["France", "Germany", "Spain"]},
        "product_mix": {str(int(p)): float(v) for p, v in pmix.items()},
    }
M["clusters"] = prof

# deviation ("snake") profile in population-SD units
snake_feats = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
               "HasCrCard", "IsActiveMember", "EstimatedSalary"]
mu, sd = rec[snake_feats].mean(), rec[snake_feats].std()
M["snake"] = {
    "features": snake_feats,
    "clusters": {str(int(k)): ((rec.loc[rec["Cluster"] == k, snake_feats].mean() - mu) / sd)
                 .round(3).tolist() for k in sorted(rec["Cluster"].unique())},
}

# separation tests (effect sizes) - what may legitimately name a cluster
sep_rows = []
for c in ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]:
    groups = [g[c].values for _, g in rec.groupby("Cluster")]
    H, p = stats.kruskal(*groups)
    eps2 = max((H - len(groups) + 1) / (len(rec) - len(groups)), 0)
    sep_rows.append({"feature": c, "test": "Kruskal-Wallis", "effect": round(eps2, 3),
                     "metric": "epsilon^2"})
for c in ["Geography", "Gender", "Exited", "IsActiveMember", "HasCrCard", "NumOfProducts"]:
    ct = pd.crosstab(rec["Cluster"], rec[c])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    V = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    sep_rows.append({"feature": c, "test": "Chi-square", "effect": round(V, 3),
                     "metric": "Cramer's V"})
M["separation"] = sorted(sep_rows, key=lambda r: -r["effect"])

M["validation"] = {
    "k_range": K_RANGE,
    "inertia": [round(v, 0) for v in inertias],
    "silhouette": [round(v, 4) for v in sils],
    "chosen_k": 3,
    "sil_k3": round(sils[K_RANGE.index(3)], 4),
    "sil_k2": round(sils[K_RANGE.index(2)], 4),
    "sil_ward": round(sil_ward, 4) if sil_ward else None,
    "ari_kmeans_ward": round(ari, 4) if ari else None,
    "nmi_kmeans_ward": round(nmi, 4) if nmi else None,
    "pca_evr": [round(float(e), 1) for e in evr],
    "pca_loadings": {c: [round(float(load.loc[c, "PC1"]), 2), round(float(load.loc[c, "PC2"]), 2)]
                     for c in load.index},
}

# DBSCAN summary
db_sizes = pd.Series(db_labels).value_counts().sort_index()
M["dbscan"] = {
    "eps": 1.25, "min_samples": 10,
    "clusters": int(len(db_sizes[db_sizes.index != -1])),
    "sizes": {str(int(k)): int(v) for k, v in db_sizes.items()},
    "noise_n": n_noise,
    "noise_pct": round(n_noise / len(rec) * 100, 2),
    "noise_churn": round(float(rec.loc[db_labels == -1, "Exited"].mean() * 100), 1),
    "noise_age_mean": round(float(rec.loc[db_labels == -1, "Age"].mean()), 1),
    "noise_products_mean": round(float(rec.loc[db_labels == -1, "NumOfProducts"].mean()), 2),
    "core_products": {str(int(k)): round(float(rec.loc[db_labels == k, "NumOfProducts"].mean()), 2)
                      for k in db_sizes.index if k != -1},
}

# --- Phase 3: hypothesis verification (direct from raw data) ----------------
mask_a = (rec["Geography"] == "Germany") & (rec["IsActiveMember"] == 0) & (rec["NumOfProducts"] == 1)
mask_f = mask_a & (rec["Exited"] == 1)
conf = mask_f.sum() / mask_a.sum()
M["hypothesis"] = {
    "antecedent_n": int(mask_a.sum()),
    "churned_n": int(mask_f.sum()),
    "support_pct": round(mask_f.mean() * 100, 2),
    "confidence_pct": round(conf * 100, 1),
    "lift": round(conf / baseline, 2),
}

# --- Phase 4: anomaly aggregates ---------------------------------------------
print("Anomaly aggregates ...", flush=True)
methods = [("IQR_flag", "IQR (1.5 x IQR)", "Univariate"),
           ("ZScore_flag", "Z-score (|z| > 3)", "Univariate"),
           ("IF_flag", "Isolation Forest (5%)", "Multivariate"),
           ("MAHA_flag", "Robust Mahalanobis (chi2 99.9%)", "Multivariate"),
           ("LOF_flag", "Local Outlier Factor (k=20)", "Multivariate"),
           ("DBSCAN_flag", "DBSCAN noise (Phase 2)", "Multivariate")]
M["anomaly_methods"] = [{
    "method": label, "family": fam,
    "flagged": int(anom[c].sum()),
    "pct": round(anom[c].mean() * 100, 2),
    "churn_pct": round(float(anom.loc[anom[c] == 1, "Exited"].mean() * 100), 1),
} for c, label, fam in methods]

comp = anom.groupby("Composite_Anomaly_Score")["Exited"].agg(["mean", "count", "sum"])
M["composite"] = {
    "scores": comp.index.astype(int).tolist(),
    "n": comp["count"].astype(int).tolist(),
    "churn_pct": (comp["mean"] * 100).round(1).tolist(),
}

seg_order = ["Neither", "Univariate only", "Both families", "Multivariate only"]
seg = anom.groupby("UniMV_Segment")["Exited"].agg(["mean", "count"]).reindex(seg_order)
mv_only = anom["UniMV_Segment"] == "Multivariate only"
hidden = int((anom.loc[mv_only, "Max_Abs_Z"] < 3).sum())
M["uni_mv"] = {
    "segments": seg_order,
    "n": seg["count"].astype(int).tolist(),
    "churn_pct": (seg["mean"] * 100).round(1).tolist(),
    "mv_only_hidden": hidden,
    "mv_only_hidden_pct": round(hidden / int(mv_only.sum()) * 100, 0),
    "jaccard": round(float(((anom["UNI_flag"] == 1) & (anom["MV_flag"] == 1)).sum()
                     / ((anom["UNI_flag"] == 1) | (anom["MV_flag"] == 1)).sum()), 3),
    "kappa": round(float(cohen_kappa_score(anom["UNI_flag"], anom["MV_flag"])), 3),
}

flagged = anom[anom["Composite_Anomaly_Score"] >= 1]
cls = flagged.groupby("Anomaly_Class")["Exited"].agg(["count", "mean"])
M["anomaly_classes"] = [{
    "cls": i, "n": int(r["count"]),
    "pct_of_flagged": round(r["count"] / len(flagged) * 100, 1),
    "churn_pct": round(r["mean"] * 100, 1),
} for i, r in cls.sort_values("count", ascending=False).iterrows()]

pairs, flags4 = [], ["IQR_flag", "ZScore_flag", "IF_flag", "DBSCAN_flag"]
for a, b in combinations(flags4, 2):
    inter = int(((anom[a] == 1) & (anom[b] == 1)).sum())
    union = int(((anom[a] == 1) | (anom[b] == 1)).sum())
    pairs.append({"pair": f"{a.replace('_flag','')} vs {b.replace('_flag','')}",
                  "intersection": inter,
                  "jaccard": round(inter / union, 3) if union else 0,
                  "kappa": round(float(cohen_kappa_score(anom[a], anom[b])), 3)})
M["pairwise"] = sorted(pairs, key=lambda r: -r["jaccard"])

hb = rec[(rec["Balance"] > rec["Balance"].quantile(0.75)) & (rec["Exited"] == 1)]
M["high_balance"] = {
    "p75": round(float(rec["Balance"].quantile(0.75)), 0),
    "n": int(len(hb)),
    "balance_mean": round(float(hb["Balance"].mean()), 0),
    "germany_pct": round(float((hb["Geography"] == "Germany").mean() * 100), 1),
    "churn_rate_hb": round(float(rec.loc[rec["Balance"] > rec["Balance"].quantile(0.75), "Exited"].mean() * 100), 1),
    "inactive_pct": round(float((hb["IsActiveMember"] == 0).mean() * 100), 1),
}

# cross-reference: Phase 2 cluster outliers vs Phase 4 statistical anomalies
if_db = int(((anom["IF_flag"] == 1) & (anom["DBSCAN_flag"] == 1)).sum())
M["cross_ref"] = {
    "if_dbscan_overlap": if_db,
    "if_dbscan_churn": round(float(anom.loc[(anom["IF_flag"] == 1) & (anom["DBSCAN_flag"] == 1), "Exited"].mean() * 100), 1),
    "noise_by_cluster": {str(int(k)): int(v) for k, v in
                         rec.loc[rec["DBSCAN_flag"] == 1].groupby("Cluster").size().items()},
    "flagged_churn_by_cluster": {str(int(k)): round(float(v) * 100, 1) for k, v in
                                 rec.loc[rec["Composite_Anomaly_Score"] >= 1]
                                 .groupby("Cluster")["Exited"].mean().items()},
}

with open(OUT / "metrics.json", "w") as f:
    json.dump(M, f, indent=1)

# ---------------------------------------------------------------------------
# 4. Rules -> rules.json (top-10 deliverable + commentary)
# ---------------------------------------------------------------------------
print("Formatting rules ...", flush=True)

PRETTY = {
    "Age_Band_Senior": "Senior (46-60)", "Age_Band_Elderly": "Elderly (60+)",
    "Age_Band_Middle_Aged": "Middle-aged", "Age_Band_Young_Adult": "Young adult",
    "Active_Status_Inactive": "Inactive member", "Active_Status_Active": "Active member",
    "Products_Label_Products_1": "1 product only", "Products_Label_Products_2": "2 products",
    "CrCard_Status_Has_CrCard": "Has credit card", "CrCard_Status_No_CrCard": "No credit card",
    "Gender_Female": "Female", "Gender_Male": "Male",
    "Geography_Germany": "Germany", "Geography_France": "France", "Geography_Spain": "Spain",
    "Balance_Band_Insured_Balance": "Insured balance (0-100K)",
    "Balance_Band_Above_DGS_Ceiling": "Above DGS ceiling (>100K)",
    "Balance_Band_Zero_Balance": "Zero balance",
    "Churn_Status_Churned": "CHURNED",
}

COMMENTARY = {
    frozenset({"Active_Status_Inactive", "Age_Band_Senior", "Products_Label_Products_1"}):
        ("The single strongest churn profile found. Inactive seniors holding only one product "
         "churn at ~77% - nearly 4x the 20.4% base rate. Action: proactive retention call before a "
         "second consecutive inactive quarter, plus a bundled-product offer."),
    frozenset({"Active_Status_Inactive", "Age_Band_Senior"}):
        ("Inactivity alone is a much weaker signal - it is the AGE interaction that drives risk. "
         "Seniors who go quiet tend to disengage permanently, while younger inactive customers often "
         "re-engage on their own."),
    frozenset({"Active_Status_Inactive", "Age_Band_Senior", "CrCard_Status_Has_CrCard"}):
        ("Holding a credit card does not protect inactive seniors at all - the card is a shallow "
         "anchor. Risk is essentially identical to the card-free version of this profile."),
    frozenset({"Age_Band_Senior", "Geography_Germany"}):
        ("The most surprising rule: German seniors churn at >3x baseline regardless of activity or "
         "product count. It points to a product-fit or service-quality issue specific to the German "
         "operation, not just an age effect."),
    frozenset({"Age_Band_Senior", "Gender_Female", "Products_Label_Products_1"}):
        ("A gender-age interaction invisible in simple cross-tabs: female seniors with a single "
         "product churn at 66%. Combine cross-sell with age-appropriate engagement."),
    frozenset({"Age_Band_Senior", "Products_Label_Products_1"}):
        ("Single-product seniors churn at 61%. Cross-sell is the obvious lever - and the data shows "
         "the bank has historically failed to deepen exactly this segment."),
    frozenset({"Age_Band_Senior", "CrCard_Status_Has_CrCard", "Products_Label_Products_1"}):
        ("A credit card does NOT reduce churn risk for single-product seniors. Card-only "
         "relationships are shallow relationships."),
    frozenset({"Age_Band_Senior", "Gender_Female"}):
        ("Senior + female alone - without inactivity or product-count conditions - already clears "
         "the 2.5x lift bar. The two demographics compound."),
    frozenset({"Active_Status_Inactive", "Age_Band_Senior", "Balance_Band_Above_DGS_Ceiling"}):
        ("New with the DGS-anchored binning: inactive seniors holding more than the EUR 100K "
         "deposit-guarantee ceiling churn at 72.6%. Money above the state guarantee is the most "
         "mobile money in the book - the highest-priority relationship-manager list."),
    frozenset({"Age_Band_Senior", "Balance_Band_Above_DGS_Ceiling", "Products_Label_Products_1"}):
        ("Single-product seniors above the insured ceiling - high-value, shallow-anchored, "
         "uninsured excess: the costliest churn profile per customer."),
    frozenset({"Age_Band_Senior", "Balance_Band_Above_DGS_Ceiling"}):
        ("Seniors above the EUR 100K ceiling churn at 57.7% (~3x baseline). Under the old 50-125K "
         "binning this pattern was split across two bands and read as 'the bank retains the "
         "wealthy' - the regulatory boundary reverses that conclusion: uninsured excess is the "
         "most flight-prone money."),
    frozenset({"Age_Band_Senior", "CrCard_Status_Has_CrCard", "Gender_Female"}):
        ("Female senior cardholders - the same story again: the card alone does not anchor the "
         "relationship."),
    frozenset({"Products_Label_Products_1", "Geography_Germany", "Active_Status_Inactive"}):
        ("Validates the assigned project hypothesis: inactive German single-product customers churn "
         "at >2.5x baseline. Together with the German-senior rule, evidence that Germany has a "
         "structural retention problem."),
}

def parse_frozen(s):
    return frozenset(ast.literal_eval(s.replace("frozenset(", "").rstrip(")")))

rules_out = []
for i, r in top_rules.iterrows():
    items = parse_frozen(r["antecedents"])
    rules_out.append({
        "rank": i + 1,
        "letter": chr(65 + i),
        "if_items": sorted(PRETTY.get(x, x) for x in items),
        "if_raw": sorted(items),
        "then": "Churned",
        "support_pct": float(r["Support (%)"]),
        "confidence_pct": float(r["Confidence (%)"]),
        "lift": float(r["Lift"]),
        "conviction": float(r["Conviction"]),
        "customers": int(round(r["Support (%)"] / 100 * 10_000)),
        "commentary": COMMENTARY.get(items, "High-lift churn profile - see rule table."),
    })

# All churn-consequent rules from the full rule file (17 under the DGS bands)
def frozen_ok(s):
    try:
        return parse_frozen(s)
    except Exception:
        return frozenset()

all_rules["A"] = all_rules["antecedents"].apply(frozen_ok)
all_rules["C"] = all_rules["consequents"].apply(frozen_ok)
churn13 = all_rules[(all_rules["C"].apply(lambda x: "Churn_Status_Churned" in x)) &
                    (~all_rules["A"].apply(lambda x: "Churn_Status_Churned" in x))]
churn13 = churn13.sort_values("lift", ascending=False)
extra = []
for _, r in churn13.iterrows():
    extra.append({
        "if_items": sorted(PRETTY.get(x, x) for x in r["A"]),
        "support_pct": round(r["support"] * 100, 2),
        "confidence_pct": round(r["confidence"] * 100, 1),
        "lift": round(r["lift"], 3),
    })

with open(OUT / "rules.json", "w") as f:
    json.dump({"top10": rules_out, "all_churn_rules": extra,
               "thresholds": {"min_support": 0.03, "min_confidence": 0.50, "min_lift": 1.5},
               "n_total_rules": int(len(all_rules)),
               "n_churn_rules": int(len(churn13))}, f, indent=1)

import shutil
shutil.rmtree(CK, ignore_errors=True)   # checkpoints no longer needed

print(f"\nDone. Cache written to {OUT}")
print(f"  records.csv  ({(OUT/'records.csv').stat().st_size/1024:.0f} KB)")
print(f"  metrics.json ({(OUT/'metrics.json').stat().st_size/1024:.0f} KB)")
print(f"  rules.json   ({(OUT/'rules.json').stat().st_size/1024:.0f} KB)")
