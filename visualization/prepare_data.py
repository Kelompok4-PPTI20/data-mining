"""
Phase 5 — Dashboard data preparation (Group 5, Bank Customer Churn)

Precomputes everything the Dash app needs so that every dashboard interaction
is an in-memory filter on small dataframes (well under the 100 ms rubric bar).

Run AFTER the main pipeline notebook (notebooks/notebook.ipynb), from this folder:

    python prepare_data.py

Inputs  (produced by the notebook):
    ../data/processed/churn_clean.csv
    ../data/processed/churn_clustered.csv
    ../data/processed/dbscan_outlier_indices.npy
    ../outputs/ph3_all_association_rules.csv
    ../outputs/ph4_anomaly_report.csv

Outputs (consumed by app.py):
    viz_data/famd_coords.csv     2-D FAMD projection + labels for the cluster map
    viz_data/rules_churn.csv     parsed churn-consequent association rules
    viz_data/summary.json        KPIs, cluster profiles, anomaly method table
"""
import json
import os
import re

import numpy as np
import pandas as pd

RANDOM_STATE = 42
OUT_DIR = 'viz_data'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. FAMD 2-D projection (same method as notebook cell "FAMD 2D Projection") ──
import prince

clustered = pd.read_csv('../data/processed/churn_clustered.csv')
X_mixed = clustered.drop(columns=['Exited', 'Cluster', 'Cluster_Name'])

famd = prince.FAMD(n_components=2, random_state=RANDOM_STATE).fit(X_mixed)
coords = famd.row_coordinates(X_mixed)
coords.columns = ['Dim1', 'Dim2']
explained = list(famd.percentage_of_variance_[:2])

dbscan_noise_idx = np.load('../data/processed/dbscan_outlier_indices.npy')
coords['DBSCAN_noise'] = 0
coords.loc[dbscan_noise_idx, 'DBSCAN_noise'] = 1

for col in ['Cluster', 'Cluster_Name', 'Exited', 'Geography', 'Age',
            'Balance', 'NumOfProducts', 'IsActiveMember']:
    coords[col] = clustered[col].values

coords.to_csv(f'{OUT_DIR}/famd_coords.csv', index=False)
print(f'  famd_coords.csv  {coords.shape} | explained variance {explained[0]:.1f}% / {explained[1]:.1f}%')

# ── 2. Parse association rules (frozenset strings -> pipe-joined items) ────────
def parse_items(s):
    return sorted(re.findall(r"'([^']+)'", s))

rules = pd.read_csv('../outputs/ph3_all_association_rules.csv')
rules['ant_items'] = rules['antecedents'].apply(parse_items)
rules['con_items'] = rules['consequents'].apply(parse_items)

churn_rules = rules[
    rules['con_items'].apply(lambda x: 'Churn_Status_Churned' in x)
    & ~rules['ant_items'].apply(lambda x: 'Churn_Status_Churned' in x)
].copy()

churn_rules['antecedent'] = churn_rules['ant_items'].apply('|'.join)
churn_rules['consequent_extra'] = churn_rules['con_items'].apply(
    lambda x: '|'.join(i for i in x if i != 'Churn_Status_Churned'))
churn_rules['n_ant'] = churn_rules['ant_items'].apply(len)

keep = ['antecedent', 'consequent_extra', 'n_ant',
        'support', 'confidence', 'lift', 'conviction']
churn_rules = (churn_rules[keep]
               .sort_values('lift', ascending=False)
               .reset_index(drop=True))
churn_rules.to_csv(f'{OUT_DIR}/rules_churn.csv', index=False)
print(f'  rules_churn.csv  {len(churn_rules)} churn-consequent rules')

# ── 3. Summary KPIs / tables ───────────────────────────────────────────────────
anom = pd.read_csv('../outputs/ph4_anomaly_report.csv')
flagged = anom[anom['Composite_Anomaly_Score'] >= 1]
highbal = anom[anom['HighBal_Churn_flag'] == 1]

profiles = []
for c, g in clustered.groupby('Cluster'):
    profiles.append({
        'cluster': int(c),
        'name': g['Cluster_Name'].iloc[0],
        'n': int(len(g)),
        'pct': round(len(g) / len(clustered) * 100, 1),
        'churn_rate': round(g['Exited'].mean() * 100, 1),
        'avg_balance': round(g['Balance'].mean(), 0),
        'avg_products': round(g['NumOfProducts'].mean(), 2),
        'avg_age': round(g['Age'].mean(), 1),
        'active_rate': round(g['IsActiveMember'].mean() * 100, 1),
        'avg_tenure': round(g['Tenure'].mean(), 2),
        'germany_pct': round((g['Geography'] == 'Germany').mean() * 100, 1),
    })

method_cols = {'IQR_flag': 'IQR (1.5×IQR)', 'ZScore_flag': 'Z-score (|z|>3)',
               'IF_flag': 'Isolation Forest', 'DBSCAN_flag': 'DBSCAN noise (Phase 2)'}
methods = []
for col, label in method_cols.items():
    m = anom[anom[col] == 1]
    methods.append({'method': label, 'flagged': int(len(m)),
                    'pct': round(len(m) / len(anom) * 100, 2),
                    'churn_rate': round(m['Exited'].mean() * 100, 1)})

summary = {
    'n_customers': int(len(clustered)),
    'baseline_churn': round(clustered['Exited'].mean() * 100, 2),
    'famd_explained': [round(v, 1) for v in explained],
    'geo_churn': {g: round(v * 100, 1) for g, v in
                  clustered.groupby('Geography')['Exited'].mean().items()},
    'cluster_profiles': profiles,
    'n_churn_rules': int(len(churn_rules)),
    'anomaly_methods': methods,
    'n_flagged': int(len(flagged)),
    'n_high_conf': int((anom['Composite_Anomaly_Score'] >= 3).sum()),
    'class_counts': flagged['Anomaly_Class'].str[0].value_counts().to_dict(),
    'highbal_churners': {'n': int(len(highbal)),
                         'avg_balance': round(highbal['Balance'].mean(), 0),
                         'germany_pct': round((highbal['Geography'] == 'Germany').mean() * 100, 1)},
}
with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('  summary.json     written')
print('Done — start the dashboard with:  python app.py')
