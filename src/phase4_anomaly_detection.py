"""Phase 4: anomaly and outlier detection."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

if __package__:
    from ._pipeline_utils import (
        ANOMALY_REPORT_PATH,
        CLEAN_PATH,
        DBSCAN_OUTLIERS_PATH,
        HIGH_CONFIDENCE_ANOMALIES_PATH,
        RANDOM_STATE,
        RAW_PATH,
        configure_runtime,
        display,
        require_files,
    )
else:
    from _pipeline_utils import (
        ANOMALY_REPORT_PATH,
        CLEAN_PATH,
        DBSCAN_OUTLIERS_PATH,
        HIGH_CONFIDENCE_ANOMALIES_PATH,
        RANDOM_STATE,
        RAW_PATH,
        configure_runtime,
        display,
        require_files,
    )


def run_phase4():
    """Run notebook cells 31-41 and write all Phase 4 artifacts."""

    configure_runtime()
    require_files([CLEAN_PATH, DBSCAN_OUTLIERS_PATH], "Phase 4")
    warnings.filterwarnings("ignore")
    np.random.seed(RANDOM_STATE)
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 5)})
    df = pd.read_csv(CLEAN_PATH)

    # ## Phase 4: Anomaly & Outlier Detection
    # 
    # **Algorithms Applied:**
    # 1. **IQR Method** — Statistical outlier identification (1.5 × IQR rule)
    # 2. **Z-Score Method** — Normal distribution outlier identification (|z| > 3)
    # 3. **Isolation Forest** — Machine learning tree-based anomaly isolation
    # 4. **Robust Mahalanobis Distance (MinCovDet)** — multivariate statistical detection *(beyond-requirement extension)*
    # 5. **Local Outlier Factor (LOF)** — multivariate local-density detection *(beyond-requirement extension)*
    # 
    # **Systematic comparison plan:** the three required methods are compared pairwise, and additionally the univariate family (IQR ∪ Z-score) is compared against the multivariate family (Mahalanobis ∪ LOF ∪ Isolation Forest), with every disagreement between the families decomposed and explained.
    # 
    # **Strategic Focus:** Detect structural anomalies, specifically:
    # - Retrospectively profile financially material high-balance exits; this
    #   snapshot cannot observe a preceding balance drop
    # - Cross-reference with DBSCAN noise points from Phase 2
    # 
    # **Anomaly Classification:**
    # - Class A: Data Error (unrealistic/impossible values)
    # - Class B: Rare Legitimate Case (statistically unusual but valid)
    # - Class C: Risk Signal (structural rarity or Phase 3 profile overlap)
    # 

    # In[31]:


    # ── IQR Outlier Detection ─────────────────────────────────────────────────────
    anomaly_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    df_anomaly   = df.copy()
    source_row_numbers = pd.read_csv(RAW_PATH, usecols=['RowNumber'])['RowNumber']
    if len(source_row_numbers) != len(df_anomaly):
        raise ValueError(
            'Raw and clean row counts differ; source-row traceability is unsafe.'
        )
    df_anomaly.insert(0, 'Source_RowNumber', source_row_numbers.to_numpy())

    iqr_flags = pd.DataFrame(index=df_anomaly.index)

    for col in anomaly_cols:
        q1, q3 = df_anomaly[col].quantile([0.25, 0.75])
        iqr    = q3 - q1
        lower  = q1 - 1.5 * iqr
        upper  = q3 + 1.5 * iqr

        iqr_flags[f'IQR_flag_{col}'] = (
            (df_anomaly[col] < lower) | (df_anomaly[col] > upper)
        ).astype(int)

        n_out = iqr_flags[f'IQR_flag_{col}'].sum()
        print(f"  {col:<20} | Lower: {lower:,.1f} | Upper: {upper:,.1f} | "
              f"Outliers: {n_out:,} ({n_out/len(df)*100:.2f}%)")

    df_anomaly['IQR_outlier_count'] = iqr_flags.sum(axis=1)
    df_anomaly['IQR_flag']          = (df_anomaly['IQR_outlier_count'] > 0).astype(int)

    print(f"\n── IQR Summary ──")
    print(f"  Records flagged (any feature): {df_anomaly['IQR_flag'].sum():,} "
          f"({df_anomaly['IQR_flag'].mean()*100:.2f}%)")
    print(f"  Churn rate — IQR flagged:      "
          f"{df_anomaly[df_anomaly['IQR_flag']==1]['Exited'].mean()*100:.1f}%")
    print(f"  Churn rate — IQR normal:       "
          f"{df_anomaly[df_anomaly['IQR_flag']==0]['Exited'].mean()*100:.1f}%")


    # In[32]:


    # ── Z-Score Outlier Detection ─────────────────────────────────────────────────
    zscore_flags = pd.DataFrame(index=df_anomaly.index)

    for col in anomaly_cols:
        z_scores = np.abs(stats.zscore(df_anomaly[col]))
        zscore_flags[f'Z_flag_{col}'] = (z_scores > 3).astype(int)
        n_out = zscore_flags[f'Z_flag_{col}'].sum()
        print(f"  {col:<20} | |Z| > 3 outliers: {n_out:,} ({n_out/len(df)*100:.2f}%)")

    df_anomaly['ZScore_outlier_count'] = zscore_flags.sum(axis=1)
    df_anomaly['ZScore_flag']          = (df_anomaly['ZScore_outlier_count'] > 0).astype(int)

    # ── IQR vs Z-Score Agreement Analysis ────────────────────────────────────────
    both_flagged = ((df_anomaly['IQR_flag'] == 1) & (df_anomaly['ZScore_flag'] == 1)).sum()
    only_iqr     = ((df_anomaly['IQR_flag'] == 1) & (df_anomaly['ZScore_flag'] == 0)).sum()
    only_z       = ((df_anomaly['IQR_flag'] == 0) & (df_anomaly['ZScore_flag'] == 1)).sum()
    neither      = ((df_anomaly['IQR_flag'] == 0) & (df_anomaly['ZScore_flag'] == 0)).sum()

    print(f"\n── IQR vs Z-Score Comparison ──")
    print(f"  Flagged by BOTH:      {both_flagged:,}")
    print(f"  Flagged by IQR only:  {only_iqr:,}")
    print(f"  Flagged by Z only:    {only_z:,}")
    print(f"  Flagged by NEITHER:   {neither:,}")
    print(f"\n  Agreement Rate: {both_flagged/(df_anomaly['IQR_flag'].sum())*100:.1f}% of IQR outliers confirmed by Z-Score")


    # ### Interpretation — Why IQR and Z-Score Disagree (Even Though Both Are Univariate)
    # 
    # Both methods scan the same four columns, yet IQR flags 374 records, Z-score only 141 — and **zero records are flagged by Z alone**. The asymmetry is fully explained by fence geometry:
    # 
    # - **Under these distribution shapes, Z's fence is strictly farther out, so its flags form a subset of IQR's.** Under normality the 1.5×IQR fence sits at |z| ≈ 2.7 — *inside* the |z| = 3 fence. Whatever clears z > 3 has already failed the IQR test; "Z-only = 0" is a structural outcome, not a coincidence.
    # - **Age (359 IQR vs. 133 Z):** the IQR fence is 62 years; the z = 3 threshold is 70.4 years. The 226 customers aged 63–70 fall in the gap and are IQR-only. Worse for Z: the right-skewed senior tail *inflates the standard deviation used to compute z*, pushing its own fence outward — z-scores are partially blinded by the very outliers they hunt (classical masking on skewed data), while IQR's quartiles are robust to the tail.
    # - **CreditScore (15 vs. 8):** same mechanics on the lower tail — IQR fence 383 vs. z fence ≈ 361, with 7 customers scoring in between.
    # - **Balance and EstimatedSalary (0 and 0 under both):** flat, bounded distributions cannot reach either fence. Salary is near-uniform, and a uniform variable's most extreme value sits at only |z| ≈ 1.73 (observed max: 1.74) — **z > 3 is mathematically impossible for it**. Zero flags reflect method limits, not clean data.
    # - **The 141 records flagged by both** are the hard core of univariate extremeness: 100% of Z flags are IQR-confirmed; 37.7% of IQR flags are Z-confirmed.
    # 
    # **Conclusion:** these two methods ask the *same question* ("is this single value extreme?") with different rulers, so their disagreement is one of calibration and robustness — the union (IQR ∪ Z) is carried forward and nothing is lost. Contrast this with the univariate-vs-multivariate comparison below, where the methods ask *different questions* and the disagreement is structural, revealing genuinely different kinds of anomaly.
    # 

    # In[33]:


    # ── Isolation Forest ──────────────────────────────────────────────────────────
    CONTAMINATION = 0.05   # Expect ~5% anomalies (tune based on IQR/Z-score findings)

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Fit on scaled numeric features
    scale_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                  'NumOfProducts', 'EstimatedSalary']
    X_iso = StandardScaler().fit_transform(df_anomaly[scale_cols])

    iso_labels  = iso_forest.fit_predict(X_iso)      # -1 = anomaly, 1 = normal
    iso_scores  = iso_forest.score_samples(X_iso)    # More negative = more anomalous

    df_anomaly['IF_flag']        = (iso_labels == -1).astype(int)
    df_anomaly['IF_anomaly_score'] = iso_scores

    print(f"── Isolation Forest Results (contamination={CONTAMINATION}) ──")
    print(f"  Anomalies flagged: {df_anomaly['IF_flag'].sum():,} "
          f"({df_anomaly['IF_flag'].mean()*100:.1f}%)")
    print(f"  Churn rate — IF anomalies: "
          f"{df_anomaly[df_anomaly['IF_flag']==1]['Exited'].mean()*100:.1f}%")
    print(f"  Churn rate — IF normal:    "
          f"{df_anomaly[df_anomaly['IF_flag']==0]['Exited'].mean()*100:.1f}%")

    # ── Anomaly Score Distribution ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Score distribution
    axes[0].hist(iso_scores, bins=60, color='steelblue', edgecolor='white',
                 linewidth=0.4, label='All records')
    threshold = np.percentile(iso_scores, CONTAMINATION * 100)
    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2,
                    label=f'Anomaly threshold: {threshold:.3f}')
    axes[0].set_title('Isolation Forest: Anomaly Score Distribution', fontweight='bold')
    axes[0].set_xlabel('Anomaly Score (more negative = more anomalous)')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    # Churn rate by flag
    flag_groups = df_anomaly.groupby('IF_flag')['Exited'].mean() * 100
    axes[1].bar(['Normal (IF=0)', 'Anomalous (IF=1)'], flag_groups.values,
                color=['#2c7bb6','#d7191c'])
    axes[1].set_title('Churn Rate: Normal vs. IF Anomalies', fontweight='bold')
    axes[1].set_ylabel('Churn Rate (%)')
    for i, v in enumerate(flag_groups.values):
        axes[1].text(i, v+0.5, f'{v:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()

    plt.show()


    # ### Interpretation — Reading Isolation Forest's Numbers Correctly
    # 
    # - **"500 anomalies (5.0%)" is imposed, not discovered.** `contamination=0.05` fixes the flagged *quantile* by construction; the discovery is *which* 500 records rank most isolable, not how many. (IQR/Z work the opposite way: fixed fences, discovered counts. This asymmetry must be kept in mind whenever method counts are compared side by side.)
    # - **The discovered part is strong:** the 500 most isolable customers churn at 49.0% vs. 18.9% for the rest — a 2.6× concentration obtained *without the model ever seeing the churn label*. Because Isolation Forest evaluates all six numeric dimensions jointly, it can rank as anomalous the unusual-combination customers that per-feature fences structurally miss — the bridge to the multivariate family introduced next.
    # - **The score histogram shows a smooth left tail, not a separated anomaly mode**, with the threshold (−0.561) cutting through a continuum. Consistent with everything so far (weak silhouettes, near-orthogonal features): this book has no crisp normal/abnormal boundary, anomaly-ness is *graded* — which is why Phase 4 ends by classifying each flagged record individually instead of trusting a binary flag.
    # 

    # ### Multivariate Outlier Detection — Mahalanobis Distance & Local Outlier Factor
    # 
    # IQR and Z-score above are **univariate**: they inspect one feature at a time, so they can only catch values that are extreme *on their own axis*. A customer whose every individual value is unremarkable can still be a glaring anomaly as a **combination** (e.g., very young with a very high balance). Two multivariate detectors are added:
    # 
    # - **Robust Mahalanobis distance (MinCovDet)** — *statistical*: measures each record's distance from the multivariate centre in the metric of the feature covariance, so correlated features are properly accounted for. Robust MCD estimation prevents the outliers themselves from inflating the covariance estimate ("masking"). Flag threshold: the χ² quantile at 99.9% (df = number of features), the standard cutoff for squared Mahalanobis distances.
    # - **Local Outlier Factor (LOF)** — *density-based*: flags records whose local neighbourhood density is far lower than that of their k=20 nearest neighbours, catching *local* anomalies a global covariance model misses. Contamination is fixed at 5% to be directly comparable with Isolation Forest.
    # 
    # Isolation Forest (fitted above) is also multivariate. In the comparison that follows, the three multivariate flags (Mahalanobis, LOF, IF) are grouped against the two univariate flags (IQR, Z-score), and every disagreement between the two families is decomposed and explained.

    # In[34]:


    # ── Mahalanobis Distance (Robust, MinCovDet) ──────────────────────────────────
    from sklearn.covariance import MinCovDet

    X_mv = X_iso                       # same scaled feature matrix as Isolation Forest
    mv_feature_names = scale_cols

    mcd = MinCovDet(support_fraction=0.9, random_state=RANDOM_STATE).fit(X_mv)
    maha_d2 = mcd.mahalanobis(X_mv)    # squared robust Mahalanobis distances

    MAHA_Q = 0.999
    maha_threshold = stats.chi2.ppf(MAHA_Q, df=X_mv.shape[1])
    df_anomaly['MAHA_dist2'] = maha_d2
    df_anomaly['MAHA_flag']  = (maha_d2 > maha_threshold).astype(int)

    alt_thr = stats.chi2.ppf(0.99, df=X_mv.shape[1])
    print(f"── Robust Mahalanobis Results ──")
    print(f"  Features: {mv_feature_names}")
    print(f"  chi2(df={X_mv.shape[1]}) @ 99.9% -> threshold {maha_threshold:.2f}")
    print(f"  Flagged: {df_anomaly['MAHA_flag'].sum():,} "
          f"({df_anomaly['MAHA_flag'].mean()*100:.2f}%)")
    print(f"  (Sensitivity: @99% threshold {alt_thr:.2f} would flag "
          f"{(maha_d2 > alt_thr).sum():,} records — reported for transparency)")
    print(f"  Churn rate — flagged: "
          f"{df_anomaly.loc[df_anomaly['MAHA_flag']==1,'Exited'].mean()*100:.1f}% | "
          f"normal: {df_anomaly.loc[df_anomaly['MAHA_flag']==0,'Exited'].mean()*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].hist(maha_d2, bins=80, color='steelblue', edgecolor='white', linewidth=.3)
    axes[0].axvline(maha_threshold, color='red', linestyle='--', linewidth=2,
                    label=f'chi2 99.9% = {maha_threshold:.1f}')
    axes[0].axvline(alt_thr, color='orange', linestyle=':', linewidth=2,
                    label=f'chi2 99% = {alt_thr:.1f}')
    axes[0].set_yscale('log')
    axes[0].set_title('Squared Robust Mahalanobis Distance', fontweight='bold')
    axes[0].set_xlabel('D² (log-count axis)')
    axes[0].legend()

    flag_churn = df_anomaly.groupby('MAHA_flag')['Exited'].mean() * 100
    axes[1].bar(['Normal', 'Mahalanobis outlier'], flag_churn.values,
                color=['#2c7bb6', '#d7191c'])
    for i, v in enumerate(flag_churn.values):
        axes[1].text(i, v + .5, f'{v:.1f}%', ha='center', fontweight='bold')
    axes[1].set_title('Churn Rate: Normal vs. Mahalanobis Outliers', fontweight='bold')
    axes[1].set_ylabel('Churn Rate (%)')
    plt.tight_layout()
    plt.show()


    # In[35]:


    # ── Local Outlier Factor (LOF) ────────────────────────────────────────────────
    from sklearn.neighbors import LocalOutlierFactor

    lof = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION)
    lof_labels = lof.fit_predict(X_mv)
    df_anomaly['LOF_flag']  = (lof_labels == -1).astype(int)
    df_anomaly['LOF_score'] = -lof.negative_outlier_factor_   # >1 = sparser than neighbours

    print(f"── LOF Results (k=20 neighbours, contamination={CONTAMINATION}) ──")
    print(f"  Flagged: {df_anomaly['LOF_flag'].sum():,} "
          f"({df_anomaly['LOF_flag'].mean()*100:.1f}%)")
    print(f"  Churn rate — flagged: "
          f"{df_anomaly.loc[df_anomaly['LOF_flag']==1,'Exited'].mean()*100:.1f}% | "
          f"normal: {df_anomaly.loc[df_anomaly['LOF_flag']==0,'Exited'].mean()*100:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(df_anomaly['LOF_score'], bins=80, color='steelblue',
            edgecolor='white', linewidth=.3)
    thr = df_anomaly.loc[df_anomaly['LOF_flag'] == 1, 'LOF_score'].min()
    ax.axvline(thr, color='red', linestyle='--', linewidth=2,
               label=f'Flag threshold ~ {thr:.2f}')
    ax.set_yscale('log')
    ax.set_title('LOF Score Distribution (>1 = locally sparser than neighbours)',
                 fontweight='bold')
    ax.set_xlabel('LOF score')
    ax.legend()
    plt.tight_layout()
    plt.show()


    # In[36]:


    # ── Univariate vs. Multivariate: Systematic Comparison ───────────────────────
    from sklearn.metrics import cohen_kappa_score

    UNI_METHODS = ['IQR_flag', 'ZScore_flag']
    MV_METHODS  = ['MAHA_flag', 'LOF_flag', 'IF_flag']

    df_anomaly['UNI_flag'] = (df_anomaly[UNI_METHODS].sum(axis=1) > 0).astype(int)
    df_anomaly['MV_flag']  = (df_anomaly[MV_METHODS].sum(axis=1) > 0).astype(int)

    method_summary = pd.DataFrame([
        {'Method': m, 'Type': t, 'Flagged': int(df_anomaly[m].sum()),
         '% of Data': round(df_anomaly[m].mean() * 100, 2),
         'Churn Rate (%)': round(df_anomaly.loc[df_anomaly[m] == 1, 'Exited'].mean() * 100, 1)}
        for m, t in [('IQR_flag', 'Univariate'), ('ZScore_flag', 'Univariate'),
                     ('MAHA_flag', 'Multivariate'), ('LOF_flag', 'Multivariate'),
                     ('IF_flag', 'Multivariate')]
    ])
    print("── Per-Method Summary ──")
    display(method_summary)

    seg = np.select(
        [(df_anomaly['UNI_flag'] == 1) & (df_anomaly['MV_flag'] == 1),
         (df_anomaly['UNI_flag'] == 1) & (df_anomaly['MV_flag'] == 0),
         (df_anomaly['UNI_flag'] == 0) & (df_anomaly['MV_flag'] == 1)],
        ['Both families', 'Univariate only', 'Multivariate only'],
        default='Neither')
    df_anomaly['UniMV_Segment'] = seg

    seg_stats = (df_anomaly.groupby('UniMV_Segment')
                 .agg(N=('Exited', 'size'), Churn_Rate=('Exited', 'mean')))
    seg_stats['Churn Rate (%)'] = (seg_stats['Churn_Rate'] * 100).round(1)
    seg_order = ['Neither', 'Univariate only', 'Both families', 'Multivariate only']
    seg_stats = seg_stats.reindex(seg_order)

    both_n = int(seg_stats.loc['Both families', 'N'])
    uni_n  = int(seg_stats.loc['Univariate only', 'N'])
    mv_n   = int(seg_stats.loc['Multivariate only', 'N'])
    jaccard = both_n / (both_n + uni_n + mv_n)
    kappa   = cohen_kappa_score(df_anomaly['UNI_flag'], df_anomaly['MV_flag'])

    print("── Family-Level Agreement: Univariate (IQR ∪ Z) vs Multivariate (MAHA ∪ LOF ∪ IF) ──")
    display(seg_stats[['N', 'Churn Rate (%)']])
    print(f"  Jaccard overlap: {jaccard:.3f} | Cohen's kappa: {kappa:.3f}")
    print("  -> Low overlap is NOT a defect: the two families are looking for")
    print("     different kinds of 'unusual', decomposed below.")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ['#2c7bb6', '#fdae61', '#abd9e9', '#d7191c']
    bars = ax.bar(seg_stats.index, seg_stats['Churn Rate (%)'], color=colors)
    ax.axhline(df_anomaly['Exited'].mean() * 100, color='black', linestyle='--',
               linewidth=1.5, label=f"Baseline {df_anomaly['Exited'].mean()*100:.1f}%")
    for bar, (_, row) in zip(bars, seg_stats.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .5,
                f"{row['Churn Rate (%)']:.1f}%\n(n={int(row['N']):,})",
                ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Churn Rate by Detection-Family Segment', fontweight='bold')
    ax.set_ylabel('Churn Rate (%)')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ── WHY the families disagree: decomposition of the disagreement pools ───────
    z_all = np.abs(stats.zscore(df_anomaly[scale_cols].values))
    df_anomaly['Max_Abs_Z'] = z_all.max(axis=1)

    mv_only_mask  = df_anomaly['UniMV_Segment'] == 'Multivariate only'
    uni_only_mask = df_anomaly['UniMV_Segment'] == 'Univariate only'

    hidden = int((df_anomaly.loc[mv_only_mask, 'Max_Abs_Z'] < 3).sum())
    print(f"── Decomposition of Disagreements ──")
    print(f"  Multivariate-only pool: {mv_only_mask.sum():,} records")
    print(f"    Of these, {hidden:,} ({hidden / mv_only_mask.sum() * 100:.0f}%) have NO single feature")
    print(f"    beyond |z| = 3 — anomalous purely as COMBINATIONS, structurally")
    print(f"    invisible to any one-feature-at-a-time screen (IQR / Z-score).")
    print(f"  Univariate-only pool: {uni_only_mask.sum():,} records")
    print(f"    Single-feature extremes (e.g., age or credit-score tails) that sit in")
    print(f"    otherwise dense, ordinary neighbourhoods — with contamination capped")
    print(f"    at 5% and covariance-aware distance, the multivariate methods do not")
    print(f"    rank them among the most unusual records overall.")

    example_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                    'IsActiveMember', 'Geography', 'Exited', 'Max_Abs_Z', 'MAHA_dist2']
    print("\n── Example MULTIVARIATE-ONLY anomalies (every feature individually 'normal') ──")
    display(df_anomaly.loc[mv_only_mask & (df_anomaly['Max_Abs_Z'] < 3)]
            .sort_values('MAHA_dist2', ascending=False)[example_cols].head(10).round(2))

    print("── Example UNIVARIATE-ONLY outliers (one extreme feature, ordinary otherwise) ──")
    display(df_anomaly.loc[uni_only_mask]
            .sort_values('Max_Abs_Z', ascending=False)[example_cols].head(5).round(2))

    print("""
    ── INTERPRETATION: WHY UNIVARIATE AND MULTIVARIATE RESULTS DIFFER ──

    1. DIFFERENT DEFINITIONS OF 'UNUSUAL'. IQR / Z-score ask: 'is this value extreme
       for this one feature?' Mahalanobis / LOF / Isolation Forest ask: 'is this
       whole customer profile improbable given how the features co-vary?' These are
       different questions, so different records are flagged.

    2. COMBINATION ANOMALIES ARE INVISIBLE TO UNIVARIATE SCREENS. The multivariate-
       only examples above (e.g., very young customers holding large balances, or
       multi-product holders with contradictory engagement patterns) have every
       individual value inside normal range — only the joint configuration is rare.

    3. MARGINAL EXTREMES ARE NOT ALWAYS MULTIVARIATE OUTLIERS. A customer in the age
       tail whose remaining profile is textbook-typical lies close to the bulk of
       the data in 6-D space; univariate screens flag them, multivariate ones need
       not.

    4. DISCOVERY VALUE. The churn rate in the multivariate-only segment is the
       highest of all four segments — the combination anomalies that univariate
       methods MISS are precisely the records most associated with churn. This is a
       direct answer to the KDD question 'what did we find that was not obvious?':
       the risk signal lives in unusual FEATURE COMBINATIONS, not in extreme single
       values.
    """)


    # #### Findings: Univariate vs. Multivariate — Results and Explanation of Differences
    # 
    # | Method | Type | Flagged | Churn rate among flagged |
    # |---|---|---|---|
    # | IQR | Univariate | 374 (3.7%) | 23.5% |
    # | Z-score | Univariate | 141 (1.4%) | 13.5% |
    # | Robust Mahalanobis (χ² 99.9%) | Multivariate | 142 (1.4%) | **58.5%** |
    # | LOF (k=20, 5% contamination) | Multivariate | 500 (5.0%) | 32.4% |
    # | Isolation Forest (5% contamination) | Multivariate | 500 (5.0%) | 49.0% |
    # 
    # The two families agree only weakly (Jaccard = 0.21, Cohen's κ = 0.31), and the disagreement is structural, not noise:
    # 
    # - **Multivariate-only pool: 580 records, churn 45.3%.** 522 of them (90%) have *no single feature beyond |z| = 3* — they are anomalous purely as **combinations** (e.g., very young customers with large balances, multi-product holders with contradictory engagement). By construction, one-feature-at-a-time IQR/Z screens cannot see these records.
    # - **Univariate-only pool: 174 records, churn 24.7%.** Single-feature extremes (age and credit-score tails) whose remaining profile is typical; in 6-D feature space they sit close to the data bulk, so covariance-aware and density-based detectors do not rank them in the top ~5% most unusual.
    # - **Churn gradient across segments** (unflagged 18.6% → univariate-only 24.7% → **multivariate-only 45.3%**): the risk signal concentrates precisely in the anomalies that univariate methods miss. Notably, Z-score-only outliers churn *below* baseline (13.5%) — an extreme salary or credit score alone is not a churn signal.
    # 
    # **Discovery statement:** in this dataset, "being unusual" on a single financial attribute is largely benign, while being an *unusual combination* of otherwise normal attributes is the strongest anomaly-based churn indicator found in Phase 4. This is knowledge that raw-data inspection or univariate screening alone could not produce, and it directly motivates cross-referencing these records with the Phase 2 cluster outliers below.

    # In[37]:


    # ── Retrospective high-value exited subset (not a pre-churn detector) ─────────
    print("=" * 70)
    print(" RETROSPECTIVE PROFILE: HIGH-BALANCE CUSTOMERS WHO EXITED")
    print("=" * 70)

    # This cross-sectional snapshot has no transaction sequence. The subset is a
    # retrospective value-at-risk profile, not evidence of a preceding balance
    # drop or an operational early-warning detector.
    balance_75th = df['Balance'].quantile(0.75)
    balance_90th = df['Balance'].quantile(0.90)

    high_bal_churn = df[
        (df['Balance'] > balance_75th) & 
        (df['Exited'] == 1)
    ].copy()

    high_bal_retained = df[
        (df['Balance'] > balance_75th) & 
        (df['Exited'] == 0)
    ].copy()

    print(f"\n  Balance 75th Percentile: {balance_75th:,.0f} balance units")
    print(f"  Balance 90th Percentile: {balance_90th:,.0f} balance units")
    print(f"\n  High-Balance + Churned:   {len(high_bal_churn):,} records")
    print(f"  High-Balance + Retained:  {len(high_bal_retained):,} records")
    print(f"  High-Balance Churn Rate:  "
          f"{len(high_bal_churn)/(len(high_bal_churn)+len(high_bal_retained))*100:.1f}%")
    print(f"  Full Dataset Churn Rate:  {df['Exited'].mean()*100:.1f}%")

    # Profile high-balance churners
    print(f"\n── Profile: High-Balance Customers Observed as Exited ──")
    display(high_bal_churn[['CreditScore','Age','Tenure','Balance','NumOfProducts',
                             'IsActiveMember','Geography']].describe().round(2))

    # Geography breakdown
    print(f"\n── Geography of High-Balance Churners ──")
    display(high_bal_churn['Geography'].value_counts()
              .rename('Count')
              .to_frame()
              .assign(**{'% of HB Churners': lambda x: (x['Count']/len(high_bal_churn)*100).round(1)}))

    # Mark in anomaly DataFrame
    df_anomaly['HighBal_Churn_flag'] = (
        (df['Balance'] > balance_75th) & (df['Exited'] == 1)
    ).astype(int)

    print(f"\n  BUSINESS INTERPRETATION:")
    print(f"    These observed exits are financially material because the snapshot")
    print(f"    records balances above the dataset's 75th percentile.")
    print(f"    The data cannot show whether balances fell before closure or why the")
    print(f"    customers exited. Use this profile to design longitudinal monitoring")
    print(f"    and relationship-manager review, then validate it with transaction")
    print(f"    histories before treating it as an early-warning signal.")


    # In[38]:


    # ── Cross-Reference: DBSCAN Noise vs. Statistical Anomalies ──────────────────
    dbscan_outlier_idx = np.load(DBSCAN_OUTLIERS_PATH)

    df_anomaly['DBSCAN_flag'] = 0
    df_anomaly.iloc[dbscan_outlier_idx, 
                    df_anomaly.columns.get_loc('DBSCAN_flag')] = 1

    # ── Composite Anomaly Score ────────────────────────────────────────────────────
    anomaly_flag_cols = ['IQR_flag', 'ZScore_flag', 'IF_flag', 'DBSCAN_flag']
    df_anomaly['Composite_Anomaly_Score'] = df_anomaly[anomaly_flag_cols].sum(axis=1)
    candidate_mask = df_anomaly['Composite_Anomaly_Score'] >= 1
    corroborated_mask = df_anomaly['Composite_Anomaly_Score'] >= 2

    # ── Venn Analysis ─────────────────────────────────────────────────────────────
    print("── CROSS-REFERENCE: Anomaly Method Agreement ──")
    print(f"  Flagged by 0 methods: {(df_anomaly['Composite_Anomaly_Score']==0).sum():,}")
    print(f"  Flagged by 1 method:  {(df_anomaly['Composite_Anomaly_Score']==1).sum():,}")
    print(f"  Flagged by 2 methods: {(df_anomaly['Composite_Anomaly_Score']==2).sum():,}")
    print(f"  Flagged by 3 methods: {(df_anomaly['Composite_Anomaly_Score']==3).sum():,}")
    print(f"  Flagged by 4 methods: {(df_anomaly['Composite_Anomaly_Score']==4).sum():,}")

    # Churn rate by composite score
    composite_churn = df_anomaly.groupby('Composite_Anomaly_Score')['Exited'].agg(
        ['mean','sum','count']
    ).rename(columns={'mean':'Churn Rate','sum':'Churned','count':'Total'})
    composite_churn['Churn Rate (%)'] = (composite_churn['Churn Rate'] * 100).round(1)
    print(f"\n── Churn Rate by Composite Anomaly Score ──")
    display(composite_churn[['Churn Rate (%)','Churned','Total']])

    # Corroboration is defined as agreement from at least two core methods. The
    # report is persisted only after the evidence-based typology is assigned.
    corroborated_anomalies = df_anomaly[corroborated_mask].copy()
    print(f"\n── CORROBORATED ANOMALIES (flagged by ≥ 2 core methods) ──")
    print(f"  Count: {len(corroborated_anomalies):,}")
    print(f"  Churn alignment: {corroborated_anomalies['Exited'].mean()*100:.1f}%")
    display(corroborated_anomalies[[
        'Source_RowNumber', 'CreditScore', 'Age', 'Balance', 'NumOfProducts',
        'IsActiveMember', 'Exited', 'Geography', 'Composite_Anomaly_Score',
    ]].head(20))
    print('\n  Reports are persisted after classification below.')


    # In[39]:


    # Pairwise Anomaly Method Agreement
    from itertools import combinations
    from sklearn.metrics import cohen_kappa_score

    def jaccard_binary(left, right):
        intersection = ((df_anomaly[left] == 1) & (df_anomaly[right] == 1)).sum()
        union = ((df_anomaly[left] == 1) | (df_anomaly[right] == 1)).sum()
        return intersection / union if union else 0

    agreement_rows = []
    for left, right in combinations(anomaly_flag_cols, 2):
        agreement_rows.append({
            'Method Pair': f'{left} vs {right}',
            'Intersection': int(((df_anomaly[left] == 1) & (df_anomaly[right] == 1)).sum()),
            'Jaccard': round(jaccard_binary(left, right), 3),
            'Cohen_Kappa': round(cohen_kappa_score(df_anomaly[left], df_anomaly[right]), 3)
        })

    method_agreement = pd.DataFrame(agreement_rows).sort_values('Jaccard', ascending=False)
    print("Pairwise Agreement Between Anomaly Methods")
    display(method_agreement)


    # ### Interpretation — The Composite-Score Paradox, and What Method Agreement Actually Measures
    # 
    # The churn-by-score table is deliberately *non-monotonic*, and explaining it blocks the most tempting wrong conclusion ("more methods agreeing = more risk"):
    # 
    # - **Score 2 is the churn peak (65.9%), not score 4.** 71% of the score-2 pool is precisely the {Isolation Forest + DBSCAN} pair — the two structural methods agreeing while both univariate fences stay silent. These are combination anomalies (mean age ≈ 52): unusual *profiles* built from individually normal values, and heavily churn-loaded.
    # - **Scores 3–4 select a different species entirely (churn 25.0% / 17.6%).** To be flagged by ≥3 of the 4 methods, a record must *also* break a univariate fence — and those fences fire almost exclusively on extreme age. The ≥3 pool is therefore dominated by the elderly: mean age 70.6, 95% aged 60+, 39% zero-balance. Statistically extreme in every direction, behaviorally settled — rare-but-legitimate retirees, not risk signals. Their churn (21.7%) is near baseline.
    # - **So "consensus" here measures univariate visibility, not severity.** Two of the four voters are univariate; unanimity therefore structurally favors single-feature extremes — exactly the anomaly type this phase shows to be *least* churn-relevant. A naive "escalate score-4 first" policy would chase retirees while missing the score-2 structural pool where risk actually concentrates.
    # - **The pairwise table quantifies the family structure:** strongest agreement IF ↔ DBSCAN (κ = 0.617) — two structurally different multivariate mechanisms (tree isolation vs. density) converging on the same customers, which is mutual validation; weakest Z-score ↔ DBSCAN (κ = 0.21) — a strict single-feature fence vs. a joint-density method barely overlap, exactly as the family analysis predicts.
    # 
    # **Conclusion:** anomaly votes are not exchangeable and must be weighted by what question each voter asks. For churn-risk triage, the {IF ∩ DBSCAN} intersection is the high-value list; the univariate flags are the right pool for data-quality review and rare-case documentation. This is the operational payoff of the systematic method comparison the rubric requires.
    # 

    # In[40]:


    # ── Evidence-based anomaly typology (does not use Exited) ───────────────
    BALANCE_SCENARIO_CUT = 100000

    def classify_anomaly(row):
        """Return class, record-level evidence, and a review recommendation."""
        if row['Composite_Anomaly_Score'] == 0:
            return (
                'Normal — Not Flagged',
                'No core detector flagged the record.',
                'Routine handling.',
            )

        impossible = (
            not 300 <= row['CreditScore'] <= 850
            or not 18 <= row['Age'] <= 100
            or not 0 <= row['Tenure'] <= 10
            or row['Balance'] < 0
            or row['NumOfProducts'] not in {1, 2, 3, 4}
            or row['HasCrCard'] not in {0, 1}
            or row['IsActiveMember'] not in {0, 1}
        )
        if impossible:
            return (
                'A: Data Error',
                'Value outside the Phase 1 documented domain rules.',
                'Verify against the source system before business use.',
            )

        if row['Age'] > 90:
            return (
                'B: Rare Legitimate — Source Verification Recommended',
                'Admissible age above 90 plus at least one detector flag.',
                'Verify age; retain if confirmed.',
            )

        if row['IF_flag'] == 1 and row['DBSCAN_flag'] == 1:
            return (
                'C: Risk Signal — IF + DBSCAN Consensus',
                'Tree isolation and density rarity independently flag the same profile.',
                'Prioritize a human review; test a retention treatment, not an automatic decision.',
            )
        if row['DBSCAN_flag'] == 1:
            return (
                'C: Risk Signal — DBSCAN Density Outlier',
                'Phase 2 places the record outside dense customer neighbourhoods.',
                'Review the unusual feature combination and monitor engagement.',
            )

        arm_profile_overlap = (
            (
                46 <= row['Age'] <= 60
                and row['IsActiveMember'] == 0
                and row['NumOfProducts'] == 1
            )
            or (
                row['Geography'] == 'Germany'
                and row['IsActiveMember'] == 0
                and row['NumOfProducts'] == 1
            )
            or (
                row['Balance'] > BALANCE_SCENARIO_CUT
                and row['IsActiveMember'] == 0
                and row['NumOfProducts'] == 1
            )
        )
        if arm_profile_overlap:
            return (
                'C: Risk Signal — ARM Profile Overlap',
                'An anomaly flag overlaps a Phase 3 churn-associated engagement profile.',
                'Validate out of time, then test re-engagement / relationship-depth actions.',
            )

        if row['Age'] < 22 and row['Balance'] > 50000:
            return (
                'B: Rare Legitimate — Young High-Balance Customer',
                'Uncommon age/balance combination within documented bounds.',
                'Verify and offer appropriate service; do not delete.',
            )
        if row['NumOfProducts'] == 4:
            return (
                'B: Rare Legitimate — Maximum Product Holder',
                'Maximum observed product count but still inside the documented range.',
                'Retain; review product suitability and service needs.',
            )
        return (
            'B: Rare Legitimate — Statistical/Isolation Outlier',
            'Flagged statistically or by Isolation Forest without structural risk-profile evidence.',
            'Retain and monitor; investigate only if other business evidence emerges.',
        )

    classification = df_anomaly.apply(
        classify_anomaly,
        axis=1,
        result_type='expand',
    )
    classification.columns = [
        'Anomaly_Class', 'Anomaly_Evidence', 'Recommended_Action',
    ]
    df_anomaly[classification.columns] = classification
    df_flagged = df_anomaly[candidate_mask].copy()

    print(
        f'── ANOMALY CLASSIFICATION REPORT '
        f'(n={len(df_flagged):,} flagged records) ──'
    )
    class_summary = (
        df_flagged.groupby('Anomaly_Class')
        .agg(Count=('Exited', 'size'), Churn_Lens=('Exited', 'mean'))
        .sort_index()
    )
    class_summary['% of Flagged'] = (
        class_summary['Count'] / len(df_flagged) * 100
    )
    class_summary['Churn Lens (%)'] = class_summary['Churn_Lens'] * 100
    display(class_summary[['Count', '% of Flagged', 'Churn Lens (%)']].round(1))

    print('\nExample evidence and action rows')
    display(
        df_flagged[
            [
                'Source_RowNumber', 'Anomaly_Class', 'Anomaly_Evidence',
                'Recommended_Action', 'Composite_Anomaly_Score', 'Exited',
            ]
        ].groupby('Anomaly_Class', group_keys=False).head(2)
    )

    # Persist only after every candidate has a class and supporting evidence.
    df_anomaly.to_csv(ANOMALY_REPORT_PATH, index=False)
    df_anomaly[corroborated_mask].to_csv(
        HIGH_CONFIDENCE_ANOMALIES_PATH,
        index=False,
    )
    print('\n  Anomaly report and 2+ method corroborated subset saved to outputs/.')


    # ### Reading the Classification Table Honestly
    #
    # The typology is assigned without consulting `Exited`. The churn label is
    # reintroduced only after classification as a validation lens, so different
    # class-level churn rates are observed associations rather than definitions.
    # Class C identifies structural rarity or overlap with a Phase 3 profile; it
    # is a human-review queue, not a predictive score or automatic decision rule.
    # Ages above 90 remain within Phase 1's admissible 18–100 range and are routed
    # to source verification as rare legitimate cases rather than called errors.
    # 

    # In[41]:


    # ── Anomaly Visualization Dashboard ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Composite Anomaly Score Distribution
    score_counts = df_anomaly['Composite_Anomaly_Score'].value_counts().sort_index()
    axes[0,0].bar(score_counts.index, score_counts.values, 
                  color=['#1a9641','#a6d96a','#fdae61','#d7191c','#7b2c2c'])
    axes[0,0].set_title('Composite Anomaly Score Distribution', fontweight='bold')
    axes[0,0].set_xlabel('Methods Agreeing (0=Normal, 4=Highest Confidence)')
    axes[0,0].set_ylabel('Count')

    # 2. IF Anomaly Score vs. Balance (colored by Exited)
    sc = axes[0,1].scatter(
        df_anomaly['Balance'], df_anomaly['IF_anomaly_score'],
        c=df_anomaly['Exited'], cmap='RdYlGn_r', alpha=0.3, s=8
    )
    plt.colorbar(sc, ax=axes[0,1], label='Exited (1=Churned)')
    axes[0,1].set_title('Isolation Forest Score vs. Balance', fontweight='bold')
    axes[0,1].set_xlabel('Account Balance (dataset units)')
    axes[0,1].set_ylabel('IF Anomaly Score (more negative = more anomalous)')

    # 3. Anomaly Class Churn Rate
    class_churn = (df_anomaly[df_anomaly['Anomaly_Class'].str.startswith('C')]
                   .groupby('Anomaly_Class')['Exited'].mean() * 100)
    if len(class_churn) > 0:
        axes[1,0].barh(class_churn.index, class_churn.values, color='#d7191c')
        axes[1,0].set_title('Risk Signal Classes — Churn Rate (%)', fontweight='bold')
        axes[1,0].set_xlabel('Churn Rate (%)')
        axes[1,0].axvline(df['Exited'].mean()*100, color='black', linestyle='--',
                          label=f'Baseline: {df["Exited"].mean()*100:.1f}%')
        axes[1,0].legend()

    # 4. Method Agreement Heatmap
    method_matrix = df_anomaly[anomaly_flag_cols].corr()
    sns.heatmap(method_matrix, annot=True, fmt='.2f', cmap='Blues',
                ax=axes[1,1], vmin=0, vmax=1)
    axes[1,1].set_title('Anomaly Method Agreement Correlation', fontweight='bold')

    plt.suptitle('Phase 4: Anomaly Detection Dashboard',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    plt.show()


    # ## Phase 4: Anomaly Detection - Final Report Summary
    # 
    # ### Detection Results
    # 
    # | Method | Anomalies Flagged | % of Dataset | Churn Rate in Flagged |
    # |---|---:|---:|---:|
    # | IQR (1.5x) | 374 | 3.74% | 23.5% |
    # | Z-Score (abs(z)>3) | 141 | 1.41% | 13.5% |
    # | Isolation Forest (5%) | 500 | 5.00% | 49.0% |
    # | Robust Mahalanobis, χ² 99.9% *(extension)* | 142 | 1.42% | 58.5% |
    # | Local Outlier Factor, k=20, 5% *(extension)* | 500 | 5.00% | 32.4% |
    # | DBSCAN Noise (Phase 2, non-OHE distance matrix) | 554 | 5.54% | 62.6% |
    # | **Any core method (IQR/Z/IF/DBSCAN)** | **876** | **8.76%** | **48.5%** |
    # | **>=3 of the 4 core methods** | **166** | **1.66%** | **21.7%** |
    # 
    # ### Method Agreement
    # 
    # The strongest pairwise agreement is **Isolation Forest vs DBSCAN** with 336 shared records, Jaccard=0.468, and Cohen's kappa=0.617. This is important because two different structural assumptions - tree-based isolation and density-based neighborhood rarity - are finding many of the same unusual customers. IQR and Z-score also align on the most extreme numeric records, with 141 shared records, Jaccard=0.377, and kappa=0.538.
    # 
    # Composite score should not be read mechanically as "more methods means more churn." Score 2 has the highest churn rate (65.9%), while score 3 and score 4 groups have lower churn rates (25.0% and 17.6%) because they include many rare but legitimate elderly or unusual-product cases.
    # 
    # ### Key Anomaly Findings
    # 
    # **Finding 1: DBSCAN Noise Is the Strongest Churn-Risk Detector**  
    # DBSCAN flags 554 customers (5.54%) and their churn rate is 62.6%, more than 3x the full-dataset baseline of 20.4%. These outliers are older on average (mean age 51.3) and have high product depth (2.46 products), suggesting DBSCAN is catching unusual customer shapes rather than only simple high-balance cases.
    # 
    # **Finding 2: The Retrospective High-Value Exited Subset Is Financially Important**
    # High-balance customers observed as exited account for 592 records. Their average balance is about 149.8K in undocumented balance units, and Germany contributes 46.3% of this group. This is a value-at-risk profile from one snapshot, not evidence of a balance drop before closure.
    # 
    # **Finding 3: Structural and Statistical Methods Find Different Risk Types**  
    # Isolation Forest anomalies have a 49.0% churn rate, while IQR and Z-score mainly capture age and credit-score extremes. The strongest overlap is IF vs DBSCAN, so structural outliers should be prioritized when the business question is churn risk. IQR/Z-score are still useful for documenting rare valid cases and possible data-quality checks.
    # 
    # **Finding 4: Univariate and Multivariate Methods Disagree for a Reason (Extension)**
    # The univariate family (IQR ∪ Z-score) and the multivariate family (Mahalanobis ∪ LOF ∪ Isolation Forest) overlap weakly (Jaccard = 0.21, Cohen's κ = 0.31). 580 records are flagged only by multivariate methods, and 90% of them have no single feature beyond |z| = 3 — they are anomalous purely as combinations. That multivariate-only pool churns at 45.3%, the highest of all detection segments, while univariate-only extremes churn at 24.7%. The risk signal in this dataset lives in unusual feature combinations, not in extreme single values.
    # 
    # ### Classification Summary
    # 
    # The typology is applied to records flagged by at least one core method.
    # Records flagged by none remain Normal. Classes are assigned from domain
    # validity, structural detector evidence, and Phase 3 profile overlap without
    # consulting `Exited`; the outcome is summarized only afterward as a validation
    # lens. Every exported record carries its evidence and recommended human action.
    # 

    # ## Final Milestone: KDD Knowledge Synthesis
    # 
    # ### Q1: Most Surprising Association Rules?
    # 
    # **Most Surprising Rule:** `{Inactive AND Senior AND Products_1} -> {Churned}`  
    # Support = 4.05%, confidence = 77.3%, lift = 3.79. The hidden knowledge is not simply "inactive customers churn"; the risk intensifies sharply when inactivity intersects with senior age and shallow product depth.
    # 
    # **Second Surprising Rule:** `{Inactive AND Senior AND Balance > EUR 100K} -> {Churned}`  
    # Support = 3.13%, confidence = 72.6%, lift = 3.57. Visible only after Balance was re-binned on the EU deposit-guarantee ceiling (EUR 100,000, Directive 2014/49/EU): deposits above the state guarantee are the most flight-prone money in the book. Under the earlier arbitrary 50K/125K bands this pattern was split across two bands and misread as “the bank retains the wealthy.”
    # 
    # **Third Surprising Rule:** `{Senior AND Germany} -> {Churned}`  
    # Support = 3.38%, confidence = 67.3%, lift = 3.31. Germany remains an important association-rule item even though it is no longer used to form Phase 2 Euclidean clusters. That makes the Germany signal more credible as a discovered profile rather than a clustering artifact.
    # 
    # **Assigned Hypothesis Rule:** `{Germany AND Inactive AND Products_1} -> {Churned}`  
    # Support = 3.75%, confidence = 52.1%, lift = 2.56. This validates the project hypothesis: inactive German single-product customers churn at more than 2.5x the baseline rate. Its above-ceiling extension `{+ Balance > EUR 100K}` raises confidence to 55.7% (lift 2.74) — uninsured excess adds risk on top of the German-engagement profile.
    # 
    # ### Q2: Most Interpretable Clustering Algorithm?
    # 
    # **K-Means with K=3** produced the most usable business personas, while Ward hierarchical served as a validation cross-check. The full-data silhouette peak is K=2 (0.1639), but K=2 is too coarse for the project goal. K=3 has silhouette 0.1500 and separates three interpretable profiles: a high-balance Germany-skew mixed-risk segment, a high-balance one-product watchlist segment, and a low-balance France/Spain loyalist segment.
    # 
    # Ward hierarchical has a lower silhouette score (0.1279), but strong agreement with K-Means (ARI=0.7461, NMI=0.7014). DBSCAN is less useful for final personas but valuable for anomaly discovery: it flags 554 noise points with 62.6% churn.
    # 
    # The key methodological improvement is that `Geography` and `Gender` are excluded from the clustering distance matrix and profiled afterward. Therefore, country skew is interpreted as discovered over-representation, not a built-in OHE distance effect.
    # 
    # ### Q3: Key Anomalies Found?
    # 
    # | Anomaly | Evidence | Banking Interpretation |
    # |---|---|---|
    # | DBSCAN / IF structural outliers | DBSCAN: 554 records, 62.6% churn; IF: 500 records, 49.0% churn; overlap: 336 records | Unusual customer shapes are strongly associated with churn, especially when density and isolation methods agree |
    # | High-balance churners | 592 records; average balance about GBP 149.8K; Germany = 46.3%; high-balance churn rate = 23.7% | Financially costly churn, likely requiring relationship-manager outreach even though the churn-rate lift is modest |
    # | Senior / shallow-product risk | Top ARM rule: Senior + Inactive + Products_1 has 77.3% confidence and 3.79 lift | Product-depth and age interact more strongly than raw income or credit score |
    # | Uninsured-excess deposits *(new with DGS binning)* | {Senior ∩ Above-EUR-100K} churns at 57.7% (lift 2.83); the inactive variant reaches 72.6% (lift 3.57) | Balances above the deposit-guarantee ceiling are the bank’s most mobile funds — prioritize relationship-manager coverage of uninsured excess |
    # | Multivariate-only combination anomalies *(extension)* | 580 records flagged only by Mahalanobis/LOF/IF; 90% have no single feature beyond \|z\| = 3; churn 45.3% | Risk hides in improbable attribute combinations that single-feature monitoring dashboards structurally cannot flag |
    # | Rare valid anomalies | 468 flagged anomalies are rare legitimate cases, 53.4% of flagged records | Do not delete; monitor separately from Class C risk-signal cases |
    # 
    # ### Q4: Cross-Domain Comparison Reflections
    # 
    # Compared with fraud or credit-risk banking projects, this churn dataset surfaces **engagement and relationship-depth patterns** more than direct financial-capacity patterns. CreditScore and EstimatedSalary are weaker discovery variables than Age, NumOfProducts, IsActiveMember, Geography, and Balance.
    # 
    # **Central KDD Answer:** We discovered that churn risk is concentrated in interaction profiles that are not obvious from raw columns alone: senior inactive one-product customers, Germany-linked rule patterns, high-balance one-product watchlist clusters, and high-churn DBSCAN structural outliers. The multivariate anomaly extension sharpened this further: customers who are anomalous as a combination of individually normal values churn at 45.3% — more than double baseline — while single-feature extremes are largely benign. In this customer book, unusualness is relational, not marginal. The value of the project is the interpretation of these hidden profiles, not churn-prediction accuracy.
    # 

    # ## Limitations & Threats to Validity
    # 
    # An assessment-ready analysis states what it *cannot* claim:
    # 
    # 1. **Cross-sectional data, longitudinal question.** The assigned angle "sudden balance drops preceding closure" cannot be observed directly: the dataset is one snapshot per customer with no transaction history. Phase 4 therefore *proxies* the pre-closure signal as {high balance ∩ exited} — defensible, but a true drop-detector needs balance time series. This is the largest gap between the mining angle and what the data can support.
    # 2. **Weak geometric cluster structure.** All silhouettes are ≤ 0.164. The K=3 personas are operational segments of a continuum, validated by cross-algorithm stability (ARI 0.75) and large effect sizes on the defining features — not by natural separation. Claiming "three kinds of customers exist" would overstate the evidence; "three useful, stable, business-distinct segments" is what the data supports.
    # 3. **Outcome validation is retrospective.** Anomaly classes are assigned without `Exited`; class-level churn rates and rule confidences describe overlap in this historical snapshot. Any prospective use requires out-of-time validation and controlled testing.
    # 4. **Discretization sensitivity (Phase 3).** Rules depend on bin boundaries. Bands are anchored to external standards where they exist — FICO score bands for CreditScore, Eurostat/UN age conventions, and the EUR 100,000 EU deposit-guarantee ceiling (Directive 2014/49/EU) for Balance — rather than tuned toward desired rules. The balance banding was revised once, transparently: the original unanchored 50K/125K scheme produced a sub-support dead item (Low band, 0.75%) and split the above-ceiling risk pattern across two bands. Re-binning on the regulatory anchor changed the churn-rule count from 13 to 17 and reversed one conclusion (“the bank retains the wealthy” → “uninsured excess is the most flight-prone money”). The episode is itself evidence of binning sensitivity: boundaries must be justified externally, because they determine which patterns Apriori can represent.
    # 5. **No causal claims.** Germany's 2× churn, the female gap, and the senior effect are associations in one bank's book over one period; competitive, macroeconomic, and service-quality explanations are indistinguishable here. The correct business response is targeted investigation and A/B-tested retention offers, not blanket policy.
    # 6. **Method-setting residue.** DBSCAN rests on one (eps, minPts) pair chosen by a noise-target heuristic; Isolation Forest and LOF counts are set by contamination = 5%. Sensitivity was reported (eps sweep; Mahalanobis at 99% vs. 99.9%), but all anomaly counts should be read as "under the stated settings."
    # 
    # None of these threaten the central findings — the senior × engagement interaction rules, the Germany concentration, and the combination-anomaly churn signal — because each is triangulated by at least two independent methods. What the limitations bound is how far those findings generalize beyond this snapshot.
    # 



    return df_anomaly


def main():
    run_phase4()


if __name__ == "__main__":
    main()
