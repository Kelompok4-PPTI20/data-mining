# D’NEXTRY — Bank Customer Churn Knowledge Discovery

This repository contains an end-to-end **Knowledge Discovery in Databases (KDD)** project on a retail banking customer churn dataset.

The project focuses on discovering:

- Meaningful customer segments
- Non-obvious churn-associated profiles
- Unusual customer records and structural risk signals
- Actionable insights for customer retention

This is a **knowledge discovery project**, not a churn-prediction benchmark. No classification model is trained to predict whether a customer will churn.

---

## Project Overview

The analysis applies a five-phase KDD workflow to the **Bank Customer Churn Dataset**, which contains demographic, financial, and bank-relationship information for retail banking customers from France, Germany, and Spain.

| Item | Description |
|---|---|
| Domain | Retail banking customer churn |
| Records | 10,000 customers |
| Original columns | 14 |
| Analytical columns | 11 |
| Countries | France, Germany, and Spain |
| Overall churn rate | 20.4% |
| Analysis type | Unsupervised and descriptive knowledge discovery |
| Main tools | Python, pandas, scikit-learn, mlxtend, SciPy, Plotly Dash |

Three non-analytical columns are removed during preprocessing:

- `RowNumber`
- `CustomerId`
- `Surname`

The remaining attributes describe customer demographics, financial condition, product ownership, relationship depth, activity status, and churn status.

---

## Research Objective

The project addresses the following central question:

> **What useful customer knowledge can be discovered that is not immediately visible from the raw data?**

The analysis specifically aims to:

1. Identify meaningful customer segments using clustering.
2. Discover combinations of customer characteristics associated with churn.
3. Detect unusual customer profiles using multiple anomaly-detection methods.
4. Translate technical results into practical banking actions.

---

## KDD Workflow

```text
Raw Dataset
    │
    ▼
Phase 1 — Data Understanding and Preprocessing
    ├── Data quality assessment
    ├── Feature relevance analysis
    ├── Path A: scaling for clustering
    └── Path B: categorical binning for Apriori
           │
           ├───────────────────────┐
           ▼                       ▼
Phase 2 — Clustering       Phase 3 — Association Rules
    ├── K-Means                ├── Apriori
    ├── Hierarchical           ├── Support
    ├── DBSCAN                 ├── Confidence
    └── Cluster profiling      ├── Lift
           │                   └── Redundancy filtering
           └──────────┬────────────┘
                      ▼
Phase 4 — Anomaly Detection
    ├── IQR
    ├── Z-score
    ├── Isolation Forest
    ├── Robust Mahalanobis Distance
    ├── Local Outlier Factor
    ├── DBSCAN cross-reference
    └── Business classification
                      │
                      ▼
Phase 5 — Visualization and Knowledge Presentation
    ├── Interactive dashboard
    ├── Knowledge Discovery Report
    └── Final presentation