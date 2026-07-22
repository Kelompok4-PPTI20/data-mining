# Group 5 — Bank Customer Churn KDD

This repository implements the five-phase Data Mining project as **knowledge discovery**, not a churn-prediction benchmark. `Exited` is used only for post-discovery profiling and validation.

## Reproducible notebook run

Use Python 3.11.9 and install the pinned dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open Jupyter from the repository root with that environment, then run:

1. `notebooks/first-eda.ipynb` — optional orientation only
2. `notebooks/phase1_preprocessing.ipynb`
3. `notebooks/phase2_clustering.ipynb`
4. `notebooks/phase3_association_rules.ipynb`
5. `notebooks/phase4_anomaly_detection.ipynb`
6. `notebooks/phase5_knowledge_presentation.ipynb`

The phase notebooks fail early when a required upstream artifact is missing. Quantitative report values are consolidated in `outputs/evaluation_metrics.csv` and summarized in `appendix.md`.

## Interactive dashboard

After running Phases 1–4:

```powershell
.\.venv\Scripts\python.exe visualization\prepare_data.py
Set-Location visualization
..\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:8050`. The Phase 5 notebook includes the report handoff, presentation outline, QA gate, and a server-side interaction benchmark. Browser latency should also be measured on the presentation machine.
