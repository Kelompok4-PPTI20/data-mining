"""Loads the precomputed dashboard cache (run prepare_data.py first)."""

import json
from pathlib import Path

import pandas as pd

DD = Path(__file__).resolve().parent / "dashboard_data"

if not DD.exists():
    raise SystemExit(
        "dashboard_data/ not found - run `python prepare_data.py` once "
        "from the visualization/ folder before starting the app."
    )

REC = pd.read_csv(DD / "records.csv")
with open(DD / "metrics.json") as f:
    M = json.load(f)
with open(DD / "rules.json") as f:
    R = json.load(f)

BASELINE = M["kpi"]["churn_rate"]  # 20.37 (%)
