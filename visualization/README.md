# Phase 5 — Knowledge Discovery Dashboard (Python Dash)

Interactive dashboard presenting the full KDD pipeline (Phases 1–4) for the
Bank Customer Churn project, built for a non-technical audience. Visuals implement the **KDD Design Spec v1.0**
(cool-slate neutrals, single Cobalt accent, Geist / Geist Mono, 56px header +
248px sidebar shell): KPI cards, plain-language insight strips under every chart, and an explicit answer to the central question — *what did we
discover that was not already obvious from the raw data?*

## Run it

From this `visualization/` folder, using the project's environment
(`requirements.txt` at repo root already covers everything):

```bash
python prepare_data.py   # once — builds dashboard_data/ from Phases 1-4 outputs
python app.py            # then open http://127.0.0.1:8050
```

`prepare_data.py` only needs re-running if the notebook regenerates the files
in `data/processed/` or `outputs/`. Heavy steps are checkpointed — if it gets
interrupted, just run it again.

## Rubric coverage (5.5 Excellent)

- **Interactive < 100 ms** — every figure is precomputed at startup; callbacks
  swap ready-made objects (measured 2–30 ms server-side). Interactions:
  tab navigation, feature/dimension explorers (Phase 1), algorithm & color
  switching on the cluster map (Phase 2), rule selector with business
  commentary (Phase 3), outlier-map color modes (Phase 4), plus hover detail
  on all 10,000-point maps.
- **All required visualizations** — cluster maps (PCA, 3 algorithms), rule
  network + rule-quality scatter + ranked top-10 table, outlier plots
  (method comparison, consensus paradox, outlier map, class donut), and key
  distributions (retained vs churned, churn by dimension).
- **Accessible to non-technical audience** — every card has a "what am I
  looking at" subtitle and a "What this tells us" conclusion; the Knowledge
  Report tab answers the central discovery question directly, covers the four
  Mining Expo questions, and states limitations honestly.

## Files

| File | Role |
|---|---|
| `app.py` | Dash app: layout, 6 tabs, callbacks |
| `figures.py` | All Plotly figures (built once at import) |
| `components.py` | Cards, KPI tiles, insight strips, tables, personas |
| `theme.py` | Palette + shared Plotly template (Google-inspired) |
| `data.py` | Loads the precomputed cache |
| `prepare_data.py` | Assembles `dashboard_data/` from Phase 1–4 artifacts |
| `assets/styles.css` | Looker-style UI (auto-loaded by Dash) |
| `dashboard_data/` | Generated cache: `records.csv`, `metrics.json`, `rules.json` |

## Numbers trace to the notebook

Everything shown is derived from `notebooks/notebook.ipynb` outputs
(`data/processed/`, `outputs/`) with `random_state=42`, full 10,000 records.
Recomputed pieces were verified against the notebook: Ward silhouette 0.1279,
ARI 0.7461, NMI 0.7014, K-sweep silhouettes to 4 decimals, DBSCAN noise 554.
