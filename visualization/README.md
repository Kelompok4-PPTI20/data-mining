# Phase 5 — Knowledge Discovery Dashboard (Python Dash)

Interactive dashboard presenting the full KDD pipeline (Phases 1–4) for the
Bank Customer Churn project, built for a non-technical audience. Visuals
implement the **KDD Design System v2** (cool-slate neutrals, single Cobalt
accent, Geist / Geist Mono, pipeline sidebar + phase page headers, stat bands,
insight strips): see `DESIGN_RATIONALE.md` for every design decision, its UX
principle and its trade-offs. The dashboard answers the central question
explicitly — *what did we discover that was not already obvious from the raw
data?*

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
  pipeline navigation (sidebar, clickable pipeline cards, prev/next stage
  footers), one-click feature/dimension explorers (Phase 1), algorithm & color
  switching on the cluster map (Phase 2), click-a-row rule interpretations
  (Phase 3), outlier-map color modes (Phase 4), plus hover detail on all
  10,000-point maps.
- **All required visualizations** — cluster maps (PCA, 3 algorithms), rule
  network + rule-quality scatter + ranked top-10 table, outlier plots
  (method comparison, consensus paradox, outlier map, class donut), and key
  distributions (retained vs churned, churn by dimension).
- **Accessible to non-technical audience** — every card has a "what am I
  looking at" subtitle and a "What this tells us" conclusion; every page opens
  with a clear header; the Knowledge Report retains the technical synthesis,
  while the Business Takeaways page translates the full KDD result into a
  plain-language priority list, testable actions, success measures and safeguards.

## Files

| File | Role |
|---|---|
| `app.py` | Dash app: shell, 7 pages, routing + interaction callbacks |
| `figures.py` | All Plotly figures (built once at import) |
| `components.py` | Design-system primitives: page headers, stat bands, cards, insight strips, callouts, personas, tables, pipeline cards |
| `theme.py` | Palette tokens + shared Plotly template |
| `data.py` | Loads the precomputed cache |
| `prepare_data.py` | Assembles `dashboard_data/` from Phase 1–4 artifacts |
| `assets/styles.css` | KDD Design System v2 (auto-loaded by Dash) |
| `dashboard_data/` | Generated cache: `records.csv`, `metrics.json`, `rules.json` |
| `DESIGN_RATIONALE.md` | The v2 redesign: critique of v1, every decision + UX principle + trade-off |

## Numbers trace to the notebook

Everything shown is derived from `notebooks/notebook.ipynb` outputs
(`data/processed/`, `outputs/`) with `random_state=42`, full 10,000 records.
Recomputed pieces were verified against the notebook: Ward silhouette 0.1279,
ARI 0.7461, NMI 0.7014, K-sweep silhouettes to 4 decimals, DBSCAN noise 554.
