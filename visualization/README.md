# Phase 5 — Visualization & Knowledge Presentation (Group 5)

Deliverables for the final KDD phase: an interactive dashboard, the Knowledge
Discovery Report, and the 10-minute group presentation. Everything here is
generated from the Phase 1–4 pipeline (`notebooks/notebook.ipynb`) and answers
the central question: *what did we discover that was not already obvious from
the raw data?*

## Contents

| File | What it is |
|---|---|
| `app.py` | Interactive dashboard (Python Dash) — 5 tabs: The Discovery, Customer Segments, Churn Rules, Anomalies, Distributions |
| `prepare_data.py` | One-time data prep: FAMD 2-D coordinates, parsed association rules, summary KPIs → `viz_data/` |
| `viz_data/` | Precomputed dashboard inputs (generated; re-runnable) |
| `assets/style.css` | Dashboard styling |
| `assets/previews/` | Static PNG exports of key views (used in the deck; regenerable) |
| `Knowledge_Discovery_Report.docx` | Final written report in plain business language (PDF copy alongside) |
| `Group5_Presentation.pptx` | 12-slide, 10-minute deck — timed speaker notes on every slide, expo Q&A prep in slide 12's notes |

## Running the dashboard

```bash
cd visualization
python prepare_data.py   # once, after the notebook has produced data/processed + outputs
python app.py            # then open http://127.0.0.1:8050
```

Dependencies are the project root `requirements.txt` (dash, plotly, pandas,
prince). Tableau Public / Power BI were acceptable alternatives, but Dash keeps
the whole pipeline in Python and the dashboard reproducible from code.

**Styling:** the UI follows the Google Looker Studio visual language — white
Material cards on a grey canvas, Roboto typography, scorecard KPIs, and blue
page tabs — while remaining a pure Python Dash app (`assets/style.css` carries
the theme). Optional: `pip install orjson` makes Plotly's JSON serialization
faster for the 10K-point cluster map.

## How the <100 ms interactivity bar is met

1. Heavy computation (FAMD projection, rule parsing, profile aggregation)
   happens once in `prepare_data.py`, never at click time.
2. Every callback filters preloaded in-memory dataframes — no disk or network I/O.
3. Every control has a small finite input space, so callback results are
   memoized (`lru_cache`) and pre-warmed at startup: measured callback time
   after warm-up is ~0.01 ms, and full HTTP round-trips measure ~15 ms.

## Rubric mapping

- Cluster maps → “Customer Segments” tab (FAMD projection, persona profiles)
- Rule networks → “Churn Rules” tab (network + ranked table, lift slider)
- Outlier plots → “Anomalies” tab (IF score scatter, method agreement, classes)
- Key distributions → “Distributions” tab (market filter)
- Central discovery question → answered on the landing tab, in the report
  (§11), and on the closing slide
- Interactive, not static: all five tabs respond to controls in-memory

## Presentation timing (10:00)

Slide notes carry the script and cues: 1–2 intro (1:15), 3 method (0:45),
4 personas (1:00), 5–8 the four discoveries (3:30), 9 rulebook (0:45),
10 dashboard/demo (0:45), 11 recommendations (0:50), 12 the answer + Q&A
hand-off (1:10). Slide 12's notes include prepared answers to the four
Mining Expo questions.
