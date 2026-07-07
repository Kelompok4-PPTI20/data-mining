# KDD Dashboard — Design Rationale (v2 redesign)

This documents the redesign of the Phase-5 dashboard: what was weak in v1, what
changed, why each change is better, the UX principle behind it, and its
trade-offs. Content, figures, numbers and all interactions are preserved —
this is a redesign of *how the knowledge is presented*, not of the knowledge.

---

## 1 · Honest critique of v1

v1 was solid on tokens (good palette, type pairing, 8px grid) but weak on
**hierarchy and scanability** — the two things a KDD dashboard lives on:

1. **Everything had the same visual weight.** Card titles were 14px, the page
   had no H1, five identical KPI cards competed equally, and every section
   looked like every other. A user could not tell in one glance what mattered
   most on a page.
2. **Wall-of-text heroes.** The Overview hero was a ~150-word paragraph that
   *duplicated* the four discovery cards directly below it. Reading it twice
   is work; skipping it is guilt.
3. **Boxes inside boxes.** Standalone conclusions ("Verdict", "Glossary",
   "Reproducibility") were insight strips wrapped in cards: double borders,
   double backgrounds, pure container noise.
4. **Navigation carried no meaning.** Sidebar entries were flat labels with
   decorative counters; nothing told you where you were in the 5-phase KDD
   pipeline, the pipeline strip at the bottom of the Overview *looked* like
   navigation but wasn't clickable, and there was no way to walk the stages
   in order.
5. **Decorative color on numbers.** KPI values were tinted blue/purple/amber
   as ornament. When color sometimes means risk and sometimes means nothing,
   it stops meaning anything.
6. **Redundant table content.** The rule table spent a column on ten identical
   "Churned" chips, and rule selection lived in a disconnected dropdown below
   the table.
7. **Overloaded chart labels.** e.g. the PCA x-axis title contained an entire
   reading guide ("LEFT = higher balance | RIGHT = more products").
8. **Small annoyances.** Scroll position persisted across page switches;
   dropdowns hid 6–7 options behind two clicks; no reduced-motion support.

---

## 2 · The redesign, decision by decision

### 2.1 Sidebar → KDD pipeline navigator
- **Changed:** Flat radio labels became buttons with a phase badge (◆, 1–5),
  a title, and a phase subtitle ("Phase 3 · association mining"). Footer
  relabeled PROVENANCE.
- **Why better:** The sidebar now *teaches the methodology while you navigate*
  — you always know which KDD stage you are in and what comes next. Directly
  serves "monitor KDD progress" and "navigate between stages efficiently".
- **Principle:** Match between system and the real world (Nielsen #2);
  recognition over recall.
- **Trade-off:** Slightly taller nav items; on <1024px the subtitles are
  dropped to keep the horizontal strip compact.

### 2.2 Three coordinated ways to move, one routing model
- **Changed:** (a) sidebar for random access, (b) the Overview pipeline cards
  are now real click targets ("Open →"), (c) every page ends with prev/next
  stage buttons. All funnel into a single `route` store + one callback;
  page switches scroll to top.
- **Why better:** KDD review is inherently sequential (P1 → P5). Reviewers can
  now walk the pipeline like a document without returning to the sidebar; the
  pipeline strip stopped being a dead picture of navigation.
- **Principle:** Flexibility & efficiency of use; "don't show a map you can't
  travel".
- **Trade-off:** Three entry points to the same destination is mild redundancy
  — acceptable because they serve different moments (orientation, reading
  flow, jumping).

### 2.3 Persistent dataset status in the header
- **Changed:** Header right side now shows "10,000 customers · 20.4% baseline
  churn · 0 nulls · 0 dupes" as chips, on every page.
- **Why better:** "Understand dataset status quickly" was the top user need;
  the baseline (20.4%) is the denominator of literally every chart, so it
  should never be more than a glance away.
- **Principle:** Visibility of system status.
- **Trade-off:** Hidden below 1150px to protect the title; the Overview stat
  band still carries the same facts.

### 2.4 Page headers with real hierarchy
- **Changed:** Every phase page opens with kicker ("PHASE 2 OF 5 ·
  SEGMENTATION VIA CLUSTERING") → 23px H1 → one-line purpose → neutral fact
  chips (e.g. "K-Means · Ward · DBSCAN", "ARI 0.75").
- **Why better:** v1's largest recurring text was a 14px card title, so pages
  had no summit. Now each page states in ~2 seconds what stage you're in,
  what it claims, and its headline parameters.
- **Principle:** Visual hierarchy through typographic scale; inverted pyramid.
- **Trade-off:** ~90px of vertical space per page — cheap for what it buys.

### 2.5 Five KPI cards → one stat band
- **Changed:** KPI rows merged into a single region with hairline dividers
  (1px-gap grid trick); one stat per band may be marked `primary` (larger,
  tinted background) — e.g. baseline churn on the Overview, top lift on
  Phase 3.
- **Why better:** Five borders + five shadows = ten lines of container noise
  around five numbers. One region reads as one scannable row, and "primary"
  finally makes the most important metric *look* most important.
- **Principle:** Gestalt common region/proximity; Tufte's data-ink ratio;
  emphasis requires contrast.
- **Trade-off:** Individual stats are less individually "card-like"; the band
  wraps to a 2-col grid (then 1-col) on small screens.

### 2.6 The one-container rule (no boxes in boxes)
- **Changed:** New `callout` component for standalone conclusions; all
  card-wrapped-insight constructs (Phase-2 verdict, Phase-3 glossary,
  Phase-4 agreement, Report reproducibility) converted.
- **Why better:** Same voice, half the chrome. Conclusion blocks now sit
  directly on the canvas as first-class content instead of double-framed
  afterthoughts.
- **Principle:** Minimize non-data ink; consistent component semantics
  (insight = inside a card; callout = standalone).
- **Trade-off:** None observed.

### 2.7 Hero discipline — and cutting duplicated prose
- **Changed:** Dark heroes are reserved for the two narrative bookends: the
  question (Overview) and the answer (Report). The Overview hero shrank from
  ~150 words to ~55 plus four hero-stats (77% · 2× · 7.6% · 45%) that anchor
  the four discovery cards below. The Phase-3 hypothesis block became a light
  green **success banner** with stat chips (confidence/lift/support/n) instead
  of a third dark hero.
- **Why better:** The old hero duplicated the discovery cards nearly verbatim
  — readers paid twice for the same content. The stats now do the summarizing
  and the cards do the explaining. Dark surfaces regain meaning because they
  are rare; green-tinted = status, which is what a confirmed hypothesis is.
- **Principle:** Progressive disclosure; DRY for content; semantic color;
  emphasis through scarcity.
- **Trade-off:** Someone who reads *only* the hero gets less detail — by
  design, the detail is one glance lower.

### 2.8 Dropdowns → segmented pills (Phase 1 explorers)
- **Changed:** The 6-feature and 7-dimension dropdowns became segmented pill
  controls (same component ids/values, callbacks untouched).
- **Why better:** All options are visible at once and comparison becomes
  one-click instead of open-scan-click-repeat — exactly the "inspect
  preprocessing results" loop this page exists for.
- **Principle:** Recognition over recall; minimize interaction cost.
- **Trade-off:** Pills cost horizontal space; viable at ≤7 short options
  (which is why the pattern is used only here).

### 2.9 Rule table = the selector (direct manipulation)
- **Changed:** The rule-selection dropdown is gone; table rows are clickable
  (pointer cursor, hover, cobalt selected state with inset accent bar) and the
  interpretation panel updates below. The "THEN" column was deleted — every
  consequent is "Churned", stated once in the card subtitle. The detail panel
  gained the support value.
- **Why better:** The deliverable table *is* the natural selection surface;
  selecting where you're reading removes a disconnected control and a
  redundant column. Rows are far larger click targets than a dropdown.
- **Principle:** Direct manipulation; Fitts's law; don't repeat invariants.
- **Trade-off:** Row-click is less self-announcing than a labeled dropdown —
  mitigated by the subtitle instruction, cursor, hover and a preselected
  row (A). Keyboard: rows are focusable (tabIndex) but Dash `Tr` clicks are
  mouse events; keyboard users still get the full default interpretation.
  Documented, accepted for a course deliverable.

### 2.10 Semantic-only color on numbers
- **Changed:** Stat values are ink by default; red strictly = churn/risk
  (baseline, risk signals, top lift), green = verified-good (0 nulls,
  0 dupes), amber = attention (retained outliers, rare-but-valid). Decorative
  blue/purple values removed.
- **Why better:** When color is reserved for state, a red number is an alarm
  rather than a theme choice. Scanning for risk becomes pre-attentive.
- **Principle:** Color encodes meaning, consistently or not at all;
  accessibility (contrast-checked tokens unchanged).
- **Trade-off:** Bands look quieter — intended.

### 2.11 Typography, measure and rhythm
- **Changed:** Real heading semantics (H1 page titles, H2 sections, H3 card
  titles); section rows gained a muted right-hand meta slot ("explicitly
  graded", "top 10 rules, ranked by lift"); prose blocks capped at ~76–100ch;
  36px section rhythm on the 8px grid; `prefers-reduced-motion` respected.
- **Why better:** Semantic headings improve screen-reader outlines and visual
  scanning simultaneously; capped measure fixes the full-width paragraph
  problem on wide monitors; section meta answers "why am I looking at this"
  without another sentence in the body.
- **Principle:** Typographic hierarchy; optimal line length (45–100ch);
  consistent spatial rhythm.
- **Trade-off:** None meaningful.

### 2.12 Chart labeling
- **Changed:** PCA axes now read "PC1 · 23% of variance"; the balance/products
  reading guide moved into the card subtitle, where prose belongs.
- **Why better:** Axis titles are for identifying axes; teaching text set in
  11px axis type was the hardest-to-read sentence on the page.
- **Principle:** Each surface does one job; layered reading (title → subtitle
  → chart → conclusion).
- **Trade-off:** The guide is one line further from the axes.

---

## 3 · Deliberately kept

- **The insight-strip pattern** ("What this tells us") — it is the project's
  interpretive voice and the rubric's substance; v2 only standardizes its
  titles by kind (default / Read with care / Key risk / Confirmed).
- **All Plotly figure types** — horizontal bars for ranked categorical
  comparisons, the rule network, the elbow/silhouette pair, the donut triage:
  all appropriate encodings; redesigning them would be change for its own sake.
- **The token system** (cool-slate neutrals, single cobalt accent, Geist) and
  the **precomputed-figures architecture** that keeps every interaction well
  under the 100ms rubric budget.
- **Every number and analytical claim.** Copy was tightened only where it
  duplicated adjacent content.

## 4 · Functionality inventory (nothing lost)

Six pages; feature-distribution explorer (6 features); churn-by-dimension
explorer (7 dimensions); cluster-map algorithm × color switching (6 states);
outlier-map color modes (2); rule interpretations (all 10, now row-click);
decision log, action table, personas, Expo Q&A, limitations, reproducibility —
all present. Navigation gained: clickable pipeline cards, prev/next stage
walking, scroll-to-top on page change, persistent dataset status.

## 5 · Verification

- `import app` builds all pages and figures; all pure callbacks exercised for
  every input value; every page serializes to Dash JSON.
- Live server test: routing callback returns the correct page + active nav
  state over HTTP; algorithm-switch callback returns the DBSCAN figure.
- Rubric check (§6 "Excellent", Presentation): interactive < 100ms unchanged
  (precomputed swaps), all required visualizations present, the central
  discovery question is answered explicitly on the Overview and the Report.
