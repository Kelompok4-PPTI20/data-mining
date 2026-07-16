"""
Bank Customer Churn: Knowledge Discovery Dashboard (Phase 5, Group 5)
=======================================================================
Interactive Plotly Dash app presenting the full KDD pipeline (Phases 1–4)
to a non-technical audience: what was mined, what was found, and what it means.

    1) python prepare_data.py     (once; builds dashboard_data/)
    2) python app.py              -> http://127.0.0.1:8050

Design notes (v2; see DESIGN_RATIONALE.md for every decision)
--------------------------------------------------------------
* Information architecture mirrors the KDD pipeline: the sidebar is a stage
  navigator, every page opens with a phase header, and prev/next footers walk
  the pipeline in order. Pipeline cards on the Overview are click targets.
* All figures are precomputed at startup; callbacks only swap ready objects,
  so every interaction responds well under the 100 ms rubric budget.
* Every chart card still ends with a plain-language conclusion strip -
  the dashboard answers the discovery question, it does not just plot data.
"""

from dash import ALL, Dash, Input, Output, ctx, dcc, html, no_update

import components as C
import figures as F
import theme as T
from data import BASELINE, M, R

card, insight, callout, chip, section, stat = (
    C.card, C.insight, C.callout, C.chip, C.section, C.stat)

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="Churn KDD Dashboard: Group 5",
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap"
    ],
)

# ===========================================================================
# NAVIGATION MODEL: one ordered pipeline, three entry styles
# (sidebar · pipeline cards · prev/next footers), one routing callback
# ===========================================================================

NAV = [
    ("overview", "The Discovery", "Executive summary", "◆"),
    ("ph1", "Data & Preprocessing", "Phase 1 · quality & features", "1"),
    ("ph2", "Customer Segments", "Phase 2 · clustering", "2"),
    ("ph3", "Churn Rules", "Phase 3 · association mining", "3"),
    ("ph4", "Anomalies", "Phase 4 · outlier triage", "4"),
    ("report", "Knowledge Report", "Phase 5 · the answer", "5"),
    ("business", "Business Takeaways", "Plain-language action summary", "B"),
]
ORDER = [p for p, _, _, _ in NAV]

# ===========================================================================
# PAGE 1: OVERVIEW
# ===========================================================================

overview = html.Div([
    # The central question, answered in two sentences; the four numbers below
    # anchor the four discovery cards; details live there, not in the hero.
    html.Div([
        html.Div("THE CENTRAL KDD QUESTION", className="hero-kicker"),
        html.H1("What did we discover that was not already obvious "
                "from the raw data?", className="hero-q"),
        html.P([
            "That churn risk in this bank is ", html.B("relational, not marginal"),
            ": no single column predicts leaving (max |r| = 0.29), but specific "
            "combinations of ordinary values multiply risk up to 4× the 20.4% "
            "baseline. The four discoveries below are that hidden knowledge; "
            "each one invisible to a column-at-a-time report.",
        ], className="hero-a"),
        html.Div([
            C.herostat("77%", "churn · inactive single-product seniors"),
            C.herostat("2×", "Germany vs France & Spain"),
            C.herostat("7.6%", "churn floor at exactly two products"),
            C.herostat("45%", "churn among combination-anomalies"),
        ], className="herostats"),
    ], className="hero"),

    C.statband([
        stat("10,000", "Customers analyzed", "14 raw features · France · Germany · Spain"),
        stat(f"{BASELINE:.1f}%", "Baseline churn", "2,037 left; the yardstick for every "
             "number here", tone=T.CRITICAL, primary=True),
        stat("3", "Customer personas", "found by clustering, method-stable (ARI 0.75)"),
        stat(f"{M['kpi']['n_churn_rules']}", "High-lift churn rules",
             "all ≥ 2.5× baseline risk · top rule 3.79×"),
        stat("876", "Anomalies triaged", "406 risk signals · 468 rare-but-valid · 2 data errors"),
    ]),

    section("The four discoveries", meta="none visible in a one-column report"),
    html.Div([
        card("01 · Churn compounds along the senior path",
             "Churn rate as conditions stack (each bar adds one condition)",
             C.graph(F.OV_SENIOR),
             insight(["Age is the strongest single signal (seniors churn at 51%), but the "
                      "knowledge is the ", html.B("interaction"), ": inactivity roughly "
                      "doubles a senior's risk, and a single product pushes it to ",
                      html.B("77%, or 3.8× baseline"), "."])),
        card("02 · Germany churns at double the rate",
             "Churn rate by country: Germany loses 814 of 2,509 customers",
             C.graph(F.OV_GEO),
             insight(["The gap survives every control: even restricted to inactive "
                      "single-product customers, Germany still churns at 52.1% (lift 2.56×). "
                      "This points to a ", html.B("structural problem in the German operation"),
                      ", not an unlucky customer mix."], kind="bad")),
        card("03 · Products follow a U-curve, not a line",
             "Churn rate by number of products held",
             C.graph(F.OV_PROD),
             insight(["Two products = safety (7.6%). One = weak anchor (27.7%). Three or four "
                      "= near-certain exit (83% / 100%). Correlation rates this feature "
                      "'weak' (r = 0.05); ", html.B("the entropy lens in Phase 1 is what "
                      "caught it"), "."])),
        card("04 · Risk hides in combinations, not extremes",
             "Churn rate by type of statistical unusualness (Phase 4)",
             C.graph(F.OV_COMBO),
             insight(["Customers flagged ONLY as unusual combinations of individually normal "
                      "values churn at 45.3%; single-value extremes are largely benign "
                      "(24.7%). ", html.B("Monitoring columns one at a time structurally "
                      "cannot see the riskiest customers"), "."], kind="warn")),
    ], className="grid g2"),

    section("How we got here", meta="the KDD pipeline; click any phase to open it"),
    C.pipeline_cards(),
    C.phase_footer(nxt=("ph1", "Phase 1 · Data & Preprocessing")),
], className="page")

# ===========================================================================
# PAGE 2: PHASE 1 · DATA & PREPROCESSING
# ===========================================================================

phase1 = html.Div([
    C.page_header(
        "PHASE 1 OF 5 · DATA UNDERSTANDING & PREPROCESSING",
        "A verified base and two tailored data paths",
        "Quality audit, transformations and feature relevance; every decision "
        "justified in writing before any mining ran.",
        chips=["scaled matrix → clustering", "discretized items → Apriori"]),

    C.statband([
        stat("0", "Missing values", "no imputation needed; verified, not assumed",
             tone=T.SUCCESS),
        stat("0", "Duplicate records", "10,000 unique customers", tone=T.SUCCESS),
        stat("3", "Columns dropped", "RowNumber, CustomerId, Surname (IDs & PII)"),
        stat("374", "Outliers retained", "real segments, not noise; re-examined in Phase 4",
             tone=T.WARNING),
        stat("6 / 10", "Features selected", "by correlation + entropy; the rest kept for "
             "unsupervised discovery"),
    ]),

    section("Know the data", meta="one-click explorers; pick a feature or dimension"),
    html.Div([
        card("Feature distributions: retained vs churned",
             "Red = the 2,037 customers who left; blue = the 7,963 who stayed. "
             "Bars show % of each group, so the shapes are directly comparable.",
             [C.pills("ph1-dist-dd", [{"label": c, "value": c} for c in F.DIST_FEATURES],
                      "Age"),
              dcc.Graph(id="ph1-dist-graph", config=T.GRAPH_CONFIG,
                        style={"height": "300px"}),
              html.Div(id="ph1-dist-insight")]),
        card("Who churns? Churn rate by customer group",
             "The dashed line is the 20.4% bank average; red bars sit clearly above it, "
             "green clearly below.",
             [C.pills("ph1-dim-dd", [{"label": v, "value": k}
                                     for k, v in F.DIM_LABELS.items()], "Geography"),
              dcc.Graph(id="ph1-dim-graph", config=T.GRAPH_CONFIG,
                        style={"height": "300px"}),
              html.Div(id="ph1-dim-insight")]),
    ], className="grid g2"),

    section("Feature selection", meta="the rubric's two lenses, side by side"),
    card("Correlation AND entropy",
         "Left: linear association with churn (Pearson |r|, undefined for nominal "
         "Geography/Gender, which is precisely why a second lens is required). "
         "Right: information-theoretic relevance (mutual information + Shannon information gain).",
         C.graph(F.FEATSEL_FIG),
         insight(["The two lenses agree at the top (Age, IsActiveMember), but ", html.B(
             "NumOfProducts is nearly invisible to correlation (|r| = 0.05) and #1–2 by "
             "information gain"), " because its U-shaped churn relation cancels out in a "
             "linear coefficient. Correlation-only selection would have discarded one of the "
             "most informative features in the dataset. Selected: Age, NumOfProducts, "
             "Geography, IsActiveMember, Balance, Gender."])),

    section("Every decision, justified", meta="the rubric grades justification, not just action"),
    card("Preprocessing decision log",
         "Each choice below is documented in the Phase 1 report; the two data paths "
         "(scaled matrix for clustering, discretized items for Apriori) come from these decisions.",
         C.decisions_table()),

    C.phase_footer(prev=("overview", "The Discovery"),
                   nxt=("ph2", "Phase 2 · Customer Segments")),
], className="page")

# ===========================================================================
# PAGE 3: PHASE 2 · SEGMENTS
# ===========================================================================

phase2 = html.Div([
    C.page_header(
        "PHASE 2 OF 5 · SEGMENTATION VIA CLUSTERING",
        "Three personas, named only on what truly separates them",
        "K-Means, Ward and DBSCAN on 8 behavioral dimensions; geography, gender "
        "and churn deliberately held out, then reintroduced as validation lenses.",
        chips=["K-Means · Ward · DBSCAN", "K = 3", "ARI 0.75"]),

    html.Div([C.persona_card(1), C.persona_card(0), C.persona_card(2)], className="grid g3"),

    section("The cluster map", meta="10,000 customers · hover any dot"),
    card("Customer segmentation in 2-D (PCA projection of the clustering space)",
         "Each dot is one customer, projected from the 8-D behavioral space the algorithms "
         "actually clustered (PC1+PC2 = 36% of variance; orientation, not proof). "
         "Reading guide: PC1 runs from higher balance (left) to more products (right); "
         "PC2 mixes salary and tenure. Switch algorithm to see how three definitions of "
         "'group' read the same customers.",
         [html.Div([
             C.control("Algorithm", C.pills("ph2-algo", [
                 {"label": "K-Means (K=3): the personas", "value": "kmeans"},
                 {"label": "Hierarchical Ward (K=3)", "value": "ward"},
                 {"label": "DBSCAN (density)", "value": "dbscan"}], "kmeans")),
             C.control("Color by", C.pills("ph2-color", [
                 {"label": "Segment", "value": "segment"},
                 {"label": "Churn status", "value": "churn"}], "segment")),
         ], className="controls"),
             dcc.Graph(id="ph2-pca-graph", config=T.GRAPH_CONFIG,
                       style={"height": "430px"})],
         insight(["K-Means and Ward find nearly the same three segments (ARI 0.75) ordered "
                  "along PC1: single-product high-balance → multi-product high-balance → "
                  "zero-balance. DBSCAN instead splits on the density valley between 1 and 2 "
                  "products and isolates 554 'noise' customers (red ✕); ", html.B(
                      "those noise customers churn at 62.6%, the strongest anomaly-churn "
                      "signal in the whole project"), " (picked up again in Phase 4). The "
                  "banded look is real data: products only take 4 discrete values."])),

    section("How K was chosen", meta="both criteria, honestly read"),
    html.Div([
        card("Elbow & silhouette",
             "The two required validity checks disagree, and the disagreement is informative.",
             C.graph(F.ELBOW_SIL_FIG),
             insight(["Silhouette peaks at K=2 (0.164), but that split is just 'has money vs "
                      "doesn't' (the bimodal Balance column restated). The elbow shows no sharp "
                      "knee, only diminishing returns from K=3–5. ", html.B(
                          "K=3 is chosen inside both windows"), ": one level finer than the "
                      "trivial split; exactly where product depth enters and the segmentation "
                      "becomes actionable. All silhouettes ≤ 0.16 mean these are descriptive "
                      "segments of a continuum, not natural islands; validated instead by "
                      "cross-algorithm stability (ARI 0.75) and the effect sizes on the right."],
                     kind="warn", title="Honest reading")),
        card("What actually separates the segments",
             "Effect sizes across the 3 clusters (Kruskal-Wallis ε² / Cramér's V; the larger "
             "of the two per feature). At n = 10,000 everything is 'significant'; only effect "
             "size is allowed to name a persona.",
             C.graph(F.EFFECT_FIG),
             insight(["Balance (0.73) and product depth (0.71) define the segments; geography "
                      "(0.33) differs strongly as a ", html.B("post-clustering discovery"),
                      ". Age, tenure, salary, credit score and activity do NOT separate these "
                      "clusters, so the persona names deliberately never mention them."])),
    ], className="grid g2"),

    html.Div([
        card("Segment fingerprints (snake plot)",
             "How far each segment's average sits from the bank average, in standard "
             "deviations; all features on one comparable scale.",
             C.graph(F.SNAKE_FIG),
             insight(["The three lines only fan apart on Balance and NumOfProducts; the same "
                      "verdict as the effect sizes, from a third angle. C1 (purple) = high "
                      "balance, one product; C2 (teal) = zero balance, multi-product; "
                      "C0 (cobalt) = high on both."])),
        card("Churn and geography by segment",
             "Left: the churn validation lens (never used to form clusters). Right: where "
             "each segment's customers live.",
             C.graph(F.CLUSTER_CHURN_FIG),
             insight(["Churn spans 13.6% → 25.6% across segments; a spread the weak "
                      "silhouette knows nothing about. And Germany concentrates in the "
                      "high-balance clusters (52% of C0) while being ", html.B(
                          "almost absent (0.7%) from the loyalist zero-balance segment"),
                      "; German customers essentially never hold the bank's 'safe' "
                      "salary-account profile. Discovered after clustering, so it is a "
                      "finding, not an artifact."])),
    ], className="grid g2"),

    callout(["K-Means K=3 gives the most interpretable personas (silhouette 0.150). Ward "
             "reproduces them (ARI 0.746, NMI 0.701); the structure is method-stable, not a "
             "K-Means artifact. Single linkage collapses into one chain (99.8% of customers "
             "in one branch); itself evidence that the book is a continuum. DBSCAN is the "
             "wrong tool for personas here (it just splits 1-product vs 2-product) but the "
             "best discovery tool: its noise set is the most churn-dense subgroup any method "
             "found. Each algorithm was used for what it is structurally good at."],
            title="Cross-algorithm verdict (Mining Expo Q2)"),

    C.phase_footer(prev=("ph1", "Phase 1 · Data & Preprocessing"),
                   nxt=("ph3", "Phase 3 · Churn Rules")),
], className="page")

# ===========================================================================
# PAGE 4: PHASE 3 · ASSOCIATION RULES
# ===========================================================================

hyp = M["hypothesis"]

phase3 = html.Div([
    C.page_header(
        "PHASE 3 OF 5 · ASSOCIATION RULE MINING",
        f"{M['kpi']['n_churn_rules']} rules that more than double churn risk",
        "Apriori on domain-anchored bins; support, confidence and lift computed for "
        "every rule, filtered to the non-trivial, high-lift findings.",
        chips=["min support 3%", "confidence ≥ 50%", "lift ≥ 1.5"]),

    # the assigned hypothesis; a status banner, not a second dark hero
    html.Div([
        html.Div("ASSIGNED HYPOTHESIS: CONFIRMED", className="banner-kicker"),
        html.H2("“Customers from Germany holding only one product who are inactive "
                "represent a strong churn profile.”", className="banner-title"),
        html.P(f"{hyp['antecedent_n']} customers match the profile and {hyp['churned_n']} of "
               "them churned; verified by direct computation on the raw data. The German "
               "retention problem is real, and Phase 3 shows it is one of TWO independent "
               "risk vectors (the other: senior age) that compound when combined.",
               className="banner-body"),
        html.Div([
            C.herostat(f"{hyp['confidence_pct']}%", "confidence (vs 20.4% baseline)"),
            C.herostat(f"{hyp['lift']}×", "lift"),
            C.herostat(f"{hyp['support_pct']}%", "support"),
            C.herostat(f"{hyp['antecedent_n']:,}", "customers match the IF"),
        ], className="herostats"),
    ], className="banner banner--success"),

    C.statband([
        stat("4,105", "Frequent itemsets", "Apriori, min support 3% (≥ ~300 customers)"),
        stat("645", "Rules pass filters", "confidence ≥ 50% and lift ≥ 1.5"),
        stat("17", "Churn-consequent rules", "the rubric needs 10; every one ≥ 2.5× baseline"),
        stat("3.79×", "Strongest lift", "inactive senior, 1 product → 77.3% churn",
             tone=T.CRITICAL, primary=True),
    ]),

    callout(["With a 20.4% base rate, demanding confidence ≥ 50% mathematically forces lift "
             "≥ 2.45, so ", html.B("every rule that survived more than doubles churn risk"),
             ". Only 17 of 645 rules concern churn: that is the filter working, not a "
             "shortage. The other 628 are structural co-occurrences (geography ↔ balance "
             "bands etc.), kept in the saved file for transparency. Re-binning balance on "
             "the EUR 100K deposit-guarantee ceiling (Directive 2014/49/EU) raised the churn-"
             "rule count from 13 to 17; five rules now involve above-ceiling balances."],
            title="Why so few rules survive"),

    section("The rule network", meta="how the top-10 rules share attributes"),
    html.Div([
        card("How the top-10 rules connect",
             "Blue circles = customer attributes (bigger = appears in more rules). Diamonds = "
             "rules A–J (redder = higher lift; hover for details). Every rule points to churn.",
             C.graph(F.RULE_NETWORK_FIG),
             insight(["One attribute sits at the heart of almost every rule: ", html.B(
                 "Senior (46–60)"), ". Churn risk compounds when senior age meets inactivity, "
                 "single-product holdings, female gender, German geography or a balance above "
                 "the EUR 100K deposit-guarantee ceiling. The bank is not "
                 "losing customers at random; it is losing ", html.B(
                     "a specific, describable population"), "."])),
        card("Rule quality at a glance",
             "Each bubble is a rule: right = more reliable (confidence), up = stronger vs "
             "baseline (lift), bigger = more customers, darker = more conditions.",
             C.graph(F.RULE_SCATTER_FIG),
             insight(["Rules stack up along the dotted frontier because lift = confidence ÷ "
                      "20.4%. The prize is the top-right: rule A (3 conditions, 77% "
                      "confidence) describes 405 real churners; specific enough to action, "
                      "big enough to matter."])),
    ], className="grid g2"),

    section("The deliverable", meta="top 10 rules, ranked by lift"),
    card("Top-10 churn rules with business interpretation",
         "Every rule's consequent is 'customer churned'. Support floor 3% means the weakest "
         "rule still describes ~300 customers; no micro-segment flukes. "
         "Click any row for its plain-language interpretation.",
         [C.rule_table(selected="A"),
          html.Div(C.rule_detail("A"), id="ph3-rule-detail")]),

    callout([html.B("Support"), " = share of ALL customers matching rule + churn. ",
             html.B("Confidence"), " = of customers matching the IF, the % who churned. ",
             html.B("Lift"), " = confidence ÷ 20.4% baseline; how many times more likely. ",
             html.B("Conviction"), " = how much more often the rule would have to fail if IF "
             "and churn were independent (higher = stronger). These are historical "
             "associations, not causal claims; the correct response is targeted "
             "investigation and A/B-tested retention offers."],
            title="Reading the metrics"),

    C.phase_footer(prev=("ph2", "Phase 2 · Customer Segments"),
                   nxt=("ph4", "Phase 4 · Anomalies")),
], className="page")

# ===========================================================================
# PAGE 5: PHASE 4 · ANOMALIES
# ===========================================================================

xr = M["cross_ref"]
u = M["uni_mv"]

phase4 = html.Div([
    C.page_header(
        "PHASE 4 OF 5 · ANOMALY & OUTLIER DETECTION",
        "Six detectors, one triage: error, rare-but-valid, or risk",
        "IQR, Z-score, Isolation Forest, DBSCAN, LOF and Mahalanobis systematically "
        "compared, every flagged record classified with evidence.",
        chips=["IQR + Z-score", "Isolation Forest", "cross-referenced with Phase 2"]),

    C.statband([
        stat("876", "Customers flagged", "by ≥ 1 of the 4 core methods (8.8% of book)",
             primary=True),
        stat("406", "Risk signals (C)", "churn-linked patterns → escalate to retention",
             tone=T.CRITICAL),
        stat("468", "Rare but valid (B)", "churn just 4.1%; SAFER than average; do not delete",
             tone=T.WARNING),
        stat("2", "Suspected data errors (A)", "ages 91–92; manual review, nothing depends "
             "on them"),
        stat("336", "IF ∩ DBSCAN overlap", "two structural methods agreeing · κ = 0.617"),
    ]),

    section("Six detectors, systematically compared"),
    card("Who finds the risky anomalies?",
         "All six methods, ranked by churn rate among the customers they flag. Blue = "
         "multivariate (judge the whole profile); gray = univariate (judge one value at a "
         "time). Note IF/LOF fix their count at 5% by construction; IQR/Z discover theirs.",
         C.graph(F.METHOD_FIG),
         insight(["A clean split: the structural, whole-profile methods (DBSCAN 62.6%, "
                  "Mahalanobis 58.5%, Isolation Forest 49.0%) flag churn-dense customers, "
                  "while single-value screens flag benign extremes (IQR 23.5%, Z-score "
                  "13.5%; ", html.B("below baseline"), "). Different methods answer "
                  "different questions; for churn risk, trust the structural family."])),

    html.Div([
        card("The consensus trap",
             "Churn rate by how many of the 4 core methods (IQR, Z-score, Isolation Forest, "
             "DBSCAN) agree a customer is anomalous.",
             C.graph(F.COMPOSITE_FIG),
             insight(["More votes ≠ more risk. The peak is at score 2 (65.9%); mostly the "
                      "IF + DBSCAN pair agreeing while both univariate fences stay silent. "
                      "Score 3–4 requires breaking a univariate fence, which fires almost "
                      "only on extreme age; flagging settled retirees (mean age 71, churn "
                      "~18–25%). ", html.B("A naive 'escalate the most-flagged first' policy "
                      "would chase retirees and miss the real risk pool.")],
                     kind="warn", title="The paradox")),
        card("Combinations beat extremes",
             f"Churn by anomaly family. Of the {u['n'][3]} multivariate-only customers, "
             f"{u['mv_only_hidden']} ({u['mv_only_hidden_pct']:.0f}%) have NO single value "
             "beyond |z| = 3; every number looks normal; only the combination is rare.",
             C.graph(F.UNIMV_FIG),
             insight(["The two families barely overlap (Jaccard 0.21, κ 0.31) because they "
                      "ask different questions. The risk gradient is 18.6% → 24.7% → 45.3%; "
                      "shows churn concentrating exactly in the anomalies univariate screens ",
                      html.B("structurally cannot see"), ": young customers with large "
                      "balances, multi-product holders with contradictory engagement. This "
                      "is the Phase-4 discovery."], kind="bad")),
    ], className="grid g2"),

    section("The outlier map", meta="10,000 customers · hover any dot"),
    card("Every customer by balance and Isolation-Forest score",
         "Dots below the dashed line are the 5% most isolable profiles. Color by anomaly "
         "class to see the triage, or by churn to see how risk concentrates toward the bottom.",
         [html.Div([C.control("Color by", C.pills("ph4-color", [
             {"label": "Anomaly class (triage)", "value": "class"},
             {"label": "Churn status", "value": "churn"}], "class"))], className="controls"),
          dcc.Graph(id="ph4-scatter", config=T.GRAPH_CONFIG, style={"height": "430px"})],
         insight(["The most anomalous region (bottom) is dominated by red risk-signal cases "
                  "at BOTH ends of the balance axis: high-balance pre-churn departures on "
                  "the right (592 churners averaging £149.8K; 46% German), and unusual "
                  "zero-balance profiles on the left. Amber rare-valid cases sit just below "
                  "the threshold; unusual, but safe."])),

    section("Classification & action", meta="the anomaly typology, applied to all 876 records"),
    html.Div([
        card("Every flagged record classified",
             "The rubric's three classes, applied to all 876 flagged records.",
             C.graph(F.CLASS_DONUT_FIG)),
        card("Subtypes",
             "The four risk-signal templates the bank should monitor prospectively.",
             C.graph(F.SUBTYPE_FIG)),
    ], className="grid g2-narrow"),
    card("Recommended actions per class", None,
         [C.action_table(),
          html.Div("* Class C churn = 100% by construction: the typology uses the observed "
                   "churn label retrospectively as supporting evidence, as the brief "
                   "prescribes. Deployed prospectively, expect the Phase-3/4 lift levels "
                   "(≈2.5–3.8×), never 100%; this figure must not be quoted as model "
                   "performance.", className="footnote")]),

    section("Cross-reference with Phase 2", meta="explicitly graded"),
    html.Div([
        card("Where DBSCAN's noise lives across the personas",
             "The 554 Phase-2 density outliers, mapped into the three K-Means segments.",
             C.graph(F.CROSSREF_FIG),
             insight(["The noise concentrates (10.6%) inside C0; the multi-product "
                      "Germany-skew segment, and is rarest in the loyalist C2 (1.7%). "
                      "Cluster outliers and statistical anomalies point at the same "
                      "neighbourhood of customers: unusual multi-product, older, "
                      "high-engagement-contradiction profiles."])),
        callout([f"IF ∩ DBSCAN share {xr['if_dbscan_overlap']} customers (Jaccard 0.47, "
                 f"κ 0.617, churn {xr['if_dbscan_churn']}%); two structurally different "
                 "multivariate mechanisms (tree isolation vs density) converging on the same "
                 "people is mutual validation. Weakest pair: Z-score vs DBSCAN (κ 0.21); a "
                 "strict single-value fence and a joint-density method barely overlap, "
                 "exactly as the family analysis predicts. Anomaly votes must be weighted by "
                 "the question each voter asks: the IF ∩ DBSCAN intersection is the "
                 "high-value retention list; univariate flags are the data-quality and "
                 "rare-case documentation pool."],
                title="Agreement between anomaly views"),
    ], className="grid g2"),

    C.phase_footer(prev=("ph3", "Phase 3 · Churn Rules"),
                   nxt=("report", "Phase 5 · Knowledge Report")),
], className="page")

# ===========================================================================
# PAGE 6: KNOWLEDGE REPORT
# ===========================================================================

report = html.Div([
    html.Div([
        html.Div("KNOWLEDGE DISCOVERY REPORT: THE DIRECT ANSWER", className="hero-kicker"),
        html.H1("We discovered that churn risk is relational: it lives in profiles, "
                "not in columns.", className="hero-q"),
        html.P([
            "Raw-data inspection shows a 20% churn rate and mild demographic tilts. Mining "
            "the same 10,000 customers end-to-end revealed four pieces of knowledge, none "
            "visible in a univariate report: (1) a compounding ", html.B("senior × engagement "
            "interaction"), " that escalates from 51% to 77% churn as conditions stack; "
            "(2) a ", html.B("structural German retention problem"), " that doubles churn "
            "independently of customer mix; (3) a customer book organized by ",
            html.B("balance × product depth"), "; whose largest segment (41.7%) keeps "
            "six-figure balances anchored by only one product, and (4) the fact that ",
            html.B("unusual combinations of normal values"), " predict churn (45.3%) far "
            "better than extreme single values (24.7%). The value of this project is the "
            "interpretation of these hidden profiles; not prediction accuracy.",
        ], className="hero-a"),
    ], className="hero"),

    section("Findings & recommended actions"),
    card(None, None, [
        C.finding("1", "The bank is hemorrhaging seniors with shallow product engagement",
                  "Senior (46–60) is the dominant antecedent in every top rule. Inactive "
                  "single-product seniors churn at 77.3% (lift 3.79×, 405 customers). Action: "
                  "retention call before a second consecutive inactive quarter; bundled second "
                  "product; senior-specific engagement program."),
        C.finding("2", "Germany has a structural retention problem",
                  "32.4% churn vs ~16% elsewhere; 46% of high-balance churners are German; "
                  "German seniors churn at 67% regardless of activity. Action: investigate "
                  "product fit and service quality in the German operation specifically; "
                  "this is not a customer-mix effect."),
        C.finding("3", "The riskiest mainstream segment is high-balance / single-product",
                  "Persona C1: 4,168 customers (41.7%) with ~£120K average balance and "
                  "exactly one product; churns at 25.6%, the highest of the three segments. "
                  "Money without product depth is unanchored. Action: cross-sell into C1 "
                  "before the money leaves; measure product depth, not balance, as the "
                  "loyalty KPI."),
        C.finding("4", "Monitor combinations, not thresholds",
                  "Customers anomalous only as combinations churn at 45.3%, and 90% of them "
                  "trip no single-value alarm. The 554 DBSCAN noise customers churn at 62.6%. "
                  "Action: add a multivariate anomaly score (IF/DBSCAN-style) to the CRM "
                  "watchlist alongside the existing per-column limits."),
    ]),

    section("Mining Expo: the four questions"),
    card(None, None, [
        C.qa("Q1 · Which association rules were the most surprising, and why?",
             [html.B("{Inactive ∩ Senior ∩ 1 product} → churn"), " (77.3% confidence, lift "
              "3.79); not because inactivity matters, but because the AGE interaction "
              "transforms it: younger inactives usually re-engage, senior inactives leave "
              "for good. Second: ", html.B("{Inactive ∩ Senior ∩ balance > EUR 100K} → churn"),
              " (72.6%, lift 3.57); visible only after balance was re-binned on the EU "
              "deposit-guarantee ceiling: money above the state guarantee is the most "
              "flight-prone in the book. Third: ", html.B("{Senior ∩ Germany} → churn"),
              " (67.3%, lift 3.31); it survives despite geography being excluded from "
              "clustering distance, making the German signal a discovered profile rather "
              "than an artifact. The assigned hypothesis {Germany ∩ Inactive ∩ 1 product} "
              "was confirmed at 52.1% confidence, lift 2.56 (55.7% for its above-ceiling "
              "variant)."]),
        C.qa("Q2 · Which clustering method produced the most interpretable segments?",
             ["K-Means at K=3; chosen over the silhouette-peak K=2, which merely restates "
              "the bimodal balance column. Ward hierarchical validates the partition "
              "(ARI 0.746); DBSCAN is the better discovery tool (its 554 noise points churn "
              "at 62.6%) but a worse persona tool (it only splits on product count). Honest "
              "caveat: all silhouettes ≤ 0.164; these are stable, business-distinct "
              "operational segments of a continuum, not natural species."]),
        C.qa("Q3 · What anomalies were found, and what do they suggest in a real banking "
             "context?",
             ["Three species: (a) 406 risk signals; high-balance pre-churn departures "
              "(avg £149.8K, 46% German: relationship-manager outreach), disengaged "
              "single-product churners, and density outliers; (b) 468 rare-but-valid "
              "profiles (settled elderly, zero-balance) churning at just 4.1%; flagging "
              "them for 'cleaning' would have destroyed real segments; (c) 2 suspected data "
              "errors (ages 91–92) for manual verification. The banking lesson: risk hides "
              "in unusual profiles, so monitoring should be multivariate."]),
        C.qa("Q4 · How do the findings compare to other banking domains?",
             ["Fraud and credit-risk datasets (Groups 3/8, 4) typically surface financial-"
              "capacity variables. Our churn book is the opposite: CreditScore and Salary "
              "carry almost zero signal, while engagement and relationship-depth variables "
              "(Age, NumOfProducts, IsActiveMember, Geography, Balance) carry it all. Same "
              "KDD pipeline, different knowledge; the meaning of 'anomaly' is domain-shaped: "
              "here it is a disengaging customer, in fraud it is a transaction pattern."]),
    ]),

    section("Limitations: what we cannot claim"),
    card(None, None, [html.Ul([
        html.Li([html.B("Snapshot data, longitudinal question. "), "The assigned 'sudden "
                 "balance drop before closure' cannot be observed in one snapshot per "
                 "customer; Phase 4 proxies it as {high balance ∩ exited}. A true "
                 "drop-detector needs transaction time series."]),
        html.Li([html.B("Weak geometric separation. "), "All silhouettes ≤ 0.164: three "
                 "useful, stable, business-distinct segments; not three natural species."]),
        html.Li([html.B("Retrospective constructs. "), "Rule confidences (52–77%) and "
                 "Class-C rates describe this historical snapshot; deployed prospectively, "
                 "expect regression toward the lift values."]),
        html.Li([html.B("Discretization sensitivity. "), "Rules depend on bin boundaries "
                 "(Senior = 46–60), fixed from domain conventions before mining."]),
        html.Li([html.B("No causal claims. "), "Germany's 2× churn and the senior effect are "
                 "associations in one bank's book over one period. Correct response: targeted "
                 "investigation and A/B-tested offers, not blanket policy."]),
        html.Li([html.B("Method-setting residue. "), "DBSCAN counts depend on (eps=1.25, "
                 "minPts=10); IF/LOF on contamination=5%. Sensitivity was reported; every "
                 "count reads 'under the stated settings'."]),
    ], className="tight"),
        html.Div("None of these threaten the central findings; each is triangulated by at "
                 "least two independent methods. The limitations bound how far the findings "
                 "generalize beyond this snapshot.", className="footnote")]),

    callout(["Pipeline: notebooks/notebook.ipynb (Phases 1–4, random_state = 42, full "
             "10,000 records, no sampling) → prepare_data.py (assembles this dashboard's "
             "cache; recomputed values verified against the notebook: Ward ARI 0.7461, "
             "NMI 0.7014, silhouettes to 4 decimals) → app.py (Plotly Dash). Stack: pandas, "
             "scikit-learn, mlxtend, SciPy, Plotly Dash."],
            title="Reproducibility"),

    C.phase_footer(prev=("ph4", "Phase 4 · Anomalies"),
                   nxt=("business", "Business Takeaways")),
], className="page")

# ===========================================================================
# PAGE 7: BUSINESS TAKEAWAYS
# An interpretation layer for general audiences, deliberately not "Phase 6".
# ===========================================================================

_business_top_rule = R["top10"][0]
_business_top_match_n = round(
    _business_top_rule["customers"] / (_business_top_rule["confidence_pct"] / 100))
_business_geo = dict(zip(M["churn_by"]["Geography"]["labels"],
                         M["churn_by"]["Geography"]["churn_pct"]))
_business_products = dict(zip(M["churn_by"]["NumOfProducts"]["labels"],
                              M["churn_by"]["NumOfProducts"]["churn_pct"]))
_business_watchlist = M["clusters"]["1"]
_business_hidden = M["uni_mv"]
_business_rare = [a for a in M["anomaly_classes"] if a["cls"].startswith("B:")]
_business_rare_n = sum(a["n"] for a in _business_rare)
_business_data_error_n = sum(
    a["n"] for a in M["anomaly_classes"] if a["cls"].startswith("A:"))

business = html.Div([
    C.page_header(
        "BUSINESS VIEW · PLAIN-LANGUAGE SUMMARY",
        "What these findings mean for the bank",
        "Customer departures are concentrated where relationships are shallow: one product, "
        "low activity, and certain age and country combinations. The practical response is "
        "targeted retention, a focused investigation in Germany, and better monitoring of "
        "the whole customer relationship.",
        chips=["10,000 customers", f"{BASELINE:.1f}% left", "patterns, not predictions"]),

    callout([
        html.B("Money in the account is not the same as loyalty. "),
        "Customers with a useful, deeper relationship left much less often, while "
        "high-balance customers holding only one product remained exposed. The bank should "
        "move from broad retention campaigns to a few focused, testable actions."
    ], kind="good", title="Bottom line"),

    C.statband([
        stat(f"{BASELINE:.1f}%", "Customers who left", "2,037 of 10,000 customers"),
        stat(f"{_business_top_rule['confidence_pct']:.1f}%", "Highest-priority pattern",
             f"aged 46–60 · inactive · one product · "
             f"{_business_top_rule['customers']:,} of {_business_top_match_n:,} left",
             tone=T.CRITICAL, primary=True),
        stat(f"{_business_geo['Germany']:.1f}%", "Germany departure rate",
             f"France {_business_geo['France']:.1f}% · Spain {_business_geo['Spain']:.1f}%",
             tone=T.WARNING),
        stat(f"{_business_products['2']:.1f}%", "Two-product departure rate",
             f"compared with {_business_products['1']:.1f}% for one product",
             tone=T.SUCCESS),
    ]),

    section("Four business takeaways", meta="the five KDD phases, translated into decisions"),
    html.Div([
        card("01 · Shallow relationships are the clearest warning",
             f"Customers with one product left at {_business_products['1']:.1f}%, compared "
             f"with {_business_products['2']:.1f}% among customers with exactly two. A high "
             "balance by itself did not protect the relationship.",
             insight(["Test whether a relevant second product and a relationship review improve "
                      "retention. Product count is not a goal by itself, so avoid indiscriminate "
                      "cross-selling."], kind="good", title="Business response")),
        card("02 · One group should head the retention queue",
             f"Among customers aged 46–60 who were inactive and held one product, "
             f"{_business_top_rule['confidence_pct']:.1f}% left; "
             f"{_business_top_rule['customers']:,} of {_business_top_match_n:,} matching "
             "customers in this data.",
             insight(["Prioritize a relevant service conversation with this combined profile. "
                      "Do not run a blanket campaign aimed at everyone in the age band."],
                     kind="bad", title="Business response")),
        card("03 · Germany needs its own investigation",
             f"Germany's departure rate was {_business_geo['Germany']:.1f}%, roughly twice "
             f"France ({_business_geo['France']:.1f}%) and Spain "
             f"({_business_geo['Spain']:.1f}%). It remained 52.1% among inactive, "
             "one-product German customers.",
             insight(["Review local product fit, service experience and customer feedback in "
                      "Germany before applying a bank-wide remedy. Geography is a clue to "
                      "investigate, not a fault of the customer."],
                     kind="warn", title="Business response")),
        card("04 · Single-field alerts miss hidden risk",
             f"{_business_hidden['n'][3]:,} customers were flagged only by whole-profile "
             f"checks, and {_business_hidden['churn_pct'][3]:.1f}% had left. Of this group, "
             f"{_business_hidden['mv_only_hidden']:,} "
             f"({_business_hidden['mv_only_hidden_pct']:.0f}%) had no individually extreme "
             "value.",
             insight(["Keep existing account limits, but add a whole-customer review list so "
                      "risky combinations are visible. A person should review the context "
                      "before any outreach."], title="Business response")),
    ], className="grid g2"),

    section("What the bank should do next", meta="priority order, not a blanket campaign"),
    card(None, None, [
        C.finding("1", "Build a targeted retention queue",
                  "Start with customers aged 46–60 who are inactive and hold one product. "
                  "Invite them into a relevant service conversation and learn why activity "
                  "fell; do not target the whole age group."),
        C.finding("2", "Deepen high-value, one-product relationships",
                  f"This segment contains {_business_watchlist['n']:,} customers and had a "
                  f"{_business_watchlist['churn']:.1f}% departure rate. Test a relevant "
                  "second-product offer and relationship-manager contact, then compare "
                  "retention with a similar group that did not receive the pilot."),
        C.finding("3", "Investigate Germany separately",
                  "Review product fit, service quality and customer feedback in Germany. "
                  "Test local remedies before assuming the same response will work in every "
                  "country."),
        C.finding("4", "Add whole-customer monitoring, with a human check",
                  f"Use profile-level warnings alongside existing single-field checks, but "
                  f"separate them from data-quality alerts. Of the unusual cases, "
                  f"{_business_rare_n:,} were classified as rare but legitimate, while only "
                  f"{_business_data_error_n:,} were suspected data errors."),
    ]),

    callout([
        html.B("Measure the pilots, not just the outreach volume. "),
        "Track retention in the targeted groups against a fair comparison group, movement "
        "from inactivity back to activity, suitable one-to-two-product adoption, and whether "
        "the Germany gap narrows. Also watch complaints and opt-outs as safeguards."
    ], title="How to know the response is working"),

    callout([
        "These are historical patterns, not causes or individual predictions. The data is one "
        "snapshot, so it cannot show sudden balance drops before departure. Use the findings "
        "to decide what to investigate and test; never to automatically deny a product, "
        "penalize a customer, or assume that every unusual profile is risky."
    ], kind="warn", title="Use these findings responsibly"),

    html.Div("All figures come from the full 10,000-customer project snapshot. The Knowledge "
             "Report and phase pages retain the technical evidence for readers who need it.",
             className="footnote"),
    C.phase_footer(prev=("report", "Phase 5 · Knowledge Report")),
], className="page")

# ===========================================================================
# Shell: header · pipeline sidebar · canvas
# ===========================================================================

def nav_button(page, title, sub, badge):
    return html.Button([
        html.Span(badge, className="nav-badge"),
        html.Span([html.Span(title, className="nav-title"),
                   html.Span(sub, className="nav-sub")], className="nav-text"),
    ], id=f"nav-{page}", n_clicks=0,
        className="nav-item" + (" nav-item--active" if page == "overview" else ""))


app.layout = html.Div([
    dcc.Store(id="route", data="overview"),
    html.Div(id="scroll-sink", style={"display": "none"}),

    # ── header: identity left · persistent dataset status right ───────────
    html.Header(html.Div([
        html.Div([
            html.Div("K", className="brand-mark"),
            html.Span("CHURN · KDD", className="brand-word"),
            html.Span("GROUP 5", className="env-chip"),
        ], className="brand-zone"),
        html.Div([
            html.Div("Bank Customer Churn: Knowledge Discovery", className="hdr-title"),
            html.Div("Kaggle · 10,000 retail customers · France · Germany · Spain",
                     className="hdr-sub"),
        ], className="hdr-context"),
        html.Div([
            chip("10,000 customers", "gray"),
            chip(f"{BASELINE:.1f}% baseline churn", "red"),
            chip("0 nulls · 0 dupes", "green"),
        ], className="hdr-chips"),
    ], className="hdr-row"), className="app-header"),

    # ── sidebar = KDD stage navigator · canvas ─────────────────────────────
    html.Div([
        html.Nav([
            html.Div("KDD PIPELINE", className="side-label"),
            html.Div([nav_button(*n) for n in NAV], className="side-nav"),
            html.Div([
                html.Div("PROVENANCE", className="side-foot-overline"),
                html.Div("Phases 1–4 mined in Jupyter (random_state 42, all 10,000 "
                         "records); every number here traces to the notebook.",
                         className="side-foot-text"),
            ], className="side-foot"),
        ], className="sidebar"),
        html.Main(html.Div(id="tab-content"), className="canvas"),
    ], className="app-body"),
])

PAGES = {"overview": overview, "ph1": phase1, "ph2": phase2,
         "ph3": phase3, "ph4": phase4, "report": report,
         "business": business}


# ===========================================================================
# Callbacks
# ===========================================================================

# -- routing: sidebar buttons, pipeline cards and prev/next footers all
#    funnel into one store; guard ignores component-mount pseudo-triggers.
@app.callback(
    Output("route", "data"),
    Input({"type": "goto", "page": ALL, "src": ALL}, "n_clicks"),
    [Input(f"nav-{p}", "n_clicks") for p in ORDER],
    prevent_initial_call=True)
def go(*_):
    if not ctx.triggered or not ctx.triggered[0]["value"]:
        return no_update
    t = ctx.triggered_id
    if isinstance(t, dict):
        return t.get("page", no_update)
    if isinstance(t, str) and t.startswith("nav-"):
        return t[4:]
    return no_update


@app.callback(
    [Output("tab-content", "children")] +
    [Output(f"nav-{p}", "className") for p in ORDER],
    Input("route", "data"))
def render(page):
    page = page if page in PAGES else "overview"
    classes = ["nav-item" + (" nav-item--active" if p == page else "") for p in ORDER]
    return [PAGES[page]] + classes


# scroll to top on page change (content swaps in place otherwise)
app.clientside_callback(
    "function(p){window.scrollTo({top:0,left:0,behavior:'auto'}); return '';}",
    Output("scroll-sink", "children"), Input("route", "data"))


@app.callback(Output("ph1-dist-graph", "figure"), Output("ph1-dist-insight", "children"),
              Input("ph1-dist-dd", "value"))
def update_dist(col):
    return F.DIST_FIGS[col], insight(F.DIST_INSIGHTS[col])


@app.callback(Output("ph1-dim-graph", "figure"), Output("ph1-dim-insight", "children"),
              Input("ph1-dim-dd", "value"))
def update_dim(dim):
    return F.DIM_FIGS[dim], insight(F.DIM_INSIGHTS[dim])


@app.callback(Output("ph2-pca-graph", "figure"),
              Input("ph2-algo", "value"), Input("ph2-color", "value"))
def update_pca(algo, colorby):
    return F.PCA_FIGS[(algo, colorby)]


# -- rule table: rows are the selector (click a row → interpretation below)
@app.callback(
    Output("ph3-rule-detail", "children"),
    Output({"type": "rulerow", "letter": ALL}, "className"),
    Input({"type": "rulerow", "letter": ALL}, "n_clicks"),
    prevent_initial_call=True)
def select_rule(_):
    if not ctx.triggered or not ctx.triggered[0]["value"]:
        return no_update, no_update
    letter = ctx.triggered_id["letter"]
    letters = [r["letter"] for r in R["top10"]]
    return (C.rule_detail(letter),
            ["rrow" + (" rrow--sel" if l == letter else "") for l in letters])


@app.callback(Output("ph4-scatter", "figure"), Input("ph4-color", "value"))
def update_outlier(colorby):
    return F.OUTLIER_FIGS[colorby]


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
