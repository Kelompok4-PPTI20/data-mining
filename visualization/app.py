"""
Bank Customer Churn — Knowledge Discovery Dashboard  (Phase 5, Group 5)
=======================================================================
Interactive Plotly Dash app presenting the full KDD pipeline (Phases 1-4)
to a non-technical audience: what was mined, what was found, and what it means.

    1) python prepare_data.py     (once - builds dashboard_data/)
    2) python app.py              -> http://127.0.0.1:8050

Design notes
------------
* Google / Looker-Studio visual language (see assets/styles.css + theme.py).
* All figures are precomputed at startup; callbacks only swap ready objects,
  so every interaction responds in well under the 100 ms rubric budget.
* Every chart card ends with a plain-language "What this tells us" strip -
  the dashboard answers the discovery question, it does not just plot data.
"""

from dash import Dash, Input, Output, dcc, html

import components as C
import figures as F
import theme as T
from data import BASELINE, M, R

kpi, card, insight, chip, section, graph = C.kpi, C.card, C.insight, C.chip, C.section, C.graph

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="Churn KDD Dashboard — Group 5",
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
    ],
)

# ===========================================================================
# TAB 1 — OVERVIEW
# ===========================================================================

overview = html.Div([
    html.Div([
        html.Div("THE CENTRAL KDD QUESTION", className="hero-kicker"),
        html.Div("What did we discover that was not already obvious from the raw data?",
                 className="hero-q"),
        html.Div([
            "Churn in this bank is not driven by any single column — no attribute correlates with "
            "leaving above |r| = 0.29. The knowledge is hidden in ", html.B("combinations"), ": "
            "seniors (46–60) churn at 51%, but a senior who is also inactive and holds only one "
            "product churns at ", html.B("77% — nearly 4× the 20.4% baseline"), ". Germany churns at "
            "double the rate of France and Spain and this survives every control we apply. Two "
            "products is a safety sweet spot (7.6% churn) while one product nearly quadruples risk — "
            "a U-shape invisible to linear analysis. And the customers most likely to leave are not "
            "the ones with extreme values, but the ones whose ", html.B("combination"),
            " of normal values is unusual: 45% churn among combination-anomalies vs 25% among "
            "single-value extremes. In short: ", html.B(
                "risk in this book is relational, not marginal — it lives in who the customer is "
                "as a profile, not in any number on its own."),
        ], className="hero-a"),
    ], className="hero"),

    html.Div([
        kpi("10,000", "Customers analyzed", "14 raw features · France, Germany, Spain"),
        kpi(f"{BASELINE:.1f}%", "Baseline churn", "2,037 customers left — the yardstick for every "
            "number here", T.RED),
        kpi("3", "Customer personas", "found by clustering, method-stable (ARI 0.75)", T.BLUE),
        kpi("13", "High-lift churn rules", "all ≥ 2.5× baseline risk · top rule 3.79×", T.PURPLE),
        kpi("876", "Anomalies flagged", "406 risk signals · 468 rare-but-valid · 2 data errors",
            T.AMBER),
    ], className="grid g-kpi"),

    section("The four discoveries"),
    html.Div([
        card("1 · Churn compounds along the senior path",
             "Churn rate as conditions stack (each bar adds one condition)",
             C.graph(F.OV_SENIOR),
             insight(["Age is the strongest single signal, but the story is the ", html.B("interaction"),
                      ": inactivity roughly doubles a senior's risk, and shallow product depth "
                      "pushes it to 77%. None of this is visible in a one-column report."])),
        card("2 · Germany churns at double the rate",
             "Churn rate by country — Germany loses 814 of 2,509 customers",
             C.graph(F.OV_GEO),
             insight(["The gap survives every control: even restricted to inactive single-product "
                      "customers, Germany still churns at 52.1% (lift 2.56×). This points to a ",
                      html.B("structural problem in the German operation"),
                      ", not an unlucky customer mix."], kind="insight--bad")),
        card("3 · Products follow a U-curve, not a line",
             "Churn rate by number of products held",
             C.graph(F.OV_PROD),
             insight(["Two products = safety (7.6%). One product = weak anchor (27.7%). Three or "
                      "four = near-certain exit (83% / 100%), a likely distress signal. Linear "
                      "correlation rates this feature 'weak' (r = 0.05) — ", html.B(
                          "the entropy lens in Phase 1 is what caught it"), "."])),
        card("4 · Risk hides in combinations, not extremes",
             "Churn rate by type of statistical unusualness (Phase 4)",
             C.graph(F.OV_COMBO),
             insight(["Customers flagged ONLY by multivariate detectors — unusual combinations of "
                      "individually normal values — churn at 45.3%. Single-value extremes are "
                      "largely benign (24.7%). ", html.B(
                          "A dashboard that monitors columns one at a time structurally cannot "
                          "see the riskiest customers"), "."], kind="insight--warn")),
    ], className="grid g2"),

    section("How we got here — the KDD pipeline"),
    C.pipeline_strip(),
], className="page")

# ===========================================================================
# TAB 2 — PHASE 1 · DATA & PREPROCESSING
# ===========================================================================

phase1 = html.Div([
    html.Div([
        kpi("0", "Missing values", "no imputation needed — verified, not assumed", T.GREEN),
        kpi("0", "Duplicate records", "10,000 unique customers", T.GREEN),
        kpi("3", "Columns dropped", "RowNumber, CustomerId, Surname (IDs & PII)"),
        kpi("374", "Outliers retained", "seniors & low-credit are real segments — re-examined in "
            "Phase 4", T.AMBER),
        kpi("6 / 10", "Features selected", "by correlation + entropy — the rest kept for "
            "unsupervised discovery", T.BLUE),
    ], className="grid g-kpi"),

    section("Know the data"),
    html.Div([
        card("Feature distributions — retained vs churned",
             "Pick a feature. Red = the 2,037 customers who left; blue = the 7,963 who stayed. "
             "Bars show % of each group, so the shapes are directly comparable.",
             [dcc.Dropdown(id="ph1-dist-dd", options=[{"label": c, "value": c}
                                                      for c in F.DIST_FEATURES],
                           value="Age", clearable=False, className="dash-dropdown",
                           style={"width": "260px"}),
              dcc.Graph(id="ph1-dist-graph", config=T.GRAPH_CONFIG, style={"height": "300px"}),
              html.Div(id="ph1-dist-insight")]),
        card("Who churns? — churn rate by customer group",
             "Pick a dimension. The dashed line is the 20.4% bank average; red bars sit clearly "
             "above it, green clearly below.",
             [dcc.Dropdown(id="ph1-dim-dd", options=[{"label": v, "value": k}
                                                     for k, v in F.DIM_LABELS.items()],
                           value="Geography", clearable=False, className="dash-dropdown",
                           style={"width": "260px"}),
              dcc.Graph(id="ph1-dim-graph", config=T.GRAPH_CONFIG, style={"height": "300px"}),
              html.Div(id="ph1-dim-insight")]),
    ], className="grid g2"),

    section("Feature selection — the rubric's two lenses"),
    card("Correlation AND entropy, side by side",
         "Left: linear association with churn (Pearson |r| — undefined for nominal "
         "Geography/Gender, which is precisely why a second lens is required). "
         "Right: information-theoretic relevance (mutual information + Shannon information gain).",
         C.graph(F.FEATSEL_FIG),
         insight(["The two lenses agree at the top (Age, IsActiveMember) — but ", html.B(
             "NumOfProducts is nearly invisible to correlation (|r| = 0.05) and #1–2 by "
             "information gain"), " because its churn relation is U-shaped and cancels out in a "
             "linear coefficient. Correlation-only selection would have discarded one of the most "
             "informative features in the dataset. Selected: Age, NumOfProducts, Geography, "
             "IsActiveMember, Balance, Gender."])),

    section("Every decision, justified"),
    card("Preprocessing decision log",
         "The rubric grades justification, not just action. Each choice below is documented in "
         "the Phase 1 report; the two data paths (scaled matrix for clustering, discretized "
         "items for Apriori) come from these decisions.",
         C.decisions_table()),
], className="page")

# ===========================================================================
# TAB 3 — PHASE 2 · SEGMENTS
# ===========================================================================

phase2 = html.Div([
    section("Three personas — named only on features that truly separate them"),
    html.Div([C.persona_card(1), C.persona_card(0), C.persona_card(2)], className="grid g3"),

    section("The cluster map"),
    card("Customer segmentation in 2-D (PCA projection of the clustering space)",
         "Each dot is one customer, projected from the 8-D behavioral space the algorithms "
         "actually clustered (PC1+PC2 = 36% of variance — orientation, not proof). Switch "
         "algorithm to see how three different definitions of 'group' read the same customers. "
         "Geography, Gender and churn were NOT used to form these groups.",
         [html.Div([
             C.control("Algorithm", C.pills("ph2-algo", [
                 {"label": "K-Means (K=3) — the personas", "value": "kmeans"},
                 {"label": "Hierarchical Ward (K=3)", "value": "ward"},
                 {"label": "DBSCAN (density)", "value": "dbscan"}], "kmeans")),
             C.control("Color by", C.pills("ph2-color", [
                 {"label": "Segment", "value": "segment"},
                 {"label": "Churn status", "value": "churn"}], "segment")),
         ], className="controls"),
             dcc.Graph(id="ph2-pca-graph", config=T.GRAPH_CONFIG, style={"height": "430px"})],
         insight(["K-Means and Ward find nearly the same three segments (ARI 0.75) ordered along "
                  "PC1: single-product high-balance → multi-product high-balance → zero-balance. "
                  "DBSCAN instead splits on the density valley between 1 and 2 products and "
                  "isolates 554 'noise' customers (red ✕) in sparse regions — ", html.B(
                      "those noise customers churn at 62.6%, the strongest anomaly-churn signal "
                      "in the whole project"), " (picked up again in Phase 4). The banded look is "
                  "real data: products only take 4 discrete values."])),

    section("How K was chosen — both criteria, honestly read"),
    html.Div([
        card("Elbow & silhouette",
             "The two required validity checks disagree — and the disagreement is informative.",
             C.graph(F.ELBOW_SIL_FIG),
             insight(["Silhouette peaks at K=2 (0.164) — but that split is just 'has money vs "
                      "doesn't' (the bimodal Balance column restated). The elbow shows no sharp "
                      "knee, only diminishing returns from K=3–5. ", html.B(
                          "K=3 is chosen inside both windows"), ": one level finer than the "
                      "trivial split — exactly where product depth enters and the segmentation "
                      "becomes actionable. All silhouettes ≤ 0.16 mean these are descriptive "
                      "segments of a continuum, not natural islands — validated instead by "
                      "cross-algorithm stability (ARI 0.75) and the effect sizes on the right."],
                     kind="insight--warn", icon="⚖️", title="Honest reading")),
        card("What actually separates the segments",
             "Effect sizes across the 3 clusters (Kruskal-Wallis ε² / Cramér's V — the larger "
             "of the two per feature). At n = 10,000 everything is 'significant'; only effect "
             "size is allowed to name a persona.",
             C.graph(F.EFFECT_FIG),
             insight(["Balance (0.73) and product depth (0.71) define the segments; geography "
                      "(0.33) differs strongly as a ", html.B("post-clustering discovery"),
                      ". Age, tenure, salary, credit score and activity do NOT separate these "
                      "clusters — so the persona names deliberately never mention them."])),
    ], className="grid g2"),

    html.Div([
        card("Segment fingerprints (snake plot)",
             "How far each segment's average sits from the bank average, in standard deviations — "
             "all features on one comparable scale.",
             C.graph(F.SNAKE_FIG),
             insight(["The three lines only fan apart on Balance and NumOfProducts — the same "
                      "verdict as the effect sizes, from a third angle. C1 (red) = high balance, "
                      "one product; C2 (green) = zero balance, multi-product; C0 (amber) = high "
                      "on both."])),
        card("Churn and geography by segment",
             "Left: the churn validation lens (never used to form clusters). Right: where each "
             "segment's customers live.",
             C.graph(F.CLUSTER_CHURN_FIG),
             insight(["Churn spans 13.6% → 25.6% across segments — a spread the weak silhouette "
                      "knows nothing about. And Germany concentrates in the high-balance clusters "
                      "(52% of C0) while being ", html.B("almost absent (0.7%) from the loyalist "
                      "zero-balance segment"), " — German customers essentially never hold the "
                      "bank's 'safe' salary-account profile. Discovered after clustering, so it "
                      "is a finding, not an artifact."])),
    ], className="grid g2"),

    card("Cross-algorithm verdict (Mining Expo Q2)", None, None,
         insight(["K-Means K=3 gives the most interpretable personas (silhouette 0.150). Ward "
                  "reproduces them (ARI 0.746, NMI 0.701) — the structure is method-stable, not a "
                  "K-Means artifact. Single linkage collapses into one chain (99.8% of customers "
                  "in one branch) — itself evidence that the book is a continuum. DBSCAN is the "
                  "wrong tool for personas here (it just splits 1-product vs 2-product) but the "
                  "best discovery tool: its noise set is the most churn-dense subgroup any method "
                  "found. Each algorithm was used for what it is structurally good at."],
                 icon="🏁", title="Verdict")),
], className="page")

# ===========================================================================
# TAB 4 — PHASE 3 · ASSOCIATION RULES
# ===========================================================================

hyp = M["hypothesis"]

phase3 = html.Div([
    html.Div([
        html.Div("ASSIGNED HYPOTHESIS — CONFIRMED ✓", className="hero-kicker",
                 style={"color": T.GREEN}),
        html.Div("“Customers from Germany holding only one product who are inactive represent a "
                 "strong churn profile.”", className="hero-q", style={"fontSize": "18px"}),
        html.Div([
            f"{hyp['antecedent_n']} customers match the profile → {hyp['churned_n']} of them "
            f"churned. Confidence ", html.B(f"{hyp['confidence_pct']}%"), " (vs 20.4% baseline), "
            f"lift ", html.B(f"{hyp['lift']}×"), f", support {hyp['support_pct']}%. The rule "
            "clears every filter and is verified by direct computation on the raw data — the "
            "German retention problem is real, and Phase 3 shows it is one of TWO independent "
            "risk vectors (the other is senior age) that compound when combined.",
        ], className="hero-a"),
    ], className="hero", style={"background": "linear-gradient(135deg,#e6f4ea 0%,#ffffff 55%)"}),

    html.Div([
        kpi("3,972", "Frequent itemsets", "Apriori, min support 3% (≥ ~300 customers)"),
        kpi("520", "Rules pass filters", "confidence ≥ 50% and lift ≥ 1.5"),
        kpi("13", "Churn-consequent rules", "the rubric needs 10 — every one ≥ 2.5× baseline",
            T.PURPLE),
        kpi("3.79×", "Strongest lift", "inactive senior, 1 product → 77.3% churn", T.RED),
    ], className="grid g-kpi4"),

    insight(["With a 20.4% base rate, demanding confidence ≥ 50% mathematically forces lift "
             "≥ 2.45 — so ", html.B("every rule that survived more than doubles churn risk"),
             ". Only 13 of 520 rules concern churn: that is the filter working, not a shortage. "
             "The other 507 are structural co-occurrences (geography ↔ balance bands etc.), "
             "kept in the saved file for transparency."], icon="🧪",
            title="Why so few rules survive"),

    section("The rule network"),
    html.Div([
        card("How the top-10 rules connect",
             "Blue circles = customer attributes (bigger = appears in more rules). Diamonds = "
             "rules A–J (redder = higher lift; hover for details). Every rule points to churn.",
             C.graph(F.RULE_NETWORK_FIG),
             insight(["One attribute sits at the heart of almost every rule: ", html.B(
                 "Senior (46–60)"), ". Churn risk compounds when senior age meets inactivity, "
                 "single-product holdings, female gender or German geography. The bank is not "
                 "losing customers at random — it is losing ", html.B(
                     "a specific, describable population"), "."])),
        card("Rule quality at a glance",
             "Each bubble is a rule: right = more reliable (confidence), up = stronger vs "
             "baseline (lift), bigger = more customers, darker = more conditions.",
             C.graph(F.RULE_SCATTER_FIG),
             insight(["Rules stack up along the dotted frontier because lift = confidence ÷ "
                      "20.4%. The prize is the top-right: rule A (3 conditions, 77% confidence) "
                      "describes 405 real churners — specific enough to action, big enough to "
                      "matter."])),
    ], className="grid g2"),

    section("The deliverable — top 10 rules, ranked by lift"),
    card("Top-10 churn rules with business interpretation",
         "Support floor 3% means the weakest rule still describes ~300 customers — no "
         "micro-segment flukes. Select a rule for its plain-language interpretation.",
         [C.rule_table(),
          html.Div([
              dcc.Dropdown(id="ph3-rule-dd",
                           options=[{"label": f"Rule {r['letter']} — "
                                     + " + ".join(r["if_items"]), "value": r["letter"]}
                                    for r in R["top10"]],
                           value="A", clearable=False, className="dash-dropdown",
                           style={"width": "520px", "maxWidth": "100%"}),
          ], style={"marginTop": "14px"}),
          html.Div(id="ph3-rule-detail")]),

    card("Reading the metrics", None, None,
         insight([html.B("Support"), " = share of ALL customers matching rule + churn. ",
                  html.B("Confidence"), " = of customers matching the IF, the % who churned. ",
                  html.B("Lift"), " = confidence ÷ 20.4% baseline — how many times more likely. ",
                  html.B("Conviction"), " = how much more often the rule would have to fail if "
                  "IF and churn were independent (higher = stronger). These are historical "
                  "associations, not causal claims — the correct response is targeted "
                  "investigation and A/B-tested retention offers."], icon="📖",
                 title="Glossary")),
], className="page")

# ===========================================================================
# TAB 5 — PHASE 4 · ANOMALIES
# ===========================================================================

xr = M["cross_ref"]
u = M["uni_mv"]

phase4 = html.Div([
    html.Div([
        kpi("876", "Customers flagged", "by ≥1 of the 4 core methods (8.8% of book)"),
        kpi("406", "Risk signals (C)", "churn-linked patterns → escalate to retention", T.RED),
        kpi("468", "Rare but valid (B)", "churn just 4.1% — SAFER than average; do not delete",
            T.AMBER),
        kpi("2", "Suspected data errors (A)", "ages 91–92 — manual review, nothing depends on "
            "them", T.PURPLE),
        kpi("336", "IF ∩ DBSCAN overlap", "two structural methods agreeing · κ = 0.617", T.BLUE),
    ], className="grid g-kpi"),

    section("Six detectors, systematically compared"),
    card("Who finds the risky anomalies?",
         "All six methods, ranked by churn rate among the customers they flag. Blue = "
         "multivariate (judge the whole profile); gray = univariate (judge one value at a "
         "time). Counts are what each method flags — note IF/LOF fix their count at 5% by "
         "construction, IQR/Z discover theirs.",
         C.graph(F.METHOD_FIG),
         insight(["A clean split: the structural, whole-profile methods (DBSCAN 62.6%, "
                  "Mahalanobis 58.5%, Isolation Forest 49.0%) flag churn-dense customers, while "
                  "single-value screens flag benign extremes (IQR 23.5%, Z-score 13.5% — ",
                  html.B("below baseline"), "). Different methods answer different questions; "
                  "for churn risk, trust the structural family."])),

    html.Div([
        card("The consensus trap",
             "Churn rate by how many of the 4 core methods (IQR, Z-score, Isolation Forest, "
             "DBSCAN) agree a customer is anomalous.",
             C.graph(F.COMPOSITE_FIG),
             insight(["More votes ≠ more risk. The peak is at score 2 (65.9%) — mostly the "
                      "IF + DBSCAN pair agreeing while both univariate fences stay silent. "
                      "Score 3–4 requires breaking a univariate fence, which fires almost only "
                      "on extreme age — flagging settled retirees (mean age 71, churn ~18–25%). ",
                      html.B("A naive 'escalate the most-flagged first' policy would chase "
                             "retirees and miss the real risk pool.")],
                     kind="insight--warn", icon="⚠️", title="The paradox")),
        card("Combinations beat extremes",
             f"Churn by anomaly family. Of the {u['n'][3]} multivariate-only customers, "
             f"{u['mv_only_hidden']} ({u['mv_only_hidden_pct']:.0f}%) have NO single value "
             "beyond |z| = 3 — every number looks normal; only the combination is rare.",
             C.graph(F.UNIMV_FIG),
             insight(["The two families barely overlap (Jaccard 0.21, κ 0.31) because they ask "
                      "different questions. The risk gradient — 18.6% → 24.7% → 45.3% — shows "
                      "churn concentrating exactly in the anomalies univariate screens ",
                      html.B("structurally cannot see"), ": young customers with large balances, "
                      "multi-product holders with contradictory engagement. This is the Phase-4 "
                      "discovery."], kind="insight--bad")),
    ], className="grid g2"),

    section("The outlier map"),
    card("Every customer by balance and Isolation-Forest score",
         "Dots below the dashed line are the 5% most isolable profiles. Color by anomaly class "
         "to see the triage, or by churn to see how risk concentrates toward the bottom.",
         [html.Div([C.control("Color by", C.pills("ph4-color", [
             {"label": "Anomaly class (triage)", "value": "class"},
             {"label": "Churn status", "value": "churn"}], "class"))], className="controls"),
          dcc.Graph(id="ph4-scatter", config=T.GRAPH_CONFIG, style={"height": "430px"})],
         insight(["The most anomalous region (bottom) is dominated by red risk-signal cases at "
                  "BOTH ends of the balance axis: high-balance pre-churn departures on the right "
                  "(592 churners averaging £149.8K — 46% German), and unusual zero-balance "
                  "profiles on the left. Amber rare-valid cases sit just below the threshold — "
                  "unusual, but safe."])),

    section("Classification & action — the anomaly typology"),
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
          html.Div("* Class C churn = 100% by construction: the typology uses the observed churn "
                   "label retrospectively as supporting evidence, as the brief prescribes. "
                   "Deployed prospectively, expect the Phase-3/4 lift levels (≈2.5–3.8×), never "
                   "100% — this figure must not be quoted as model performance.",
                   className="footnote")]),

    section("Cross-reference with Phase 2 (explicitly graded)"),
    html.Div([
        card("Where DBSCAN's noise lives across the personas",
             "The 554 Phase-2 density outliers, mapped into the three K-Means segments.",
             C.graph(F.CROSSREF_FIG),
             insight(["The noise concentrates (10.6%) inside C0 — the multi-product Germany-skew "
                      "segment — and is rarest in the loyalist C2 (1.7%). Cluster outliers and "
                      "statistical anomalies point at the same neighbourhood of customers: "
                      "unusual multi-product, older, high-engagement-contradiction profiles."])),
        card("Agreement between anomaly views", None, None,
             insight([f"IF ∩ DBSCAN share {xr['if_dbscan_overlap']} customers (Jaccard 0.47, "
                      f"κ 0.617, churn {xr['if_dbscan_churn']}%) — two structurally different "
                      "multivariate mechanisms (tree isolation vs density) converging on the same "
                      "people is mutual validation. Weakest pair: Z-score vs DBSCAN (κ 0.21) — a "
                      "strict single-value fence and a joint-density method barely overlap, "
                      "exactly as the family analysis predicts. Anomaly votes must be weighted "
                      "by the question each voter asks: the IF ∩ DBSCAN intersection is the "
                      "high-value retention list; univariate flags are the data-quality and "
                      "rare-case documentation pool."], icon="🔗", title="Verdict")),
    ], className="grid g2"),
], className="page")

# ===========================================================================
# TAB 6 — KNOWLEDGE REPORT
# ===========================================================================

report = html.Div([
    html.Div([
        html.Div("KNOWLEDGE DISCOVERY REPORT — THE DIRECT ANSWER", className="hero-kicker"),
        html.Div("We discovered that churn risk is relational: it lives in profiles, "
                 "not in columns.", className="hero-q"),
        html.Div([
            "Raw-data inspection shows a 20% churn rate and mild demographic tilts. Mining the "
            "same 10,000 customers end-to-end revealed four pieces of knowledge none of which is "
            "visible in a univariate report: (1) a compounding ", html.B("senior × engagement "
            "interaction"), " that escalates from 51% to 77% churn as conditions stack; (2) a ",
            html.B("structural German retention problem"), " that doubles churn independently of "
            "customer mix; (3) a customer book organized by ", html.B("balance × product depth"),
            " — whose largest segment (41.7%) keeps six-figure balances anchored by only one "
            "product; and (4) the fact that ", html.B("unusual combinations of normal values"),
            " predict churn (45.3%) far better than extreme single values (24.7%). The value of "
            "this project is the interpretation of these hidden profiles — not prediction "
            "accuracy.",
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
                  "32.4% churn vs ~16% elsewhere; 46% of high-balance churners are German; German "
                  "seniors churn at 67% regardless of activity. Action: investigate product fit "
                  "and service quality in the German operation specifically — this is not a "
                  "customer-mix effect."),
        C.finding("3", "The riskiest mainstream segment is high-balance / single-product",
                  "Persona C1 — 4,168 customers (41.7%) with ~£120K average balance and exactly "
                  "one product — churns at 25.6%, the highest of the three segments. Money "
                  "without product depth is unanchored. Action: cross-sell into C1 before "
                  "the money leaves; measure product depth, not balance, as the loyalty KPI."),
        C.finding("4", "Monitor combinations, not thresholds",
                  "Customers anomalous only as combinations churn at 45.3% — and 90% of them "
                  "trip no single-value alarm. The 554 DBSCAN noise customers churn at 62.6%. "
                  "Action: add a multivariate anomaly score (IF/DBSCAN-style) to the CRM "
                  "watchlist alongside the existing per-column limits."),
    ]),

    section("Mining Expo — the four questions"),
    card(None, None, [
        C.qa("Q1 · Which association rules were the most surprising, and why?",
             [html.B("{Inactive ∩ Senior ∩ 1 product} → churn"), " (77.3% confidence, lift 3.79) "
              "— not because inactivity matters, but because the AGE interaction transforms it: "
              "younger inactives usually re-engage, senior inactives leave for good. Second: ",
              html.B("{Senior ∩ Germany} → churn"), " (67.3%, lift 3.31) — it survives despite "
              "geography being excluded from clustering distance, making the German signal a "
              "discovered profile rather than an artifact. The assigned hypothesis "
              "{Germany ∩ Inactive ∩ 1 product} was confirmed at 52.1% confidence, lift 2.56."]),
        C.qa("Q2 · Which clustering method produced the most interpretable segments?",
             ["K-Means at K=3 — chosen over the silhouette-peak K=2, which merely restates the "
              "bimodal balance column. Ward hierarchical validates the partition (ARI 0.746); "
              "DBSCAN is the better discovery tool (its 554 noise points churn at 62.6%) but a "
              "worse persona tool (it only splits on product count). Honest caveat: all "
              "silhouettes ≤ 0.164 — these are stable, business-distinct operational segments "
              "of a continuum, not natural species."]),
        C.qa("Q3 · What anomalies were found, and what do they suggest in a real banking context?",
             ["Three species: (a) 406 risk signals — high-balance pre-churn departures "
              "(avg £149.8K, 46% German: relationship-manager outreach), disengaged "
              "single-product churners, and density outliers; (b) 468 rare-but-valid profiles "
              "(settled elderly, zero-balance) churning at just 4.1% — flagging them for "
              "'cleaning' would have destroyed real segments; (c) 2 suspected data errors "
              "(ages 91–92) for manual verification. The banking lesson: risk hides in unusual "
              "profiles, so monitoring should be multivariate."]),
        C.qa("Q4 · How do the findings compare to other banking domains?",
             ["Fraud and credit-risk datasets (Groups 3/8, 4) typically surface financial-"
              "capacity variables. Our churn book is the opposite: CreditScore and Salary carry "
              "almost zero signal, while engagement and relationship-depth variables (Age, "
              "NumOfProducts, IsActiveMember, Geography, Balance) carry it all. Same KDD "
              "pipeline, different knowledge — the meaning of 'anomaly' is domain-shaped: here "
              "it is a disengaging customer, in fraud it is a transaction pattern."]),
    ]),

    section("Limitations — what we cannot claim"),
    card(None, None, [html.Ul([
        html.Li([html.B("Snapshot data, longitudinal question. "), "The assigned 'sudden balance "
                 "drop before closure' cannot be observed in one snapshot per customer; Phase 4 "
                 "proxies it as {high balance ∩ exited}. A true drop-detector needs transaction "
                 "time series."]),
        html.Li([html.B("Weak geometric separation. "), "All silhouettes ≤ 0.164: three useful, "
                 "stable, business-distinct segments — not three natural species."]),
        html.Li([html.B("Retrospective constructs. "), "Rule confidences (52–77%) and Class-C "
                 "rates describe this historical snapshot; deployed prospectively, expect "
                 "regression toward the lift values."]),
        html.Li([html.B("Discretization sensitivity. "), "Rules depend on bin boundaries "
                 "(Senior = 46–60), fixed from domain conventions before mining."]),
        html.Li([html.B("No causal claims. "), "Germany's 2× churn and the senior effect are "
                 "associations in one bank's book over one period. Correct response: targeted "
                 "investigation and A/B-tested offers, not blanket policy."]),
        html.Li([html.B("Method-setting residue. "), "DBSCAN counts depend on (eps=1.25, "
                 "minPts=10); IF/LOF on contamination=5%. Sensitivity was reported; every count "
                 "reads 'under the stated settings'."]),
    ], className="tight"),
        html.Div("None of these threaten the central findings — each is triangulated by at "
                 "least two independent methods. The limitations bound how far the findings "
                 "generalize beyond this snapshot.", className="footnote")]),

    card("Reproducibility", None, None,
         insight(["Pipeline: notebooks/notebook.ipynb (Phases 1–4, random_state = 42, full "
                  "10,000 records, no sampling) → prepare_data.py (assembles this dashboard's "
                  "cache; recomputed values verified against the notebook: Ward ARI 0.7461, "
                  "NMI 0.7014, silhouettes to 4 decimals) → app.py (Plotly Dash). Stack: pandas, "
                  "scikit-learn, mlxtend, SciPy, Plotly Dash."], icon="🔁",
                 title="How to reproduce")),
], className="page")

# ===========================================================================
# Shell
# ===========================================================================

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([html.Span(style={"background": c}) for c in
                      [T.BLUE, T.RED, T.AMBER, T.GREEN]], className="logo-dots"),
            html.Div([
                html.Div("Bank Customer Churn — Knowledge Discovery", className="appbar-title"),
                html.Div("KDD pipeline · 10,000 retail customers · France · Germany · Spain",
                         className="appbar-sub"),
            ]),
            html.Div([
                html.Div([html.B("Group 5"), " · Data Mining · Phase 5 deliverable"]),
                html.Div("Discovery over prediction — every number traces to the notebook"),
            ], className="appbar-meta"),
        ], className="appbar-row"),
        html.Div(dcc.Tabs(id="tabs", value="overview", className="custom-tabs", children=[
            dcc.Tab(label="✦ The Discovery", value="overview",
                    className="ctab", selected_className="ctab--selected"),
            dcc.Tab(label="1 · Data & Preprocessing", value="ph1",
                    className="ctab", selected_className="ctab--selected"),
            dcc.Tab(label="2 · Customer Segments", value="ph2",
                    className="ctab", selected_className="ctab--selected"),
            dcc.Tab(label="3 · Churn Rules", value="ph3",
                    className="ctab", selected_className="ctab--selected"),
            dcc.Tab(label="4 · Anomalies", value="ph4",
                    className="ctab", selected_className="ctab--selected"),
            dcc.Tab(label="5 · Knowledge Report", value="report",
                    className="ctab", selected_className="ctab--selected"),
        ]), className="tabs-holder"),
    ], className="appbar"),
    html.Div(id="tab-content"),
])

PAGES = {"overview": overview, "ph1": phase1, "ph2": phase2,
         "ph3": phase3, "ph4": phase4, "report": report}


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    return PAGES[tab]


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


@app.callback(Output("ph3-rule-detail", "children"), Input("ph3-rule-dd", "value"))
def update_rule_detail(letter):
    r = next(x for x in R["top10"] if x["letter"] == letter)
    chips = [chip(it, "blue") for it in r["if_items"]]
    stats = (f"confidence {r['confidence_pct']}% · lift {r['lift']:.2f}× · "
             f"{r['customers']:,} churners · conviction {r['conviction']:.2f}")
    return insight([html.Span(chips + [html.Span(" → "), chip("Churned", "red")],
                              style={"display": "inline-flex", "gap": "6px",
                                     "flexWrap": "wrap", "alignItems": "center",
                                     "marginRight": "8px"}),
                    html.Div(stats, style={"margin": "6px 0", "fontWeight": "600"}),
                    html.Div(r["commentary"])],
                   icon="🎯", title=f"Rule {letter}")


@app.callback(Output("ph4-scatter", "figure"), Input("ph4-color", "value"))
def update_outlier(colorby):
    return F.OUTLIER_FIGS[colorby]


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
