"""Reusable layout building blocks, styled per the KDD Design Spec.

Card anatomy: 52px header (title 14/600) · 1px divider · 24px body.
Insight strips: 3px semantic accent bar + tint surface + mono overline.
KPIs: overline label · 28px mono figure · meta line.
"""

from dash import dcc, html

import theme as T
from data import M, R

CFG = T.GRAPH_CONFIG


def graph(fig, gid=None, height=None):
    style = {"height": f"{height}px"} if height else {}
    kw = {"id": gid} if gid else {}
    return dcc.Graph(figure=fig, config=CFG, style=style, **kw)


def card(title, subtitle=None, children=None, insight=None, kind="", extra_class=""):
    """Spec card: header + divider + body. title=None -> plain card, no header."""
    body = []
    if subtitle:
        body.append(html.P(subtitle, className="card-sub"))
    if children:
        body += children if isinstance(children, list) else [children]
    if insight is not None:
        body.append(insight)
    parts = []
    cls = f"card {extra_class}".strip()
    if title:
        parts.append(html.Div(html.Div(title, className="card-title"), className="card-head"))
    else:
        cls += " card--plain"
    parts.append(html.Div(body, className="card-body"))
    return html.Div(parts, className=cls)


def insight(text, kind="", icon=None, title="What this tells us"):
    """Accent-bar insight panel. `icon` kept for API compatibility (unused:
    the spec's icon language is Lucide-outline; emoji would read as noise)."""
    return html.Div([
        html.Div([html.Span(className="insight-dot"),
                  html.Span(title, className="insight-title")], className="insight-head"),
        html.Div(text if isinstance(text, list) else [text], className="insight-body"),
    ], className=f"insight {kind}".strip())


def kpi(value, label, sub="", color=None):
    style = {"color": color} if color else {}
    return html.Div([
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value", style=style),
        html.Div(sub, className="kpi-sub"),
    ], className="kpi")


def chip(text, kind="gray"):
    return html.Span(text, className=f"chip chip--{kind}")


def section(text):
    return html.Div(text, className="section-title")


def control(label, component):
    return html.Div([html.Div(label, className="control-label"), component])


def pills(pid, options, value):
    return dcc.RadioItems(
        id=pid, options=options, value=value, className="radio-pills",
        inputStyle={"display": "none"}, labelStyle={"display": "inline-flex"})


# ---------------------------------------------------------------------------
# persona cards (Phase 2) — identity color from the categorical series;
# risk state as a semantic chip (color = state, per the spec).
# ---------------------------------------------------------------------------

PERSONA_TAGLINE = {
    1: "41.7% of the book keeps six-figure money here but only ONE product. Nothing but inertia "
       "anchors them - and they churn the most.",
    0: "Engaged, wealthy, multi-product - and over half German. Churn is only average, but when "
       "they do leave, it is the costliest money walking out.",
    2: "Zero balance yet TWO products on average - salary-account style usage. Germany is almost "
       "absent (0.7%). The bank's quiet retention anchor.",
}

PERSONA_NOTE = {
    1: "Highest-churn segment - deepen the relationship (second product) before the money leaves.",
    0: "Risk is individual, not segment-wide - monitor high-value accounts, don't blanket-target.",
    2: "Lowest-risk group - a model of what product breadth does for retention.",
}

PERSONA_RISK = {
    1: ("Watchlist · 1.26×", "red"),
    0: ("Mixed risk · 1.07×", "amber"),
    2: ("Loyalist · 0.67×", "green"),
}


def persona_card(k):
    c = M["clusters"][str(k)]
    color = T.CLUSTER_COLORS[k]
    geo = max(c["geo_mix"], key=c["geo_mix"].get)
    risk_label, risk_kind = PERSONA_RISK[k]
    stats = [
        (f"{c['n']:,}", f"customers ({c['share']}%)"),
        (f"{c['churn']}%", "churn rate"),
        (f"£{c['balance_mean']:,.0f}", "avg balance"),
        (f"{c['products_mean']:.2f}", "avg products"),
        (f"{c['active_pct']}%", "active members"),
        (f"{geo} {c['geo_mix'][geo]:.0f}%", "largest country"),
    ]
    return html.Div(html.Div([
        html.Div([html.Span(f"CLUSTER {k}", style={"color": color}),
                  chip(risk_label, risk_kind)], className="persona-overline"),
        html.Div(c["name"], className="persona-name"),
        html.Div(PERSONA_TAGLINE[k], className="persona-tag"),
        html.Div([html.Div([html.Div(v, className="pstat-v"), html.Div(l, className="pstat-l")])
                  for v, l in stats], className="persona-stats"),
        html.Div(PERSONA_NOTE[k], className="persona-note"),
    ], className="card-body"), className="card card--plain persona",
        style={"borderTopColor": color})


# ---------------------------------------------------------------------------
# rule table (Phase 3)
# ---------------------------------------------------------------------------

def _lift_bar(lift, max_lift=3.8):
    pct = min(lift / max_lift * 100, 100)
    color = T.CRITICAL if lift >= 3.2 else "#EC6142" if lift >= 2.9 else T.WARNING
    return html.Div([
        html.Div(className="bar-fill", style={"width": f"{pct}%", "background": color})
    ], className="bar-track")


def rule_table():
    head = html.Tr([html.Th("#"), html.Th("IF the customer is ..."), html.Th("THEN"),
                    html.Th("Confidence"), html.Th("Lift"), html.Th(""),
                    html.Th("Churners"), html.Th("Conviction")])
    rows = []
    for r in R["top10"]:
        rows.append(html.Tr([
            html.Td(html.B(r["letter"], className="num")),
            html.Td(" + ".join(r["if_items"])),
            html.Td(chip("Churned", "red")),
            html.Td(f"{r['confidence_pct']:.1f}%", className="num"),
            html.Td(f"{r['lift']:.2f}×", className="num"),
            html.Td(_lift_bar(r["lift"]), style={"width": "120px"}),
            html.Td(f"{r['customers']:,}", className="num"),
            html.Td(f"{r['conviction']:.2f}", className="num"),
        ]))
    return html.Table([html.Thead(head), html.Tbody(rows)], className="dtable")


# ---------------------------------------------------------------------------
# preprocessing decision log (Phase 1)
# ---------------------------------------------------------------------------

DECISIONS = [
    ("Identifier & PII columns", "Dropped RowNumber, CustomerId, Surname",
     "Index artifacts and personally identifying text carry no behavioral signal and risk leakage."),
    ("Missing values", "None found - no imputation",
     "0 nulls confirmed across all 14 columns; imputing would only add artificial structure."),
    ("Duplicates & consistency", "None found - no removal",
     "0 duplicate rows / IDs; all values inside domain-valid ranges; category labels clean."),
    ("Outliers (Age, CreditScore)", "RETAINED, not deleted",
     "Seniors and low-score customers are real segments. Deleting them would have erased the "
     "project's main finding - they are re-examined in Phase 4 instead."),
    ("Scaling (Path A - clustering)", "StandardScaler on 6 numeric fields",
     "Balance/Salary live in the 100K range vs products in 1-4; unscaled, Euclidean distance "
     "would be a balance ruler."),
    ("Geography & Gender", "EXCLUDED from clustering distance",
     "Nominal categories one-hot-encoded into Euclidean space would force country/gender splits. "
     "They are reintroduced AFTER clustering - so any country skew is a discovery, not an artifact."),
    ("Binning (Path B - rules)", "Domain-anchored bands, not equal-width",
     "Credit-score tiers, life-stage ages (Senior = 46-60), a dedicated Zero-Balance bin for the "
     "36% spike, salary quartiles. Equal-width bins would bury the balance spike."),
    ("Feature selection", "Correlation + entropy, used as a guide",
     "Both lenses agree on Age/IsActiveMember; entropy alone catches NumOfProducts. Weak features "
     "stay in unsupervised phases - churn is a validation lens, not a target."),
]


def decisions_table():
    head = html.Tr([html.Th("Decision"), html.Th("Action"), html.Th("Why (as documented in Phase 1)")])
    rows = [html.Tr([html.Td(html.B(d)), html.Td(a), html.Td(w)]) for d, a, w in DECISIONS]
    return html.Table([html.Thead(head), html.Tbody(rows)], className="dtable")


# ---------------------------------------------------------------------------
# anomaly action table (Phase 4)
# ---------------------------------------------------------------------------

def action_table():
    rows_data = [
        ("A - Suspected data error", "2", "0%",
         "Ages 91-92: legal but implausible. Verify against source systems; exclude from decisions.",
         "gray"),
        ("B - Rare but legitimate", "468", "4.1%",
         "Settled elderly, zero-balance profiles, 4-product holders. Monitor only - churn is BELOW "
         "average. Do not waste retention budget here.", "amber"),
        ("C - Risk signal", "406", "100%*",
         "High-balance pre-churn, disengaged single-product, density outliers. Escalate to "
         "relationship managers; templates to monitor prospectively.", "red"),
    ]
    head = html.Tr([html.Th("Class"), html.Th("Records"), html.Th("Churn"), html.Th("Recommended action")])
    rows = [html.Tr([html.Td(chip(c, k)), html.Td(n, className="num"),
                     html.Td(ch, className="num"), html.Td(a)])
            for c, n, ch, a, k in rows_data]
    return html.Table([html.Thead(head), html.Tbody(rows)], className="dtable")


# ---------------------------------------------------------------------------
# misc blocks
# ---------------------------------------------------------------------------

def qa(question, answer):
    return html.Div([html.Div(question, className="qa-q"),
                     html.Div(answer, className="qa-a")], className="qa")


def finding(num, title, body):
    return html.Div([
        html.Div(num, className="finding-num"),
        html.Div([html.Div(title, className="finding-title"),
                  html.Div(body, className="finding-body")]),
    ], className="finding")


def pipeline_strip():
    steps = [
        ("Phase 1", "Understand & preprocess", "0 nulls, 0 duplicates - two tailored data paths; "
         "feature relevance via correlation + entropy."),
        ("Phase 2", "Segment (clustering)", "K-Means / Ward / DBSCAN on 10,000 customers -> 3 named "
         "personas, method-stable (ARI 0.75)."),
        ("Phase 3", "Mine rules (Apriori)", "3,972 itemsets -> 520 rules -> 13 churn rules at "
         ">2.5x lift. Hypothesis confirmed."),
        ("Phase 4", "Detect anomalies", "6 detectors compared; risk concentrates in unusual "
         "COMBINATIONS (45% churn), not extremes."),
        ("Phase 5", "Communicate", "This dashboard + knowledge report: what we found and what "
         "the bank should do."),
    ]
    return html.Div([html.Div([
        html.Div(p, className="pipe-num"), html.Div(n, className="pipe-name"),
        html.Div(d, className="pipe-desc")], className="pipe-step")
        for p, n, d in steps], className="pipeline")
