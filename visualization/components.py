"""Reusable layout primitives — KDD Design System v2.

Rules the system enforces (see DESIGN_RATIONALE.md):
* One container level: a boxed element never wraps another boxed element.
  Cards hold content; standalone conclusions use `callout`, not card+insight.
* KPI rows are a single `statband` region with hairline dividers — five
  numbers read as one scannable row, not five competing cards.
* Every page opens with a `page_header` (phase kicker · H1 · one-line purpose).
* Navigation targets are real <button>s (keyboard-focusable), wired through
  one pattern-matching callback in app.py.
* Color on numbers is semantic only (risk = red, good = green); never decorative.
"""

from dash import dcc, html

import theme as T
from data import M, R

CFG = T.GRAPH_CONFIG


# ===========================================================================
# atoms
# ===========================================================================

def graph(fig, gid=None, height=None):
    style = {"height": f"{height}px"} if height else {}
    kw = {"id": gid} if gid else {}
    return dcc.Graph(figure=fig, config=CFG, style=style, **kw)


def chip(text, kind="gray"):
    return html.Span(text, className=f"chip chip--{kind}")


def goto(page, children, cls, src):
    """Navigation button. `src` must be unique per call site (id uniqueness)."""
    return html.Button(children, id={"type": "goto", "page": page, "src": src},
                       n_clicks=0, className=cls)


def control(label, component):
    return html.Div([html.Div(label, className="control-label"), component])


def pills(pid, options, value):
    """Segmented control. Same id/value contract as the old dropdowns, so
    existing callbacks keep working — one click instead of open-scan-click."""
    return dcc.RadioItems(
        id=pid, options=options, value=value, className="radio-pills",
        inputStyle={"display": "none"}, labelStyle={"display": "inline-flex"})


# ===========================================================================
# page scaffolding
# ===========================================================================

def page_header(kicker, title, desc=None, chips=None):
    """Every page opens identically: phase kicker · H1 · purpose · fact chips.
    Gives each KDD stage a clear identity and a consistent scanning anchor."""
    left = [html.Div(kicker, className="ph-kicker"),
            html.H1(title, className="ph-title")]
    if desc:
        left.append(html.P(desc, className="ph-desc"))
    row = [html.Div(left, className="ph-main")]
    if chips:
        row.append(html.Div([chip(c, "outline") for c in chips], className="ph-chips"))
    return html.Header(row, className="page-head")


def section(text, meta=None):
    kids = [html.H2(text, className="section-title")]
    if meta:
        kids.append(html.Span(meta, className="section-meta"))
    return html.Div(kids, className="section-row")


def phase_footer(prev=None, nxt=None):
    """Sequential prev/next stage navigation at the bottom of every page."""
    kids = []
    if prev:
        kids.append(goto(prev[0], ["← ", html.Span(prev[1])],
                         "foot-nav foot-nav--prev", f"prev-{prev[0]}"))
    kids.append(html.Div(className="foot-spacer"))
    if nxt:
        kids.append(goto(nxt[0], [html.Span(nxt[1]), " →"],
                         "foot-nav foot-nav--next", f"next-{nxt[0]}"))
    return html.Div(kids, className="page-foot")


# ===========================================================================
# stats
# ===========================================================================

def stat(value, label, sub="", tone=None, primary=False):
    style = {"color": tone} if tone else {}
    return html.Div([
        html.Div(label, className="stat-label"),
        html.Div(value, className="stat-value", style=style),
        html.Div(sub, className="stat-sub"),
    ], className="stat" + (" stat--primary" if primary else ""))


def statband(stats):
    """One region, hairline dividers — replaces N separate KPI cards."""
    return html.Div(stats, className="statband")


def herostat(value, label):
    return html.Div([html.Span(value, className="hs-v"),
                     html.Span(label, className="hs-l")], className="herostat")


# ===========================================================================
# cards, insights, callouts
# ===========================================================================

def card(title, subtitle=None, children=None, insight=None, extra_class=""):
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
        parts.append(html.Div(html.H3(title, className="card-title"),
                              className="card-head"))
    else:
        cls += " card--plain"
    parts.append(html.Div(body, className="card-body"))
    return html.Div(parts, className=cls)


_INSIGHT_TITLES = {"": "What this tells us", "warn": "Read with care",
                   "bad": "Key risk", "good": "Confirmed"}


def insight(text, kind="", icon=None, title=None):
    """Conclusion strip inside a card. `kind` in {'', 'warn', 'bad', 'good'}."""
    title = title or _INSIGHT_TITLES.get(kind, "What this tells us")
    cls = f"insight insight--{kind}" if kind else "insight"
    return html.Div([
        html.Div([html.Span(className="insight-dot"),
                  html.Span(title, className="insight-title")], className="insight-head"),
        html.Div(text if isinstance(text, list) else [text], className="insight-body"),
    ], className=cls)


def callout(text, kind="", title="Verdict"):
    """Standalone conclusion block — same voice as insight strips but WITHOUT
    a wrapping card, so verdicts never render as a box inside a box."""
    cls = f"callout callout--{kind}" if kind else "callout"
    return html.Div([
        html.Div([html.Span(className="insight-dot"),
                  html.Span(title, className="insight-title")], className="insight-head"),
        html.Div(text if isinstance(text, list) else [text], className="insight-body"),
    ], className=cls)


# ===========================================================================
# persona cards (Phase 2)
# ===========================================================================

PERSONA_TAGLINE = {
    1: "41.7% of the book keeps six-figure money here but only ONE product. Nothing but inertia "
       "anchors them — and they churn the most.",
    0: "Engaged, wealthy, multi-product — and over half German. Churn is only average, but when "
       "they do leave, it is the costliest money walking out.",
    2: "Zero balance yet TWO products on average — salary-account style usage. Germany is almost "
       "absent (0.7%). The bank's quiet retention anchor.",
}

PERSONA_NOTE = {
    1: "Highest-churn segment — deepen the relationship (second product) before the money leaves.",
    0: "Risk is individual, not segment-wide — monitor high-value accounts, don't blanket-target.",
    2: "Lowest-risk group — a model of what product breadth does for retention.",
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


# ===========================================================================
# rule table (Phase 3) — rows ARE the selector (direct manipulation)
# ===========================================================================

def _lift_bar(lift, max_lift=3.8):
    pct = min(lift / max_lift * 100, 100)
    color = T.CRITICAL if lift >= 3.2 else "#EC6142" if lift >= 2.9 else T.WARNING
    return html.Div([
        html.Div(className="bar-fill", style={"width": f"{pct}%", "background": color})
    ], className="bar-track")


def rule_table(selected="A"):
    """Consequent column removed: every rule points to churn (stated once in
    the card subtitle), so ten identical 'Churned' chips were pure noise."""
    head = html.Tr([html.Th("#"), html.Th("IF the customer is …"),
                    html.Th("Confidence"), html.Th("Lift"), html.Th(""),
                    html.Th("Churners"), html.Th("Conviction")])
    rows = []
    for r in R["top10"]:
        sel = " rrow--sel" if r["letter"] == selected else ""
        rows.append(html.Tr([
            html.Td(html.B(r["letter"], className="num")),
            html.Td(" + ".join(r["if_items"])),
            html.Td(f"{r['confidence_pct']:.1f}%", className="num"),
            html.Td(f"{r['lift']:.2f}×", className="num"),
            html.Td(_lift_bar(r["lift"]), style={"width": "110px"}),
            html.Td(f"{r['customers']:,}", className="num"),
            html.Td(f"{r['conviction']:.2f}", className="num"),
        ], id={"type": "rulerow", "letter": r["letter"]}, n_clicks=0,
            className=f"rrow{sel}", tabIndex="0",
            title=f"Show interpretation for rule {r['letter']}"))
    return html.Div(html.Table([html.Thead(head), html.Tbody(rows)], className="dtable"),
                    className="table-scroll")


def rule_detail(letter):
    r = next(x for x in R["top10"] if x["letter"] == letter)
    chips = [chip(it, "blue") for it in r["if_items"]]
    stats = (f"confidence {r['confidence_pct']}% · lift {r['lift']:.2f}× · "
             f"support {r['support_pct']}% · {r['customers']:,} churners · "
             f"conviction {r['conviction']:.2f}")
    return insight([
        html.Span(chips + [html.Span(" → "), chip("Churned", "red")],
                  className="rule-chips"),
        html.Div(stats, className="rule-stats"),
        html.Div(r["commentary"]),
    ], title=f"Rule {letter} · business interpretation")


# ===========================================================================
# preprocessing decision log (Phase 1)
# ===========================================================================

DECISIONS = [
    ("Identifier & PII columns", "Dropped RowNumber, CustomerId, Surname",
     "Index artifacts and personally identifying text carry no behavioral signal and risk leakage."),
    ("Missing values", "None found — no imputation",
     "0 nulls confirmed across all 14 columns; imputing would only add artificial structure."),
    ("Duplicates & consistency", "None found — no removal",
     "0 duplicate rows / IDs; all values inside domain-valid ranges; category labels clean."),
    ("Outliers (Age, CreditScore)", "RETAINED, not deleted",
     "Seniors and low-score customers are real segments. Deleting them would have erased the "
     "project's main finding — they are re-examined in Phase 4 instead."),
    ("Scaling (Path A — clustering)", "StandardScaler on 6 numeric fields",
     "Balance/Salary live in the 100K range vs products in 1–4; unscaled, Euclidean distance "
     "would be a balance ruler."),
    ("Geography & Gender", "EXCLUDED from clustering distance",
     "Nominal categories one-hot-encoded into Euclidean space would force country/gender splits. "
     "They are reintroduced AFTER clustering — so any country skew is a discovery, not an artifact."),
    ("Binning (Path B — rules)", "Domain-anchored bands, not equal-width",
     "Credit-score tiers, life-stage ages (Senior = 46–60), a dedicated Zero-Balance bin for the "
     "36% spike, salary quartiles. Equal-width bins would bury the balance spike."),
    ("Feature selection", "Correlation + entropy, used as a guide",
     "Both lenses agree on Age/IsActiveMember; entropy alone catches NumOfProducts. Weak features "
     "stay in unsupervised phases — churn is a validation lens, not a target."),
]


def decisions_table():
    head = html.Tr([html.Th("Decision"), html.Th("Action"),
                    html.Th("Why (as documented in Phase 1)")])
    rows = [html.Tr([html.Td(html.B(d)), html.Td(a), html.Td(w)]) for d, a, w in DECISIONS]
    return html.Div(html.Table([html.Thead(head), html.Tbody(rows)], className="dtable"),
                    className="table-scroll")


# ===========================================================================
# anomaly action table (Phase 4)
# ===========================================================================

def action_table():
    rows_data = [
        ("A — Suspected data error", "2", "0%",
         "Ages 91–92: legal but implausible. Verify against source systems; exclude from decisions.",
         "gray"),
        ("B — Rare but legitimate", "468", "4.1%",
         "Settled elderly, zero-balance profiles, 4-product holders. Monitor only — churn is BELOW "
         "average. Do not waste retention budget here.", "amber"),
        ("C — Risk signal", "406", "100%*",
         "High-balance pre-churn, disengaged single-product, density outliers. Escalate to "
         "relationship managers; templates to monitor prospectively.", "red"),
    ]
    head = html.Tr([html.Th("Class"), html.Th("Records"), html.Th("Churn"),
                    html.Th("Recommended action")])
    rows = [html.Tr([html.Td(chip(c, k)), html.Td(n, className="num"),
                     html.Td(ch, className="num"), html.Td(a)])
            for c, n, ch, a, k in rows_data]
    return html.Div(html.Table([html.Thead(head), html.Tbody(rows)], className="dtable"),
                    className="table-scroll")


# ===========================================================================
# report blocks
# ===========================================================================

def qa(question, answer):
    return html.Div([html.Div(question, className="qa-q"),
                     html.Div(answer, className="qa-a")], className="qa")


def finding(num, title, body):
    return html.Div([
        html.Div(num, className="finding-num"),
        html.Div([html.Div(title, className="finding-title"),
                  html.Div(body, className="finding-body")]),
    ], className="finding")


# ===========================================================================
# pipeline cards (Overview) — clickable, they ARE navigation
# ===========================================================================

PIPELINE = [
    ("ph1", "Phase 1", "Understand & preprocess",
     "0 nulls, 0 duplicates → two tailored data paths; relevance via correlation + entropy."),
    ("ph2", "Phase 2", "Segment (clustering)",
     "K-Means / Ward / DBSCAN → 3 named personas, method-stable (ARI 0.75)."),
    ("ph3", "Phase 3", "Mine rules (Apriori)",
     "4,105 itemsets → 645 rules → 17 churn rules at ≥ 2.5× lift. Hypothesis confirmed."),
    ("ph4", "Phase 4", "Detect anomalies",
     "6 detectors compared; risk lives in unusual COMBINATIONS (45% churn), not extremes."),
    ("report", "Phase 5", "Communicate",
     "This dashboard + the knowledge report: what we found and what the bank should do."),
]


def pipeline_cards():
    return html.Div([
        goto(page, [
            html.Div(num, className="pipe-num"),
            html.Div(name, className="pipe-name"),
            html.Div(desc, className="pipe-desc"),
            html.Div("Open →", className="pipe-cta"),
        ], "pipe-step", f"pipe-{page}")
        for page, num, name, desc in PIPELINE
    ], className="pipeline")
