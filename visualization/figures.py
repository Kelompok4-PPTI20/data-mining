"""
All Plotly figures for the dashboard.

Every figure is built ONCE at import time from the precomputed cache, so
callbacks only swap ready-made objects and interactions stay <100 ms.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import theme as T
from data import BASELINE, M, R, REC

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _base(height=320, **layout):
    fig = go.Figure()
    fig.update_layout(height=height, **layout)
    return fig


def _baseline_hline(fig, y=BASELINE, text=None, row=None, col=None, pos="top right"):
    fig.add_hline(y=y, line_dash="dash", line_color=T.INK, line_width=1.2, layer="below",
                  annotation_text=text or f"bank average {y:.1f}%",
                  annotation_font=dict(size=10.5, color=T.INK2),
                  annotation_position=pos, row=row, col=col)


def risk_color(v, hi=1.15, lo=0.85):
    if v >= BASELINE * hi:
        return T.RED
    if v <= BASELINE * lo:
        return T.GREEN
    return T.BLUE


# ---------------------------------------------------------------------------
# OVERVIEW mini charts
# ---------------------------------------------------------------------------

def ov_bar(labels, values, colors, height=225, suffix="%", texts=None):
    fig = _base(height)
    fig.add_bar(x=values, y=labels, orientation="h",
                marker_color=colors, marker_line_width=0,
                text=texts or [f"{v:.1f}{suffix}" for v in values],
                textposition="outside", textfont=dict(size=11.5),
                hoverinfo="skip")
    fig.update_layout(margin=dict(l=10, r=30, t=6, b=6), bargap=0.42,
                      xaxis=dict(range=[0, max(values) * 1.24], showgrid=True,
                                 ticksuffix=suffix, visible=False),
                      yaxis=dict(autorange="reversed"))
    return fig


OV_SENIOR = ov_bar(
    ["All customers", "Senior (46-60)", "Senior + inactive", "Senior + inactive + 1 product"],
    [BASELINE, M["churn_by"]["Age_Band"]["churn_pct"][2], 68.4, 77.3],
    ["#B7BEC9", T.AMBER, "#EC6142", T.RED])

OV_GEO = ov_bar(
    [f"{l}  ({t:,} customers)" for l, t in zip(M["churn_by"]["Geography"]["labels"],
                                               M["churn_by"]["Geography"]["total"])],
    M["churn_by"]["Geography"]["churn_pct"],
    [T.BLUE, T.RED, T.BLUE])

OV_PROD = ov_bar(
    [f"{l} product{'s' if l != '1' else ''}  ({t:,})" for l, t in
     zip(M["churn_by"]["NumOfProducts"]["labels"], M["churn_by"]["NumOfProducts"]["total"])],
    M["churn_by"]["NumOfProducts"]["churn_pct"],
    [T.AMBER, T.GREEN, T.RED, "#A61B37"])

OV_COMBO = ov_bar(
    ["Not an outlier", "Extreme on ONE value", "Extreme + unusual combo", "Unusual COMBINATION only"],
    M["uni_mv"]["churn_pct"],
    ["#B7BEC9", "#7FA0FF", T.AMBER, T.RED])


# ---------------------------------------------------------------------------
# PHASE 1 — distributions
# ---------------------------------------------------------------------------

DIST_FEATURES = ["Age", "Balance", "NumOfProducts", "CreditScore", "Tenure", "EstimatedSalary"]

DIST_INSIGHTS = {
    "Age": ("Churned customers are visibly OLDER: the red distribution is shifted right, "
            "peaking around 45-55 while retained customers peak around 30-37. Age is the single "
            "strongest churn correlate (|r| = 0.29) and the seed of every top Phase-3 rule."),
    "Balance": ("Two populations, not one: 36.2% of customers hold EXACTLY zero balance (left spike), "
                "the rest form a bell around 120K. The mean (76.5K) describes almost nobody - and "
                "counter-intuitively, churners are MORE common among positive balances (r = +0.12). "
                "Money does not buy loyalty in this book."),
    "NumOfProducts": ("Half the bank holds one product, 46% hold two, and 3-4 products are rare. "
                      "Churn is U-shaped across these values (28% / 8% / 83% / 100%) - a pattern a "
                      "linear correlation structurally cannot see."),
    "CreditScore": ("The only near-normal feature - and churned vs retained overlap almost perfectly. "
                    "Creditworthiness tells us nearly nothing about who leaves (|r| = 0.03)."),
    "Tenure": ("Near-uniform: customers are spread evenly across relationship years, and the two "
               "curves overlap. Loyalty in years does NOT protect against churn here (|r| = 0.01)."),
    "EstimatedSalary": ("Almost perfectly uniform from 0 to 200K, and identical for churned vs "
                        "retained. Salary carries essentially zero churn information - it cannot even "
                        "produce statistical outliers (max |z| = 1.74)."),
}


def make_dist_fig(col):
    fig = _base(300)
    nbins = {"NumOfProducts": 4, "Tenure": 11}.get(col, 45)
    for status, color in [("Retained", "#7FA0FF"), ("Churned", T.RED)]:
        sub = REC.loc[REC["Churn_Status"] == status, col]
        fig.add_histogram(x=sub, name=status, nbinsx=nbins, histnorm="percent",
                          marker_color=color, opacity=0.62,
                          marker_line=dict(width=0.4, color="white"))
    fig.update_layout(barmode="overlay", height=300,
                      xaxis_title=col, yaxis_title="% of group",
                      margin=dict(l=10, r=14, t=8, b=10))
    return fig


DIST_FIGS = {c: make_dist_fig(c) for c in DIST_FEATURES}

DIM_LABELS = {
    "Geography": "Country", "Gender": "Gender", "Age_Band": "Age band",
    "Active_Status": "Activity status", "NumOfProducts": "Number of products",
    "Balance_Band": "Balance band", "Tenure_Band": "Tenure band",
}

DIM_INSIGHTS = {
    "Geography": ("Germany churns at 32.4% - double France (16.2%) and Spain (16.7%) - despite being "
                  "a quarter of the book. Germany loses 814 customers from 2,509 while France loses "
                  "810 from 5,014. This single table motivated the project's Germany hypothesis, "
                  "formally confirmed as an association rule in Phase 3."),
    "Gender": ("Women churn at 25.1% vs 16.5% for men - a 1.5x gap invisible in the balanced "
               "population split. Phase 3 shows it widens further when combined with age "
               "(senior + female rules reach 57-66% churn)."),
    "Age_Band": ("The senior band (46-60) churns at 51% - 2.5x baseline - while young adults sit at "
                 "7.5%. Note the drop for 60+: the elderly who stayed are settled savers. This "
                 "non-monotonic shape is why age dominates every Phase-3 rule."),
    "Active_Status": ("Inactive members churn at 26.9% vs 14.3% for active ones. Meaningful, but "
                      "modest alone - Phase 3 shows inactivity only becomes dangerous when it "
                      "meets senior age (68%) and shallow product depth (77%)."),
    "NumOfProducts": ("The U-curve: two products is the safety sweet spot (7.6%), one product nearly "
                      "quadruples the risk (27.7%), and 3-4 products are near-certain leavers "
                      "(83% / 100%) - likely distressed or mis-sold relationships."),
    "Balance_Band": ("Churn rises monotonically with the band: zero balance 13.8%, insured "
                     "(0-100K) 20.6%, above the EUR 100K deposit-guarantee ceiling 25.2%. Balance "
                     "does not anchor loyalty - deposits beyond the state guarantee are the most "
                     "flight-prone, and senior variants of this band reach 58-73% churn in the "
                     "Phase-3 rules."),
    "Tenure_Band": ("Churn is flat across tenure (~19-22%). Years with the bank buy almost no "
                    "protection - engagement depth, not relationship length, is what matters."),
}


def make_dim_fig(dim):
    d = M["churn_by"][dim]
    colors = [risk_color(v) for v in d["churn_pct"]]
    fig = _base(300)
    fig.add_bar(x=d["labels"], y=d["churn_pct"], marker_color=colors, marker_line_width=0,
                text=[f"{v:.1f}%" for v in d["churn_pct"]], textposition="outside",
                textfont=dict(size=11.5),
                customdata=np.stack([d["churned"], d["total"]], axis=-1),
                hovertemplate="<b>%{x}</b><br>churn %{y:.1f}%<br>%{customdata[0]:,} of "
                              "%{customdata[1]:,} customers<extra></extra>")
    _baseline_hline(fig)
    fig.update_layout(height=300, yaxis_title="churn rate (%)",
                      yaxis_range=[0, max(d["churn_pct"]) * 1.28],
                      margin=dict(l=10, r=14, t=8, b=10))
    return fig


DIM_FIGS = {d: make_dim_fig(d) for d in DIM_LABELS}


# feature selection: two lenses side by side
def make_featsel_fig():
    fs = M["feature_selection"]
    order = np.argsort(fs["mutual_info"])            # ascending for horizontal bars
    feats = [fs["features"][i] for i in order]
    pear = [fs["pearson_abs"][i] for i in order]
    mi = [fs["mutual_info"][i] for i in order]
    ig = [fs["info_gain"][i] for i in order]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=("Lens 1 - Correlation  (Pearson |r|)",
                                        "Lens 2 - Entropy  (mutual information & info gain, bits)"))
    fig.add_bar(y=feats, x=[p if p is not None else 0 for p in pear], orientation="h",
                marker_color=["#D8E1FF" if p is None else T.BLUE for p in pear],
                text=["n/a (nominal)" if p is None else f"{p:.3f}" for p in pear],
                textposition="outside", textfont=dict(size=10.5),
                name="Pearson |r|", row=1, col=1, hoverinfo="skip")
    fig.add_bar(y=feats, x=mi, orientation="h", marker_color=T.PURPLE, name="Mutual information",
                text=[f"{v:.3f}" for v in mi], textposition="outside",
                textfont=dict(size=10.5), row=1, col=2, hoverinfo="skip")
    fig.add_bar(y=feats, x=ig, orientation="h", marker_color="#C7B5FD", name="Information gain",
                row=1, col=2, hovertemplate="info gain %{x:.4f} bits<extra></extra>")
    fig.update_layout(height=380, barmode="group", bargap=0.28,
                      legend=dict(y=1.12), margin=dict(l=10, r=20, t=42, b=10),
                      xaxis1=dict(range=[0, 0.36]),
                      xaxis2=dict(range=[0, max(max(mi), max(ig)) * 1.3]))
    fig.update_annotations(font_size=12)
    return fig


FEATSEL_FIG = make_featsel_fig()


# ---------------------------------------------------------------------------
# PHASE 2 — cluster map, validation, profiles
# ---------------------------------------------------------------------------

def _scatter(fig, x, y, name, color, symbol="circle", size=3.6, opacity=0.45, hover=None):
    fig.add_scattergl(x=x, y=y, mode="markers", name=name,
                      marker=dict(color=color, size=size, symbol=symbol, opacity=opacity,
                                  line=dict(width=0)),
                      hovertext=hover, hoverinfo="text" if hover is not None else "name")


def _pca_layout(fig, height=430):
    # reading guide (what left/right/up/down mean) lives in the card subtitle,
    # so the axes carry only what an axis should: name + % of variance.
    evr = M["validation"]["pca_evr"]
    fig.update_layout(
        height=height,
        xaxis_title=f"PC1 · {evr[0]}% of variance",
        yaxis_title=f"PC2 · {evr[1]}% of variance",
        legend=dict(itemsizing="constant"),
        margin=dict(l=10, r=14, t=10, b=10))


def make_pca_fig(algo, colorby):
    fig = _base()
    hover = ("Balance " + REC["Balance"].map("{:,.0f}".format) + " | products " +
             REC["NumOfProducts"].astype(str) + " | age " + REC["Age"].astype(str) +
             " | " + REC["Geography"] + " | " + REC["Churn_Status"])

    if colorby == "churn":
        for status in ["Retained", "Churned"]:
            m = REC["Churn_Status"] == status
            _scatter(fig, REC.loc[m, "PC1"], REC.loc[m, "PC2"],
                     f"{status} ({m.sum():,})", T.CHURN_COLORS[status],
                     opacity=0.4 if status == "Retained" else 0.55,
                     hover=hover[m])
    elif algo == "kmeans":
        for k in [1, 0, 2]:
            m = REC["Cluster"] == k
            _scatter(fig, REC.loc[m, "PC1"], REC.loc[m, "PC2"],
                     f"{T.CLUSTER_SHORT[k]} ({m.sum():,})", T.CLUSTER_COLORS[k], hover=hover[m])
    elif algo == "ward":
        # color Ward groups with the K-Means palette of their best-matching cluster
        mapping = {}
        for w in sorted(REC["Ward_Label"].unique()):
            mapping[w] = REC.loc[REC["Ward_Label"] == w, "Cluster"].mode()[0]
        for w, k in mapping.items():
            m = REC["Ward_Label"] == w
            _scatter(fig, REC.loc[m, "PC1"], REC.loc[m, "PC2"],
                     f"Ward {w} - matches C{k} ({m.sum():,})", T.CLUSTER_COLORS[k], hover=hover[m])
    else:  # dbscan
        palette = {0: T.BLUE, 1: T.TEAL, 2: T.PURPLE}
        for lab in sorted(REC["DBSCAN_Label"].unique()):
            m = REC["DBSCAN_Label"] == lab
            if lab == -1:
                _scatter(fig, REC.loc[m, "PC1"], REC.loc[m, "PC2"],
                         f"Noise / outliers ({m.sum():,}) - 62.6% churn", T.RED,
                         symbol="x", size=5, opacity=0.8, hover=hover[m])
            else:
                prod = M["dbscan"]["core_products"][str(lab)]
                _scatter(fig, REC.loc[m, "PC1"], REC.loc[m, "PC2"],
                         f"Core {lab} - {prod:.0f}-product customers ({m.sum():,})",
                         palette.get(lab, T.INK3), hover=hover[m])
    _pca_layout(fig)
    return fig


PCA_FIGS = {(a, c): make_pca_fig(a, c)
            for a in ["kmeans", "ward", "dbscan"] for c in ["segment", "churn"]}


def make_elbow_sil_fig():
    v = M["validation"]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=("Elbow method - inertia (WCSS)",
                                        "Silhouette score - separation quality"))
    fig.add_scatter(x=v["k_range"], y=v["inertia"], mode="lines+markers",
                    marker=dict(size=7, color=T.BLUE), line=dict(color=T.BLUE, width=2.4),
                    name="Inertia", row=1, col=1,
                    hovertemplate="K=%{x}<br>inertia %{y:,.0f}<extra></extra>")
    fig.add_vrect(x0=2.7, x1=5.3, fillcolor=T.BLUE, opacity=0.06, line_width=0, row=1, col=1)
    fig.add_annotation(x=4, y=v["inertia"][1], text="diminishing returns<br>zone (K=3-5)",
                       showarrow=False, font=dict(size=10.5, color=T.INK2), row=1, col=1)

    fig.add_scatter(x=v["k_range"], y=v["silhouette"], mode="lines+markers",
                    marker=dict(size=7, color=T.PURPLE), line=dict(color=T.PURPLE, width=2.4),
                    name="Silhouette", row=1, col=2,
                    hovertemplate="K=%{x}<br>silhouette %{y:.4f}<extra></extra>")
    fig.add_scatter(x=[2], y=[v["sil_k2"]], mode="markers+text", text=["  peak = trivial balance split"],
                    textposition="middle right", textfont=dict(size=10, color=T.INK2),
                    marker=dict(size=11, color="white", line=dict(color=T.PURPLE, width=2)),
                    showlegend=False, row=1, col=2, hoverinfo="skip")
    fig.add_scatter(x=[3], y=[v["sil_k3"]], mode="markers+text", text=["chosen K = 3"],
                    textposition="bottom center", textfont=dict(size=10.5, color=T.GREEN),
                    marker=dict(size=13, color=T.GREEN, symbol="star"),
                    showlegend=False, row=1, col=2, hoverinfo="skip")
    fig.update_layout(height=320, showlegend=False, margin=dict(l=10, r=14, t=42, b=10))
    fig.update_xaxes(title_text="number of clusters K", dtick=1)
    fig.update_yaxes(range=[0.122, 0.174], row=1, col=2)
    fig.update_annotations(font_size=12)
    return fig


ELBOW_SIL_FIG = make_elbow_sil_fig()


def make_snake_fig():
    s = M["snake"]
    fig = _base(330)
    for k in ["1", "0", "2"]:
        fig.add_scatter(x=s["features"], y=s["clusters"][k], mode="lines+markers",
                        name=T.CLUSTER_SHORT[int(k)],
                        line=dict(color=T.CLUSTER_COLORS[int(k)], width=2.6),
                        marker=dict(size=7),
                        hovertemplate="%{x}: %{y:+.2f} SD<extra></extra>")
    fig.add_hline(y=0, line_color=T.INK3, line_width=1)
    fig.update_layout(height=330, yaxis_title="deviation from bank average (SD units)",
                      margin=dict(l=10, r=14, t=10, b=10))
    return fig


SNAKE_FIG = make_snake_fig()


def make_effect_fig():
    best = {}
    for r in M["separation"]:
        if r["feature"] not in best or r["effect"] > best[r["feature"]]["effect"]:
            best[r["feature"]] = r
    rows = sorted(best.values(), key=lambda r: r["effect"])
    fig = _base(330)
    fig.add_bar(y=[r["feature"] for r in rows], x=[r["effect"] for r in rows], orientation="h",
                marker_color=[T.BLUE if r["effect"] >= 0.14 else "#D3D9E1" for r in rows],
                text=[f"{r['effect']:.2f}" for r in rows], textposition="outside",
                textfont=dict(size=10.5),
                customdata=[f"{r['test']} - {r['metric']}" for r in rows],
                hovertemplate="<b>%{y}</b><br>effect size %{x:.3f}<br>%{customdata}<extra></extra>")
    fig.add_vline(x=0.14, line_dash="dash", line_color=T.INK,
                  annotation_text="large effect", annotation_font_size=10)
    fig.update_layout(height=330, xaxis_title="effect size across the 3 segments",
                      xaxis_range=[0, 0.85], margin=dict(l=10, r=24, t=10, b=10))
    return fig


EFFECT_FIG = make_effect_fig()


def make_cluster_churn_fig():
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Churn rate per segment (validation lens)",
                                        "Where each segment lives (geography mix)"),
                        column_widths=[0.45, 0.55])
    ks = [1, 0, 2]
    names = [f"C{k}" for k in ks]
    churn = [M["clusters"][str(k)]["churn"] for k in ks]
    fig.add_bar(x=names, y=churn, marker_color=[T.CLUSTER_COLORS[k] for k in ks],
                text=[f"{v}%" for v in churn], textposition="outside",
                textfont=dict(size=12), row=1, col=1,
                customdata=[[M["clusters"][str(k)]["n"]] for k in ks],
                hovertemplate="<b>%{x}</b> churn %{y}%<br>%{customdata[0]:,} customers<extra></extra>",
                showlegend=False)
    _baseline_hline(fig, row=1, col=1)

    geo_colors = {"France": T.BLUE, "Germany": T.RED, "Spain": "#E8A93B"}
    for geo in ["France", "Germany", "Spain"]:
        fig.add_bar(y=names, x=[M["clusters"][str(k)]["geo_mix"][geo] for k in ks],
                    orientation="h", name=geo, marker_color=geo_colors[geo],
                    text=[f"{M['clusters'][str(k)]['geo_mix'][geo]:.0f}%" for k in ks],
                    textposition="inside", textfont=dict(size=10.5, color="white"),
                    hovertemplate=f"{geo}: %{{x:.1f}}%<extra></extra>", row=1, col=2)
    fig.update_layout(height=320, barmode="stack",
                      legend=dict(y=1.14, x=0.55), margin=dict(l=10, r=14, t=42, b=10))
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_yaxes(range=[0, 32], title_text="churn %", row=1, col=1)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", row=1, col=2)
    fig.update_annotations(font_size=12)
    return fig


CLUSTER_CHURN_FIG = make_cluster_churn_fig()


# ---------------------------------------------------------------------------
# PHASE 3 — rule network, scatter
# ---------------------------------------------------------------------------

def make_rule_network():
    rules = R["top10"]
    n = len(rules)
    # rule nodes on an inner ring, item nodes on an outer ring
    rule_angle = {i: (90 - i * (360 / n)) * np.pi / 180 for i in range(n)}
    items = {}
    for i, rule in enumerate(rules):
        for it in rule["if_items"]:
            items.setdefault(it, []).append(i)

    # place items at the circular mean of their rules' angles, then spread collisions
    item_angle = {}
    for it, idxs in items.items():
        sins = np.mean([np.sin(rule_angle[i]) for i in idxs])
        coss = np.mean([np.cos(rule_angle[i]) for i in idxs])
        item_angle[it] = np.arctan2(sins, coss)
    order = sorted(item_angle, key=lambda k: -item_angle[k])
    min_gap = 2 * np.pi / max(len(order), 1) * 0.9
    for j in range(1, len(order)):
        prev, cur = item_angle[order[j - 1]], item_angle[order[j]]
        if prev - cur < min_gap:
            item_angle[order[j]] = prev - min_gap

    R_RULE, R_ITEM = 1.0, 2.05
    fig = _base(470)

    # edges item -> rule
    ex, ey = [], []
    for it, idxs in items.items():
        ix, iy = R_ITEM * np.cos(item_angle[it]), R_ITEM * np.sin(item_angle[it])
        for i in idxs:
            rx, ry = R_RULE * np.cos(rule_angle[i]), R_RULE * np.sin(rule_angle[i])
            ex += [ix, rx, None]
            ey += [iy, ry, None]
    fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(color="#D3D9E1", width=1.1),
                    hoverinfo="skip", showlegend=False)

    # edges rule -> churn (width & color by lift)
    max_lift = max(r["lift"] for r in rules)
    for i, rule in enumerate(rules):
        rx, ry = R_RULE * np.cos(rule_angle[i]), R_RULE * np.sin(rule_angle[i])
        w = 1 + 5 * (rule["lift"] - 2.5) / (max_lift - 2.5)
        fig.add_scatter(x=[rx, 0], y=[ry, 0], mode="lines",
                        line=dict(color=T.RED, width=max(w, 1.2)),
                        opacity=0.5, hoverinfo="skip", showlegend=False)

    # item nodes
    fig.add_scatter(
        x=[R_ITEM * np.cos(item_angle[i]) for i in items],
        y=[R_ITEM * np.sin(item_angle[i]) for i in items],
        mode="markers+text", text=list(items.keys()),
        textposition=["middle right" if np.cos(item_angle[i]) >= 0.25 else
                      "middle left" if np.cos(item_angle[i]) <= -0.25 else
                      "top center" if np.sin(item_angle[i]) > 0 else "bottom center"
                      for i in items],
        textfont=dict(size=11, color=T.INK),
        marker=dict(size=[10 + 3.2 * len(v) for v in items.values()],
                    color=T.BLUE, opacity=0.9, line=dict(color="white", width=1.5)),
        customdata=[len(v) for v in items.values()],
        hovertemplate="<b>%{text}</b><br>appears in %{customdata} of the top-10 rules<extra></extra>",
        showlegend=False)

    # rule nodes
    fig.add_scatter(
        x=[R_RULE * np.cos(rule_angle[i]) for i in range(n)],
        y=[R_RULE * np.sin(rule_angle[i]) for i in range(n)],
        mode="markers+text", text=[r["letter"] for r in rules],
        textfont=dict(size=10, color="white", family=T.FONT),
        textposition="middle center",
        marker=dict(size=21, color=[T.RED if r["lift"] >= 3.2 else "#EC6142" if r["lift"] >= 2.9
                                    else T.AMBER for r in rules],
                    symbol="diamond", line=dict(color="white", width=1.5)),
        customdata=[[" + ".join(r["if_items"]), r["confidence_pct"], r["lift"], r["customers"]]
                    for r in rules],
        hovertemplate="<b>Rule %{text}</b><br>IF %{customdata[0]}<br>THEN churned - "
                      "confidence %{customdata[1]}% | lift %{customdata[2]:.2f}x | "
                      "%{customdata[3]:,} churners<extra></extra>",
        showlegend=False)

    # churn node
    fig.add_scatter(x=[0], y=[0], mode="markers+text", text=["CHURN"],
                    textfont=dict(size=11, color="white"), textposition="middle center",
                    marker=dict(size=52, color=T.RED, line=dict(color="#A61B37", width=2)),
                    hovertemplate="All 10 rules point here: customer leaves the bank<extra></extra>",
                    showlegend=False)

    fig.update_layout(height=470, xaxis=dict(visible=False, range=[-3.3, 3.3]),
                      yaxis=dict(visible=False, range=[-2.65, 2.65], scaleanchor="x"),
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


RULE_NETWORK_FIG = make_rule_network()


def make_rule_scatter():
    rules = R["top10"]
    fig = _base(360)
    # minimum-lift frontier: lift = confidence / base rate
    xs = np.linspace(50, 80, 40)
    fig.add_scatter(x=xs, y=xs / BASELINE, mode="lines",
                    line=dict(color=T.INK3, dash="dot", width=1.4),
                    name="lift implied by confidence (conf / 20.4%)",
                    hoverinfo="skip")
    sizes = [10 + r["support_pct"] * 6 for r in rules]
    fig.add_scatter(
        x=[r["confidence_pct"] for r in rules], y=[r["lift"] for r in rules],
        mode="markers+text", text=[r["letter"] for r in rules],
        textposition="top center", textfont=dict(size=11, color=T.INK2),
        marker=dict(size=sizes, color=[len(r["if_items"]) for r in rules],
                    colorscale=[[0, "#7FA0FF"], [0.5, T.BLUE], [1, T.BLUE_D]],
                    opacity=0.85, line=dict(color="white", width=1.5),
                    showscale=False),
        customdata=[[" + ".join(r["if_items"]), r["customers"], r["support_pct"], len(r["if_items"])]
                    for r in rules],
        hovertemplate="<b>Rule %{text}</b>  (%{customdata[3]} conditions)<br>"
                      "IF %{customdata[0]}<br>confidence %{x}% | lift %{y:.2f}x<br>"
                      "%{customdata[1]:,} churners (support %{customdata[2]}%)<extra></extra>",
        name="top-10 rules")
    fig.update_layout(height=360, xaxis_title="confidence - % of matching customers who churned",
                      yaxis_title="lift vs 20.4% baseline",
                      xaxis=dict(ticksuffix="%"),
                      legend=dict(y=1.1), margin=dict(l=10, r=14, t=8, b=10))
    return fig


RULE_SCATTER_FIG = make_rule_scatter()


# ---------------------------------------------------------------------------
# PHASE 4 — anomaly figures
# ---------------------------------------------------------------------------

def make_method_fig():
    rows = sorted(M["anomaly_methods"], key=lambda r: r["churn_pct"])
    fig = _base(340)
    fig.add_bar(
        y=[r["method"] for r in rows], x=[r["churn_pct"] for r in rows], orientation="h",
        marker_color=[T.BLUE if r["family"] == "Multivariate" else "#B7BEC9" for r in rows],
        text=[f"{r['churn_pct']}%  ({r['flagged']:,} flagged)" for r in rows],
        textposition="outside", textfont=dict(size=11),
        customdata=[[r["flagged"], r["pct"], r["family"]] for r in rows],
        hovertemplate="<b>%{y}</b> (%{customdata[2]})<br>flags %{customdata[0]:,} customers "
                      "(%{customdata[1]}% of book)<br>churn among flagged: %{x}%<extra></extra>")
    fig.add_vline(x=BASELINE, line_dash="dash", line_color=T.INK, layer="below",
                  annotation_text=f"baseline {BASELINE:.1f}%", annotation_font_size=10.5,
                  annotation_position="bottom right")
    fig.update_layout(height=340, xaxis_title="churn rate among flagged customers (%)",
                      xaxis_range=[0, 84], margin=dict(l=10, r=30, t=8, b=10))
    return fig


METHOD_FIG = make_method_fig()


def make_composite_fig():
    c = M["composite"]
    fig = _base(320)
    colors = ["#B7BEC9", T.AMBER, T.RED, "#8A93A3", "#8A93A3"]
    fig.add_bar(x=[str(s) for s in c["scores"]], y=c["churn_pct"],
                marker_color=colors,
                text=[f"{v}%<br><span style='font-size:10px'>n={n:,}</span>"
                      for v, n in zip(c["churn_pct"], c["n"])],
                textposition="outside", textfont=dict(size=11),
                customdata=c["n"],
                hovertemplate="flagged by %{x} methods<br>churn %{y}%<br>%{customdata:,} "
                              "customers<extra></extra>")
    fig.add_annotation(x="2", y=c["churn_pct"][2] + 13, text="<b>peak risk here</b><br>(IF + DBSCAN agree)",
                       showarrow=False, font=dict(size=11, color=T.RED))
    fig.add_annotation(x="4", y=c["churn_pct"][4] + 27, text="'full consensus' =<br>mostly benign retirees",
                       showarrow=False, font=dict(size=10.5, color=T.INK2))
    fig.update_layout(height=320,
                      xaxis_title="how many of the 4 core methods flagged the customer "
                                  "(bank average churn = 20.4%)",
                      yaxis_title="churn rate (%)", yaxis_range=[0, 92],
                      margin=dict(l=10, r=14, t=8, b=10))
    return fig


COMPOSITE_FIG = make_composite_fig()


def make_unimv_fig():
    u = M["uni_mv"]
    labels = ["Flagged by neither family", "Univariate only<br>(one extreme value)",
              "Both families", "Multivariate only<br>(unusual combination)"]
    fig = _base(320)
    fig.add_bar(x=labels, y=u["churn_pct"],
                marker_color=["#B7BEC9", "#7FA0FF", T.AMBER, T.RED],
                text=[f"{v}%<br><span style='font-size:10px'>n={n:,}</span>"
                      for v, n in zip(u["churn_pct"], u["n"])],
                textposition="outside", textfont=dict(size=11),
                hoverinfo="skip")
    _baseline_hline(fig)
    fig.update_layout(height=320, yaxis_title="churn rate (%)", yaxis_range=[0, 58],
                      margin=dict(l=10, r=14, t=8, b=10))
    return fig


UNIMV_FIG = make_unimv_fig()


def make_outlier_scatter(colorby):
    fig = _base(430)
    hover = ("Balance " + REC["Balance"].map("{:,.0f}".format) + " | age " +
             REC["Age"].astype(str) + " | products " + REC["NumOfProducts"].astype(str) +
             " | " + REC["Geography"] + " | flags: " +
             REC["Composite_Anomaly_Score"].astype(str) + "/4")
    if colorby == "churn":
        for status in ["Retained", "Churned"]:
            m = REC["Churn_Status"] == status
            _scatter(fig, REC.loc[m, "Balance"], REC.loc[m, "IF_score"],
                     f"{status} ({m.sum():,})", T.CHURN_COLORS[status],
                     opacity=0.35 if status == "Retained" else 0.55, hover=hover[m])
    else:
        groups = [("Normal — Not Flagged", "Not flagged", "#B7BEC9", 3.0, 0.3),
                  ("B:", "B - Rare but legitimate", T.AMBER, 4.5, 0.75),
                  ("C:", "C - RISK SIGNAL", T.RED, 4.5, 0.8),
                  ("A:", "A - Suspected data error", T.PURPLE, 9, 1.0)]
        for prefix, name, color, size, op in groups:
            m = REC["Anomaly_Class"].str.startswith(prefix)
            _scatter(fig, REC.loc[m, "Balance"], REC.loc[m, "IF_score"],
                     f"{name} ({m.sum():,})", color, size=size, opacity=op, hover=hover[m])
    thr = REC.loc[REC["IF_flag"] == 1, "IF_score"].max()
    fig.add_hline(y=thr, line_dash="dash", line_color=T.INK,
                  annotation_text="Isolation-Forest flag threshold (5% most isolable)",
                  annotation_font_size=10.5, annotation_position="bottom right")
    fig.update_layout(height=430, xaxis_title="account balance",
                      yaxis_title="Isolation Forest score  (lower = more anomalous)",
                      margin=dict(l=10, r=14, t=10, b=10))
    return fig


OUTLIER_FIGS = {c: make_outlier_scatter(c) for c in ["class", "churn"]}


def make_class_donut():
    flagged = REC[REC["Composite_Anomaly_Score"] >= 1]
    counts = {
        "C - Risk signal": int(flagged["Anomaly_Class"].str.startswith("C").sum()),
        "B - Rare but valid": int(flagged["Anomaly_Class"].str.startswith("B").sum()),
        "A - Suspected data error": int(flagged["Anomaly_Class"].str.startswith("A").sum()),
    }
    fig = _base(300)
    fig.add_pie(labels=list(counts.keys()), values=list(counts.values()), hole=0.62,
                marker=dict(colors=[T.RED, T.AMBER, T.PURPLE],
                            line=dict(color="white", width=2)),
                texttemplate="%{percent:.1%}", textfont=dict(size=12),
                hovertemplate="<b>%{label}</b><br>%{value:,} records (%{percent})<extra></extra>")
    fig.add_annotation(text=f"<b>{len(flagged):,}</b><br><span style='font-size:11px;"
                            f"color:{T.INK2}'>flagged<br>records</span>",
                       showarrow=False, font=dict(size=20))
    fig.update_layout(height=300, legend=dict(orientation="v", y=0.5, x=1.02),
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


CLASS_DONUT_FIG = make_class_donut()


def make_subtype_fig():
    rows = sorted(M["anomaly_classes"], key=lambda r: r["n"])
    short = {
        "B: Rare Valid — Statistically Unusual Pattern": "B - Unusual but valid pattern",
        "B: Rare Valid — Young High-Balance Customer": "B - Young, high balance",
        "B: Rare Valid — Maximum Product Holder": "B - Holds 4 products",
        "C: Risk Signal — Density Outlier + Churned": "C - Density outlier + churned",
        "C: Risk Signal — High-Balance Pre-Churn": "C - High-balance pre-churn",
        "C: Risk Signal — Disengaged Single-Product Churn": "C - Disengaged single-product",
        "C: Risk Signal — Senior High-Value Departure": "C - Senior high-value departure",
        "A: Data Error": "A - Suspected data error",
    }
    fig = _base(300)
    fig.add_bar(y=[short.get(r["cls"], r["cls"]) for r in rows], x=[r["n"] for r in rows],
                orientation="h",
                marker_color=[T.RED if r["cls"].startswith("C") else
                              T.PURPLE if r["cls"].startswith("A") else T.AMBER for r in rows],
                text=[f"{r['n']:,}" for r in rows], textposition="outside",
                textfont=dict(size=11),
                customdata=[[r["pct_of_flagged"], r["churn_pct"]] for r in rows],
                hovertemplate="<b>%{y}</b><br>%{x:,} records (%{customdata[0]}% of flagged)<br>"
                              "churn %{customdata[1]}%<extra></extra>")
    fig.update_layout(height=300, xaxis_title="flagged records", xaxis_range=[0, 560],
                      margin=dict(l=10, r=30, t=8, b=10))
    return fig


SUBTYPE_FIG = make_subtype_fig()


def make_crossref_fig():
    ks = [1, 0, 2]
    noise = [M["cross_ref"]["noise_by_cluster"].get(str(k), 0) for k in ks]
    share = [n / M["clusters"][str(k)]["n"] * 100 for n, k in zip(noise, ks)]
    fig = _base(300)
    fig.add_bar(x=[f"C{k} - {T.CLUSTER_SHORT[k].split('· ')[1]}" for k in ks], y=noise,
                marker_color=[T.CLUSTER_COLORS[k] for k in ks],
                text=[f"{n:,}<br><span style='font-size:10px'>{s:.1f}% of segment</span>"
                      for n, s in zip(noise, share)],
                textposition="outside", textfont=dict(size=11), hoverinfo="skip")
    fig.update_layout(height=300, yaxis_title="DBSCAN noise points inside segment",
                      yaxis_range=[0, max(noise) * 1.4], margin=dict(l=10, r=14, t=8, b=10))
    return fig


CROSSREF_FIG = make_crossref_fig()
