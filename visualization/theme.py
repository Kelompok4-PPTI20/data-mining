"""
Design tokens + Plotly template — implements the KDD Design Spec (v1.0).

Cool-slate neutral ramp, one Cobalt accent, semantic solid+tint pairs,
8-series categorical dataviz palette, Geist / Geist Mono typography.
Legacy alias names (BLUE, RED, ...) are preserved so figures/components
keep working; their values now point at spec tokens.
"""

import plotly.graph_objects as go
import plotly.io as pio

# -- neutrals · cool slate ----------------------------------------------------
SURFACE_0, SURFACE_50, SURFACE_100 = "#FFFFFF", "#F7F8FA", "#F1F3F6"
BORDER_200, BORDER_300 = "#E4E8EE", "#D3D9E1"
INK_300, INK_400, INK_500 = "#B7BEC9", "#8A93A3", "#5A6474"
INK_700, INK_800, INK_900 = "#3A4453", "#1A2130", "#0B0F1A"

# -- brand · Cobalt is the single interactive accent --------------------------
COBALT, COBALT_HOVER, COBALT_ACTIVE = "#2D5BFF", "#1E48E0", "#173AC0"
COBALT_TINT = "#EAEFFF"
COBALT_SOFT = "#7FA0FF"            # light cobalt (spec cover accent)

# -- semantic · solid + 10% tint, reserved strictly for state -----------------
SUCCESS, SUCCESS_TINT = "#17966B", "#E3F6EE"
WARNING, WARNING_TINT = "#C77700", "#FCF0DC"
CRITICAL, CRITICAL_TINT = "#D6304A", "#FCE7EB"
INFO, INFO_TINT = COBALT, COBALT_TINT

# -- data-visualization palette · assign in order, never skip -----------------
SERIES = ["#2D5BFF", "#7C5CFC", "#12A594", "#E8A93B",
          "#EC6142", "#3AA655", "#D6467F", "#6B7480"]

# -- legacy aliases (semantic roles preserved across figures/components) ------
BLUE, BLUE_D, BLUE_BG = COBALT, COBALT_ACTIVE, COBALT_TINT
RED, RED_BG = CRITICAL, CRITICAL_TINT
GREEN, GREEN_BG = SUCCESS, SUCCESS_TINT
AMBER, AMBER_BG = WARNING, WARNING_TINT
PURPLE, TEAL = SERIES[1], SERIES[2]
INK, INK2, INK3, LINE = INK_900, INK_500, INK_400, BORDER_200

COLORWAY = SERIES

# cluster IDENTITY colors = categorical series in order (spec rule);
# churn/risk STATE keeps the semantic palette.
CLUSTER_COLORS = {0: SERIES[0], 1: SERIES[1], 2: SERIES[2]}
CLUSTER_SHORT = {0: "C0 · Germany-skew multi-product", 1: "C1 · Single-product watchlist",
                 2: "C2 · Zero-balance loyalists"}
CHURN_COLORS = {"Retained": COBALT_SOFT, "Churned": CRITICAL}

FONT = "Geist, Inter, system-ui, -apple-system, sans-serif"
FONT_MONO = "'Geist Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

# -- plotly template -----------------------------------------------------------
pio.templates["gkdd"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT, size=12.5, color=INK_700),
        title=dict(font=dict(size=14, color=INK_900)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        margin=dict(l=10, r=14, t=26, b=10),
        xaxis=dict(gridcolor=BORDER_200, griddash="dash", zerolinecolor=BORDER_200,
                   linecolor=BORDER_200, ticks="outside", tickcolor="rgba(0,0,0,0)",
                   automargin=True,
                   title=dict(font=dict(size=11.5, color=INK_500)),
                   tickfont=dict(size=11, color=INK_400)),
        yaxis=dict(gridcolor=BORDER_200, griddash="dash", zerolinecolor=BORDER_200,
                   linecolor="rgba(0,0,0,0)", automargin=True,
                   title=dict(font=dict(size=11.5, color=INK_500)),
                   tickfont=dict(size=11, color=INK_400)),
        legend=dict(font=dict(size=11.5, color=INK_500), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="white", bordercolor=BORDER_200,
                        font=dict(family=FONT, size=12, color=INK_900)),
        bargap=0.35,
    )
)
pio.templates.default = "gkdd"

GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}


def pct(x, d=1):
    return f"{x:.{d}f}%"


def n_fmt(x):
    return f"{x:,.0f}"
