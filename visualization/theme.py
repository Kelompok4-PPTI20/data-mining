"""Shared visual identity: Google-inspired palette + a clean Plotly template."""

import plotly.graph_objects as go
import plotly.io as pio

# -- palette -----------------------------------------------------------------
BLUE, BLUE_D, BLUE_BG = "#1a73e8", "#174ea6", "#e8f0fe"
RED, RED_BG = "#d93025", "#fce8e6"
GREEN, GREEN_BG = "#188038", "#e6f4ea"
AMBER, AMBER_BG = "#f9ab00", "#fef7e0"
PURPLE, TEAL = "#9334e6", "#12b5cb"
INK, INK2, INK3, LINE = "#202124", "#5f6368", "#80868b", "#e8eaed"

COLORWAY = [BLUE, RED, AMBER, GREEN, PURPLE, TEAL, "#e8710a", "#7cacf8"]

# semantic colors for the three K-Means personas (risk-coded)
CLUSTER_COLORS = {0: AMBER, 1: RED, 2: GREEN}
CLUSTER_SHORT = {0: "C0 · Germany-skew multi-product", 1: "C1 · Single-product watchlist",
                 2: "C2 · Zero-balance loyalists"}
CHURN_COLORS = {"Retained": "#9ec3f7", "Churned": RED}

FONT = "Inter, Roboto, Segoe UI, system-ui, sans-serif"

# -- template ----------------------------------------------------------------
pio.templates["gkdd"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT, size=12.5, color=INK),
        title=dict(font=dict(size=14, color=INK)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        margin=dict(l=10, r=14, t=26, b=10),
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                   ticks="outside", tickcolor="rgba(0,0,0,0)", automargin=True,
                   title=dict(font=dict(size=11.5, color=INK2)),
                   tickfont=dict(size=11, color=INK2)),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor="rgba(0,0,0,0)",
                   automargin=True,
                   title=dict(font=dict(size=11.5, color=INK2)),
                   tickfont=dict(size=11, color=INK2)),
        legend=dict(font=dict(size=11.5, color=INK2), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="white", bordercolor=LINE,
                        font=dict(family=FONT, size=12, color=INK)),
        bargap=0.35,
    )
)
pio.templates.default = "gkdd"

GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}


def pct(x, d=1):
    return f"{x:.{d}f}%"


def n_fmt(x):
    return f"{x:,.0f}"
