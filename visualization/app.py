"""
Phase 5 — Interactive Knowledge Discovery Dashboard (Group 5, Bank Customer Churn)

Python Dash app visualizing the four KDD phases for a non-technical audience:
cluster maps (FAMD), association-rule network, outlier plots, key distributions,
and the central discovery answer.

Design for the <100 ms interactivity rubric bar:
  1. All heavy computation happens once in prepare_data.py.
  2. Every callback is an in-memory filter on a preloaded 10K-row frame.
  3. Every control has a small finite input space, so callback results are
     memoized (lru_cache) and pre-warmed at startup — interactions are cache hits.

Run from this folder (after prepare_data.py):

    python app.py          ->  http://127.0.0.1:8050
"""
import json
import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, dash_table

# ── Load precomputed data ──────────────────────────────────────────────────────
if not os.path.exists('viz_data/summary.json'):
    sys.exit('viz_data/ not found — run `python prepare_data.py` first.')

SUMMARY = json.load(open('viz_data/summary.json'))
FAMD = pd.read_csv('viz_data/famd_coords.csv')
FAMD[['Dim1', 'Dim2']] = FAMD[['Dim1', 'Dim2']].round(4)
RULES = pd.read_csv('viz_data/rules_churn.csv').fillna({'consequent_extra': ''})
CUST = pd.read_csv('../data/processed/churn_clustered.csv')
ANOM = pd.read_csv('../outputs/ph4_anomaly_report.csv')
ANOM['IF_anomaly_score'] = ANOM['IF_anomaly_score'].round(4)
ANOM['Balance'] = ANOM['Balance'].round(0)

# Default cluster-map viewport: central 99.6% of customers (a few extreme FAMD
# outliers otherwise flatten the whole cloud; they stay plotted — just pan/zoom)
_q = FAMD[['Dim1', 'Dim2']].quantile([0.002, 0.998])
_px = (_q.loc[0.998, 'Dim1'] - _q.loc[0.002, 'Dim1']) * 0.06
_py = (_q.loc[0.998, 'Dim2'] - _q.loc[0.002, 'Dim2']) * 0.06
XRANGE = [_q.loc[0.002, 'Dim1'] - _px, _q.loc[0.998, 'Dim1'] + _px]
YRANGE = [_q.loc[0.002, 'Dim2'] - _py, _q.loc[0.998, 'Dim2'] + _py]

BASELINE = SUMMARY['baseline_churn']
PERSONA_COLORS = {0: '#2a9d8f', 1: '#e76f51', 2: '#6d597a'}
CHURN_RED, RETAIN_BLUE, GREY = '#d62828', '#457b9d', '#cfd6dd'
TEMPLATE = 'plotly_white'

# Looker-style typography inside figures
import plotly.io as pio
pio.templates[TEMPLATE].layout.font = dict(family='Roboto, "Segoe UI", Arial, sans-serif',
                                           color='#202124')
pio.templates[TEMPLATE].layout.title.font.size = 14

# Plain-business-language names for transaction items
ITEM_LABELS = {
    'Age_Band_Senior': 'Age 46–60', 'Age_Band_Elderly': 'Age over 60',
    'Age_Band_Middle_Aged': 'Age 31–45', 'Age_Band_Young_Adult': 'Age up to 30',
    'Active_Status_Inactive': 'Inactive member', 'Active_Status_Active': 'Active member',
    'Products_Label_Products_1': 'Single product', 'Products_Label_Products_2': 'Two products',
    'Geography_Germany': 'Germany', 'Geography_France': 'France', 'Geography_Spain': 'Spain',
    'Gender_Female': 'Female', 'Gender_Male': 'Male',
    'CrCard_Status_Has_CrCard': 'Has credit card', 'CrCard_Status_No_CrCard': 'No credit card',
    'Balance_Band_Zero_Balance': 'Zero balance', 'Balance_Band_Low_Balance': 'Balance under £50K',
    'Balance_Band_Mid_Balance': 'Balance £50–125K', 'Balance_Band_High_Balance': 'Balance over £125K',
    'Tenure_Band_New_Customer': 'Customer 0–2 yrs', 'Tenure_Band_Established': 'Customer 3–5 yrs',
    'Tenure_Band_Loyal': 'Customer 6–10 yrs',
}


def nice(item):
    return ITEM_LABELS.get(item, item.replace('_', ' '))


def kpi(label, value, sub=''):
    return html.Div([html.Div(value, className='kpi-value'),
                     html.Div(label, className='kpi-label'),
                     html.Div(sub, className='kpi-sub')], className='kpi-card')


def disco_card(num, title, stat, body, action):
    return html.Div([
        html.Div(f'Discovery {num}', className='disco-tag'),
        html.H3(title), html.Div(stat, className='disco-stat'),
        html.P(body), html.P([html.B('What to do: '), action], className='disco-action'),
    ], className='disco-card')


# ══ TAB 1 — The Discovery (central question answered up front) ═════════════════
hb = SUMMARY['highbal_churners']
discovery_tab = html.Div([
    html.Div([
        html.H2('“What did we discover that was not already obvious from the raw data?”'),
        html.P('The raw file says one in five customers leaves the bank. It does not say '
               'which ones, why, or what it costs. Mining the data as one connected '
               'discovery exercise — segments, rules, anomalies — surfaced four findings '
               'no single column or average reveals.', className='lede'),
    ], className='question-banner'),
    html.Div([
        disco_card(1, 'The mid-life churn cliff',
                   '77% churn — nearly 4× the average',
                   'Customers aged 46–60 churn at 51% vs 14% for everyone else. When that age '
                   'band combines with inactivity and a single product, churn hits 77.3% '
                   '(lift 3.8). Age alone, inactivity alone, or product count alone never '
                   'shows this — it is the interaction that drives the risk.',
                   'Trigger a retention call when a 46–60 customer with one product goes '
                   'inactive for a quarter — before the second quarter.'),
        disco_card(2, 'Germany is a structural problem',
                   '32% churn — double France and Spain',
                   'German customers churn at 32.4% vs ~16% elsewhere, German seniors at '
                   '67.3% (lift 3.3), and Germany supplies 46.3% of all high-balance '
                   'churners. This is market-level, not customer-level: a product-fit or '
                   'service issue specific to the German operation.',
                   'Commission a German market review; benchmark fees, rates and service '
                   'against local competitors.'),
        disco_card(3, 'Engagement beats balance as a loyalty signal',
                   '£6K loyalists stay; £122K holders drift',
                   'The most loyal segment holds the least money (avg £6.2K) but nearly two '
                   'products each — churn 15.7%. The wealthiest segments (£115–122K) hold '
                   '~1.3 products and churn at 23–24%. One-product customers churn at 27.7%; '
                   'two-product customers at 7.6%.',
                   'Measure relationship depth, not balance, as the loyalty KPI; cross-sell '
                   'a second product to single-product holders.'),
        disco_card(4, 'High-value money leaves quietly',
                   f"{hb['n']} churners took ~£{hb['avg_balance']/1000:.0f}K each",
                   'Structurally unusual customers (caught independently by density- and '
                   'tree-based detectors) churn at 49–52% — 2.5× baseline. Among them sit '
                   f"{hb['n']} high-balance churners averaging £{hb['avg_balance']:,.0f}, "
                   'the costliest losses in the book.',
                   'Alert a relationship manager whenever a high-balance account picks up '
                   'an anomaly flag — these are the most expensive customers to lose.'),
    ], className='disco-grid'),
], className='tab-body')

# ══ TAB 2 — Customer Segments (cluster maps) ═══════════════════════════════════
segments_tab = html.Div([
    html.Div([
        html.Div([
            html.Label('Color the map by'),
            dcc.Dropdown(id='seg-color', clearable=False, value='persona',
                         options=[{'label': 'Customer persona (K-Means)', 'value': 'persona'},
                                  {'label': 'Churn status', 'value': 'churn'},
                                  {'label': 'DBSCAN structural outliers', 'value': 'noise'}]),
            dcc.Graph(id='seg-map', config={'displayModeBar': False}),
            html.P('2-D FAMD projection of all 10,000 customers (mixed numeric + categorical '
                   'data). Axes are abstract pattern dimensions, not raw features. '
                   'Default view shows the central 99.6% — zoom out for extremes.',
                   className='fineprint'),
        ], className='col-7'),
        html.Div([
            html.Label('Inspect a persona'),
            dcc.Dropdown(id='seg-persona', clearable=False, value=0,
                         options=[{'label': f"C{p['cluster']} — {p['name']}", 'value': p['cluster']}
                                  for p in SUMMARY['cluster_profiles']]),
            html.Div(id='seg-profile'),
        ], className='col-5'),
    ], className='row'),
], className='tab-body')

# ══ TAB 3 — Churn Rules (rule network) ═════════════════════════════════════════
rules_tab = html.Div([
    html.Div([
        html.Label('Minimum lift (how many times above the 20.4% baseline churn rate)'),
        dcc.Slider(id='rule-lift', min=1.5, max=3.8, step=0.1, value=2.5,
                   marks={1.5: '1.5×', 2.0: '2×', 2.5: '2.5×', 3.0: '3×', 3.5: '3.5×', 3.8: '3.8×'}),
    ], className='control-strip'),
    html.Div([
        html.Div(dcc.Graph(id='rule-net', config={'displayModeBar': False}), className='col-7'),
        html.Div([
            html.H4(id='rule-count'),
            dash_table.DataTable(
                id='rule-table',
                columns=[{'name': 'IF a customer is…', 'id': 'ant'},
                         {'name': 'Support', 'id': 'sup'},
                         {'name': 'Churn rate', 'id': 'conf'},
                         {'name': 'Lift', 'id': 'lift'}],
                style_cell={'fontFamily': 'inherit', 'fontSize': 13, 'padding': '6px 10px',
                            'whiteSpace': 'normal', 'height': 'auto', 'textAlign': 'left'},
                style_header={'fontWeight': '500', 'backgroundColor': '#f8f9fa', 'color': '#5f6368', 'textTransform': 'uppercase', 'fontSize': 11.5, 'letterSpacing': '0.4px', 'borderBottom': '1px solid #dadce0'},
                style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}],
                page_size=8),
            html.P('Reading guide: “Support” = share of all 10,000 customers matching the '
                   'profile AND churning. “Churn rate” = confidence — of customers matching '
                   'the profile, the share who churned (baseline 20.4%). “Lift” = how many '
                   'times the baseline that is.', className='fineprint'),
        ], className='col-5'),
    ], className='row'),
], className='tab-body')

# ══ TAB 4 — Anomalies & Outliers ═══════════════════════════════════════════════
mt = pd.DataFrame(SUMMARY['anomaly_methods'])
anomaly_tab = html.Div([
    html.Div([
        html.Label('Show customers flagged by'),
        dcc.Checklist(id='anom-methods', inline=True,
                      value=['IF_flag', 'DBSCAN_flag'],
                      options=[{'label': ' IQR', 'value': 'IQR_flag'},
                               {'label': ' Z-score', 'value': 'ZScore_flag'},
                               {'label': ' Isolation Forest', 'value': 'IF_flag'},
                               {'label': ' DBSCAN noise', 'value': 'DBSCAN_flag'}]),
    ], className='control-strip'),
    html.Div([
        html.Div(dcc.Graph(id='anom-scatter', config={'displayModeBar': False}), className='col-7'),
        html.Div(dcc.Graph(id='anom-composite', config={'displayModeBar': False}), className='col-5'),
    ], className='row'),
    html.Div([
        html.Div([
            html.H4('Detection methods compared'),
            dash_table.DataTable(
                data=mt.rename(columns={'method': 'Method', 'flagged': 'Flagged',
                                        'pct': '% of book', 'churn_rate': 'Churn % in flagged'})
                       .to_dict('records'),
                style_cell={'fontFamily': 'inherit', 'fontSize': 13, 'padding': '6px 10px'},
                style_header={'fontWeight': '500', 'backgroundColor': '#f8f9fa', 'color': '#5f6368', 'textTransform': 'uppercase', 'fontSize': 11.5, 'letterSpacing': '0.4px', 'borderBottom': '1px solid #dadce0'}),
            html.P(f"{SUMMARY['n_flagged']:,} customers were flagged by at least one method; "
                   f"{SUMMARY['n_high_conf']} by three or four. Each flagged record was "
                   'classified as Data Error (2), Rare-but-Legitimate (583) or Risk Signal '
                   '(469).', className='fineprint'),
        ], className='col-6'),
        html.Div(dcc.Graph(id='anom-class', config={'displayModeBar': False}), className='col-6'),
    ], className='row'),
], className='tab-body')

# ══ TAB 5 — Distributions ══════════════════════════════════════════════════════
dist_tab = html.Div([
    html.Div([
        html.Label('Market'),
        dcc.Dropdown(id='dist-geo', clearable=False, value='All',
                     options=[{'label': g, 'value': g} for g in
                              ['All', 'France', 'Germany', 'Spain']],
                     style={'width': '220px'}),
    ], className='control-strip'),
    html.Div([
        html.Div(dcc.Graph(id='dist-geo-churn', config={'displayModeBar': False}), className='col-6'),
        html.Div(dcc.Graph(id='dist-age', config={'displayModeBar': False}), className='col-6'),
    ], className='row'),
    html.Div([
        html.Div(dcc.Graph(id='dist-balance', config={'displayModeBar': False}), className='col-6'),
        html.Div(dcc.Graph(id='dist-products', config={'displayModeBar': False}), className='col-6'),
    ], className='row'),
], className='tab-body')

# ══ App shell ══════════════════════════════════════════════════════════════════
app = Dash(__name__, title='Group 5 — Churn Knowledge Discovery')

app.layout = html.Div([
    html.Div([
        html.H1('Where the Bank Quietly Loses Its Customers'),
        html.P('Knowledge discovery in a 10,000-customer retail-banking book · '
               'Group 5 · Bank Customer Churn · KDD Phases 1–5', className='byline'),
        html.Div([
            kpi('Customers analysed', f"{SUMMARY['n_customers']:,}", '3 markets'),
            kpi('Baseline churn', f'{BASELINE:.1f}%', '1 in 5 leaves'),
            kpi('Customer personas', '3', 'K-Means · Ward · DBSCAN'),
            kpi('Churn rules mined', str(SUMMARY['n_churn_rules']), 'lift up to 3.8×'),
            kpi('Anomalies flagged', f"{SUMMARY['n_flagged']:,}", '4 methods'),
            kpi('High-value churners', str(hb['n']), f"~£{hb['avg_balance']/1000:.0f}K each"),
        ], className='kpi-strip'),
    ], className='header'),
    dcc.Tabs(className='looker-tabs', children=[
        dcc.Tab(label='The Discovery', children=discovery_tab, className='looker-tab', selected_className='looker-tab--selected'),
        dcc.Tab(label='Customer Segments', children=segments_tab, className='looker-tab', selected_className='looker-tab--selected'),
        dcc.Tab(label='Churn Rules', children=rules_tab, className='looker-tab', selected_className='looker-tab--selected'),
        dcc.Tab(label='Anomalies', children=anomaly_tab, className='looker-tab', selected_className='looker-tab--selected'),
        dcc.Tab(label='Distributions', children=dist_tab, className='looker-tab', selected_className='looker-tab--selected'),
    ]),
    html.Div('Group 5 · Data Mining · KDD methodology — discovery, not prediction. '
             'All figures recomputable from notebooks/notebook.ipynb.', className='footer'),
])


# ══ Cached view builders (memoized: every control has a finite input space) ════
@lru_cache(maxsize=8)
def build_segment_map(mode):
    fig = go.Figure()
    # constant-per-trace label goes in the template, not in 10K customdata rows
    cd = np.column_stack([FAMD['Geography'], FAMD['Age'],
                          FAMD['Balance'].round(0)])

    def tmpl(label):
        return (f'<b>{label}</b><br>%{{customdata[0]}} · Age %{{customdata[1]}}'
                '<br>Balance £%{customdata[2]:,.0f}<extra></extra>')

    if mode == 'persona':
        for c in sorted(FAMD['Cluster'].unique()):
            m = (FAMD['Cluster'] == c).values
            label = FAMD.loc[m, 'Cluster_Name'].iloc[0]
            fig.add_trace(go.Scattergl(
                x=FAMD.loc[m, 'Dim1'], y=FAMD.loc[m, 'Dim2'], mode='markers',
                name=f'C{c}', marker=dict(size=4, color=PERSONA_COLORS[c], opacity=0.45),
                customdata=cd[m], hovertemplate=tmpl(label)))
    elif mode == 'churn':
        for val, name, color in [(0, 'Retained', RETAIN_BLUE), (1, 'Churned', CHURN_RED)]:
            m = (FAMD['Exited'] == val).values
            fig.add_trace(go.Scattergl(
                x=FAMD.loc[m, 'Dim1'], y=FAMD.loc[m, 'Dim2'], mode='markers', name=name,
                marker=dict(size=4, color=color, opacity=0.45 if val else 0.25),
                customdata=cd[m], hovertemplate=tmpl(name)))
    else:
        for val, name, color, op in [(0, 'Core customers', GREY, 0.3),
                                     (1, 'Structural outliers (52% churn)', CHURN_RED, 0.8)]:
            m = (FAMD['DBSCAN_noise'] == val).values
            fig.add_trace(go.Scattergl(
                x=FAMD.loc[m, 'Dim1'], y=FAMD.loc[m, 'Dim2'], mode='markers', name=name,
                marker=dict(size=4 if not val else 5, color=color, opacity=op),
                customdata=cd[m], hovertemplate=tmpl(name)))
    fig.update_layout(template=TEMPLATE, height=520, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation='h', y=1.06),
                      xaxis=dict(title='Pattern dimension 1', range=XRANGE),
                      yaxis=dict(title='Pattern dimension 2', range=YRANGE))
    return fig


@lru_cache(maxsize=8)
def build_segment_profile(c):
    p = next(x for x in SUMMARY['cluster_profiles'] if x['cluster'] == c)
    pop = {'churn_rate': BASELINE, 'active_rate': CUST['IsActiveMember'].mean() * 100,
           'germany_pct': (CUST['Geography'] == 'Germany').mean() * 100,
           'avg_products': CUST['NumOfProducts'].mean(),
           'avg_balance': CUST['Balance'].mean(), 'avg_tenure': CUST['Tenure'].mean()}
    rows = [('Churn rate', f"{p['churn_rate']}%", f"{pop['churn_rate']:.1f}%"),
            ('Average balance', f"£{p['avg_balance']:,.0f}", f"£{pop['avg_balance']:,.0f}"),
            ('Products held', f"{p['avg_products']}", f"{pop['avg_products']:.2f}"),
            ('Active members', f"{p['active_rate']}%", f"{pop['active_rate']:.1f}%"),
            ('Based in Germany', f"{p['germany_pct']}%", f"{pop['germany_pct']:.1f}%"),
            ('Years with bank', f"{p['avg_tenure']}", f"{pop['avg_tenure']:.2f}")]
    return html.Div([
        html.Div([html.Span(f"C{p['cluster']}", className='persona-chip',
                            style={'background': PERSONA_COLORS[c]}),
                  html.H3(p['name'])], className='persona-head'),
        html.P(f"{p['n']:,} customers ({p['pct']}% of the book) · average age {p['avg_age']}"),
        html.Table([html.Thead(html.Tr([html.Th('Metric'), html.Th('This persona'),
                                        html.Th('Whole book')]))] +
                   [html.Tbody([html.Tr([html.Td(a), html.Td(b, className='strong'),
                                         html.Td(cv)]) for a, b, cv in rows])],
                   className='profile-table'),
    ], className='persona-card')


@lru_cache(maxsize=32)
def build_rule_network(min_lift):
    sel = RULES[RULES['lift'] >= min_lift].nlargest(20, 'lift')
    items = sorted({i for ant in sel['antecedent'] for i in ant.split('|')})
    n = max(len(items), 1)
    pos = {it: (1.05 * np.cos(2 * np.pi * k / n), 1.05 * np.sin(2 * np.pi * k / n))
           for k, it in enumerate(items)}
    fig = go.Figure()
    for _, r in sel.iterrows():
        ants = r['antecedent'].split('|')
        cx = np.mean([pos[a][0] for a in ants]) * 0.45
        cy = np.mean([pos[a][1] for a in ants]) * 0.45
        for a in ants:
            fig.add_trace(go.Scatter(x=[pos[a][0], cx], y=[pos[a][1], cy], mode='lines',
                                     line=dict(color='#c9d4de', width=1.2),
                                     hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=[cx, 0], y=[cy, 0], mode='lines',
                                 line=dict(color=CHURN_RED, width=1 + 2.2 * (r['lift'] - 1.5)),
                                 opacity=0.5, hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode='markers',
            marker=dict(size=7 + 26 * r['support'], symbol='diamond', color='#f4a261',
                        line=dict(color='#7a4d12', width=1)),
            hovertemplate=('IF ' + ' + '.join(nice(a) for a in ants) +
                           f"<br>THEN churned · churn rate {r['confidence']*100:.1f}%"
                           f"<br>support {r['support']*100:.2f}% · lift {r['lift']:.2f}×"
                           '<extra></extra>'),
            showlegend=False))
    fig.add_trace(go.Scatter(
        x=[pos[i][0] for i in items], y=[pos[i][1] for i in items],
        mode='markers+text', text=[nice(i) for i in items], textposition='top center',
        textfont=dict(size=11), marker=dict(size=13, color=RETAIN_BLUE),
        hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text', text=['CHURNED'],
                             textposition='middle center',
                             textfont=dict(size=11, color='white'),
                             marker=dict(size=58, color=CHURN_RED), hoverinfo='skip',
                             showlegend=False))
    fig.update_layout(template=TEMPLATE, height=520,
                      margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(visible=False, range=[-1.45, 1.45]),
                      yaxis=dict(visible=False, range=[-1.35, 1.35]))
    table = tuple({'ant': ' + '.join(nice(a) for a in r['antecedent'].split('|')),
                   'sup': f"{r['support']*100:.2f}%", 'conf': f"{r['confidence']*100:.1f}%",
                   'lift': f"{r['lift']:.2f}×"} for _, r in sel.iterrows())
    label = f'{len(sel)} rules at lift ≥ {min_lift:.1f}×'
    return fig, table, label


@lru_cache(maxsize=16)
def build_anomaly_views(methods):
    methods = list(methods)
    flagged = ANOM[methods].max(axis=1) == 1 if methods else pd.Series(False, index=ANOM.index)
    fig = go.Figure()
    bg = ANOM[~flagged]
    fig.add_trace(go.Scattergl(x=bg['Balance'], y=bg['IF_anomaly_score'], mode='markers',
                               name='Not flagged', marker=dict(size=3, color=GREY, opacity=0.35),
                               hoverinfo='skip'))
    for val, nm, color in [(0, 'Flagged · retained', RETAIN_BLUE), (1, 'Flagged · churned', CHURN_RED)]:
        m = flagged & (ANOM['Exited'] == val)
        d = ANOM[m]
        fig.add_trace(go.Scattergl(
            x=d['Balance'], y=d['IF_anomaly_score'], mode='markers', name=nm,
            marker=dict(size=5, color=color, opacity=0.75),
            customdata=d[['Geography', 'Age', 'Anomaly_Class']].values,
            hovertemplate='%{customdata[0]} · age %{customdata[1]}<br>Balance £%{x:,.0f}'
                          '<br>%{customdata[2]}<extra></extra>'))
    fig.update_layout(template=TEMPLATE, height=430, margin=dict(l=10, r=10, t=40, b=10),
                      title='Isolation-Forest anomaly score vs balance',
                      xaxis_title='Account balance (£)',
                      yaxis_title='Anomaly score (lower = stranger)',
                      legend=dict(orientation='h', y=1.12))

    comp = ANOM['Composite_Anomaly_Score'].value_counts().sort_index()
    churn_by = ANOM.groupby('Composite_Anomaly_Score')['Exited'].mean() * 100
    fig2 = go.Figure(go.Bar(x=comp.index, y=comp.values,
                            marker_color=['#1a9641', '#a6d96a', '#fdae61', '#d7191c', '#7b2c2c'],
                            text=[f'{churn_by[i]:.0f}% churn' for i in comp.index],
                            textposition='outside'))
    fig2.update_layout(template=TEMPLATE, height=430, margin=dict(l=10, r=10, t=40, b=10),
                       title='How many methods agree per customer',
                       xaxis_title='Methods flagging the customer', yaxis_title='Customers',
                       yaxis_range=[0, comp.max() * 1.18])

    fl = ANOM[ANOM['Composite_Anomaly_Score'] >= 1]
    cc = fl['Anomaly_Class'].value_counts().sort_values()
    fig3 = go.Figure(go.Bar(x=cc.values, y=[c.replace(' — ', '<br>') for c in cc.index],
                            orientation='h', marker_color='#6d597a',
                            text=cc.values, textposition='outside'))
    fig3.update_layout(template=TEMPLATE, height=430, margin=dict(l=10, r=10, t=40, b=10),
                       title=f'Classification of the {len(fl):,} flagged anomalies',
                       xaxis_range=[0, cc.max() * 1.2])
    return fig, fig2, fig3


@lru_cache(maxsize=8)
def build_distributions(geo):
    d = CUST if geo == 'All' else CUST[CUST['Geography'] == geo]
    gc = SUMMARY['geo_churn']
    colors = [CHURN_RED if (geo in (g, 'All')) else GREY for g in gc]
    f1 = go.Figure(go.Bar(x=list(gc.keys()), y=list(gc.values()), marker_color=colors,
                          text=[f'{v}%' for v in gc.values()], textposition='outside'))
    f1.add_hline(y=BASELINE, line_dash='dot', annotation_text=f'baseline {BASELINE:.1f}%')
    f1.update_layout(template=TEMPLATE, height=360, title='Churn rate by market',
                     margin=dict(l=10, r=10, t=40, b=10), yaxis_range=[0, 40])

    def hist_fig(col, title, bins):
        # numpy-binned bars instead of raw-point histograms keep the figure
        # payload tiny (≤50 bars instead of 10K points per trace)
        edges = np.histogram_bin_edges(CUST[col], bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        width = (edges[1] - edges[0]) * 0.92
        f = go.Figure()
        for val, nm, color in [(0, 'Retained', RETAIN_BLUE), (1, 'Churned', CHURN_RED)]:
            counts, _ = np.histogram(d[d['Exited'] == val][col], bins=edges)
            f.add_trace(go.Bar(x=centers, y=counts, width=width, name=nm,
                               marker_color=color, opacity=0.65))
        f.update_layout(template=TEMPLATE, height=360, barmode='overlay', title=title,
                        margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation='h', y=1.1))
        return f

    f2 = hist_fig('Age', f'Age distribution — {geo}', 40)
    f2.add_vrect(x0=46, x1=60, fillcolor='#f4a261', opacity=0.15, line_width=0,
                 annotation_text='46–60', annotation_position='top left')

    f3 = hist_fig('Balance', f'Balance distribution — {geo} (36% of the book holds £0)', 50)

    pr = d.groupby('NumOfProducts')['Exited'].agg(['mean', 'size'])
    f4 = go.Figure(go.Bar(x=pr.index, y=pr['mean'] * 100,
                          marker_color=['#e76f51', '#2a9d8f', '#d62828', '#7b2c2c'][:len(pr)],
                          text=[f"{v*100:.1f}%<br>(n={s:,})" for v, s in zip(pr['mean'], pr['size'])],
                          textposition='outside'))
    f4.add_hline(y=BASELINE, line_dash='dot', annotation_text='baseline')
    f4.update_layout(template=TEMPLATE, height=360,
                     title=f'Churn rate by products held — {geo}',
                     xaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4]),
                     margin=dict(l=10, r=10, t=40, b=10), yaxis_range=[0, 115])
    return f1, f2, f3, f4


# ══ Thin registered callbacks delegating to the cached builders ════════════════
@app.callback(Output('seg-map', 'figure'), Input('seg-color', 'value'))
def segment_map(mode):
    return build_segment_map(mode)


@app.callback(Output('seg-profile', 'children'), Input('seg-persona', 'value'))
def segment_profile(c):
    return build_segment_profile(int(c))


@app.callback(Output('rule-net', 'figure'), Output('rule-table', 'data'),
              Output('rule-count', 'children'), Input('rule-lift', 'value'))
def rule_network(min_lift):
    fig, table, label = build_rule_network(round(float(min_lift), 1))
    return fig, list(table), label


@app.callback(Output('anom-scatter', 'figure'), Output('anom-composite', 'figure'),
              Output('anom-class', 'figure'), Input('anom-methods', 'value'))
def anomaly_views(methods):
    return build_anomaly_views(tuple(sorted(methods or [])))


@app.callback(Output('dist-geo-churn', 'figure'), Output('dist-age', 'figure'),
              Output('dist-balance', 'figure'), Output('dist-products', 'figure'),
              Input('dist-geo', 'value'))
def distributions(geo):
    return build_distributions(geo)


# ── Pre-warm every cacheable view at startup (~2 s once) so the first user
#    interaction on each control is already a cache hit ────────────────────────
for _m in ['persona', 'churn', 'noise']:
    build_segment_map(_m)
for _c in [0, 1, 2]:
    build_segment_profile(_c)
for _l in [1.5, 2.0, 2.5, 3.0, 3.5, 3.8]:
    build_rule_network(_l)
for _g in ['All', 'France', 'Germany', 'Spain']:
    build_distributions(_g)
build_anomaly_views(('DBSCAN_flag', 'IF_flag'))
build_anomaly_views(('DBSCAN_flag', 'IF_flag', 'IQR_flag', 'ZScore_flag'))


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8050)
