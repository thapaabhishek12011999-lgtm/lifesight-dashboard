"""
Lifesight - Executive-grade UI (styled to match provided screenshots)
- Tabbed: Executive / CMO / CFO
- Soft purple header, rounded cards, compact KPI tiles
- Funnel shows counts + % of impressions; delta semantics: CPM/refund decrease = green
- Cohort heatmap removed from UI
- Exports removed
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# -----------------------
# Mock data generator (kept from your pipeline)
# -----------------------
@st.cache_data
def generate_mock_data(months_before=3, months_after=3, seed=42):
    np.random.seed(seed)
    today = pd.to_datetime(datetime.utcnow().date())
    start = today - pd.DateOffset(months=months_before)
    end = today + pd.DateOffset(months=months_after)
    dates = pd.date_range(start=start, end=end, freq="D")

    channels = {
        "Meta": {"ctr": 0.02, "cpc": 0.35, "cv_rate": 0.03, "aov": 45, "cogs_pct": 0.45, "return_rate": 0.03},
        "Google": {"ctr": 0.035, "cpc": 0.50, "cv_rate": 0.05, "aov": 60, "cogs_pct": 0.40, "return_rate": 0.025},
        "Amazon": {"ctr": 0.06, "cpc": 0.60, "cv_rate": 0.10, "aov": 55, "cogs_pct": 0.50, "return_rate": 0.04},
        "TikTok": {"ctr": 0.015, "cpc": 0.20, "cv_rate": 0.02, "aov": 35, "cogs_pct": 0.50, "return_rate": 0.05}
    }

    rows = []
    for d in dates:
        day_multiplier = 1 + 0.05 * np.sin((d.dayofyear % 30) / 30 * 2 * np.pi)
        for channel, params in channels.items():
            for camp_i in range(1, 4):
                campaign = f"{channel}_Camp_{camp_i}"
                for adset_i in range(1, 3):
                    ad_set = f"{campaign}_AS{adset_i}"
                    for creative_i in range(1, 3):
                        creative = f"{ad_set}_CR{creative_i}"

                        base_imp = {"Meta":120000, "Google":80000, "Amazon":40000, "TikTok":150000}[channel]
                        impressions = max(0, int(np.random.normal(base_imp*0.02, base_imp*0.004)))
                        impressions = int(impressions * day_multiplier * (1 + 0.01 * (camp_i-2)))

                        ctr = max(0.001, np.random.normal(params["ctr"], params["ctr"]*0.25))
                        clicks = np.random.binomial(impressions, ctr) if impressions>0 else 0

                        cv_rate = max(0.001, np.random.normal(params["cv_rate"], params["cv_rate"]*0.25))
                        conversions = np.random.binomial(clicks, cv_rate) if clicks>0 else 0

                        cpc = max(0.05, np.random.normal(params["cpc"], params["cpc"]*0.12))
                        spend = round(clicks * cpc, 2)

                        aov = max(5, np.random.normal(params["aov"], params["aov"]*0.12))
                        revenue = round(conversions * aov, 2)
                        orders = conversions

                        cogs_pct = max(0.2, np.random.normal(params["cogs_pct"], 0.05))
                        cogs = round(revenue * cogs_pct, 2)

                        return_rate = max(0.0, np.random.normal(params["return_rate"], params["return_rate"]*0.3))
                        returns = int(round(orders * return_rate))
                        returned_value = round(returns * aov * 0.9, 2)

                        if orders > 0:
                            new_pct = np.random.beta(2,5)
                            new_customers = int(round(orders * new_pct))
                            returning_customers = orders - new_customers
                        else:
                            new_customers = 0
                            returning_customers = 0

                        ctr_actual = round((clicks / impressions) if impressions>0 else 0, 4)
                        cvr_actual = round((conversions / clicks) if clicks>0 else 0, 4)
                        cac = round(spend / conversions, 2) if conversions>0 else np.nan
                        roas = round(revenue / spend, 2) if spend>0 else np.nan
                        profit = round(revenue - cogs - returned_value - spend, 2)

                        rows.append({
                            "date": d.date().isoformat(),
                            "channel": channel,
                            "campaign": campaign,
                            "ad_set": ad_set,
                            "creative": creative,
                            "impressions": impressions,
                            "clicks": clicks,
                            "spend": spend,
                            "conversions": conversions,
                            "orders": orders,
                            "revenue": revenue,
                            "cogs": cogs,
                            "returns": returns,
                            "returned_value": returned_value,
                            "new_customers": new_customers,
                            "returning_customers": returning_customers,
                            "ctr": ctr_actual,
                            "cvr": cvr_actual,
                            "aov": round(aov, 2),
                            "cac": cac,
                            "roas": roas,
                            "profit": profit,
                            "funnel_step": "total"
                        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df

# -----------------------
# THEME & CSS to mimic screenshots
# -----------------------
PRIMARY = "#6B21A8"
ACCENT = "#7c3aed"
MUTED = "#6b7280"
CARD_BG = "#FBF9FF"   # very light purple card background
WHITE = "#FFFFFF"
TEXT = "#0f172a"

def inject_css():
    st.markdown(
        f"""
        <style>
        /* page & container sizes */
        .stApp {{ background: #fbfbfd; color: {TEXT}; font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial; }}
        .block-container{{max-width:1400px; padding-top:10px; padding-left:20px; padding-right:20px;}}

        /* top header + pills */
        .topband {{
            background: linear-gradient(90deg, {PRIMARY}20 0%, {ACCENT}12 100%);
            border: 1px solid #f0ecf9;
            padding: 16px 18px;
            border-radius: 10px;
            margin-bottom: 14px;
        }}
        .title-main {{ font-size:20px; font-weight:700; color: #2b2540; }}
        .title-sub {{ font-size:13px; color:{MUTED}; margin-top:3px; }}

        /* tab pills styling tweak (affects tab labels visually) */
        .css-1d391kg .stTabs [role="tab"] {{
            border-radius: 999px !important;
            padding: 6px 12px !important;
            font-weight:600;
        }}

        /* KPI card */
        .kpi-card {{
            background: {CARD_BG};
            border-radius: 12px;
            padding: 14px;
            box-shadow: 0 6px 18px rgba(115,85,200,0.04);
            border: 1px solid #f0ecf9;
        }}
        .kpi-label {{ color:{MUTED}; font-size:12px; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.6px; }}
        .kpi-value {{ font-size:22px; font-weight:800; color:#251f3a; }}
        .kpi-target {{ color:{MUTED}; font-size:12px; margin-top:6px; }}

        /* small delta badges */
        .delta-up {{ color:#059669; font-weight:700; }}
        .delta-down {{ color:#dc2626; font-weight:700; }}

        /* section header card */
        .section-header {{
            background: linear-gradient(90deg, #faf7ff, #fff);
            border-radius:10px;
            padding:12px 14px;
            border:1px solid #f0ecf9;
            margin-bottom:12px;
        }}
        .section-title {{ font-size:15px; font-weight:700; color:#281b3f; }}

        /* compact metrics row spacing */
        .metrics-row > div {{ padding-right:8px; }}

        /* smaller helpers */
        .muted {{ color:{MUTED}; font-size:13px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# -----------------------
# KPI helpers & delta logic
# -----------------------
def compute_exec_kpis(df):
    total_revenue = df["revenue"].sum()
    total_spend = df["spend"].sum()
    mer = round(total_revenue / total_spend, 2) if total_spend>0 else np.nan
    total_cogs = df["cogs"].sum()
    gross_margin = round((total_revenue - total_cogs) / total_revenue, 4) if total_revenue>0 else np.nan
    total_conversions = df["conversions"].sum()
    total_new_cust = df["new_customers"].sum()
    cac = round(df["spend"].sum() / total_conversions, 2) if total_conversions>0 else np.nan
    ltv = round(total_revenue / max(1, total_new_cust), 2) if total_new_cust>0 else np.nan
    ltv_cac = round(ltv / cac, 2) if cac and cac>0 else np.nan
    profit = df["profit"].sum()
    return {
        "total_revenue": total_revenue,
        "total_spend": total_spend,
        "mer": mer,
        "gross_margin": gross_margin,
        "cac": cac,
        "ltv": ltv,
        "ltv_cac": ltv_cac,
        "profit": profit
    }

def compute_pop(cur, prev):
    if prev is None or prev == 0 or np.isnan(prev):
        return None
    try:
        return (cur - prev) / abs(prev)
    except:
        return None

def delta_html_small(cur, prev, invert=False):
    pop = compute_pop(cur, prev)
    if pop is None:
        return "<span class='muted'>No prev</span>"
    sym = "▲" if pop > 0 else "▼"
    if invert:
        color_cls = "delta-up" if pop < 0 else "delta-down"
    else:
        color_cls = "delta-up" if pop > 0 else "delta-down"
    return f"<span class='{color_cls}' style='font-size:13px'>{sym} {abs(pop)*100:.1f}%</span>"

# -----------------------
# Plot building (kept but adjusted styling)
# -----------------------
def plot_spend_revenue_trend(df):
    ts = df.groupby("date").agg({"revenue":"sum","spend":"sum"}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ts["date"], y=ts["spend"], name="Spend", marker_color=ACCENT, yaxis="y2", opacity=0.14))
    fig.add_trace(go.Scatter(x=ts["date"], y=ts["revenue"], name="Revenue", mode="lines", line=dict(color="#10b981", width=3)))
    fig.update_layout(
        title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Revenue", gridcolor="rgba(0,0,0,0.06)"),
        yaxis2=dict(title="Spend", overlaying="y", side="right"),
        margin=dict(l=30, r=30, t=8, b=30),
        template="plotly_white",
        height=360
    )
    return fig

def plot_roas_by_channel(df):
    agg = df.groupby("channel").agg({"revenue":"sum","spend":"sum"}).reset_index()
    agg["roas"] = agg["revenue"] / agg["spend"]
    color_map = {"Meta":"#2563EB","Google":"#FBBF24","Amazon":"#FB923C","TikTok":"#F43F5E"}
    fig = px.bar(agg.sort_values("roas", ascending=False), x="roas", y="channel", orientation="h",
                 text=agg["roas"].round(2), color="channel", color_discrete_map=color_map)
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=100,t=6,b=6))
    fig.update_xaxes(title="ROAS (Revenue / Spend)")
    return fig

def plot_contribution_waterfall(df):
    rev = df["revenue"].sum()
    cogs = df["cogs"].sum()
    returns = df["returned_value"].sum()
    spend = df["spend"].sum()
    profit = rev - cogs - returns - spend
    measures = ["relative","relative","relative","relative","total"]
    fig = go.Figure(go.Waterfall(
        name="Contribution",
        orientation="v",
        measure=measures,
        x=["Revenue","COGS","Returns","Marketing Spend","Net Profit"],
        text=[f"{rev:,.0f}", f"-{cogs:,.0f}", f"-{returns:,.0f}", f"-{spend:,.0f}", f"{profit:,.0f}"],
        y=[rev, -cogs, -returns, -spend, profit],
        decreasing=dict(marker=dict(color="#f97373")),
        increasing=dict(marker=dict(color="#10b981")),
        totals=dict(marker=dict(color=PRIMARY))
    ))
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=30,r=30,t=6,b=6))
    return fig

def plot_cac_trend(df):
    ts = df.groupby("date").apply(lambda x: pd.Series({"cac": x["spend"].sum() / x["conversions"].sum() if x["conversions"].sum()>0 else np.nan})).reset_index()
    fig = px.line(ts, x="date", y="cac", title="", markers=False)
    fig.update_layout(template="plotly_white", height=260, margin=dict(l=30,r=20,t=8,b=20))
    fig.update_yaxes(title="CAC")
    return fig

def plot_funnel_bars(df):
    impressions = df["impressions"].sum()
    clicks = df["clicks"].sum()
    product_views = int(df["orders"].sum() * 3.5)
    add_to_cart = int(product_views * 0.25)
    purchases = int(df["orders"].sum())

    steps = pd.DataFrame({
        "step": ["Impressions", "Clicks", "Product Views", "Add to Cart", "Purchases"],
        "value": [impressions, clicks, product_views, add_to_cart, purchases]
    })

    steps["pct_of_impr"] = (steps["value"] / impressions * 100).fillna(0) if impressions>0 else 0.0
    pct_prev = []
    prev_val = None
    for v in steps["value"]:
        if prev_val is None:
            pct_prev.append(100.0)
        else:
            pct_prev.append((v / prev_val * 100) if prev_val>0 else 0.0)
        prev_val = v
    steps["pct_prev"] = pct_prev
    # show label as "count (percent of impressions)" similar to screenshots
    steps["label"] = steps.apply(lambda r: f"{int(r['value']):,}   {r['pct_of_impr']:.1f}%", axis=1)

    # order top->bottom impressions->purchases
    order_map = {"Impressions":5,"Clicks":4,"Product Views":3,"Add to Cart":2,"Purchases":1}
    steps["order"] = steps["step"].map(order_map)
    plot_df = steps.sort_values("order", ascending=False)

    customdata = np.column_stack((plot_df["value"], plot_df["pct_of_impr"], plot_df["pct_prev"]))
    fig = px.bar(plot_df, x="value", y="step", orientation="h", text="label", color_discrete_sequence=["#a78bfa"])
    fig.update_traces(textposition="inside", insidetextanchor="end",
                      hovertemplate=("<b>%{y}</b><br>Count: %{customdata[0]:,}<br>% of impressions: %{customdata[1]:.2f}%<br>% vs prev step: %{customdata[2]:.2f}%<extra></extra>"),
                      customdata=customdata)
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=120,t=6,b=6), xaxis=dict(title="Count"))
    return fig

# -----------------------
# Streamlit layout - Header + Tabs
# -----------------------
st.set_page_config(page_title="Marketing Performance Dashboard", layout="wide", initial_sidebar_state="collapsed")
inject_css()

# Top header + small controls on the right
header_col, controls_col = st.columns([3,1])
with header_col:
    st.markdown(f"<div class='title-main'>Marketing Performance Dashboard</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='title-sub'>E-Commerce Brand Analytics • Multi-Platform Advertising Performance</div>", unsafe_allow_html=True)
with controls_col:
    # compact controls to mimic screenshots
    with st.container():
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        _ = st.selectbox("", ["Last 1 Month","Last 3 Months","Last 6 Months"], index=2, key="__range", help="Select date range")
        if st.button("Refresh"):
            st.experimental_rerun()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# Tabs (Executive / CMO / CFO) as pill tabs at top
tab_exec, tab_cmo, tab_cfo = st.tabs(["Executive Summary", "CMO View", "CFO View"])

# Load data
df = generate_mock_data(months_before=3, months_after=3)

# Sidebar-like filters but kept minimal (top-of-page style)
filters_col1, filters_col2, filters_col3 = st.columns([1,1,1])
with filters_col1:
    channel = st.selectbox("Channel", ["All"] + sorted(df["channel"].unique().tolist()), index=0, key="filter_channel")
with filters_col2:
    campaign = st.selectbox("Campaign", ["All"] + sorted(df["campaign"].unique().tolist()), index=0, key="filter_campaign")
with filters_col3:
    start_date = st.date_input("Start date", value=df["date"].min().date())
    end_date = st.date_input("End date", value=df["date"].max().date())

# Subset
subset = df.copy()
if channel != "All":
    subset = subset[subset["channel"]==channel]
if campaign != "All":
    subset = subset[subset["campaign"]==campaign]
subset = subset[(subset["date"]>=pd.to_datetime(start_date)) & (subset["date"]<=pd.to_datetime(end_date))]

if subset.empty:
    st.warning("No data for the selected filters/date range. Adjust filters.")
    st.stop()

# prev period computation (same length)
curr_start, curr_end = pd.to_datetime(start_date), pd.to_datetime(end_date)
period_days = (curr_end - curr_start).days + 1
prev_end = curr_start - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=period_days-1)
prev_subset = df[(df["date"]>=prev_start) & (df["date"]<=prev_end)]

cur_kpis = compute_exec_kpis(subset)
prev_kpis = compute_exec_kpis(prev_subset)

# -----------------------
# EXECUTIVE TAB content
# -----------------------
with tab_exec:
    st.markdown("<div class='section-header'><div class='section-title'>Executive Summary</div><div class='muted'>High-level KPIs linking marketing investment to company financial health</div></div>", unsafe_allow_html=True)
    # KPI row - similar to screenshots (4 cards across)
    cols = st.columns([1.2,1,1,1], gap="large")
    # Total Net Revenue (formatted friendly: K / M)
    def fmt_money(v):
        if v >= 1e6:
            return f"${v/1e6:.2f}M"
        if v >= 1e3:
            return f"${v/1e3:.0f}K"
        return f"${v:.0f}"

    with cols[0]:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>TOTAL NET REVENUE</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{fmt_money(cur_kpis['total_revenue'])}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-target'>{delta_html_small(cur_kpis['total_revenue'], prev_kpis['total_revenue'])} &nbsp; <span class='muted'>Target: $4.5M</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[1]:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>MARKETING EFFICIENCY RATIO</div>", unsafe_allow_html=True)
        mer_val = cur_kpis["mer"]
        st.markdown(f"<div class='kpi-value'>{mer_val:.2f}x</div>" if not np.isnan(mer_val) else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-target'>{delta_html_small(mer_val, prev_kpis['mer'])} &nbsp; <span class='muted'>Target: 3.5x</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[2]:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>LTV : CAC RATIO</div>", unsafe_allow_html=True)
        ltv_cac = cur_kpis.get("ltv_cac", np.nan)
        st.markdown(f"<div class='kpi-value'>{ltv_cac:.2f}x</div>" if not np.isnan(ltv_cac) else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-target'>{delta_html_small(ltv_cac, prev_kpis.get('ltv_cac', np.nan))} &nbsp; <span class='muted'>Target: 4x</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[3]:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>GROSS PROFIT MARGIN</div>", unsafe_allow_html=True)
        gpm = cur_kpis["gross_margin"]
        st.markdown(f"<div class='kpi-value'>{gpm*100:.0f}%</div>" if not np.isnan(gpm) else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-target'>{delta_html_small(gpm, prev_kpis['gross_margin'])} &nbsp; <span class='muted'>Target: 60%</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Big trend chart
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Revenue & Marketing Efficiency Trend</div>", unsafe_allow_html=True)
    fig = plot_spend_revenue_trend(subset)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# CMO VIEW
# -----------------------
with tab_cmo:
    st.markdown("<div class='section-header'><div class='section-title'>CMO View: Marketing Effectiveness</div><div class='muted'>Campaign diagnostics, channel performance & acquisition insights</div></div>", unsafe_allow_html=True)

    # KPI row for marketing
    c_cols = st.columns([1,1,1,1], gap="large")
    prev_impr = prev_subset["impressions"].sum()
    prev_clicks = prev_subset["clicks"].sum()
    prev_spend = prev_subset["spend"].sum()
    prev_conv = prev_subset["conversions"].sum()

    with c_cols[0]:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>TOTAL IMPRESSIONS</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{int(subset['impressions'].sum()):,}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-target'>{delta_html_small(subset['impressions'].sum(), prev_impr)} &nbsp; <span class='muted'> </span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_cols[1]:
        ctr_val = subset['clicks'].sum() / subset['impressions'].sum() if subset['impressions'].sum()>0 else 0
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>CLICK THROUGH RATE</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{ctr_val*100:.2f}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-target'>{delta_html_small(ctr_val, (prev_clicks/prev_impr) if prev_impr>0 else None)} </div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_cols[2]:
        cpm_val = subset['spend'].sum() / (subset['impressions'].sum()/1000) if subset['impressions'].sum()>0 else np.nan
        prev_cpm = prev_spend / (prev_impr/1000) if prev_impr>0 else None
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>AVG. CPM</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{'${:.2f}'.format(cpm_val) if not np.isnan(cpm_val) else 'N/A'}</div>", unsafe_allow_html=True)
        # inverted semantics (decrease = green)
        st.markdown(f"<div class='kpi-target'>{delta_html_small(cpm_val, prev_cpm, invert=True)}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_cols[3]:
        freq = ((subset['impressions'].sum() / max(1, subset['orders'].sum())) if subset['orders'].sum()>0 else np.nan)
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>AVG. FREQUENCY</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{freq:.1f}x</div>" if not np.isnan(freq) else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-target'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ROAS + Funnel side by side
    r1, r2 = st.columns([1.4,1], gap="large")
    with r1:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>ROAS by Channel</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_roas_by_channel(subset), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with r2:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Marketing Funnel</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_funnel_bars(subset), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    # campaign diagnostics expander
    with st.expander("Campaign & Creative Performance (Top 100)", expanded=False):
        diag = subset.groupby(["channel","campaign","ad_set","creative"]).agg({
            "spend":"sum","impressions":"sum","clicks":"sum","conversions":"sum","revenue":"sum"
        }).reset_index()
        diag["ctr"] = (diag["clicks"] / diag["impressions"]).round(4)
        diag["cvr"] = (diag["conversions"] / diag["clicks"]).round(4)
        diag["cpa"] = (diag["spend"] / diag["conversions"]).round(2).replace([np.inf, -np.inf], pd.NA)
        st.dataframe(diag.sort_values("revenue", ascending=False).head(100), use_container_width=True)

# -----------------------
# CFO VIEW
# -----------------------
with tab_cfo:
    st.markdown("<div class='section-header'><div class='section-title'>CFO View: Financial Efficiency & Profitability</div><div class='muted'>Cost control, margin analysis and ROI tracking for fiscal planning</div></div>", unsafe_allow_html=True)

    fcols = st.columns([1,1,1,1], gap="large")

    with fcols[0]:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        roi_val = (subset["revenue"].sum() - subset["spend"].sum()) / subset["spend"].sum() if subset["spend"].sum()>0 else None
        st.markdown("<div class='kpi-label'>MARKETING ROI</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{(roi_val*100):.0f}%</div>" if roi_val is not None else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-target'>{delta_html_small(roi_val, (prev_kpis['profit']-prev_kpis['total_spend'])/prev_kpis['total_spend'] if prev_kpis['total_spend']>0 else None)} </div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with fcols[1]:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        gm_val = (subset["revenue"].sum() - subset["cogs"].sum()) / subset["revenue"].sum() if subset["revenue"].sum()>0 else None
        st.markdown("<div class='kpi-label'>GROSS MARGIN RATE</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{(gm_val*100):.1f}%</div>" if gm_val is not None else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-target'>{delta_html_small(gm_val, (prev_kpis['total_revenue']-prev_kpis['total_spend'])/prev_kpis['total_revenue'] if prev_kpis['total_revenue']>0 else None)} </div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with fcols[2]:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        aov_val = subset['aov'].mean() if not subset.empty else None
        st.markdown("<div class='kpi-label'>AVERAGE ORDER VALUE</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>${aov_val:.0f}</div>" if aov_val is not None else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-target'>{delta_html_small(aov_val, prev_subset['aov'].mean() if not prev_subset.empty else None)} </div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with fcols[3]:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        refund_rate = subset["returns"].sum() / subset["orders"].sum() if subset["orders"].sum()>0 else np.nan
        prev_ref = prev_subset["returns"].sum() / prev_subset["orders"].sum() if prev_subset["orders"].sum()>0 else None
        st.markdown("<div class='kpi-label'>REFUND & RETURN RATE</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{(refund_rate*100):.1f}%</div>" if not np.isnan(refund_rate) else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-target'>{delta_html_small(refund_rate, prev_ref, invert=True)}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Contribution & CAC side-by-side
    left, right = st.columns([1.4,1], gap="large")
    with left:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Contribution Margin Breakdown</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_contribution_waterfall(subset), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Customer Acquisition Cost (CAC) Trend</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_cac_trend(subset), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # small financial metrics (bottom)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Net Profit Margin</div>", unsafe_allow_html=True)
        net_profit = cur_kpis['profit']
        total_rev = cur_kpis['total_revenue']
        npm = (net_profit / total_rev) if total_rev>0 else np.nan
        st.markdown(f"<div class='kpi-value'>{npm*100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b2:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Total Costs</div>", unsafe_allow_html=True)
        total_costs = subset['cogs'].sum() + subset['spend'].sum() + subset['returned_value'].sum()
        st.markdown(f"<div class='kpi-value'>${total_costs/1e3:.0f}K</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b3:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Average Order Value</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>${subset['aov'].mean():.0f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# footer caption
st.caption("Dashboard styled to match the provided screenshots. Tell me any color/spacing tweaks and I'll update it.")
