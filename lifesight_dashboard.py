"""
Lifesight - Clean & Professional Marketing Performance Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# -----------------------
# Mock data generator (same as prior)
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
# Theme & CSS for clean/professional look
# -----------------------
PRIMARY = "#6B21A8"
ACCENT_GREEN = "#14B8A6"
LIGHT_BG = "#FFFFFF"
CARD_BG = "#FAFAFB"
TEXT = "#0f172a"
MUTED = "#6b7280"

def inject_css():
    st.markdown(
        f"""
        <style>
        /* Page layout */
        .stApp {{ background: {LIGHT_BG}; color: {TEXT}; font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }}
        .block-container{{ padding-top: 0.8rem; padding-left: 1.2rem; padding-right:1.2rem; max-width: 1400px; }}

        /* Top band */
        .topband {{
            background: linear-gradient(90deg, {PRIMARY} 0%, #7c3aed 100%);
            color: white;
            padding: 10px 18px;
            border-radius: 8px;
            margin-bottom: 14px;
        }}
        /* KPI card */
        .kpi-card {{
            background: {CARD_BG};
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 6px 18px rgba(15,23,42,0.06);
        }}
        .kpi-label {{ color: {MUTED}; font-size:13px; margin-bottom:4px; }}
        .kpi-value {{ font-size:20px; font-weight:700; color:{TEXT}; }}
        .kpi-delta {{ font-size:12px; font-weight:600; margin-top:6px; }}

        /* Smaller helpers */
        .muted {{ color: {MUTED}; font-size:13px; }}
        .section-title {{ font-size:16px; font-weight:700; margin-bottom:6px; }}
        .chart-card {{ background: transparent; padding: 6px 0; }}
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

def delta_html(cur, prev, invert=False):
    """Return small inline HTML showing percent change. If invert=True, decrease=green."""
    pop = compute_pop(cur, prev)
    if pop is None:
        return "<div class='kpi-delta muted'>No prev</div>"
    sym = "▲" if pop > 0 else "▼"
    if invert:
        color = "#059669" if pop < 0 else "#dc2626"
    else:
        color = "#059669" if pop > 0 else "#dc2626"
    return f"<div class='kpi-delta' style='color:{color}'>{sym} {abs(pop)*100:.1f}%</div>"

# -----------------------
# Plot: Spend & Revenue trend, ROAS, Contribution, CAC (unchanged)
# -----------------------
def plot_spend_revenue_trend(df):
    ts = df.groupby("date").agg({"revenue":"sum","spend":"sum"}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ts["date"], y=ts["spend"], name="Spend", marker_color="#8b5cf6", yaxis="y2", opacity=0.6))
    fig.add_trace(go.Scatter(x=ts["date"], y=ts["revenue"], name="Revenue", mode="lines", line=dict(color=ACCENT_GREEN, width=3)))
    fig.update_layout(
        title=dict(text="Revenue & Marketing Spend Trend", x=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Revenue"),
        yaxis2=dict(title="Spend", overlaying="y", side="right"),
        margin=dict(l=30, r=30, t=50, b=30),
        template="plotly_white",
        height=360
    )
    return fig

def plot_roas_by_channel(df):
    agg = df.groupby("channel").agg({"revenue":"sum","spend":"sum"}).reset_index()
    agg["roas"] = agg["revenue"] / agg["spend"]
    color_map = {"Meta":"#1D4ED8","Google":"#FACC15","Amazon":"#F59E0B","TikTok":"#F43F5E"}
    fig = px.bar(agg.sort_values("roas", ascending=False), x="roas", y="channel", orientation="h",
                 text=agg["roas"].round(2), color="channel", color_discrete_map=color_map)
    fig.update_layout(title_text="ROAS by Channel", template="plotly_white", height=300, margin=dict(l=120,t=40,b=20))
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
        decreasing=dict(marker=dict(color="#ef4444")),
        increasing=dict(marker=dict(color="#10b981")),
        totals=dict(marker=dict(color="#6B21A8"))
    ))
    fig.update_layout(title_text="Contribution Margin Breakdown", template="plotly_white", height=340, margin=dict(l=30,r=30,t=50,b=20))
    return fig

def plot_cac_trend(df):
    ts = df.groupby("date").apply(lambda x: pd.Series({"cac": x["spend"].sum() / x["conversions"].sum() if x["conversions"].sum()>0 else np.nan})).reset_index()
    fig = px.line(ts, x="date", y="cac", title="Customer Acquisition Cost (CAC) Trend")
    fig.update_layout(template="plotly_white", height=260, margin=dict(l=30,r=20,t=40,b=20))
    fig.update_yaxes(title="CAC")
    return fig

# -----------------------
# Funnel: now includes percent of impressions (and % of previous step in hover)
# -----------------------
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

    steps["label"] = steps.apply(lambda r: f"{int(r['value']):,}\\n({r['pct_of_impr']:.2f}%)", axis=1)

    # sort so Impressions is top row in the horizontal chart
    steps_plot = steps.copy()
    steps_plot["order"] = [5,4,3,2,1]  # impressions top to purchases bottom
    steps_plot = steps_plot.sort_values("order", ascending=False)

    customdata = np.column_stack((steps_plot["value"], steps_plot["pct_of_impr"], steps_plot["pct_prev"]))

    fig = px.bar(steps_plot, x="value", y="step", orientation="h", text="label")
    fig.update_traces(textposition="outside", marker_color="#93c5fd",
                      hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Count: %{customdata[0]:,}<br>"
                        "% of impressions: %{customdata[1]:.2f}%<br>"
                        "% vs previous step: %{customdata[2]:.2f}%<extra></extra>"
                      ),
                      customdata=customdata
                      )
    fig.update_layout(title_text="Marketing Funnel: Impression → Purchase", template="plotly_white", height=320, margin=dict(l=130,t=40,b=20))
    fig.update_xaxes(title="Count")
    return fig

# -----------------------
# Streamlit layout (cleaned)
# -----------------------
st.set_page_config(page_title="Lifesight - Clean Dashboard", layout="wide", initial_sidebar_state="expanded")
inject_css()

# top brand / header
st.markdown(f"<div class='topband'><strong style='font-size:18px'>Lifesight</strong> — Marketing Performance</div>", unsafe_allow_html=True)

# Load data
df = generate_mock_data(months_before=3, months_after=3)

# Sidebar filters (clean & compact)
st.sidebar.header("Filters")
channel = st.sidebar.selectbox("Channel", ["All"] + sorted(df["channel"].unique().tolist()), index=0)
campaign = st.sidebar.selectbox("Campaign", ["All"] + sorted(df["campaign"].unique().tolist()), index=0)
creative = st.sidebar.selectbox("Creative", ["All"] + sorted(df["creative"].unique().tolist()), index=0)
start_date = st.sidebar.date_input("Start date", value=df["date"].min().date())
end_date = st.sidebar.date_input("End date", value=df["date"].max().date())
refresh = st.sidebar.button("Refresh")

# Subset
subset = df.copy()
if channel != "All":
    subset = subset[subset["channel"]==channel]
if campaign != "All":
    subset = subset[subset["campaign"]==campaign]
if creative != "All":
    subset = subset[subset["creative"]==creative]
subset = subset[(subset["date"]>=pd.to_datetime(start_date)) & (subset["date"]<=pd.to_datetime(end_date))]

if subset.empty:
    st.warning("No data for the selected filters/date range. Adjust filters.")
    st.stop()

# compute prev period for deltas
curr_start, curr_end = pd.to_datetime(start_date), pd.to_datetime(end_date)
period_days = (curr_end - curr_start).days + 1
prev_end = curr_start - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=period_days-1)
prev_subset = df[(df["date"]>=prev_start) & (df["date"]<=prev_end)]

cur_kpis = compute_exec_kpis(subset)
prev_kpis = compute_exec_kpis(prev_subset)

# KPI strip (compact cards)
kpi_cols = st.columns([1.6,1,1,1,1], gap="large")
with kpi_cols[0]:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-label'>Total Net Revenue</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-value'>₹{cur_kpis['total_revenue']:,.0f}</div>", unsafe_allow_html=True)
    st.markdown(delta_html(cur_kpis['total_revenue'], prev_kpis['total_revenue']), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    gpm = cur_kpis["gross_margin"]
    pgm = prev_kpis["gross_margin"]
    st.markdown("<div class='kpi-label'>Gross Margin</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-value'>{(gpm*100):.1f}%</div>" if not np.isnan(gpm) else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
    st.markdown(delta_html(gpm, pgm), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with kpi_cols[2]:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    mer = cur_kpis["mer"]
    pmer = prev_kpis["mer"]
    st.markdown("<div class='kpi-label'>MER (Revenue / Spend)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-value'>{mer:.2f}</div>" if not np.isnan(mer) else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
    st.markdown(delta_html(mer, pmer), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with kpi_cols[3]:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    ltvcac = cur_kpis["ltv_cac"]
    pltv = prev_kpis["ltv_cac"]
    st.markdown("<div class='kpi-label'>LTV : CAC</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-value'>{ltvcac:.2f}</div>" if not np.isnan(ltvcac) else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
    st.markdown(delta_html(ltvcac, pltv), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with kpi_cols[4]:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    profit = cur_kpis["profit"]
    pprofit = prev_kpis["profit"]
    st.markdown("<div class='kpi-label'>Profit</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-value'>₹{profit:,.0f}</div>", unsafe_allow_html=True)
    st.markdown(delta_html(profit, pprofit), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Layout: Two-column main area: left charts (overview), right diagnostics / deeper charts
left_col, right_col = st.columns([2,1], gap="large")

with left_col:
    st.markdown("<div class='section-title'>Revenue & Efficiency</div>", unsafe_allow_html=True)
    fig_spend_rev = plot_spend_revenue_trend(subset)
    st.plotly_chart(fig_spend_rev, use_container_width=True)

    # Funnel + ROAS side-by-side
    f1, f2 = st.columns([1.2,1], gap="large")
    with f1:
        st.markdown("<div class='section-title'>Full Marketing Funnel</div>", unsafe_allow_html=True)
        fig_funnel = plot_funnel_bars(subset)
        st.plotly_chart(fig_funnel, use_container_width=True)
    with f2:
        st.markdown("<div class='section-title'>ROAS by Channel</div>", unsafe_allow_html=True)
        fig_roas = plot_roas_by_channel(subset)
        st.plotly_chart(fig_roas, use_container_width=True)

    st.markdown("<div class='section-title'>Contribution & CAC</div>", unsafe_allow_html=True)
    fig_contrib = plot_contribution_waterfall(subset)
    st.plotly_chart(fig_contrib, use_container_width=True)
    fig_cac = plot_cac_trend(subset)
    st.plotly_chart(fig_cac, use_container_width=True)

with right_col:
    st.markdown("<div class='section-title'>Diagnostics & Campaign Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Top performing campaigns and creatives (click to expand)</div>", unsafe_allow_html=True)

    # Diagnostics table in expander (keeps page tidy)
    with st.expander("Campaign & Creative Diagnostics (Top 100 rows)", expanded=False):
        diag = subset.groupby(["channel","campaign","ad_set","creative"]).agg({
            "spend":"sum","impressions":"sum","clicks":"sum","conversions":"sum","revenue":"sum"
        }).reset_index()
        diag["ctr"] = (diag["clicks"] / diag["impressions"]).round(4)
        diag["cvr"] = (diag["conversions"] / diag["clicks"]).round(4)
        diag["cpa"] = (diag["spend"] / diag["conversions"]).round(2).replace([np.inf, -np.inf], pd.NA)
        st.dataframe(diag.sort_values("revenue", ascending=False).head(100), use_container_width=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Key CMO KPIs compact
    st.markdown("<div class='section-title'>Marketing KPIs</div>", unsafe_allow_html=True)
    mk_cols = st.columns(2, gap="small")
    prev_impr = prev_subset["impressions"].sum()
    prev_clicks = prev_subset["clicks"].sum()
    prev_spend = prev_subset["spend"].sum()
    prev_conv = prev_subset["conversions"].sum()

    with mk_cols[0]:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        total_imp = subset['impressions'].sum()
        st.markdown("<div class='kpi-label'>Impressions</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{total_imp:,}</div>", unsafe_allow_html=True)
        st.markdown(delta_html(total_imp, prev_impr), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Avg CPM (inverted semantics: decrease = green)
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        cpm_val = subset['spend'].sum() / (subset['impressions'].sum()/1000) if subset['impressions'].sum() > 0 else np.nan
        prev_cpm = prev_spend / (prev_impr/1000) if prev_impr > 0 else None
        st.markdown("<div class='kpi-label'>Avg. CPM</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{'₹{:.2f}'.format(cpm_val) if not np.isnan(cpm_val) else 'N/A'}</div>", unsafe_allow_html=True)
        st.markdown(delta_html(cpm_val, prev_cpm, invert=True), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with mk_cols[1]:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        ctr_val = subset['clicks'].sum() / subset['impressions'].sum() if subset['impressions'].sum() > 0 else 0
        prev_ctr = prev_clicks / prev_impr if prev_impr > 0 else None
        st.markdown("<div class='kpi-label'>Click-Through Rate</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{ctr_val*100:.2f}%</div>", unsafe_allow_html=True)
        st.markdown(delta_html(ctr_val, prev_ctr), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        cvr_val = subset['conversions'].sum() / subset['clicks'].sum() if subset['clicks'].sum() > 0 else 0
        prev_cvr = prev_conv / prev_clicks if prev_clicks > 0 else None
        st.markdown("<div class='kpi-label'>Conversion Rate</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{cvr_val*100:.2f}%</div>", unsafe_allow_html=True)
        st.markdown(delta_html(cvr_val, prev_cvr), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Financial KPIs (compact)
    st.markdown("<div class='section-title'>Financial KPIs</div>", unsafe_allow_html=True)
    fcols = st.columns(2)
    with fcols[0]:
        st.metric("Average Order Value (AOV)", f"₹{subset['aov'].mean():.2f}")
    with fcols[1]:
        refund_rate = subset["returns"].sum() / subset["orders"].sum() if subset["orders"].sum() > 0 else np.nan
        st.metric("Refund / Return Rate", f"{(refund_rate*100):.2f}%" if not np.isnan(refund_rate) else "N/A")

st.caption("Dashboard layout: streamlined for clarity — filters in sidebar, KPIs upfront, diagnostics tucked away.")
