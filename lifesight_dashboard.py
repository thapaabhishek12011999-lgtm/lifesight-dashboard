# lifesight_dashboard.py
"""
Lifesight - Marketing Performance Streamlit Dashboard (mock data)
Generates mock data (3 months before -> 3 months after today) and renders
an Executive Summary + CMO + CFO views using Plotly visuals.

Structure and metrics follow the assignment brief. See uploaded doc for rationale. :contentReference[oaicite:1]{index=1}
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# -----------------------
# Utility: generate mock data
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
        # day-level seasonality & noise
        day_multiplier = 1 + 0.05 * np.sin((d.dayofyear % 30) / 30 * 2 * np.pi)
        for channel, params in channels.items():
            for camp_i in range(1, 4):                # 3 campaigns
                campaign = f"{channel}_Camp_{camp_i}"
                for adset_i in range(1, 3):           # 2 adsets
                    ad_set = f"{campaign}_AS{adset_i}"
                    for creative_i in range(1, 3):    # 2 creatives
                        creative = f"{ad_set}_CR{creative_i}"

                        base_imp = {"Meta":120000, "Google":80000, "Amazon":40000, "TikTok":150000}[channel]
                        impressions = max(0, int(np.random.normal(base_imp*0.02, base_imp*0.004)) )
                        impressions = int(impressions * day_multiplier * (1 + 0.01 * (camp_i-2)))

                        ctr = max(0.001, np.random.normal(params["ctr"], params["ctr"]*0.25))
                        clicks = np.random.binomial(impressions, ctr) if impressions>0 else 0

                        cv_rate = max(0.001, np.random.normal(params["cv_rate"], params["cv_rate"]*0.25))
                        conversions = np.random.binomial(clicks, cv_rate) if clicks>0 else 0

                        # money metrics
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
    # keep types tidy
    df["date"] = pd.to_datetime(df["date"])
    return df

# -----------------------
# Metric helpers
# -----------------------
def agg_period(df, start=None, end=None):
    sub = df.copy()
    if start is not None:
        sub = sub[sub["date"] >= pd.to_datetime(start)]
    if end is not None:
        sub = sub[sub["date"] <= pd.to_datetime(end)]
    return sub

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

# -----------------------
# Plot builders (Plotly)
# -----------------------
def plot_spend_revenue_trend(df, date_col="date"):
    ts = df.groupby(date_col).agg({"revenue":"sum","spend":"sum"}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ts[date_col], y=ts["spend"], name="Spend", marker_color="#7f7fff", yaxis="y2", opacity=0.6))
    fig.add_trace(go.Scatter(x=ts[date_col], y=ts["revenue"], name="Revenue", mode="lines+markers", line=dict(color="#0fb78a", width=3)))
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title=""),
        yaxis=dict(title="Revenue", showgrid=False),
        yaxis2=dict(title="Spend", overlaying="y", side="right", showgrid=False),
        margin=dict(l=40, r=40, t=10, b=40),
        template="plotly_dark",
        height=350
    )
    return fig

def plot_roas_by_channel(df):
    agg = df.groupby("channel").agg({"revenue":"sum","spend":"sum"}).reset_index()
    agg["roas"] = agg["revenue"] / agg["spend"]
    agg = agg.sort_values("roas", ascending=False)
    fig = px.bar(agg, x="roas", y="channel", orientation="h", text=agg["roas"].round(2), color="channel",
                 color_discrete_map={"Meta":"#1D4ED8","Google":"#FACC15","Amazon":"#F59E0B","TikTok":"#F43F5E"})
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=80,r=20,t=30,b=20), showlegend=False)
    fig.update_xaxes(title="ROAS (Revenue / Spend)")
    return fig

def plot_cac_trend(df):
    ts = df.groupby("date").apply(lambda x: pd.Series({"cac": x["spend"].sum() / x["conversions"].sum() if x["conversions"].sum()>0 else np.nan})).reset_index()
    fig = px.line(ts, x="date", y="cac", title="CAC Trend")
    fig.update_layout(template="plotly_dark", height=250, margin=dict(l=40,r=20,t=30,b=20))
    fig.update_yaxes(title="CAC")
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
        totals=dict(marker=dict(color="#06b6d4"))
    ))
    fig.update_layout(template="plotly_dark", height=360, margin=dict(l=40,r=20,t=20,b=20))
    return fig

def plot_funnel_bars(df):
    # funnel numbers aggregated
    impressions = df["impressions"].sum()
    clicks = df["clicks"].sum()
    product_views = int(df["orders"].sum() * 3.5)  # fake but representative
    add_to_cart = int(product_views * 0.25)
    purchases = int(df["orders"].sum())
    steps = pd.DataFrame({
        "step":["Impressions","Clicks","Product Views","Add to Cart","Purchases"],
        "value":[impressions, clicks, product_views, add_to_cart, purchases]
    })
    steps["pct"] = steps["value"] / steps["value"].iloc[0]
    fig = px.bar(steps, y="step", x="value", orientation="h", text=steps["value"].apply(lambda v: f"{v:,}"))
    fig.update_layout(template="plotly_dark", height=340, margin=dict(l=120,r=20,t=20,b=20))
    fig.update_xaxes(showgrid=False)
    return fig

# -----------------------
# Streamlit layout
# -----------------------
st.set_page_config(page_title="Lifesight - Marketing Performance", layout="wide", initial_sidebar_state="collapsed")
st.title("Marketing Performance Dashboard")
st.markdown("**Executive Summary** — High-level KPIs linking marketing investment to company financial health.")

# load data
df = generate_mock_data(months_before=3, months_after=3)

# Top controls
with st.sidebar:
    st.header("Filters & Controls")
    ch_options = ["All"] + sorted(df["channel"].unique().tolist())
    selected_channel = st.selectbox("Channel", ch_options, index=0)
    start_date = st.date_input("Start date", value=df["date"].min().date())
    end_date = st.date_input("End date", value=df["date"].max().date())

# page-scoped filter & subset
subset = df.copy()
if selected_channel != "All":
    subset = subset[subset["channel"] == selected_channel]
subset = subset[(subset["date"] >= pd.to_datetime(start_date)) & (subset["date"] <= pd.to_datetime(end_date))]

# Tabs: Exec / CMO / CFO
tab1, tab2, tab3 = st.tabs(["Executive Summary", "CMO View", "CFO View"])

# Executive Summary
with tab1:
    st.subheader("Executive Snapshot")
    kpis = compute_exec_kpis(subset)

    col1, col2, col3, col4, col5 = st.columns([2,1,1,1,1], gap="large")
    col1.metric("Total Net Revenue", f"₹{kpis['total_revenue']:,.0f}")
    col2.metric("Gross Profit Margin", f"{kpis['gross_margin']*100:.1f}%" if not np.isnan(kpis['gross_margin']) else "N/A")
    col3.metric("Marketing Efficiency Ratio (MER)", f"{kpis['mer']}" if not np.isnan(kpis['mer']) else "N/A")
    col4.metric("LTV : CAC Ratio", f"{kpis['ltv_cac']}" if not np.isnan(kpis['ltv_cac']) else "N/A")
    col5.metric("Total Profit", f"₹{kpis['profit']:,.0f}")

    st.plotly_chart(plot_spend_revenue_trend(subset, date_col="date"), use_container_width=True)

# CMO View
with tab2:
    st.subheader("CMO — Marketing Effectiveness")
    # ROAS by channel
    left, right = st.columns([1,1], gap="large")
    with left:
        st.plotly_chart(plot_roas_by_channel(subset), use_container_width=True)
    with right:
        st.plotly_chart(plot_funnel_bars(subset), use_container_width=True)

    # Campaign/creative diagnostics table (top performance)
    st.markdown("#### Campaign & Creative Diagnostics")
    diag = subset.groupby(["channel","campaign","ad_set","creative"]).agg({
        "spend":"sum","impressions":"sum","clicks":"sum","conversions":"sum","revenue":"sum"
    }).reset_index()
    diag["ctr"] = (diag["clicks"] / diag["impressions"]).round(4)
    diag["cvr"] = (diag["conversions"] / diag["clicks"]).round(4)
    diag["cpa"] = (diag["spend"] / diag["conversions"]).round(2).replace([np.inf, -np.inf], np.nan)
    st.dataframe(diag.sort_values("revenue", ascending=False).head(30), use_container_width=True)

# CFO View
with tab3:
    st.subheader("CFO — Financial Efficiency & Profitability")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.plotly_chart(plot_contribution_waterfall(subset), use_container_width=True)
    with col2:
        st.plotly_chart(plot_cac_trend(subset), use_container_width=True)

    # ROI vs Budget mock
    st.markdown("#### Marketing ROI vs Mock Budget")
    budget = subset.groupby("date").agg(budget=("spend", lambda s: s.sum()*1.1), spend=("spend","sum"), revenue=("revenue","sum")).reset_index()
    budget["roi"] = (budget["revenue"] - budget["spend"]) / budget["spend"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=budget["date"], y=budget["budget"], name="Budget (mock)", marker_color="#6b7280", opacity=0.6))
    fig.add_trace(go.Bar(x=budget["date"], y=budget["spend"], name="Actual Spend", marker_color="#7f7fff", opacity=0.8))
    fig.add_trace(go.Scatter(x=budget["date"], y=budget["roi"], name="ROI", yaxis="y2", line=dict(color="#10b981", width=3)))
    fig.update_layout(template="plotly_dark", height=360, margin=dict(l=40,r=40,t=20,b=30),
                      yaxis=dict(title="Amount"), yaxis2=dict(title="ROI", overlaying="y", side="right", tickformat=".0%"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Cost Efficiency Metrics")
    aov = subset["aov"].mean()
    refund_rate = subset["returns"].sum() / subset["orders"].sum() if subset["orders"].sum()>0 else np.nan
    st.metric("Average Order Value (AOV)", f"₹{aov:,.2f}")
    st.metric("Refund / Return Rate", f"{refund_rate*100:.2f}%" if not np.isnan(refund_rate) else "N/A")

st.markdown("---")
st.caption("Mock dataset generated for demonstration purposes. Dashboard design & metric selection follow the Lifesight assignment brief. :contentReference[oaicite:2]{index=2}")
