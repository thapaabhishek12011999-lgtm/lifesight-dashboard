# lifesight_dashboard_improved.py
"""
Improved Lifesight Streamlit dashboard:
- Dark theme CSS
- Horizontal filters (channel, campaign, creative, date range)
- KPI cards with period-over-period deltas
- Auto insight callout (semi-dynamic)
- Download filtered CSV
- Cohort LTV heatmap
- Polished Plotly charts
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Utilities & Mock data (same generator as before)
# ---------------------------
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

# ---------------------------
# Styling (dark theme tweaks)
# ---------------------------
DARK_BG = "#0f172a"
CARD_BG = "#0b1220"
ACCENT = "#14B8A6"
PURPLE = "#8B5CF6"

def inject_css():
    st.markdown(
        f"""
    <style>
    .stApp {{ background-color: {DARK_BG}; color: #e6eef8; }}
    .reportview-container .main .block-container{{ padding-top:1rem; padding-left:2rem; padding-right:2rem;}}
    .kpi-card {{ background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding: 14px; border-radius: 12px; }}
    .insight-box {{ background: #071024; padding:14px; border-radius:10px; color:#cbd5e1; }}
    .filter-box .stSelectbox, .filter-box .stDateInput {{ background: #0b1220; border-radius:8px; color:#e6eef8; }}
    </style>
    """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Core metrics/helpers
# ---------------------------
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

def period_summary(df, start_date, end_date, prev_start, prev_end):
    # returns two dicts (current, previous)
    cur = compute_exec_kpis(df[(df["date"]>=pd.to_datetime(start_date)) & (df["date"]<=pd.to_datetime(end_date))])
    prev = compute_exec_kpis(df[(df["date"]>=pd.to_datetime(prev_start)) & (df["date"]<=pd.to_datetime(prev_end))])
    return cur, prev

# ---------------------------
# Charts (Plotly)
# ---------------------------
def plot_spend_revenue_trend(df):
    ts = df.groupby("date").agg({"revenue":"sum","spend":"sum"}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ts["date"], y=ts["spend"], name="Spend", marker_color="#7f7fff", yaxis="y2", opacity=0.6))
    fig.add_trace(go.Scatter(x=ts["date"], y=ts["revenue"], name="Revenue", mode="lines+markers", line=dict(color=ACCENT, width=3)))
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title=""),
        yaxis=dict(title="Revenue", showgrid=False),
        yaxis2=dict(title="Spend", overlaying="y", side="right", showgrid=False),
        margin=dict(l=40, r=40, t=10, b=40),
        template="plotly_dark",
        height=340
    )
    return fig

def plot_roas_by_channel(df):
    agg = df.groupby("channel").agg({"revenue":"sum","spend":"sum"}).reset_index()
    agg["roas"] = agg["revenue"] / agg["spend"]
    agg = agg.sort_values("roas", ascending=False)
    color_map = {"Meta":"#1D4ED8","Google":"#FACC15","Amazon":"#F59E0B","TikTok":"#F43F5E"}
    fig = px.bar(agg, x="roas", y="channel", orientation="h", text=agg["roas"].round(2), color="channel",
                 color_discrete_map=color_map)
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

def cohort_ltv_heatmap(df, months=6):
    # cohort by first purchase month -> cumulative revenue per customer over next N months
    orders = df[df["orders"]>0].copy()
    # Simulate first_purchase_date as date (in our mock this is same as order date)
    orders["cohort_month"] = orders["date"].dt.to_period("M").dt.to_timestamp()
    orders["order_month"] = orders["date"].dt.to_period("M").dt.to_timestamp()
    # customer_id not present - so we use creative-level pseudo customers: sum revenue per cohort-month bucket as proxy
    pivot = orders.groupby(["cohort_month","order_month"]).agg({"revenue":"sum","orders":"sum"}).reset_index()
    pivot["months_since"] = ((pivot["order_month"].dt.year - pivot["cohort_month"].dt.year)*12 + (pivot["order_month"].dt.month - pivot["cohort_month"].dt.month))
    # only months >=0 and <=months
    pivot = pivot[(pivot["months_since"]>=0) & (pivot["months_since"]<months)]
    heat = pivot.pivot_table(index="cohort_month", columns="months_since", values="revenue", aggfunc="sum").fillna(0)
    # normalize by cohort size (orders) to get per-customer-like LTV (approx)
    # for display, convert to thousands
    fig = go.Figure(data=go.Heatmap(
        z=heat.values,
        x=[f"M+{c}" for c in heat.columns],
        y=[d.strftime("%Y-%m") for d in heat.index],
        colorscale="Viridis"
    ))
    fig.update_layout(template="plotly_dark", height=360, margin=dict(l=80,r=20,t=30,b=20))
    return fig

# ---------------------------
# App layout & UI
# ---------------------------
st.set_page_config(page_title="Lifesight - Marketing Performance (Improved)", layout="wide", initial_sidebar_state="collapsed")
inject_css()

df = generate_mock_data(months_before=3, months_after=3)

# Top filter row (applies to page)
st.markdown("### Executive Summary")
# build filter controls in a single horizontal row using columns
fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns([1.2,1.2,1.2,1.6,0.6])
with fcol1:
    channel_options = ["All"] + sorted(df["channel"].unique().tolist())
    channel = st.selectbox("Channel", channel_options, index=0, key="f_channel")
with fcol2:
    campaign_options = ["All"] + sorted(df["campaign"].unique().tolist())
    campaign = st.selectbox("Campaign", campaign_options, index=0, key="f_campaign")
with fcol3:
    creative_options = ["All"] + sorted(df["creative"].unique().tolist())
    creative = st.selectbox("Creative", creative_options, index=0, key="f_creative")
with fcol4:
    start_date = st.date_input("Start date", value=df["date"].min().date())
    end_date = st.date_input("End date", value=df["date"].max().date())
with fcol5:
    if st.button("Download CSV"):
        filtered = df.copy()
        if channel != "All":
            filtered = filtered[filtered["channel"] == channel]
        if campaign != "All":
            filtered = filtered[filtered["campaign"] == campaign]
        if creative != "All":
            filtered = filtered[filtered["creative"] == creative]
        filtered = filtered[(filtered["date"]>=pd.to_datetime(start_date)) & (filtered["date"]<=pd.to_datetime(end_date))]
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download", data=csv, file_name="lifesight_filtered.csv", mime="text/csv")

# subset data
subset = df.copy()
if channel != "All":
    subset = subset[subset["channel"] == channel]
if campaign != "All":
    subset = subset[subset["campaign"] == campaign]
if creative != "All":
    subset = subset[subset["creative"] == creative]
subset = subset[(subset["date"] >= pd.to_datetime(start_date)) & (subset["date"] <= pd.to_datetime(end_date))]

# If subset empty, show message
if subset.empty:
    st.warning("No data for selected filters / date range. Adjust filters.")
    st.stop()

# compute current and previous period (same length) for PoP
curr_start, curr_end = pd.to_datetime(start_date), pd.to_datetime(end_date)
period_days = (curr_end - curr_start).days + 1
prev_end = curr_start - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=period_days-1)

cur_kpis, prev_kpis = period_summary(df, curr_start, curr_end, prev_start, prev_end)

# KPI cards row
kcols = st.columns([1.6,1,1,1,1], gap="large")
def render_kpi(col, label, cur_value, prev_value, prefix="₹", is_pct=False):
    # compute delta
    if prev_value is None or prev_value==0 or np.isnan(prev_value):
        delta = None
    else:
        try:
            delta = (cur_value - prev_value) / abs(prev_value)
        except:
            delta = None
    # format
    if is_pct:
        display = f"{cur_value*100:.1f}%"
    else:
        display = f"{prefix}{cur_value:,.0f}" if not np.isnan(cur_value) else "N/A"
    if delta is None:
        delta_display = "N/A"
    else:
        sign = "▲" if delta>0 else "▼"
        color = "green" if delta>0 else "red"
        delta_display = f"{sign} {abs(delta*100):.1f}%"
    with col:
        st.markdown(f"**{label}**")
        st.markdown(f"<div class='kpi-card'><h2 style='margin:0;color:#e6eef8'>{display}</h2><div style='color:{'#14B8A6' if (delta and delta>0) else '#ef4444' if (delta and delta<0) else '#94a3b8'}; font-size:14px'>{delta_display if delta_display else ''}</div></div>", unsafe_allow_html=True)

render_kpi(kcols[0], "Total Net Revenue", cur_kpis["total_revenue"], prev_kpis["total_revenue"])
render_kpi(kcols[1], "Gross Profit Margin", cur_kpis["gross_margin"], prev_kpis["gross_margin"], prefix="", is_pct=True)
render_kpi(kcols[2], "Marketing Efficiency Ratio (MER)", cur_kpis["mer"], prev_kpis["mer"], prefix="", is_pct=False)
render_kpi(kcols[3], "LTV : CAC Ratio", cur_kpis["ltv_cac"], prev_kpis["ltv_cac"], prefix="", is_pct=False)
render_kpi(kcols[4], "Total Profit", cur_kpis["profit"], prev_kpis["profit"])

# Auto insight callout (semi-dynamic)
insight_lines = []
try:
    rev_delta = None if np.isnan(prev_kpis["total_revenue"]) or prev_kpis["total_revenue"]==0 else (cur_kpis["total_revenue"] - prev_kpis["total_revenue"]) / prev_kpis["total_revenue"]
    if rev_delta is not None:
        if rev_delta > 0.05:
            insight_lines.append(f"Revenue increased by {rev_delta*100:.1f}% vs previous period — strong growth.")
        elif rev_delta < -0.05:
            insight_lines.append(f"Revenue decreased by {abs(rev_delta)*100:.1f}% vs previous period — investigate conversion or channel performance.")
    mer_delta = None if np.isnan(prev_kpis["mer"]) or prev_kpis["mer"]==0 else (cur_kpis["mer"] - prev_kpis["mer"]) / prev_kpis["mer"]
    if mer_delta is not None and mer_delta < -0.05:
        insight_lines.append("MER declined — spend is less efficient; consider pausing low-performing campaigns.")
    # top channel by revenue change
    ch_current = subset.groupby("channel")["revenue"].sum().sort_values(ascending=False)
    if not ch_current.empty:
        top_ch = ch_current.idxmax()
        insight_lines.append(f"Top channel in the selected period: {top_ch}.")
except Exception as e:
    insight_lines.append("Insights not available due to limited data.")

insight_text = "<br>".join(insight_lines) if insight_lines else "No significant changes detected."
st.markdown(f"<div class='insight-box'><strong>Insights:</strong><br>{insight_text}</div>", unsafe_allow_html=True)

# Revenue & Spend trend
st.plotly_chart(plot_spend_revenue_trend(subset), use_container_width=True)

# Tabs for CMO / CFO
tab1, tab2 = st.tabs(["CMO View", "CFO View"])
with tab1:
    st.subheader("CMO — Marketing Effectiveness")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(plot_roas_by_channel(subset), use_container_width=True)
    with c2:
        st.plotly_chart(plot_cac_trend(subset), use_container_width=True)
    st.markdown("#### Campaign & Creative Diagnostics (top rows)")
    diag = subset.groupby(["channel","campaign","ad_set","creative"]).agg({
        "spend":"sum","impressions":"sum","clicks":"sum","conversions":"sum","revenue":"sum"
    }).reset_index()
    diag["ctr"] = (diag["clicks"] / diag["impressions"]).round(4)
    diag["cvr"] = (diag["conversions"] / diag["clicks"]).round(4)
    diag["cpa"] = (diag["spend"] / diag["conversions"]).round(2).replace([np.inf, -np.inf], np.nan)
    st.dataframe(diag.sort_values("revenue", ascending=False).head(50), use_container_width=True)

with tab2:
    st.subheader("CFO — Financial Efficiency")
    w1, w2 = st.columns(2, gap="large")
    with w1:
        st.plotly_chart(plot_contribution_waterfall(subset), use_container_width=True)
    with w2:
        st.plotly_chart(plot_cac_trend(subset), use_container_width=True)
    st.markdown("#### LTV Cohort Heatmap (Revenue by cohort month)")
    st.plotly_chart(cohort_ltv_heatmap(subset, months=6), use_container_width=True)

st.caption("Mock dataset and visualization for Lifesight assignment. Ask Zoro to add or tweak any visual/style.")
