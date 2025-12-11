# lifesight_dashboard_v4.py
"""
Lifesight dashboard v4

Based on prior file/layers. :contentReference[oaicite:1]{index=1}
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# optionally used libraries
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False

# Try kaleido for PNG export
try:
    import kaleido  # noqa: F401
    HAS_KALEIDO = True
except Exception:
    HAS_KALEIDO = False

# Try img2pdf for merging PNGs into PDF (optional)
try:
    import img2pdf  # noqa: F401
    HAS_IMG2PDF = True
except Exception:
    HAS_IMG2PDF = False

# -----------------------
# Mock data generator (daily, 3 months before + 3 months after today)
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
# Theme / Styling (light and dark)
# -----------------------
# Define colors
PRIMARY_PURPLE = "#6B21A8"
ACCENT_GREEN = "#14B8A6"
TEXT_LIGHT = "#eef2ff"
TEXT_DARK = "#0f172a"

def inject_css(theme="light"):
    if theme == "dark":
        page_bg = "#0b1220"
        card_bg = "#0f172a"
        text_color = TEXT_LIGHT
        small_text = "#cbd5e1"
    else:
        page_bg = "#ffffff"
        card_bg = "#ffffff"
        text_color = TEXT_DARK
        small_text = "#6b7280"

    st.markdown(
        f"""
    <style>
    .stApp {{ background: {page_bg}; color: {text_color}; }}
    .block-container{{padding-top:1rem; padding-left:1.5rem; padding-right:1.5rem; max-width:1600px;}}
    .topband {{
        background: linear-gradient(90deg,{PRIMARY_PURPLE} 0%, #7c3aed 100%);
        color: white;
        padding: 8px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }}
    .kpi-large {{
        background: {card_bg};
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        color: {text_color};
    }}
    .kpi-small {{
        background: {card_bg};
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.04);
        color: {text_color};
    }}
    .insight {{
        background: {card_bg};
        padding: 10px;
        border-radius: 8px;
        color: {small_text};
        border: 1px solid rgba(0,0,0,0.04);
    }}
    .filter-row {{ position: sticky; top: 8px; background: transparent; z-index:100; padding-bottom:10px; }}
    .filter-label {{ font-size:13px; color:{small_text}; font-weight:600; }}
    </style>
    """,
        unsafe_allow_html=True,
    )

# -----------------------
# KPI helpers & calculations
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

# -----------------------
# Plot builders with titles/subtitles (respecting template variable)
# -----------------------
def plot_spend_revenue_trend(df, template="plotly_white"):
    ts = df.groupby("date").agg({"revenue":"sum","spend":"sum"}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ts["date"], y=ts["spend"], name="Spend", marker_color=PRIMARY_PURPLE, yaxis="y2", opacity=0.6))
    fig.add_trace(go.Scatter(x=ts["date"], y=ts["revenue"], name="Revenue", mode="lines+markers", line=dict(color=ACCENT_GREEN, width=3)))
    fig.update_layout(
        title=dict(text="Revenue & Marketing Spend Trend", x=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Revenue"),
        yaxis2=dict(title="Spend", overlaying="y", side="right"),
        margin=dict(l=40, r=40, t=60, b=40),
        template=template,
        height=360
    )
    return fig

def plot_roas_by_channel(df, template="plotly_white"):
    agg = df.groupby("channel").agg({"revenue":"sum","spend":"sum"}).reset_index()
    agg["roas"] = agg["revenue"] / agg["spend"]
    color_map = {"Meta":"#1D4ED8","Google":"#FACC15","Amazon":"#F59E0B","TikTok":"#F43F5E"}
    fig = px.bar(agg.sort_values("roas", ascending=False), x="roas", y="channel", orientation="h",
                 text=agg["roas"].round(2), color="channel", color_discrete_map=color_map, labels={"roas":"ROAS"})
    fig.update_layout(title_text="ROAS by Channel — click a bar to drilldown", template=template, height=320, margin=dict(l=120,t=50,b=20), showlegend=False)
    fig.update_xaxes(title="ROAS (Revenue / Spend)")
    return fig

def plot_funnel_bars(df, template="plotly_white"):
    impressions = df["impressions"].sum()
    clicks = df["clicks"].sum()
    product_views = int(df["orders"].sum() * 3.5)
    add_to_cart = int(product_views * 0.25)
    purchases = int(df["orders"].sum())
    steps = pd.DataFrame({
        "step":["Impressions","Clicks","Product Views","Add to Cart","Purchases"],
        "value":[impressions, clicks, product_views, add_to_cart, purchases]
    })
    fig = px.bar(steps, x="value", y="step", orientation="h", text=steps["value"].apply(lambda v: f"{v:,}"))
    fig.update_layout(title_text="Marketing Funnel: Impression → Purchase", template=template, height=320, margin=dict(l=120,t=50,b=20))
    fig.update_xaxes(title="Count")
    return fig

def plot_cac_trend(df, template="plotly_white"):
    ts = df.groupby("date").apply(lambda x: pd.Series({"cac": x["spend"].sum() / x["conversions"].sum() if x["conversions"].sum()>0 else np.nan})).reset_index()
    fig = px.line(ts, x="date", y="cac", title="Customer Acquisition Cost (CAC) Trend")
    fig.update_layout(template=template, height=260, margin=dict(l=40,r=20,t=50,b=20))
    fig.update_yaxes(title="CAC")
    return fig

def plot_contribution_waterfall(df, template="plotly_white"):
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
        totals=dict(marker=dict(color=PRIMARY_PURPLE))
    ))
    fig.update_layout(title_text="Contribution Margin Breakdown", template=template, height=360, margin=dict(l=40,r=20,t=60,b=20))
    return fig

def cohort_ltv_heatmap(df, months=6, template="plotly_white"):
    orders = df[df["orders"]>0].copy()
    if orders.empty:
        return go.Figure()
    orders["cohort_month"] = orders["date"].dt.to_period("M").dt.to_timestamp()
    orders["order_month"] = orders["date"].dt.to_period("M").dt.to_timestamp()
    pivot = orders.groupby(["cohort_month","order_month"]).agg({"revenue":"sum"}).reset_index()
    pivot["months_since"] = ((pivot["order_month"].dt.year - pivot["cohort_month"].dt.year)*12 + (pivot["order_month"].dt.month - pivot["cohort_month"].dt.month))
    pivot = pivot[(pivot["months_since"]>=0) & (pivot["months_since"]<months)]
    heat = pivot.pivot_table(index="cohort_month", columns="months_since", values="revenue", aggfunc="sum").fillna(0)
    fig = go.Figure(data=go.Heatmap(
        z=heat.values,
        x=[f"M+{c}" for c in heat.columns],
        y=[d.strftime("%Y-%m") for d in heat.index],
        colorscale="Purples"
    ))
    fig.update_layout(title_text="Cohort LTV Heatmap (Revenue by Cohort Month)", template=template, height=360, margin=dict(l=80,r=20,t=60,b=20))
    return fig

# -----------------------
# Export helpers
# -----------------------
def fig_to_png_bytes(fig, width=1200, height=600, scale=2):
    if not HAS_KALEIDO:
        raise RuntimeError("kaleido not available. Install `kaleido` to enable PNG export.")
    img = pio.to_image(fig, format="png", width=width, height=height, scale=scale)
    return img

def figures_to_pdf_bytes(figs):
    # renders each fig to PNG bytes and then optionally combine via img2pdf
    pngs = []
    for fig in figs:
        try:
            pngs.append(fig_to_png_bytes(fig))
        except Exception as e:
            raise
    if not HAS_IMG2PDF:
        # fallback: return first PNG
        return pngs[0]
    # combine
    pdf_bytes = img2pdf.convert(pngs)
    return pdf_bytes

# -----------------------
# Natural language insights (improved)
# -----------------------
def generate_insights(df_subset, cur_kpis, prev_kpis, df_full, prev_start, prev_end):
    insights = []
    # Revenue delta
    try:
        rev_cur = cur_kpis["total_revenue"]
        rev_prev = prev_kpis["total_revenue"]
        if rev_prev and rev_prev>0:
            rev_delta = (rev_cur - rev_prev)/rev_prev
            if rev_delta > 0.08:
                insights.append(f"Revenue ↑ {rev_delta*100:.1f}% vs previous period — strong topline growth.")
            elif rev_delta < -0.08:
                insights.append(f"Revenue ↓ {abs(rev_delta)*100:.1f}% vs previous period — investigate conversion or traffic drops.")
            else:
                insights.append(f"Revenue change is {rev_delta*100:.1f}% vs previous period.")
    except Exception:
        pass

    # MER
    try:
        if prev_kpis["mer"] and not np.isnan(prev_kpis["mer"]):
            mer_delta = (cur_kpis["mer"] - prev_kpis["mer"]) / prev_kpis["mer"]
            if mer_delta < -0.05:
                insights.append("MER declined >5% — efficiency dropping; review low-ROAS campaigns.")
    except Exception:
        pass

    # CAC
    try:
        if prev_kpis["cac"] and not np.isnan(prev_kpis["cac"]):
            cac_delta = (cur_kpis["cac"] - prev_kpis["cac"]) / prev_kpis["cac"]
            if cac_delta > 0.1:
                insights.append(f"CAC increased {cac_delta*100:.1f}% — acquisition getting more expensive.")
    except Exception:
        pass

    # Channel winners/losers (compare to previous period)
    try:
        ch_curr = df_subset.groupby("channel")["revenue"].sum().sort_values(ascending=False)
        ch_prev = df_full[(df_full["date"]>=prev_start) & (df_full["date"]<=prev_end)].groupby("channel")["revenue"].sum()
        ch_changes = {}
        for ch in ch_curr.index:
            p = ch_prev.get(ch, 0)
            c = ch_curr.loc[ch]
            if p>0:
                ch_changes[ch] = (c-p)/p
        if ch_changes:
            best = max(ch_changes, key=ch_changes.get)
            worst = min(ch_changes, key=ch_changes.get)
            insights.append(f"Top channel: {best} ({ch_changes[best]*100:+.1f}% vs prev). Worst: {worst} ({ch_changes[worst]*100:+.1f}%).")
    except Exception:
        pass

    if not insights:
        insights = ["No significant changes detected for the selected filters/time range."]
    return insights

# -----------------------
# App layout & interactions
# -----------------------
st.set_page_config(page_title="Lifesight - Marketing Performance", layout="wide")
# Theme toggle
theme_choice = st.sidebar.radio("Theme", ["Lifesight (Light)", "Dark Mode"], index=0)
theme = "dark" if theme_choice == "Dark Mode" else "light"
inject_css(theme=theme)
plot_template = "plotly_dark" if theme=="dark" else "plotly_white"

# Top branding band (sticky effect)
st.markdown(f"<div class='topband'><strong style='font-size:18px;color:white'>Lifesight</strong> — Marketing Performance Dashboard</div>", unsafe_allow_html=True)

# Load data
df = generate_mock_data(months_before=3, months_after=3)

# --- Sticky filter row header
st.markdown("<div class='filter-row'>", unsafe_allow_html=True)
st.markdown("<div class='filter-label'>Filters</div>", unsafe_allow_html=True)

# Filters: channel, campaign, creative, date range, with headers
col1, col2, col3, col4, col5 = st.columns([1.2,1.2,1.2,1.6,0.6])
with col1:
    st.markdown("<div style='font-size:13px; margin-bottom:4px;'>Channel</div>", unsafe_allow_html=True)
    channel_sel = st.selectbox(" ", ["All"] + sorted(df["channel"].unique().tolist()), index=0, key="channel_filter")
with col2:
    st.markdown("<div style='font-size:13px; margin-bottom:4px;'>Campaign</div>", unsafe_allow_html=True)
    campaign_sel = st.selectbox(" ", ["All"] + sorted(df["campaign"].unique().tolist()), index=0, key="campaign_filter")
with col3:
    st.markdown("<div style='font-size:13px; margin-bottom:4px;'>Creative</div>", unsafe_allow_html=True)
    creative_sel = st.selectbox(" ", ["All"] + sorted(df["creative"].unique().tolist()), index=0, key="creative_filter")
with col4:
    st.markdown("<div style='font-size:13px; margin-bottom:4px;'>Date Range</div>", unsafe_allow_html=True)
    start_date = st.date_input(" ", value=df["date"].min().date())
    end_date = st.date_input(" ", value=df["date"].max().date())
with col5:
    st.markdown("<div style='font-size:13px; margin-bottom:4px;'>Download</div>", unsafe_allow_html=True)
    # placeholder buttons (export is implemented later)
    st.write("")

st.markdown("</div>", unsafe_allow_html=True)

# apply filters
subset = df.copy()
if channel_sel != "All":
    subset = subset[subset["channel"] == channel_sel]
if campaign_sel != "All":
    subset = subset[subset["campaign"] == campaign_sel]
if creative_sel != "All":
    subset = subset[subset["creative"] == creative_sel]
subset = subset[(subset["date"]>=pd.to_datetime(start_date)) & (subset["date"]<=pd.to_datetime(end_date))]

if subset.empty:
    st.warning("No data for selected filters / date range. Adjust filters.")
    st.stop()

# compute current and previous period for PoP
curr_start = pd.to_datetime(start_date); curr_end = pd.to_datetime(end_date)
period_days = (curr_end - curr_start).days + 1
prev_end = curr_start - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=period_days-1)
cur_kpis = compute_exec_kpis(subset)
prev_kpis = compute_exec_kpis(df[(df["date"]>=prev_start) & (df["date"]<=prev_end)])

# KPI Row (F-pattern) with tooltips (via title attr)
left_col, right_col = st.columns([2,3], gap="large")
with left_col:
    # big revenue card
    rev = cur_kpis["total_revenue"]
    rev_delta = compute_pop(cur_kpis["total_revenue"], prev_kpis["total_revenue"])
    delta_html = f"<span title='Period-over-period change vs previous {period_days} days'>{'▲' if rev_delta and rev_delta>0 else '▼' if rev_delta and rev_delta<0 else ''} {abs(rev_delta)*100:.1f}%</span>" if rev_delta is not None else "<span title='No previous period available'>N/A</span>"
    st.markdown(f"<div class='kpi-large'><div style='font-size:14px;color:#6b7280'>Total Net Revenue</div><div style='font-size:28px;font-weight:700'>₹{rev:,.0f}</div><div style='margin-top:6px;color:#374151'>{delta_html}</div></div>", unsafe_allow_html=True)

with right_col:
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    # Gross Profit Margin
    with k1:
        gpm = cur_kpis["gross_margin"]
        gpm_delta = compute_pop(gpm, prev_kpis["gross_margin"])
        st.markdown(f"<div class='kpi-small'><div style='font-size:13px;color:#6b7280'>Gross Profit Margin</div><div style='font-size:20px;font-weight:700'>{gpm*100:.1f}%</div><div title='Gross margin = (revenue - COGS)/revenue' style='color:#6b7280'>{'▲' if gpm_delta and gpm_delta>0 else '▼' if gpm_delta and gpm_delta<0 else ''} {abs(gpm_delta*100):.1f}%</div></div>", unsafe_allow_html=True)
    # MER
    with k2:
        mer = cur_kpis["mer"]
        mer_delta = compute_pop(mer, prev_kpis["mer"])
        st.markdown(f"<div class='kpi-small'><div style='font-size:13px;color:#6b7280'>Marketing Efficiency Ratio (MER)</div><div style='font-size:20px;font-weight:700'>{mer:.2f}</div><div title='MER = revenue / marketing spend'>{'▲' if mer_delta and mer_delta>0 else '▼' if mer_delta and mer_delta<0 else ''} {abs(mer_delta*100):.1f}%</div></div>", unsafe_allow_html=True)
    # LTV:CAC
    with k3:
        ltvcac = cur_kpis["ltv_cac"]
        ltv_delta = compute_pop(ltvcac, prev_kpis["ltv_cac"])
        st.markdown(f"<div class='kpi-small'><div style='font-size:13px;color:#6b7280'>LTV : CAC</div><div style='font-size:20px;font-weight:700'>{ltvcac:.2f}</div><div title='LTV divided by CAC'>{'▲' if ltv_delta and ltv_delta>0 else '▼' if ltv_delta and ltv_delta<0 else ''} {abs(ltv_delta*100):.1f}%</div></div>", unsafe_allow_html=True)
    # Total Profit
    with k4:
        profit = cur_kpis["profit"]
        pf_delta = compute_pop(profit, prev_kpis["profit"])
        st.markdown(f"<div class='kpi-small'><div style='font-size:13px;color:#6b7280'>Total Profit</div><div style='font-size:20px;font-weight:700'>₹{profit:,.0f}</div><div title='Net profit after ads, COGS, returns'>{'▲' if pf_delta and pf_delta>0 else '▼' if pf_delta and pf_delta<0 else ''} {abs(pf_delta*100):.1f}%</div></div>", unsafe_allow_html=True)

# Auto insights
insights = generate_insights(subset, cur_kpis, prev_kpis, df, prev_start, prev_end)
insight_html = "<br>".join(insights)
st.markdown(f"<div class='insight'><strong>Automated Insights</strong><br>{insight_html}</div>", unsafe_allow_html=True)
st.markdown("---")

# Tabs
tabs = st.tabs(["Executive (Overview)", "CMO View (Marketing)", "CFO View (Finance)"])

# ---- EXECUTIVE TAB
with tabs[0]:
    st.subheader("Executive — Revenue & Spend Overview")
    fig_rev_spend = plot_spend_revenue_trend(subset, template=plot_template)
    st.plotly_chart(fig_rev_spend, use_container_width=True)

    # stacked new vs returning
    st.markdown("**Customer Acquisition & Retention** — New vs Returning revenue (proxy)")
    subset_agg = subset.groupby("date").agg({"revenue":"sum", "new_customers":"sum","returning_customers":"sum"}).reset_index()
    subset_agg["new_rev"] = subset_agg["revenue"] * (subset_agg["new_customers"] / (subset_agg["new_customers"] + subset_agg["returning_customers"] + 1e-9))
    subset_agg["ret_rev"] = subset_agg["revenue"] - subset_agg["new_rev"]
    fig_nr = go.Figure()
    fig_nr.add_trace(go.Scatter(x=subset_agg["date"], y=subset_agg["ret_rev"], stackgroup='one', name='Returning', line=dict(color="#8b5cf6")))
    fig_nr.add_trace(go.Scatter(x=subset_agg["date"], y=subset_agg["new_rev"], stackgroup='one', name='New', line=dict(color=ACCENT_GREEN)))
    fig_nr.update_layout(template=plot_template, height=300, margin=dict(t=30))
    st.plotly_chart(fig_nr, use_container_width=True)

    # export buttons for executive charts
    export_col1, export_col2, export_col3 = st.columns([1,1,1])
    with export_col1:
        if st.button("Export Revenue Chart PNG"):
            try:
                png = fig_to_png_bytes(fig_rev_spend)
                st.download_button("Download PNG", data=png, file_name="revenue_trend.png", mime="image/png")
            except Exception as e:
                st.error(f"PNG export failed: {e}. Ensure `kaleido` is installed.")
    with export_col2:
        if st.button("Export New/Returning PNG"):
            try:
                png = fig_to_png_bytes(fig_nr)
                st.download_button("Download PNG", data=png, file_name="new_retention.png", mime="image/png")
            except Exception as e:
                st.error(f"PNG export failed: {e}. Ensure `kaleido` is installed.")
    with export_col3:
        if st.button("Export Executive PDF (PNG-based)"):
            try:
                figs = [fig_rev_spend, fig_nr]
                pdf_bytes = figures_to_pdf_bytes(figs)
                st.download_button("Download PDF", data=pdf_bytes, file_name="executive_report.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF export failed: {e}. Install `img2pdf` and `kaleido` for full PDF export.")

# ---- CMO View
with tabs[1]:
    st.header("CMO — Marketing Effectiveness & Channel Diagnostics")
    st.markdown("**ROAS by Channel** (click a bar to filter by channel)")
    fig_roas = plot_roas_by_channel(subset, template=plot_template)
    # Render & capture click → only if streamlit-plotly-events installed
    if HAS_PLOTLY_EVENTS:
        selected = plotly_events(fig_roas, click_event=True, hover_event=False)
        st.plotly_chart(fig_roas, use_container_width=True)
        if selected:
            # selected returns dicts with keys like 'y' for horizontal bar
            val = selected[0].get("y") or selected[0].get("label") or None
            if val:
                # set session_state drill filter
                st.session_state["drill_channel"] = val
                st.success(f"Drilldown: filtering to channel {val}")
    else:
        st.plotly_chart(fig_roas, use_container_width=True)
        st.info("Click-to-drilldown requires `streamlit-plotly-events`. Install it to enable click interactions.")

    # If drill filter present, apply it and show a small badge
    if st.session_state.get("drill_channel"):
        dd = st.session_state["drill_channel"]
        st.markdown(f"**Drill filter active:** {dd} — use the Channel filter to clear.")
        # apply it to subset for downstream visuals in this tab
        local_subset = subset[subset["channel"]==dd]
    else:
        local_subset = subset

    st.markdown("**Marketing Funnel** — identify drop-off points")
    st.plotly_chart(plot_funnel_bars(local_subset, template=plot_template), use_container_width=True)

    # CMO KPI Scorecards (Total Impressions, CTR, Avg CPM, Avg Conversion Rate)
    st.markdown("**CMO KPIs**")
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        tot_impr = local_subset["impressions"].sum()
        st.markdown(f"<div class='kpi-small'><div style='font-size:13px;color:#6b7280'>Total Impressions</div><div style='font-size:20px;font-weight:700'>{tot_impr:,}</div></div>", unsafe_allow_html=True)
    with c2:
        ctr = (local_subset["clicks"].sum() / local_subset["impressions"].sum()) if local_subset["impressions"].sum()>0 else 0
        st.markdown(f"<div class='kpi-small' title='Click-through rate = clicks / impressions'><div style='font-size:13px;color:#6b7280'>Click-Through Rate</div><div style='font-size:20px;font-weight:700'>{ctr*100:.2f}%</div></div>", unsafe_allow_html=True)
    with c3:
        # avg CPM = (spend / impressions) * 1000
        cpm = (local_subset["spend"].sum() / local_subset["impressions"].sum()) * 1000 if local_subset["impressions"].sum()>0 else np.nan
        st.markdown(f"<div class='kpi-small' title='CPM = cost per thousand impressions'><div style='font-size:13px;color:#6b7280'>Avg. CPM</div><div style='font-size:20px;font-weight:700'>₹{cpm:,.2f}</div></div>", unsafe_allow_html=True)
    with c4:
        cvr = (local_subset["conversions"].sum() / local_subset["clicks"].sum()) if local_subset["clicks"].sum()>0 else 0
        st.markdown(f"<div class='kpi-small' title='Conversion rate = conversions / clicks'><div style='font-size:13px;color:#6b7280'>Avg. Conversion Rate</div><div style='font-size:20px;font-weight:700'>{cvr*100:.2f}%</div></div>", unsafe_allow_html=True)

    # Campaign diagnostics table
    st.markdown("**Campaign & Creative Diagnostics**")
    diag = local_subset.groupby(["channel","campaign","ad_set","creative"]).agg({
        "spend":"sum","impressions":"sum","clicks":"sum","conversions":"sum","revenue":"sum"
    }).reset_index()
    diag["ctr"] = (diag["clicks"] / diag["impressions"]).round(4)
    diag["cvr"] = (diag["conversions"] / diag["clicks"]).round(4)
    diag["cpa"] = (diag["spend"] / diag["conversions"]).round(2).replace([np.inf, -np.inf], pd.NA)
    st.dataframe(diag.sort_values("revenue", ascending=False).head(80), use_container_width=True)

# ---- CFO View
with tabs[2]:
    st.header("CFO — Financial Efficiency & Profitability")
    st.plotly_chart(plot_contribution_waterfall(subset, template=plot_template), use_container_width=True)
    st.markdown("**CAC Trend & Cost Efficiency**")
    st.plotly_chart(plot_cac_trend(subset, template=plot_template), use_container_width=True)

    # CFO KPIs (Marketing ROI, Gross Margin Rate, AOV, Refund & Return Rate)
    st.markdown("**CFO KPIs**")
    f1, f2, f3, f4 = st.columns(4, gap="large")
    with f1:
        roi = (subset["revenue"].sum() - subset["spend"].sum()) / subset["spend"].sum() if subset["spend"].sum()>0 else np.nan
        st.markdown(f"<div class='kpi-small' title='Marketing ROI = (Revenue - Spend) / Spend'><div style='font-size:13px;color:#6b7280'>Marketing ROI</div><div style='font-size:20px;font-weight:700'>{roi:.2f}</div></div>", unsafe_allow_html=True)
    with f2:
        gross_margin = cur_kpis["gross_margin"]
        st.markdown(f"<div class='kpi-small' title='Gross margin rate = (Revenue - COGS) / Revenue'><div style='font-size:13px;color:#6b7280'>Gross Margin Rate</div><div style='font-size:20px;font-weight:700'>{gross_margin*100:.1f}%</div></div>", unsafe_allow_html=True)
    with f3:
        aov = subset["aov"].mean()
        st.markdown(f"<div class='kpi-small' title='Average Order Value'><div style='font-size:13px;color:#6b7280'>Average Order Value (AOV)</div><div style='font-size:20px;font-weight:700'>₹{aov:,.2f}</div></div>", unsafe_allow_html=True)
    with f4:
        refund_rate = subset["returns"].sum() / subset["orders"].sum() if subset["orders"].sum()>0 else np.nan
        st.markdown(f"<div class='kpi-small' title='Refund & return rate = returns / orders'><div style='font-size:13px;color:#6b7280'>Refund & Return Rate</div><div style='font-size:20px;font-weight:700'>{refund_rate*100:.2f}%</div></div>", unsafe_allow_html=True)

    st.markdown("**Cohort LTV (heatmap)**")
    st.plotly_chart(cohort_ltv_heatmap(subset, months=6, template=plot_template), use_container_width=True)

# Footer / notes
st.markdown("---")
st.caption("This dashboard is a Lifesight-themed demo. Use the filters above to slice data. Click ROAS bars to drilldown (requires `streamlit-plotly-events`). PNG/PDF export requires `kaleido` and optional `img2pdf` for combined PDF output.")
