# lifesight_dashboard_v3_updated.py
"""
Lifesight-themed Marketing Performance Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from pathlib import Path

# Optional PDF library; use if available
try:
    from fpdf import FPDF
    _HAS_FPDF = True
except Exception:
    _HAS_FPDF = False

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
# Styling: Lifesight-like (purple + white)
# -----------------------
PRIMARY_PURPLE = "#6B21A8"
ACCENT_GREEN = "#14B8A6"
CARD_BG = "#ffffff"
PAGE_BG = "#ffffff"
TEXT_COLOR = "#0f172a"

def inject_css():
    st.markdown(
        f"""
    <style>
    /* Page background & header bar */
    .stApp {{ background: {PAGE_BG}; color: {TEXT_COLOR}; }}
    .reportview-container .main .block-container{{padding-top:1.5rem; padding-left:2rem; padding-right:2rem;}}
    /* Top header band to mimic Lifesight */
    .topband {{
        background: linear-gradient(90deg,{PRIMARY_PURPLE} 0%, #7c3aed 100%);
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        margin-bottom: 18px;
    }}
    /* KPI card style */
    .kpi-large {{
        background: {CARD_BG};
        padding: 18px;
        border-radius: 10px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        color: {TEXT_COLOR};
    }}
    .kpi-small {{
        background: {CARD_BG};
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.06);
        color: {TEXT_COLOR};
    }}
    .insight {{
        background: #f8fafc;
        padding: 12px;
        border-radius: 8px;
        color: {TEXT_COLOR};
    }}
    /* Small text tweaks */
    .kpi-label {{ font-size:14px; color:#374151; }}
    .kpi-value {{ font-size:28px; font-weight:700; color:{TEXT_COLOR}; }}
    .kpi-delta {{ font-size:13px; color:#059669; }}
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
    # compute percent change safely
    if prev is None or prev == 0 or np.isnan(prev):
        return None
    try:
        return (cur - prev) / abs(prev)
    except:
        return None

# -----------------------
# Plot builders with titles/subtitles
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
        margin=dict(l=40, r=40, t=60, b=40),
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
    fig.update_layout(title_text="ROAS by Channel", template="plotly_white", height=320, margin=dict(l=120,t=50,b=20))
    fig.update_xaxes(title="ROAS (Revenue / Spend)")
    return fig

def plot_funnel_bars(df):
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
    fig.update_layout(title_text="Marketing Funnel: Impression → Purchase", template="plotly_white", height=320, margin=dict(l=120,t=50,b=20))
    fig.update_xaxes(title="Count")
    return fig

def plot_cac_trend(df):
    ts = df.groupby("date").apply(lambda x: pd.Series({"cac": x["spend"].sum() / x["conversions"].sum() if x["conversions"].sum()>0 else np.nan})).reset_index()
    fig = px.line(ts, x="date", y="cac", title="Customer Acquisition Cost (CAC) Trend")
    fig.update_layout(template="plotly_white", height=260, margin=dict(l=40,r=20,t=50,b=20))
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
        totals=dict(marker=dict(color="#6B21A8"))
    ))
    fig.update_layout(title_text="Contribution Margin Breakdown", template="plotly_white", height=360, margin=dict(l=40,r=20,t=60,b=20))
    return fig

def cohort_ltv_heatmap(df, months=6):
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
    fig.update_layout(title_text="Cohort LTV Heatmap (Revenue by Cohort Month)", template="plotly_white", height=360, margin=dict(l=80,r=20,t=60,b=20))
    return fig

# -----------------------
# Export utilities: PDF report & figure -> PNG bytes
# -----------------------
def create_pdf_bytes(df, kpis, title="Dashboard Report"):
    """
    Returns PDF bytes containing title, KPIs and sample rows.
    Uses fpdf if available, else returns a plain-text bytes fallback.
    """
    if _HAS_FPDF:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt=title, ln=True, align='C')
        pdf.ln(6)
        pdf.set_font("Arial", size=10)
        for k, v in kpis.items():
            pdf.cell(0, 8, txt=f"{k}: {v}", ln=True)
        pdf.ln(4)
        pdf.cell(0, 8, txt="Sample data (top 10 rows):", ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", size=8)
        for _, row in df.head(10).iterrows():
            # keep each line short
            text = ", ".join([f"{c}:{row[c]}" for c in ["date","channel","campaign","impressions","clicks","spend","revenue"] if c in row])
            pdf.multi_cell(0, 6, txt=text)
        out = io.BytesIO()
        pdf.output(out)
        return out.getvalue()
    else:
        # fallback: plain text "report" bytes
        buf = io.StringIO()
        buf.write(f"{title}\n\n")
        buf.write("KPIs:\n")
        for k, v in kpis.items():
            buf.write(f"{k}: {v}\n")
        buf.write("\nSample rows (top 10):\n")
        buf.write(df.head(10).to_string(index=False))
        return buf.getvalue().encode("utf-8")

def fig_to_png_bytes(fig):
    """
    Attempt to convert a Plotly figure to PNG bytes.
    Requires kaleido or orca available in the environment for fig.to_image to work.
    Will return None if conversion fails.
    """
    try:
        # fig.to_image uses kaleido by default if installed
        img_bytes = fig.to_image(format="png")
        return img_bytes
    except Exception:
        try:
            import plotly.io as pio
            img_bytes = pio.to_image(fig, format="png")
            return img_bytes
        except Exception:
            return None

# -----------------------
# Streamlit app layout
# -----------------------
st.set_page_config(page_title="Lifesight - Marketing Performance", layout="wide")
inject_css()

# Top purple band (branding)
st.markdown(f"<div class='topband'><strong style='font-size:18px'>Lifesight</strong> — Marketing Performance Dashboard</div>", unsafe_allow_html=True)

# Load data
df = generate_mock_data(months_before=3, months_after=3)

# FILTER ROW (horizontal) — channel, campaign, creative, date range
st.markdown("### Filters")
st.markdown("Use the filters below to customise all calculations and charts. Filters apply to all tabs and exports.")
f1, f2, f3, f4, f5 = st.columns([1.2,1.2,1.2,1.6,0.4])
with f1:
    channel = st.selectbox("Select Channel", ["All"] + sorted(df["channel"].unique().tolist()), index=0)
with f2:
    campaign = st.selectbox("Select Campaign", ["All"] + sorted(df["campaign"].unique().tolist()), index=0)
with f3:
    creative = st.selectbox("Select Creative", ["All"] + sorted(df["creative"].unique().tolist()), index=0)
with f4:
    start_date = st.date_input("Start Date", value=df["date"].min().date())
    end_date = st.date_input("End Date", value=df["date"].max().date())

# subset data according to filters
subset = df.copy()
if channel != "All":
    subset = subset[subset["channel"]==channel]
if campaign != "All":
    subset = subset[subset["campaign"]==campaign]
if creative != "All":
    subset = subset[subset["creative"]==creative]
subset = subset[(subset["date"]>=pd.to_datetime(start_date)) & (subset["date"]<=pd.to_datetime(end_date))]

# handle empty subset
if subset.empty:
    st.warning("No data for the selected filters/date range. Adjust filters.")
    st.stop()

# compute current and previous period (same length) for PoP
curr_start, curr_end = pd.to_datetime(start_date), pd.to_datetime(end_date)
period_days = (curr_end - curr_start).days + 1
prev_end = curr_start - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=period_days-1)
cur_kpis = compute_exec_kpis(subset)
prev_subset = df[(df["date"]>=prev_start) & (df["date"]<=prev_end)]
prev_kpis = compute_exec_kpis(prev_subset)

# KPI area — F-PATTERN (Executive)
left, right = st.columns([2,3], gap="large")
with left:
    st.markdown("<div class='kpi-large'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-label'>Total Net Revenue</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-value'>₹{cur_kpis['total_revenue']:,.0f}</div>", unsafe_allow_html=True)
    rev_delta = compute_pop(cur_kpis['total_revenue'], prev_kpis['total_revenue'])
    if rev_delta is None:
        st.markdown("<div class='kpi-delta'>No previous period</div>", unsafe_allow_html=True)
    else:
        sym = "▲" if rev_delta > 0 else "▼"
        color = "#059669" if rev_delta > 0 else "#dc2626"
        st.markdown(f"<div style='color:{color}; font-weight:600'>{sym} {abs(rev_delta)*100:.1f}% vs prev period</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    cols = st.columns(4, gap="medium")
    gpm = cur_kpis["gross_margin"]
    prev_gpm = prev_kpis["gross_margin"]
    gpm_delta = compute_pop(gpm, prev_gpm)
    with cols[0]:
        st.markdown("<div class='kpi-small'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Gross Profit Margin</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{gpm*100:.1f}%</div>" if not np.isnan(gpm) else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        if gpm_delta is not None:
            colc = "#059669" if gpm_delta>0 else "#dc2626"
            st.markdown(f"<div style='color:{colc}'>{'▲' if gpm_delta>0 else '▼'} {abs(gpm_delta)*100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    mer = cur_kpis["mer"]
    prev_mer = prev_kpis["mer"]
    mer_delta = compute_pop(mer, prev_mer)
    with cols[1]:
        st.markdown("<div class='kpi-small'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Marketing Efficiency Ratio (MER)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{mer:.2f}</div>" if not np.isnan(mer) else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        if mer_delta is not None:
            colc = "#059669" if mer_delta>0 else "#dc2626"
            st.markdown(f"<div style='color:{colc}'>{'▲' if mer_delta>0 else '▼'} {abs(mer_delta)*100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    ltv_cac = cur_kpis["ltv_cac"]
    prev_ltv_cac = prev_kpis["ltv_cac"]
    ltv_delta = compute_pop(ltv_cac, prev_ltv_cac)
    with cols[2]:
        st.markdown("<div class='kpi-small'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>LTV : CAC Ratio</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{ltv_cac:.2f}</div>" if not np.isnan(ltv_cac) else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        if ltv_delta is not None:
            colc = "#059669" if ltv_delta>0 else "#dc2626"
            st.markdown(f"<div style='color:{colc}'>{'▲' if ltv_delta>0 else '▼'} {abs(ltv_delta)*100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    profit = cur_kpis["profit"]
    prev_profit = prev_kpis["profit"]
    profit_delta = compute_pop(profit, prev_profit)
    with cols[3]:
        st.markdown("<div class='kpi-small'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Total Profit</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>₹{profit:,.0f}</div>", unsafe_allow_html=True)
        if profit_delta is not None:
            colc = "#059669" if profit_delta>0 else "#dc2626"
            st.markdown(f"<div style='color:{colc}'>{'▲' if profit_delta>0 else '▼'} {abs(profit_delta)*100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Auto insight (semi-dynamic)
insight = []
if rev_delta is not None:
    if rev_delta > 0.05:
        insight.append(f"Revenue rose {rev_delta*100:.1f}% vs previous period — growth momentum.")
    elif rev_delta < -0.05:
        insight.append(f"Revenue fell {abs(rev_delta)*100:.1f}% vs previous period — review channel performance.")
if mer_delta is not None and mer_delta < -0.05:
    insight.append("MER decreased — spend efficiency may be worsening.")
top_channel = subset.groupby("channel")["revenue"].sum().sort_values(ascending=False).index[0]
insight.append(f"Top channel (by revenue) in the selection: {top_channel}.")

st.markdown(f"<div class='insight'><strong>Insights:</strong><br>{'<br>'.join(insight)}</div>", unsafe_allow_html=True)

st.markdown("---")

# TABS: Executive (already), CMO, CFO
tabs = st.tabs(["Executive (Overview)", "CMO View (Marketing)", "CFO View (Finance)"])

# ===== Executive Overview tab =====
with tabs[0]:
    st.subheader("Executive Overview — Revenue & Spend")

    fig_spend_rev = plot_spend_revenue_trend(subset)
    st.plotly_chart(fig_spend_rev, use_container_width=True)

    # Download buttons: export PNG of this chart (if environment supports)
    img_bytes = fig_to_png_bytes(fig_spend_rev)
    if img_bytes is not None:
        st.download_button("⬇ Download Trend PNG", img_bytes, file_name="revenue_spend_trend.png", mime="image/png")
    else:
        st.info("PNG export for chart not available in this environment (requires kaleido).")

    st.markdown("**Customer Acquisition & Retention** — New vs Returning revenue over time (quick view)")
    subset_agg = subset.groupby("date").agg({"revenue":"sum", "new_customers":"sum","returning_customers":"sum"}).reset_index()
    subset_agg["new_rev"] = subset_agg["revenue"] * (subset_agg["new_customers"] / (subset_agg["new_customers"] + subset_agg["returning_customers"] + 1e-9))
    subset_agg["ret_rev"] = subset_agg["revenue"] - subset_agg["new_rev"]
    fig_nr = go.Figure()
    fig_nr.add_trace(go.Scatter(x=subset_agg["date"], y=subset_agg["ret_rev"], stackgroup='one', name='Returning Customers', line=dict(color="#8b5cf6")))
    fig_nr.add_trace(go.Scatter(x=subset_agg["date"], y=subset_agg["new_rev"], stackgroup='one', name='New Customers', line=dict(color=ACCENT_GREEN)))
    fig_nr.update_layout(template="plotly_white", height=300, margin=dict(t=40))
    st.plotly_chart(fig_nr, use_container_width=True)

    # Add report export (PDF)
    report_bytes = create_pdf_bytes(subset, {
        "Total Revenue (₹)": f"{cur_kpis['total_revenue']:,.2f}",
        "Total Spend (₹)": f"{cur_kpis['total_spend']:,.2f}",
        "MER": cur_kpis['mer'],
        "Gross Margin": f"{cur_kpis['gross_margin']*100:.2f}%"
    }, title="Lifesight - Executive Report")
    st.download_button("⬇ Download Dashboard Report (PDF)", report_bytes, file_name="lifesight_dashboard_report.pdf", mime="application/pdf")

# ===== CMO View =====
with tabs[1]:
    st.header("CMO View — Marketing Effectiveness & Diagnostics")

    # --- CMO KPI Scorecards (placed at the top)
    st.subheader("CMO KPIs")
    cmo1, cmo2, cmo3, cmo4 = st.columns(4)
    with cmo1:
        st.metric("Total Impressions", f"{subset['impressions'].sum():,}")
    with cmo2:
        ctr_val = subset['clicks'].sum() / subset['impressions'].sum() if subset['impressions'].sum() > 0 else 0
        st.metric("Click-Through Rate", f"{ctr_val*100:.2f}%")
    with cmo3:
        cpm_val = subset['spend'].sum() / (subset['impressions'].sum()/1000) if subset['impressions'].sum() > 0 else np.nan
        st.metric("Avg. CPM", f"₹{cpm_val:.2f}" if not np.isnan(cpm_val) else "N/A")
    with cmo4:
        cvr_val = subset['conversions'].sum() / subset['clicks'].sum() if subset['clicks'].sum() > 0 else 0
        st.metric("Avg. Conversion Rate", f"{cvr_val*100:.2f}%")

    st.markdown("**ROAS by Channel** — quick comparison to guide budget allocation")
    fig_roas = plot_roas_by_channel(subset)
    st.plotly_chart(fig_roas, use_container_width=True)
    img_roas = fig_to_png_bytes(fig_roas)
    if img_roas is not None:
        st.download_button("⬇ Download ROAS Chart (PNG)", img_roas, file_name="roas_by_channel.png", mime="image/png")

    st.markdown("**Full Marketing Funnel** — identify drop-off points")
    fig_funnel = plot_funnel_bars(subset)
    st.plotly_chart(fig_funnel, use_container_width=True)
    img_funnel = fig_to_png_bytes(fig_funnel)
    if img_funnel is not None:
        st.download_button("⬇ Download Funnel Chart (PNG)", img_funnel, file_name="marketing_funnel.png", mime="image/png")

    st.markdown("**Campaign & Creative Diagnostics** (Top performing rows)")
    diag = subset.groupby(["channel","campaign","ad_set","creative"]).agg({
        "spend":"sum","impressions":"sum","clicks":"sum","conversions":"sum","revenue":"sum"
    }).reset_index()
    diag["ctr"] = (diag["clicks"] / diag["impressions"]).round(4)
    diag["cvr"] = (diag["conversions"] / diag["clicks"]).round(4)
    diag["cpa"] = (diag["spend"] / diag["conversions"]).round(2).replace([np.inf, -np.inf], pd.NA)
    st.dataframe(diag.sort_values("revenue", ascending=False).head(100), use_container_width=True)
    # allow downloading the diagnostics table as CSV
    csv_buf = diag.sort_values("revenue", ascending=False).to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Diagnostics (CSV)", csv_buf, file_name="campaign_diagnostics.csv", mime="text/csv")

# ===== CFO View =====
with tabs[2]:
    st.header("CFO View — Financial Efficiency & Profitability")

    # --- CFO KPI Scorecards (placed at the top)
    st.subheader("CFO KPIs")
    cfo1, cfo2, cfo3, cfo4 = st.columns(4)
    with cfo1:
        # Marketing ROI = (Revenue - Spend) / Spend
        if subset["spend"].sum() > 0:
            roi_val = (subset["revenue"].sum() - subset["spend"].sum()) / subset["spend"].sum()
            st.metric("Marketing ROI", f"{roi_val*100:.2f}%")
        else:
            st.metric("Marketing ROI", "N/A")
    with cfo2:
        if subset["revenue"].sum() > 0:
            gm_val = (subset["revenue"].sum() - subset["cogs"].sum()) / subset["revenue"].sum()
            st.metric("Gross Margin Rate", f"{gm_val*100:.2f}%")
        else:
            st.metric("Gross Margin Rate", "N/A")
    with cfo3:
        aov_val = subset['aov'].mean()
        st.metric("Average Order Value", f"₹{aov_val:.2f}")
    with cfo4:
        refund_rate = subset["returns"].sum() / subset["orders"].sum() if subset["orders"].sum() > 0 else np.nan
        st.metric("Refund & Return Rate", f"{refund_rate*100:.2f}%" if not np.isnan(refund_rate) else "N/A")

    fig_contrib = plot_contribution_waterfall(subset)
    st.plotly_chart(fig_contrib, use_container_width=True)
    img_contrib = fig_to_png_bytes(fig_contrib)
    if img_contrib is not None:
        st.download_button("⬇ Download Contribution Chart (PNG)", img_contrib, file_name="contribution_waterfall.png", mime="image/png")

    st.markdown("**CAC Trend & Cost Efficiency** — monitor CAC vs historical performance")
    fig_cac = plot_cac_trend(subset)
    st.plotly_chart(fig_cac, use_container_width=True)
    img_cac = fig_to_png_bytes(fig_cac)
    if img_cac is not None:
        st.download_button("⬇ Download CAC Chart (PNG)", img_cac, file_name="cac_trend.png", mime="image/png")

    st.markdown("**Cohort LTV (heatmap)** — revenue evolution for acquisition cohorts")
    fig_cohort = cohort_ltv_heatmap(subset, months=6)
    st.plotly_chart(fig_cohort, use_container_width=True)
    img_cohort = fig_to_png_bytes(fig_cohort)
    if img_cohort is not None:
        st.download_button("⬇ Download Cohort Heatmap (PNG)", img_cohort, file_name="cohort_ltv_heatmap.png", mime="image/png")

    st.markdown("**Key financial KPIs**")
    aov = subset["aov"].mean()
    refund_rate = subset["returns"].sum() / subset["orders"].sum() if subset["orders"].sum()>0 else np.nan
    st.metric("Average Order Value (AOV)", f"₹{aov:,.2f}")
    st.metric("Refund / Return Rate", f"{refund_rate*100:.2f}%" if not np.isnan(refund_rate) else "N/A")

st.caption("Dashboard structure, KPI selection and layout follow the Lifesight assignment brief and executive needs.")
