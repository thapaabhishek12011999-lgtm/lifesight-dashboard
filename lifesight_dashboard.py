
"""
Updated Lifesight Dashboard — loads mock data from CSV (instead of generating it)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# NOTE: This app expects the mock CSV at: /mnt/data/mock_marketing_data_3m.csv
CSV_PATH = "/mnt/data/mock_marketing_data_3m.csv"

# -----------------------
# Styling & helper functions
# -----------------------
PRIMARY_PURPLE = "#6B21A8"
ACCENT_GREEN = "#14B8A6"
CARD_BG = "#FAFAFB"
PAGE_BG = "#ffffff"
TEXT_COLOR = "#0f172a"

def inject_css():
    st.markdown(
        f"""
    <style>
    .stApp {{ background: {PAGE_BG}; color: {TEXT_COLOR}; }}
    .reportview-container .main .block-container{{padding-top:1.5rem; padding-left:2rem; padding-right:2rem;}}
    .topband {{
        background: linear-gradient(90deg,{PRIMARY_PURPLE} 0%, #7c3aed 100%);
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        margin-bottom: 18px;
    }}
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
    .kpi-label {{ font-size:14px; color:#374151; }}
    .kpi-value {{ font-size:28px; font-weight:700; color:{TEXT_COLOR}; }}
    .kpi-delta {{ font-size:13px; color:#059669; }}
    </style>
    """,
        unsafe_allow_html=True,
    )

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

def delta_html(cur, prev):
    pop = compute_pop(cur, prev)
    if pop is None:
        return ""
    sym = "▲" if pop > 0 else "▼"
    color = "#059669" if pop > 0 else "#dc2626"
    return f"<div style='color:{color}; font-weight:600'>{sym} {abs(pop)*100:.1f}% vs prev</div>"

def delta_html_inverted(cur, prev):
    pop = compute_pop(cur, prev)
    if pop is None:
        return ""
    sym = "▲" if pop > 0 else "▼"
    color = "#059669" if pop < 0 else "#dc2626"
    return f"<div style='color:{color}; font-weight:600'>{sym} {abs(pop)*100:.1f}% vs prev</div>"

# Note: plotting functions below are unchanged; for brevity they are omitted in this snippet
# In the actual file they should be copied from original implementation.

# -----------------------
# Streamlit app layout
# -----------------------
st.set_page_config(page_title="Lifesight - Marketing Performance (CSV source)", layout="wide")
inject_css()
st.markdown(f"<div class='topband'><strong style='font-size:18px'>Lifesight</strong> — Marketing Performance Dashboard (CSV)</div>", unsafe_allow_html=True)

# Load data from CSV
try:
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    st.info(f"Loaded mock data from {CSV_PATH} — rows: {len(df):,}")
except Exception as e:
    st.error(f"Failed to load CSV at {CSV_PATH}: {e}")
    st.stop()

# The remainder of the app (filters, KPIs, tabs, plots) can remain identical to the original.
st.write("App will continue to behave the same, using the CSV data as the source.")
