"""Tab 2: Cash & Cutout Markets — price trends, quality spreads, anomaly flags."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from shared.data_loader import load_csv
from shared.constants import get_data_root, CHART_THEME, PALETTE
from shared.analytics import zscore_anomalies


def render():
    st.header("💵 Cash & Cutout Markets")
    data_root = get_data_root()

    sub = st.radio("Select view:", ["Cash Cattle Prices", "Cutout & Quality Spreads", "Anomaly Scanner"], horizontal=True)

    if sub == "Cash Cattle Prices":
        _render_cash(data_root)
    elif sub == "Cutout & Quality Spreads":
        _render_cutout(data_root)
    else:
        _render_anomaly(data_root)


def _render_cash(data_root):
    fp = data_root / "Cash Cattle.csv"
    if not fp.exists():
        st.error("Cash Cattle.csv not found."); return
    df = load_csv(str(fp))
    if df.empty:
        st.error("Empty dataset."); return

    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

    # Sidebar-like filters in expander
    with st.expander("🔧 Filters", expanded=True):
        fc1, fc2 = st.columns(2)
        with fc1:
            if "selling_basis_description" in df.columns:
                opts = df["selling_basis_description"].dropna().unique().tolist()
                sel = st.multiselect("Selling Basis", opts, default=opts[:1] if opts else [])
                if sel:
                    df = df[df["selling_basis_description"].isin(sel)]
        with fc2:
            if "grade_description" in df.columns:
                opts = df["grade_description"].dropna().unique().tolist()
                sel = st.multiselect("Grade", opts, default=opts[:1] if opts else [])
                if sel:
                    df = df[df["grade_description"].isin(sel)]

    if "report_date" not in df.columns or "weighted_avg_price" not in df.columns:
        st.warning("Expected columns not found."); return

    df = df.sort_values("report_date")
    trend = df.groupby("report_date")["weighted_avg_price"].mean().reset_index()

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Records", f"{len(df):,}")
    if len(trend) > 1:
        latest = trend["weighted_avg_price"].iloc[-1]
        prev = trend["weighted_avg_price"].iloc[-min(8, len(trend))]
        m2.metric("Latest Avg Price", f"${latest:.2f}")
        m3.metric("Period Change", f"${latest - prev:+.2f}")

    fig = px.line(trend, x="report_date", y="weighted_avg_price",
                  title="Cash Cattle — Weighted Average Price ($/cwt)",
                  color_discrete_sequence=[CHART_THEME["primary"]])
    fig.update_layout(xaxis_title="Date", yaxis_title="Price ($)",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw Data"):
        st.dataframe(df.tail(200), use_container_width=True)


def _render_cutout(data_root):
    fp = data_root / "Cutout (Select_Choice).csv"
    if not fp.exists():
        st.error("Cutout data not found."); return
    df = load_csv(str(fp))
    if df.empty:
        st.error("Empty dataset."); return

    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    if "Attribute" in df.columns and "Value" in df.columns:
        df = df.rename(columns={"Attribute": "grade_description", "Value": "wtd_avg"})

    if "grade_description" not in df.columns or "wtd_avg" not in df.columns:
        st.warning("Expected cutout columns not found."); return

    df = df.sort_values("report_date")
    pivot = df.pivot_table(index="report_date", columns="grade_description", values="wtd_avg", aggfunc="mean").reset_index()
    choice_col = next((c for c in pivot.columns if "choice" in str(c).lower()), None)
    select_col = next((c for c in pivot.columns if "select" in str(c).lower()), None)

    if choice_col and select_col:
        pivot["Choice-Select Spread"] = pivot[choice_col] - pivot[select_col]
        m1, m2, m3 = st.columns(3)
        latest = pivot.iloc[-1]
        m1.metric(f"Latest {choice_col}", f"${latest[choice_col]:.2f}")
        m2.metric(f"Latest {select_col}", f"${latest[select_col]:.2f}")
        m3.metric("Spread", f"${latest['Choice-Select Spread']:.2f}")

    # Trend
    fig = px.line(df, x="report_date", y="wtd_avg", color="grade_description",
                  title="Cutout Value Over Time",
                  color_discrete_sequence=PALETTE)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    if choice_col and select_col:
        fig2 = px.area(pivot, x="report_date", y="Choice-Select Spread",
                       title="Choice–Select Spread ($/cwt)",
                       color_discrete_sequence=[CHART_THEME["accent"]])
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)


def _render_anomaly(data_root):
    fp = data_root / "Cash Cattle.csv"
    if not fp.exists():
        st.error("Cash Cattle.csv not found."); return
    df = load_csv(str(fp))
    if df.empty:
        return

    date_col = next((c for c in df.columns if "date" in str(c).lower()), None)
    price_col = next((c for c in df.columns if "price" in str(c).lower()), None)
    if not date_col or not price_col:
        st.warning("Cannot detect date/price columns."); return

    df = df.sort_values(date_col).dropna(subset=[price_col])
    anomaly_df = zscore_anomalies(df[price_col].reset_index(drop=True))

    latest = anomaly_df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Price", f"${latest['value']:.2f}")
    c2.metric("30-Day Mean", f"${latest['rolling_mean']:.2f}")
    c3.metric("Z-Score", f"{latest['z_score']:+.2f}")

    if latest["z_score"] > 2:
        st.error(f"🚨 **ANOMALY:** Price is {latest['z_score']:.2f}σ above the 30-day average — potentially overbought.")
    elif latest["z_score"] < -2:
        st.error(f"🚨 **ANOMALY:** Price is {latest['z_score']:.2f}σ below the 30-day average — potentially oversold.")
    else:
        st.success("✅ Price is within normal historical ranges.")

    chart_df = pd.DataFrame({"Price": anomaly_df["value"].values, "30d Mean": anomaly_df["rolling_mean"].values})
    st.line_chart(chart_df.tail(200))
