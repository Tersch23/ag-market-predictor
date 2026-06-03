"""Tab 3: Futures — nearby futures trends and forward contract curves."""
import streamlit as st
import pandas as pd
import plotly.express as px
from shared.data_loader import load_csv, load_xlsx
from shared.constants import get_data_root, CHART_THEME, PALETTE


def render():
    st.header("📈 Futures")
    data_root = get_data_root()

    sub = st.radio("Select view:", ["Nearby Continuous", "Contract Curves"], horizontal=True)

    if sub == "Nearby Continuous":
        _render_nearby(data_root)
    else:
        _render_contracts(data_root)


def _render_nearby(data_root):
    fp = data_root / "Nearby Futures.csv"
    if not fp.exists():
        st.error("Nearby Futures.csv not found."); return
    df = load_csv(str(fp))
    if df.empty:
        return

    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

    regions = df["Name"].dropna().unique().tolist() if "Name" in df.columns else []
    selected = st.selectbox("Region", regions, index=0) if regions else None

    if selected:
        plot_df = df[df["Name"] == selected].sort_values("report_date")
        price_col = "price_5day" if "price_5day" in plot_df.columns else "wtd_avg"

        if price_col in plot_df.columns:
            fig = px.line(plot_df, x="report_date", y=price_col,
                          title=f"Nearby Futures — {selected}",
                          color_discrete_sequence=[CHART_THEME["secondary"]])
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Data"):
                st.dataframe(plot_df.tail(100), use_container_width=True)


def _render_contracts(data_root):
    st.subheader("Forward Contract Curves")
    c1, c2 = st.columns(2)
    with c1:
        commodity = st.selectbox("Commodity", ["Live Cattle", "Hog", "Corn", "Soybeans"])
    with c2:
        year = st.selectbox("Year", ["2025"])

    dir_path = data_root / f"{commodity} {year} Futures"
    if not dir_path.exists() or not dir_path.is_dir():
        st.warning(f"No contract data found for {commodity} {year}."); return

    files = list(dir_path.glob("*.xlsx"))
    if not files:
        st.warning("No .xlsx files in this directory."); return

    dfs = []
    for f in files:
        try:
            raw = pd.read_excel(str(f))
            # These xlsx files have headers in row 0 of data (Unnamed: columns)
            # Detect this and fix it
            if any("Unnamed" in str(c) for c in raw.columns):
                raw.columns = raw.iloc[0].astype(str)
                raw = raw.iloc[1:].reset_index(drop=True)
            # Convert numeric columns
            for col in ["Open", "High", "Low", "Close", "Volume", "Open Interest"]:
                if col in raw.columns:
                    raw[col] = pd.to_numeric(raw[col], errors="coerce")
            # Parse date from 'Time' column
            if "Time" in raw.columns:
                raw["Date"] = pd.to_datetime(raw["Time"], errors="coerce")
            raw["Contract"] = f.stem.split(" ")[0]
            dfs.append(raw)
        except Exception:
            continue

    if not dfs:
        st.warning("Could not parse any contract files."); return
    combined = pd.concat(dfs, ignore_index=True)

    # Find date and price columns
    if "Date" not in combined.columns:
        date_cols = [c for c in combined.columns if "date" in str(c).lower() or "time" in str(c).lower()]
        if date_cols:
            combined["Date"] = pd.to_datetime(combined[date_cols[0]], errors="coerce")

    price_col = "Close" if "Close" in combined.columns else None
    if not price_col:
        price_cols = [c for c in combined.columns if str(c).lower() in ["close", "settle", "last"]]
        price_col = price_cols[0] if price_cols else None
    if not price_col:
        num_cols = combined.select_dtypes("number").columns
        price_col = num_cols[-1] if len(num_cols) else None

    if "Date" in combined.columns and price_col:
        combined = combined.dropna(subset=["Date", price_col]).sort_values("Date")

        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Contracts", combined["Contract"].nunique())
        m2.metric("Total Records", f"{len(combined):,}")
        latest = combined.sort_values("Date").groupby("Contract")[price_col].last()
        m3.metric("Avg Latest Price", f"${latest.mean():.2f}")

        fig = px.line(combined, x="Date", y=price_col, color="Contract",
                      title=f"{commodity} {year} Futures Contracts",
                      color_discrete_sequence=PALETTE)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Contract Data"):
            st.dataframe(combined.head(100), use_container_width=True)
    else:
        st.warning("Could not detect Date and Price columns in the contract data.")
