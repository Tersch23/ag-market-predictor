"""Tab 4: Cross-Market Correlations — heatmaps and lead-lag relationships across datasets."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from shared.data_loader import load_csv
from shared.constants import get_data_root, DATASET_GROUPS, PALETTE
from shared.analytics import align_datasets_on_date, compute_static_correlation, sanitize_columns


# Key datasets with known usable numeric + date columns for correlation analysis
_CORR_CANDIDATES = {
    "Cash Cattle": ("Cash Cattle.csv", "weighted_avg_price"),
    "Cutout Choice/Select": ("Cutout (Select_Choice).csv", "wtd_avg"),
    "Nearby Futures": ("Nearby Futures.csv", "price_5day"),
    "Beef Production": ("Beef Production.csv", None),
    "Pork Production": ("Pork Production.csv", None),
    "Fed Cattle": ("Fed Cattle.csv", None),
    "Carcass Weights": ("Carcass Weights.csv", "avg_carcass_weight"),
    "Weekly Harvest": ("Weekly Harvest.csv", "slaughter"),
    "Indexes": ("Indexes.csv", None),
}


def render():
    st.header("🔬 Cross-Market Correlations")
    st.markdown(
        "Discover how different agricultural markets move together. "
        "Select datasets to merge on date and visualize their pairwise correlations."
    )

    data_root = get_data_root()

    # Let users pick datasets to correlate
    available = []
    for label, (fname, _) in _CORR_CANDIDATES.items():
        if (data_root / fname).exists():
            available.append(label)

    if len(available) < 2:
        st.warning("Need at least 2 available datasets for correlation analysis.")
        return

    selected = st.multiselect(
        "Select datasets to compare (pick 2–6 for best results):",
        available,
        default=available[:4],
    )

    if len(selected) < 2:
        st.info("Select at least 2 datasets.")
        return

    # Load and merge
    dfs_to_merge = []
    for label in selected:
        fname, _ = _CORR_CANDIDATES[label]
        fp = data_root / fname
        df = load_csv(str(fp))
        if not df.empty:
            dfs_to_merge.append((label, df))

    if len(dfs_to_merge) < 2:
        st.error("Could not load enough datasets.")
        return

    with st.spinner("Aligning datasets on date…"):
        merged = align_datasets_on_date(dfs_to_merge)

    if merged.empty or merged.shape[1] < 3:
        st.error("Not enough overlapping data to compute correlations.")
        return

    st.success(f"Merged {len(dfs_to_merge)} datasets → {len(merged):,} aligned rows, {merged.shape[1]-1} numeric features.")

    # Remove data leakage columns (IndexbyYear, RankbyAttribute, redundant futures)
    merged = sanitize_columns(merged)
    st.caption("ℹ️ Calendar artifacts (IndexbyYear, RankbyAttribute) and redundant columns removed automatically.")

    # Correlation heatmap
    st.subheader("Pearson Correlation Heatmap")
    corr = compute_static_correlation(merged.drop(columns=["Date_Index"], errors="ignore"))

    if corr.empty:
        st.warning("Cannot compute correlations.")
        return

    # Truncate long column names for readability
    short = {c: (c[:35] + "…" if len(c) > 35 else c) for c in corr.columns}
    corr_display = corr.rename(columns=short, index=short)

    fig = px.imshow(
        corr_display,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Pairwise Correlation Matrix",
        aspect="auto",
    )
    fig.update_layout(height=max(500, 40 * len(corr_display)))
    st.plotly_chart(fig, use_container_width=True)

    # Top correlated pairs
    st.subheader("Strongest Cross-Market Relationships")
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({"Feature A": cols[i], "Feature B": cols[j], "Correlation": corr.iloc[i, j]})
    pairs_df = pd.DataFrame(pairs).dropna()
    pairs_df["Abs Corr"] = pairs_df["Correlation"].abs()
    pairs_df = pairs_df.sort_values("Abs Corr", ascending=False).head(15)

    st.dataframe(
        pairs_df[["Feature A", "Feature B", "Correlation"]].reset_index(drop=True),
        use_container_width=True,
    )

    # Time series overlay of top pair
    if not pairs_df.empty:
        st.subheader("Top Correlated Pair — Time Series Overlay")
        top = pairs_df.iloc[0]
        fa, fb = top["Feature A"], top["Feature B"]
        if fa in merged.columns and fb in merged.columns:
            overlay = merged[["Date_Index", fa, fb]].copy()
            overlay["Date_Index"] = pd.to_datetime(overlay["Date_Index"])
            fig2 = px.line(overlay, x="Date_Index", y=[fa, fb],
                           title=f"{fa}  vs  {fb}  (r = {top['Correlation']:.3f})",
                           color_discrete_sequence=PALETTE[:2])
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               yaxis_title="Value", legend_title="")
            st.plotly_chart(fig2, use_container_width=True)
