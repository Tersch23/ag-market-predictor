"""Tab 1: Market Overview — Data manifest and explorer for all Track 1 datasets."""
import streamlit as st
import pandas as pd
from shared.data_loader import discover_track1_files, load_csv, load_xlsx, get_column_info
from shared.constants import get_data_root, DATASET_GROUPS


def render():
    st.header("📊 Market Overview")
    st.markdown("A top-down view of all available Track 1 datasets, their sizes, and contents.")

    data_root = get_data_root()
    manifest = discover_track1_files(str(data_root))

    if manifest.empty:
        st.error(f"No data files found at `{data_root}`. Update the data path in the sidebar.")
        return

    # ── Summary metrics ──
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Datasets", len(manifest))
    c2.metric("Total Size", f"{manifest['size_mb'].sum():.1f} MB")
    c3.metric("File Types", ", ".join(manifest["extension"].unique()))

    # ── Grouped catalog ──
    st.subheader("Dataset Catalog")
    for group_name, files in DATASET_GROUPS.items():
        found = manifest[manifest["filename"].isin(files)]
        if found.empty:
            continue
        with st.expander(f"**{group_name}** ({len(found)} files)", expanded=False):
            st.dataframe(
                found[["filename", "category", "size_mb", "extension"]],
                use_container_width=True, hide_index=True,
            )

    st.divider()

    # ── Interactive data explorer ──
    st.subheader("Data Explorer")
    selected_file = st.selectbox("Select a dataset to preview:", manifest["filename"].tolist())

    if selected_file:
        file_info = manifest[manifest["filename"] == selected_file].iloc[0]
        fp = file_info["path"]
        df = load_csv(fp) if file_info["extension"] == ".csv" else load_xlsx(fp)

        if not df.empty:
            col_l, col_r = st.columns([3, 1])
            with col_l:
                st.dataframe(df.head(100), use_container_width=True)
            with col_r:
                st.markdown(f"**Rows:** {len(df):,}")
                st.markdown(f"**Columns:** {len(df.columns)}")
                info = get_column_info(df)
                if info["date_cols"]:
                    st.markdown("**Date Columns:**")
                    for c in info["date_cols"]:
                        st.markdown(f"- `{c}`")
                if info["numeric_cols"]:
                    st.markdown(f"**Numeric Columns:** {len(info['numeric_cols'])}")

                # Quick summary stats
                st.markdown("**Quick Stats (numeric):**")
                st.dataframe(df.describe().T[["mean", "min", "max"]].head(8), use_container_width=True)
        else:
            st.warning("Could not load this dataset.")
