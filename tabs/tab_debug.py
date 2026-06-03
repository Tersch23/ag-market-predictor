"""Tab 7: Debug — system diagnostics, data health, and FRED API connectivity."""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import importlib.util
from pathlib import Path
from datetime import datetime


def render():
    st.header("🔧 Debug & Diagnostics")
    st.markdown(
        "System health checks, data directory scan, FRED API connectivity test, "
        "and merge diagnostics. Use this panel to troubleshoot issues."
    )

    _render_system_info()
    st.divider()
    _render_data_health()
    st.divider()
    _render_fred_diagnostics()
    st.divider()
    _render_merge_diagnostics()


def _render_system_info():
    st.subheader("💻 System Info")
    cols = st.columns(3)
    cols[0].metric("Python", sys.version.split()[0])
    cols[1].metric("Platform", sys.platform)
    cols[2].metric("Timestamp", datetime.now().strftime("%H:%M:%S"))

    # Package versions
    packages = {
        "streamlit": "streamlit",
        "pandas": "pandas",
        "numpy": "numpy",
        "xgboost": "xgboost",
        "scikit-learn": "sklearn",
        "plotly": "plotly",
        "fredapi": "fredapi",
        "openpyxl": "openpyxl",
    }
    rows = []
    for label, mod_name in packages.items():
        spec = importlib.util.find_spec(mod_name)
        if spec:
            try:
                mod = __import__(mod_name)
                version = getattr(mod, "__version__", "✅ installed")
            except Exception:
                version = "✅ installed"
            rows.append({"Package": label, "Version": version, "Status": "✅"})
        else:
            rows.append({"Package": label, "Version": "—", "Status": "❌ Missing"})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_data_health():
    st.subheader("📂 Data Directory Health")

    from shared.constants import get_data_root
    data_root = get_data_root()

    if not data_root.exists():
        st.error(f"❌ Data root NOT found: `{data_root}`")
        return

    st.success(f"✅ Data root: `{data_root}`")

    # Scan all CSV/XLSX files
    files = []
    for fp in sorted(data_root.iterdir()):
        if fp.is_file() and not fp.name.startswith("."):
            size_mb = fp.stat().st_size / (1024 * 1024)
            if fp.suffix in (".csv", ".xlsx"):
                try:
                    if fp.suffix == ".csv":
                        df = pd.read_csv(str(fp), low_memory=False, nrows=1)
                    else:
                        df = pd.read_excel(str(fp), nrows=1)
                    ncols = len(df.columns)
                    # Quick full row count
                    if fp.suffix == ".csv":
                        full_rows = sum(1 for _ in open(str(fp))) - 1
                    else:
                        full_rows = "?"
                except Exception:
                    ncols = "?"
                    full_rows = "?"
                files.append({
                    "File": fp.name,
                    "Size (MB)": f"{size_mb:.2f}",
                    "Rows": full_rows,
                    "Columns": ncols,
                    "Status": "✅"
                })
        elif fp.is_dir() and not fp.name.startswith("."):
            sub_files = list(fp.glob("*.xlsx")) + list(fp.glob("*.csv"))
            files.append({
                "File": f"📁 {fp.name}/",
                "Size (MB)": "—",
                "Rows": f"{len(sub_files)} files",
                "Columns": "—",
                "Status": "✅"
            })

    if files:
        st.dataframe(pd.DataFrame(files), use_container_width=True, hide_index=True)
    else:
        st.warning("No data files found.")


def _render_fred_diagnostics():
    st.subheader("📡 FRED API Connectivity")

    # Check if fredapi is installed
    if not importlib.util.find_spec("fredapi"):
        st.error("❌ `fredapi` package not installed. Run: `pip install fredapi`")
        return

    st.caption("✅ fredapi package installed")

    # Check API key
    fred_key = st.session_state.get("fred_api_key", "")
    if not fred_key:
        fred_key = os.environ.get("FRED_API_KEY", "")

    if not fred_key:
        st.warning("⚠️ No FRED API key configured. Enter one in the sidebar or set `FRED_API_KEY` environment variable.")
        st.info("Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    st.caption("✅ FRED API key found")

    # Live connectivity test
    if st.button("🧪 Test FRED API Connection", type="primary"):
        from shared.fred_loader import FRED_SERIES, FRED_DESCRIPTIONS
        from fredapi import Fred

        with st.spinner("Testing FRED API…"):
            try:
                fred = Fred(api_key=fred_key)
                results = []
                for name, sid in FRED_SERIES.items():
                    try:
                        import time
                        t0 = time.time()
                        data = fred.get_series(sid, observation_start="2024-01-01")
                        latency = (time.time() - t0) * 1000
                        if data is not None and not data.empty:
                            latest = data.dropna().iloc[-1]
                            latest_date = data.dropna().index[-1]
                            results.append({
                                "Series": name,
                                "FRED ID": sid,
                                "Latest Value": f"{latest:.4f}",
                                "Latest Date": str(latest_date.date()),
                                "Latency": f"{latency:.0f}ms",
                                "Status": "✅",
                            })
                        else:
                            results.append({
                                "Series": name, "FRED ID": sid,
                                "Latest Value": "—", "Latest Date": "—",
                                "Latency": "—", "Status": "⚠️ Empty",
                            })
                    except Exception as e:
                        results.append({
                            "Series": name, "FRED ID": sid,
                            "Latest Value": "—", "Latest Date": "—",
                            "Latency": "—", "Status": f"❌ {e}",
                        })

                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

                passes = sum(1 for r in results if r["Status"] == "✅")
                if passes == len(results):
                    st.success(f"✅ All {len(results)} FRED series fetched successfully!")
                else:
                    st.warning(f"⚠️ {passes}/{len(results)} series OK")

            except Exception as e:
                st.error(f"❌ FRED API connection failed: {e}")


def _render_merge_diagnostics():
    st.subheader("🔗 Cross-Market Merge Diagnostics")

    from shared.constants import get_data_root
    from shared.data_loader import load_csv
    from shared.analytics import align_datasets_on_date

    data_root = get_data_root()
    if not data_root.exists():
        st.error("Data root not found."); return

    test_files = [
        "Cash Cattle.csv", "Cutout (Select_Choice).csv", "Nearby Futures.csv",
        "Beef Production.csv", "Carcass Weights.csv", "Weekly Harvest.csv",
    ]

    available = [fn for fn in test_files if (data_root / fn).exists()]

    if st.button("🧪 Run Merge Diagnostic"):
        with st.spinner("Loading and merging datasets…"):
            # Individual dataset info
            ds_info = []
            dfs = []
            for fn in available:
                df = load_csv(str(data_root / fn))
                if df.empty:
                    ds_info.append({"File": fn, "Rows": 0, "Num Cols": 0,
                                    "Date Range": "—", "All-NaN Cols": "—"})
                    continue
                date_cols = [c for c in df.columns if "date" in str(c).lower()]
                date_range = "—"
                if date_cols:
                    dates = pd.to_datetime(df[date_cols[0]], errors="coerce").dropna()
                    if not dates.empty:
                        date_range = f"{dates.min().date()} → {dates.max().date()}"

                num_cols = df.select_dtypes(include=[np.number]).columns
                all_nan = [c for c in num_cols if df[c].isna().all()]

                ds_info.append({
                    "File": fn,
                    "Rows": f"{len(df):,}",
                    "Num Cols": len(num_cols),
                    "Date Range": date_range,
                    "All-NaN Cols": ", ".join(all_nan) if all_nan else "None ✅",
                })
                dfs.append((fn, df))

            st.markdown("**Individual Datasets:**")
            st.dataframe(pd.DataFrame(ds_info), use_container_width=True, hide_index=True)

            # Merge result
            if len(dfs) >= 2:
                merged = align_datasets_on_date(dfs)
                if merged.empty:
                    st.error(f"❌ Merge produced 0 rows!")
                else:
                    st.success(f"✅ Merged {len(dfs)} datasets → {len(merged):,} rows × {merged.shape[1]} columns")
                    st.caption(f"Date range: {merged['Date_Index'].min()} → {merged['Date_Index'].max()}")
                    with st.expander("Preview merged data"):
                        st.dataframe(merged.head(20), use_container_width=True)
