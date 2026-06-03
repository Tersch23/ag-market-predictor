"""
Ag Market Predictor Dashboard
==============================
A unified, locally-run Streamlit application for cross-market agricultural
commodity analysis and prediction using Track 1 data from the Pro-Ag
Data Science Competition.

Run:  streamlit run app.py
"""
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Ag Market Predictor Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar: Data Path Configuration ──
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Syracuse_Orange_logo.svg/120px-Syracuse_Orange_logo.svg.png", width=60)
    st.title("⚙️ Configuration")
    default_path = str(Path(__file__).resolve().parent.parent / "pro_ag_comp" / "ProAg_data")
    current = st.session_state.get("custom_data_root", default_path)
    new_path = st.text_input("Data Directory", value=current,
                             help="Absolute or relative path to the ProAg_data folder.")
    if new_path != current:
        st.session_state["custom_data_root"] = new_path
        st.rerun()

    p = Path(new_path).resolve()
    if p.exists() and p.is_dir():
        st.success(f"✅ Data: {p.name}")
    else:
        st.error(f"❌ Not found: {p}")

    st.divider()

    # System health
    import importlib.util
    checks = {
        "XGBoost": "xgboost",
        "Plotly": "plotly",
        "Scikit-learn": "sklearn",
    }
    for label, mod in checks.items():
        if importlib.util.find_spec(mod):
            st.caption(f"✅ {label}")
        else:
            st.caption(f"❌ {label} missing")

    # FRED API key (optional — for external macro data)
    st.divider()
    st.markdown("**📡 External Data (FRED)**")
    fred_key = st.text_input(
        "FRED API Key",
        value=st.session_state.get("fred_api_key", ""),
        type="password",
        help="Free key from https://fred.stlouisfed.org/docs/api/api_key.html — enables corn prices, oil, interest rates, etc.",
    )
    if fred_key:
        st.session_state["fred_api_key"] = fred_key
        st.caption("✅ FRED connected")
    else:
        st.caption("ℹ️ Optional — adds macro indicators")

    st.divider()
    st.caption("All processing runs 100% locally.")
    st.caption("Syracuse University — CCDS Pro-Ag Competition 2026")

# ── Header ──
st.title("🌾 Ag Market Predictor Dashboard")
st.markdown(
    "Cross-market agricultural commodity analysis and prediction engine. "
    "Select a tab below to explore Track 1 data, discover correlations, "
    "train predictive models, and generate AI-powered market briefings."
)

# ── Tabs ──
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Market Overview",
    "💵 Cash & Cutout",
    "📈 Futures",
    "🔬 Correlations",
    "🧠 Predictive Engine",
    "📝 AI Briefing",
    "🔧 Debug",
])

from tabs import tab_overview, tab_cash_cutout, tab_futures, tab_correlations, tab_predictor, tab_briefing, tab_debug

with tab1:
    tab_overview.render()
with tab2:
    tab_cash_cutout.render()
with tab3:
    tab_futures.render()
with tab4:
    tab_correlations.render()
with tab5:
    tab_predictor.render()
with tab6:
    tab_briefing.render()
with tab7:
    tab_debug.render()

