"""
Centralized constants and configuration for the Ag Market Predictor Dashboard.
"""
from pathlib import Path
import streamlit as st

# ─────────────────────────────────────────────
# Color Palette — Professional Agricultural Theme
# ─────────────────────────────────────────────
BULLISH_COLOR = "#16a34a"   # Green — price up
BEARISH_COLOR = "#dc2626"   # Red — price down
NEUTRAL_COLOR = "#6b7280"   # Gray — flat / baseline
PRIMARY_COLOR = "#2563eb"   # Blue — primary accent
SECONDARY_COLOR = "#f59e0b" # Amber — secondary accent
ACCENT_COLOR = "#8b5cf6"    # Violet — highlights

CHART_THEME = {
    "primary": PRIMARY_COLOR,
    "secondary": SECONDARY_COLOR,
    "bullish": BULLISH_COLOR,
    "bearish": BEARISH_COLOR,
    "neutral": NEUTRAL_COLOR,
    "accent": ACCENT_COLOR,
}

# Plotly-compatible sequential palette for multi-series charts
PALETTE = [
    "#2563eb", "#f59e0b", "#16a34a", "#dc2626",
    "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16",
    "#f97316", "#14b8a6",
]

# ─────────────────────────────────────────────
# Dataset Catalog — human-readable grouping of Track 1 files
# ─────────────────────────────────────────────
DATASET_GROUPS = {
    "Cash Markets": [
        "Cash Cattle.csv",
        "Western Cornbelt.csv",
        "Western Colt.csv",
        "National.csv",
        "IASWMN.csv",
        "National SOW Prices.csv",
        "Isowean Cash Market Volume.csv",
    ],
    "Futures & Derivatives": [
        "Nearby Futures.csv",
        "lookup_CME.csv",
        "LRP Quotes.csv",
        "LRP Quotes Futures.csv",
    ],
    "Cutout & Primal Values": [
        "Cutout (Select_Choice).csv",
        "Cattle Primal Values.csv",
        "Pork Primal Values.csv",
    ],
    "Production & Harvest": [
        "Beef Production.csv",
        "Pork Production.csv",
        "Fed Cattle.csv",
        "Cow Harvest.csv",
        "SOW Harvest.csv",
        "Harvest - USDA.csv",
        "Harvest 2.csv",
        "Harvest 3.csv",
        "Historical Harvest.csv",
        "Weekly Harvest.csv",
        "Carcass Weights.csv",
    ],
    "Supply, Demand & Trade": [
        "WASDE.csv",
        "Sales.csv",
        "Indexes.csv",
    ],
    "Reference": [
        "EndPointList.csv",
    ],
}

# Flat list of all track 1 CSV filenames
ALL_TRACK1_FILES = [f for group in DATASET_GROUPS.values() for f in group]

# ─────────────────────────────────────────────
# Data Root — dynamic, with sidebar override
# ─────────────────────────────────────────────
_DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "pro_ag_comp" / "ProAg_data"

def get_data_root() -> Path:
    """Return the active data root, respecting any sidebar override."""
    try:
        if "custom_data_root" in st.session_state and st.session_state["custom_data_root"]:
            p = Path(st.session_state["custom_data_root"]).resolve()
            if p.exists():
                return p
    except Exception:
        pass
    return _DEFAULT_DATA_PATH.resolve()
