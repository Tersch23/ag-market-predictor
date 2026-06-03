"""
FRED (Federal Reserve Economic Data) integration for the Ag Market Predictor.

Why FRED?
---------
FRED is the authoritative, free, programmatic source for U.S. macroeconomic data
maintained by the Federal Reserve Bank of St. Louis.  Unlike other providers
(EIA, BLS, NOAA), FRED aggregates data from multiple agencies into a single
consistent API, reducing integration complexity from N APIs to one.

Key series for agricultural commodity prediction:
  - DCOILWTICO  : WTI crude oil price (transportation/energy costs)
  - FEDFUNDS    : Federal funds effective rate (cost of capital, hedging)
  - CPIAUCSL    : Consumer Price Index (downstream demand signal)
  - WPU0223     : PPI - Corn (direct feed cost for cattle/hog producers)
  - PCOTTINDUSDM: Cotton price (alternative ag benchmark)

Usage:
  Requires a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
  Pass the key via Streamlit sidebar or set FRED_API_KEY env variable.
"""
import pandas as pd
import numpy as np
import streamlit as st
import os
from datetime import datetime, timedelta

# FRED series relevant to agricultural commodity markets
FRED_SERIES = {
    "Corn Price (PPI)": "WPU0223",
    "WTI Crude Oil": "DCOILWTICO",
    "Fed Funds Rate": "FEDFUNDS",
    "CPI (All Urban)": "CPIAUCSL",
    "USD Index (Broad)": "DTWEXBGS",
}

FRED_DESCRIPTIONS = {
    "Corn Price (PPI)": "Producer Price Index for corn — the single largest input cost for cattle and hog producers (feed = 60–70% of production cost). Rising corn prices squeeze margins and ultimately reduce supply.",
    "WTI Crude Oil": "West Texas Intermediate crude oil — a proxy for transportation and energy costs that directly affect the cash-futures basis and delivery economics.",
    "Fed Funds Rate": "The federal funds effective rate — drives the cost of capital for hedging and forward contracting. Rate hikes increase carry costs for futures positions.",
    "CPI (All Urban)": "Consumer Price Index — a downstream demand signal. When consumer prices rise broadly, retail beef/pork demand can soften, eventually feeding back into wholesale prices.",
    "USD Index (Broad)": "Trade-weighted U.S. dollar index — a strong dollar makes U.S. beef exports more expensive, reducing export demand and depressing domestic prices.",
}


@st.cache_data(ttl=3600, show_spinner="Fetching FRED data…")
def fetch_fred_series(series_id: str, api_key: str, start: str = "2000-01-01") -> pd.DataFrame:
    """Fetch a single FRED series and return as a DataFrame with Date_Index."""
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id, observation_start=start)
        if data is None or data.empty:
            return pd.DataFrame()
        df = data.reset_index()
        df.columns = ["Date_Index", series_id]
        df["Date_Index"] = pd.to_datetime(df["Date_Index"]).dt.date
        df = df.dropna()
        return df
    except Exception as e:
        st.warning(f"Could not fetch FRED series {series_id}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="Fetching FRED macro data…")
def fetch_multiple_fred(selected_names: list, api_key: str) -> pd.DataFrame:
    """
    Fetch multiple FRED series and merge them on date.
    Returns a single DataFrame with Date_Index and one column per series.
    """
    from fredapi import Fred
    fred = Fred(api_key=api_key)
    merged = pd.DataFrame()

    for name in selected_names:
        sid = FRED_SERIES.get(name)
        if not sid:
            continue
        try:
            data = fred.get_series(sid, observation_start="2000-01-01")
            if data is None or data.empty:
                continue
            df = data.reset_index()
            df.columns = ["Date_Index", f"{name} (FRED)"]
            df["Date_Index"] = pd.to_datetime(df["Date_Index"]).dt.date
            df = df.dropna()

            if merged.empty:
                merged = df
            else:
                merged = pd.merge(merged, df, on="Date_Index", how="outer")
        except Exception:
            continue

    if not merged.empty:
        merged = merged.sort_values("Date_Index").ffill().dropna()
    return merged


def get_fred_api_key() -> str:
    """Get the FRED API key from session state or environment."""
    key = st.session_state.get("fred_api_key", "")
    if not key:
        key = os.environ.get("FRED_API_KEY", "")
    return key
