"""
Reusable analytics functions for the Ag Market Predictor Dashboard.
Covers feature engineering, correlation analysis, anomaly detection,
and cross-market merging logic.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────
# Date Alignment & Multi-Dataset Merging
# ─────────────────────────────────────────────

def align_datasets_on_date(
    dataframes: List[Tuple[str, pd.DataFrame]],
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    Merge a list of (name, df) tuples on a common Date_Index column.
    Each df should already have a Date_Index column (datetime.date).
    Numeric columns are suffixed with the dataset name to prevent collisions.
    Missing values from outer joins are filled using the specified method.
    """
    merged = pd.DataFrame()
    for name, df in dataframes:
        if df.empty:
            continue
        # Identify date column — check multiple patterns
        date_col = None
        for c in df.columns:
            cl = str(c).lower()
            if cl in ("report_date", "date", "tradingday", "trading_day", "slaughter_date"):
                date_col = c
                break
            if "date" in cl or "time" in cl:
                date_col = c
                break
        if date_col is None:
            continue

        temp = df.copy()
        parsed = pd.to_datetime(temp[date_col], errors="coerce")
        if parsed.isna().all():
            continue
        temp["Date_Index"] = parsed.dt.date
        num_cols = temp.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            continue
        temp = temp[["Date_Index"] + num_cols]
        base = name.replace(".csv", "")
        temp = temp.rename(columns={c: f"{c} ({base})" for c in num_cols})
        temp = temp.groupby("Date_Index").mean().reset_index()

        if merged.empty:
            merged = temp
        else:
            merged = pd.merge(merged, temp, on="Date_Index", how="outer")

    if merged.empty:
        return merged

    merged = merged.sort_values("Date_Index")
    # Drop columns that are entirely NaN (e.g. Cutout 'trend') — they
    # cannot be forward-filled and would cause dropna() to remove all rows.
    merged = merged.dropna(axis=1, how="all")
    if fill_method == "ffill":
        merged = merged.ffill()
    elif fill_method == "interpolate":
        merged = merged.interpolate(method="linear")
    merged = merged.dropna()
    return merged


# ─────────────────────────────────────────────
# Data Leakage & Redundancy Sanitization
# ─────────────────────────────────────────────

# Patterns that indicate calendar/ranking artifacts, NOT genuine market signals.
# These correlate with the target only because the target trends over time.
_LEAKAGE_PATTERNS = [
    "IndexbyYear",
    "indexbyyear",
    "Week of Year",
    "week of year",
    "RankbyAttribute",
    "rankbyattribute",
    "CommodityRowCount",
]

# Bare 'Year' columns are calendar leakage — they only correlate with
# the target because prices trend upward over time.
_YEAR_COLUMN_PATTERN = "Year"

# Columns that are perfectly redundant (r=1.00 with another column).
# Keep only the first; drop the rest.
_REDUNDANT_GROUPS = [
    # Nearby Futures: price_5day ≡ avg ≡ wtd_avg
    {"keep": "price_5day", "drop": ["avg", "wtd_avg"]},
]


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove data-leakage columns and deduplicate redundant features.

    Drops:
      - Calendar artifacts (IndexbyYear, Week of Year) that only correlate
        with the target because price trends over time.
      - RankbyAttribute (ordinal ranking proxy, not a market signal).
      - Redundant columns (e.g. avg/wtd_avg when price_5day is present).
    """
    cols_to_drop = []

    for col in df.columns:
        col_lower = col.lower()
        # Check leakage patterns
        for pattern in _LEAKAGE_PATTERNS:
            if pattern.lower() in col_lower:
                cols_to_drop.append(col)
                break
        # Drop bare 'Year' columns (e.g. 'Year (Cash Cattle)') — calendar leakage
        col_base = col.split(" (")[0].strip()
        if col_base == _YEAR_COLUMN_PATTERN:
            cols_to_drop.append(col)

    # Check redundancy: only drop avg/wtd_avg from Nearby Futures
    for group in _REDUNDANT_GROUPS:
        keep_cols = [c for c in df.columns if group["keep"] in c]
        if keep_cols:
            for drop_name in group["drop"]:
                for c in df.columns:
                    c_lower = c.lower()
                    # Only match if it's from Nearby Futures (not generic 'avg' in other columns)
                    if drop_name in c_lower and c not in keep_cols and "nearby futures" in c_lower:
                        cols_to_drop.append(c)

    cols_to_drop = list(set(cols_to_drop))
    if cols_to_drop:
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df


# ─────────────────────────────────────────────
# Feature Engineering for XGBoost
# ─────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    target_col: str,
    lags: List[int] = None,
    rolling_windows: List[int] = None,
    predict_returns: bool = False,
) -> pd.DataFrame:
    """
    Add time-series features proven to drive predictions.

    Trimmed to ~15 features based on feature importance analysis:
      - roc_1 (39.8%), momentum_7 (26.9%), bollinger_z_14 (9.6%)
      - lag_1/lag_3, roll_std_7/14, roll_mean_30, sma_50_position
      - roc_14/roc_30, roll_std_30, day_of_week_sin/cos

    Removed 0%-importance features: lag_7/14/21, roc_7, momentum_30,
    bollinger_z_30, roll_mean_7/14, sma_200, month_sin/cos, quarter_sin/cos.

    Parameters
    ----------
    predict_returns : bool
        If True, transform target to pct_change (returns) instead of
        raw price levels. The original price column is preserved as
        '_original_price' for inverse transformation.
    """
    if lags is None:
        lags = [1, 3]
    if rolling_windows is None:
        rolling_windows = [7, 14, 30]

    out = df.copy()

    # ── Lagged values (only 1 & 3 — longer lags had 0% importance) ──
    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)

    # ── Rolling statistics ──
    shifted = out[target_col].shift(1)
    # Rolling std for all windows (all >1% importance)
    for w in rolling_windows:
        out[f"roll_std_{w}"] = shifted.rolling(w).std()
    # Rolling mean only for 30-day (7 & 14 had 0% importance)
    out["roll_mean_30"] = shifted.rolling(30).mean()

    # ── Rate of change (1, 14, 30 — roc_7 had 0% importance) ──
    for period in [1, 14, 30]:
        out[f"roc_{period}"] = out[target_col].pct_change(period)

    # ── Momentum (7-day only — 30-day had 0% importance) ──
    out["momentum_7"] = out[target_col] - shifted.rolling(7).mean()

    # ── SMA position (50-day only — 200-day had 0% importance) ──
    sma_50 = shifted.rolling(50).mean()
    out["sma_50_position"] = (out[target_col] - sma_50) / sma_50.replace(0, np.nan)

    # ── Bollinger z-score (14-day only — 30-day had 0% importance) ──
    rmean = shifted.rolling(14).mean()
    rstd = shifted.rolling(14).std()
    out["bollinger_z_14"] = (out[target_col] - rmean) / rstd.replace(0, np.nan)

    # ── Day-of-week only (month/quarter had 0% importance) ──
    if "Date_Index" in out.columns:
        dates = pd.to_datetime(out["Date_Index"])
        out["day_of_week_sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
        out["day_of_week_cos"] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)

    # ── Returns mode: transform target to pct_change ──
    if predict_returns:
        out["_original_price"] = out[target_col]
        out[target_col] = out[target_col].pct_change()

    out = out.dropna()
    return out


def inverse_transform_returns(
    predictions: np.ndarray,
    last_known_price: float,
) -> np.ndarray:
    """
    Convert predicted returns back to price levels (cumulative).
    price_t = price_{t-1} * (1 + return_t)
    """
    prices = np.zeros(len(predictions))
    prices[0] = last_known_price * (1 + predictions[0])
    for i in range(1, len(predictions)):
        prices[i] = prices[i - 1] * (1 + predictions[i])
    return prices


def inverse_transform_returns_1step(
    predictions: np.ndarray,
    previous_actual_prices: np.ndarray,
) -> np.ndarray:
    """
    Convert predicted returns back to price levels for a 1-step-ahead forecast.
    Instead of accumulating errors iteratively, this uses the actual known price
    at t-1 to compute the predicted price at t.
    """
    return previous_actual_prices * (1 + predictions)

# ─────────────────────────────────────────────
# Cross-Market Correlation Analysis
# ─────────────────────────────────────────────

def rolling_correlation_matrix(
    df: pd.DataFrame,
    window: int = 30,
    min_periods: int = 15,
) -> pd.DataFrame:
    """
    Compute rolling pairwise correlations and return the latest snapshot.
    Only considers numeric columns.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2 or len(num_df) < min_periods:
        return pd.DataFrame()
    corr = num_df.rolling(window=window, min_periods=min_periods).corr()
    # Get the last complete snapshot
    latest_idx = corr.index.get_level_values(0)[-1]
    return corr.loc[latest_idx]


def compute_static_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Full-period Pearson correlation matrix on numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return pd.DataFrame()
    return num_df.corr()


# ─────────────────────────────────────────────
# Anomaly / Z-Score Detection
# ─────────────────────────────────────────────

def zscore_anomalies(
    series: pd.Series,
    window: int = 30,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Compute rolling z-scores and flag anomalies.
    Returns a DataFrame with rolling_mean, rolling_std, z_score, and is_anomaly.
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    z_score = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return pd.DataFrame({
        "value": series,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "z_score": z_score,
        "is_anomaly": z_score.abs() > threshold,
    })


# ─────────────────────────────────────────────
# Basis Calculation (Cash – Futures)
# ─────────────────────────────────────────────

def compute_basis(
    cash_series: pd.Series,
    futures_series: pd.Series,
) -> pd.Series:
    """
    Compute the cash-futures basis.
    Positive basis → cash premium; negative basis → cash discount.
    """
    return cash_series - futures_series
