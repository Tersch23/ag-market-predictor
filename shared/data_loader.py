"""
Cached data loading utilities for the Ag Market Predictor Dashboard.
Handles CSV and XLSX files with automatic date detection and type coercion.
"""
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import warnings
import os

# Suppress openpyxl default-style warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


@st.cache_data(show_spinner="Loading CSV…")
def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV into a DataFrame with automatic date parsing. Cached."""
    if not os.path.exists(file_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, low_memory=False)
        return _auto_detect_dates(df)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner="Loading XLSX…")
def load_xlsx(file_path: str, sheet_name=0) -> pd.DataFrame:
    """Load an Excel file into a DataFrame. Cached."""
    if not os.path.exists(file_path):
        return pd.DataFrame()
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return _auto_detect_dates(df)
    except Exception:
        return pd.DataFrame()


def _auto_detect_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristically convert columns that look like dates into datetime."""
    df = df.copy()
    date_keywords = ["date", "time", "week", "month"]
    for col in df.columns:
        if df[col].dtype == "object":
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in date_keywords):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass
    return df


def get_column_info(df: pd.DataFrame) -> dict:
    """Classify columns by type for downstream logic."""
    info = {"date_cols": [], "numeric_cols": [], "categorical_cols": []}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            info["date_cols"].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            info["numeric_cols"].append(col)
        else:
            info["categorical_cols"].append(col)
    return info


@st.cache_data(show_spinner="Scanning data directory…")
def discover_track1_files(data_root: str) -> pd.DataFrame:
    """Recursively discover all CSV/XLSX files under data_root and return a manifest DataFrame."""
    root = Path(data_root)
    if not root.exists():
        return pd.DataFrame()

    rows = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith("."):
                continue
            if fn.endswith(".csv") or fn.endswith(".xlsx"):
                fp = Path(dirpath) / fn
                # Skip Track 2 subdirectory
                if "Track 2" in fp.parts:
                    continue
                parent = fp.parent.name
                category = parent if parent != root.name else "Root"
                rows.append({
                    "path": str(fp),
                    "filename": fn,
                    "extension": fp.suffix,
                    "size_mb": round(fp.stat().st_size / (1024 * 1024), 2),
                    "parent_dir": parent,
                    "category": category,
                })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
