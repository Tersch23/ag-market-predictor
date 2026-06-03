"""
Data Analysis & Model Diagnostics — Output Capture Script
Mirrors data_analysis_model.ipynb, saves all figures + text to notebook_outputs/
Run:  python run_analysis.py
"""
import sys, os, io
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Setup ──────────────────────────────────────────────────────────
OUT = Path('notebook_outputs')
OUT.mkdir(exist_ok=True)
LOG = open(OUT / 'analysis_log.txt', 'w')
fig_num = [0]

def log(msg=''):
    print(msg)
    LOG.write(str(msg) + '\n')

def save_fig(name):
    fig_num[0] += 1
    fname = f'{fig_num[0]:02d}_{name}.png'
    plt.savefig(OUT / fname, dpi=150, bbox_inches='tight')
    log(f'[SAVED] {fname}')
    plt.close('all')

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['figure.dpi'] = 100

DATA_ROOT = Path('../pro_ag_comp/ProAg_data')
if not DATA_ROOT.exists():
    DATA_ROOT = Path('pro_ag_comp/ProAg_data')
log(f'Data root exists: {DATA_ROOT.exists()}')
log(f'Data root: {DATA_ROOT.resolve()}')

# ── §1 Data Inventory ─────────────────────────────────────────────
log('\n' + '='*60)
log('§1  DATA INVENTORY & QUALITY ASSESSMENT')
log('='*60)
rows = []
for dirpath, _, filenames in os.walk(DATA_ROOT):
    for fn in filenames:
        fp = Path(dirpath) / fn
        if fn.startswith('.') or 'Track 2' in str(fp): continue
        if fn.endswith('.csv'):
            try:
                full = pd.read_csv(fp, low_memory=False)
                date_cols = [c for c in full.columns if 'date' in c.lower()]
                dr = ''
                if date_cols:
                    dates = pd.to_datetime(full[date_cols[0]], errors='coerce').dropna()
                    if not dates.empty: dr = f'{dates.min().date()} to {dates.max().date()}'
                nan_pct = full.isna().mean().mean() * 100
                all_nan = [c for c in full.select_dtypes(include=[np.number]).columns if full[c].isna().all()]
                rows.append({'File': fn, 'Rows': len(full), 'Cols': len(full.columns),
                             'Size_MB': round(fp.stat().st_size/1e6, 2), 'Date Range': dr,
                             'NaN %': f'{nan_pct:.1f}%', 'All-NaN Cols': ', '.join(all_nan) or 'None'})
            except:
                rows.append({'File': fn, 'Rows': '?', 'Cols': '?', 'Size_MB': round(fp.stat().st_size/1e6,2),
                             'Date Range': 'ERROR', 'NaN %': '?', 'All-NaN Cols': '?'})

manifest = pd.DataFrame(rows).sort_values('File')
log(f'Discovered {len(manifest)} Track 1 CSV files, {manifest["Size_MB"].sum():.1f} MB total')
log(manifest.to_string(index=False))

# ── §2 Cash Cattle Deep Dive ──────────────────────────────────────
log('\n' + '='*60)
log('§2  CASH CATTLE DEEP DIVE')
log('='*60)
cash = pd.read_csv(DATA_ROOT / 'Cash Cattle.csv', low_memory=False)
cash['report_date'] = pd.to_datetime(cash['report_date'], errors='coerce')
cash = cash.sort_values('report_date')
log(f'Cash Cattle: {cash.shape[0]:,} rows, {cash.shape[1]} columns')
log(f'Date range: {cash["report_date"].min().date()} to {cash["report_date"].max().date()}')
log(f'Selling bases: {cash["selling_basis_description"].unique().tolist()}')
log(f'Grade descriptions: {cash["grade_description"].unique().tolist() if "grade_description" in cash.columns else "N/A"}')
log(cash.describe().to_string())

daily_cash = cash.groupby('report_date')['weighted_avg_price'].mean()
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
daily_cash.plot(ax=axes[0], color='#2563eb', linewidth=0.8)
axes[0].set_title('Cash Cattle — Daily Weighted Average Price ($/cwt)', fontsize=14)
axes[0].set_ylabel('Price ($)')
axes[0].axhline(daily_cash.mean(), color='red', linestyle='--', alpha=0.5, label=f'Mean: ${daily_cash.mean():.2f}')
axes[0].legend()
rolling_std = daily_cash.rolling(30).std()
rolling_std.plot(ax=axes[1], color='#dc2626', linewidth=0.8)
axes[1].set_title('30-Day Rolling Volatility', fontsize=14)
axes[1].set_ylabel('Std Dev ($)')
plt.tight_layout()
save_fig('cash_cattle_price_volatility')

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
daily_cash.hist(bins=50, ax=axes[0], color='#2563eb', edgecolor='white')
axes[0].set_title('Price Distribution')
axes[0].axvline(daily_cash.mean(), color='red', linestyle='--')
cash['year'] = cash['report_date'].dt.year
cash.boxplot(column='weighted_avg_price', by='year', ax=axes[1])
axes[1].set_title('Price by Year')
plt.suptitle('')
plt.tight_layout()
save_fig('cash_cattle_distribution')

log(f'\n=== ANALYST NOTES ===')
log(f'Price range: ${daily_cash.min():.2f} to ${daily_cash.max():.2f}')
log(f'Mean: ${daily_cash.mean():.2f}, Median: ${daily_cash.median():.2f}')
log(f'Skewness: {daily_cash.skew():.3f}')
log(f'Max volatility period: {rolling_std.idxmax()}')

# ── §3 Cutout Analysis ────────────────────────────────────────────
log('\n' + '='*60)
log('§3  CUTOUT (CHOICE vs SELECT) ANALYSIS')
log('='*60)
cutout = pd.read_csv(DATA_ROOT / 'Cutout (Select_Choice).csv', low_memory=False)
cutout['report_date'] = pd.to_datetime(cutout['report_date'], errors='coerce')
if 'Attribute' in cutout.columns:
    cutout = cutout.rename(columns={'Attribute': 'grade_description', 'Value': 'wtd_avg'})
cutout = cutout.sort_values('report_date')
pivot = cutout.pivot_table(index='report_date', columns='grade_description', values='wtd_avg', aggfunc='mean')
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
pivot.plot(ax=axes[0], linewidth=0.8)
axes[0].set_title('Cutout Values Over Time ($/cwt)', fontsize=14)
axes[0].set_ylabel('Price ($)')
choice_col = next((c for c in pivot.columns if 'choice' in str(c).lower()), None)
select_col = next((c for c in pivot.columns if 'select' in str(c).lower()), None)
if choice_col and select_col:
    spread = pivot[choice_col] - pivot[select_col]
    spread.plot(ax=axes[1], color='#8b5cf6', linewidth=0.8)
    axes[1].axhline(0, color='gray', linestyle='--')
    axes[1].set_title('Choice-Select Spread ($/cwt)', fontsize=14)
    log(f'Spread stats: mean=${spread.mean():.2f}, std=${spread.std():.2f}')
plt.tight_layout()
save_fig('cutout_choice_select')

# ── §4 Nearby Futures ─────────────────────────────────────────────
log('\n' + '='*60)
log('§4  NEARBY FUTURES ANALYSIS')
log('='*60)
futures = pd.read_csv(DATA_ROOT / 'Nearby Futures.csv', low_memory=False)
futures['report_date'] = pd.to_datetime(futures['report_date'], errors='coerce')
futures = futures.sort_values('report_date')
fig, ax = plt.subplots(figsize=(14, 5))
for name in futures['Name'].dropna().unique():
    sub = futures[futures['Name'] == name]
    ax.plot(sub['report_date'], sub['price_5day'], label=name, linewidth=0.8)
ax.set_title('Nearby Futures — 5-Day Price by Region (HOG/PORK Carcass)', fontsize=14)
ax.set_ylabel('Price ($/cwt)')
ax.legend()
plt.tight_layout()
save_fig('nearby_futures')
log(f'Regions: {futures["Name"].unique().tolist()}')
log(f'Purchase type: {futures["purchase_type"].unique().tolist()}')
log(f'Price range: ${futures["price_5day"].min():.2f} to ${futures["price_5day"].max():.2f}')

# ── §5 Production & Harvest ───────────────────────────────────────
log('\n' + '='*60)
log('§5  PRODUCTION & HARVEST DATA')
log('='*60)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, (fn, title) in enumerate([
    ('Beef Production.csv', 'Beef Production (LBS)'),
    ('Pork Production.csv', 'Pork Production'),
    ('Weekly Harvest.csv', 'Weekly Harvest Volume'),
    ('Carcass Weights.csv', 'Carcass Weights')]):
    ax = axes[i//2, i%2]
    fp = DATA_ROOT / fn
    if not fp.exists(): continue
    df = pd.read_csv(str(fp), low_memory=False)
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    num_cols = df.select_dtypes(include=[np.number]).columns
    if date_cols and len(num_cols) > 0:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
        df = df.sort_values(date_cols[0])
        ax.plot(df[date_cols[0]], df[num_cols[0]], linewidth=0.8, color='#16a34a')
    ax.set_title(title, fontsize=12)
plt.tight_layout()
save_fig('production_harvest')

# ── §6 Futures Contract Curves ────────────────────────────────────
log('\n' + '='*60)
log('§6  FUTURES CONTRACT CURVES')
log('='*60)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, commodity in enumerate(['Live Cattle', 'Hog', 'Corn', 'Soybeans']):
    ax = axes[idx//2, idx%2]
    d = DATA_ROOT / f'{commodity} 2025 Futures'
    if not d.exists():
        ax.set_title(f'{commodity} — No data')
        log(f'{commodity}: directory not found')
        continue
    for f in sorted(d.glob('*.xlsx')):
        try:
            raw = pd.read_excel(str(f))
            if any('Unnamed' in str(c) for c in raw.columns):
                raw.columns = raw.iloc[0].astype(str)
                raw = raw.iloc[1:].reset_index(drop=True)
            if 'Time' in raw.columns and 'Close' in raw.columns:
                raw['Date'] = pd.to_datetime(raw['Time'], errors='coerce')
                raw['Close'] = pd.to_numeric(raw['Close'], errors='coerce')
                raw = raw.dropna(subset=['Date','Close']).sort_values('Date')
                ax.plot(raw['Date'], raw['Close'], label=f.stem.split(' ')[0], linewidth=0.7)
        except: pass
    ax.set_title(f'{commodity} 2025 Futures', fontsize=12)
    ax.legend(fontsize=7)
plt.tight_layout()
save_fig('futures_contract_curves')

# ── §7 Cross-Market Correlation ───────────────────────────────────
log('\n' + '='*60)
log('§7  CROSS-MARKET CORRELATION ANALYSIS')
log('='*60)
def load_and_prep(filename):
    fp = DATA_ROOT / filename
    if not fp.exists(): return pd.DataFrame()
    df = pd.read_csv(fp, low_memory=False)
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    if not date_cols: return pd.DataFrame()
    df['Date_Index'] = pd.to_datetime(df[date_cols[0]], errors='coerce').dt.date
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols: return pd.DataFrame()
    base = filename.replace('.csv', '')
    df = df[['Date_Index'] + num_cols].rename(columns={c: f'{c} ({base})' for c in num_cols})
    return df.groupby('Date_Index').mean().reset_index()

datasets = ['Cash Cattle.csv', 'Cutout (Select_Choice).csv', 'Nearby Futures.csv',
            'Beef Production.csv', 'Carcass Weights.csv', 'Weekly Harvest.csv']
merged = pd.DataFrame()
for fn in datasets:
    temp = load_and_prep(fn)
    if temp.empty: continue
    if merged.empty: merged = temp
    else: merged = pd.merge(merged, temp, on='Date_Index', how='outer')

merged = merged.sort_values('Date_Index').dropna(axis=1, how='all').ffill().dropna()
log(f'Merged: {merged.shape[0]:,} rows × {merged.shape[1]} columns')

num = merged.select_dtypes(include=[np.number])
top_cols = num.std().nlargest(15).index.tolist()
corr = num[top_cols].corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, linewidths=0.5, ax=ax,
            xticklabels=[c[:30] for c in corr.columns], yticklabels=[c[:30] for c in corr.columns])
ax.set_title('Cross-Market Correlation Heatmap (Top 15 Features)', fontsize=14)
plt.tight_layout()
save_fig('correlation_heatmap')

# ── §8 FRED Macro Indicators ──────────────────────────────────────
log('\n' + '='*60)
log('§8  FRED MACROECONOMIC INDICATORS')
log('='*60)
FRED_API_KEY = '137417237dd06cd5b46636ee9af306fe'
try:
    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)
    series = {'Corn PPI': 'WPU0223', 'WTI Oil': 'DCOILWTICO', 'Fed Funds': 'FEDFUNDS',
              'CPI': 'CPIAUCSL', 'USD Index': 'DTWEXBGS'}
    fig, axes = plt.subplots(len(series), 1, figsize=(14, 3*len(series)), sharex=True)
    for i, (name, sid) in enumerate(series.items()):
        data = fred.get_series(sid, observation_start='2019-01-01')
        if data is not None and not data.empty:
            axes[i].plot(data.index, data.values, linewidth=0.8, color='#2563eb')
            axes[i].set_title(f'{name} ({sid})', fontsize=11)
            axes[i].set_ylabel('Value')
            log(f'✅ {name}: latest={data.dropna().iloc[-1]:.4f} ({data.dropna().index[-1].date()})')
    plt.tight_layout()
    save_fig('fred_macro_indicators')
except ImportError:
    log('fredapi not installed. Run: pip install fredapi')

# ── §9 XGBoost Model Diagnostics ─────────────────────────────────
log('\n' + '='*60)
log('§9  XGBOOST MODEL — FULL DIAGNOSTICS')
log('='*60)
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

target = 'weighted_avg_price (Cash Cattle)'
if target not in merged.columns:
    price_cols = [c for c in merged.columns if 'price' in c.lower()]
    target = price_cols[0] if price_cols else None
    log(f'Using fallback target: {target}')

ml = merged.drop(columns=['Date_Index']).copy()
for lag in [1, 3, 7, 14, 21]:
    ml[f'lag_{lag}'] = ml[target].shift(lag)
for w in [7, 14, 30]:
    ml[f'roll_mean_{w}'] = ml[target].shift(1).rolling(w).mean()
    ml[f'roll_std_{w}'] = ml[target].shift(1).rolling(w).std()
ml['roc_1'] = ml[target].pct_change(1)
ml['roc_7'] = ml[target].pct_change(7)
ml['momentum_7'] = ml[target] - ml[target].shift(1).rolling(7).mean()
ml = ml.dropna()

X = ml.drop(columns=[target])
y = ml[target]
log(f'Feature matrix: {X.shape[0]} rows × {X.shape[1]} features')
log(f'Target: {target}')

# CV
tscv = TimeSeriesSplit(n_splits=5)
fold_results = []
fig, axes = plt.subplots(5, 1, figsize=(14, 20))
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
    ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
    m = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)
    m.fit(Xtr, ytr, verbose=False)
    preds = m.predict(Xte)
    mae = mean_absolute_error(yte, preds)
    r2 = r2_score(yte, preds)
    mape = mean_absolute_percentage_error(yte, preds) * 100
    fold_results.append({'Fold': fold+1, 'MAE': mae, 'R2': r2, 'MAPE': mape,
                         'Train Size': len(Xtr), 'Test Size': len(Xte),
                         'Test Mean': yte.mean(), 'Test Std': yte.std()})
    axes[fold].plot(yte.values, label='Actual', color='#2563eb')
    axes[fold].plot(preds, label='Predicted', color='#f59e0b', linestyle='--')
    axes[fold].set_title(f'Fold {fold+1}: MAE={mae:.2f}, R²={r2:.4f}, MAPE={mape:.1f}%', fontsize=11)
    axes[fold].legend()
plt.suptitle('Cross-Validation Folds — Actual vs Predicted', fontsize=14, y=1.01)
plt.tight_layout()
save_fig('cv_folds')

cv_df = pd.DataFrame(fold_results)
log('\n=== CV RESULTS ===')
log(cv_df.to_string(index=False))
log(f'\nMean MAE: {cv_df["MAE"].mean():.4f} ± {cv_df["MAE"].std():.4f}')
log(f'Mean R²:  {cv_df["R2"].mean():.4f} ± {cv_df["R2"].std():.4f}')
log(f'Mean MAPE: {cv_df["MAPE"].mean():.2f}% ± {cv_df["MAPE"].std():.2f}%')

# Final model
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
final_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)
final_model.fit(X_train, y_train, verbose=False)
preds = final_model.predict(X_test)

log(f'\nTest MAE:  {mean_absolute_error(y_test, preds):.4f}')
log(f'Test MAPE: {mean_absolute_percentage_error(y_test, preds)*100:.2f}%')
log(f'Test R²:   {r2_score(y_test, preds):.4f}')

importance = pd.DataFrame({'Feature': X.columns, 'Importance': final_model.feature_importances_})
importance = importance.sort_values('Importance', ascending=True).tail(20)
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
axes[0].barh(importance['Feature'].apply(lambda x: x[:40]), importance['Importance'], color='#2563eb')
axes[0].set_title('Top 20 Feature Importances', fontsize=14)
axes[0].set_xlabel('Importance (gain)')
axes[1].plot(y_test.values, label='Actual', color='#2563eb')
axes[1].plot(preds, label='Predicted', color='#f59e0b', linestyle='--')
axes[1].set_title('Test Set: Actual vs Predicted', fontsize=14)
axes[1].legend()
plt.tight_layout()
save_fig('model_importance_and_test')

# ── §10 Residual Analysis ────────────────────────────────────────
log('\n' + '='*60)
log('§10  RESIDUAL ANALYSIS')
log('='*60)
residuals = y_test.values - preds
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].hist(residuals, bins=40, color='#8b5cf6', edgecolor='white')
axes[0,0].axvline(0, color='red', linestyle='--')
axes[0,0].set_title('Residual Distribution')
axes[0,0].set_xlabel('Residual (Actual - Predicted)')
axes[0,1].scatter(range(len(residuals)), residuals, alpha=0.5, s=10, color='#dc2626')
axes[0,1].axhline(0, color='gray', linestyle='--')
axes[0,1].set_title('Residuals Over Time (Test Set)')
axes[0,1].set_xlabel('Observation')
axes[1,0].scatter(preds, residuals, alpha=0.5, s=10, color='#2563eb')
axes[1,0].axhline(0, color='red', linestyle='--')
axes[1,0].set_title('Residuals vs Predicted Value')
axes[1,0].set_xlabel('Predicted')
axes[1,0].set_ylabel('Residual')
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(pd.Series(residuals), ax=axes[1,1])
axes[1,1].set_title('Residual Autocorrelation')
plt.tight_layout()
save_fig('residual_analysis')

log(f'Mean residual: {residuals.mean():.4f} (should be ~0)')
log(f'Std residual:  {residuals.std():.4f}')
log(f'Skewness: {pd.Series(residuals).skew():.4f}')
log(f'Max overpredict: ${abs(residuals.min()):.2f}')
log(f'Max underpredict: ${residuals.max():.2f}')
bias = 'UNDERESTIMATES' if residuals.mean() > 1 else 'OVERESTIMATES' if residuals.mean() < -1 else 'approximately unbiased'
log(f'\n⚠️ Model {bias} (mean residual = {residuals.mean():.2f})')

# Save baseline metrics for comparison
baseline_mae = mean_absolute_error(y_test, preds)
baseline_r2 = r2_score(y_test, preds)
baseline_mape = mean_absolute_percentage_error(y_test, preds) * 100
baseline_resid_mean = residuals.mean()

# ── §11 IMPROVED MODEL — All Fixes Applied ───────────────────────
log('\n' + '='*60)
log('§11  IMPROVED MODEL — BASELINE vs FIXED')
log('='*60)
log('Fixes applied:')
log('  1. Sanitize data leakage columns (IndexbyYear, RankbyAttribute, redundant futures)')
log('  2. Predict returns (pct_change) instead of price levels')
log('  3. Add trend-aware features (Bollinger, SMA position, longer momentum, cyclical dates)')
log('  4. Regularized XGBoost (L1, L2, subsample, colsample)')

# --- Sanitize ---
_LEAKAGE = ['IndexbyYear', 'indexbyyear', 'Week of Year', 'week of year',
            'RankbyAttribute', 'rankbyattribute', 'CommodityRowCount']
cols_dropped = []
for col in merged.columns:
    if col == target:  # Never drop the target
        continue
    cl = col.lower()
    if any(p.lower() in cl for p in _LEAKAGE):
        cols_dropped.append(col)
    # Drop bare 'Year' columns (calendar leakage)
    col_base = col.split(' (')[0].strip()
    if col_base == 'Year':
        cols_dropped.append(col)
    # Drop redundant Nearby Futures columns (avg, wtd_avg) — keep price_5day
    if 'nearby futures' in cl and ('avg' in cl or 'wtd_avg' in cl) and 'price_5day' not in cl:
        cols_dropped.append(col)
cols_dropped = list(set(cols_dropped))
merged_clean = merged.drop(columns=[c for c in cols_dropped if c in merged.columns])
log(f'Dropped {len(cols_dropped)} leakage/redundant columns: {cols_dropped}')

# --- Feature engineering (trimmed to features with >0% importance) ---
ml2 = merged_clean.drop(columns=['Date_Index']).copy()
shifted = ml2[target].shift(1)

for lag in [1, 3]:  # lag_7/14/21 had 0% importance
    ml2[f'lag_{lag}'] = ml2[target].shift(lag)
for w in [7, 14, 30]:  # roll_std all matter
    ml2[f'roll_std_{w}'] = shifted.rolling(w).std()
ml2['roll_mean_30'] = shifted.rolling(30).mean()  # only 30-day mean matters
for period in [1, 14, 30]:  # roc_7 had 0% importance
    ml2[f'roc_{period}'] = ml2[target].pct_change(period)
ml2['momentum_7'] = ml2[target] - shifted.rolling(7).mean()  # momentum_30 had 0%

# SMA position (50-day only)
sma50 = shifted.rolling(50).mean()
ml2['sma_50_pos'] = (ml2[target] - sma50) / sma50.replace(0, np.nan)

# Bollinger z-score (14-day only — 30-day had 0%)
rm14 = shifted.rolling(14).mean()
rs14 = shifted.rolling(14).std()
ml2['boll_z_14'] = (ml2[target] - rm14) / rs14.replace(0, np.nan)

# Day-of-week only (month/quarter had 0%)
dates = pd.to_datetime(merged_clean['Date_Index'])
ml2['dow_sin'] = np.sin(2 * np.pi * dates.dt.dayofweek / 7).values
ml2['dow_cos'] = np.cos(2 * np.pi * dates.dt.dayofweek / 7).values

# --- Returns target ---
ml2['_orig_price'] = ml2[target]
ml2[target] = ml2[target].pct_change()
ml2 = ml2.dropna()

orig_prices = ml2['_orig_price'].values
ml2 = ml2.drop(columns=['_orig_price'])

X2 = ml2.drop(columns=[target])
y2 = ml2[target]
log(f'Improved feature matrix: {X2.shape[0]} rows × {X2.shape[1]} features')

# --- Regularized XGBoost ---
xgb_params = dict(n_estimators=200, learning_rate=0.05, max_depth=4,
                  reg_alpha=0.1, reg_lambda=1.0, subsample=0.8,
                  colsample_bytree=0.8, random_state=42)

# CV
tscv2 = TimeSeriesSplit(n_splits=5)
fold_results2 = []
fig, axes = plt.subplots(5, 1, figsize=(14, 20))
for fold, (train_idx, test_idx) in enumerate(tscv2.split(X2)):
    Xtr, Xte = X2.iloc[train_idx], X2.iloc[test_idx]
    ytr, yte = y2.iloc[train_idx], y2.iloc[test_idx]
    m2 = xgb.XGBRegressor(**xgb_params)
    m2.fit(Xtr, ytr, verbose=False)
    p2 = m2.predict(Xte)

    # Inverse transform to prices for interpretable metrics (1-step ahead)
    actual_prices = orig_prices[test_idx]
    prev_prices = orig_prices[test_idx - 1]
    pred_prices = prev_prices * (1 + p2)

    mae_p = mean_absolute_error(actual_prices, pred_prices)
    r2_p = r2_score(actual_prices, pred_prices)
    mape_p = mean_absolute_percentage_error(actual_prices, pred_prices) * 100
    dir_actual = np.sign(np.diff(actual_prices))
    dir_pred = np.sign(np.diff(pred_prices))
    dir_acc = np.mean(dir_actual == dir_pred) * 100 if len(dir_actual) > 0 else 0

    fold_results2.append({'Fold': fold+1, 'MAE': mae_p, 'R2': r2_p, 'MAPE': mape_p,
                          'Dir_Acc': dir_acc})

    axes[fold].plot(actual_prices, label='Actual', color='#2563eb')
    axes[fold].plot(pred_prices, label='Predicted', color='#16a34a', linestyle='--')
    axes[fold].set_title(f'Fold {fold+1}: MAE=${mae_p:.2f}, R²={r2_p:.4f}, Dir={dir_acc:.0f}%', fontsize=11)
    axes[fold].legend()
plt.suptitle('IMPROVED MODEL — CV Folds (Returns → Price)', fontsize=14, y=1.01)
plt.tight_layout()
save_fig('improved_cv_folds')

cv2_df = pd.DataFrame(fold_results2)
log('\n=== IMPROVED CV RESULTS ===')
log(cv2_df.to_string(index=False))
log(f'\nMean MAE: {cv2_df["MAE"].mean():.4f} ± {cv2_df["MAE"].std():.4f}')
log(f'Mean R²:  {cv2_df["R2"].mean():.4f} ± {cv2_df["R2"].std():.4f}')
log(f'Mean Dir Acc: {cv2_df["Dir_Acc"].mean():.1f}%')

# Final 80/20
split2 = int(len(X2) * 0.8)
X2_train, X2_test = X2.iloc[:split2], X2.iloc[split2:]
y2_train, y2_test = y2.iloc[:split2], y2.iloc[split2:]
fm2 = xgb.XGBRegressor(**xgb_params)
fm2.fit(X2_train, y2_train, verbose=False)
p2_final = fm2.predict(X2_test)

# Inverse transform (1-step ahead)
act_p = orig_prices[split2:]
prev_p = orig_prices[split2 - 1 : split2 - 1 + len(y2_test)]
prd_p = prev_p * (1 + p2_final)

imp_mae = mean_absolute_error(act_p, prd_p)
imp_r2 = r2_score(act_p, prd_p)
imp_mape = mean_absolute_percentage_error(act_p, prd_p) * 100
imp_resid = (act_p - prd_p).mean()

log(f'\nImproved Test MAE:  {imp_mae:.4f}')
log(f'Improved Test R²:   {imp_r2:.4f}')
log(f'Improved Test MAPE: {imp_mape:.2f}%')
log(f'Improved Mean Resid: {imp_resid:.4f}')

# Feature importance
imp2 = pd.DataFrame({'Feature': X2.columns, 'Importance': fm2.feature_importances_})
imp2 = imp2.sort_values('Importance', ascending=True).tail(20)
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
axes[0].barh(imp2['Feature'].apply(lambda x: x[:40]), imp2['Importance'], color='#16a34a')
axes[0].set_title('IMPROVED — Top 20 Feature Importances', fontsize=14)
axes[1].plot(act_p, label='Actual', color='#2563eb')
axes[1].plot(prd_p, label='Predicted', color='#16a34a', linestyle='--')
axes[1].set_title('IMPROVED — Test Set: Actual vs Predicted', fontsize=14)
axes[1].legend()
plt.tight_layout()
save_fig('improved_model_importance_and_test')

# Residuals
resid2 = act_p - prd_p
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].hist(resid2, bins=40, color='#16a34a', edgecolor='white')
axes[0,0].axvline(0, color='red', linestyle='--')
axes[0,0].set_title('IMPROVED — Residual Distribution')
axes[0,1].scatter(range(len(resid2)), resid2, alpha=0.5, s=10, color='#16a34a')
axes[0,1].axhline(0, color='gray', linestyle='--')
axes[0,1].set_title('IMPROVED — Residuals Over Time')
axes[1,0].scatter(prd_p, resid2, alpha=0.5, s=10, color='#16a34a')
axes[1,0].axhline(0, color='red', linestyle='--')
axes[1,0].set_title('IMPROVED — Residuals vs Predicted')
autocorrelation_plot(pd.Series(resid2), ax=axes[1,1])
axes[1,1].set_title('IMPROVED — Residual Autocorrelation')
plt.tight_layout()
save_fig('improved_residual_analysis')

# ── §12 COMPARISON TABLE ─────────────────────────────────────────
log('\n' + '='*60)
log('§12  BASELINE vs IMPROVED — COMPARISON')
log('='*60)
comparison = pd.DataFrame({
    'Metric': ['Test MAE ($)', 'Test R²', 'Test MAPE (%)', 'Mean Residual ($)',
               'Mean CV MAE ($)', 'Mean CV R²'],
    'Baseline': [f'{baseline_mae:.2f}', f'{baseline_r2:.4f}', f'{baseline_mape:.2f}',
                 f'{baseline_resid_mean:.2f}',
                 f'{cv_df["MAE"].mean():.2f}', f'{cv_df["R2"].mean():.4f}'],
    'Improved': [f'{imp_mae:.2f}', f'{imp_r2:.4f}', f'{imp_mape:.2f}',
                 f'{imp_resid:.2f}',
                 f'{cv2_df["MAE"].mean():.2f}', f'{cv2_df["R2"].mean():.4f}'],
})
log(comparison.to_string(index=False))
log(f'\n{"✅ IMPROVED MODEL IS BETTER" if imp_r2 > baseline_r2 else "⚠️ Review results"}')

# ── Done ──────────────────────────────────────────────────────────
LOG.close()
print(f'\n✅ All outputs saved to: {OUT.resolve()}')
print(f'   Files: {sorted(f.name for f in OUT.iterdir())}')
