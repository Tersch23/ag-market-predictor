"""Tab 5: Predictive Engine (XAI) — dynamic XGBoost training with explainability."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from shared.data_loader import load_csv
from shared.constants import get_data_root, CHART_THEME, PALETTE
from shared.analytics import (
    align_datasets_on_date, engineer_features, sanitize_columns,
    inverse_transform_returns,
)


def render():
    st.header("🧠 Predictive Engine (Explainable AI)")
    st.markdown(
        "Select Track 1 datasets, choose a target variable, and train an XGBoost model "
        "in real-time. The engine extracts mathematical feature importance to explain **why** "
        "the market is predicted to move."
    )

    data_root = get_data_root()
    if not data_root.exists():
        st.error("Data directory not found."); return

    # ── 1. Dataset selection ──
    csv_files = sorted([f.name for f in data_root.glob("*.csv") if f.is_file()])
    if not csv_files:
        st.error("No CSV files found."); return

    col_cfg, col_preview = st.columns([1, 2])

    with col_cfg:
        st.subheader("1 · Configure")
        selected_files = st.multiselect("Datasets to merge", csv_files,
                                        default=[csv_files[0]] if csv_files else [])
        if not selected_files:
            st.warning("Select at least one dataset."); return

    # Load and merge
    dfs = []
    for fn in selected_files:
        df = load_csv(str(data_root / fn))
        if not df.empty:
            dfs.append((fn, df))

    if not dfs:
        st.error("Could not load any selected datasets."); return

    merged = align_datasets_on_date(dfs)
    if merged.empty or "Date_Index" not in merged.columns:
        st.error("Could not align datasets on a common date column."); return

    # ── Sanitize data leakage columns ──
    merged = sanitize_columns(merged)

    # ── Optional: Enrich with FRED macro data ──
    fred_key = st.session_state.get("fred_api_key", "")
    if fred_key:
        from shared.fred_loader import FRED_SERIES, FRED_DESCRIPTIONS, fetch_multiple_fred
        with col_cfg:
            st.markdown("---")
            st.markdown("**📡 External Macro Indicators (FRED)**")
            fred_options = list(FRED_SERIES.keys())
            fred_selected = st.multiselect(
                "Add macro features",
                fred_options,
                default=[],
                help="These indicators are fetched from the Federal Reserve (FRED) and merged with your Track 1 data.",
            )
            if fred_selected:
                with st.spinner("Fetching FRED data…"):
                    fred_df = fetch_multiple_fred(fred_selected, fred_key)
                if not fred_df.empty:
                    merged = pd.merge(merged, fred_df, on="Date_Index", how="left")
                    merged = merged.sort_values("Date_Index").ffill().dropna()
                    st.success(f"+ {len(fred_selected)} FRED indicator(s) merged")

    with col_cfg:
        st.success(f"Aligned {len(dfs)} dataset(s) → {len(merged):,} rows")
        num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.error("No numeric columns available."); return
        target_col = st.selectbox("Target variable to predict", num_cols)

        # Date range filter
        min_d, max_d = merged["Date_Index"].min(), merged["Date_Index"].max()
        try:
            dr = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
            if len(dr) == 2:
                merged = merged[(merged["Date_Index"] >= dr[0]) & (merged["Date_Index"] <= dr[1])]
        except Exception:
            pass

        # ── Model mode toggle ──
        st.markdown("---")
        st.markdown("**Model Configuration**")
        predict_mode = st.radio(
            "Prediction target",
            ["📈 Returns (recommended)", "💲 Price Levels (baseline)"],
            help="**Returns mode** predicts daily percent changes — eliminates trend bias. "
                 "**Price Levels** predicts raw prices — struggles with regime changes.",
        )
        use_returns = predict_mode.startswith("📈")

        n_folds = st.slider("Cross-validation folds", 2, 10, 5)
        train_btn = st.button("🚀 Train Model & Explain", type="primary", use_container_width=True)

    with col_preview:
        st.subheader("Merged Dataset Preview")
        st.caption("Gaps from differing report frequencies are filled via forward-fill. "
                   "Data leakage columns (IndexbyYear, RankbyAttribute) removed automatically.")
        st.dataframe(merged.head(50), use_container_width=True)

    st.divider()

    # ── 2. Train ──
    if train_btn:
        if len(merged) < 50:
            st.error("Need ≥ 50 rows for reliable training."); return

        with st.spinner("Engineering features & training XGBoost…"):
            ml_df = engineer_features(
                merged, target_col, predict_returns=use_returns,
            )

            if len(ml_df) < 20:
                st.error("Not enough rows after feature engineering."); return

            # Save original prices for inverse transform if in returns mode
            original_prices = None
            if use_returns and "_original_price" in ml_df.columns:
                original_prices = ml_df["_original_price"].values
                ml_df = ml_df.drop(columns=["_original_price"])

            # Drop Date_Index before training
            if "Date_Index" in ml_df.columns:
                ml_df = ml_df.drop(columns=["Date_Index"])

            X = ml_df.drop(columns=[target_col])
            y = ml_df[target_col]

            # XGBoost with regularization to reduce overfitting
            xgb_params = dict(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                reg_alpha=0.1,    # L1 regularization
                reg_lambda=1.0,   # L2 regularization
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )

            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_folds)
            fold_results = []

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
                ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
                m = xgb.XGBRegressor(**xgb_params)
                m.fit(Xtr, ytr, verbose=False)
                preds = m.predict(Xte)
                mae = mean_absolute_error(yte, preds)
                r2 = r2_score(yte, preds)
                mape = mean_absolute_percentage_error(yte, preds) * 100

                # Directional accuracy
                if len(yte) > 1:
                    actual_dir = np.sign(np.diff(yte.values))
                    pred_dir = np.sign(np.diff(preds))
                    dir_acc = np.mean(actual_dir == pred_dir) * 100
                else:
                    dir_acc = 0.0

                fold_results.append({
                    "Fold": fold + 1, "MAE": mae, "R²": r2,
                    "MAPE": f"{mape:.1f}%", "Dir. Acc.": f"{dir_acc:.0f}%",
                    "Train": len(Xtr), "Test": len(Xte),
                })

            # Final model on 80/20 split for display
            split = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train, y_train, verbose=False)
            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions) * 100
            r2 = r2_score(y_test, predictions)

            # Directional accuracy on test set
            if len(y_test) > 1:
                actual_dir = np.sign(np.diff(y_test.values))
                pred_dir = np.sign(np.diff(predictions))
                dir_acc = np.mean(actual_dir == pred_dir) * 100
            else:
                dir_acc = 0.0

            importance = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False)

            # Inverse transform for display if in returns mode
            display_actual = y_test.values
            display_predicted = predictions
            if use_returns and original_prices is not None:
                from shared.analytics import inverse_transform_returns_1step
                prev_prices = original_prices[split - 1 : split - 1 + len(y_test)]
                display_actual = original_prices[split : split + len(y_test)]
                display_predicted = inverse_transform_returns_1step(predictions, prev_prices)

        # ── 3. Results ──
        mode_label = "Returns → Price" if use_returns else "Price Levels"
        st.subheader(f"🎯 Model Performance ({mode_label})")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("MAE", f"{mae:.4f}")
        m2.metric("MAPE", f"{mape:.2f}%")
        m3.metric("R²", f"{r2:.4f}")
        m4.metric(f"CV MAE ({n_folds}-fold)", f"{np.mean([f['MAE'] for f in fold_results]):.4f}")
        m5.metric("Dir. Accuracy", f"{dir_acc:.0f}%")

        # Per-fold CV table
        st.subheader("📊 Cross-Validation Results (Per Fold)")
        cv_df = pd.DataFrame(fold_results)
        st.dataframe(cv_df, use_container_width=True, hide_index=True)

        cv_maes = [f["MAE"] for f in fold_results]
        cv_r2s = [f["R²"] for f in fold_results]
        st.caption(
            f"Mean MAE: {np.mean(cv_maes):.4f} ± {np.std(cv_maes):.4f}  |  "
            f"Mean R²: {np.mean(cv_r2s):.4f} ± {np.std(cv_r2s):.4f}"
        )

        col_imp, col_pred = st.columns([1, 2])

        with col_imp:
            st.markdown("**Top Feature Drivers**")
            st.dataframe(importance.head(10), use_container_width=True, hide_index=True)

        with col_pred:
            fig = px.bar(importance.head(10), x="Importance", y="Feature", orientation="h",
                         title=f"What drives {target_col}?",
                         color_discrete_sequence=[CHART_THEME["primary"]])
            fig.update_layout(yaxis={"categoryorder": "total ascending"},
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        # Actual vs Predicted (using display values — price levels even in returns mode)
        st.subheader("Actual vs Predicted")
        avp = pd.DataFrame({"Actual": display_actual, "Predicted": display_predicted})
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=avp["Actual"], mode="lines", name="Actual",
                                  line=dict(color=CHART_THEME["primary"])))
        fig2.add_trace(go.Scatter(y=avp["Predicted"], mode="lines", name="Predicted",
                                  line=dict(color=CHART_THEME["secondary"], dash="dash")))
        y_label = f"{target_col} ($/cwt)" if not use_returns else f"{target_col} (reconstructed price)"
        fig2.update_layout(title="Test Set: Actual vs Predicted",
                           xaxis_title="Observation", yaxis_title=y_label,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

        # Residual analysis
        st.subheader("Residual Analysis")
        residuals = display_actual - display_predicted
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            fig3 = px.histogram(residuals, nbins=40, title="Residual Distribution",
                                color_discrete_sequence=[CHART_THEME["accent"]])
            fig3.add_vline(x=0, line_dash="dash", line_color="red")
            fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               xaxis_title="Residual", yaxis_title="Count")
            st.plotly_chart(fig3, use_container_width=True)
        with rcol2:
            fig4 = px.scatter(x=display_predicted, y=residuals,
                              title="Residuals vs Predicted",
                              color_discrete_sequence=[CHART_THEME["primary"]])
            fig4.add_hline(y=0, line_dash="dash", line_color="red")
            fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               xaxis_title="Predicted", yaxis_title="Residual")
            st.plotly_chart(fig4, use_container_width=True)

        mean_resid = residuals.mean()
        st.caption(
            f"Mean residual: {mean_resid:.2f} (should be ~0)  |  "
            f"Std: {residuals.std():.2f}  |  "
            f"{'⚠️ Systematic bias detected' if abs(mean_resid) > 2 else '✅ Approximately unbiased'}"
        )

        # Store results in session for the Briefing tab
        top3 = importance.head(3)
        feature_text = ", ".join([f"{r['Feature']} ({r['Importance']*100:.1f}%)" for _, r in top3.iterrows()])
        recent_val = display_actual[-1]
        pred_val = display_predicted[-1]
        direction = "up" if pred_val > recent_val else "down"
        st.session_state["xai_facts"] = (
            f"The target variable '{target_col}' recently closed at {recent_val:.2f}. "
            f"The XGBoost model predicts it will move {direction} to {pred_val:.2f}. "
            f"MAE={mae:.4f}, R²={r2:.4f}, CV MAE={np.mean(cv_maes):.4f}, "
            f"Directional Accuracy={dir_acc:.0f}%. "
            f"Mode: {'returns-based (trend-adjusted)' if use_returns else 'price-level (baseline)'}. "
            f"The top mathematical drivers are: {feature_text}."
        )
        st.session_state["xai_target"] = target_col
