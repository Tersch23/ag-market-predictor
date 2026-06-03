"""Tab 6: AI Executive Briefing — LLM-powered or template-fallback market summary."""
import streamlit as st
from pathlib import Path


def render():
    st.header("📝 AI Executive Briefing")
    st.markdown(
        "Generate a natural-language market briefing synthesized from the Predictive Engine's "
        "mathematical feature coefficients. The LLM acts as an **Explainable AI translator**, "
        "converting dense model output into actionable business language."
    )

    # Check if the predictor tab has produced facts
    facts = st.session_state.get("xai_facts")
    target = st.session_state.get("xai_target", "target variable")

    if not facts:
        st.info(
            "⬅️ Please run the **Predictive Engine** tab first to generate model results. "
            "The briefing is synthesized from those results."
        )
        return

    with st.expander("🔍 Model Facts (context sent to the AI)", expanded=False):
        safe = facts.replace("$", "\\$")
        st.info(safe)

    st.divider()

    # LLM model selection
    col_settings, col_output = st.columns([1, 2])

    with col_settings:
        st.subheader("Settings")

        # ── Display mode toggle ──
        display_mode = st.radio(
            "Briefing Detail Level",
            ["📋 Executive Summary", "📊 Detailed Analysis"],
            help="Executive Summary: 3–4 concise bullets for leadership.\n"
                 "Detailed Analysis: full technical breakdown with metrics and interpretation.",
        )
        is_detailed = display_mode == "📊 Detailed Analysis"

        st.divider()

        model_dir = Path(__file__).resolve().parent.parent.parent / "pro_ag_comp_test" / "cattleedge" / "models"
        # Also check a local models/ dir
        local_model_dir = Path(__file__).resolve().parent.parent / "models"
        local_model_dir.mkdir(exist_ok=True)

        gguf_files = []
        for d in [model_dir, local_model_dir]:
            if d.exists():
                gguf_files.extend(list(d.glob("*.gguf")))

        model_choice = None
        model_path = None
        if gguf_files:
            model_names = [f.name for f in gguf_files]
            model_names.append("Template Fallback (No LLM)")
            model_choice = st.selectbox("Local Model (Llama.cpp)", model_names)
            if model_choice != "Template Fallback (No LLM)":
                model_path = next(f for f in gguf_files if f.name == model_choice)
        else:
            st.warning("No .gguf model files found. Using template fallback.")
            st.markdown(
                "**To enable LLM briefings:** download a GGUF model "
                "(e.g. [Llama-3-8B-Instruct](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF)) "
                "and place it in the `models/` directory."
            )
            model_choice = "Template Fallback (No LLM)"

        generate_btn = st.button("📝 Generate Briefing", type="primary", use_container_width=True)

    with col_output:
        label = "Executive Summary" if not is_detailed else "Detailed Analysis"
        st.subheader(f"Generated Briefing — {label}")

        if generate_btn:
            with st.spinner("Generating briefing…"):
                if model_path and model_choice != "Template Fallback (No LLM)":
                    _generate_llm_briefing(model_path, facts, target, is_detailed)
                else:
                    _generate_template_briefing(facts, target, is_detailed)


def _generate_llm_briefing(model_path, facts, target, is_detailed):
    """Use a local LlamaCpp model to generate the briefing."""
    try:
        from langchain_community.llms import LlamaCpp
        from langchain_core.prompts import PromptTemplate

        if is_detailed:
            max_tokens = 600
            template = """You are a senior cross-market agricultural commodities analyst.
You use Explainable AI (XAI) to translate machine learning mathematics into business strategy.

Write a DETAILED technical briefing based ONLY on the following model facts.
Structure it with these sections (use markdown headers):
### Prediction
One sentence: what the model predicts and the direction of movement.
### Key Drivers
List each top feature driver with its importance percentage. Explain what each driver means in agricultural market terms.
### Model Confidence
Interpret the MAE, R², and CV MAE values. Explain what they mean for decision-making.
### Market Implications
2-3 sentences on what this means for producers, traders, or analysts.

CRITICAL: Use markdown formatting. Be precise. Cite the numbers from the facts.

MODEL FACTS:
{facts}

BRIEFING:
"""
        else:
            max_tokens = 250
            template = """You are a senior cross-market agricultural commodities analyst.

Write a 3-bullet executive summary based ONLY on the following model facts.
Each bullet should be ONE concise sentence. No repetition. No filler.

- Bullet 1: The prediction (direction + target value)
- Bullet 2: The top driver and what it means
- Bullet 3: Model confidence (one metric)

CRITICAL: Exactly 3 bullets using "- " prefix. Maximum 25 words per bullet.

MODEL FACTS:
{facts}

BRIEFING:
"""

        llm = LlamaCpp(
            model_path=str(model_path),
            temperature=0.2,
            max_tokens=max_tokens,
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=False,
        )

        prompt = PromptTemplate(template=template, input_variables=["facts"])
        chain = prompt | llm
        briefing = chain.invoke({"facts": facts})
        briefing = briefing.replace("•", "\n- ")
        safe = briefing.replace("$", "\\$")

        with st.container(border=True):
            st.markdown(safe)
        st.caption(f"Generated by Llama.cpp ({model_path.name}) running 100% locally.")

    except Exception as e:
        st.error("LLM inference failed. Falling back to template.")
        st.code(str(e))
        _generate_template_briefing(facts, target, is_detailed)


def _generate_template_briefing(facts, target, is_detailed):
    """Rule-based fallback when no LLM is available."""
    import re

    direction = "upward" if "move up" in facts else "downward"
    drivers = re.findall(r"([\w\s()]+)\s+\((\d+\.\d+)%\)", facts)

    # Extract metrics from facts string
    mae_match = re.search(r"MAE=([\d.]+)", facts)
    r2_match = re.search(r"R²=([\d.]+)", facts)
    cv_mae_match = re.search(r"CV MAE=([\d.]+)", facts)
    pred_match = re.search(r"to ([\d.]+)", facts)
    recent_match = re.search(r"at ([\d.]+)", facts)

    mae_val = mae_match.group(1) if mae_match else "N/A"
    r2_val = r2_match.group(1) if r2_match else "N/A"
    cv_mae_val = cv_mae_match.group(1) if cv_mae_match else "N/A"
    pred_val = pred_match.group(1) if pred_match else "N/A"
    recent_val = recent_match.group(1) if recent_match else "N/A"

    if not is_detailed:
        # ── Executive Summary: 3-4 concise bullets ──
        with st.container(border=True):
            st.markdown(f"- **{target}** is forecast to trend **{direction}** (current: {recent_val} → predicted: {pred_val}).")
            if len(drivers) >= 1:
                st.markdown(f"- Primary driver: **{drivers[0][0].strip()}** ({drivers[0][1]}% importance).")
            st.markdown(f"- Model confidence: R² = {r2_val}, MAE = {mae_val}.")
    else:
        # ── Detailed Analysis: structured technical breakdown ──
        with st.container(border=True):
            st.markdown("### 📈 Prediction")
            st.markdown(
                f"The XGBoost model forecasts **{target}** to move **{direction}** "
                f"from a recent close of **{recent_val}** to a predicted value of **{pred_val}**."
            )

            st.markdown("### 🔑 Key Drivers")
            if drivers:
                for i, (name, pct) in enumerate(drivers[:5]):
                    rank = ["Primary", "Secondary", "Tertiary"][i] if i < 3 else f"#{i+1}"
                    bar_len = int(float(pct) / 2)
                    bar = "█" * max(bar_len, 1)
                    st.markdown(f"- **{rank}:** {name.strip()} — `{pct}%` {bar}")
            else:
                st.markdown("- Feature driver data not available.")

            # Interpret what the top driver means
            if drivers:
                top_name = drivers[0][0].strip().lower()
                if "lag" in top_name:
                    st.caption("↳ Lagged price features indicate strong short-term autocorrelation — recent price history is the strongest predictor.")
                elif "roll" in top_name or "mean" in top_name:
                    st.caption("↳ Rolling averages dominate, suggesting trend-following behavior in this market.")
                elif "roc" in top_name or "momentum" in top_name:
                    st.caption("↳ Rate-of-change / momentum features suggest the market is momentum-driven.")
                elif "fred" in top_name.lower() or "cpi" in top_name.lower() or "oil" in top_name.lower():
                    st.caption("↳ Macro-economic indicators are driving predictions — external factors outweigh internal price history.")

            st.markdown("### 📊 Model Confidence")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("R²", r2_val, help="1.0 = perfect fit. >0.8 is strong.")
            mc2.metric("MAE", mae_val, help="Average absolute prediction error in price units.")
            mc3.metric("CV MAE", cv_mae_val, help="Cross-validated MAE — measures generalization ability.")

            # Interpret R²
            try:
                r2_num = float(r2_val)
                if r2_num > 0.90:
                    st.success("✅ Excellent predictive power — model captures >90% of price variance.")
                elif r2_num > 0.70:
                    st.success("✅ Strong predictive power — suitable for short-term forecasting.")
                elif r2_num > 0.50:
                    st.info("ℹ️ Moderate fit — useful for directional guidance but not precise point forecasts.")
                elif r2_num > 0.0:
                    st.warning("⚠️ Weak fit — predictions should be treated as directional signals only.")
                else:
                    st.error(f"❌ Negative R² ({r2_val}) — model performs worse than predicting the mean. "
                             "Consider switching to Returns mode in the Predictive Engine.")
            except ValueError:
                pass

            st.markdown("### 💡 Market Implications")
            st.markdown(
                f"The model's reliance on {'momentum-based features' if any('lag' in d[0].lower() or 'roll' in d[0].lower() or 'roc' in d[0].lower() for d in drivers) else 'cross-market features'} "
                f"suggests that **{target}** is primarily driven by "
                f"{'recent price trends and short-term autocorrelation' if any('lag' in d[0].lower() for d in drivers) else 'broader market relationships'}. "
                f"{'Producers and hedgers should monitor the rate of change for early reversal signals.' if direction == 'upward' else 'Risk management and downside protection may be warranted.'}"
            )

    mode = "rule-based template (no LLM)"
    st.caption(f"Generated using {mode}.")
