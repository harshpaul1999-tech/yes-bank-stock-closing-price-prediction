from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

from yes_bank_ml import (  # noqa: E402
    FEATURE_COLUMNS,
    build_modeling_frame,
    evaluate_models,
    fit_final_model,
    load_yes_bank_data,
    model_coefficients,
    next_forecast_date,
    permutation_feature_importance,
    predict_close_price,
)


st.set_page_config(
    page_title="Regression - Yes Bank Stock Closing Price Prediction",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(248, 214, 120, 0.22), transparent 32%),
                radial-gradient(circle at top right, rgba(61, 123, 189, 0.18), transparent 30%),
                linear-gradient(180deg, #f8f5ee 0%, #eef3f9 100%);
            color: #0f2233;
            font-family: "Aptos", "Segoe UI", sans-serif;
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(14, 35, 52, 0.96), rgba(26, 70, 104, 0.92));
            color: #f5f7fb;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 20px 50px rgba(17, 38, 58, 0.20);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.15;
        }
        .hero p {
            margin: 0.45rem 0 0;
            color: #d8e4f2;
            font-size: 1rem;
        }
        .section-card {
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(15, 34, 51, 0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 24px rgba(20, 40, 60, 0.08);
        }
        .tiny-note {
            color: #495d6d;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_raw_data() -> pd.DataFrame:
    return load_yes_bank_data(PROJECT_ROOT / "data" / "yes_bank_stock_prices.csv")


@st.cache_resource(show_spinner=False)
def get_project_assets() -> Dict[str, object]:
    raw_df = get_raw_data()
    model_df = build_modeling_frame(raw_df)
    metrics_df, _, split_payload = evaluate_models(model_df, FEATURE_COLUMNS)
    final_model, final_payload, _ = fit_final_model(model_df, FEATURE_COLUMNS)
    permutation_df = permutation_feature_importance(
        final_model,
        final_payload["x_test"],
        final_payload["y_test"],
        FEATURE_COLUMNS,
    )
    coefficient_df = model_coefficients(final_model, FEATURE_COLUMNS)
    return {
        "raw_df": raw_df,
        "model_df": model_df,
        "metrics_df": metrics_df,
        "final_model": final_model,
        "final_payload": final_payload,
        "permutation_df": permutation_df,
        "coefficient_df": coefficient_df,
        "split_payload": split_payload,
    }


def get_azure_client() -> OpenAI:
    if OpenAI is None:
        raise ValueError(
            "The `openai` package is not installed. Install dependencies from requirements.txt to enable the Azure assistant."
        )
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    if not endpoint or not api_key:
        raise ValueError(
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY to enable the Azure assistant."
        )
    return OpenAI(api_key=api_key, base_url="{0}/openai/v1/".format(endpoint))


def azure_assistant_enabled() -> bool:
    if OpenAI is None:
        return False
    required_vars = [
        os.getenv("AZURE_OPENAI_ENDPOINT"),
        os.getenv("AZURE_OPENAI_API_KEY"),
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    ]
    return all(required_vars)


def build_llm_context(assets: Dict[str, object]) -> str:
    raw_df = assets["raw_df"]
    metrics_df = assets["metrics_df"]
    permutation_df = assets["permutation_df"]
    latest_row = raw_df.iloc[-1]

    context_lines = [
        "Project: Regression - Yes Bank Stock Closing Price Prediction",
        "Subtitle: Machine Learning & GenAI with Microsoft Azure",
        "Dataset span: {0:%b %Y} to {1:%b %Y}".format(raw_df["Date"].min(), raw_df["Date"].max()),
        "Rows: {0}".format(len(raw_df)),
        "Target: Monthly closing price",
        "Latest actual close: {0:.2f} in {1:%b %Y}".format(latest_row["Close"], latest_row["Date"]),
        "Top model comparison rows:",
        metrics_df.head(3).to_string(index=False),
        "Most influential features by permutation importance:",
        permutation_df.head(5).to_string(index=False),
        "Business context: governance concerns after 2018 and the COVID-era shock are modeled as structural break flags.",
        "Guardrail: explain outcomes only from the supplied project context; do not invent unseen performance claims.",
    ]
    return "\n".join(context_lines)


def ask_azure_assistant(question: str, assets: Dict[str, object], chat_history: List[Dict[str, str]]) -> str:
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
    if not deployment_name:
        raise ValueError("Set AZURE_OPENAI_DEPLOYMENT_NAME to enable the Azure assistant.")

    transcript = []
    for item in chat_history[-6:]:
        transcript.append("{0}: {1}".format(item["role"].upper(), item["content"]))

    system_prompt = (
        "You are helping a candidate explain a Yes Bank stock price regression project in interviews. "
        "Keep answers practical, concise, and tied to the supplied project context. "
        "When discussing risk, mention that same-month Open/High/Low inputs make this closer to a price estimation system "
        "than a pure ex-ante trading forecast."
    )
    user_prompt = "\n\n".join(
        [
            build_llm_context(assets),
            "Recent chat history:",
            "\n".join(transcript) if transcript else "No previous messages.",
            "User question:",
            question,
        ]
    )

    client = get_azure_client()
    response = client.responses.create(
        model=deployment_name,
        instructions=system_prompt,
        input=user_prompt,
    )
    return response.output_text.strip()


def render_overview_tab(assets: Dict[str, object]) -> None:
    raw_df = assets["raw_df"]
    metrics_df = assets["metrics_df"]
    final_payload = assets["final_payload"]
    permutation_df = assets["permutation_df"]

    latest_row = raw_df.iloc[-1]
    holdout_metrics = final_payload["holdout_metrics"]

    metric_columns = st.columns(4)
    metric_columns[0].metric("Dataset Rows", len(raw_df))
    metric_columns[1].metric("Latest Close", "{0:.2f}".format(latest_row["Close"]))
    metric_columns[2].metric("Holdout RMSE", "{0:.2f}".format(holdout_metrics["RMSE"]))
    metric_columns[3].metric("Holdout R2", "{0:.3f}".format(holdout_metrics["R2"]))

    left_col, right_col = st.columns((1.6, 1))

    with left_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Closing Price Trend")
        chart_df = raw_df.copy()
        chart_df["MA_3"] = chart_df["Close"].rolling(3).mean()
        chart_df["MA_12"] = chart_df["Close"].rolling(12).mean()

        fig, ax = plt.subplots(figsize=(11, 4.6))
        ax.plot(chart_df["Date"], chart_df["Close"], color="#103c5a", linewidth=2.2, label="Close")
        ax.plot(chart_df["Date"], chart_df["MA_3"], color="#d2872c", linewidth=1.8, label="3M MA")
        ax.plot(chart_df["Date"], chart_df["MA_12"], color="#7a8b99", linewidth=1.6, label="12M MA")
        ax.axvline(pd.Timestamp("2018-01-01"), linestyle="--", color="#9c3a2b", linewidth=1.2)
        ax.text(pd.Timestamp("2018-03-01"), chart_df["Close"].max() * 0.92, "2018 stress period", color="#9c3a2b")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(frameon=False, loc="upper left")
        ax.grid(alpha=0.16)
        st.pyplot(fig, clear_figure=True)
        st.caption(
            "The post-2018 period is visibly more unstable, which is why the project adds structural-break indicators."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Model Comparison")
        st.dataframe(
            metrics_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Regularized linear models outperform the tree-based models here, which suggests a strong linear relationship between monthly price levels and the closing price."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns((1, 1))

    with bottom_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Top Drivers")
        fig, ax = plt.subplots(figsize=(8, 4.6))
        top_features = permutation_df.head(8).iloc[::-1]
        ax.barh(top_features["feature"], top_features["importance_mean"], color="#174d72")
        ax.set_xlabel("Permutation Importance")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.15)
        st.pyplot(fig, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with bottom_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Interview Talking Points")
        st.markdown(
            """
            - The holdout split is chronological, not random, to respect time order.
            - Target conditioning was tested with `log1p`, but the raw target performed better.
            - Multicollinearity is expected in OHLC data, so the project uses both correlation inspection and regularized models.
            - The deployed app is intentionally framed as a monthly close estimation tool, not a trading signal generator.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_predictor_tab(assets: Dict[str, object]) -> None:
    raw_df = assets["raw_df"]
    final_model = assets["final_model"]
    default_row = raw_df.iloc[-1]
    suggested_date = next_forecast_date(raw_df)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Scenario-Based Close Price Estimator")
    st.markdown(
        "Enter a monthly price scenario to estimate the closing price. This works best when you already have a plausible range for that month."
    )

    col1, col2 = st.columns(2)
    with col1:
        forecast_date = st.date_input(
            "Forecast month",
            value=suggested_date.to_pydatetime().date(),
            help="The app uses the month and the recent close history to build the feature row.",
        )
        open_price = st.number_input("Open price", min_value=0.01, value=float(default_row["Open"]))
    with col2:
        high_price = st.number_input("High price", min_value=0.01, value=float(default_row["High"]))
        low_price = st.number_input("Low price", min_value=0.01, value=float(default_row["Low"]))

    if st.button("Predict Closing Price", type="primary"):
        try:
            prediction = predict_close_price(
                fitted_pipeline=final_model,
                history_df=raw_df,
                forecast_date=pd.Timestamp(forecast_date),
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                feature_columns=FEATURE_COLUMNS,
            )
            expected_return = ((prediction - open_price) / open_price) * 100 if open_price else 0.0
            pred_cols = st.columns(3)
            pred_cols[0].metric("Predicted Close", "{0:.2f}".format(prediction))
            pred_cols[1].metric("Expected Return vs Open", "{0:.2f}%".format(expected_return))
            pred_cols[2].metric("Monthly Range", "{0:.2f}".format(high_price - low_price))
            st.success(
                "Scenario estimated successfully. Use this to discuss how the model reacts to different monthly trading ranges."
            )
        except Exception as exc:  # pylint: disable=broad-except
            st.error(str(exc))

    st.caption(
        "Because the model uses same-month Open/High/Low information, this predictor is best presented as a scenario estimator rather than a pure ahead-of-time forecast."
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_genai_tab(assets: Dict[str, object]) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Azure GenAI Copilot")
    st.markdown(
        "Use Azure OpenAI to answer interview questions about the project, the model choice, and the business insights."
    )

    if "azure_messages" not in st.session_state:
        st.session_state.azure_messages = []

    if not azure_assistant_enabled():
        if OpenAI is None:
            st.info(
                "Install the `openai` package from `requirements.txt`, then add the Azure environment variables to enable the chat assistant."
            )
        else:
            st.info(
                "Add `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_DEPLOYMENT_NAME` to enable the chat assistant."
            )
        st.code(
            "\n".join(
                [
                    "pip install -r requirements.txt",
                    "AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE-NAME.openai.azure.com",
                    "AZURE_OPENAI_API_KEY=your_key_here",
                    "AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-mini",
                ]
            ),
            language="bash",
        )
    else:
        for message in st.session_state.azure_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        prompt = st.chat_input("Ask about the project, model trade-offs, or deployment story")
        if prompt:
            st.session_state.azure_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            try:
                answer = ask_azure_assistant(prompt, assets, st.session_state.azure_messages)
            except Exception as exc:  # pylint: disable=broad-except
                answer = "Azure assistant error: {0}".format(exc)

            st.session_state.azure_messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)

    st.markdown("</div>", unsafe_allow_html=True)


def render_deployment_tab(assets: Dict[str, object]) -> None:
    final_payload = assets["final_payload"]
    coefficient_df = assets["coefficient_df"]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Deployment Story")
    st.markdown(
        """
        1. Train the regularized regression pipeline on the cleaned monthly stock dataset.
        2. Serve the model through Streamlit for interactive scenario prediction.
        3. Add Azure OpenAI to answer natural-language questions about model logic and business implications.
        4. Deploy on Streamlit Community Cloud, Azure App Service, or a container platform after setting the Azure environment variables.
        """
    )

    st.markdown("**Best tuned parameters**")
    st.json(final_payload["best_params"])

    st.markdown("**Most influential coefficients**")
    st.dataframe(
        coefficient_df.head(8).style.format(
            {"coefficient": "{:.4f}", "abs_coefficient": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_styles()
    assets = get_project_assets()

    st.markdown(
        """
        <div class="hero">
            <h1>Regression - Yes Bank Stock Closing Price Prediction</h1>
            <p>Machine Learning &amp; GenAI with Microsoft Azure</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "Executive Dashboard",
            "Scenario Predictor",
            "Azure GenAI Copilot",
            "Deployment Notes",
        ]
    )

    with tabs[0]:
        render_overview_tab(assets)
    with tabs[1]:
        render_predictor_tab(assets)
    with tabs[2]:
        render_genai_tab(assets)
    with tabs[3]:
        render_deployment_tab(assets)


if __name__ == "__main__":
    main()
