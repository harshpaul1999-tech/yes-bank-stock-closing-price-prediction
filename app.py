from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

try:
    from google import genai
except ModuleNotFoundError:
    genai = None

import dashboard_view  # noqa: E402
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
    resolve_default_data_path,
)


st.set_page_config(
    page_title="Yes Bank Stock Closing Price Prediction",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 210, 120, 0.25), transparent 28%),
                radial-gradient(circle at top right, rgba(25, 99, 156, 0.20), transparent 30%),
                linear-gradient(180deg, #f8f4ee 0%, #eef3f8 100%);
            color: #112336;
            font-family: "Aptos", "Segoe UI", sans-serif;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: #102738;
        }
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp span,
        .stApp div {
            color: #173047;
        }
        .hero, .hero h1, .hero p {
            color: #f7f9fc !important;
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #102738 0%, #174e72 58%, #206b81 100%);
            color: #f7f9fc;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 18px 42px rgba(16, 39, 56, 0.24);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.15;
        }
        .hero p {
            margin: 0.45rem 0 0;
            color: #d7e5f1;
            font-size: 1rem;
        }
        .section-card {
            background: rgba(255, 255, 255, 0.80);
            border: 1px solid rgba(17, 35, 54, 0.08);
            border-radius: 18px;
            padding: 1rem 1.15rem;
            box-shadow: 0 10px 24px rgba(20, 40, 60, 0.08);
            margin-bottom: 1rem;
        }
        .mini-note {
            color: #516273;
            font-size: 0.92rem;
        }
        div[data-baseweb="tab-list"] button {
            color: #173047 !important;
            font-weight: 700 !important;
        }
        div[data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #0d4672 !important;
            background: rgba(23, 78, 114, 0.10) !important;
            border-radius: 10px !important;
        }
        label[data-testid="stWidgetLabel"] p,
        .stDateInput label p,
        .stNumberInput label p,
        .stTextInput label p,
        .stSelectbox label p {
            color: #173047 !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricLabel"] p {
            color: #355066 !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #102738 !important;
        }
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea,
        .stTextInput input,
        .stTextArea textarea {
            color: #102738 !important;
            -webkit-text-fill-color: #102738 !important;
        }
        .stMarkdown a {
            color: #0d5b8f !important;
        }
        .stDataFrame, .stTable {
            color: #102738 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_raw_data() -> pd.DataFrame:
    data_path = resolve_default_data_path(PROJECT_ROOT)
    return load_yes_bank_data(data_path)


@st.cache_resource(show_spinner=False)
def get_project_assets() -> Dict[str, object]:
    raw_df = get_raw_data()
    model_df = build_modeling_frame(raw_df)
    metrics_df, _, split_payload = evaluate_models(model_df, FEATURE_COLUMNS)
    final_model, final_payload, _ = fit_final_model(model_df, FEATURE_COLUMNS)
    holdout_results = pd.DataFrame(
        {
            "Date": final_payload["test_df"]["Date"].dt.strftime("%b-%Y").to_numpy(),
            "Actual": final_payload["y_test"].to_numpy(),
            "Predicted": final_payload["holdout_predictions"],
        }
    )
    holdout_results["Residual"] = holdout_results["Actual"] - holdout_results["Predicted"]
    holdout_results["Absolute Error"] = holdout_results["Residual"].abs()

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
        "split_payload": split_payload,
        "final_model": final_model,
        "final_payload": final_payload,
        "holdout_results": holdout_results,
        "permutation_df": permutation_df,
        "coefficient_df": coefficient_df,
    }


def gemini_enabled() -> bool:
    return genai is not None and bool(os.getenv("GEMINI_API_KEY"))


def get_gemini_client():
    if genai is None:
        raise ValueError(
            "The `google-genai` package is not installed. Install requirements.txt to enable Gemini."
        )
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY to enable the Gemini copilot.")
    return genai.Client(api_key=api_key)


def build_gemini_context(assets: Dict[str, object]) -> str:
    raw_df = assets["raw_df"]
    metrics_df = assets["metrics_df"]
    holdout_results = assets["holdout_results"]
    permutation_df = assets["permutation_df"]
    latest_row = raw_df.iloc[-1]

    return "\n".join(
        [
            "Project: Yes Bank Stock Closing Price Prediction",
            "Stack: Regression notebook, Streamlit dashboard, Gemini-powered Q&A",
            "Dataset rows: {0}".format(len(raw_df)),
            "Date range: {0:%b %Y} to {1:%b %Y}".format(raw_df["Date"].min(), raw_df["Date"].max()),
            "Latest actual close: {0:.2f} ({1:%b %Y})".format(latest_row["Close"], latest_row["Date"]),
            "Evaluation summary:",
            metrics_df.to_string(index=False),
            "Top feature importance rows:",
            permutation_df.head(8).to_string(index=False),
            "Recent holdout rows:",
            holdout_results.tail(6).to_string(index=False),
            "Important framing: this project estimates the monthly closing price using same-month OHLC behavior plus historical context, so it is closer to a scenario-based close estimation system than a pure ahead-of-time trading signal.",
        ]
    )


def ask_gemini(question: str, assets: Dict[str, object], history: List[Dict[str, str]]) -> str:
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    client = get_gemini_client()

    transcript = "\n".join(
        "{0}: {1}".format(item["role"].upper(), item["content"]) for item in history[-8:]
    )
    prompt = "\n\n".join(
        [
            "You are helping a candidate explain a Yes Bank stock closing price prediction project.",
            "Keep answers practical, crisp, and grounded in the supplied project facts.",
            build_gemini_context(assets),
            "Recent chat history:",
            transcript if transcript else "No previous messages.",
            "User question:",
            question,
        ]
    )

    response = client.models.generate_content(model=model_name, contents=prompt)
    return (response.text or "").strip()


def render_overview_metrics(assets: Dict[str, object]) -> None:
    raw_df = assets["raw_df"]
    final_payload = assets["final_payload"]
    latest_row = raw_df.iloc[-1]
    holdout_metrics = final_payload["holdout_metrics"]

    metric_columns = st.columns(5)
    metric_columns[0].metric("Rows", len(raw_df))
    metric_columns[1].metric("Latest Close", "{0:.2f}".format(latest_row["Close"]))
    metric_columns[2].metric("Best Holdout RMSE", "{0:.2f}".format(holdout_metrics["RMSE"]))
    metric_columns[3].metric("Best Holdout R²", "{0:.3f}".format(holdout_metrics["R2"]))
    metric_columns[4].metric("Forecast Month", "{0:%b %Y}".format(next_forecast_date(raw_df)))


def render_model_lab(assets: Dict[str, object]) -> None:
    raw_df = assets["raw_df"]
    metrics_df = assets["metrics_df"]
    final_model = assets["final_model"]
    final_payload = assets["final_payload"]
    coefficient_df = assets["coefficient_df"]
    permutation_df = assets["permutation_df"]
    holdout_results = assets["holdout_results"]
    default_row = raw_df.iloc[-1]
    suggested_date = next_forecast_date(raw_df)

    col1, col2 = st.columns((1, 1))

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Evaluation Summary")
        st.dataframe(
            metrics_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Linear and regularized linear models dominate the holdout summary, while KNN and tree-based models trail once chronology is preserved."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Best Model Parameters")
        st.json(final_payload["best_params"])
        st.markdown("**Top coefficient drivers**")
        st.dataframe(
            coefficient_df.head(8).style.format(
                {"coefficient": "{:.4f}", "abs_coefficient": "{:.4f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    col3, col4 = st.columns((1, 1))

    with col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Holdout Actual vs Predicted")
        st.dataframe(
            holdout_results.style.format(
                {
                    "Actual": "{:.2f}",
                    "Predicted": "{:.2f}",
                    "Residual": "{:.2f}",
                    "Absolute Error": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=420,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Permutation Feature Importance")
        st.dataframe(
            permutation_df.head(10).style.format(
                {"importance_mean": "{:.4f}", "importance_std": "{:.4f}"}
            ),
            use_container_width=True,
            hide_index=True,
            height=420,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Scenario-Based Closing Price Estimator")
    input_col1, input_col2 = st.columns(2)
    with input_col1:
        forecast_date = st.date_input(
            "Forecast month",
            value=suggested_date.to_pydatetime().date(),
        )
        open_price = st.number_input("Open price", min_value=0.01, value=float(default_row["Open"]))
    with input_col2:
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
            result_cols = st.columns(3)
            result_cols[0].metric("Predicted Close", "{0:.2f}".format(prediction))
            result_cols[1].metric(
                "Expected Return vs Open",
                "{0:.2f}%".format(((prediction - open_price) / open_price) * 100 if open_price else 0.0),
            )
            result_cols[2].metric("Monthly Range", "{0:.2f}".format(high_price - low_price))
        except Exception as exc:  # pylint: disable=broad-except
            st.error(str(exc))
    st.caption(
        "This estimator works best when the monthly open, high, and low are known or can be sensibly assumed."
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_gemini_tab(assets: Dict[str, object]) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Gemini Copilot")
    st.markdown(
        "Ask Gemini to explain the business context, model trade-offs, evaluation results, and project story in interview-ready language."
    )

    if "gemini_messages" not in st.session_state:
        st.session_state.gemini_messages = []

    if not gemini_enabled():
        st.info(
            "Install `google-genai` and set `GEMINI_API_KEY` to enable Gemini-powered Q&A."
        )
        st.code(
            "\n".join(
                [
                    "pip install -r requirements.txt",
                    "export GEMINI_API_KEY='your_api_key_here'",
                    "export GEMINI_MODEL_NAME='gemini-2.5-flash'",
                ]
            ),
            language="bash",
        )
    else:
        for message in st.session_state.gemini_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        prompt = st.chat_input("Ask Gemini about the notebook, metrics, findings, or conclusions")
        if prompt:
            st.session_state.gemini_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            try:
                answer = ask_gemini(prompt, assets, st.session_state.gemini_messages)
            except Exception as exc:  # pylint: disable=broad-except
                answer = "Gemini error: {0}".format(exc)

            st.session_state.gemini_messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)

    st.markdown("</div>", unsafe_allow_html=True)


def render_documents_tab() -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Project Deliverables")
    st.markdown(
        """
        - `Yes_bank_stock_closing_price_prediction.ipynb`
        - `summary.pdf`
        - `Yes_Bank Technical Documentation.pdf`
        - `data_YesBank_StockPrices.csv`
        - `dashboard/README.md`
        - `dashboard/dashboard_preview.png`
        - `dashboard_view.py`
        - `app.py`
        - `README.md`
        """
    )

    for filename in [
        "Yes_bank_stock_closing_price_prediction.ipynb",
        "summary.pdf",
        "Yes_Bank Technical Documentation.pdf",
        "data_YesBank_StockPrices.csv",
        "dashboard/dashboard_preview.png",
        "dashboard/README.md",
        "dashboard_view.py",
        "app.py",
        "README.md",
    ]:
        path = PROJECT_ROOT / filename
        if path.exists():
            with open(path, "rb") as handle:
                st.download_button(
                    label="Download {0}".format(filename),
                    data=handle.read(),
                    file_name=filename,
                    use_container_width=True,
                )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_styles()
    assets = get_project_assets()

    st.markdown(
        """
        <div class="hero">
            <h1>Yes Bank Stock Closing Price Prediction</h1>
            <p>Standalone Regression Capstone with Gemini-powered explanations and a presentation-heavy dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_overview_metrics(assets)

    tabs = st.tabs(
        [
            "Insight Dashboard",
            "Model Lab",
            "Gemini Copilot",
            "Project Deliverables",
        ]
    )

    with tabs[0]:
        dashboard_view.render_dashboard(assets)
    with tabs[1]:
        render_model_lab(assets)
    with tabs[2]:
        render_gemini_tab(assets)
    with tabs[3]:
        render_documents_tab()


if __name__ == "__main__":
    main()
