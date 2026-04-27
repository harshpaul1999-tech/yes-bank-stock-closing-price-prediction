from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "yes_bank_ml.py"
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "regression_yes_bank_stock_closing_price_prediction.ipynb"
)


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _to_lines(source),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _to_lines(source),
    }


def _to_lines(source: str) -> list:
    cleaned = textwrap.dedent(source).strip("\n") + "\n"
    return cleaned.splitlines(keepends=True)


def build_cells() -> list:
    helper_source = MODULE_PATH.read_text()

    return [
        md_cell(
            """
            # **Project Name**
            - **Regression - Yes Bank Stock Closing Price Prediction**

            #### **Project Type**
            - Regression

            #### **Contribution**
            - Individual

            #### **Project Theme**
            - **Machine Learning & GenAI with Microsoft Azure**
            """
        ),
        md_cell(
            """
            # **Problem Statement**

            Yes Bank is one of the well-known private sector banks in India. The bank saw sharp market sentiment changes after the governance concerns and fraud-related news cycle that intensified from 2018 onward. This creates an interesting modeling challenge because stock behavior can shift abruptly during stress periods.

            The primary objective of this project is to **predict the monthly closing price of Yes Bank stock** using the monthly `Open`, `High`, `Low`, and engineered time-based features.
            """
        ),
        md_cell(
            """
            # **Business Context**

            - Investors and analysts want to understand how market range information relates to the month-end close.
            - The project also explores whether a standard regression setup can still remain useful when the bank goes through structural breaks and abnormal sentiment.
            - From an interview perspective, the project demonstrates EDA, feature engineering, time-aware evaluation, regularization, explainability, and deployment readiness.
            """
        ),
        md_cell(
            """
            # **Project Checklist**

            1. Problem statement and business context
            2. Data understanding
            3. Dataset loading and cleanup
            4. EDA and storytelling
            5. Feature engineering and multicollinearity handling
            6. Target feature conditioning
            7. Train-test split, model fitting, testing, evaluation, and regularization
            8. Model explainability
            9. Streamlit + Microsoft Azure GenAI deployment plan
            10. Conclusion and Git commit checkpoints
            """
        ),
        md_cell("# ***Let's Begin !***"),
        md_cell("## ***1. Know Your Data***"),
        md_cell("### Import Libraries and Configure the Notebook"),
        code_cell(
            """
            import io
            import warnings
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from IPython.display import display

            warnings.filterwarnings("ignore")
            plt.style.use("seaborn-v0_8-whitegrid")
            sns.set_theme(style="whitegrid")

            pd.set_option("display.max_columns", 100)
            pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
            """
        ),
        md_cell("### Project Helper Functions"),
        code_cell(helper_source),
        md_cell("### Dataset Loading"),
        code_cell(
            """
            candidate_roots = [
                Path.cwd(),
                Path.cwd().parent,
                Path.cwd() / "yes_bank_regression_azure",
                Path.cwd().parent / "yes_bank_regression_azure",
            ]

            PROJECT_ROOT = None
            for candidate in candidate_roots:
                if (candidate / "data" / "yes_bank_stock_prices.csv").exists():
                    PROJECT_ROOT = candidate
                    break

            if PROJECT_ROOT is None:
                raise FileNotFoundError(
                    "Could not locate the project root. Make sure the notebook is opened with the project folder available."
                )

            DATA_PATH = PROJECT_ROOT / "data" / "yes_bank_stock_prices.csv"
            raw_df = load_yes_bank_data(DATA_PATH)

            print("Project root:", PROJECT_ROOT)
            print("Dataset path:", DATA_PATH)
            """
        ),
        md_cell("### Dataset First View"),
        code_cell(
            """
            raw_df.head()
            """
        ),
        md_cell("### Dataset Rows & Columns Count"),
        code_cell(
            """
            print(f"Rows    : {raw_df.shape[0]}")
            print(f"Columns : {raw_df.shape[1]}")
            """
        ),
        md_cell("### Dataset Information"),
        code_cell(
            """
            info_buffer = io.StringIO()
            raw_df.info(buf=info_buffer)
            print(info_buffer.getvalue())
            """
        ),
        md_cell("#### Duplicate Values"),
        code_cell(
            """
            duplicate_count = raw_df.duplicated().sum()
            print(f"Duplicate rows: {duplicate_count}")
            """
        ),
        md_cell("#### Missing Values / Null Values"),
        code_cell(
            """
            missing_df = raw_df.isna().sum().to_frame("missing_values")
            missing_df["missing_pct"] = (missing_df["missing_values"] / len(raw_df)) * 100
            missing_df
            """
        ),
        md_cell("### Statistical Summary"),
        code_cell(
            """
            raw_df.describe(include="all").transpose()
            """
        ),
        md_cell(
            """
            ### What did we learn from the raw dataset?

            - The dataset contains monthly stock records from **July 2005 to November 2020**.
            - There are no missing values and no duplicate rows.
            - `Open`, `High`, `Low`, and `Close` are numeric and highly related, which is expected for OHLC market data.
            - Because this is a short monthly time series, evaluation must preserve chronology instead of using a random split.
            """
        ),
        md_cell("## ***2. Understanding Your Variables***"),
        md_cell("### Variables Description"),
        code_cell(
            """
            variable_description = pd.DataFrame(
                {
                    "Variable": ["Date", "Open", "High", "Low", "Close"],
                    "Description": [
                        "Month of the observation",
                        "Opening stock price for the month",
                        "Highest stock price touched during the month",
                        "Lowest stock price touched during the month",
                        "Closing stock price for the month (target variable)",
                    ],
                }
            )
            variable_description
            """
        ),
        md_cell("### Target Variable Overview"),
        code_cell(
            """
            target_summary = pd.DataFrame(
                {
                    "metric": ["min", "max", "mean", "median", "skewness"],
                    "value": [
                        raw_df["Close"].min(),
                        raw_df["Close"].max(),
                        raw_df["Close"].mean(),
                        raw_df["Close"].median(),
                        raw_df["Close"].skew(),
                    ],
                }
            )
            target_summary
            """
        ),
        md_cell(
            """
            The closing price is positively skewed because the stock experienced a dramatic rise before the later correction phases. We will test target conditioning later, but we should keep the business interpretation on the original rupee scale whenever possible.
            """
        ),
        md_cell("## ***3. Dataset Cleanup and Wrangling***"),
        md_cell("### Data Wrangling Code"),
        code_cell(
            """
            clean_df = raw_df.copy()
            clean_df = clean_df.sort_values("Date").reset_index(drop=True)
            clean_df["YearMonth"] = clean_df["Date"].dt.to_period("M").astype(str)

            print("Is the dataset chronologically sorted?", clean_df["Date"].is_monotonic_increasing)
            clean_df.head()
            """
        ),
        md_cell(
            """
            ### What cleanup did we perform?

            - Converted the `Date` column to proper datetime format.
            - Sorted the records chronologically.
            - Created a readable monthly label for later charting.
            - Since there were no missing values or duplicates, the cleanup stage was intentionally light.
            """
        ),
        md_cell(
            """
            ## ***4. EDA, Storytelling, and Chart Experiments***

            The goal of EDA here is not just plotting charts, but understanding trend shifts, volatility bursts, seasonality, and the effect of structurally different market periods.
            """
        ),
        md_cell("#### Chart 1 - Monthly Closing Price Trend"),
        code_cell(
            """
            plt.figure(figsize=(14, 5))
            plt.plot(clean_df["Date"], clean_df["Close"], color="#123b5d", linewidth=2.2, label="Close")
            plt.axvline(pd.Timestamp("2018-01-01"), linestyle="--", linewidth=1.4, color="#a43a2a", label="2018 stress period")
            plt.title("Yes Bank Monthly Closing Price Trend")
            plt.xlabel("Date")
            plt.ylabel("Closing Price")
            plt.legend()
            plt.show()
            """
        ),
        md_cell(
            """
            **Insight:** The stock saw a long growth phase followed by a sharp correction, especially after the 2018 stress period. This makes it important to include event-aware features instead of treating the full series as perfectly stable.
            """
        ),
        md_cell("#### Chart 2 - Open, High, Low, and Close Comparison"),
        code_cell(
            """
            plt.figure(figsize=(14, 5))
            plt.plot(clean_df["Date"], clean_df["Open"], label="Open", linewidth=1.8, color="#305f72")
            plt.plot(clean_df["Date"], clean_df["High"], label="High", linewidth=1.6, color="#e09f3e")
            plt.plot(clean_df["Date"], clean_df["Low"], label="Low", linewidth=1.6, color="#8c2f39")
            plt.plot(clean_df["Date"], clean_df["Close"], label="Close", linewidth=2.0, color="#1d3557")
            plt.title("OHLC Behavior Over Time")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(ncol=4)
            plt.show()
            """
        ),
        md_cell(
            """
            **Insight:** The OHLC series move together very closely, which is useful for prediction but also creates strong multicollinearity. We will inspect that explicitly before modeling.
            """
        ),
        md_cell("#### Chart 3 - Seasonality View by Month"),
        code_cell(
            """
            seasonality_df = clean_df.copy()
            seasonality_df["Month"] = seasonality_df["Date"].dt.strftime("%b")
            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            plt.figure(figsize=(13, 5))
            sns.boxplot(data=seasonality_df, x="Month", y="Close", order=month_order, color="#5c88b4")
            plt.title("Distribution of Closing Price by Calendar Month")
            plt.xlabel("Month")
            plt.ylabel("Closing Price")
            plt.show()
            """
        ),
        md_cell(
            """
            **Insight:** There is no clean seasonal pattern strong enough to dominate the series, but month encoding can still help the model capture softer cyclical effects.
            """
        ),
        md_cell("#### Chart 4 - Moving Averages and Volatility Proxy"),
        code_cell(
            """
            eda_df = clean_df.copy()
            eda_df["close_ma_3"] = eda_df["Close"].rolling(3).mean()
            eda_df["close_ma_12"] = eda_df["Close"].rolling(12).mean()
            eda_df["range_value"] = eda_df["High"] - eda_df["Low"]

            fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

            axes[0].plot(eda_df["Date"], eda_df["Close"], label="Close", color="#143d59", linewidth=2)
            axes[0].plot(eda_df["Date"], eda_df["close_ma_3"], label="3M Moving Average", color="#f4b942", linewidth=1.8)
            axes[0].plot(eda_df["Date"], eda_df["close_ma_12"], label="12M Moving Average", color="#6c7a89", linewidth=1.8)
            axes[0].set_title("Trend Smoothing with Moving Averages")
            axes[0].legend()

            axes[1].bar(eda_df["Date"], eda_df["range_value"], color="#9d4d4d", alpha=0.8)
            axes[1].set_title("Monthly Trading Range (High - Low)")
            axes[1].set_ylabel("Range")

            plt.tight_layout()
            plt.show()
            """
        ),
        md_cell(
            """
            **Insight:** Moving averages smooth the long-term trend, while the monthly range highlights the stress periods where volatility expanded materially.
            """
        ),
        md_cell("#### Chart 5 - Pre-2018 vs Post-2018 Closing Price Distribution"),
        code_cell(
            """
            comparison_df = clean_df.copy()
            comparison_df["Regime"] = np.where(comparison_df["Date"] < pd.Timestamp("2018-01-01"), "Before 2018", "2018 and After")

            plt.figure(figsize=(8, 5))
            sns.boxplot(data=comparison_df, x="Regime", y="Close", palette=["#7a9e9f", "#bc4b51"])
            plt.title("Closing Price Distribution Across Market Regimes")
            plt.xlabel("")
            plt.ylabel("Closing Price")
            plt.show()
            """
        ),
        md_cell(
            """
            **Insight:** The distribution changes visibly after 2018, which supports the decision to add structural-break indicator variables in the feature engineering stage.
            """
        ),
        md_cell("#### Chart 6 - Correlation Heatmap"),
        code_cell(
            """
            plt.figure(figsize=(8, 6))
            sns.heatmap(clean_df[["Open", "High", "Low", "Close"]].corr(), annot=True, cmap="Blues", fmt=".2f")
            plt.title("Correlation Heatmap of Raw Price Variables")
            plt.show()
            """
        ),
        md_cell(
            """
            **Insight:** All core price variables are strongly correlated, which is useful for predictive accuracy but needs thoughtful handling to keep the final model stable and interpretable.
            """
        ),
        md_cell("## ***5. Feature Engineering and Pre-processing***"),
        md_cell("### Feature Creation, Encoding, and Modeling Frame"),
        code_cell(
            """
            model_df = build_modeling_frame(clean_df)
            print("Modeling rows available after lag/rolling feature creation:", len(model_df))
            print("Total engineered features used:", len(FEATURE_COLUMNS))
            model_df.head()
            """
        ),
        md_cell(
            """
            **Feature engineering choices made in this project**

            - Time features: `year`, `time_index`
            - Cyclical encoding: `month_sin`, `month_cos`
            - Price spread features: `range_value`, `range_pct`, `high_open_gap`, `open_low_gap`
            - Lag features: prior monthly closes
            - Rolling features: 3-month and 6-month moving averages and standard deviations
            - Structural break indicators: `post_2018_crisis`, `covid_shock`
            """
        ),
        md_cell("### Multicollinearity Inspection"),
        code_cell(
            """
            correlation_pairs = high_correlation_pairs(model_df, FEATURE_COLUMNS, threshold=0.95)
            correlation_pairs.head(15)
            """
        ),
        md_cell(
            """
            **Multicollinearity handling strategy**

            - OHLC and lag features are naturally correlated in financial data.
            - Instead of dropping every strong predictor, we keep business-relevant variables and control coefficient instability through **L1/L2 regularization**.
            - This gives a better interview narrative than blindly deleting most of the signal.
            """
        ),
        md_cell("### Target Feature Conditioning"),
        code_cell(
            """
            conditioning_df = evaluate_target_conditioning(model_df, FEATURE_COLUMNS, alpha=0.2)
            conditioning_df
            """
        ),
        md_cell(
            """
            **Decision:** The raw target performs better than `log1p(Close)` on the holdout set, so the final model keeps the target in the original scale for easier business interpretation.
            """
        ),
        md_cell("### Train-Test Split and Scaling Strategy"),
        code_cell(
            """
            x_train, x_test, y_train, y_test, train_df, test_df = time_aware_split(model_df, FEATURE_COLUMNS, test_ratio=0.2)

            split_summary = pd.DataFrame(
                {
                    "split": ["Train", "Test"],
                    "start_date": [train_df["Date"].min(), test_df["Date"].min()],
                    "end_date": [train_df["Date"].max(), test_df["Date"].max()],
                    "rows": [len(train_df), len(test_df)],
                }
            )
            split_summary
            """
        ),
        md_cell(
            """
            The split is chronological, not random. Feature scaling is applied inside the model pipeline, which avoids data leakage from the test period.
            """
        ),
        md_cell("## ***6. Model Implementation***"),
        md_cell("### Baseline and Candidate Model Comparison"),
        code_cell(
            """
            metrics_df, fitted_models, split_payload = evaluate_models(model_df, FEATURE_COLUMNS, test_ratio=0.2)
            metrics_df
            """
        ),
        md_cell(
            """
            **Modeling takeaway:** Regularized linear models outperform the tree-based models here. That means the relationship between the closing price and the engineered monthly price features is mostly linear and very strong.
            """
        ),
        md_cell("### Time-Series-Only vs Same-Month Estimation Framing"),
        code_cell(
            """
            lag_only_features = [
                "close_lag_1",
                "close_lag_3",
                "close_lag_6",
                "close_ma_3",
                "close_ma_6",
                "close_std_3",
                "close_std_6",
                "month_sin",
                "month_cos",
                "year",
                "time_index",
                "post_2018_crisis",
                "covid_shock",
            ]

            lag_only_metrics, _, _ = evaluate_models(model_df, lag_only_features, test_ratio=0.2)

            framing_comparison = pd.DataFrame(
                {
                    "Framing": ["Final feature set", "Lag-only feature set"],
                    "Lasso RMSE": [
                        metrics_df.loc[metrics_df["Model"] == "Lasso Regression", "RMSE"].iloc[0],
                        lag_only_metrics.loc[lag_only_metrics["Model"] == "Lasso Regression", "RMSE"].iloc[0],
                    ],
                    "Lasso R2": [
                        metrics_df.loc[metrics_df["Model"] == "Lasso Regression", "R2"].iloc[0],
                        lag_only_metrics.loc[lag_only_metrics["Model"] == "Lasso Regression", "R2"].iloc[0],
                    ],
                }
            )
            framing_comparison
            """
        ),
        md_cell(
            """
            **Interpretation:** A pure lag-only setup performs noticeably worse. That is why the final project should be presented as a **monthly close estimation system** using the month's trading range, not as a pure ahead-of-time trading forecast.
            """
        ),
        md_cell("### Hyperparameter Tuning and Regularization"),
        code_cell(
            """
            tuning_df, tuned_searches = tune_regularized_models(model_df, FEATURE_COLUMNS)
            tuning_df
            """
        ),
        md_cell(
            """
            **Why regularization matters here**

            - The data contains many correlated predictors.
            - Lasso helps shrink weaker coefficients and improves interpretability.
            - Ridge and ElasticNet act as stability-oriented alternatives.
            """
        ),
        md_cell("### Final Model Fit"),
        code_cell(
            """
            final_model, final_payload, tuned_searches = fit_final_model(model_df, FEATURE_COLUMNS)

            final_metrics_df = pd.DataFrame([final_payload["holdout_metrics"]])
            final_metrics_df
            """
        ),
        md_cell("### Actual vs Predicted Performance on the Holdout Set"),
        code_cell(
            """
            holdout_results = pd.DataFrame(
                {
                    "Date": final_payload["test_df"]["Date"].to_numpy(),
                    "Actual": final_payload["y_test"].to_numpy(),
                    "Predicted": final_payload["holdout_predictions"],
                }
            )
            holdout_results["Residual"] = holdout_results["Actual"] - holdout_results["Predicted"]
            holdout_results.head()
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            axes[0].plot(holdout_results["Date"], holdout_results["Actual"], label="Actual", linewidth=2, color="#123b5d")
            axes[0].plot(holdout_results["Date"], holdout_results["Predicted"], label="Predicted", linewidth=2, color="#d3872b")
            axes[0].set_title("Actual vs Predicted Close Price")
            axes[0].legend()

            axes[1].scatter(holdout_results["Predicted"], holdout_results["Residual"], color="#8f3b45", alpha=0.8)
            axes[1].axhline(0, linestyle="--", color="black", linewidth=1)
            axes[1].set_title("Residual Plot")
            axes[1].set_xlabel("Predicted Close")
            axes[1].set_ylabel("Residual")

            plt.tight_layout()
            plt.show()
            """
        ),
        md_cell("## ***7. Model Explainability***"),
        md_cell("### Coefficient-Based Importance and Permutation Importance"),
        code_cell(
            """
            coefficient_df = model_coefficients(final_model, FEATURE_COLUMNS)
            importance_df = permutation_feature_importance(
                final_model,
                final_payload["x_test"],
                final_payload["y_test"],
                FEATURE_COLUMNS,
            )

            display(coefficient_df.head(10))
            display(importance_df.head(10))
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))

            coef_plot = coefficient_df.head(8).iloc[::-1]
            axes[0].barh(coef_plot["feature"], coef_plot["abs_coefficient"], color="#19486a")
            axes[0].set_title("Top Absolute Coefficients")
            axes[0].set_xlabel("Absolute Coefficient")

            perm_plot = importance_df.head(8).iloc[::-1]
            axes[1].barh(perm_plot["feature"], perm_plot["importance_mean"], color="#d3872b")
            axes[1].set_title("Top Permutation Importances")
            axes[1].set_xlabel("Importance Mean")

            plt.tight_layout()
            plt.show()
            """
        ),
        md_cell(
            """
            **Explainability summary**

            - Current-month price levels and spread-related features dominate prediction quality.
            - Structural break indicators help the model account for post-2018 and post-COVID behavior shifts.
            - The final Lasso model is explainable enough for interview discussion while still giving strong holdout performance.
            """
        ),
        md_cell("## ***8. Deployment - Streamlit + Microsoft Azure GenAI***"),
        md_cell(
            """
            ### Deployment Strategy

            - Build an interactive Streamlit app for historical trend analysis, model metrics, and scenario-based prediction.
            - Integrate **Azure OpenAI** so users can ask natural-language questions such as:
              - Why was Lasso chosen?
              - What changed after 2018?
              - Is this a forecast or an estimation model?
            - Keep Azure credentials in environment variables rather than hardcoding them.
            """
        ),
        code_cell(
            """
            print("Streamlit launch command:")
            print("streamlit run app.py")

            print("\\nRecommended Azure environment variables:")
            print("AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE-NAME.openai.azure.com")
            print("AZURE_OPENAI_API_KEY=your_api_key_here")
            print("AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-mini")
            """
        ),
        md_cell("## ***9. Conclusion***"),
        md_cell(
            """
            - The dataset is clean, compact, and highly informative despite having only monthly observations.
            - Yes Bank shows a visible regime shift after 2018, so event-aware features improve the storytelling and modeling logic.
            - Regularized linear models outperform tree-based models for this problem.
            - A lag-only, pure time-series framing is weaker than the final feature set, so the best business framing is **monthly close estimation** using the month's trading range.
            - The project is deployment-ready through Streamlit, and Azure GenAI adds a strong presentation layer for interviews.
            """
        ),
        md_cell("## ***10. Suggested Git Commit Checkpoints***"),
        md_cell(
            """
            1. `git commit -m "feat: add yes bank stock dataset and project scaffold"`
            2. `git commit -m "feat: complete yes bank eda and feature engineering notebook sections"`
            3. `git commit -m "feat: add regression modeling, tuning, and explainability workflow"`
            4. `git commit -m "feat: build streamlit app with azure genai integration"`
            5. `git commit -m "docs: add project readme and deployment notes"`
            """
        ),
        md_cell("### ***Hurrah! You have successfully completed the project notebook structure.***"),
    ]


def main() -> None:
    notebook = {
        "cells": build_cells(),
        "metadata": {
            "colab": {
                "name": "Regression - Yes Bank Stock Closing Price Prediction.ipynb",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2))
    print("Notebook written to:", NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
