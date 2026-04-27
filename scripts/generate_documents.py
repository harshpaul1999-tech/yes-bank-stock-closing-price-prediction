#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from yes_bank_ml import load_yes_bank_data, resolve_default_data_path  # noqa: E402


NOTEBOOK_NAME = "Yes_bank_stock_closing_price_prediction.ipynb"
SUMMARY_PDF = PROJECT_ROOT / "summary.pdf"
TECHNICAL_PDF = PROJECT_ROOT / "Yes_Bank Technical Documentation.pdf"
SEED = 42

warnings.filterwarnings("ignore", category=RuntimeWarning)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_true, y_pred)),
    }


def engineer_capstone_features(frame: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    data = frame.copy()
    data["month"] = data["Date"].dt.month
    data["year"] = data["Date"].dt.year
    data["time_index"] = np.arange(len(data))
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)
    data["range_value"] = data["High"] - data["Low"]
    data["range_pct"] = np.where(data["Open"] != 0, data["range_value"] / data["Open"], 0.0)
    data["high_open_gap"] = data["High"] - data["Open"]
    data["open_low_gap"] = data["Open"] - data["Low"]
    data["post_2018_crisis"] = (data["Date"] >= pd.Timestamp("2018-01-01")).astype(int)
    data["covid_shock"] = (data["Date"] >= pd.Timestamp("2020-03-01")).astype(int)

    for lag in (1, 3, 6):
        data[f"close_lag_{lag}"] = data["Close"].shift(lag)
    for window in (3, 6):
        shifted_close = data["Close"].shift(1)
        data[f"close_ma_{window}"] = shifted_close.rolling(window=window).mean()
        data[f"close_std_{window}"] = shifted_close.rolling(window=window).std()

    feature_columns = [
        "Open",
        "High",
        "Low",
        "range_value",
        "range_pct",
        "high_open_gap",
        "open_low_gap",
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
    modeling_df = data[["Date", "Close"] + feature_columns].dropna().reset_index(drop=True)
    return modeling_df, feature_columns


def evaluate_core_models(modeling_df: pd.DataFrame, feature_columns: list[str]) -> Dict[str, object]:
    split_index = int(len(modeling_df) * 0.8)
    train_df = modeling_df.iloc[:split_index].copy()
    test_df = modeling_df.iloc[split_index:].copy()

    x_train = train_df[feature_columns]
    x_test = test_df[feature_columns]
    y_train = train_df["Close"]
    y_test = test_df["Close"]
    cv = TimeSeriesSplit(n_splits=4)

    models = {
        "Linear Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "KNN Regressor": GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())]),
            param_grid={"model__n_neighbors": [2, 3, 4, 5, 6, 7]},
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
        ),
        "Random Forest": GridSearchCV(
            RandomForestRegressor(random_state=SEED),
            param_grid={
                "n_estimators": [150, 300],
                "max_depth": [5, 8, None],
                "min_samples_leaf": [1, 2],
            },
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
        ),
        "Ridge Regression": GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
            param_grid={"model__alpha": [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]},
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
        ),
        "Lasso Regression": GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=20000))]),
            param_grid={"model__alpha": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]},
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
        ),
        "ElasticNet Regression": GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(max_iter=20000))]),
            param_grid={
                "model__alpha": [0.01, 0.05, 0.1, 0.2, 0.5],
                "model__l1_ratio": [0.3, 0.5, 0.7, 0.9],
            },
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
        ),
    }

    evaluation_rows = []
    model_store: Dict[str, object] = {}
    predictions_store: Dict[str, np.ndarray] = {}
    best_params_store: Dict[str, object] = {}

    for model_name, model_object in models.items():
        model_object.fit(x_train, y_train)
        if hasattr(model_object, "best_estimator_"):
            best_model = model_object.best_estimator_
            best_params = model_object.best_params_
        else:
            best_model = model_object
            best_params = {}

        y_pred = best_model.predict(x_test)
        metrics = regression_metrics(y_test, y_pred)
        metrics["Model"] = model_name
        metrics["Best Params"] = str(best_params) if best_params else "Default"
        evaluation_rows.append(metrics)
        model_store[model_name] = best_model
        predictions_store[model_name] = y_pred
        best_params_store[model_name] = best_params

    evaluation_summary = pd.DataFrame(evaluation_rows)[["Model", "MAE", "RMSE", "R2", "Best Params"]]
    evaluation_summary = evaluation_summary.sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    return {
        "evaluation_summary": evaluation_summary,
        "model_store": model_store,
        "predictions_store": predictions_store,
        "best_params_store": best_params_store,
        "train_df": train_df,
        "test_df": test_df,
        "x_test": x_test,
        "y_test": y_test,
    }


def evaluate_target_conditioning(modeling_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    split_index = int(len(modeling_df) * 0.8)
    train_df = modeling_df.iloc[:split_index].copy()
    test_df = modeling_df.iloc[split_index:].copy()
    x_train = train_df[feature_columns]
    x_test = test_df[feature_columns]
    y_train = train_df["Close"]
    y_test = test_df["Close"]

    rows = []
    raw_model = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.2, max_iter=20000))])
    raw_model.fit(x_train, y_train)
    raw_pred = raw_model.predict(x_test)
    raw_metrics = regression_metrics(y_test, raw_pred)
    raw_metrics["Target Strategy"] = "Raw Close"
    rows.append(raw_metrics)

    log_model = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.2, max_iter=20000))])
    log_model.fit(x_train, np.log1p(y_train))
    log_pred = np.expm1(log_model.predict(x_test))
    log_metrics = regression_metrics(y_test, log_pred)
    log_metrics["Target Strategy"] = "log1p(Close)"
    rows.append(log_metrics)

    return pd.DataFrame(rows)[["Target Strategy", "MAE", "RMSE", "R2"]].sort_values("RMSE")


def build_payload() -> Dict[str, object]:
    raw_df = load_yes_bank_data(resolve_default_data_path(PROJECT_ROOT))
    raw_df["Range_Value"] = raw_df["High"] - raw_df["Low"]
    raw_df["Intramonth_Return_Pct"] = np.where(
        raw_df["Open"] != 0,
        ((raw_df["Close"] - raw_df["Open"]) / raw_df["Open"]) * 100,
        0.0,
    )
    raw_df["Regime"] = np.where(raw_df["Date"] < pd.Timestamp("2018-01-01"), "Before 2018", "2018 onward")
    raw_df["MA_3"] = raw_df["Close"].rolling(3).mean()
    raw_df["MA_12"] = raw_df["Close"].rolling(12).mean()

    modeling_df, feature_columns = engineer_capstone_features(raw_df)
    model_payload = evaluate_core_models(modeling_df, feature_columns)
    target_conditioning_df = evaluate_target_conditioning(modeling_df, feature_columns)

    evaluation_summary = model_payload["evaluation_summary"]
    lasso_model = model_payload["model_store"]["Lasso Regression"]
    lasso_coef = lasso_model.named_steps["model"].coef_
    coefficient_df = pd.DataFrame({"Feature": feature_columns, "Coefficient": lasso_coef})
    coefficient_df["Absolute Coefficient"] = coefficient_df["Coefficient"].abs()
    coefficient_df = coefficient_df.sort_values("Absolute Coefficient", ascending=False).reset_index(drop=True)

    holdout_results = pd.DataFrame(
        {
            "Date": model_payload["test_df"]["Date"].dt.strftime("%b-%Y").to_numpy(),
            "Actual": model_payload["y_test"].to_numpy(),
            "Predicted": model_payload["predictions_store"]["Lasso Regression"],
        }
    )
    holdout_results["Residual"] = holdout_results["Actual"] - holdout_results["Predicted"]
    holdout_results["Absolute Error"] = holdout_results["Residual"].abs()
    holdout_metrics = regression_metrics(
        model_payload["y_test"], model_payload["predictions_store"]["Lasso Regression"]
    )

    regime_summary = (
        raw_df.groupby("Regime")
        .agg(
            Avg_Close=("Close", "mean"),
            Avg_Range=("Range_Value", "mean"),
            Mean_Return_Pct=("Intramonth_Return_Pct", "mean"),
            Observations=("Date", "count"),
        )
        .round(2)
        .reset_index()
    )
    close_correlations = (
        raw_df[["Open", "High", "Low", "Close", "Range_Value", "Intramonth_Return_Pct"]]
        .corr()["Close"]
        .sort_values(ascending=False)
        .drop("Close")
        .round(4)
        .reset_index()
    )
    close_correlations.columns = ["Feature", "Correlation with Close"]
    top_volatility = (
        raw_df.nlargest(5, "Range_Value")[["Date", "Range_Value", "Close", "Intramonth_Return_Pct"]]
        .copy()
    )
    top_volatility["Date"] = top_volatility["Date"].dt.strftime("%b-%Y")
    top_volatility = top_volatility.round(2)

    high_corr_pairs = []
    corr_matrix = modeling_df[feature_columns].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    for column in upper_triangle.columns:
        for row_name, value in upper_triangle[column].dropna().items():
            if value >= 0.95:
                high_corr_pairs.append((row_name, column, round(float(value), 3)))
    high_corr_df = pd.DataFrame(high_corr_pairs, columns=["Feature 1", "Feature 2", "Correlation"]).head(12)

    return {
        "raw_df": raw_df,
        "modeling_df": modeling_df,
        "feature_columns": feature_columns,
        "evaluation_summary": evaluation_summary,
        "target_conditioning_df": target_conditioning_df,
        "coefficient_df": coefficient_df,
        "holdout_results": holdout_results,
        "holdout_metrics": holdout_metrics,
        "regime_summary": regime_summary,
        "close_correlations": close_correlations,
        "top_volatility": top_volatility,
        "high_corr_df": high_corr_df,
        "best_params_store": model_payload["best_params_store"],
    }


def _safe_text(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
    return str(value)


def df_table(frame: pd.DataFrame, col_widths=None) -> Table:
    table_data = [list(frame.columns)] + [[_safe_text(value) for value in row] for row in frame.to_numpy()]
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#143d59")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cad5df")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fa")]),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ProjectTitle",
            parent=styles["Title"],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#143d59"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#174d72"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyTextSmall",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#233645"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletSmall",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=13,
            leftIndent=10,
            bulletIndent=0,
            spaceAfter=4,
            textColor=colors.HexColor("#233645"),
        )
    )
    return styles


def bullet(text: str, styles) -> Paragraph:
    return Paragraph(f"• {text}", styles["BulletSmall"])


def generate_charts(payload: Dict[str, object], temp_dir: Path) -> Dict[str, Path]:
    plt.style.use("seaborn-v0_8-whitegrid")
    raw_df = payload["raw_df"]
    evaluation_summary = payload["evaluation_summary"]

    line_path = temp_dir / "close_trend.png"
    plt.figure(figsize=(9, 4))
    plt.plot(raw_df["Date"], raw_df["Close"], label="Close", color="#143d59", linewidth=2.3)
    plt.plot(raw_df["Date"], raw_df["MA_3"], label="3M Moving Avg", color="#f4b942", linewidth=1.8)
    plt.plot(raw_df["Date"], raw_df["MA_12"], label="12M Moving Avg", color="#6c7a89", linewidth=1.8)
    plt.axvline(pd.Timestamp("2018-01-01"), linestyle="--", color="#a43a2a", linewidth=1.2)
    plt.title("Closing Price Trend with Rolling Averages")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.legend()
    plt.tight_layout()
    plt.savefig(line_path, dpi=180)
    plt.close()

    heatmap_path = temp_dir / "correlation_heatmap.png"
    plt.figure(figsize=(7, 5))
    corr = raw_df[["Open", "High", "Low", "Close", "Range_Value", "Intramonth_Return_Pct"]].corr()
    plt.imshow(corr, cmap="Blues", aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    for row_idx in range(len(corr.index)):
        for col_idx in range(len(corr.columns)):
            plt.text(col_idx, row_idx, f"{corr.iloc[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(label="Correlation")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=180)
    plt.close()

    rmse_path = temp_dir / "model_rmse.png"
    rmse_frame = evaluation_summary[["Model", "RMSE"]].sort_values("RMSE").copy()
    plt.figure(figsize=(8, 4.5))
    plt.barh(rmse_frame["Model"], rmse_frame["RMSE"], color="#2c6e91")
    plt.title("Model Comparison by RMSE")
    plt.xlabel("RMSE")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(rmse_path, dpi=180)
    plt.close()

    return {"line": line_path, "heatmap": heatmap_path, "rmse": rmse_path}


def build_summary_pdf(payload: Dict[str, object], chart_paths: Dict[str, Path]) -> None:
    styles = make_styles()
    story = []
    raw_df = payload["raw_df"]
    evaluation_summary = payload["evaluation_summary"]
    regime_summary = payload["regime_summary"]
    holdout_metrics = payload["holdout_metrics"]
    close_correlations = payload["close_correlations"]

    story.append(Paragraph("Yes Bank Stock Closing Price Prediction - Summary", styles["ProjectTitle"]))
    story.append(
        Paragraph(
            "Standalone capstone summary aligned to the reference Yes Bank project, "
            "but updated with an original time-aware workflow, expanded EDA, and a Gemini-enabled dashboard experience.",
            styles["BodyTextSmall"],
        )
    )

    snapshot_df = pd.DataFrame(
        {
            "Metric": [
                "Dataset rows",
                "Date range",
                "Peak close",
                "Best raw correlation with close",
                "Selected deployment model",
                "Final Lasso holdout RMSE",
            ],
            "Value": [
                len(raw_df),
                f"{raw_df['Date'].min():%b %Y} to {raw_df['Date'].max():%b %Y}",
                f"{raw_df['Close'].max():.2f}",
                f"{close_correlations.iloc[0, 0]} ({close_correlations.iloc[0, 1]:.4f})",
                "Lasso Regression",
                f"{holdout_metrics['RMSE']:.2f}",
            ],
        }
    )
    story.append(Paragraph("Project Snapshot", styles["SectionTitle"]))
    story.append(df_table(snapshot_df, col_widths=[2.2 * inch, 4.2 * inch]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Image(str(chart_paths["line"]), width=6.8 * inch, height=3.0 * inch))

    story.append(Paragraph("Findings That Stay Close to the Reference Project", styles["SectionTitle"]))
    story.append(bullet("The dataset is clean, with no missing values and no duplicate rows.", styles))
    story.append(bullet("All price variables are positively skewed, and the raw OHLC variables remain highly correlated with closing price.", styles))
    story.append(bullet("The post-2018 period is meaningfully more volatile, so the broader reference story about crisis-era instability still clearly holds.", styles))
    story.append(bullet("Regularized linear models remain strong performers on the same dataset, which keeps the final project direction close to the reference repo.", styles))

    story.append(Paragraph("Evaluation Summary", styles["SectionTitle"]))
    summary_table = evaluation_summary[["Model", "MAE", "RMSE", "R2"]].round(4)
    story.append(df_table(summary_table, col_widths=[2.2 * inch, 1.0 * inch, 1.0 * inch, 0.8 * inch]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        Paragraph(
            "Linear Regression and the regularized linear family dominate the holdout comparison. "
            "For deployment, this project chooses Lasso because it stays highly accurate while handling multicollinearity more gracefully than plain linear regression.",
            styles["BodyTextSmall"],
        )
    )

    story.append(Paragraph("Regime Comparison", styles["SectionTitle"]))
    story.append(df_table(regime_summary, col_widths=[1.4 * inch, 1.0 * inch, 1.0 * inch, 1.2 * inch, 1.0 * inch]))
    story.append(
        Paragraph(
            "The regime table shows why the 2018 break matters: the average monthly range rises sharply and the mean intramonth return turns negative in the later phase.",
            styles["BodyTextSmall"],
        )
    )

    doc = SimpleDocTemplate(str(SUMMARY_PDF), pagesize=A4, rightMargin=32, leftMargin=32, topMargin=32, bottomMargin=32)
    doc.build(story)


def build_technical_pdf(payload: Dict[str, object], chart_paths: Dict[str, Path]) -> None:
    styles = make_styles()
    story = []
    raw_df = payload["raw_df"]
    target_conditioning_df = payload["target_conditioning_df"]
    evaluation_summary = payload["evaluation_summary"]
    high_corr_df = payload["high_corr_df"]
    coefficient_df = payload["coefficient_df"]
    holdout_results = payload["holdout_results"]
    close_correlations = payload["close_correlations"]
    top_volatility = payload["top_volatility"]
    best_params_store = payload["best_params_store"]

    story.append(Paragraph("Yes Bank Technical Documentation", styles["ProjectTitle"]))
    story.append(
        Paragraph(
            "This document describes the standalone Yes Bank stock closing price prediction project, "
            "including business framing, EDA, preprocessing, model comparison, evaluation summary, and final conclusions.",
            styles["BodyTextSmall"],
        )
    )

    story.append(Paragraph("1. Abstract", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "The project predicts monthly closing price using OHLC data and engineered time-aware features. "
            "The same dataset used by the reference project is retained, so the main EDA conclusions remain aligned: high correlation among price fields, visible positive skew, and strong multicollinearity. "
            "The updated workflow adds deeper regime analysis and a more explicit holdout evaluation story.",
            styles["BodyTextSmall"],
        )
    )

    story.append(Paragraph("2. Data Description", styles["SectionTitle"]))
    data_profile = pd.DataFrame(
        {
            "Attribute": ["Rows", "Columns", "Start Month", "End Month", "Missing Values", "Duplicate Rows"],
            "Value": [
                len(raw_df),
                raw_df.shape[1],
                f"{raw_df['Date'].min():%b %Y}",
                f"{raw_df['Date'].max():%b %Y}",
                int(raw_df.isna().sum().sum()),
                int(raw_df.duplicated().sum()),
            ],
        }
    )
    story.append(df_table(data_profile, col_widths=[2.1 * inch, 1.4 * inch]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Image(str(chart_paths["line"]), width=6.8 * inch, height=3.0 * inch))

    story.append(Paragraph("3. Exploratory Data Analysis", styles["SectionTitle"]))
    story.append(bullet("All four price variables are positively skewed, which is consistent with the stock’s long rise followed by a sharp reset.", styles))
    story.append(bullet("The strongest raw correlation with close comes from Low, followed by High and Open.", styles))
    story.append(bullet("The most volatile months cluster around the 2018 crisis and the 2020 COVID shock.", styles))
    story.append(bullet("The average monthly range is much larger after 2018, confirming a structural regime change.", styles))
    story.append(df_table(close_correlations.round(4), col_widths=[2.6 * inch, 1.5 * inch]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Image(str(chart_paths["heatmap"]), width=5.5 * inch, height=4.1 * inch))

    story.append(Paragraph("4. Feature Engineering and Preprocessing", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "Feature creation includes monthly range statistics, open-high-low gaps, lagged close values, rolling mean and rolling standard deviation features, cyclical month encoding, and regime flags for post-2018 stress and the COVID period. "
            "Instead of dropping core price variables to fight multicollinearity, the workflow keeps them and lets regularization absorb the instability.",
            styles["BodyTextSmall"],
        )
    )
    if not high_corr_df.empty:
        story.append(df_table(high_corr_df, col_widths=[2.0 * inch, 2.0 * inch, 1.0 * inch]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(df_table(target_conditioning_df.round(4), col_widths=[2.2 * inch, 1.0 * inch, 1.0 * inch, 0.8 * inch]))
    story.append(
        Paragraph(
            "Target conditioning checks show that the raw close performs better than log-transformed close on the chronological holdout set, so the final notebook keeps the target on its natural rupee scale.",
            styles["BodyTextSmall"],
        )
    )

    story.append(Paragraph("5. Model Implementation", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "The core notebook comparison uses a chronological 80:20 split and TimeSeriesSplit-based hyperparameter tuning. "
            "This keeps the evaluation closer to a realistic time-aware setup while preserving the same overall problem framing as the reference project.",
            styles["BodyTextSmall"],
        )
    )
    story.append(df_table(evaluation_summary[["Model", "MAE", "RMSE", "R2"]].round(4), col_widths=[2.2 * inch, 1.0 * inch, 1.0 * inch, 0.8 * inch]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Image(str(chart_paths["rmse"]), width=6.2 * inch, height=3.5 * inch))
    story.append(
        Paragraph(
            "Best tuned parameters for the regularized family: "
            + f"Lasso {best_params_store['Lasso Regression']}, "
            + f"Ridge {best_params_store['Ridge Regression']}, "
            + f"ElasticNet {best_params_store['ElasticNet Regression']}.",
            styles["BodyTextSmall"],
        )
    )

    story.append(Paragraph("6. Evaluation Summary", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "The main outcome stays aligned with the reference project in an important way: the linear and regularized linear family is the strongest part of the benchmark. "
            "In this standalone version, Lasso is selected for deployment because it keeps errors low and remains easy to explain in an interview setting.",
            styles["BodyTextSmall"],
        )
    )
    story.append(df_table(holdout_results.head(12).round(4), col_widths=[1.1 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.1 * inch]))

    story.append(Paragraph("7. Model Explainability", styles["SectionTitle"]))
    story.append(df_table(coefficient_df.head(12).round(4), col_widths=[2.4 * inch, 1.2 * inch, 1.4 * inch]))
    story.append(
        Paragraph(
            "The most influential features are still dominated by same-month price levels and short-memory close history. "
            "That is consistent with the central intuition of the dataset: the monthly close is tightly anchored to contemporaneous price structure.",
            styles["BodyTextSmall"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("8. Additional Findings", styles["SectionTitle"]))
    story.append(df_table(top_volatility, col_widths=[1.3 * inch, 1.2 * inch, 1.0 * inch, 1.4 * inch]))
    story.append(bullet("The strongest downside shocks line up with known stress windows, not random isolated months.", styles))
    story.append(bullet("The data is simple in shape but not in regime behavior; that is why validation strategy matters more than brute-force model complexity.", styles))
    story.append(bullet("The updated Streamlit dashboard adds a more presentation-ready layer than the reference repo by surfacing trend, volatility, correlation, and prediction quality in one place.", styles))

    story.append(Paragraph("9. Final Conclusion", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "Because the dataset is the same, the broad analytical outcome remains intentionally close to the reference project: the data is clean, price variables are highly correlated, regularized regression is a strong fit, and the 2018 crisis materially changes the stock behavior. "
            "The main difference in this standalone repository is that the analysis is more explicit about regime change, chronological validation, and model explainability. "
            "That makes the project stronger for interviews while still remaining faithful to the underlying dataset story.",
            styles["BodyTextSmall"],
        )
    )

    doc = SimpleDocTemplate(str(TECHNICAL_PDF), pagesize=A4, rightMargin=32, leftMargin=32, topMargin=32, bottomMargin=32)
    doc.build(story)


def main() -> None:
    payload = build_payload()
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        chart_paths = generate_charts(payload, tmp_dir)
        build_summary_pdf(payload, chart_paths)
        build_technical_pdf(payload, chart_paths)
    print(f"Generated {SUMMARY_PDF.name}")
    print(f"Generated {TECHNICAL_PDF.name}")


if __name__ == "__main__":
    main()
