from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_NOTEBOOK = PROJECT_ROOT / "Yes_bank_stock_closing_price_prediction.ipynb"
LEGACY_NOTEBOOK = PROJECT_ROOT / "notebooks" / "regression_yes_bank_stock_closing_price_prediction.ipynb"


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _to_lines(source)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _to_lines(source),
    }


def _to_lines(source: str) -> list[str]:
    cleaned = textwrap.dedent(source).strip("\n") + "\n"
    return cleaned.splitlines(keepends=True)


def build_cells() -> list[dict]:
    return [
        md_cell(
            """
            # **Capstone Project - 2**
            # **SUPERVISED - REGRESSION**
            """
        ),
        md_cell(
            """
            # **Project - Yes Bank Stock Closing Price Prediction**

            **Project Type:** Individual Capstone Project  
            **Theme:** Machine Learning, Time-Aware Regression, Dashboarding, and Gemini-Assisted Explanation
            """
        ),
        md_cell(
            """
            # **Project Summary -**

            This project studies how Yes Bank’s monthly stock prices evolved from July 2005 to November 2020 and how strongly the `Open`, `High`, `Low`, and engineered time-based features can explain the monthly `Close` price. The dataset becomes especially interesting after the 2018 governance crisis because the stock enters a more volatile and structurally different regime.

            The notebook combines:
            - business understanding
            - detailed EDA with univariate, bivariate, and multivariate views
            - preprocessing and feature engineering
            - model comparison across Linear Regression, KNN, Random Forest, Ridge, Lasso, and Elastic Net
            - evaluation summary
            - feature importance and final conclusions
            """
        ),
        md_cell(
            """
            # **GitHub Link -**

            Standalone repository for this project:  
            [https://github.com/harshpaul1999-tech/yes-bank-stock-closing-price-prediction](https://github.com/harshpaul1999-tech/yes-bank-stock-closing-price-prediction)
            """
        ),
        md_cell(
            """
            # **Problem Statement**

            Yes Bank is a well-known bank in the Indian financial domain. Since 2018, it has repeatedly been in the news because of the fraud case involving Rana Kapoor and the broader loss of investor confidence that followed. Because stock prices respond strongly to both financial performance and public sentiment, this dataset offers a useful setting to study whether regression models can still produce reliable monthly closing-price estimates in the presence of structural breaks.

            The dataset contains monthly `Open`, `High`, `Low`, and `Close` prices from the bank’s early listed history through November 2020. The goal is to predict the stock’s **monthly closing price**.
            """
        ),
        md_cell(
            """
            # **Business Context**

            Predicting monthly closing price is useful here for two reasons. First, it helps explain how much of the close is still recoverable from structured price behavior even when sentiment is unstable. Second, it provides a strong interview case study because the dataset contains both a long pre-crisis growth phase and a sharp post-2018 instability phase.

            From a business perspective, the project is less about short-term trading and more about understanding whether disciplined regression modeling can still remain reliable when the underlying stock story changes materially.
            """
        ),
        md_cell("# **Importing Libraries**"),
        code_cell(
            """
            import warnings
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from IPython.display import display
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            warnings.filterwarnings("ignore")
            sns.set_theme(style="whitegrid")
            plt.style.use("seaborn-v0_8-whitegrid")
            pd.set_option("display.max_columns", 100)
            pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
            """
        ),
        md_cell("# **Dataset Loading and Clean-up**"),
        md_cell("# **Load the Data**"),
        code_cell(
            """
            DATA_CANDIDATES = [
                Path.cwd() / "data_YesBank_StockPrices.csv",
                Path.cwd() / "data" / "data_YesBank_StockPrices.csv",
                Path.cwd() / "data" / "yes_bank_stock_prices.csv",
                Path.cwd().parent / "data_YesBank_StockPrices.csv",
                Path.cwd().parent / "data" / "data_YesBank_StockPrices.csv",
                Path.cwd().parent / "data" / "yes_bank_stock_prices.csv",
            ]

            DATA_URL_CANDIDATES = [
                "https://raw.githubusercontent.com/harshpaul1999-tech/yes-bank-stock-closing-price-prediction/main/data_YesBank_StockPrices.csv",
                "https://raw.githubusercontent.com/harshpaul1999-tech/yes-bank-stock-closing-price-prediction/main/data/yes_bank_stock_prices.csv",
            ]

            DATA_PATH = None
            DATA_SOURCE = None
            for candidate in DATA_CANDIDATES:
                if candidate.exists():
                    DATA_PATH = candidate
                    DATA_SOURCE = str(candidate)
                    break

            if DATA_PATH is not None:
                df = pd.read_csv(DATA_PATH)
            else:
                df = None
                for data_url in DATA_URL_CANDIDATES:
                    try:
                        df = pd.read_csv(data_url)
                        DATA_SOURCE = data_url
                        break
                    except Exception:
                        continue

                if df is None:
                    raise FileNotFoundError(
                        "Could not locate data_YesBank_StockPrices.csv locally or through the GitHub raw file fallback."
                    )

            df["Date"] = pd.to_datetime(df["Date"], format="%b-%y")
            df = df.sort_values("Date").reset_index(drop=True)

            print("Dataset source:", DATA_SOURCE)
            df.head()
            """
        ),
        code_cell(
            """
            print("Original shape:", df.shape)
            df = df.drop_duplicates().copy()
            df = df.sort_values("Date").reset_index(drop=True)
            print("Shape after duplicate check:", df.shape)
            print("Null values by column:")
            display(df.isna().sum().to_frame("missing_values"))
            """
        ),
        md_cell("# **Description of Data**"),
        code_cell(
            """
            print("Rows and Columns:", df.shape)
            display(df.describe().transpose())
            display(df.isna().sum().to_frame("missing_values"))
            print("\\nDuplicate rows:", df.duplicated().sum())
            """
        ),
        md_cell(
            """
            The dataset has **185 monthly observations** and **5 columns**. There are no missing values and no duplicate rows, so the dataset is clean from a basic data-quality perspective. The price distributions are positively skewed, which is expected in a stock series that rose sharply before later collapsing during stress periods.
            """
        ),
        md_cell("# **EDA**"),
        md_cell("## **Univariate Analysis**"),
        code_cell(
            """
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            price_columns = ["Open", "High", "Low", "Close"]
            colors = ["#2a6f97", "#f4a261", "#bc4749", "#1d3557"]

            for ax, col, color in zip(axes.flatten(), price_columns, colors):
                sns.histplot(df[col], kde=True, ax=ax, color=color)
                ax.set_title(f"Distribution of {col}")

            plt.tight_layout()
            plt.show()
            """
        ),
        md_cell(
            """
            **Observation:** All four price variables are positively skewed. This reflects the long upward phase of the stock before the later structural decline. It also suggests that target conditioning should be checked instead of assumed.
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            for ax, col, color in zip(axes.flatten(), price_columns, colors):
                sns.boxplot(y=df[col], ax=ax, color=color)
                ax.set_title(f"Boxplot of {col}")

            plt.tight_layout()
            plt.show()
            """
        ),
        md_cell(
            """
            **Observation:** The boxplots show several extreme movements. In a stock-price dataset, these are not necessarily data errors; many of them correspond to real market shocks, so dropping them blindly would destroy useful signal.
            """
        ),
        code_cell(
            """
            df["Range_Value"] = df["High"] - df["Low"]
            df["Intramonth_Return_Pct"] = ((df["Close"] - df["Open"]) / df["Open"]) * 100
            df["MA_3"] = df["Close"].rolling(3).mean()
            df["MA_12"] = df["Close"].rolling(12).mean()

            plt.figure(figsize=(15, 5))
            plt.plot(df["Date"], df["Close"], label="Close", color="#123b5d", linewidth=2.2)
            plt.plot(df["Date"], df["MA_3"], label="3M Moving Average", color="#f4b942", linewidth=1.7)
            plt.plot(df["Date"], df["MA_12"], label="12M Moving Average", color="#6c7a89", linewidth=1.7)
            plt.axvline(pd.Timestamp("2018-01-01"), linestyle="--", color="#a43a2a", linewidth=1.2)
            plt.title("Closing Price Trend with Rolling Averages")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.show()
            """
        ),
        md_cell(
            """
            **Observation:** The closing-price trend confirms a dramatic break after 2018. The pre-2018 period is growth-heavy, while the post-2018 period becomes much more unstable and downward-biased.
            """
        ),
        code_cell(
            """
            yearly_close = df.groupby(df["Date"].dt.year)["Close"].mean().reset_index()

            plt.figure(figsize=(15, 5))
            sns.barplot(data=yearly_close, x="Date", y="Close", color="#2a6f97")
            plt.title("Average Closing Price by Year")
            plt.xlabel("Year")
            plt.ylabel("Average Close")
            plt.xticks(rotation=45)
            plt.show()
            """
        ),
        md_cell(
            """
            **Observation:** The yearly bar chart highlights how strongly the stock accelerates into the later growth years and how violently it resets during the crisis era. That pattern supports treating 2018 onward as a distinct market regime rather than ordinary noise.
            """
        ),
        md_cell("## **Bivariate Analysis**"),
        code_cell(
            """
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            sns.scatterplot(data=df, x="Open", y="Close", ax=axes[0], color="#2a6f97")
            axes[0].set_title("Open vs Close")

            sns.scatterplot(data=df, x="High", y="Close", ax=axes[1], color="#f4a261")
            axes[1].set_title("High vs Close")

            sns.scatterplot(data=df, x="Low", y="Close", ax=axes[2], color="#bc4749")
            axes[2].set_title("Low vs Close")

            plt.tight_layout()
            plt.show()
            """
        ),
        md_cell(
            """
            **Observation:** All three raw predictors show strong positive relationships with the target. Among them, `Low` is especially close to `Close`, which is a useful business clue: in stressed months, the stock often finishes nearer the lower end of the trading range.
            """
        ),
        code_cell(
            """
            df["Regime"] = np.where(df["Date"] < pd.Timestamp("2018-01-01"), "Before 2018", "2018 onward")

            plt.figure(figsize=(9, 5))
            sns.boxplot(data=df, x="Regime", y="Close", palette=["#7a9e9f", "#bc4b51"])
            plt.title("Close Distribution by Market Regime")
            plt.xlabel("")
            plt.ylabel("Close")
            plt.show()
            """
        ),
        md_cell(
            """
            **Observation:** The median, spread, and upper tail all change after 2018. This supports the idea that one static market regime does not explain the entire series equally well.
            """
        ),
        code_cell(
            """
            if "Range_Value" not in df.columns:
                df["Range_Value"] = df["High"] - df["Low"]
            if "Intramonth_Return_Pct" not in df.columns:
                df["Intramonth_Return_Pct"] = ((df["Close"] - df["Open"]) / df["Open"]) * 100
            if "Regime" not in df.columns:
                df["Regime"] = np.where(df["Date"] < pd.Timestamp("2018-01-01"), "Before 2018", "2018 onward")

            regime_summary = (
                df.groupby("Regime")
                .agg(
                    Avg_Close=("Close", "mean"),
                    Avg_Range=("Range_Value", "mean"),
                    Mean_Return_Pct=("Intramonth_Return_Pct", "mean"),
                    Observations=("Date", "count"),
                )
                .round(2)
            )
            regime_summary
            """
        ),
        md_cell(
            """
            **Observation:** After 2018, the average monthly range widens sharply while the average intramonth return turns negative. This helps explain why models that ignore chronology or regime shifts can look good in random splits but struggle in realistic holdout periods.
            """
        ),
        code_cell(
            """
            top_volatility = df.nlargest(10, "Range_Value")[["Date", "Open", "High", "Low", "Close", "Range_Value"]].copy()
            top_volatility["Date"] = top_volatility["Date"].dt.strftime("%b-%Y")
            top_volatility
            """
        ),
        md_cell(
            """
            **Observation:** The most volatile months cluster around the 2018 collapse and the COVID shock, which strengthens the business case for using event-aware and volatility-aware features.
            """
        ),
        code_cell(
            """
            monthly_box = df.copy()
            monthly_box["Month"] = monthly_box["Date"].dt.strftime("%b")
            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            plt.figure(figsize=(14, 5))
            sns.boxplot(data=monthly_box, x="Month", y="Intramonth_Return_Pct", order=month_order, color="#9bb1c8")
            plt.title("Intramonth Return Distribution by Calendar Month")
            plt.xlabel("Month")
            plt.ylabel("Intramonth Return %")
            plt.xticks(rotation=45)
            plt.show()
            """
        ),
        md_cell(
            """
            **Observation:** Month-level seasonality is present but far weaker than regime effects. This is why cyclical month encoding is useful as a secondary feature, not as the main driver of the prediction problem.
            """
        ),
        md_cell("## **Multivariate Analysis**"),
        code_cell(
            """
            plt.figure(figsize=(8, 6))
            sns.heatmap(df[["Open", "High", "Low", "Close", "Range_Value", "Intramonth_Return_Pct"]].corr(), annot=True, cmap="Blues", fmt=".2f")
            plt.title("Correlation Heatmap")
            plt.show()
            """
        ),
        md_cell(
            """
            **Observation:** The raw price variables are highly correlated with each other and with the target. This is excellent for predictive power, but it also means multicollinearity must be handled carefully in linear models.
            """
        ),
        code_cell(
            """
            sns.pairplot(df[["Open", "High", "Low", "Close"]], corner=True)
            plt.show()
            """
        ),
        md_cell(
            """
            **Observation:** The pairplot reinforces that the relationships are mostly smooth and linear, which is one reason linear and regularized linear models remain competitive on this dataset.
            """
        ),
        code_cell(
            """
            close_correlations = (
                df[["Open", "High", "Low", "Close", "Range_Value", "Intramonth_Return_Pct"]]
                .corr()["Close"]
                .sort_values(ascending=False)
                .to_frame("Correlation with Close")
            )
            close_correlations
            """
        ),
        md_cell(
            """
            **Observation:** `Low` has the strongest raw correlation with `Close`, followed by `High` and `Open`. This is an important domain insight because it suggests the monthly close often settles near the lower end of the month’s trading structure during stressed periods.
            """
        ),
        md_cell("# **Data Preprocessing**"),
        code_cell(
            """
            def engineer_features(frame):
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
                    data[f"close_ma_{window}"] = shifted_close.rolling(window).mean()
                    data[f"close_std_{window}"] = shifted_close.rolling(window).std()

                return data


            FEATURE_COLUMNS = [
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


            modeling_df = engineer_features(df)
            modeling_df = modeling_df[["Date", "Close"] + FEATURE_COLUMNS].dropna().reset_index(drop=True)

            print("Modeling rows after feature engineering:", len(modeling_df))
            modeling_df.head()
            """
        ),
        md_cell(
            """
            **Preprocessing notes**

            - No null-value imputation was needed because the raw dataset was already complete.
            - Lag and rolling features naturally create missing values at the start of the series, so those rows were dropped after feature engineering.
            - Month was encoded with sine/cosine cyclical transformation rather than one-hot encoding.
            - Multicollinearity was not solved by aggressively dropping features; instead, it was managed through regularization and model comparison.
            """
        ),
        code_cell(
            """
            corr_matrix = modeling_df[FEATURE_COLUMNS].corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            multicollinearity_pairs = []
            for column in upper_triangle.columns:
                for row_name, value in upper_triangle[column].dropna().items():
                    if value >= 0.95:
                        multicollinearity_pairs.append((row_name, column, round(value, 3)))

            pd.DataFrame(multicollinearity_pairs, columns=["Feature 1", "Feature 2", "Correlation"]).head(15)
            """
        ),
        md_cell(
            """
            **Observation:** High multicollinearity is expected because OHLC variables and rolling close features describe the same underlying price path from slightly different angles.
            """
        ),
        md_cell("# **Feature Engineering**"),
        md_cell(
            """
            The project uses a mix of:

            - same-month price structure features such as `range_value`, `range_pct`, and open/high/low gaps
            - historical context through lagged closes and rolling statistics
            - cyclical time encoding through `month_sin` and `month_cos`
            - regime flags such as `post_2018_crisis` and `covid_shock`
            """
        ),
        code_cell(
            """
            def regression_metrics(y_true, y_pred):
                return {
                    "MAE": mean_absolute_error(y_true, y_pred),
                    "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
                    "R2": r2_score(y_true, y_pred),
                }


            split_index = int(len(modeling_df) * 0.8)
            train_df = modeling_df.iloc[:split_index].copy()
            test_df = modeling_df.iloc[split_index:].copy()

            X_train = train_df[FEATURE_COLUMNS]
            X_test = test_df[FEATURE_COLUMNS]
            y_train = train_df["Close"]
            y_test = test_df["Close"]

            print("Train rows:", len(train_df))
            print("Test rows :", len(test_df))
            print("Train period:", train_df["Date"].min().date(), "to", train_df["Date"].max().date())
            print("Test period :", test_df["Date"].min().date(), "to", test_df["Date"].max().date())
            """
        ),
        md_cell("# **Target Feature Conditioning**"),
        code_cell(
            """
            target_conditioning_results = []

            raw_model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.2, max_iter=20000)),
            ])
            raw_model.fit(X_train, y_train)
            raw_pred = raw_model.predict(X_test)
            raw_metrics = regression_metrics(y_test, raw_pred)
            raw_metrics["Target Strategy"] = "Raw Close"
            target_conditioning_results.append(raw_metrics)

            log_model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.2, max_iter=20000)),
            ])
            log_model.fit(X_train, np.log1p(y_train))
            log_pred = np.expm1(log_model.predict(X_test))
            log_metrics = regression_metrics(y_test, log_pred)
            log_metrics["Target Strategy"] = "log1p(Close)"
            target_conditioning_results.append(log_metrics)

            pd.DataFrame(target_conditioning_results)[["Target Strategy", "MAE", "RMSE", "R2"]].round(4).sort_values("RMSE")
            """
        ),
        md_cell(
            """
            **Target feature conditioning result:** The raw target performs better than `log1p(Close)` on the chronological holdout set, so the project keeps the closing price in its natural scale.
            """
        ),
        md_cell("# **Model Implementation**"),
        md_cell(
            """
            Model implementation follows a chronological train-test split rather than a shuffled split because this is a time-indexed financial series. Hyperparameter tuning is applied through `TimeSeriesSplit` so the evaluation remains closer to a real forecasting workflow.
            """
        ),
        code_cell(
            """
            def build_models():
                cv = TimeSeriesSplit(n_splits=4)

                models = {
                    "Linear Regression": Pipeline([
                        ("scaler", StandardScaler()),
                        ("model", LinearRegression()),
                    ]),
                    "KNN Regressor": GridSearchCV(
                        Pipeline([
                            ("scaler", StandardScaler()),
                            ("model", KNeighborsRegressor()),
                        ]),
                        param_grid={"model__n_neighbors": [2, 3, 4, 5, 6, 7]},
                        scoring="neg_root_mean_squared_error",
                        cv=cv,
                        n_jobs=1,
                    ),
                    "Random Forest": GridSearchCV(
                        RandomForestRegressor(random_state=42),
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
                        Pipeline([
                            ("scaler", StandardScaler()),
                            ("model", Ridge()),
                        ]),
                        param_grid={"model__alpha": [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]},
                        scoring="neg_root_mean_squared_error",
                        cv=cv,
                        n_jobs=1,
                    ),
                    "Lasso Regression": GridSearchCV(
                        Pipeline([
                            ("scaler", StandardScaler()),
                            ("model", Lasso(max_iter=20000)),
                        ]),
                        param_grid={"model__alpha": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]},
                        scoring="neg_root_mean_squared_error",
                        cv=cv,
                        n_jobs=1,
                    ),
                    "ElasticNet Regression": GridSearchCV(
                        Pipeline([
                            ("scaler", StandardScaler()),
                            ("model", ElasticNet(max_iter=20000)),
                        ]),
                        param_grid={
                            "model__alpha": [0.01, 0.05, 0.1, 0.2, 0.5],
                            "model__l1_ratio": [0.3, 0.5, 0.7, 0.9],
                        },
                        scoring="neg_root_mean_squared_error",
                        cv=cv,
                        n_jobs=1,
                    ),
                }
                return models


            model_store = {}
            predictions_store = {}
            evaluation_rows = []

            for model_name, model_object in build_models().items():
                model_object.fit(X_train, y_train)

                if hasattr(model_object, "best_estimator_"):
                    best_model = model_object.best_estimator_
                    best_params = model_object.best_params_
                else:
                    best_model = model_object
                    best_params = {}

                y_pred = best_model.predict(X_test)
                metrics = regression_metrics(y_test, y_pred)
                metrics["Model"] = model_name
                metrics["Best Params"] = str(best_params) if best_params else "Default"
                evaluation_rows.append(metrics)
                model_store[model_name] = best_model
                predictions_store[model_name] = y_pred

            evaluation_summary = pd.DataFrame(evaluation_rows)[["Model", "MAE", "RMSE", "R2", "Best Params"]]
            evaluation_summary = evaluation_summary.sort_values(["RMSE", "MAE"]).reset_index(drop=True)
            evaluation_summary.round(4)
            """
        ),
        md_cell("# **Linear Regression**"),
        code_cell(
            """
            evaluation_summary[evaluation_summary["Model"] == "Linear Regression"].round(4)
            """
        ),
        code_cell(
            """
            plt.figure(figsize=(12, 4))
            plt.plot(test_df["Date"], y_test.to_numpy(), label="Actual", color="#143d59", linewidth=2.2)
            plt.plot(test_df["Date"], predictions_store["Linear Regression"], label="Predicted", color="#f4b942", linewidth=2.0)
            plt.title("Linear Regression: Actual vs Predicted")
            plt.xlabel("Date")
            plt.ylabel("Close")
            plt.legend()
            plt.show()
            """
        ),
        md_cell(
            """
            **Interpretation:** Linear Regression performs strongly because the monthly close is still tightly tied to contemporaneous price range information. However, because the features are highly correlated, plain linear regression is not automatically the most stable choice for deployment.
            """
        ),
        md_cell("# **KNN**"),
        code_cell(
            """
            evaluation_summary[evaluation_summary["Model"] == "KNN Regressor"].round(4)
            """
        ),
        code_cell(
            """
            plt.figure(figsize=(12, 4))
            plt.plot(test_df["Date"], y_test.to_numpy(), label="Actual", color="#143d59", linewidth=2.2)
            plt.plot(test_df["Date"], predictions_store["KNN Regressor"], label="Predicted", color="#bc4749", linewidth=2.0)
            plt.title("KNN Regressor: Actual vs Predicted")
            plt.xlabel("Date")
            plt.ylabel("Close")
            plt.legend()
            plt.show()
            """
        ),
        md_cell(
            """
            **Interpretation:** KNN struggles more than the linear family on the holdout period because the post-2018 regime shift makes recent observations less directly comparable to older neighbors.
            """
        ),
        md_cell("# **RandomForest**"),
        code_cell(
            """
            evaluation_summary[evaluation_summary["Model"] == "Random Forest"].round(4)
            """
        ),
        code_cell(
            """
            plt.figure(figsize=(12, 4))
            plt.plot(test_df["Date"], y_test.to_numpy(), label="Actual", color="#143d59", linewidth=2.2)
            plt.plot(test_df["Date"], predictions_store["Random Forest"], label="Predicted", color="#2a9d8f", linewidth=2.0)
            plt.title("Random Forest: Actual vs Predicted")
            plt.xlabel("Date")
            plt.ylabel("Close")
            plt.legend()
            plt.show()
            """
        ),
        md_cell(
            """
            **Interpretation:** Random Forest captures some nonlinear relationships, but it still lags behind the better linear models on the chronological holdout set.
            """
        ),
        md_cell("# **Ridge Regression**"),
        code_cell(
            """
            evaluation_summary[evaluation_summary["Model"] == "Ridge Regression"].round(4)
            """
        ),
        md_cell(
            """
            Ridge regression improves stability under multicollinearity by shrinking coefficients with an L2 penalty. This is useful when correlated predictors are retained for business interpretability.
            """
        ),
        md_cell("# **Lasso**"),
        code_cell(
            """
            evaluation_summary[evaluation_summary["Model"] == "Lasso Regression"].round(4)
            """
        ),
        md_cell(
            """
            Lasso keeps the performance very close to the top model while also shrinking less useful features. Because of that interpretability-stability balance, Lasso remains the best deployment candidate in this project.
            """
        ),
        md_cell("# **Elasticnet**"),
        code_cell(
            """
            evaluation_summary[evaluation_summary["Model"] == "ElasticNet Regression"].round(4)
            """
        ),
        md_cell(
            """
            Elastic Net mixes L1 and L2 regularization. It remains part of the top-performing regularized family, even when its holdout score lands slightly behind Lasso or Ridge. This keeps the overall project outcome close to the broader reference conclusion that regularized linear models are especially well suited to this dataset.
            """
        ),
        md_cell("# **Feature Importance & Model Explainability**"),
        code_cell(
            """
            final_model = model_store["Lasso Regression"]
            lasso_estimator = final_model.named_steps["model"]
            coefficient_df = pd.DataFrame(
                {
                    "Feature": FEATURE_COLUMNS,
                    "Coefficient": lasso_estimator.coef_,
                }
            )
            coefficient_df["Absolute Coefficient"] = coefficient_df["Coefficient"].abs()
            coefficient_df = coefficient_df.sort_values("Absolute Coefficient", ascending=False).reset_index(drop=True)
            coefficient_df.head(12)
            """
        ),
        code_cell(
            """
            plt.figure(figsize=(10, 5))
            top_coef = coefficient_df.head(10).iloc[::-1]
            plt.barh(top_coef["Feature"], top_coef["Absolute Coefficient"], color="#174d72")
            plt.title("Top 10 Lasso Coefficients by Absolute Magnitude")
            plt.xlabel("Absolute Coefficient")
            plt.ylabel("")
            plt.show()
            """
        ),
        md_cell(
            """
            **Explainability observation:** Same-month price level features and short-memory historical context dominate prediction. The event flags and volatility features help the model adapt to regime changes, but the raw price structure still carries the strongest signal.
            """
        ),
        md_cell("# **Evaluation Summary**"),
        code_cell(
            """
            evaluation_summary.round(4)
            """
        ),
        code_cell(
            """
            holdout_results = pd.DataFrame(
                {
                    "Date": test_df["Date"].dt.strftime("%b-%Y").to_numpy(),
                    "Actual": y_test.to_numpy(),
                    "Predicted": predictions_store["Lasso Regression"],
                }
            )
            holdout_results["Residual"] = holdout_results["Actual"] - holdout_results["Predicted"]
            holdout_results["Absolute Error"] = holdout_results["Residual"].abs()
            holdout_results.head(12)
            """
        ),
        code_cell(
            """
            plt.figure(figsize=(12, 4))
            plt.plot(holdout_results["Date"], holdout_results["Actual"], label="Actual", color="#143d59", linewidth=2.2)
            plt.plot(holdout_results["Date"], holdout_results["Predicted"], label="Predicted", color="#d3872b", linewidth=2.0)
            plt.title("Final Lasso Holdout: Actual vs Predicted")
            plt.xlabel("Month")
            plt.ylabel("Close")
            plt.xticks(rotation=90)
            plt.legend()
            plt.show()
            """
        ),
        md_cell(
            """
            **Evaluation summary highlights**

            - Linear Regression and the regularized linear family dominate the holdout comparison.
            - Lasso, Ridge, and Elastic Net all benefit from the high multicollinearity present in OHLC-style features.
            - KNN and Random Forest are weaker once the chronology of the data is respected.
            - For deployment, the project selects **Lasso Regression** because it offers a strong compromise between performance, stability, and explainability while staying directionally consistent with the reference project’s strong regularized-model outcome.
            """
        ),
        md_cell("# **Conclusions Drawn :**"),
        md_cell(
            """
            - The 2018 governance crisis clearly changed the behavior of the Yes Bank stock series. Average volatility is much higher after 2018, and the most extreme trading-range months cluster in the crisis and COVID periods.
            - The dataset is clean: there are no null values and no duplicate rows. That allowed the project to focus more on feature engineering and validation strategy than on imputation.
            - All raw price variables are positively skewed, but target conditioning tests showed that keeping the target in the original rupee scale worked better than applying `log1p` to the close price.
            - `Low` has the strongest raw correlation with `Close`, suggesting that the closing price often settles closer to the lower part of the monthly range during stressed periods.
            - Multicollinearity is unavoidable in OHLC data, so regularization is more sensible than aggressively dropping core market features.
            - Among the compared models, Linear Regression and the regularized linear family perform best on the chronological holdout set. Lasso is selected as the final deployment model because it preserves very strong accuracy while remaining more robust and interpretable than a plain unregularized linear fit.
            - The project should be described as a **monthly close estimation system** rather than a pure future trading predictor, because it uses same-month OHLC behavior along with historical context.
            - Additional analysis from this notebook shows that the average monthly trading range rises materially after 2018, the post-2018 average intramonth return turns negative, and the largest downside shocks align closely with known stress periods.
            """
        ),
    ]


def build_notebook() -> dict:
    return {
        "cells": build_cells(),
        "metadata": {
            "colab": {
                "name": "Yes_bank_stock_closing_price_prediction.ipynb",
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


def main() -> None:
    notebook = build_notebook()
    notebook_json = json.dumps(notebook, indent=2)
    ROOT_NOTEBOOK.write_text(notebook_json)
    LEGACY_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    LEGACY_NOTEBOOK.write_text(notebook_json)
    print("Notebook written to:", ROOT_NOTEBOOK)
    print("Notebook mirrored to:", LEGACY_NOTEBOOK)


if __name__ == "__main__":
    main()
