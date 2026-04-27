"""Utilities for the Yes Bank stock closing price regression project."""

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
DATE_FORMAT = "%b-%y"

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

TARGET_COLUMN = "Close"


def resolve_default_data_path(project_root: Union[str, Path]) -> Path:
    """Resolve the preferred raw data path inside the standalone project."""
    root = Path(project_root)
    candidates = [
        root / "data_YesBank_StockPrices.csv",
        root / "data" / "data_YesBank_StockPrices.csv",
        root / "data" / "yes_bank_stock_prices.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the Yes Bank dataset. Expected one of: {0}".format(
            ", ".join(str(item) for item in candidates)
        )
    )


def load_yes_bank_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load and sort the monthly Yes Bank stock price dataset."""
    frame = pd.read_csv(csv_path)
    frame["Date"] = pd.to_datetime(frame["Date"], format=DATE_FORMAT)
    frame = frame.sort_values("Date").reset_index(drop=True)
    return frame


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create time-aware and market-behavior features used by the models."""
    df = frame.copy()
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["year"] = df["Date"].dt.year
    df["time_index"] = np.arange(len(df))

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    df["range_value"] = df["High"] - df["Low"]
    df["range_pct"] = np.where(df["Open"] != 0, df["range_value"] / df["Open"], 0.0)
    df["high_open_gap"] = df["High"] - df["Open"]
    df["open_low_gap"] = df["Open"] - df["Low"]

    df["post_2018_crisis"] = (df["Date"] >= pd.Timestamp("2018-01-01")).astype(int)
    df["covid_shock"] = (df["Date"] >= pd.Timestamp("2020-03-01")).astype(int)

    for lag in (1, 3, 6):
        df["close_lag_{0}".format(lag)] = df["Close"].shift(lag)

    for window in (3, 6):
        shifted_close = df["Close"].shift(1)
        df["close_ma_{0}".format(window)] = shifted_close.rolling(window=window).mean()
        df["close_std_{0}".format(window)] = shifted_close.rolling(window=window).std()

    return df


def build_modeling_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the final modeling frame with complete feature rows only."""
    model_df = engineer_features(frame)
    required_columns = ["Date", TARGET_COLUMN] + FEATURE_COLUMNS
    model_df = model_df[required_columns].dropna().reset_index(drop=True)
    return model_df


def build_preprocessor(feature_columns: Optional[Iterable[str]] = None) -> ColumnTransformer:
    """Build the preprocessing pipeline for numeric features."""
    columns = list(feature_columns or FEATURE_COLUMNS)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, columns)],
        remainder="drop",
    )


def make_pipeline(model, feature_columns: Optional[Iterable[str]] = None) -> Pipeline:
    """Create a full preprocessing + model pipeline."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(feature_columns)),
            ("model", model),
        ]
    )


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute a consistent set of regression metrics."""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_true, y_pred)),
    }


def time_aware_split(
    model_df: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Split the dataset without shuffling to preserve chronology."""
    columns = list(feature_columns or FEATURE_COLUMNS)
    split_index = int(len(model_df) * (1 - test_ratio))

    train_df = model_df.iloc[:split_index].copy()
    test_df = model_df.iloc[split_index:].copy()

    x_train = train_df[columns]
    x_test = test_df[columns]
    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    return x_train, x_test, y_train, y_test, train_df, test_df


def naive_baseline_metrics(model_df: pd.DataFrame, test_ratio: float = 0.2) -> Dict[str, float]:
    """Use previous month's close as a naive baseline."""
    _, _, _, y_test, _, test_df = time_aware_split(model_df, test_ratio=test_ratio)
    baseline_pred = test_df["close_lag_1"].to_numpy()
    metrics = compute_metrics(y_test, baseline_pred)
    metrics["Model"] = "Naive Previous Close"
    return metrics


def get_candidate_models() -> Dict[str, object]:
    """Return interview-friendly baseline model candidates."""
    return {
        "Linear Regression": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=3),
        "Lasso Regression": Lasso(alpha=0.2, max_iter=20000),
        "Ridge Regression": Ridge(alpha=1.0),
        "ElasticNet Regression": ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=20000),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            random_state=SEED,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            random_state=SEED,
        ),
    }


def evaluate_models(
    model_df: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Dict[str, object]]:
    """Train and evaluate the candidate models on a chronological holdout set."""
    columns = list(feature_columns or FEATURE_COLUMNS)
    x_train, x_test, y_train, y_test, train_df, test_df = time_aware_split(
        model_df, feature_columns=columns, test_ratio=test_ratio
    )

    metrics_rows = [naive_baseline_metrics(model_df, test_ratio=test_ratio)]
    fitted_models: Dict[str, Pipeline] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for model_name, estimator in get_candidate_models().items():
            pipeline = make_pipeline(estimator, columns)
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)
            row = compute_metrics(y_test, predictions)
            row["Model"] = model_name
            metrics_rows.append(row)
            fitted_models[model_name] = pipeline

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df[["Model", "MAE", "RMSE", "R2"]].sort_values(
        by=["RMSE", "MAE"], ascending=[True, True]
    )
    metrics_df = metrics_df.reset_index(drop=True)

    split_payload = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
    }
    return metrics_df, fitted_models, split_payload


def tune_regularized_models(
    model_df: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
    cv_splits: int = 4,
) -> Tuple[pd.DataFrame, Dict[str, GridSearchCV]]:
    """Tune the regularized linear models with time series cross-validation."""
    columns = list(feature_columns or FEATURE_COLUMNS)
    x_train, x_test, y_train, y_test, _, _ = time_aware_split(
        model_df, feature_columns=columns
    )
    cv = TimeSeriesSplit(n_splits=cv_splits)

    search_space = {
        "Lasso Regression": (
            Lasso(max_iter=20000),
            {"model__alpha": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]},
        ),
        "Ridge Regression": (
            Ridge(),
            {"model__alpha": [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]},
        ),
        "ElasticNet Regression": (
            ElasticNet(max_iter=20000),
            {
                "model__alpha": [0.01, 0.05, 0.1, 0.2, 0.5],
                "model__l1_ratio": [0.3, 0.5, 0.7, 0.9],
            },
        ),
    }

    tuning_rows: List[Dict[str, object]] = []
    tuned_searches: Dict[str, GridSearchCV] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for model_name, (estimator, params) in search_space.items():
            search = GridSearchCV(
                estimator=make_pipeline(estimator, columns),
                param_grid=params,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=1,
            )
            search.fit(x_train, y_train)
            predictions = search.best_estimator_.predict(x_test)
            metrics = compute_metrics(y_test, predictions)
            metrics["Model"] = model_name
            metrics["Best Params"] = str(search.best_params_)
            metrics["CV RMSE"] = float(-search.best_score_)
            tuning_rows.append(metrics)
            tuned_searches[model_name] = search

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df = tuning_df[
        ["Model", "Best Params", "CV RMSE", "MAE", "RMSE", "R2"]
    ].sort_values(by=["RMSE", "CV RMSE"])
    tuning_df = tuning_df.reset_index(drop=True)
    return tuning_df, tuned_searches


def evaluate_target_conditioning(
    model_df: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
    alpha: float = 0.2,
) -> pd.DataFrame:
    """Compare raw-target modeling versus log1p target conditioning."""
    columns = list(feature_columns or FEATURE_COLUMNS)
    x_train, x_test, y_train, y_test, _, _ = time_aware_split(
        model_df, feature_columns=columns
    )
    baseline_model = make_pipeline(Lasso(alpha=alpha, max_iter=20000), columns)

    rows: List[Dict[str, float]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        baseline_model.fit(x_train, y_train)
        raw_predictions = baseline_model.predict(x_test)
        raw_metrics = compute_metrics(y_test, raw_predictions)
        raw_metrics["Target Strategy"] = "Raw Close"
        rows.append(raw_metrics)

        baseline_model.fit(x_train, np.log1p(y_train))
        log_predictions = np.expm1(baseline_model.predict(x_test))
        log_metrics = compute_metrics(y_test, log_predictions)
        log_metrics["Target Strategy"] = "log1p(Close)"
        rows.append(log_metrics)

    result_df = pd.DataFrame(rows)
    result_df = result_df[["Target Strategy", "MAE", "RMSE", "R2"]].sort_values(
        by=["RMSE", "MAE"]
    )
    return result_df.reset_index(drop=True)


def fit_final_model(
    model_df: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
    preferred_model: str = "Lasso Regression",
) -> Tuple[Pipeline, Dict[str, object], Dict[str, object]]:
    """Tune the preferred model, then fit it on the full modeling frame."""
    columns = list(feature_columns or FEATURE_COLUMNS)
    tuning_df, tuned_searches = tune_regularized_models(model_df, columns)
    best_search = tuned_searches.get(preferred_model)

    if best_search is None:
        raise ValueError("Preferred model '{0}' was not tuned.".format(preferred_model))

    x_train, x_test, y_train, y_test, train_df, test_df = time_aware_split(
        model_df, feature_columns=columns
    )
    holdout_predictions = best_search.best_estimator_.predict(x_test)
    holdout_metrics = compute_metrics(y_test, holdout_predictions)

    final_pipeline = best_search.best_estimator_
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        final_pipeline.fit(model_df[columns], model_df[TARGET_COLUMN])

    payload = {
        "tuning_table": tuning_df,
        "best_params": best_search.best_params_,
        "holdout_metrics": holdout_metrics,
        "train_df": train_df,
        "test_df": test_df,
        "x_test": x_test,
        "y_test": y_test,
        "holdout_predictions": holdout_predictions,
    }
    return final_pipeline, payload, tuned_searches


def model_coefficients(
    fitted_pipeline: Pipeline,
    feature_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Extract absolute coefficient magnitude for linear models."""
    columns = list(feature_columns or FEATURE_COLUMNS)
    estimator = fitted_pipeline.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        raise ValueError("The fitted estimator does not expose coefficients.")

    coef_df = pd.DataFrame(
        {
            "feature": columns,
            "coefficient": estimator.coef_,
        }
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)
    return coef_df


def permutation_feature_importance(
    fitted_pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: Optional[Iterable[str]] = None,
    n_repeats: int = 20,
) -> pd.DataFrame:
    """Compute model-agnostic feature importance on the holdout set."""
    columns = list(feature_columns or FEATURE_COLUMNS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = permutation_importance(
            fitted_pipeline,
            x_test,
            y_test,
            n_repeats=n_repeats,
            random_state=SEED,
            scoring="neg_root_mean_squared_error",
        )

    importance_df = pd.DataFrame(
        {
            "feature": columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    importance_df = importance_df.sort_values(
        by="importance_mean", ascending=False
    ).reset_index(drop=True)
    return importance_df


def high_correlation_pairs(
    model_df: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """List strongly correlated feature pairs for multicollinearity inspection."""
    columns = list(feature_columns or FEATURE_COLUMNS)
    corr_matrix = model_df[columns].corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    rows: List[Dict[str, object]] = []
    for column in upper_triangle.columns:
        correlations = upper_triangle[column].dropna()
        for index_name, value in correlations.items():
            if value >= threshold:
                rows.append(
                    {
                        "feature_1": index_name,
                        "feature_2": column,
                        "correlation": float(value),
                    }
                )

    return pd.DataFrame(rows).sort_values(by="correlation", ascending=False).reset_index(
        drop=True
    )


def next_forecast_date(frame: pd.DataFrame) -> pd.Timestamp:
    """Return the next monthly period after the latest available record."""
    last_date = pd.to_datetime(frame["Date"]).max()
    return pd.Timestamp(last_date) + pd.offsets.MonthBegin(1)


def build_prediction_frame(
    history_df: pd.DataFrame,
    forecast_date: Union[str, pd.Timestamp],
    open_price: float,
    high_price: float,
    low_price: float,
    feature_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Create a one-row feature frame for interactive scenario prediction."""
    columns = list(feature_columns or FEATURE_COLUMNS)
    if high_price < open_price or low_price > open_price or low_price > high_price:
        raise ValueError(
            "Expected Low <= Open <= High and Low <= High for a valid monthly scenario."
        )

    forecast_timestamp = pd.Timestamp(forecast_date).replace(day=1)
    close_history = history_df["Close"].to_numpy()

    if len(close_history) < 6:
        raise ValueError("At least 6 historical close values are required for prediction.")

    month = forecast_timestamp.month
    feature_values = {
        "Open": float(open_price),
        "High": float(high_price),
        "Low": float(low_price),
        "range_value": float(high_price - low_price),
        "range_pct": float((high_price - low_price) / open_price if open_price else 0.0),
        "high_open_gap": float(high_price - open_price),
        "open_low_gap": float(open_price - low_price),
        "close_lag_1": float(close_history[-1]),
        "close_lag_3": float(close_history[-3]),
        "close_lag_6": float(close_history[-6]),
        "close_ma_3": float(close_history[-3:].mean()),
        "close_ma_6": float(close_history[-6:].mean()),
        "close_std_3": float(pd.Series(close_history[-3:]).std()),
        "close_std_6": float(pd.Series(close_history[-6:]).std()),
        "month_sin": float(np.sin(2 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * month / 12.0)),
        "year": int(forecast_timestamp.year),
        "time_index": int(len(history_df)),
        "post_2018_crisis": int(forecast_timestamp >= pd.Timestamp("2018-01-01")),
        "covid_shock": int(forecast_timestamp >= pd.Timestamp("2020-03-01")),
    }

    return pd.DataFrame([feature_values], columns=columns)


def predict_close_price(
    fitted_pipeline: Pipeline,
    history_df: pd.DataFrame,
    forecast_date: Union[str, pd.Timestamp],
    open_price: float,
    high_price: float,
    low_price: float,
    feature_columns: Optional[Iterable[str]] = None,
) -> float:
    """Predict the closing price for a user-supplied monthly scenario."""
    prediction_frame = build_prediction_frame(
        history_df=history_df,
        forecast_date=forecast_date,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        feature_columns=feature_columns,
    )
    prediction = fitted_pipeline.predict(prediction_frame)[0]
    return float(prediction)
