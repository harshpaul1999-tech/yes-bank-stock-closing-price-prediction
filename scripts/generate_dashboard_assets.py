#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from yes_bank_ml import (  # noqa: E402
    FEATURE_COLUMNS,
    build_modeling_frame,
    evaluate_models,
    fit_final_model,
    load_yes_bank_data,
    resolve_default_data_path,
)


OUTPUT_DIR = PROJECT_ROOT / "dashboard"
OUTPUT_PATH = OUTPUT_DIR / "dashboard_preview.png"


def build_dashboard_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["Regime"] = np.where(df["Date"] < pd.Timestamp("2018-01-01"), "Before 2018", "2018 onward")
    df["Range Value"] = df["High"] - df["Low"]
    df["Intramonth Return %"] = np.where(
        df["Open"] != 0, ((df["Close"] - df["Open"]) / df["Open"]) * 100, 0.0
    )
    df["Direction"] = np.where(df["Close"] >= df["Open"], "Closed Above Open", "Closed Below Open")
    df["MA_3"] = df["Close"].rolling(3).mean()
    df["MA_12"] = df["Close"].rolling(12).mean()
    df["Date Label"] = df["Date"].dt.strftime("%b-%Y")
    return df


def add_metric_box(fig, xy, title: str, value: str, color: str) -> None:
    x, y = xy
    fig.text(
        x,
        y,
        f"{title}\n{value}",
        ha="left",
        va="top",
        fontsize=11,
        color="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor="none"),
    )


def render_dashboard_preview() -> Path:
    raw_df = load_yes_bank_data(resolve_default_data_path(PROJECT_ROOT))
    dashboard_df = build_dashboard_dataframe(raw_df)
    model_df = build_modeling_frame(raw_df)
    metrics_df, _, _ = evaluate_models(model_df, FEATURE_COLUMNS)
    _, final_payload, _ = fit_final_model(model_df, FEATURE_COLUMNS)

    best_model_row = metrics_df.sort_values("RMSE").iloc[0]
    holdout_results = pd.DataFrame(
        {
            "Date": final_payload["test_df"]["Date"],
            "Actual": final_payload["y_test"].to_numpy(),
            "Predicted": final_payload["holdout_predictions"],
        }
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(18, 11), facecolor="#eef3f8")
    grid = fig.add_gridspec(
        3,
        3,
        left=0.04,
        right=0.98,
        top=0.84,
        bottom=0.06,
        wspace=0.22,
        hspace=0.28,
    )

    fig.text(
        0.04,
        0.94,
        "YES BANK STOCK CLOSING PRICE DASHBOARD",
        fontsize=24,
        fontweight="bold",
        color="#143d59",
    )
    fig.text(
        0.04,
        0.905,
        "Standalone GitHub visual preview for the Streamlit dashboard",
        fontsize=12,
        color="#516273",
    )

    add_metric_box(fig, (0.04, 0.875), "Rows", f"{len(raw_df)}", "#174e72")
    add_metric_box(fig, (0.17, 0.875), "Peak Close", f"{dashboard_df['Close'].max():.2f}", "#1f7a8c")
    add_metric_box(fig, (0.34, 0.875), "Best Model", str(best_model_row["Model"]), "#bc4b51")
    add_metric_box(fig, (0.55, 0.875), "Best RMSE", f"{best_model_row['RMSE']:.2f}", "#d3872b")
    add_metric_box(fig, (0.70, 0.875), "Latest Close", f"{dashboard_df.iloc[-1]['Close']:.2f}", "#5c88b4")

    ax1 = fig.add_subplot(grid[0, :2])
    ax1.plot(dashboard_df["Date"], dashboard_df["Close"], color="#143d59", linewidth=2.4, label="Close")
    ax1.plot(dashboard_df["Date"], dashboard_df["MA_3"], color="#f4b942", linewidth=1.8, label="3M Moving Avg")
    ax1.plot(dashboard_df["Date"], dashboard_df["MA_12"], color="#6c7a89", linewidth=1.8, label="12M Moving Avg")
    ax1.axvline(pd.Timestamp("2018-01-01"), color="#a43a2a", linestyle="--", linewidth=1.1)
    ax1.set_title("Closing Price Trend with Rolling Averages", fontsize=13, fontweight="bold")
    ax1.set_xlabel("")
    ax1.set_ylabel("Close")
    ax1.legend(loc="upper left", frameon=False, ncol=3)

    ax2 = fig.add_subplot(grid[0, 2])
    direction_counts = dashboard_df["Direction"].value_counts()
    colors = ["#1f7a8c", "#bc4b51"]
    ax2.pie(
        direction_counts.values,
        labels=direction_counts.index,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white"),
        textprops=dict(fontsize=9),
    )
    ax2.set_title("Monthly Return Direction", fontsize=13, fontweight="bold")

    ax3 = fig.add_subplot(grid[1, 0])
    regime_range = (
        dashboard_df.groupby("Regime")["Range Value"].mean().reindex(["Before 2018", "2018 onward"])
    )
    ax3.bar(regime_range.index, regime_range.values, color=["#7a9e9f", "#bc4b51"])
    ax3.set_title("Average Monthly Range by Regime", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Range Value")
    ax3.tick_params(axis="x", rotation=10)

    ax4 = fig.add_subplot(grid[1, 1])
    corr = dashboard_df[["Open", "High", "Low", "Close", "Range Value", "Intramonth Return %"]].corr()
    im = ax4.imshow(corr, cmap="Blues", aspect="auto")
    ax4.set_xticks(range(len(corr.columns)))
    ax4.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax4.set_yticks(range(len(corr.index)))
    ax4.set_yticklabels(corr.index, fontsize=8)
    for row_idx in range(len(corr.index)):
        for col_idx in range(len(corr.columns)):
            ax4.text(col_idx, row_idx, f"{corr.iloc[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=7)
    ax4.set_title("Correlation Heatmap", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(grid[1, 2])
    volatile_months = dashboard_df.nlargest(8, "Range Value")[["Date Label", "Range Value"]].iloc[::-1]
    ax5.barh(volatile_months["Date Label"], volatile_months["Range Value"], color="#d3872b")
    ax5.set_title("Top Volatile Months", fontsize=13, fontweight="bold")
    ax5.set_xlabel("Range Value")

    ax6 = fig.add_subplot(grid[2, :2])
    ax6.plot(holdout_results["Date"], holdout_results["Actual"], color="#143d59", linewidth=2.2, label="Actual")
    ax6.plot(holdout_results["Date"], holdout_results["Predicted"], color="#d3872b", linewidth=2.0, label="Predicted")
    ax6.set_title("Holdout Actual vs Predicted", fontsize=13, fontweight="bold")
    ax6.set_ylabel("Close")
    ax6.legend(loc="upper right", frameon=False)

    ax7 = fig.add_subplot(grid[2, 2])
    ax7.axis("off")
    findings = [
        "1. Post-2018 volatility is materially higher.",
        "2. Low is the strongest raw correlate of Close.",
        "3. Regularized linear models remain the most stable fit.",
        "4. Large downside shocks cluster around crisis periods.",
        "5. The dashboard combines EDA, regime analysis, and prediction quality in one view.",
    ]
    ax7.text(
        0,
        1,
        "Dashboard Story\n\n" + "\n".join(findings),
        va="top",
        fontsize=11,
        color="#233645",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="white", edgecolor="#cad5df"),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return OUTPUT_PATH


def main() -> None:
    output_path = render_dashboard_preview()
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
