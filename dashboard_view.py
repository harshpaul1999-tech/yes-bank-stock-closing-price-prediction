from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _prepare_dashboard_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["Regime"] = np.where(df["Date"] < pd.Timestamp("2018-01-01"), "Before 2018", "2018 onward")
    df["Return Direction"] = np.where(df["Close"] >= df["Open"], "Closed Above Open", "Closed Below Open")
    df["Range Value"] = df["High"] - df["Low"]
    df["Range %"] = np.where(df["Open"] != 0, ((df["High"] - df["Low"]) / df["Open"]) * 100, 0.0)
    df["Intramonth Return %"] = np.where(df["Open"] != 0, ((df["Close"] - df["Open"]) / df["Open"]) * 100, 0.0)
    df["Month"] = df["Date"].dt.strftime("%b")
    df["Year"] = df["Date"].dt.year
    df["Date Label"] = df["Date"].dt.strftime("%b-%Y")
    df["MA_3"] = df["Close"].rolling(3).mean()
    df["MA_12"] = df["Close"].rolling(12).mean()
    return df


def _correlation_figure(df: pd.DataFrame) -> go.Figure:
    corr = df[["Open", "High", "Low", "Close", "Range Value", "Intramonth Return %"]].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_layout(
        title="Correlation Heatmap",
        margin=dict(l=10, r=10, t=50, b=10),
        coloraxis_colorbar_title="Corr",
    )
    return fig


def render_dashboard(assets: dict) -> None:
    raw_df = assets["raw_df"]
    metrics_df = assets["metrics_df"]
    holdout_results = assets["holdout_results"]
    dashboard_df = _prepare_dashboard_frame(raw_df)

    latest = dashboard_df.iloc[-1]
    best_model = metrics_df.sort_values("RMSE").iloc[0]["Model"]

    top_cols = st.columns(4)
    top_cols[0].metric("Best Model", best_model)
    top_cols[1].metric("Peak Close", "{0:.2f}".format(dashboard_df["Close"].max()))
    top_cols[2].metric("Latest Monthly Range", "{0:.2f}".format(latest["Range Value"]))
    top_cols[3].metric("Positive Months", int((dashboard_df["Close"] >= dashboard_df["Open"]).sum()))

    row1_col1, row1_col2 = st.columns((1.5, 1))
    with row1_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        line_fig = go.Figure()
        line_fig.add_trace(
            go.Scatter(
                x=dashboard_df["Date"],
                y=dashboard_df["Close"],
                name="Close",
                line=dict(color="#143d59", width=3),
            )
        )
        line_fig.add_trace(
            go.Scatter(
                x=dashboard_df["Date"],
                y=dashboard_df["MA_3"],
                name="3M Moving Avg",
                line=dict(color="#f4b942", width=2),
            )
        )
        line_fig.add_trace(
            go.Scatter(
                x=dashboard_df["Date"],
                y=dashboard_df["MA_12"],
                name="12M Moving Avg",
                line=dict(color="#6c7a89", width=2),
            )
        )
        line_fig.add_vline(x=pd.Timestamp("2018-01-01"), line_dash="dash", line_color="#a43a2a")
        line_fig.update_layout(
            title="Closing Price Trend with Moving Averages",
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01),
        )
        st.plotly_chart(line_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row1_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        donut_fig = px.pie(
            dashboard_df,
            names="Return Direction",
            hole=0.55,
            color="Return Direction",
            color_discrete_map={
                "Closed Above Open": "#1f7a8c",
                "Closed Below Open": "#bc4b51",
            },
            title="Monthly Return Direction",
        )
        donut_fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(donut_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    row2_col1, row2_col2 = st.columns((1, 1))
    with row2_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        regime_counts = dashboard_df["Regime"].value_counts().rename_axis("Regime").reset_index(name="Months")
        pie_fig = px.pie(
            regime_counts,
            names="Regime",
            values="Months",
            title="Observation Share by Market Regime",
            color="Regime",
            color_discrete_map={"Before 2018": "#7a9e9f", "2018 onward": "#bc4b51"},
        )
        pie_fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(pie_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row2_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        volatile_months = dashboard_df.nlargest(10, "Range Value")[["Date Label", "Range Value"]].iloc[::-1]
        bar_fig = px.bar(
            volatile_months,
            x="Range Value",
            y="Date Label",
            orientation="h",
            title="Top 10 Most Volatile Months",
            color="Range Value",
            color_continuous_scale="Oranges",
        )
        bar_fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False)
        st.plotly_chart(bar_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    row3_col1, row3_col2 = st.columns((1, 1))
    with row3_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        box_fig = px.box(
            dashboard_df,
            x="Regime",
            y="Close",
            color="Regime",
            title="Closing Price Distribution by Regime",
            color_discrete_map={"Before 2018": "#5c88b4", "2018 onward": "#d3872b"},
        )
        box_fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
        st.plotly_chart(box_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row3_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        scatter_fig = px.scatter(
            dashboard_df,
            x="Open",
            y="Close",
            color="Regime",
            size="Range Value",
            hover_name="Date Label",
            title="Open vs Close with Volatility Bubble Size",
            color_discrete_map={"Before 2018": "#1f7a8c", "2018 onward": "#bc4b51"},
        )
        scatter_fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(scatter_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    row4_col1, row4_col2 = st.columns((1, 1))
    with row4_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        heatmap_fig = _correlation_figure(dashboard_df)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row4_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        holdout_fig = go.Figure()
        holdout_fig.add_trace(
            go.Scatter(
                x=holdout_results["Date"],
                y=holdout_results["Actual"],
                mode="lines+markers",
                name="Actual",
                line=dict(color="#143d59", width=2.6),
            )
        )
        holdout_fig.add_trace(
            go.Scatter(
                x=holdout_results["Date"],
                y=holdout_results["Predicted"],
                mode="lines+markers",
                name="Predicted",
                line=dict(color="#d3872b", width=2.4),
            )
        )
        holdout_fig.update_layout(
            title="Holdout Actual vs Predicted",
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="Month",
            yaxis_title="Close",
        )
        st.plotly_chart(holdout_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Key Findings")
    st.markdown(
        """
        - The stock’s volatility regime changed sharply after 2018, with the average monthly trading range jumping materially.
        - `Low` remains the most tightly aligned raw price variable with the final closing price, which helps explain why same-month OHLC features predict closing price strongly.
        - The biggest downside shocks cluster in the 2018 governance crisis and the 2020 COVID period.
        - The chronological holdout set shows that simpler linear models remain more stable than KNN and tree-heavy alternatives for this dataset.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
