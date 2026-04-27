# Regression - Yes Bank Stock Closing Price Prediction

This standalone repository presents a capstone-style machine learning project focused on predicting the **monthly closing price of Yes Bank stock**. The dataset spans **July 2005 to November 2020** and becomes especially interesting after the **2018 governance crisis**, when the stock behavior shifts into a visibly more volatile regime.

The project is designed for GitHub review, Google Colab walkthroughs, interview demos, and Streamlit-based storytelling. It includes a polished notebook, project PDFs, the raw dataset, reusable modeling utilities, and a Gemini-enabled interactive app.

## Problem Statement

Yes Bank is a well-known name in the Indian banking sector. Since 2018, the bank has remained in the news because of the Rana Kapoor fraud case and the subsequent confidence shock faced by the business. This raises an important machine learning question:

Can regression models still estimate the monthly closing stock price accurately when the stock history contains a major structural break?

The target variable in this project is the **monthly closing price**.

## Business Context

This project should be understood as a **monthly close estimation system** rather than a pure trading-signal engine. It uses same-month OHLC behavior together with historical context to explain how the final close behaves under both normal and stress periods.

That makes it a strong interview project because it combines:

- business framing
- financial data storytelling
- EDA and feature engineering
- time-aware model evaluation
- explainability
- an interactive GenAI-enabled app

## Repository Deliverables

```text
.
├── Yes_bank_stock_closing_price_prediction.ipynb
├── Yes_Bank Technical Documentation.pdf
├── summary.pdf
├── data_YesBank_StockPrices.csv
├── app.py
├── dashboard/
│   ├── README.md
│   └── dashboard_preview.png
├── data/
│   └── yes_bank_stock_prices.csv
├── notebooks/
│   └── regression_yes_bank_stock_closing_price_prediction.ipynb
├── scripts/
│   ├── colab_link.py
│   ├── generate_dashboard_assets.py
│   ├── generate_documents.py
│   └── generate_notebook.py
├── dashboard_view.py
└── src/
    └── yes_bank_ml.py
```

## Notebook Coverage

The Colab-style notebook includes the following sections in order:

- problem statement
- business context
- dataset loading and clean-up
- data understanding
- univariate, bivariate, and multivariate EDA
- feature engineering
- multicollinearity inspection
- target feature conditioning
- model implementation
- evaluation summary
- feature importance and explainability
- conclusions drawn

## Key Findings

- The dataset contains **185 monthly observations** with no missing values and no duplicate rows.
- The strongest raw relationship with the closing price comes from **Low**, followed by **High** and **Open**.
- The stock behaves very differently after **2018**, with a sharp increase in average monthly trading range and a negative average intramonth return.
- The biggest volatility shocks cluster around the **2018 crisis period** and the **2020 COVID shock**.
- Chronology matters. When the holdout set is preserved in time order, simpler linear and regularized linear models outperform KNN and tree-heavy alternatives.

## Evaluation Summary

Chronological holdout results from the final notebook workflow:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 10.08 | 14.48 | 0.987 |
| Lasso Regression | 11.05 | 16.43 | 0.983 |
| Ridge Regression | 11.48 | 16.83 | 0.983 |
| ElasticNet Regression | 13.11 | 19.26 | 0.977 |
| Gradient Boosting | 22.63 | 33.62 | 0.930 |
| Random Forest | 29.11 | 36.85 | 0.916 |
| Naive Previous Close | 23.80 | 38.82 | 0.907 |
| KNN Regressor | 58.70 | 74.81 | 0.655 |

For deployment, the project uses **Lasso Regression** because it offers a strong balance of predictive quality, regularization, and explainability.

## Streamlit App

The app is built with **Streamlit** and includes:

- a presentation-heavy dashboard with line charts, pie charts, donut charts, boxplots, scatterplots, heatmaps, and holdout tracking
- a model lab with evaluation tables, feature importance, and scenario-based prediction
- a **Gemini API** assistant for interview-style questions and project explanations
- downloadable project deliverables directly from the interface

For GitHub readability, the repository also includes a visual dashboard deliverable in [dashboard/README.md](./dashboard/README.md) so reviewers can see the dashboard story as a diagram/preview instead of only browsing Python code.

Run locally:

```bash
git clone https://github.com/harshpaul1999-tech/yes-bank-stock-closing-price-prediction.git
cd yes-bank-stock-closing-price-prediction
python3 scripts/generate_notebook.py
python3 scripts/generate_documents.py
python3 scripts/generate_dashboard_assets.py
streamlit run app.py
```

## Gemini Setup

```bash
export GEMINI_API_KEY="your_api_key_here"
export GEMINI_MODEL_NAME="gemini-2.5-flash"
```

If the Gemini variables are missing, the app still works in dashboard mode and shows setup guidance for the chat feature.

## Colab Workflow

This repo is designed to work cleanly with Google Colab:

1. Regenerate the notebook locally if needed.
2. Commit and push the notebook to GitHub.
3. Open the root notebook in Colab using the GitHub-backed link.

Helper command:

```bash
python3 scripts/colab_link.py Yes_bank_stock_closing_price_prediction.ipynb --open
```

## Public Deployment

The repository now includes [render.yaml](./render.yaml) so the Streamlit app can be deployed as a public Render web service directly from this GitHub repo.

Start command used by Render:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Conclusion

Yes Bank’s closing price can be estimated very effectively when same-month OHLC structure and recent price memory are used together. The dataset is not difficult because it is noisy; it is difficult because the stock enters a new regime after 2018. That makes this project a useful demonstration of why **time-aware evaluation, careful feature engineering, and regularized linear models** matter more than blindly using more complex algorithms.
