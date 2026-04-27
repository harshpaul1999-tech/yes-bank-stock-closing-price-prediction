# Regression - Yes Bank Stock Closing Price Prediction

Machine Learning & GenAI with Microsoft Azure.

## What is included

- A Colab-style notebook that walks through business context, data understanding, cleanup, EDA, feature engineering, target conditioning, model implementation, explainability, conclusion, and suggested Git commit checkpoints
- A reusable ML utility module shared by the analysis workflow and the app
- A Streamlit app for dashboarding, scenario-based close price prediction, and Azure OpenAI interview Q&A
- A local copy of the Yes Bank stock dataset

## Project structure

```text
.
├── app.py
├── COLAB_WORKFLOW.md
├── data/
│   └── yes_bank_stock_prices.csv
├── notebooks/
│   └── regression_yes_bank_stock_closing_price_prediction.ipynb
├── requirements.txt
├── scripts/
│   ├── colab_link.py
│   └── generate_notebook.py
└── src/
    └── yes_bank_ml.py
```

## Run locally

```bash
git clone https://github.com/harshpaul1999-tech/yes-bank-stock-closing-price-prediction.git
cd yes-bank-stock-closing-price-prediction
python3 scripts/generate_notebook.py
python3 -m streamlit run app.py
```

## Colab-friendly workflow

Use this pattern whenever you want Codex to build notebooks that later open cleanly in Google Colab:

1. Keep the project in a normal local folder or Git repo.
2. Let Codex create or update the notebook locally.
3. Regenerate the notebook if needed with `python3 scripts/generate_notebook.py`.
4. Commit and push the notebook to GitHub.
5. Generate a Colab URL with:

```bash
python3 scripts/colab_link.py \
  notebooks/regression_yes_bank_stock_closing_price_prediction.ipynb
```

6. Add `--open` if you want the script to open the Colab URL automatically.

More detailed notes are in [COLAB_WORKFLOW.md](./COLAB_WORKFLOW.md).

## Azure GenAI setup

Set these environment variables before launching the app if you want the Microsoft Azure assistant enabled:

```bash
export AZURE_OPENAI_ENDPOINT="https://YOUR-RESOURCE-NAME.openai.azure.com"
export AZURE_OPENAI_API_KEY="your_api_key_here"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4.1-mini"
```

The app uses Azure OpenAI through the current Responses API flow exposed by the `openai` Python package.
