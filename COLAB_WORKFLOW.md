# Colab-Friendly Workflow

This repository is set up so Codex can build and update the notebook locally, and you can reopen the same work cleanly in Google Colab through GitHub.

## Recommended Flow

1. Make code and notebook changes locally in this standalone repository.
2. Regenerate the notebook:

```bash
cd /Users/harshpaul/Documents/yes-bank-stock-closing-price-prediction
python3 scripts/generate_notebook.py
python3 scripts/generate_documents.py
```

3. Commit and push the updated files:

```bash
cd /Users/harshpaul/Documents/yes-bank-stock-closing-price-prediction
git add .
git commit -m "feat: refresh yes bank notebook and project docs"
git push origin main
```

4. Generate or open the Colab link:

```bash
cd /Users/harshpaul/Documents/yes-bank-stock-closing-price-prediction
python3 scripts/colab_link.py Yes_bank_stock_closing_price_prediction.ipynb --open
```

## Why This Works Well

- Codex can safely edit normal local files.
- GitHub becomes the single source for the notebook version you want to present.
- Colab opens the exact notebook from the repository without needing manual copy-paste into a Drive notebook.

## Primary Notebook Paths

- Root presentation notebook:
  `Yes_bank_stock_closing_price_prediction.ipynb`
- Mirrored legacy notebook path:
  `notebooks/regression_yes_bank_stock_closing_price_prediction.ipynb`

The generator updates both files together so GitHub and Colab links remain stable.
