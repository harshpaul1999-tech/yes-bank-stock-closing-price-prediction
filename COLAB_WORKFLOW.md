# Colab-Friendly Workflow

This project is set up so Codex can build everything locally first, and Colab can be used afterward as the presentation or execution layer.

## Recommended flow

1. Keep the project in a normal local folder or Git repo.
2. Let Codex create or update the notebook and supporting Python files locally.
3. Regenerate the notebook if needed:

```bash
cd yes-bank-stock-closing-price-prediction
python3 scripts/generate_notebook.py
```

4. Commit and push the notebook to GitHub:

```bash
git add .
git commit -m "feat: update yes bank colab notebook"
git push origin main
```

5. Generate a GitHub-backed Colab link:

```bash
python3 scripts/colab_link.py \
  notebooks/regression_yes_bank_stock_closing_price_prediction.ipynb
```

6. Open that Colab URL in the browser, or pass `--open` to open it automatically.

## Why this works well

- Codex edits stable local files instead of a fragile browser tab.
- GitHub becomes the source of truth for reopening the same notebook in Colab.
- The notebook can be regenerated from code if you later change the project logic.

## Example

After the notebook is pushed, the Colab URL will look like this shape:

```text
https://colab.research.google.com/github/<user>/<repo>/blob/<branch>/<path-to-notebook>.ipynb
```

## Important note

If a notebook exists only inside a live Google Drive Colab tab, Codex cannot reliably update it directly. The best long-term workflow is:

- local project files
- Git commit history
- optional GitHub push
- Colab opened from GitHub
