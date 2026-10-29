# PR Complexity Analysis

Research code for measuring pull request (PR) complexity and studying its relationship with merge time. The workflow cleans PR metadata, assigns keyword categories, calculates complexity scores, and compares score components.

## Get started

```bash
git clone https://github.com/syd-kiwi/pr-complexity-code.git
cd pr-complexity-code
python -m venv .venv
```

Activate the environment:

- **Windows PowerShell:** `.venv\Scripts\Activate.ps1`
- **macOS/Linux:** `source .venv/bin/activate`

**Setup note:** `requirements.txt` currently contains Git merge conflict markers. Resolve that block before installing dependencies. Some notebooks also need packages such as `wordcloud` that are not listed there.

```bash
python -m pip install -r requirements.txt
python -m pip install jupyterlab wordcloud
jupyter lab
```

## Run the notebooks

Open notebooks from the `notebooks/` folder. Update input and output paths for your environment before running cells in order.

| Notebook | Purpose |
| --- | --- |
| [1. Build dataset](notebooks/1.build_customer_dataset.ipynb) | Query source PR data. Requires internal Trino access and the `ecm_mlflow_libs` library. |
| [2. Clean dataset](notebooks/2.clean_dataset.ipynb) | Load and clean PR data. |
| [3. Categorize PRs](notebooks/3.categorize_prs_keywords.ipynb) | Match keyword categories in PR titles and bodies. |
| [4. Calculate complexity](notebooks/4.calculate_pr_complexity_score.ipynb) | Calculate scores and examine merge time patterns. |
| [5. Component ablation](notebooks/5.component_ablation_analysis.ipynb) | Remove one component at a time and compare results. |

Source datasets are not included. If you already have compatible PR metadata, start at notebook 2 and place your input CSV files in a local `data/` folder. Each notebook expects specific columns and file names; check its loading cells first.

## Complexity score

The score combines five percentile based components:

| Component | Weight |
| --- | --- |
| Code lines modified | 30% |
| Files touched | 20% |
| Review burden | 25% |
| Configuration lines modified | 10% |
| Keyword category breadth | 15% |

Review burden combines review comments (60%) and reviewer count (40%). The score includes review activity observed during the PR lifecycle, so it describes realized complexity rather than predicting difficulty before review.

## Other folders

- `scripts/`: GPU categorization and scoring scripts, plus merge time prediction.
- `additional_analysis/`: Optional language, topic, and modeling analyses.
- `results/`: Saved charts, summary tables, and ablation outputs.
