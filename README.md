# PR Complexity Analysis

Code for measuring pull request (PR) complexity, comparing PRs by Copilot access, and studying merge time.

## Choose a workflow

| Goal | Start here |
| --- | --- |
| Build and clean PR metadata | [Notebook 1](paper1/notebooks/1.build_customer_dataset.ipynb), then [Notebook 2](paper1/notebooks/2.clean_dataset.ipynb) |
| Assign keyword categories | [Notebook 3](paper1/notebooks/3.categorize_prs_keywords.ipynb) |
| Calculate complexity and examine merge time | [Notebook 4](paper1/notebooks/4.calculate_pr_complexity_score.ipynb) |
| Compare score components | [Notebook 5](paper1/notebooks/5.component_ablation_analysis.ipynb) |
| Compare Copilot groups and merge time models | [Experiment guide](paper2/README.md) |
| Edit the complexity manuscript | [Paper repository](https://github.com/syd-kiwi/pr-complexity-paper/tree/main/paper) |
| Edit the empirical study manuscript | [Empirical study repository](https://github.com/syd-kiwi/An-Empirical-Study-of-the-Pull-Based-Software-Development-Model/tree/main/paper) |

The `paper1/` folder holds the complexity workflow. The complexity study evaluates a measurement framework. The empirical study compares PR characteristics and merge time models. The `paper2/` directory contains code for the empirical study; its folder name is retained for compatibility.

## Empirical research questions

1. **RQ1: How do PR size, language composition, review activity, and merge time differ between authors with and without Copilot access?**
2. **RQ2: Can machine learning predict PR merge time based on limited features?**

Feature ablations, text representations, and generalization checks support RQ2. Copilot seat status measures access, not confirmed tool use or AI authorship.

## Setup

```bash
git clone https://github.com/syd-kiwi/pr-complexity-code.git
cd pr-complexity-code
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in Windows PowerShell.

### Complexity notebooks

The root `requirements.txt` contains unresolved merge conflict markers. Install the main notebook packages directly until that file is repaired:

```bash
python -m pip install jupyterlab pandas numpy matplotlib seaborn scipy scikit-learn joblib tqdm langdetect wordcloud
jupyter lab
```

Follow notebooks 1–5 in order, or start at notebook 2 with compatible metadata. Notebook 1 requires internal Trino access and `ecm_mlflow_libs`.

Notebook paths assume `paper1/notebooks/` is the working directory. Review loading and saving cells before running: some use `../data/cleaned_public.csv`, others use `../data/microsoft/`. With the notebooks now under `paper1/`, a `../data/` path resolves to `paper1/data/`. Update it if your data remains in the repository root. Notebook 2 has separate cells loading all CSVs and reloading the cleaned CSV; choose your intended input.

### Copilot comparisons and merge time models

These scripts have independent dependencies. Run commands from the repository root:

```bash
python -m pip install -r paper2/requirements.txt
python paper2/prepare.py --input data/cleaned_public.csv --output paper2/outputs/prepared01 --scope all
python paper2/analyze.py --input paper2/outputs/prepared01/prepared.parquet --output paper2/outputs/rq1_01
```

This prepares data and runs descriptive comparisons without training models. Use new or empty output directories. `--scope all` retains public and private rows; use `open-source` or `private` for a specific population.

For model comparisons, CPU development commands, GPU experiments, and output definitions, follow [paper2/README.md](paper2/README.md).

## Complexity score

| Component | Post-review weight |
| --- | --- |
| Code lines modified | 30% |
| Files touched | 20% |
| Review burden | 25% |
| Configuration lines modified | 10% |
| Keyword category count | 15% |

Review burden combines percentile ranks of review comments (60%) and reviewer count (40%). The weighted sum is multiplied by 100. Notebook 4 uses average percentile ranks after `log1p`, with total modified lines as a fallback for code lines.

Weights are researcher selected. Recompute percentiles using your own dataset and report feature definitions, ties, and missing value handling. The manuscript also defines a pre-review formulation; check its [measure section](https://github.com/syd-kiwi/pr-complexity-paper/blob/main/paper/sections/03-complexity-measure.tex) before assuming every variant is implemented in these notebooks.

Completed review counts describe realized activity. Models using current snapshots require retrospective interpretation because feature availability at PR opening has not been verified. The empirical models exclude the composite complexity score and compare review features separately.

## Other folders

| Folder | Contents |
| --- | --- |
| `paper1/scripts/` | GPU categorization, GPU scoring, and merge time prediction |
| `paper1/additional_analysis/` | Optional language, topic, text, and modeling analyses |
| `paper1/results/` | Saved charts, tables, and ablation outputs |

Source data is not included. Keep datasets and row level outputs local unless sharing is authorized. Saved results describe their original inputs and runs.
