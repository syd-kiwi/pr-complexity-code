# PR Complexity Analysis

Research code for measuring pull request (PR) complexity, comparing PR characteristics by Copilot access, and studying merge time.

## Start here

Choose the workflow that matches your task:

| Task | Start here |
| --- | --- |
| Prepare PR data and calculate complexity | Follow the [notebook workflow](#notebook-workflow) |
| Compare PRs by Copilot access | Run the [RQ1 comparison](#compare-copilot-access-groups) |
| Compare merge time prediction models | Run the [merge time experiments](#compare-merge-time-models) |
| Find existing charts and tables | Browse [results/](results/) |

Source datasets are not included. You need compatible PR metadata to run the analyses.

## Research focus

The repository supports a PR complexity measurement study and an empirical study of PR behavior and merge time.

The empirical study focuses on two questions:

1. **RQ1: How do PR size, language composition, review activity, and merge time differ between authors with and without Copilot access?**
2. **RQ2: Can machine learning predict PR merge time based on limited features?**

The scripts in `paper2/` provide descriptive comparisons, model comparisons, feature ablations, and evaluation across repositories or merge periods. See the [experiment guide](paper2/README.md) for implemented analyses and the [research plan](paper2/RESEARCH_PLAN.md) for related work and evaluation details.

Copilot seat status indicates access. It does not confirm tool use or AI authorship. Comparisons describe associations and do not establish a causal effect.

## Setup

Clone the repository and create an environment:

```bash
git clone https://github.com/syd-kiwi/pr-complexity-code.git
cd pr-complexity-code
python -m venv .venv
```

Activate it:

- **Windows PowerShell:** `.venv\Scripts\Activate.ps1`
- **macOS/Linux:** `source .venv/bin/activate`

### Notebook workflow

Follow notebooks 1–5 in order. If you already have compatible PR metadata, start at notebook 2. Use this workflow to calculate the complexity score; the merge time scripts above can run independently.

The root `requirements.txt` currently contains unresolved Git merge conflict markers and cannot be installed as written. For the core notebook workflow, install the main packages directly:

```bash
python -m pip install jupyterlab pandas numpy matplotlib seaborn scipy scikit-learn joblib tqdm langdetect wordcloud
jupyter lab
```

Internal data extraction and optional analyses require additional dependencies. Notebook 1 requires internal Trino access and `ecm_mlflow_libs`.

Open the notebooks from `notebooks/`. Their relative paths assume that folder is the working directory.

### Merge time experiments

The `paper2/` scripts have a separate requirements file and run independently of the root requirements:

```bash
python -m pip install -r paper2/requirements.txt
```

Run the following commands from the repository root. The examples use `data/cleaned_public.csv`; replace that path with your input file. Each output directory must be new or empty.

### Compare Copilot access groups

Prepare the data, then generate descriptive comparisons without training models:

```bash
python paper2/prepare.py --input data/cleaned_public.csv --output paper2/outputs/prepared01 --scope all
python paper2/analyze.py --input paper2/outputs/prepared01/prepared.parquet --output paper2/outputs/rq1_01
```

This produces group summaries, correlations, within-repository merge time comparisons, and a cumulative merge time figure. Start with `by_had_copilot_seat.csv` and `within_repository_effect.json` in `paper2/outputs/rq1_01/`.

The example retains public and private rows with `--scope all`. Use `--scope open-source` or `--scope private` for a specific population. Check `prepared01/audit.json` for actual row counts and filtering.

### Compare merge time models

For a small CPU development run:

```bash
python paper2/run_pipeline.py --input data/cleaned_public.csv --output paper2/outputs/dev01 --scope all --device cpu --trials 1 --trees 30 --permutation-rows 100
```

Use a small compatible dataset with enough repositories and examples of all three duration classes. These settings check the workflow; they are not a final experiment.

For the standard GPU experiment:

```bash
python paper2/run_pipeline.py --input data/cleaned_public.csv --output paper2/outputs/run01 --scope all --device cuda:0
```

The pipeline prepares the data, runs the Copilot comparisons, and compares XGBoost, logistic regression, and a majority baseline. It holds out entire repositories by default. GPU training requires a compatible NVIDIA setup.

See [paper2/README.md](paper2/README.md) for feature sets, evaluation metrics, text features, ordinal classification, duration regression, and saved model evaluation.

## Notebook workflow

| Step | Notebook | Purpose |
| --- | --- | --- |
| 1 | [Build dataset](notebooks/1.build_customer_dataset.ipynb) | Query source PR metadata using internal access. |
| 2 | [Clean dataset](notebooks/2.clean_dataset.ipynb) | Clean metadata and prepare title and body features. |
| 3 | [Categorize PRs](notebooks/3.categorize_prs_keywords.ipynb) | Match keyword categories in PR titles and bodies. |
| 4 | [Calculate complexity](notebooks/4.calculate_pr_complexity_score.ipynb) | Calculate scores and examine their relationship with merge time. |
| 5 | [Component ablation](notebooks/5.component_ablation_analysis.ipynb) | Remove score components and compare the resulting patterns. |

Before running, review each notebook's loading and saving cells. Paths are not fully standardized: some use `../data/cleaned_public.csv`, while others use files under `../data/microsoft/`. Set them to your own files and make sure each output matches the next notebook's input. Notebook 2 currently loads all CSVs in `data/` in one cell and reloads `cleaned_public.csv` in another; choose the intended input before executing it.

## Complexity score

The implemented score combines five percentile based components:

| Component | Input | Weight |
| --- | --- | --- |
| Code size | Code lines modified, with total modified lines as a fallback | 30% |
| File breadth | Files touched | 20% |
| Review burden | Review comments and reviewer count | 25% |
| Configuration impact | Configuration lines modified | 10% |
| Keyword breadth | Number of matched keyword categories | 15% |

Review burden combines the percentile ranks of review comments (60%) and reviewer count (40%). The final weighted score is multiplied by 100.

These weights are researcher selected and shared in the code. To calculate scores on your own dataset, compute the component percentiles using your own dataset. Notebook 4 uses average ranks for ties after a `log1p` transformation. Report the dataset, feature definitions, and missing value handling when sharing results.

Because the score includes completed review activity, it describes realized complexity. Its association with merge time does not establish prediction before review. The merge time models exclude the composite score and compare review features separately. The available snapshots also require retrospective interpretation because opening-time feature availability has not been verified.

## Repository guide

| Folder | Contents |
| --- | --- |
| `notebooks/` | Main dataset and complexity workflow |
| [`paper2/`](paper2/README.md) | Copilot comparisons and merge time experiments |
| `scripts/` | GPU categorization, GPU scoring, and a merge time prediction script |
| `additional_analysis/` | Optional language, topic, text, and modeling analyses |
| `results/` | Saved charts, summary tables, and component ablation outputs |

Keep source datasets and row level outputs local unless you have permission to share them. Saved results belong to particular inputs and runs; rerunning on another dataset may produce different findings.
