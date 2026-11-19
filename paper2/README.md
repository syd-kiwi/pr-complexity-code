# Paper 2: Copilot Access and PR Merge Time

This folder compares PRs by Copilot access and tests models that predict merge time.

## Research questions

1. **RQ1:** How do title and body length, PR size, language composition, review activity, and merge time differ between authors with and without Copilot access?
2. **RQ2:** Can machine learning predict PR merge time based on limited features?

## Setup

Run commands from the repository root. Use Python 3.10 or newer and your own compatible dataset.

```bash
python -m pip install -r paper2/requirements.txt
```

The examples use `data/cleaned_public.csv`. Change that path if your file is somewhere else. Use a new output folder for each run.

## Run the RQ1 notebook

Open [PR_Merge_Time_RQ1_RQ2.ipynb](PR_Merge_Time_RQ1_RQ2.ipynb) in Jupyter. From the repository root, run:

```bash
python -m pip install jupyterlab 'pandas>=2.2,<4' 'numpy>=1.26,<3'
python -m jupyterlab paper2/PR_Merge_Time_RQ1_RQ2.ipynb
```

Place your dataset at `data/cleaned_public.csv` or set `CSV_PATH` in the first code cell, then select **Run All**. The notebook reads the CSV in chunks and displays pooled and within-repository RQ1 comparisons. It also summarizes the completed RQ2 results. The dataset is supplied separately.

## Compare Copilot groups from the command line

```bash
python paper2/prepare.py --input data/cleaned_public.csv --output paper2/outputs/prepared01 --scope all
python paper2/analyze.py --input paper2/outputs/prepared01/prepared.parquet --output paper2/outputs/rq1_01
```

Look in `paper2/outputs/rq1_01/` for the comparison tables and merge time chart. These commands do not train models.

## Compare prediction models

```bash
python paper2/run_pipeline.py --input data/cleaned_public.csv --output paper2/outputs/run01 --scope all --device cuda:0
```

This runs the data preparation, group comparisons, and model training. It compares XGBoost, logistic regression, and a simple baseline.

The command requires a compatible NVIDIA GPU. For a quick CPU check on a small dataset, use `--device cpu --trials 1 --trees 30 --permutation-rows 100`.

## Find the results

| Folder | Contents |
| --- | --- |
| `prepared/` | Cleaned modeling data and a data audit |
| `rq1/` | Copilot group comparisons |
| `repository_seed42/` | Model scores, predictions, and saved models |

For text features and other experiments, run `python paper2/run_research.py --help`. See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for related work and experiment details.

Results describe merged PRs. Copilot status means access, not confirmed use. Review counts and other fields may reflect the final PR state, so these experiments do not establish prediction at PR opening.
