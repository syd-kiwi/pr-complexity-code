# Paper 2: Copilot Access and PR Merge Time

## RQ1. Compare PRs by Copilot access

Start with [the notebook](PR_Merge_Time_RQ1_RQ2.ipynb). From the repository root:

```bash
python -m pip install jupyterlab "pandas>=2.2,<4" "numpy>=1.26,<3"
python -m jupyterlab paper2/PR_Merge_Time_RQ1_RQ2.ipynb
```

Put your CSV at `data/cleaned_public.csv` or set `CSV_PATH` in the first cell, then select **Run All**. The notebook compares Copilot seat groups overall and within repositories. It also summarizes the completed RQ2 results. The final cell saves a timestamped `rq1_report_*.md` in `paper2/outputs/` and prints its location. Share that report when adding the RQ1 results to the manuscript.

## RQ2. Run the merge-time models

To train the baseline models on an NVIDIA GPU, run:

```bash
python -m pip install -r paper2/requirements.txt
python paper2/run_pipeline.py --input data/cleaned_public.csv --output paper2/outputs/run01 --scope all --device cuda:0
```

Model results go to `paper2/outputs/run01/`. Use a fresh output directory for each run. For a quick check on a small dataset, use `--device cpu --trials 1 --trees 30 --permutation-rows 100`.

### Compare CatBoost on RunPod

From the repository root on a pod with an NVIDIA GPU, run:

```bash
python -m pip install "catboost>=1.2,<2" "pandas>=2.2,<4" "scikit-learn>=1.4,<2"
python paper2/run_catboost.py --input data/cleaned_public.csv --output paper2/outputs/catboost01 --device cuda:0
```

The script reads the CSV directly and trains CatBoost with and without final review counts. It uses the paper's public-PR filter and repository split procedure. If you have `splits.csv.gz` from the original `a100_run01`, add `--reference-splits /path/to/splits.csv.gz` to verify PR keys, split membership, and duration labels against that run.

Upload `paper2/outputs/catboost01/summary.md` to share the results. The folder also contains metrics, confusion matrices, validation trials, and saved models. Use a new output path for each run.

See the [research plan](RESEARCH_PLAN.md) for additional experiments and [the archived runs](https://github.com/syd-kiwi/pr-complexity-code/tree/archive/paper2-classifier-results/paper2) for earlier classifier outputs.
