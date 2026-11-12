# Paper 2: PR merge time and Copilot access

**As AI tools become part of development, what explains how quickly pull requests move through review and merge?**

This folder contains scripts for the approximately 1.2 million PR dataset with the supplied 37-column schema. The scripts compute actual row counts; the dataset is not included. They run independently of the repository's existing notebooks and root requirements file.

## Research questions

| Question | Analysis | Main outputs |
| --- | --- | --- |
| RQ1: How do change characteristics, review activity, and merge time differ between authors with and without Copilot access? | Group summaries, Spearman correlations, and within-repository differences in group median merge time | `rq1/` tables, bootstrap interval, and cumulative distribution figure |
| RQ2: How accurately can PR metadata classify merge speed, and does performance generalize across repositories or merge periods? | XGBoost, logistic regression, and majority baseline on identical splits | `metrics.csv`, confusion matrices, classification reports, models, and comparison figure |
| RQ3: What predictive information do description features, Copilot access, and completed review activity add? | Paired feature ablations and subgroup evaluation | `feature_comparisons.csv`, `subgroup_metrics.csv`, and validation permutation importance |

These questions concern associations and predictive performance. Copilot seat status measures access, not confirmed use or AI authorship. No causal effect is estimated. The data contains merged PRs, so the models do not predict whether an arbitrary opened PR will merge.

## Install on a GPU machine

Use Python 3.10+ on a machine with a compatible NVIDIA driver and a CUDA-enabled XGBoost wheel. Check the CUDA requirements of the installed XGBoost version. PyTorch, RAPIDS, and a separate CUDA toolkit are not required by these scripts.

```bash
git clone https://github.com/syd-kiwi/pr-complexity-code.git
cd pr-complexity-code
python -m venv .venv-paper2
source .venv-paper2/bin/activate
python -m pip install --upgrade pip
python -m pip install -r paper2/requirements.txt
nvidia-smi
```

On Windows PowerShell, activate with `.venv-paper2\Scripts\Activate.ps1`.

The default source file is `data/cleaned_public.csv`. Run commands from the repository root so this path resolves correctly. Override it with `--input` when needed. Accepted inputs: one CSV, compressed CSV (`.csv.gz`), or Parquet file. Convert a multi-file extract to one Parquet file first. Do not concatenate multiple snapshots of the same PR.

## Run the full experiment

From the repository root:

```bash
python paper2/run_pipeline.py \
  --output paper2/outputs/run01 \
  --scope open-source \
  --device cuda:0
```

The default keeps rows where `is_open_source` is true, includes zero-minute merges, and holds out entire repositories (approximately 70% train, 15% validation, 15% test by repository count). Actual row fractions vary. Use `--scope all` to retain both public and private rows or `--scope private` for private rows. Missing open-source status is retained only under `all`. The preparation audit reports the scope and filtering counts.

The model trains on **one GPU per process**. `--device cuda:1` selects another GPU. The five feature sets run sequentially to limit GPU memory usage. GPU training is checked explicitly; unavailable CUDA produces an error instead of silently reporting CPU training as GPU training. Preparation, descriptive statistics, logistic regression, and most preprocessing run on CPU. All selected columns fit in host RAM; this is not an out-of-core or distributed training pipeline. Raw titles, bodies, and NLP text copies are not loaded.

For repeated repository splits and a separate retrospective merge-period evaluation:

```bash
python paper2/run_pipeline.py \
  --input data/cleaned_public.csv \
  --output paper2/outputs/repeated01 \
  --device cuda:0 \
  --splits repository merge-time \
  --seeds 42 43 44 \
  --trials 6 \
  --trees 1200
```

Merge-period splits are deterministic and do not change with the seed; repeated seeds vary model tuning. Repository split seeds vary both held-out repositories and model fitting. Compare runs within the same split protocol. Each run reports its own thresholds and class counts. To use identical class definitions across runs, prespecify practical thresholds with `--thresholds FAST_MINUTES SLOW_MINUTES`; do not choose these after examining test results.

For a smaller development run on your machine:

```bash
python paper2/run_pipeline.py \
  --input paper2/data/development_sample.csv \
  --output paper2/outputs/dev01 \
  --device cpu --trials 1 --trees 30 --permutation-rows 100
```

Use a sample containing enough repositories and examples of all three classes. This is a development check, not the full experiment. Output folders must be new or empty. A failed run retains its partial outputs; select another output directory when retrying.

## Run stages separately

```bash
python paper2/prepare.py --output paper2/outputs/prepared01
python paper2/analyze.py --input paper2/outputs/prepared01/prepared.parquet --output paper2/outputs/rq1_01
python paper2/train.py --input paper2/outputs/prepared01/prepared.parquet --output paper2/outputs/models01 --device cuda:0
```

`train.py --feature-sets structural description copilot` skips the review-inclusive models. `--class-weight balanced` applies training-derived class weights to XGBoost and logistic regression; run it in a separate folder as a sensitivity analysis. `--exclude-zero` in preparation or the full pipeline excludes zero-minute merges. `--permutation-rows 0` disables permutation importance.

## Features and leakage controls

| Feature set | Inputs |
| --- | --- |
| `structural` | Files added/deleted/modified/touched, total/code/configuration lines modified, code flag, language count, first listed language |
| `description` | Structural inputs plus title/body word counts, title/body presence, and short-body flag |
| `copilot` | Description inputs plus `had_copilot_seat` |
| `review` | Description inputs plus final reviewer and review-comment counts |
| `review_copilot` | Review inputs plus Copilot access |

Identifiers, merge dates, merge month, the outcome, and `complexity_score` never enter the predictors. The existing complexity score includes review information, so using it would obscure the review ablation. `is_open_source` defines cohorts and evaluation subgroups, not a predictor. User IDs are retained only when present; no author history is invented. `detected_language` refers to text detection and is not used as a programming language.

`languages_list` accepts JSON/Python list strings or native Parquet arrays. Language count is the number of distinct entries. `first_language` is exactly the first listed entry and is not assumed to be dominant. A training-fitted encoder groups infrequent categories and handles unseen languages. Counts use log(1+x), training-only median imputation with missing indicators, and training-only scaling. Missing Copilot status remains unknown, with a missingness indicator, rather than becoming false.

Required inputs are the structural fields, review counts, the five description fields, the two PR identifiers, languages list, open-source status, Copilot seat flag, and merge duration. Optional metadata includes merge day/month and user ID. Preparation selects only these columns, reports invalid numeric cells and dropped rows, removes identical analytical duplicate records, and stops on conflicting analytical records with the same repository/PR keys. Negative, infinite, or invalid predictors become missing; invalid targets are removed. Invalid boolean values or malformed language containers stop the run.

### Feature timing matters

**The current schema does not establish that features were captured when a PR opened. Every run is labeled `unverified_final_snapshot` and must be described as retrospective.** Even a model without review counts may use final diff sizes or descriptions edited during review.

`merge-time` splits keep whole merge days together and order those days into train/validation/test. This tests performance across merge periods. It is not an opening-time evaluation: PRs can open before a split boundary and merge after it. A prospective experiment requires verified opening snapshots plus opening and completion timestamps, with training outcomes available before evaluation PRs open. The scripts do not reconstruct an exact creation time from a day-rounded merge date and duration.

The dataset has no first-review or approval timestamps. It supports analysis of total merge time and aggregate review activity, but cannot separate time awaiting review, time spent revising, and time awaiting merge.

## Training and evaluation

- Class labels are fast, medium, and slow. By default, the two cutoffs are the training split's 1/3 and 2/3 merge-duration quantiles. Fast includes the first cutoff; medium includes the second. The cutoffs stay fixed for validation and test. Tied cutoffs require explicit thresholds.
- All feature sets use the same PRs, partitions, labels, and sampled hyperparameter candidates. Missing features are imputed instead of dropping different rows for different models.
- Each XGBoost search samples up to 36 distinct configurations over depth, learning rate, minimum child weight, and row sampling. The default tries six. Validation log loss controls early stopping; validation macro F1 selects the configuration. The test set is never an `eval_set` and never controls model selection.
- The default objective is multiclass probability prediction, with histogram trees and `device=cuda:0`. `--trees` is a maximum; early stopping can select fewer trees.
- Each selected XGBoost model and fixed logistic baseline is evaluated on the held-out test set. The majority baseline is fitted on training class frequencies. Metrics include macro F1, balanced accuracy, accuracy, per-class recall, log loss, and multiclass ROC AUC where all classes are present.
- Copilot and open-source subgroup metrics report support counts. Unknown statuses form separate subgroups. Subgroup balanced accuracy averages recall over classes present in that subgroup; macro F1 always includes all three classes. Missing-class ROC AUC is left empty.
- Permutation importance uses a reproducible validation sample and measures macro F1 loss when a raw feature is shuffled. Gain importance is also saved. Neither importance measure establishes causation, and correlated features may share predictive information.
- Feature comparisons report paired test macro F1 differences, without significance claims. Repeat prespecified seeds and report variation rather than selecting the best test run.

RQ1 also reports a repository-stratified descriptive comparison: each eligible repository must contain at least five PRs in each known Copilot access group. It calculates each repository's difference in group median merge time, then reports the median of those differences and a repository bootstrap interval. This weights repositories equally and does not adjust for author, language, or change-size differences. One eligible repository produces no confidence interval.

## Outputs

`prepared/audit.json` records source SHA256, actual counts, scope, missingness, removals, and dependency versions. Training manifests record input hashes, seeds, thresholds, feature definitions, class counts, and limitations. `splits.csv.gz` preserves the exact membership of every PR. `validation_trials.csv` records tuning history. `selected_on_validation.json` identifies the model chosen using validation results.

Saved `*.joblib` bundles contain the fitted transformation, model, feature-set name, thresholds, and class names. XGBoost models are also saved as `*.ubj`. Load joblib files only from trusted sources. Test predictions include probabilities for each class, enabling further analysis. Tables are CSV/JSON and figures are PNG/PDF.

Data, outputs, row identifiers, predictions, and model binaries remain local and are ignored by git within this folder. No dataset is downloaded or published by the scripts. Inspect outputs before sharing them.

## Verification

```bash
python -m unittest discover -s paper2/tests -v
```

Tests use generated synthetic data. They check boolean parsing, unknown status, filtering, deduplication conflicts, repository separation, merge-day boundaries, class definitions, CSV and Parquet input, all five model feature sets on CPU, artifact reload predictions, and output overwrite protection. Actual GPU execution and real-data conclusions require your GPU machine and dataset.

Implementation references: [XGBoost GPU support](https://xgboost.readthedocs.io/en/stable/gpu/), [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html), and [scikit-learn group splits](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).
