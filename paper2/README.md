# Copilot Access and PR Merge Time Experiments

This directory contains the empirical study code in [pr-complexity-code](../README.md). The `paper2/` name is retained for compatibility with existing commands and outputs.

The scripts run independently of the complexity notebooks and the root requirements file. Source data is not included; row counts come from the preparation audit.

## Research questions

| Question | Analysis |
| --- | --- |
| **RQ1: How do PR size, language composition, review activity, and merge time differ between authors with and without Copilot access?** | Group summaries, language summaries, correlations, and within-repository merge time comparisons |
| **RQ2: Can machine learning predict PR merge time based on limited features?** | Compare XGBoost and linear models with simple baselines; examine feature ablations and generalization |

Description features, Copilot access, completed review activity, and text representations are supporting comparisons under RQ2. Some script docstrings and the separate research plan retain earlier RQ numbering; use the two questions above for this guide.

The contribution differs from the [complexity measurement paper](https://github.com/syd-kiwi/pr-complexity-paper/tree/main/paper): this study examines group differences and predictive information. It does not redefine the complexity score. Related work and proposed extensions are in [RESEARCH_PLAN.md](RESEARCH_PLAN.md).

Copilot status records access, not confirmed use or AI authorship. Results concern merged PRs and describe associations. Final snapshots do not establish that predictors were available when a PR opened.

## Start with descriptive comparisons

After installing the dependencies below, run from the repository root:

```bash
python paper2/prepare.py --input data/cleaned_public.csv --output paper2/outputs/prepared01 --scope all
python paper2/analyze.py --input paper2/outputs/prepared01/prepared.parquet --output paper2/outputs/rq1_01
```

This runs RQ1 without training models. Inspect `prepared01/audit.json`, then `rq1_01/by_had_copilot_seat.csv`, `by_first_language.csv`, `within_repository_effect.json`, and `copilot_merge_time_ecdf.png`. The first listed language is not necessarily dominant.

All examples need compatible input data and new or empty output directories. Scope is explicit here: `all` retains public and private rows; the scripts otherwise default to `open-source`.

## Choose a model workflow

| Entry point | Use | Evaluation |
| --- | --- | --- |
| `run_pipeline.py` | Preparation, RQ1 summaries, and tabular classification | Evaluates selected models on the internal test |
| `run_research.py` | Text comparisons, ordinal classification, regression, and diagnostics | Training and validation by default |
| `evaluate_saved.py` | Evaluate an already selected model on separate data | No retraining; checks development PR overlap |

For a new enhanced run that does not require an earlier split file:

```bash
python paper2/run_research.py --input data/cleaned_public.csv --output paper2/outputs/research01 --scope all --device cuda:0 --tasks multiclass --feature-sets description review --text-methods none tfidf
```

This prepares text and runs matched tabular/text comparisons. Choose the population, split, and duration boundaries before interpreting comparisons. For continued development on a previously evaluated dataset, use the split-reuse instructions below and preserve the original outputs.

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

Tests use generated synthetic data. They check boolean parsing, unknown status, filtering, deduplication conflicts, repository separation, merge-day boundaries, class definitions, CSV and Parquet input, all five original model feature sets on CPU, artifact reload predictions, and output overwrite protection. Enhanced tests cover text vocabulary leakage, SVD/encoder reload, monotonic ordinal probabilities, ordered errors, split reuse by PR key, known-repository timing, all enhanced tasks on CPU, fixed-model evaluation, and external PR overlap rejection. Optional sentence embedding API behavior is tested with a deterministic local fake, not a model download. Actual CUDA execution and pretrained model encoding require your GPU machine and optional dependencies.

Implementation references: [XGBoost GPU support](https://xgboost.readthedocs.io/en/stable/gpu/), [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html), and [scikit-learn group splits](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).

## Enhanced experiments after the first run

The enhanced runner adds actual title/body text, training/validation diagnostics, ordered classification, and duration regression. The original `run_pipeline.py` remains available. See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the questions, related work, and evaluation plan.

On RunPod, start from `/workspace/pr-complexity-code`:

```bash
git pull
python -m pip install -r paper2/requirements.txt
python -u paper2/run_research.py \
  --output paper2/outputs/research02 \
  --device cuda:0 --threads 16 \
  --split-file paper2/outputs/a100_run01/repository_seed42/splits.csv.gz \
  --thresholds 35 826 \
  --tasks multiclass \
  --feature-sets description review \
  --text-methods none tfidf
```

This is a staged development run: full training data, three hyperparameter trials per comparison, and training/validation scores. It holds the original partitions and duration definitions fixed. It does **not** score the original test again by default. The defaults still read `data/cleaned_public.csv`. `35` and `826` minutes reproduce the already-observed first experiment; they are not universal duration cutoffs. Omit `--thresholds` for training-only tertiles on a different dataset.

Preparation now retains raw title/body only with `--include-text`; `run_research.py` enables that automatically. It writes a new prepared file without changing the earlier run. `--prepared-input` can reuse that new text-bearing Parquet in later runs, avoiding another CSV load:

```bash
python -u paper2/run_research.py \
  --prepared-input paper2/outputs/research02/prepared/prepared.parquet \
  --output paper2/outputs/research03 \
  --device cuda:0 --threads 16 \
  --split-file paper2/outputs/a100_run01/repository_seed42/splits.csv.gz \
  --thresholds 35 826 \
  --tasks multiclass ordinal regression \
  --feature-sets description copilot review review_copilot \
  --text-methods none tfidf
```

The second command runs more comparisons. Choose the experiment matrix before evaluating it and report all attempted variants. Larger searches are available through `--trials` (maximum 36), `--trees`, and `--early-stopping`, but no performance increase is promised.

For optional pretrained embeddings, install the extra dependencies and run a separate comparison:

```bash
python -m pip install -r paper2/requirements-embeddings.txt
python -u paper2/run_research.py \
  --prepared-input paper2/outputs/research02/prepared/prepared.parquet \
  --output paper2/outputs/embeddings01 \
  --device cuda:0 --threads 16 \
  --split-file paper2/outputs/a100_run01/repository_seed42/splits.csv.gz \
  --thresholds 35 826 --tasks multiclass \
  --feature-sets description review \
  --text-methods embeddings --embedding-batch-size 128
```

The default frozen encoder is `sentence-transformers/all-MiniLM-L6-v2`; specify `--encoder` for another model or local path. Use `--encoder-revision` with an immutable model commit for an archival run. The first use may download model weights. Documents are capped at 12,000 characters; the encoder also truncates to its token limit, which is recorded in its metadata. No fine-tuning occurs. Embedding encoding and XGBoost use the GPU sequentially, releasing the encoder from GPU memory before tree training. TF-IDF, SVD, scalers, and linear baselines use CPU; `--threads` controls XGBoost threads, not every library's thread pool.

### Enhanced outputs and tasks

Under each run's `models/`, inspect:

- `metrics.csv`: training sample and validation scores for all attempted models; test/external metrics only when requested.
- `generalization_gaps.csv`: training sample versus validation macro F1. By default the training diagnostics use a seeded 50,000-row sample; `--diagnostic-rows 0` evaluates all training rows.
- `feature_comparisons.csv`: matched text and tabular feature comparisons within each task, split, and model type. Positive macro F1 differences favor the added features; negative absolute class error differences favor them.
- `validation_trials.csv`: task-specific training/validation scores, selected boosting rounds, actual devices, and sampled hyperparameters.
- `*_learning_curves.csv` and selected model `*_learning_curves.png`: training sample and validation loss across boosting rounds. Ordinal models have one curve pair per cumulative threshold.
- `*_validation_confusion.csv`, `*_validation_by_repository.csv`, and `*_validation_by_had_copilot_seat.csv`: diagnostics for each selected configuration and applicable linear baseline. Per-repository/group tables report sample counts and error aggregates, rather than significance or causal claims.
- `selected_on_validation.json`: a separate selected bundle for each task. Scores for different tasks are not directly ranked against one another.
- `*_stage*_gain_importance.csv`: selected XGBoost feature importance. Text components are latent dimensions, not interpretable topics; gain importance does not establish causation.
- `text_encoder_*.joblib`, saved model bundles, and native XGBoost stage models. Keep a bundle beside its referenced text encoder when moving the output directory.

`multiclass` selects by validation macro F1. `ordinal` fits two cumulative binary XGBoost models, corrects crossing cumulative probabilities, predicts the posterior median class, and selects by validation mean absolute class error. It can reduce extreme mistakes without improving exact accuracy. `regression` predicts log(1 + minutes), selects by validation log MAE, and reports errors in both log units and actual minutes. Its durations are also mapped back to the same three classes for secondary diagnostics. Negative log predictions are clipped to zero, with an upper numerical bound of 30 before exponentiation.

Early stopping uses validation loss, with a training diagnostic sample logged first. Train/validation scores are unweighted even when classification training uses `--class-weight balanced`. Ordinal stages receive the same original class-derived row weights; regression is unweighted. The same candidates and PR partitions are used across representations and feature sets. Linear baselines use fixed parameters; `--skip-linear` omits them when only a tree diagnostic is needed, while majority/median baselines remain.

TF-IDF fits unigram/bigram vocabulary and inverse-document frequencies on training text only, followed by training-fitted SVD and scaling. Defaults are a 30,000-term vocabulary and up to 128 components. Transformations are chunked and reused across all feature sets/tasks in that run. Frozen embeddings are also scaled on training rows only. The workflow remains in host RAM; it is not distributed or out-of-core.

### Generalization and final evaluation

`experiment.py --split known-repository` uses later merge periods from repositories represented in training. Omit `--split-file` for this protocol. Repositories first seen in later periods are excluded from evaluation, with row exclusions recorded. This protocol differs from holding out entire repositories and must be reported separately. Neither protocol establishes opening-time prediction from final snapshots. To reuse a prepared table directly:

```bash
python paper2/experiment.py \
  --input paper2/outputs/research02/prepared/prepared.parquet \
  --output paper2/outputs/known_repository01 \
  --device cuda:0 --threads 16 --split known-repository \
  --thresholds 35 826 --tasks multiclass \
  --feature-sets description review --text-methods none tfidf
```

Use `--evaluate-test` only for an explicitly exploratory revisit of the internal test. Re-splitting data already used during development does not make a genuinely untouched final test. With separately collected, independently held-out data, prepare it with `--include-text` and evaluate an already fixed bundle without retraining:

```bash
python paper2/prepare.py --input data/independent_prs.csv --include-text --output paper2/outputs/independent_prepared
python paper2/evaluate_saved.py \
  --model paper2/outputs/research02/models/tfidf_review_multiclass_xgboost.joblib \
  --input paper2/outputs/independent_prepared/prepared.parquet \
  --development-input paper2/outputs/research02/prepared/prepared.parquet \
  --output paper2/outputs/independent_evaluation
```

Supply the model actually selected on validation. The evaluator rejects PR keys found in development data and uses the saved class boundaries. Independence and snapshot timing still require researcher verification. Opening-time data collection and extra history/CI features are described in the research plan; the scripts do not invent those fields.

