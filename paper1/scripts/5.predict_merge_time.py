#!/usr/bin/env python3
"""Compare PR merge-time regressors using cleaned_public.csv (Python 3.10+).

INSTALL:
  python -m pip install numpy pandas scipy scikit-learn matplotlib joblib 'xgboost>=2' sentence-transformers torch rtdl-revisiting-models==0.0.2
RUN (four existing models on all feature sets + structural FT-Transformer):
  python 5.predict_merge_time.py --input cleaned_public.csv --output merge_time_results
START HERE (compare FT-Transformer with existing structural models):
  python 5.predict_merge_time.py --input cleaned_public.csv --output structural_results --feature-sets structural --models ridge rf xgboost mlp ft_transformer
QUICK FT CHECK (CPU; no text encoder download):
  python 5.predict_merge_time.py --input cleaned_public.csv --output ft_check --feature-sets structural --models ft_transformer --max-rows 10000 --trials 1 --epochs 3 --device cpu
USE A GPU FOR FT-TRANSFORMER:
  python 5.predict_merge_time.py --input cleaned_public.csv --output ft_results --feature-sets structural --models ft_transformer --device cuda
OFFLINE TEXT OPTION (no pretrained model download):
  python 5.predict_merge_time.py --input cleaned_public.csv --text-method tfidf
QUICK CHECK:
  python 5.predict_merge_time.py --input cleaned_public.csv --max-rows 10000 --trials 1 --epochs 10 --text-method tfidf
STRUCTURAL ONLY:
  python 5.predict_merge_time.py --input cleaned_public.csv --feature-sets structural --models ridge rf
PREDICT WITH A SAVED MODEL (same raw-column schema, target not required):
  python 5.predict_merge_time.py --input new_prs.csv --predict-from merge_time_results/structural_xgboost.joblib --output new_predictions

Method:
* Deduplicate by repository_id + pull_request_id; reject invalid/negative targets.
* Keep zero-minute PRs unless --exclude-zero is specified. No tail trimming.
* Disjoint repository groups: about 70% train / 15% validation / 15% test.
  Percentages refer to repositories, so row fractions can differ substantially.
* Fit preprocessing on training rows only. Tune on validation log-MAE only.
* Train on log1p(minutes_to_merge); test once with the selected configuration.
* All experiments use the same rows and split. Review-inclusive models are
  retrospective: final review counts are not opening-time predictors.
* Structural/text features may also reflect final state. This is not proof of
  prospective opening-time performance. Merged-only data cannot predict acceptance.
* The MLP uses sklearn on CPU (three hidden layers). FT-Transformer uses the
  authors' rtdl-revisiting-models package and trains from scratch on structural
  features only. Its learned numeric feature tokens are not title/body tokens.
  It uses minibatch log-target MSE training and validation log-MAE early stopping.
* --device controls FT-Transformer and pretrained text encoding (cpu, cuda, or
  automatic if omitted). XGBoost also uses CUDA when --device cuda is explicit;
  otherwise XGBoost uses CPU. Ridge/Random Forest/sklearn MLP use CPU.
  Text encoding is frozen, not fine-tuned. This is not a DeepPull replication.
  Default text encoder truncates long descriptions.
* FT-Transformer settings here are a small experiment-specific search, not a
  reproduction of the original paper's full tuning protocol or reported results.
* Models and preprocessing are saved together. Load joblib files only if trusted.
* Full CSV is read into RAM; --max-rows performs a uniform streaming row sample
  before cleaning. It is a development option, not the full-dataset experiment.

Outputs: metrics.csv, duration_metrics.csv, predictions.csv.gz, splits.csv.gz,
validation_trials.csv, model_comparison.png, audit.json, run_config.json,
feature_schema.json, selected_on_validation.json, and fitted *.joblib bundles.
FT runs also write ft_training_history.csv with per-epoch train/validation error.
Progress: START/DONE labels identify stages; WORKING messages repeat every
--progress-interval seconds (default 30) during long calls. They report elapsed
time, not a completion percentage. Column row counts and FT minibatch counts
report actual progress within that operation. This version does not resume runs;
choose a fresh output directory when restarting after an interruption.
The FT joblib bundle stores architecture and CPU weight arrays, not a pickled
script-defined model class; use this script's --predict-from to reconstruct it.
The selected model is chosen using validation error, never test results.
Pretrained embeddings require internet on first use or a local --encoder path.

Related work:
  Maddila et al., FSE 2019: https://doi.org/10.1145/3338906.3340457
  Nudge, TOSEM 2022: https://arxiv.org/abs/2011.12468
  DeepPull, 2024: https://doi.org/10.1007/978-3-031-61753-9_6
  Pansuriya et al., 2026 preprint: https://arxiv.org/abs/2607.12057
  Gorishniy et al., NeurIPS 2021 (FT-Transformer): https://arxiv.org/abs/2106.11959
  Official implementation: https://github.com/yandex-research/rtdl-revisiting-models
"""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.metadata
import json
import logging
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Input schema: column names expected in the source CSV.
# IDs identify PRs and define repository groups; they are not model features.
# TARGET is the value to predict, measured in minutes.
# -----------------------------------------------------------------------------
LOG = logging.getLogger("merge_time")
PROGRESS_INTERVAL = 30.0


@contextmanager
def progress_step(label):
    """Announce a step and print elapsed time while it is still running.

    Some libraries do not expose a percentage during fit(). A small background
    thread reports that the call has not returned yet; it is NOT a percentage
    or a promise about completion time. This does not parallelize training.
    The thread is always stopped, including when a step raises an exception.
    """
    started = time.monotonic()
    stopped = threading.Event()
    interval = PROGRESS_INTERVAL

    def heartbeat():
        while not stopped.wait(interval):
            LOG.info(
                "WORKING | %s | elapsed %.1f s (completion unknown)",
                label,
                time.monotonic() - started,
            )

    LOG.info("START | %s", label)
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    except BaseException:
        LOG.error("STOPPED | %s | after %.1f s", label, time.monotonic() - started)
        raise
    else:
        LOG.info("DONE | %s | %.1f s", label, time.monotonic() - started)
    finally:
        stopped.set()
        worker.join(timeout=1)


def tracked(label):
    """Wrap a long function with progress messages without changing its result."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with progress_step(label):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def progress_map(series, function, label, chunk_size=50000):
    """Apply a per-row conversion in chunks, reporting real completed row counts.

    The same function and row order are retained. The percentage and ETA refer
    ONLY to this column operation, never to the entire experiment.
    """
    parts = []
    started = time.monotonic()
    with progress_step(label + " (CPU)"):
        for start in range(0, len(series), chunk_size):
            part = series.iloc[start : start + chunk_size].map(function)
            parts.append(part)
            done = min(start + chunk_size, len(series))
            elapsed = time.monotonic() - started
            eta = elapsed * (len(series) - done) / done
            LOG.info(
                "ROWS | %s | %s/%s (%.1f%%) | column ETA %.1f s",
                label,
                f"{done:,}",
                f"{len(series):,}",
                100 * done / len(series),
                eta,
            )
    return pd.concat(parts) if parts else series.map(function)


TARGET = "minutes_to_merge"
NUMERIC = [
    "files_added",
    "files_deleted",
    "files_modified",
    "files_touched",
    "lines_modified",
    "code_lines_modified",
    "config_lines_modified",
    "is_code_pr",
]
REVIEW = ["num_reviewers", "num_review_comments"]
EXTRA = ["languages_list", "lines_added_by_language", "lines_deleted_by_language"]
RAW_TEXT = ["title", "body"]
IDS = ["repository_id", "pull_request_id"]


# =============================================================================
# CSV parsing and numeric/text feature construction
# =============================================================================
def dump_json(path, value):
    """Write settings or audit information as readable UTF-8 JSON."""
    Path(path).write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )


def parse_container(value, expected):
    """Read a list/dictionary stored as text inside a CSV cell.

    Accept both JSON and Python literal syntax. Invalid or missing values become
    an empty container of the requested type; no arbitrary code is executed."""
    if isinstance(value, expected):
        return value
    if not isinstance(value, str) or not value.strip():
        return expected()
    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(value)
            if isinstance(result, expected):
                return result
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            pass
    return expected()


def numeric(series):
    # Accommodate CSV boolean strings without treating missing values as False.
    """Convert a column to floats, accepting true/false strings as 1/0.

    Invalid entries and infinity become NaN, to be imputed later using training
    data. Missing values are not automatically treated as zero."""
    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .replace({"true": "1", "false": "0"})
    )
    return (
        pd.to_numeric(normalized, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )


@tracked("Build structural features (CPU; GPU is not used during preparation)")
def structural_features(df):
    """Build numeric features describing the size and spread of a PR.

    The eight NUMERIC columns are required. If present, language lists add a
    unique-language count, and per-language dictionaries add line totals."""
    missing = [c for c in NUMERIC if c not in df]
    if missing:
        raise ValueError(f"Missing structural columns: {missing}")
    columns = {}
    for number, column in enumerate(NUMERIC, 1):
        with progress_step(f"Numeric column {number}/{len(NUMERIC)}: {column}"):
            columns[column] = numeric(df[column])
    out = pd.DataFrame(columns, index=df.index)
    if "languages_list" in df:
        languages = progress_map(
            df["languages_list"],
            lambda v: parse_container(v, list),
            "Parse languages_list",
        )
        out["language_count"] = progress_map(
            languages, lambda v: len(set(map(str, v))), "Count languages"
        )
    for source in ["lines_added_by_language", "lines_deleted_by_language"]:
        if source in df:
            parsed = progress_map(
                df[source], lambda v: parse_container(v, dict), f"Parse {source}"
            )

            def total(mapping):
                # Empty dictionaries sum to zero; avoid per-row pandas work.
                if not mapping:
                    return 0.0
                values = pd.to_numeric(
                    pd.Series(list(mapping.values()), dtype=object), errors="coerce"
                )
                return float(values.replace([np.inf, -np.inf], np.nan).sum())

            out[source + "_total"] = progress_map(parsed, total, f"Sum {source}")
    return out


def text_numeric(df):
    """Derive title/body word counts and flags indicating nonempty text."""
    title = df["title"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    # Recompute lengths/flags to keep their definitions consistent on new inputs.
    return pd.DataFrame(
        {
            "title_words": title.str.split().str.len(),
            "body_words": body.str.split().str.len(),
            "has_title": title.str.strip().ne("").astype(int),
            "has_body": body.str.strip().ne("").astype(int),
        },
        index=df.index,
    )


def text_input(df):
    """Join each PR title and body into one document for text encoding."""
    return (
        df["title"].fillna("").astype(str) + "\n\n" + df["body"].fillna("").astype(str)
    ).tolist()


# =============================================================================
# Data loading, cleaning, and repository splits
# =============================================================================
@tracked("Read input CSV (CPU/disk)")
def load_data(args):
    """Read relevant CSV columns and retain each row's original position.

    In training mode, --max-rows uniformly samples raw rows while scanning the
    whole file in chunks. Sampling happens before cleaning, so fewer rows may
    remain afterward. Without that option, all selected columns enter RAM.
    Prediction mode reads every row and does not require the target."""
    path = Path(args.input)
    LOG.info("Input: %s", path.resolve())
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {path.resolve()}")
    available = pd.read_csv(path, nrows=0).columns.tolist()
    needs_text = args.predict_from or any(
        s in args.feature_sets for s in ("text", "review")
    )
    wanted = (
        IDS + NUMERIC + EXTRA + [TARGET] + (RAW_TEXT if needs_text else []) + REVIEW
    )
    if args.include_complexity_baseline or args.predict_from:
        wanted += ["complexity_score"]
    columns = [c for c in wanted if c in available]
    dtypes = {c: "string" for c in IDS if c in columns}
    kwargs = dict(usecols=columns, dtype=dtypes, low_memory=False)
    if args.max_rows and not args.predict_from:
        # Random-key reservoir: uniform sample without loading all text into RAM.
        rng = np.random.default_rng(args.seed)
        sample = None
        seen = 0
        for chunk in pd.read_csv(path, chunksize=50000, **kwargs):
            chunk["__source_row"] = np.arange(seen, seen + len(chunk))
            seen += len(chunk)
            LOG.info("CSV sample scan: %s source rows read", f"{seen:,}")
            chunk["__sample_key"] = rng.random(len(chunk))
            sample = (
                chunk
                if sample is None
                else pd.concat([sample, chunk], ignore_index=True)
            )
            sample = sample.nsmallest(args.max_rows, "__sample_key")
        if sample is None:
            raise ValueError("CSV has no data rows")
        df = sample.drop(columns="__sample_key").sort_values("__source_row")
    else:
        df = pd.read_csv(path, **kwargs)
        df["__source_row"] = np.arange(len(df))
        seen = len(df)
    LOG.info(
        "CSV loaded: %s rows scanned; %s rows selected; %s columns",
        f"{seen:,}",
        f"{len(df):,}",
        len(columns),
    )
    return df.reset_index(drop=True), seen


@tracked("Validate identifiers, duplicates, and durations (CPU)")
def clean_training(df, exclude_zero):
    """Validate training data and return cleaned rows plus an audit summary.

    Require identifiers, structural columns, and minutes_to_merge. Drop missing
    identifiers, identical duplicate PRs, and invalid/negative durations. Raise
    an error for conflicting duplicate snapshots. Keep zero-minute PRs unless
    requested otherwise. Text/review requirements are checked later."""
    required = IDS + NUMERIC + [TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    audit = {"loaded_rows": len(df)}
    valid_ids = df[IDS].notna().all(axis=1) & df[IDS].apply(
        lambda s: s.str.strip().ne("")
    ).all(axis=1)
    audit["invalid_identifier_rows"] = int((~valid_ids).sum())
    df = df.loc[valid_ids].copy()
    duplicated = df.duplicated(IDS, keep="first")
    audit["duplicate_rows_removed"] = int(duplicated.sum())
    # Do not silently choose between inconsistent copies of the same PR.
    duplicate_rows = df.loc[df.duplicated(IDS, keep=False)]
    if len(duplicate_rows):
        check_cols = [c for c in df.columns if c != "__source_row"]
        if duplicate_rows.drop_duplicates(check_cols).duplicated(IDS).any():
            raise ValueError(
                "Conflicting rows share a PR identifier. Resolve duplicate snapshots before training."
            )
    df = df.loc[~duplicated].copy()
    y = numeric(df[TARGET])
    valid = y.notna() & y.ge(0)
    audit["invalid_or_negative_target_rows"] = int((~valid).sum())
    audit["zero_minute_rows"] = int(y.eq(0).sum())
    if exclude_zero:
        valid &= y.gt(0)
    df = df.loc[valid].copy()
    df[TARGET] = y.loc[valid]
    df = df.reset_index(drop=True)
    audit["retained_rows"] = len(df)
    audit["retained_repositories"] = int(df.repository_id.nunique())
    if len(df) < 30 or df.repository_id.nunique() < 6:
        raise ValueError(
            "Need at least 30 valid PRs and 6 repositories for train/validation/test splitting."
        )
    return df, audit


@tracked("Split repositories into training/validation/test (CPU)")
def make_splits(df, args):
    """Separate repositories into training, validation, and test groups.

    Defaults allocate roughly 70/15/15 percent of repositories. All PRs from a
    repository stay together, so percentages of PR rows can differ. Validation
    selects settings; test rows evaluate the selected settings."""
    groups = df.repository_id.to_numpy()
    trainval, test = next(
        GroupShuffleSplit(
            n_splits=1, test_size=args.test_fraction, random_state=args.seed
        ).split(df, groups=groups)
    )
    relative = args.validation_fraction / (1 - args.test_fraction)
    train_local, val_local = next(
        GroupShuffleSplit(
            n_splits=1, test_size=relative, random_state=args.seed + 1
        ).split(trainval, groups=groups[trainval])
    )
    splits = {
        "train": trainval[train_local],
        "validation": trainval[val_local],
        "test": test,
    }
    sets = [set(groups[v]) for v in splits.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    if any(len(v) < 2 for v in splits.values()):
        raise ValueError(
            "A split has fewer than two PRs; use more data or adjust split fractions."
        )
    return splits


# =============================================================================
# Text representations and cumulative feature sets
# =============================================================================
@tracked("Encode PR text")
def fit_text(df, train, args):
    """Create a text matrix and reusable text-processing state.

    embeddings: use a pretrained SentenceTransformer without fine-tuning it.
    tfidf: learn a word/bigram vocabulary and SVD reduction on training rows
    only, then transform all rows using those same fitted objects.

    Return one vector per PR and the state needed to encode future PRs."""
    texts = text_input(df)
    # The default encoder turns text into fixed vectors; it is not a merge-time
    # regressor. Only this encoding step can use --device cuda.
    if args.text_method == "embeddings":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Install sentence-transformers, or use --text-method tfidf."
            ) from e
        encoder = SentenceTransformer(args.encoder, device=args.device)
        LOG.info(
            "Encoding %s PR descriptions; long inputs are truncated at encoder token limit %s",
            len(texts),
            encoder.max_seq_length,
        )
        # All splits can be encoded together because this pretrained encoder
        # is frozen: it does not learn from these PRs or their duration labels.
        encoded = encoder.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # Bundle the actual encoder locally so inference need not redownload it.
        destination = Path(args.output).resolve() / "text_encoder"
        encoder.save(str(destination))
        return np.asarray(encoded, dtype=np.float32), {
            "method": "embeddings",
            "encoder_dir": "text_encoder",
        }
    # Offline alternative: TF-IDF weights words/bigrams, then SVD compresses
    # the sparse representation to a smaller dense matrix.
    vectorizer = TfidfVectorizer(
        max_features=args.tfidf_features,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    training = vectorizer.fit_transform([texts[i] for i in train])
    if training.shape[1] < 2:
        raise ValueError(
            "Not enough distinct text tokens for TF-IDF. Use structural-only experiments."
        )
    components = min(args.text_dimensions, training.shape[1] - 1, training.shape[0] - 1)
    svd = TruncatedSVD(n_components=components, random_state=args.seed).fit(training)
    encoded = svd.transform(vectorizer.transform(texts)).astype(np.float32)
    return encoded, {"method": "tfidf", "vectorizer": vectorizer, "svd": svd}


def transform_text(df, state, bundle_dir, args):
    """Encode new PR text using the saved vocabulary/SVD or local encoder."""
    texts = text_input(df)
    if state["method"] == "tfidf":
        return (
            state["svd"]
            .transform(state["vectorizer"].transform(texts))
            .astype(np.float32)
        )
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        str(bundle_dir / state["encoder_dir"]), device=args.device
    )
    return encoder.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def feature_frame(df, name):
    """Build one of the cumulative numeric feature sets.

    structural = size/file features plus available language summaries.
    text = structural plus title/body counts and flags.
    review = text plus reviewer and review-comment counts.
    Text vectors are appended separately in main(), not in this function."""
    frame = structural_features(df)
    if name in ("text", "review"):
        frame = pd.concat([frame, text_numeric(df)], axis=1)
    if name == "review":
        missing = [c for c in REVIEW if c not in df]
        if missing:
            raise ValueError(f"Review experiment needs columns: {missing}")
        for c in REVIEW:
            frame[c] = numeric(df[c])
    return frame


# =============================================================================
# FT-Transformer: a transformer for numeric feature columns
# =============================================================================
def ft_dependencies():
    """Import optional deep learning packages only when FT is requested.

    Other models and --help still work without PyTorch or the RTDL package.
    We use the authors' implementation rather than approximate their model.
    """
    try:
        import torch
        from rtdl_revisiting_models import FTTransformer
    except ImportError as error:
        raise ImportError(
            "FT-Transformer requires PyTorch and rtdl-revisiting-models. "
            "Install torch for your CPU/CUDA environment, then run: "
            "python -m pip install rtdl-revisiting-models==0.0.2. "
            "Or omit ft_transformer from --models."
        ) from error
    return torch, FTTransformer


def ft_device(requested):
    """Choose CPU/GPU explicitly, or prefer CUDA when no device was supplied."""
    torch, _ = ft_dependencies()
    device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable. Use --device cpu.")
    return device


class FTTransformerRegressor:
    """Small adapter connecting the official PyTorch model to this experiment.

    Think of a PR as a table row, with columns such as files_touched and
    code_lines_modified. RTDL turns EACH standardized numeric column into a
    learned vector (a feature token), adds a learned summary token, then uses
    attention blocks to combine information across columns. A regression head
    turns the summary into one predicted log1p merge duration.

    This is tabular attention, not attention over words or a sequence of PRs.
    Each PR is predicted independently. IDs and target labels are never tokens.
    The existing train-only imputer/scaler supplies the numeric inputs.
    """

    def __init__(self, n_features, parameters, args):
        """Construct an untrained model; learned weights come from fit/load."""
        torch, transformer = ft_dependencies()
        self.device = ft_device(args.device)
        self.batch_size = args.batch_size
        self.n_features = int(n_features)
        self.parameters = dict(parameters)
        self.best_epoch = None
        self.history = []

        # Seed initialization and dropout as well as minibatch shuffling.
        # Exact results can still differ across hardware/package versions.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        if args.jobs > 0:
            torch.set_num_threads(args.jobs)

        self.network = transformer(
            n_cont_features=self.n_features,
            cat_cardinalities=[],  # This script uses numeric features only.
            d_out=1,  # One regression output: log1p(minutes_to_merge).
            n_blocks=parameters["n_blocks"],
            d_block=parameters["d_block"],  # Vector width of each feature token.
            attention_n_heads=parameters["attention_n_heads"],
            attention_dropout=parameters["dropout"],
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=parameters["dropout"],
            residual_dropout=0.0,
        ).to(self.device)

    def predict(self, X):
        """Return log-duration predictions in minibatches to limit GPU memory."""
        torch, _ = ft_dependencies()
        self.network.eval()  # Disable dropout during evaluation/inference.
        output = []
        with torch.inference_mode():  # No gradients needed for predictions.
            for start in range(0, len(X), self.batch_size):
                batch = torch.as_tensor(
                    np.asarray(X[start : start + self.batch_size], dtype=np.float32),
                    device=self.device,
                )
                output.append(self.network(batch, None).reshape(-1).cpu().numpy())
        return np.concatenate(output) if output else np.empty(0, dtype=np.float32)

    def fit(self, Xtrain, ytrain, Xval, yval, args):
        """Train on training PRs; retain the epoch with best validation log-MAE.

        Training minimizes squared error in log-duration, like a standard
        regression loss. Selection uses the SAME clipped validation log-MAE
        as the existing models. Test rows never enter this method.
        """
        torch, _ = ft_dependencies()
        from torch.utils.data import DataLoader, TensorDataset

        # Keep the full training tensors on CPU. Only the current minibatch
        # moves to GPU, so dataset size does not determine GPU memory usage.
        dataset = TensorDataset(
            torch.from_numpy(np.asarray(Xtrain, dtype=np.float32)),
            torch.from_numpy(np.asarray(ytrain, dtype=np.float32)),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=0,
        )
        # RTDL excludes biases/normalizations/feature embeddings from weight
        # decay through its official parameter-group helper.
        optimizer = torch.optim.AdamW(
            self.network.make_parameter_groups(),
            lr=self.parameters["learning_rate"],
            weight_decay=self.parameters["weight_decay"],
        )
        best_state, best_loss, stale = None, np.inf, 0
        LOG.info(
            "FT-Transformer: %s numeric tokens + summary token, %s blocks, "
            "%s attention heads, token width %s, device %s",
            self.n_features,
            self.parameters["n_blocks"],
            self.parameters["attention_n_heads"],
            self.parameters["d_block"],
            self.device,
        )
        for epoch in range(1, args.epochs + 1):
            self.network.train()  # Enable dropout for training.
            squared_error = 0.0
            epoch_started = last_report = time.monotonic()
            LOG.info(
                "FT epoch %s/%s: training %s minibatches on %s",
                epoch,
                args.epochs,
                f"{len(loader):,}",
                self.device,
            )
            for batch_number, (batch, target) in enumerate(loader, 1):
                batch, target = batch.to(self.device), target.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.network(batch, None).reshape(-1)
                loss = torch.nn.functional.mse_loss(prediction, target)
                if not torch.isfinite(loss).item():
                    raise ValueError("FT-Transformer produced nonfinite training loss")
                loss.backward()  # Compute how each weight affects the loss.
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()  # Update the weights using those gradients.
                squared_error += loss.item() * len(batch)
                now = time.monotonic()
                if now - last_report >= PROGRESS_INTERVAL or batch_number == len(
                    loader
                ):
                    elapsed = now - epoch_started
                    eta = elapsed * (len(loader) - batch_number) / batch_number
                    LOG.info(
                        "FT BATCHES | epoch %s | %s/%s (%.1f%%) | training ETA %.1f s",
                        epoch,
                        f"{batch_number:,}",
                        f"{len(loader):,}",
                        100 * batch_number / len(loader),
                        eta,
                    )
                    last_report = now

            with progress_step(
                f"FT epoch {epoch}: validate {len(Xval):,} PRs on {self.device}"
            ):
                validation_loss = float(
                    mean_absolute_error(yval, np.maximum(0, self.predict(Xval)))
                )
            if not np.isfinite(validation_loss):
                raise ValueError("FT-Transformer produced nonfinite validation loss")
            self.history.append(
                {
                    "epoch": epoch,
                    "train_log_mse": squared_error / len(dataset),
                    "validation_log_mae": validation_loss,
                }
            )
            LOG.info(
                "FT epoch %s: train log-MSE %.4f; validation log-MAE %.4f",
                epoch,
                squared_error / len(dataset),
                validation_loss,
            )
            if validation_loss < best_loss - 1e-5:
                best_loss, self.best_epoch, stale = validation_loss, epoch, 0
                # CPU copies keep the best epoch without retaining a second
                # model or its gradients in GPU memory.
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.network.state_dict().items()
                }
            else:
                stale += 1
            if stale >= args.patience:
                LOG.info("FT early stopping; restoring best epoch %s", self.best_epoch)
                break
        self.network.load_state_dict(best_state)
        self.network.eval()
        return self

    def export_state(self):
        """Save plain configuration/arrays, avoiding a script-specific pickle.

        The surrounding joblib file also stores the imputer/scaler. No GPU or
        script class instance is embedded in the saved weights. A future run
        rebuilds the official model and fills it with these learned weights.
        """
        return {
            "n_features": self.n_features,
            "parameters": self.parameters,
            "best_epoch": self.best_epoch,
            "weights": {
                key: value.detach().cpu().numpy().copy()
                for key, value in self.network.state_dict().items()
            },
        }

    @classmethod
    def from_state(cls, state, args):
        """Restore a fitted model on the device chosen for this prediction run."""
        torch, _ = ft_dependencies()
        model = cls(state["n_features"], state["parameters"], args)
        model.network.load_state_dict(
            {key: torch.from_numpy(value) for key, value in state["weights"].items()}
        )
        model.best_epoch = state["best_epoch"]
        model.network.eval()
        return model


# =============================================================================
# Regressors, parameter candidates, and training
# =============================================================================
def candidates(name, args):
    """Return up to --trials predefined parameter settings for a model.

    This is a small fixed search, not an exhaustive grid or random search.
    Ridge changes regularization; forests and XGBoost change tree settings;
    the MLP changes layer sizes, regularization, and learning rate.
    FT changes token width, number of attention blocks, and learning rate.
    These are compact candidate settings, not published paper configurations.
    """
    if name == "ft_transformer":
        return [
            {
                "n_blocks": blocks,
                "d_block": width,
                "attention_n_heads": 8,
                "dropout": 0.1,
                "learning_rate": learning_rate,
                "weight_decay": 1e-5,
            }
            for blocks, width, learning_rate in (
                (2, 64, 1e-3),
                (3, 96, 5e-4),
                (3, 128, 3e-4),
                (2, 96, 1e-4),
            )
        ][: args.trials]
    if name == "ridge":
        return [{"alpha": a} for a in (1.0, 10.0, 100.0, 1000.0)][: args.trials]
    if name == "rf":
        return [
            {
                "n_estimators": 200,
                "max_depth": d,
                "min_samples_leaf": leaf,
                "max_features": 0.8,
                "n_jobs": args.jobs,
                "random_state": args.seed,
                "verbose": 1,  # sklearn prints tree-completion counts.
            }
            for d, leaf in ((16, 5), (24, 10), (None, 5), (12, 20))
        ][: args.trials]
    if name == "xgboost":
        return [
            {
                "n_estimators": n,
                "max_depth": d,
                "learning_rate": lr,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 5.0,
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "device": args.device or "cpu",
                "n_jobs": args.jobs,
                "random_state": args.seed,
            }
            for n, d, lr in (
                (300, 4, 0.05),
                (500, 6, 0.05),
                (500, 8, 0.03),
                (300, 3, 0.08),
            )
        ][: args.trials]
    return [
        {
            "hidden_layer_sizes": layers,
            "alpha": alpha,
            "learning_rate_init": lr,
            "random_state": args.seed,
            "batch_size": args.batch_size,
            "early_stopping": False,
            "shuffle": True,
        }
        for layers, alpha, lr in (
            ((256, 128, 64), 0.01, 0.001),
            ((128, 64, 32), 0.1, 0.0005),
            ((256, 128, 64), 1.0, 0.0005),
            ((128, 64, 32), 0.01, 0.001),
        )
    ][: args.trials]


@tracked("Fit model trial (see model/device line above for CPU or GPU)")
def train_candidate(name, params, Xtrain, ytrain, Xval, yval, args):
    """Train one configuration against the log-transformed target.

    Ridge, Random Forest, and XGBoost fit once. The MLP trains one epoch per
    partial_fit call, checks external validation MAE, and keeps a copy of its
    best epoch. Stop after --patience epochs without sufficient improvement.
    FT-Transformer also uses validation early stopping, in its fit() method.
    Return the model and its best epoch (None for models fitted once)."""
    if name == "ft_transformer":
        model = FTTransformerRegressor(Xtrain.shape[1], params, args)
        model.fit(Xtrain, ytrain, Xval, yval, args)
        return model, model.best_epoch
    if name == "ridge":
        model = Ridge(**params)
    elif name == "rf":
        model = RandomForestRegressor(**params)
    elif name == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(**params)
    else:
        model = MLPRegressor(**params)
        best, best_loss, stale, best_epoch = None, np.inf, 0, 0
        for epoch in range(args.epochs):
            model.partial_fit(Xtrain, ytrain)
            loss = mean_absolute_error(yval, np.maximum(0, model.predict(Xval)))
            LOG.info(
                "MLP epoch %s/%s (CPU): validation log-MAE %.4f",
                epoch + 1,
                args.epochs,
                loss,
            )
            if not np.isfinite(loss):
                raise ValueError("MLP produced nonfinite validation error")
            if loss < best_loss - 1e-5:
                best, best_loss, stale, best_epoch = (
                    copy.deepcopy(model),
                    loss,
                    0,
                    epoch + 1,
                )
            else:
                stale += 1
            if stale >= args.patience:
                LOG.info("MLP early stopping; using epoch %s", best_epoch)
                break
        return best, best_epoch
    if name == "xgboost":
        # Evaluation logs expose boosting progress. They do not change tree
        # count or select a model: validation selection is still done in main.
        model.set_params(eval_metric="mae")
        model.fit(Xtrain, ytrain, eval_set=[(Xval, yval)], verbose=50)
        actual_device = json.loads(model.get_booster().save_config())["learner"][
            "generic_param"
        ]["device"]
        LOG.info("XGBoost actual device: %s", actual_device)
        if str(params["device"]).startswith("cuda") and not actual_device.startswith(
            "cuda"
        ):
            raise RuntimeError(
                "XGBoost fell back to CPU despite --device cuda. Check your GPU/driver and CUDA-enabled XGBoost installation, or choose --device cpu."
            )
    else:
        model.fit(Xtrain, ytrain)
    return model, None


# =============================================================================
# Prediction conversion, evaluation, and plots
# =============================================================================
def minutes_from_log(values):
    """Convert predicted log1p durations back to minutes using expm1.

    Clip log predictions to [0, 50] to prevent negative durations and limit
    extreme exponentiation. Reject NaN/infinite model predictions."""
    if not np.isfinite(values).all():
        raise ValueError("Model produced nonfinite predictions")
    return np.expm1(np.clip(values, 0, 50))


@tracked("Calculate error metrics and repository averages (CPU)")
def metrics(y, pred, groups):
    """Measure test errors in minutes, log error, and prediction agreement.

    MAE = average absolute error; median AE = typical absolute error;
    RMSE penalizes large errors more; RMSLE measures log-scale error;
    R2 compares squared error with a mean predictor and can be negative;
    Spearman measures rank association. Macro repository MAE averages each
    repository's MAE equally, instead of weighting large repositories more."""
    errors = np.abs(y - pred)
    correlation = (
        pd.Series(y).corr(pd.Series(pred), method="spearman")
        if np.unique(pred).size > 1 and np.unique(y).size > 1
        else np.nan
    )
    return {
        "n": len(y),
        "mae_minutes": float(mean_absolute_error(y, pred)),
        "median_ae_minutes": float(np.median(errors)),
        "rmse_minutes": float(np.sqrt(mean_squared_error(y, pred))),
        "rmsle": float(np.sqrt(mean_squared_error(np.log1p(y), np.log1p(pred)))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
        "spearman": float(correlation),
        "macro_repository_mae_minutes": float(
            pd.DataFrame({"repo": groups, "error": errors})
            .groupby("repo")
            .error.mean()
            .mean()
        ),
    }


def plot_comparison(table, output):
    """Save a horizontal chart of test MAE for every experiment.

    This plot describes test performance; it does not select the saved winner.
    Model selection uses validation log-MAE, a different error measure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = table.sort_values("mae_minutes", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, len(table) * 0.35)))
    ax.barh(table["experiment"], table.mae_minutes, color="#246b8e")
    ax.set_xlabel("Test MAE (minutes; lower is better)")
    ax.set_title("Merge-time prediction: repositories held out from training")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


# =============================================================================
# Inference with a saved model bundle
# =============================================================================
def predict_saved(args):
    """Load a trusted model bundle and predict durations for a new CSV.

    Reuse the saved feature order, imputation, scaling, and text encoder.
    Write predictions.csv with available identifiers and predicted minutes.
    No training, target labels, deduplication, or evaluation occurs here."""
    model_path = Path(args.predict_from).resolve()
    bundle = joblib.load(model_path)
    # Older bundles still contain ordinary sklearn/XGBoost model objects.
    # FT bundles instead reconstruct the network from portable CPU weights.
    if bundle.get("model_kind") == "ft_transformer":
        bundle["model"] = FTTransformerRegressor.from_state(bundle["ft_state"], args)
    elif bundle.get("model_kind") == "xgboost":
        bundle["model"].set_params(device=args.device or "cpu")
    df, _ = load_data(args)
    if bundle["feature_set"] == "median":
        predictions = bundle["model"].predict(np.zeros((len(df), 1)))
    else:
        frame = (
            pd.DataFrame({"complexity_score": numeric(df["complexity_score"])})
            if bundle["feature_set"] == "complexity_score_only"
            else feature_frame(df, bundle["feature_set"])
        )
        frame = frame.reindex(columns=bundle["numeric_columns"])
        X = bundle["preprocessor"].transform(frame).astype(np.float32)
        if bundle["text_state"] is not None:
            X = np.hstack(
                [X, transform_text(df, bundle["text_state"], model_path.parent, args)]
            )
        predictions = minutes_from_log(bundle["model"].predict(X))
    result = df[[c for c in ["__source_row"] + IDS if c in df]].copy()
    result["predicted_minutes_to_merge"] = predictions
    result.to_csv(Path(args.output) / "predictions.csv", index=False)
    LOG.info("Wrote %s predictions", len(result))


# =============================================================================
# Command-line workflow
# =============================================================================
def main():
    """Run training/evaluation by default, or inference with --predict-from.

    Workflow: parse options -> read/clean -> split repositories -> encode text
    -> median baseline -> tune/evaluate model-feature combinations -> save.
    Each model uses the same rows and split. Preprocessors learn from training
    rows only; selected models are not refit on training plus validation."""
    global PROGRESS_INTERVAL
    run_started = time.monotonic()
    # 1. Defaults run 4 existing models x 3 feature sets, plus structural FT.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default="cleaned_public.csv")
    parser.add_argument("--output", default="merge_time_results")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["ridge", "rf", "xgboost", "mlp", "ft_transformer"],
        default=["ridge", "rf", "xgboost", "mlp", "ft_transformer"],
        help="FT-Transformer runs on structural features only; other models use all requested feature sets",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["structural", "text", "review"],
        default=["structural", "text", "review"],
    )
    parser.add_argument(
        "--text-method", choices=["embeddings", "tfidf"], default="embeddings"
    )
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--device",
        default=None,
        help="FT/text device (automatic if omitted); explicit cuda also enables GPU XGBoost. Ridge/RF/sklearn MLP remain CPU",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Seconds between elapsed-time/batch progress updates (default: 30)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--text-dimensions", type=int, default=128)
    parser.add_argument("--tfidf-features", type=int, default=20000)
    parser.add_argument("--trials", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--exclude-zero", action="store_true")
    parser.add_argument(
        "--include-complexity-baseline",
        action="store_true",
        help="Add score-only ridge benchmark; assumes score may include final review information",
    )
    parser.add_argument(
        "--predict-from",
        default=None,
        help="Path to a saved model bundle; run inference instead of training",
    )
    # 2. Validate options and require a fresh output directory.
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not np.isfinite(args.progress_interval) or args.progress_interval <= 0:
        parser.error("--progress-interval must be a positive finite number")
    PROGRESS_INTERVAL = args.progress_interval
    for field in (
        "batch_size",
        "text_dimensions",
        "tfidf_features",
        "epochs",
        "patience",
    ):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    if not (
        0 < args.test_fraction < 1
        and 0 < args.validation_fraction < 1
        and args.test_fraction + args.validation_fraction < 1
    ):
        parser.error("Split fractions must be positive and sum to less than 1")
    if not args.predict_from and "ft_transformer" in args.models:
        if "structural" not in args.feature_sets:
            parser.error(
                "FT-Transformer currently needs --feature-sets structural (you may also include text/review for other models)"
            )
        # Fail early with an actionable dependency/device error. If ONLY FT
        # was selected, skip text/review entirely: those features are unused.
        ft_device(args.device)
        if set(args.models) == {"ft_transformer"}:
            args.feature_sets = ["structural"]
            LOG.info("FT-only run: using structural features; no text encoding needed")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error(
            "Output folder is not empty. Choose a new --output to preserve prior results."
        )
    output.mkdir(parents=True, exist_ok=True)
    # Keep the script's progress messages for later troubleshooting.
    file_log = logging.FileHandler(output / "run.log", encoding="utf-8")
    file_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(file_log)
    LOG.info(
        "Run started | models=%s | feature sets=%s | trials per model=%s",
        args.models,
        args.feature_sets,
        args.trials,
    )
    LOG.info(
        "Preprocessing/Ridge/Random Forest/sklearn MLP use CPU. XGBoost requested device=%s; FT/text device=%s",
        args.device or "cpu",
        args.device or "automatic",
    )
    LOG.info(
        "Progress updates every %.1f s; output=%s", PROGRESS_INTERVAL, output.resolve()
    )
    # Prediction mode exits early: it reuses a previously fitted model.
    if args.predict_from:
        predict_saved(args)
        return
    if "xgboost" in args.models:
        try:
            import xgboost  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Install xgboost, or explicitly select --models ridge rf mlp"
            ) from e
        if int(xgboost.__version__.split(".")[0]) < 2:
            raise ImportError(
                'This script needs XGBoost >= 2 for its device option. Run: python -m pip install --upgrade "xgboost>=2"'
            )
        if str(args.device).startswith("cuda") and not xgboost.build_info().get(
            "USE_CUDA", False
        ):
            raise RuntimeError(
                "Installed XGBoost has no CUDA support. Install a CUDA-enabled XGBoost build or select --device cpu."
            )
    # 3. Clean training data and create one shared repository-disjoint split.
    df, seen = load_data(args)
    df, audit = clean_training(df, args.exclude_zero)
    splits = make_splits(df, args)
    train, val, test = (splits[k] for k in ("train", "validation", "test"))
    audit["input_rows_scanned"] = seen
    audit["split_counts"] = {
        k: {"rows": len(v), "repositories": int(df.iloc[v].repository_id.nunique())}
        for k, v in splits.items()
    }
    audit["interpretation"] = (
        "Merged-PR duration; feature snapshot timing unverified. Review-inclusive experiments are retrospective."
    )
    dump_json(output / "audit.json", audit)
    # Record the run settings and installed package versions for reproducibility.
    versions = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "sentence-transformers",
        "torch",
        "rtdl-revisiting-models",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    dump_json(
        output / "run_config.json",
        {
            "arguments": vars(args),
            "versions": versions,
            "selection_metric": "validation MAE of log1p(minutes_to_merge)",
            "refit_train_plus_validation": False,
        },
    )
    labels = np.full(len(df), "", dtype=object)
    for label, idx in splits.items():
        labels[idx] = label
    with progress_step("Save compressed repository split assignments (CPU/disk)"):
        df.assign(split=labels)[["__source_row"] + IDS + ["split"]].to_csv(
            output / "splits.csv.gz", index=False
        )
    LOG.info("Cleaned dataset: %s", audit["split_counts"])
    # 4. Encode title/body text once and reuse the vectors across experiments.
    text_matrix, text_state = None, None
    if any(s in args.feature_sets for s in ("text", "review")):
        if not all(c in df for c in RAW_TEXT):
            raise ValueError("Text experiments require original title and body columns")
        text_matrix, text_state = fit_text(df, train, args)
    y = df[TARGET].to_numpy(dtype=float)
    # Log(1 + minutes) compresses the long tail and accepts zero-minute PRs.
    # All five regressors fit this transformed target, not raw minutes.
    yl = np.log1p(y)
    predictions = df.iloc[test][["__source_row"] + IDS + [TARGET]].copy()
    results, trials, duration_results, schemas = [], [], [], {}
    ft_history = []  # Per-epoch curves, with experiment/trial labels.
    test_groups = df.iloc[test].repository_id.to_numpy()
    # 5. Baseline: predict the training-set median duration for every test PR.
    # This baseline fits raw minutes, unlike the five learned regressors.
    with progress_step("Fit median baseline and predict test durations (CPU)"):
        baseline = DummyRegressor(strategy="median").fit(
            np.zeros((len(train), 1)), y[train]
        )
        base_pred = baseline.predict(np.zeros((len(test), 1)))
    results.append(
        {
            "experiment": "median_baseline",
            "validation_log_mae": float(
                mean_absolute_error(
                    yl[val], np.log1p(baseline.predict(np.zeros((len(val), 1))))
                )
            ),
            **metrics(y[test], base_pred, test_groups),
        }
    )
    predictions["median_baseline"] = base_pred
    joblib.dump(
        {"model": baseline, "feature_set": "median"}, output / "median_baseline.joblib"
    )
    # 6. Compare cumulative feature sets. The optional complexity benchmark
    # uses only complexity_score and always fits Ridge.
    names = list(dict.fromkeys(args.feature_sets))
    if args.include_complexity_baseline:
        if "complexity_score" not in df:
            raise ValueError(
                "--include-complexity-baseline needs complexity_score column"
            )
        names += ["complexity_score_only"]
    for feature_set in names:
        # Start FT with structural features; it is not yet a hybrid text model.
        # Keep the original feature comparisons for the four existing models.
        model_names = (
            ["ridge"]
            if feature_set == "complexity_score_only"
            else [
                name
                for name in dict.fromkeys(args.models)
                if name != "ft_transformer" or feature_set == "structural"
            ]
        )
        if not model_names:
            continue
        LOG.info("FEATURE SET: %s | models: %s", feature_set, model_names)
        frame = (
            pd.DataFrame({"complexity_score": numeric(df.complexity_score)})
            if feature_set == "complexity_score_only"
            else feature_frame(df, feature_set)
        )
        # Remove unusable/constant columns based only on training data.
        with progress_step(f"{feature_set}: find usable training columns (CPU)"):
            # Slice once rather than copying the training frame per column.
            training_frame = frame.iloc[train]
            usable = [c for c in frame if training_frame[c].nunique(dropna=True) > 1]
            del training_frame
        if not usable:
            raise ValueError(f"No varying training features for {feature_set}")
        frame = frame[usable]
        # Fill missing values with training medians, add missingness indicators
        # where learned by the imputer, then standardize numeric features.
        preprocessor = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
        )
        with progress_step(
            f"{feature_set}: fit numeric imputer/scaler on training PRs (CPU)"
        ):
            Xtrain = preprocessor.fit_transform(frame.iloc[train]).astype(np.float32)
        with progress_step(f"{feature_set}: transform validation/test PRs (CPU)"):
            Xval = preprocessor.transform(frame.iloc[val]).astype(np.float32)
            Xtest = preprocessor.transform(frame.iloc[test]).astype(np.float32)
        use_text = feature_set in ("text", "review")
        if use_text:
            Xtrain = np.hstack([Xtrain, text_matrix[train]])
            Xval = np.hstack([Xval, text_matrix[val]])
            Xtest = np.hstack([Xtest, text_matrix[test]])
        schemas[feature_set] = {
            "numeric_columns": usable,
            "transformed_numeric_columns": preprocessor.get_feature_names_out().tolist(),
            "text_dimensions": int(text_matrix.shape[1]) if use_text else 0,
        }
        LOG.info(
            "Feature matrices ready: train=%s, validation=%s, test=%s",
            Xtrain.shape,
            Xval.shape,
            Xtest.shape,
        )
        # Train every requested model on this feature set.
        for model_name in model_names:
            name = f"{feature_set}_{model_name}"
            best_model, best_score, best_parameters, best_epoch = (
                None,
                np.inf,
                None,
                None,
            )
            start = time.monotonic()
            # Validation log-MAE chooses the configuration; the test split
            # is not consulted during this search.
            for number, params in enumerate(candidates(model_name, args), 1):
                device = (
                    (args.device or "automatic")
                    if model_name == "ft_transformer"
                    else (params.get("device", "cpu"))
                )
                LOG.info(
                    "MODEL: %s | trial %s/%s | device=%s | parameters=%s",
                    name,
                    number,
                    len(candidates(model_name, args)),
                    device,
                    json.dumps(params),
                )
                model, epoch = train_candidate(
                    model_name, params, Xtrain, yl[train], Xval, yl[val], args
                )
                if model_name == "ft_transformer":
                    ft_history.extend(
                        {"experiment": name, "trial": number, **row}
                        for row in model.history
                    )
                    pd.DataFrame(ft_history).to_csv(
                        output / "ft_training_history.csv", index=False
                    )
                score = float(
                    mean_absolute_error(yl[val], np.maximum(0, model.predict(Xval)))
                )
                trials.append(
                    {
                        "experiment": name,
                        "trial": number,
                        "validation_log_mae": score,
                        "best_epoch": epoch,
                        "parameters": json.dumps(params),
                    }
                )
                if score < best_score:
                    best_model, best_score, best_parameters, best_epoch = (
                        model,
                        score,
                        params,
                        epoch,
                    )
                LOG.info(
                    "TRIAL DONE: %s | trial=%s | validation log-MAE=%.4f | best so far=%.4f",
                    name,
                    number,
                    score,
                    best_score,
                )
            # 7. Evaluate the chosen configuration once on held-out test PRs.
            with progress_step(f"{name}: predict {len(test):,} held-out test PRs"):
                pred = minutes_from_log(best_model.predict(Xtest))
            result = {
                "experiment": name,
                "validation_log_mae": best_score,
                "training_seconds": time.monotonic() - start,
                **metrics(y[test], pred, test_groups),
            }
            results.append(result)
            predictions[name] = pred
            # These groups describe actual test duration for error reporting.
            # They are not classification labels or model outputs.
            duration_labels = pd.cut(
                y[test],
                [-1, 60, 1440, 10080, np.inf],
                labels=["0_to_60_minutes", "over_1h_to_1d", "over_1d_to_7d", "over_7d"],
            )
            for band in duration_labels.categories:
                mask = np.asarray(duration_labels == band)
                if mask.any():
                    duration_results.append(
                        {
                            "experiment": name,
                            "actual_duration": band,
                            "n": int(mask.sum()),
                            "mae_minutes": float(
                                np.mean(np.abs(y[test][mask] - pred[mask]))
                            ),
                            "median_ae_minutes": float(
                                np.median(np.abs(y[test][mask] - pred[mask]))
                            ),
                        }
                    )
            # Save the fitted model together with its preprocessing so future
            # inputs receive the same transformations. Text bundles also need
            # the sibling text_encoder directory when using embeddings.
            bundle = {
                # Never pickle our script-defined wrapper: this also allows
                # loading bundles when this .py file is renamed or imported.
                "model": None if model_name == "ft_transformer" else best_model,
                "model_kind": model_name,
                "preprocessor": preprocessor,
                "feature_set": feature_set,
                "numeric_columns": usable,
                "text_state": text_state if use_text else None,
                "parameters": best_parameters,
                "best_epoch": best_epoch,
                "target_transform": "log1p",
                "prediction_log_clip": [0, 50],
            }
            if model_name == "ft_transformer":
                bundle["ft_state"] = best_model.export_state()
            with progress_step(
                f"{name}: save model, metrics, and compressed predictions (CPU/disk)"
            ):
                joblib.dump(bundle, output / f"{name}.joblib", compress=3)
                # Checkpoint each experiment so interrupted long runs retain results.
                pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)
                pd.DataFrame(trials).to_csv(
                    output / "validation_trials.csv", index=False
                )
                predictions.to_csv(output / "predictions.csv.gz", index=False)
            LOG.info(
                "%s: test MAE %.2f minutes, validation log-MAE %.4f",
                name,
                result["mae_minutes"],
                best_score,
            )
    # 8. Save summaries and pick the overall winner by validation log-MAE.
    # The median baseline can win; test MAE can rank models differently.
    table = pd.DataFrame(results)
    pd.DataFrame(duration_results).to_csv(output / "duration_metrics.csv", index=False)
    dump_json(output / "feature_schema.json", schemas)
    selected = min(results, key=lambda row: row["validation_log_mae"])
    dump_json(
        output / "selected_on_validation.json",
        {
            "experiment": selected["experiment"],
            "validation_log_mae": selected["validation_log_mae"],
            "note": "Selection includes the median baseline. Review-inclusive/score-only models have a different information budget; compare within the intended use case.",
        },
    )
    LOG.info("Saving model comparison plot")
    plot_comparison(table, output / "model_comparison.png")
    LOG.info("Comparison plot saved")
    print(
        table[
            ["experiment", "mae_minutes", "median_ae_minutes", "rmsle", "r2"]
        ].to_string(index=False)
    )
    LOG.info("Saved results to %s", output.resolve())
    LOG.info(
        "RUN COMPLETE | total elapsed %.1f minutes",
        (time.monotonic() - run_started) / 60,
    )


if __name__ == "__main__":
    main()
