#!/usr/bin/env python3
"""Compare PR merge-time regressors using cleaned_public.csv (Python 3.10+).

INSTALL:
  python -m pip install numpy pandas scipy scikit-learn matplotlib joblib xgboost sentence-transformers
RUN (all four models, pretrained text embeddings):
  python predict_merge_time.py --input cleaned_public.csv --output merge_time_results
OFFLINE TEXT OPTION (no pretrained model download):
  python predict_merge_time.py --input cleaned_public.csv --text-method tfidf
QUICK CHECK:
  python predict_merge_time.py --input cleaned_public.csv --max-rows 10000 --trials 1 --epochs 10 --text-method tfidf
STRUCTURAL ONLY:
  python predict_merge_time.py --input cleaned_public.csv --feature-sets structural --models ridge rf
PREDICT WITH A SAVED MODEL (same raw-column schema, target not required):
  python predict_merge_time.py --input new_prs.csv --predict-from merge_time_results/structural_xgboost.joblib --output new_predictions

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
* The MLP uses sklearn on CPU (three hidden layers); pretrained text encoding
  can use a GPU. This architecture is inspired by mixed text/tabular approaches,
  not a replication of DeepPull. Default encoder truncates long descriptions.
* Models and preprocessing are saved together. Load joblib files only if trusted.
* Full CSV is read into RAM; --max-rows performs a uniform streaming row sample
  before cleaning. It is a development option, not the full-dataset experiment.

Outputs: metrics.csv, duration_metrics.csv, predictions.csv.gz, splits.csv.gz,
validation_trials.csv, model_comparison.png, audit.json, run_config.json,
feature_schema.json, selected_on_validation.json, and fitted *.joblib bundles.
The selected model is chosen using validation error, never test results.
Pretrained embeddings require internet on first use or a local --encoder path.

Related work:
  Maddila et al., FSE 2019: https://doi.org/10.1145/3338906.3340457
  Nudge, TOSEM 2022: https://arxiv.org/abs/2011.12468
  DeepPull, 2024: https://doi.org/10.1007/978-3-031-61753-9_6
  Pansuriya et al., 2026 preprint: https://arxiv.org/abs/2607.12057
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.metadata
import json
import logging
import time
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

LOG = logging.getLogger("merge_time")
TARGET = "minutes_to_merge"
NUMERIC = ["files_added", "files_deleted", "files_modified", "files_touched",
           "lines_modified", "code_lines_modified", "config_lines_modified", "is_code_pr"]
REVIEW = ["num_reviewers", "num_review_comments"]
EXTRA = ["languages_list", "lines_added_by_language", "lines_deleted_by_language"]
RAW_TEXT = ["title", "body"]
IDS = ["repository_id", "pull_request_id"]


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def parse_container(value, expected):
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
    normalized = series.astype("string").str.strip().str.lower().replace({"true": "1", "false": "0"})
    return pd.to_numeric(normalized, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def structural_features(df):
    missing = [c for c in NUMERIC if c not in df]
    if missing:
        raise ValueError(f"Missing structural columns: {missing}")
    out = pd.DataFrame({c: numeric(df[c]) for c in NUMERIC}, index=df.index)
    if "languages_list" in df:
        languages = df["languages_list"].map(lambda v: parse_container(v, list))
        out["language_count"] = languages.map(lambda v: len(set(map(str, v))))
    for source in ["lines_added_by_language", "lines_deleted_by_language"]:
        if source in df:
            parsed = df[source].map(lambda v: parse_container(v, dict))
            def total(mapping):
                values = pd.to_numeric(pd.Series(list(mapping.values()), dtype=object), errors="coerce")
                return float(values.replace([np.inf, -np.inf], np.nan).sum())
            out[source + "_total"] = parsed.map(total)
    return out


def text_numeric(df):
    title = df["title"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    # Recompute lengths/flags to keep their definitions consistent on new inputs.
    return pd.DataFrame({"title_words": title.str.split().str.len(),
                         "body_words": body.str.split().str.len(),
                         "has_title": title.str.strip().ne("").astype(int),
                         "has_body": body.str.strip().ne("").astype(int)}, index=df.index)


def text_input(df):
    return (df["title"].fillna("").astype(str) + "\n\n" + df["body"].fillna("").astype(str)).tolist()


def load_data(args):
    path = Path(args.input)
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {path.resolve()}")
    available = pd.read_csv(path, nrows=0).columns.tolist()
    needs_text = args.predict_from or any(s in args.feature_sets for s in ("text", "review"))
    wanted = IDS + NUMERIC + EXTRA + [TARGET] + (RAW_TEXT if needs_text else []) + REVIEW
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
            chunk["__sample_key"] = rng.random(len(chunk))
            sample = chunk if sample is None else pd.concat([sample, chunk], ignore_index=True)
            sample = sample.nsmallest(args.max_rows, "__sample_key")
        if sample is None:
            raise ValueError("CSV has no data rows")
        df = sample.drop(columns="__sample_key").sort_values("__source_row")
    else:
        df = pd.read_csv(path, **kwargs)
        df["__source_row"] = np.arange(len(df))
        seen = len(df)
    return df.reset_index(drop=True), seen


def clean_training(df, exclude_zero):
    required = IDS + NUMERIC + [TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    audit = {"loaded_rows": len(df)}
    valid_ids = df[IDS].notna().all(axis=1) & df[IDS].apply(lambda s: s.str.strip().ne("")).all(axis=1)
    audit["invalid_identifier_rows"] = int((~valid_ids).sum())
    df = df.loc[valid_ids].copy()
    duplicated = df.duplicated(IDS, keep="first")
    audit["duplicate_rows_removed"] = int(duplicated.sum())
    # Do not silently choose between inconsistent copies of the same PR.
    duplicate_rows = df.loc[df.duplicated(IDS, keep=False)]
    if len(duplicate_rows):
        check_cols = [c for c in df.columns if c != "__source_row"]
        if duplicate_rows.drop_duplicates(check_cols).duplicated(IDS).any():
            raise ValueError("Conflicting rows share a PR identifier. Resolve duplicate snapshots before training.")
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
        raise ValueError("Need at least 30 valid PRs and 6 repositories for train/validation/test splitting.")
    return df, audit


def make_splits(df, args):
    groups = df.repository_id.to_numpy()
    trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=args.test_fraction,
                                           random_state=args.seed).split(df, groups=groups))
    relative = args.validation_fraction / (1 - args.test_fraction)
    train_local, val_local = next(GroupShuffleSplit(n_splits=1, test_size=relative,
                           random_state=args.seed + 1).split(trainval, groups=groups[trainval]))
    splits = {"train": trainval[train_local], "validation": trainval[val_local], "test": test}
    sets = [set(groups[v]) for v in splits.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    if any(len(v) < 2 for v in splits.values()):
        raise ValueError("A split has fewer than two PRs; use more data or adjust split fractions.")
    return splits


def fit_text(df, train, args):
    texts = text_input(df)
    if args.text_method == "embeddings":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("Install sentence-transformers, or use --text-method tfidf.") from e
        encoder = SentenceTransformer(args.encoder, device=args.device)
        LOG.info("Encoding %s PR descriptions; long inputs are truncated at encoder token limit %s",
                 len(texts), encoder.max_seq_length)
        encoded = encoder.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                                 normalize_embeddings=True, convert_to_numpy=True)
        # Bundle the actual encoder locally so inference need not redownload it.
        destination = Path(args.output).resolve() / "text_encoder"
        encoder.save(str(destination))
        return np.asarray(encoded, dtype=np.float32), {"method": "embeddings", "encoder_dir": "text_encoder"}
    vectorizer = TfidfVectorizer(max_features=args.tfidf_features, ngram_range=(1, 2),
                                min_df=1, sublinear_tf=True, strip_accents="unicode")
    training = vectorizer.fit_transform([texts[i] for i in train])
    if training.shape[1] < 2:
        raise ValueError("Not enough distinct text tokens for TF-IDF. Use structural-only experiments.")
    components = min(args.text_dimensions, training.shape[1] - 1, training.shape[0] - 1)
    svd = TruncatedSVD(n_components=components, random_state=args.seed).fit(training)
    encoded = svd.transform(vectorizer.transform(texts)).astype(np.float32)
    return encoded, {"method": "tfidf", "vectorizer": vectorizer, "svd": svd}


def transform_text(df, state, bundle_dir, args):
    texts = text_input(df)
    if state["method"] == "tfidf":
        return state["svd"].transform(state["vectorizer"].transform(texts)).astype(np.float32)
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(str(bundle_dir / state["encoder_dir"]), device=args.device)
    return encoder.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                          show_progress_bar=True, convert_to_numpy=True).astype(np.float32)


def feature_frame(df, name):
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


def candidates(name, args):
    if name == "ridge":
        return [{"alpha": a} for a in (1., 10., 100., 1000.)][:args.trials]
    if name == "rf":
        return [{"n_estimators": 200, "max_depth": d, "min_samples_leaf": leaf,
                 "max_features": .8, "n_jobs": args.jobs, "random_state": args.seed}
                for d, leaf in ((16, 5), (24, 10), (None, 5), (12, 20))][:args.trials]
    if name == "xgboost":
        return [{"n_estimators": n, "max_depth": d, "learning_rate": lr,
                 "subsample": .8, "colsample_bytree": .8, "reg_lambda": 5.,
                 "objective": "reg:squarederror", "tree_method": "hist",
                 "n_jobs": args.jobs, "random_state": args.seed}
                for n, d, lr in ((300, 4, .05), (500, 6, .05), (500, 8, .03), (300, 3, .08))][:args.trials]
    return [{"hidden_layer_sizes": layers, "alpha": alpha, "learning_rate_init": lr,
             "random_state": args.seed, "batch_size": args.batch_size,
             "early_stopping": False, "shuffle": True}
            for layers, alpha, lr in (((256, 128, 64), .01, .001),
                                     ((128, 64, 32), .1, .0005),
                                     ((256, 128, 64), 1., .0005),
                                     ((128, 64, 32), .01, .001))][:args.trials]


def train_candidate(name, params, Xtrain, ytrain, Xval, yval, args):
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
            if not np.isfinite(loss):
                raise ValueError("MLP produced nonfinite validation error")
            if loss < best_loss - 1e-5:
                best, best_loss, stale, best_epoch = copy.deepcopy(model), loss, 0, epoch + 1
            else:
                stale += 1
            if stale >= args.patience:
                break
        return best, best_epoch
    model.fit(Xtrain, ytrain)
    return model, None


def minutes_from_log(values):
    if not np.isfinite(values).all():
        raise ValueError("Model produced nonfinite predictions")
    return np.expm1(np.clip(values, 0, 50))


def metrics(y, pred, groups):
    errors = np.abs(y - pred)
    correlation = pd.Series(y).corr(pd.Series(pred), method="spearman") if np.unique(pred).size > 1 and np.unique(y).size > 1 else np.nan
    return {"n": len(y), "mae_minutes": float(mean_absolute_error(y, pred)),
            "median_ae_minutes": float(np.median(errors)),
            "rmse_minutes": float(np.sqrt(mean_squared_error(y, pred))),
            "rmsle": float(np.sqrt(mean_squared_error(np.log1p(y), np.log1p(pred)))),
            "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
            "spearman": float(correlation),
            "macro_repository_mae_minutes": float(pd.DataFrame({"repo": groups, "error": errors}).groupby("repo").error.mean().mean())}


def plot_comparison(table, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = table.sort_values("mae_minutes", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, len(table) * .35)))
    ax.barh(table["experiment"], table.mae_minutes, color="#246b8e")
    ax.set_xlabel("Test MAE (minutes; lower is better)")
    ax.set_title("Merge-time prediction: repositories held out from training")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def predict_saved(args):
    model_path = Path(args.predict_from).resolve()
    bundle = joblib.load(model_path)
    df, _ = load_data(args)
    if bundle["feature_set"] == "median":
        predictions = bundle["model"].predict(np.zeros((len(df), 1)))
    else:
        frame = (pd.DataFrame({"complexity_score": numeric(df["complexity_score"])})
                 if bundle["feature_set"] == "complexity_score_only" else feature_frame(df, bundle["feature_set"]))
        frame = frame.reindex(columns=bundle["numeric_columns"])
        X = bundle["preprocessor"].transform(frame).astype(np.float32)
        if bundle["text_state"] is not None:
            X = np.hstack([X, transform_text(df, bundle["text_state"], model_path.parent, args)])
        predictions = minutes_from_log(bundle["model"].predict(X))
    result = df[[c for c in ["__source_row"] + IDS if c in df]].copy()
    result["predicted_minutes_to_merge"] = predictions
    result.to_csv(Path(args.output) / "predictions.csv", index=False)
    LOG.info("Wrote %s predictions", len(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="cleaned_public.csv")
    parser.add_argument("--output", default="merge_time_results")
    parser.add_argument("--models", nargs="+", choices=["ridge", "rf", "xgboost", "mlp"], default=["ridge", "rf", "xgboost", "mlp"])
    parser.add_argument("--feature-sets", nargs="+", choices=["structural", "text", "review"], default=["structural", "text", "review"])
    parser.add_argument("--text-method", choices=["embeddings", "tfidf"], default="embeddings")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default=None, help="Text encoder device: cpu, cuda, or automatic if omitted")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--text-dimensions", type=int, default=128)
    parser.add_argument("--tfidf-features", type=int, default=20000)
    parser.add_argument("--trials", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=.15)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--exclude-zero", action="store_true")
    parser.add_argument("--include-complexity-baseline", action="store_true", help="Add score-only ridge benchmark; assumes score may include final review information")
    parser.add_argument("--predict-from", default=None, help="Path to a saved model bundle; run inference instead of training")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for field in ("batch_size", "text_dimensions", "tfidf_features", "epochs", "patience"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    if not (0 < args.test_fraction < 1 and 0 < args.validation_fraction < 1 and args.test_fraction + args.validation_fraction < 1):
        parser.error("Split fractions must be positive and sum to less than 1")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output folder is not empty. Choose a new --output to preserve prior results.")
    output.mkdir(parents=True, exist_ok=True)
    if args.predict_from:
        predict_saved(args)
        return
    if "xgboost" in args.models:
        try:
            import xgboost  # noqa: F401
        except ImportError as e:
            raise ImportError("Install xgboost, or explicitly select --models ridge rf mlp") from e
    df, seen = load_data(args)
    df, audit = clean_training(df, args.exclude_zero)
    splits = make_splits(df, args)
    train, val, test = (splits[k] for k in ("train", "validation", "test"))
    audit["input_rows_scanned"] = seen
    audit["split_counts"] = {k: {"rows": len(v), "repositories": int(df.iloc[v].repository_id.nunique())} for k, v in splits.items()}
    audit["interpretation"] = "Merged-PR duration; feature snapshot timing unverified. Review-inclusive experiments are retrospective."
    dump_json(output / "audit.json", audit)
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "xgboost", "sentence-transformers", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    dump_json(output / "run_config.json", {"arguments": vars(args), "versions": versions,
                                         "selection_metric": "validation MAE of log1p(minutes_to_merge)",
                                         "refit_train_plus_validation": False})
    labels = np.full(len(df), "", dtype=object)
    for label, idx in splits.items():
        labels[idx] = label
    df.assign(split=labels)[["__source_row"] + IDS + ["split"]].to_csv(output / "splits.csv.gz", index=False)
    LOG.info("Cleaned dataset: %s", audit["split_counts"])
    text_matrix, text_state = None, None
    if any(s in args.feature_sets for s in ("text", "review")):
        if not all(c in df for c in RAW_TEXT):
            raise ValueError("Text experiments require original title and body columns")
        text_matrix, text_state = fit_text(df, train, args)
    y = df[TARGET].to_numpy(dtype=float)
    yl = np.log1p(y)
    predictions = df.iloc[test][["__source_row"] + IDS + [TARGET]].copy()
    results, trials, duration_results, schemas = [], [], [], {}
    test_groups = df.iloc[test].repository_id.to_numpy()
    baseline = DummyRegressor(strategy="median").fit(np.zeros((len(train), 1)), y[train])
    base_pred = baseline.predict(np.zeros((len(test), 1)))
    results.append({"experiment": "median_baseline", "validation_log_mae": float(mean_absolute_error(yl[val], np.log1p(baseline.predict(np.zeros((len(val), 1)))))) , **metrics(y[test], base_pred, test_groups)})
    predictions["median_baseline"] = base_pred
    joblib.dump({"model": baseline, "feature_set": "median"}, output / "median_baseline.joblib")
    names = list(dict.fromkeys(args.feature_sets))
    if args.include_complexity_baseline:
        if "complexity_score" not in df:
            raise ValueError("--include-complexity-baseline needs complexity_score column")
        names += ["complexity_score_only"]
    for feature_set in names:
        frame = (pd.DataFrame({"complexity_score": numeric(df.complexity_score)})
                 if feature_set == "complexity_score_only" else feature_frame(df, feature_set))
        # Remove unusable/constant columns based only on training data.
        usable = [c for c in frame if frame.iloc[train][c].nunique(dropna=True) > 1]
        if not usable:
            raise ValueError(f"No varying training features for {feature_set}")
        frame = frame[usable]
        preprocessor = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
        Xtrain = preprocessor.fit_transform(frame.iloc[train]).astype(np.float32)
        Xval = preprocessor.transform(frame.iloc[val]).astype(np.float32)
        Xtest = preprocessor.transform(frame.iloc[test]).astype(np.float32)
        use_text = feature_set in ("text", "review")
        if use_text:
            Xtrain = np.hstack([Xtrain, text_matrix[train]])
            Xval = np.hstack([Xval, text_matrix[val]])
            Xtest = np.hstack([Xtest, text_matrix[test]])
        schemas[feature_set] = {"numeric_columns": usable, "transformed_numeric_columns": preprocessor.get_feature_names_out().tolist(),
                                "text_dimensions": int(text_matrix.shape[1]) if use_text else 0}
        for model_name in (["ridge"] if feature_set == "complexity_score_only" else list(dict.fromkeys(args.models))):
            name = f"{feature_set}_{model_name}"
            best_model, best_score, best_parameters, best_epoch = None, np.inf, None, None
            start = time.monotonic()
            for number, params in enumerate(candidates(model_name, args), 1):
                LOG.info("%s: validation trial %s", name, number)
                model, epoch = train_candidate(model_name, params, Xtrain, yl[train], Xval, yl[val], args)
                score = float(mean_absolute_error(yl[val], np.maximum(0, model.predict(Xval))))
                trials.append({"experiment": name, "trial": number, "validation_log_mae": score,
                               "best_epoch": epoch, "parameters": json.dumps(params)})
                if score < best_score:
                    best_model, best_score, best_parameters, best_epoch = model, score, params, epoch
            pred = minutes_from_log(best_model.predict(Xtest))
            result = {"experiment": name, "validation_log_mae": best_score,
                      "training_seconds": time.monotonic() - start, **metrics(y[test], pred, test_groups)}
            results.append(result)
            predictions[name] = pred
            duration_labels = pd.cut(y[test], [-1, 60, 1440, 10080, np.inf], labels=["0_to_60_minutes", "over_1h_to_1d", "over_1d_to_7d", "over_7d"])
            for band in duration_labels.categories:
                mask = np.asarray(duration_labels == band)
                if mask.any():
                    duration_results.append({"experiment": name, "actual_duration": band,
                                             "n": int(mask.sum()), "mae_minutes": float(np.mean(np.abs(y[test][mask] - pred[mask]))),
                                             "median_ae_minutes": float(np.median(np.abs(y[test][mask] - pred[mask])))})
            bundle = {"model": best_model, "preprocessor": preprocessor, "feature_set": feature_set,
                      "numeric_columns": usable, "text_state": text_state if use_text else None,
                      "parameters": best_parameters, "best_epoch": best_epoch,
                      "target_transform": "log1p", "prediction_log_clip": [0, 50]}
            joblib.dump(bundle, output / f"{name}.joblib", compress=3)
            # Checkpoint each experiment so interrupted long runs retain results.
            pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)
            pd.DataFrame(trials).to_csv(output / "validation_trials.csv", index=False)
            predictions.to_csv(output / "predictions.csv.gz", index=False)
            LOG.info("%s: test MAE %.2f minutes, validation log-MAE %.4f", name, result["mae_minutes"], best_score)
    table = pd.DataFrame(results)
    pd.DataFrame(duration_results).to_csv(output / "duration_metrics.csv", index=False)
    dump_json(output / "feature_schema.json", schemas)
    selected = min(results, key=lambda row: row["validation_log_mae"])
    dump_json(output / "selected_on_validation.json", {"experiment": selected["experiment"], "validation_log_mae": selected["validation_log_mae"],
               "note": "Selection includes the median baseline. Review-inclusive/score-only models have a different information budget; compare within the intended use case."})
    plot_comparison(table, output / "model_comparison.png")
    print(table[["experiment", "mae_minutes", "median_ae_minutes", "rmsle", "r2"]].to_string(index=False))
    LOG.info("Saved results to %s", output.resolve())


if __name__ == "__main__":
    main()
