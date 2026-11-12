"""Shared schema, deterministic preprocessing, splits, and output helpers."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import logging
import platform
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

LOG = logging.getLogger("paper2")
KEYS = ["repository_id", "pull_request_id"]
TARGET = "minutes_to_merge"
STRUCTURAL = ["files_added", "files_deleted", "files_modified", "files_touched",
              "lines_modified", "code_lines_modified", "config_lines_modified", "is_code_pr"]
TEXT = ["title_word_count", "body_word_count", "has_title", "has_body", "short_body"]
REVIEW = ["num_reviewers", "num_review_comments"]
FLAGS = ["is_open_source", "is_code_pr", "had_copilot_seat", "has_title", "has_body", "short_body"]
CLASSES = ["fast", "medium", "slow"]
FEATURE_SETS = {
    "structural": STRUCTURAL + ["language_count"],
    "description": STRUCTURAL + ["language_count"] + TEXT,
    "copilot": STRUCTURAL + ["language_count"] + TEXT + ["had_copilot_seat"],
    "review": STRUCTURAL + ["language_count"] + TEXT + REVIEW,
    "review_copilot": STRUCTURAL + ["language_count"] + TEXT + REVIEW + ["had_copilot_seat"],
}


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def dump_json(path, value):
    def convert(v):
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, (Path, pd.Timestamp)):
            return str(v)
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n", encoding="utf-8")


def fresh_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output is not empty: {path}. Choose a new run directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def versions():
    names = ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "xgboost", "joblib",
             "sentence-transformers", "transformers", "torch"]
    result = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size,
            "sha256": digest.hexdigest()}


def numeric(series):
    return pd.to_numeric(series, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def boolean(series):
    mapping = {"true": 1., "false": 0., "1": 1., "0": 0., "1.0": 1., "0.0": 0.,
               "yes": 1., "no": 0., "": np.nan}
    text = series.astype("string").str.strip().str.lower()
    invalid = text.notna() & ~text.isin(mapping)
    if invalid.any():
        raise ValueError(f"Invalid boolean in {series.name}: {text[invalid].unique()[:5].tolist()}")
    return text.map(mapping).astype(float)


@lru_cache(maxsize=16384)
def parse_languages_string(value):
    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(value)
            if isinstance(result, (list, tuple)):
                return tuple(str(x) for x in result if x is not None and str(x).strip())
        except (ValueError, SyntaxError, TypeError, RecursionError):
            pass
    raise ValueError(f"languages_list must contain JSON/Python lists, got {value[:100]!r}")


def parse_languages(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(str(x) for x in value if x is not None and str(x).strip())
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        return parse_languages_string(value.strip()) if value.strip() else None
    raise ValueError(f"Unsupported languages_list type: {type(value).__name__}")


def read_source(path, include_text=False):
    """Read selected columns; raw title/body are opt-in for text experiments."""
    import pyarrow.parquet as pq
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Expected one CSV, CSV.GZ, or Parquet file: {path}")
    parquet = path.suffix.lower() in {".parquet", ".pq"}
    available = pq.read_schema(path).names if parquet else pd.read_csv(path, nrows=0).columns.tolist()
    required = KEYS + STRUCTURAL + REVIEW + TEXT + [TARGET, "languages_list", "is_open_source", "had_copilot_seat"]
    if include_text:
        required += ["title", "body"]
    missing = sorted(set(required) - set(available))
    if missing:
        raise ValueError(f"Missing columns: {missing}. This workflow expects the supplied 37-column schema.")
    use = list(dict.fromkeys(required + [c for c in ["merge_day", "merge_month", "user_id"] if c in available]))
    if parquet:
        df = pd.read_parquet(path, columns=use)
    else:
        df = pd.read_csv(path, usecols=use, dtype={c: "string" for c in KEYS + ["user_id"] if c in use}, low_memory=False)
    return df


def clean_data(df, scope, exclude_zero=False):
    df = df.copy()
    audit = {"input_rows": len(df), "scope": scope, "exclude_zero": exclude_zero}
    for c in ["title", "body"]:
        if c in df:
            df[c] = df[c].fillna("").astype(str)
    audit["retained_text_columns"] = [c for c in ["title", "body"] if c in df]
    for c in KEYS + (["user_id"] if "user_id" in df else []):
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA)
    valid_keys = df[KEYS].notna().all(axis=1)
    audit["missing_key_rows"] = int((~valid_keys).sum())
    df = df.loc[valid_keys].copy()
    for c in FLAGS:
        df[c] = boolean(df[c])
    counts = list(dict.fromkeys(STRUCTURAL + REVIEW + TEXT + [TARGET]))
    audit["invalid_numeric_cells"] = {}
    for c in counts:
        if c in FLAGS:
            continue
        original = df[c]
        values = numeric(original)
        bad = original.notna() & (values.isna() | values.lt(0))
        audit["invalid_numeric_cells"][c] = int(bad.sum())
        df[c] = values.mask(values.lt(0))
    df["languages_list"] = df["languages_list"].map(parse_languages)
    # Lists are no longer needed after deterministic feature extraction.
    df["language_count"] = df.languages_list.map(lambda x: float(len(set(x))) if x is not None else np.nan)
    df["first_language"] = df.languages_list.map(lambda x: x[0] if x else "__unknown__")
    df = df.drop(columns="languages_list")
    duplicates = df.duplicated(KEYS, keep=False)
    candidates = df.loc[duplicates].drop_duplicates()
    if candidates.duplicated(KEYS).any():
        raise ValueError("Conflicting records share repository_id + pull_request_id. Resolve PR snapshots before analysis.")
    before = len(df)
    df = df.drop_duplicates(KEYS).copy()
    audit["identical_duplicate_rows_removed"] = before - len(df)
    valid_target = df[TARGET].notna() & (df[TARGET].gt(0) if exclude_zero else df[TARGET].ge(0))
    audit["invalid_or_excluded_target_rows"] = int((~valid_target).sum())
    df = df.loc[valid_target].copy()
    audit["open_source_counts_before_filter"] = {str(k): int(v) for k, v in df.is_open_source.value_counts(dropna=False).items()}
    before = len(df)
    if scope != "all":
        df = df.loc[df.is_open_source.eq(1 if scope == "open-source" else 0)].copy()
    audit["scope_rows_removed"] = before - len(df)
    if df.empty:
        raise ValueError("No valid PRs remain after cleaning and scope selection.")
    if "merge_day" in df:
        df["merge_day"] = pd.to_datetime(df.merge_day, errors="coerce", utc=True).dt.floor("D")
    df = df.reset_index(drop=True)
    for c in counts + ["had_copilot_seat", "is_open_source", "language_count"]:
        df[c] = df[c].astype("float32")
    audit.update(output_rows=len(df), repositories=int(df.repository_id.nunique()),
                 zero_minute_rows=int(df[TARGET].eq(0).sum()),
                 missing_copilot_rows=int(df.had_copilot_seat.isna().sum()),
                 feature_timing="unverified_final_snapshot",
                 interpretation="Merged PRs only; retrospective classification, no causal AI attribution.")
    return df, audit


def split_data(df, strategy, seed, validation_fraction=.15, test_fraction=.15):
    """No row shuffle temporal split: equal merge days stay together."""
    if strategy == "repository":
        if df.repository_id.nunique() < 7:
            raise ValueError("Repository split needs at least 7 repositories.")
        split = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        remaining, test = next(split.split(df, groups=df.repository_id))
        inner = GroupShuffleSplit(n_splits=1, test_size=validation_fraction / (1 - test_fraction), random_state=seed + 1)
        train_rel, val_rel = next(inner.split(df.iloc[remaining], groups=df.iloc[remaining].repository_id))
        indices = (remaining[train_rel], remaining[val_rel], test)
    else:
        if "merge_day" not in df or df.merge_day.isna().any():
            raise ValueError("merge-time split requires a valid merge_day for every PR.")
        days = np.sort(df.merge_day.unique())
        if len(days) < 7:
            raise ValueError("merge-time split needs at least 7 distinct merge days.")
        i = max(1, int(len(days) * (1 - test_fraction - validation_fraction)))
        j = max(i + 1, int(len(days) * (1 - test_fraction)))
        indices = (np.flatnonzero(df.merge_day.lt(days[i])),
                   np.flatnonzero(df.merge_day.ge(days[i]) & df.merge_day.lt(days[j])),
                   np.flatnonzero(df.merge_day.ge(days[j])))
    if any(len(x) == 0 for x in indices):
        raise ValueError("A split is empty. Use more data or another split strategy.")
    return dict(zip(["train", "validation", "test"], indices))


def class_thresholds(train_minutes, explicit=None):
    cuts = np.asarray(explicit if explicit else np.quantile(train_minutes, [1 / 3, 2 / 3]), dtype=float)
    if cuts.shape != (2,) or not np.isfinite(cuts).all() or not 0 <= cuts[0] < cuts[1]:
        raise ValueError("Class thresholds must satisfy 0 <= fast < slow. Tied training quantiles require --thresholds.")
    return cuts


def labels(minutes, cuts):
    # fast <= first threshold; medium <= second; slow > second.
    return np.searchsorted(cuts, np.asarray(minutes), side="left").astype("int32")
