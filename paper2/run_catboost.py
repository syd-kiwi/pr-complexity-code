#!/usr/bin/env python3
"""Run the Paper 2 CatBoost comparison directly from the cleaned CSV."""

import argparse
import ast
import json
import logging
from functools import lru_cache
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

KEYS = ["repository_id", "pull_request_id"]
TARGET = "minutes_to_merge"
FLAGS = ["is_open_source", "is_code_pr", "had_copilot_seat", "has_title", "has_body", "short_body"]
DESCRIPTION = [
    "files_added", "files_deleted", "files_modified", "files_touched", "lines_modified",
    "code_lines_modified", "config_lines_modified", "is_code_pr", "language_count",
    "title_word_count", "body_word_count", "has_title", "has_body", "short_body",
    "first_language",
]
FEATURE_SETS = {"description": DESCRIPTION, "review": DESCRIPTION + ["num_reviewers", "num_review_comments"]}
CATEGORICAL = ["first_language"]
CLASSES = ["fast", "medium", "slow"]
PARTITIONS = ["train", "validation", "test"]
REFERENCE_COUNTS = [(951_569, 41_631), (193_978, 8_921), (192_854, 8_922)]
REFERENCE_CUTS = [35, 826]
XGBOOST_F1 = {"description": 0.4298, "review": 0.5076}
XGBOOST_REVIEW_MEDIUM_RECALL = 0.260
# Six validation trials per feature set, matching the published trial budget.
TRIALS = [(4, .03), (4, .08), (6, .03), (6, .08), (8, .03), (8, .08)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/cleaned_public.csv"), help="Cleaned CSV or CSV.GZ")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--device", default="cuda:0", help="cpu or cuda:N")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--trials", type=int, default=len(TRIALS), choices=range(1, len(TRIALS) + 1))
    parser.add_argument("--trees", type=int, default=1_200)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--reference-splits", type=Path, help="Optional splits.csv.gz from the original a100_run01")
    args = parser.parse_args()
    if args.device != "cpu" and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        parser.error("--device must be cpu or cuda:N")
    if min(args.threads, args.chunk_size, args.trees, args.early_stopping) < 1 or not 0 <= args.seed < 2**32 - 1:
        parser.error("threads, chunk-size, trees, and early-stopping must be positive and seed must be in [0, 2^32-2]")
    if not args.input.is_file():
        parser.error(f"input CSV does not exist: {args.input}")
    if args.reference_splits and not args.reference_splits.is_file():
        parser.error(f"reference splits file does not exist: {args.reference_splits}")
    if args.output.exists():
        parser.error(f"choose a new output directory: {args.output}")
    return args


@lru_cache(maxsize=16_384)
def language_features(text):
    for parse in (json.loads, ast.literal_eval):
        try:
            values = parse(text)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            continue
        if isinstance(values, (list, tuple)):
            entries = tuple(str(item) for item in values if item is not None and str(item).strip())
            return len(set(entries)), entries[0] if entries else "__unknown__"
    raise ValueError(f"languages_list must contain JSON or Python lists, got {text[:100]!r}")


def read_data(path, chunk_size):
    missing_tokens = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0,
                      "yes": 1, "no": 0, "": np.nan}
    numeric = [c for c in FEATURE_SETS["review"] if c not in FLAGS + CATEGORICAL + ["language_count"]]
    required = KEYS + numeric + FLAGS + [TARGET, "languages_list"]
    available = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(required) - set(available))
    if missing:
        raise ValueError(f"Missing columns in cleaned CSV: {', '.join(missing)}")
    optional = [c for c in ("user_id", "merge_day", "merge_month") if c in available]
    read_columns = required + optional
    chunks = []
    audit = {"input_rows": 0, "invalid_key_rows": 0}

    # Project out the raw title and body. Keep the other baseline fields for duplicate checking.
    for part in pd.read_csv(path, usecols=read_columns, dtype="string", chunksize=chunk_size):
        audit["input_rows"] += len(part)
        for c in KEYS + (["user_id"] if "user_id" in optional else []):
            part[c] = part[c].str.strip().replace("", pd.NA)
        valid_keys = part[KEYS].notna().all(axis=1)
        audit["invalid_key_rows"] += int((~valid_keys).sum())
        part = part.loc[valid_keys].copy()
        for c in FLAGS:
            flag = part[c].str.strip().str.lower()
            if (flag.notna() & ~flag.isin(missing_tokens)).any():
                raise ValueError(f"Unexpected boolean in {c}. Expected true/false, yes/no, or 1/0.")
            part[c] = flag.map(missing_tokens).astype("float64")
        for c in numeric + [TARGET]:
            value = pd.to_numeric(part[c], errors="coerce").astype("float64")
            part[c] = value.where(np.isfinite(value) & value.ge(0))
        parsed = [language_features(s) if pd.notna(s) and s else (np.nan, "__unknown__")
                  for s in part.pop("languages_list").str.strip()]
        part["language_count"] = [count for count, _ in parsed]
        part["first_language"] = [first for _, first in parsed]
        chunks.append(part)

    if not chunks:
        raise ValueError("The input CSV has no rows.")
    data = pd.concat(chunks, ignore_index=True)
    del chunks
    duplicate_key_rows = data.duplicated(KEYS, keep=False)
    if data.loc[duplicate_key_rows].drop_duplicates().duplicated(KEYS).any():
        raise ValueError("Conflicting rows share a repository and pull request ID.")
    deduplicated = data.drop_duplicates(KEYS)
    audit["duplicates_removed"] = len(data) - len(deduplicated)
    valid_target = deduplicated[TARGET].notna()
    audit["invalid_target_rows"] = int((~valid_target).sum())
    public = deduplicated["is_open_source"].eq(1)
    audit["non_public_rows_with_valid_target"] = int((valid_target & ~public).sum())
    data = deduplicated.loc[valid_target & public, KEYS + FEATURE_SETS["review"] + [TARGET]].reset_index(drop=True)
    for col in (name for name in data.columns if name not in KEYS + CATEGORICAL):
        data[col] = data[col].astype("float32")
    audit["retained_prs"] = len(data)
    audit["retained_repositories"] = int(data.repository_id.nunique())
    return data, audit


def build_splits(data, seed):
    if data.repository_id.nunique() < 7:
        raise ValueError("The repository split requires at least seven repositories.")
    outer = GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed)
    remaining, test = next(outer.split(data, groups=data.repository_id))
    inner = GroupShuffleSplit(n_splits=1, test_size=.15 / (1 - .15), random_state=seed + 1)
    train_rel, val_rel = next(inner.split(remaining, groups=data.repository_id.iloc[remaining]))
    partitions = dict(zip(PARTITIONS, (remaining[train_rel], remaining[val_rel], test)))
    cuts = np.quantile(data[TARGET].iloc[partitions["train"]], [1 / 3, 2 / 3])
    if not 0 <= cuts[0] < cuts[1]:
        raise ValueError("Training merge-duration quantiles must produce two distinct cutoffs.")
    labels = np.searchsorted(cuts, data[TARGET], side="left").astype("int32")
    if any(set(labels[indices]) != {0, 1, 2} for indices in partitions.values()):
        raise ValueError("Each partition needs all three duration classes.")
    summaries = {name: {"rows": len(indices), "repositories": int(data.repository_id.iloc[indices].nunique()),
                        "class_counts": np.bincount(labels[indices], minlength=3).tolist()}
                 for name, indices in partitions.items()}
    return partitions, labels, cuts.tolist(), summaries


def check_reference(path, data, labels, partitions):
    reference = pd.read_csv(path, usecols=KEYS + ["split", "class"], dtype={c: "string" for c in KEYS + ["split"]})
    if len(reference) != len(data) or reference.duplicated(KEYS).any():
        raise ValueError("The reference split has different rows or nonunique PR keys.")
    split_names = np.empty(len(data), dtype=object)
    for name, indices in partitions.items():
        split_names[indices] = name
    aligned = data[KEYS].merge(reference, on=KEYS, how="left", sort=False, validate="one_to_one")
    if not aligned["split"].notna().all() or not np.array_equal(aligned["split"], split_names):
        raise ValueError("PR keys or repository split membership differ from the reference.")
    if not np.array_equal(pd.to_numeric(aligned["class"], errors="coerce"), labels):
        raise ValueError("The duration class labels differ from the reference.")


def evaluate(truth, predicted):
    matrix = confusion_matrix(truth, predicted, labels=[0, 1, 2])
    return {
        "macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        **{f"recall_{name}": float(matrix[i, i] / matrix[i].sum()) for i, name in enumerate(CLASSES)},
        "fast_slow_error_rate": float((matrix[0, 2] + matrix[2, 0]) / matrix.sum()),
    }, matrix


def run_experiment(args):
    try:
        import catboost
    except ImportError as exc:
        raise SystemExit("Install the CatBoost run dependencies listed in paper2/README.md.") from exc

    data, audit = read_data(args.input, args.chunk_size)
    partitions, labels, cuts, summaries = build_splits(data, args.seed)
    counts_match = ([(s["rows"], s["repositories"]) for s in summaries.values()] == REFERENCE_COUNTS
                    and cuts == REFERENCE_CUTS and args.seed == 42)
    if args.reference_splits:
        check_reference(args.reference_splits, data, labels, partitions)
    logging.info("Cohort: %s. Duration cutoffs: %s.", audit, cuts)
    if not counts_match:
        logging.warning("This cohort or split differs from the published A100 run. Baseline deltas will be omitted.")

    args.output.mkdir(parents=True)
    manifest = {
        "arguments": vars(args), "audit": audit, "split_summary": summaries, "cutoffs_minutes": cuts,
        "class_boundaries": "fast <= t1; t1 < medium <= t2; slow > t2",
        "feature_sets": FEATURE_SETS, "matches_paper_counts_and_cutoffs": counts_match,
        "reference_keys_membership_and_labels_checked": bool(args.reference_splits),
        "versions": {"python": platform.python_version(), "catboost": catboost.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "evaluation": "Exploratory reuse of the reported holdout; retrospective merged PR classification.",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, default=str, indent=2) + "\n", encoding="utf-8")
    task = {"task_type": "CPU"} if args.device == "cpu" else {"task_type": "GPU", "devices": args.device.split(":")[1]}
    trials, scores = [], []
    for name, columns in FEATURE_SETS.items():
        features = data[columns].copy()
        for col in (c for c in columns if c not in FLAGS + CATEGORICAL):
            features[col] = np.log1p(features[col]).astype("float32")
        pools = {key: catboost.Pool(features.iloc[indices], label=labels[indices], cat_features=CATEGORICAL)
                 for key, indices in partitions.items()}
        del features
        best_score, best_model = -1., None
        for depth, learning_rate in TRIALS[:args.trials]:
            started = perf_counter()
            model = catboost.CatBoostClassifier(
                loss_function="MultiClass", eval_metric="MultiClass", iterations=args.trees,
                depth=depth, learning_rate=learning_rate, random_seed=args.seed, thread_count=args.threads,
                allow_writing_files=False, **task,
            )
            model.fit(pools["train"], eval_set=pools["validation"], use_best_model=True,
                      early_stopping_rounds=args.early_stopping, verbose=False)
            pred = np.asarray(model.predict(pools["validation"])).ravel().astype("int32")
            score = f1_score(labels[partitions["validation"]], pred, labels=[0, 1, 2], average="macro", zero_division=0)
            trials.append({"feature_set": name, "depth": depth, "learning_rate": learning_rate,
                           "trees_kept": model.tree_count_, "validation_macro_f1": score,
                           "seconds": perf_counter() - started})
            logging.info("%s depth=%d learning_rate=%.2f validation macro F1=%.4f", name, depth, learning_rate, score)
            if score > best_score:
                best_score, best_model = score, model
        predicted = np.asarray(best_model.predict(pools["test"])).ravel().astype("int32")
        metrics, matrix = evaluate(labels[partitions["test"]], predicted)
        row = {"feature_set": name, "validation_macro_f1": best_score, **metrics}
        if counts_match:
            row["reported_xgboost_macro_f1"] = XGBOOST_F1[name]
            row["delta_vs_reported_xgboost_macro_f1"] = metrics["macro_f1"] - XGBOOST_F1[name]
            if name == "review":
                row["delta_vs_reported_xgboost_medium_recall"] = metrics["recall_medium"] - XGBOOST_REVIEW_MEDIUM_RECALL
        scores.append(row)
        pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(args.output / f"{name}_confusion.csv")
        best_model.save_model(str(args.output / f"{name}_catboost.cbm"))
        del pools, model, best_model

    pd.DataFrame(trials).to_csv(args.output / "validation_trials.csv", index=False)
    pd.DataFrame(scores).to_csv(args.output / "metrics.csv", index=False)
    report = [
        "# CatBoost merge-duration comparison", "",
        f"Public merged PRs: {audit['retained_prs']:,}. Test PRs: {summaries['test']['rows']:,}.",
        f"Train-derived duration cutoffs: {cuts[0]:g} and {cuts[1]:g} minutes.", "",
        "| Features | CatBoost test macro F1 | Accuracy | Medium recall | Reported XGBoost macro F1 | Difference |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scores:
        baseline = f"{row['reported_xgboost_macro_f1']:.4f}" if counts_match else "NA"
        delta = f"{row['delta_vs_reported_xgboost_macro_f1']:+.4f}" if counts_match else "NA"
        report.append(f"| {row['feature_set']} | {row['macro_f1']:.4f} | {row['accuracy']:.2%} | {row['recall_medium']:.2%} | {baseline} | {delta} |")
    if counts_match:
        detail = ("PR keys, split membership, and duration classes match the supplied reference split."
                  if args.reference_splits else
                  "Counts and cutoffs match the paper. Exact test PR membership was not verified.")
        report.extend(["", detail,
                       f"The reported XGBoost review model had {XGBOOST_REVIEW_MEDIUM_RECALL:.1%} medium-duration recall."])
    report.extend(["", "Results are retrospective. The review feature set includes final reviewer and comment counts. "
                   "A comparison on the previously used test set is exploratory."])
    (args.output / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved outputs to {args.output.resolve()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_experiment(parse_args())
