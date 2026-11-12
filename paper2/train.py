#!/usr/bin/env python3
"""RQ2/RQ3: GPU XGBoost classification with validation tuning and paired ablations."""
from __future__ import annotations

import argparse
import itertools
import json
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

from common import (CLASSES, FEATURE_SETS, TARGET, dump_json, fingerprint, fresh_output,
                    class_thresholds, labels, setup_logging, split_data, versions, LOG)


def model_input(df, feature_set):
    columns = FEATURE_SETS[feature_set]
    X = df[columns + ["first_language"]].copy()
    # Monotonic transform for skewed counts; never transform flags.
    flags = {"is_code_pr", "has_title", "has_body", "short_body", "had_copilot_seat"}
    for c in columns:
        X[c] = X[c].astype("float32") if c in flags else np.log1p(X[c]).astype("float32")
    return X


def preprocessor(columns):
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler())
    return ColumnTransformer([
        ("numeric", numeric, columns),
        ("language", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                    max_categories=64, sparse_output=False, dtype=np.float32), ["first_language"]),
    ], verbose_feature_names_out=True)


def metrics(y, pred, probabilities):
    result = {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
              "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
              "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
              "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])), "roc_auc_ovr_macro": None}
    if len(np.unique(y)) == 3:
        result["roc_auc_ovr_macro"] = float(roc_auc_score(y, probabilities, labels=[0, 1, 2], multi_class="ovr", average="macro"))
    report = classification_report(y, pred, labels=[0, 1, 2], target_names=CLASSES, output_dict=True, zero_division=0)
    result.update({f"recall_{name}": float(report[name]["recall"]) for name in CLASSES})
    return result, report


def check_gpu(device, threads):
    """XGBoost otherwise warns and silently falls back to CPU on some builds."""
    if device == "cpu":
        return
    if not xgb.build_info().get("USE_CUDA", False):
        raise RuntimeError("This XGBoost build lacks CUDA. Install the CUDA-enabled xgboost wheel.")
    probe = xgb.XGBClassifier(device=device, tree_method="hist", n_estimators=1, n_jobs=threads)
    probe.fit(np.array([[0], [1], [2], [3]], dtype=np.float32), [0, 0, 1, 1])
    actual = json.loads(probe.get_booster().save_config())["learner"]["generic_param"]["device"]
    if not actual.startswith("cuda"):
        raise RuntimeError(f"Requested {device}, but XGBoost selected {actual}. Check NVIDIA drivers/GPU visibility or use --device cpu explicitly.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared Parquet file")
    parser.add_argument("--output", required=True, help="New run directory")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 (default), cuda:1, or cpu")
    parser.add_argument("--split", choices=["repository", "merge-time"], default="repository")
    parser.add_argument("--feature-sets", nargs="+", choices=list(FEATURE_SETS), default=list(FEATURE_SETS))
    parser.add_argument("--thresholds", nargs=2, type=float, metavar=("FAST_MINUTES", "SLOW_MINUTES"))
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--trees", type=int, default=1200)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutation-rows", type=int, default=10000, help="Validation rows for permutation importance; 0 disables")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    args = parser.parse_args()
    if any(v < 1 for v in [args.trials, args.trees, args.early_stopping, args.threads]) or args.permutation_rows < 0:
        parser.error("trials, trees, early-stopping, threads must be positive; permutation-rows must be nonnegative")
    if args.trials > 36:
        parser.error("At most 36 distinct hyperparameter trials are supported")
    if len(set(args.feature_sets)) != len(args.feature_sets):
        parser.error("feature-sets must not contain duplicates")
    if not (args.device == "cpu" or args.device == "cuda" or (args.device.startswith("cuda:") and args.device[5:].isdigit())):
        parser.error("device must be cpu, cuda, or cuda:N")
    setup_logging()
    check_gpu(args.device, args.threads)
    df = pd.read_parquet(args.input)
    partitions = split_data(df, args.split, args.seed)
    cuts = class_thresholds(df.iloc[partitions["train"]][TARGET], args.thresholds)
    y = labels(df[TARGET], cuts)
    train, val, test = [partitions[k] for k in ["train", "validation", "test"]]
    if set(y[train]) != {0, 1, 2} or set(y[val]) != {0, 1, 2}:
        raise ValueError("Training and validation must each contain fast, medium, and slow PRs. Change thresholds or supply more data.")
    out = fresh_output(args.output)
    split_labels = np.empty(len(df), dtype=object)
    for name, ix in partitions.items():
        split_labels[ix] = name
    split_table = df[["repository_id", "pull_request_id"]].copy()
    split_table["split"] = split_labels
    split_table["class"] = y
    split_table.to_csv(out / "splits.csv.gz", index=False)
    manifest = {"arguments": vars(args), "versions": versions(), "source": fingerprint(args.input),
                "thresholds_minutes": cuts.tolist(), "class_names": CLASSES,
                "class_boundaries": "fast <= t1; t1 < medium <= t2; slow > t2",
                "feature_sets": {k: FEATURE_SETS[k] + ["first_language"] for k in args.feature_sets},
                "feature_timing": "unverified_final_snapshot",
                "excluded_predictors": ["minutes_to_merge", "complexity_score", "merge_day", "merge_month", "repository_id", "pull_request_id", "user_id", "is_open_source"],
                "split_summary": {name: {"rows": len(ix), "repositories": int(df.iloc[ix].repository_id.nunique()),
                                         "class_counts": np.bincount(y[ix], minlength=3).tolist()}
                                  for name, ix in partitions.items()},
                "limitations": ["Merged PRs only; this model does not predict acceptance or outcomes for all opened PRs.",
                                "Feature snapshots are unverified; results are retrospective, not opening-time performance.",
                                "Seat status means Copilot access, not confirmed use or AI authorship.",
                                "merge-time split orders by final merge day, not opening time; never call it a prospective evaluation.",
                                "Repository split holds out repositories, but authors can occur in multiple partitions.",
                                "Review activity is a completed-process observation. Prediction/importance is not causation.",
                                "No review-stage timestamps are available; total merge time cannot isolate review waiting time."]}
    dump_json(out / "run_manifest.json", manifest)
    LOG.info("Splits: %s; thresholds (minutes): %s", {k: len(v) for k, v in partitions.items()}, cuts)
    rng = np.random.default_rng(args.seed)
    candidates = list(itertools.product([3, 5, 7], [.03, .08], [1, 5, 15], [.8, 1.]))
    selected_candidates = rng.choice(len(candidates), size=args.trials, replace=False)
    weights = None
    if args.class_weight == "balanced":
        counts = np.bincount(y[train], minlength=3)
        weights = (len(train) / (3 * counts))[y[train]]
    rows, trials, subgroups = [], [], []
    validation_scores = []
    prediction_table = df.iloc[test][["repository_id", "pull_request_id", "had_copilot_seat", "is_open_source"]].copy()
    prediction_table["true_class"] = y[test]

    def record(name, feature_set, model, Xt, Xv):
        probabilities = model.predict_proba(Xt)
        pred = probabilities.argmax(axis=1)
        score, report = metrics(y[test], pred, probabilities)
        validation_f1 = float(f1_score(y[val], model.predict(Xv), labels=[0, 1, 2], average="macro", zero_division=0))
        rows.append(dict(model=name, feature_set=feature_set, validation_macro_f1=validation_f1, **score))
        validation_scores.append((validation_f1, feature_set, name))
        stem = f"{feature_set}_{name}"
        dump_json(out / f"{stem}_classification_report.json", report)
        pd.DataFrame(confusion_matrix(y[test], pred, labels=[0, 1, 2]), index=CLASSES, columns=CLASSES).to_csv(out / f"{stem}_confusion.csv")
        prediction_table[f"{stem}_class"] = pred
        for j, cls in enumerate(CLASSES):
            prediction_table[f"{stem}_p_{cls}"] = probabilities[:, j]
        for grouping in ["had_copilot_seat", "is_open_source"]:
            for value in [0., 1., "unknown"]:
                series = df.iloc[test][grouping]
                mask = (series.isna() if value == "unknown" else series.eq(value)).to_numpy()
                if mask.any():
                    group_score, _ = metrics(y[test][mask], pred[mask], probabilities[mask])
                    subgroups.append(dict(model=name, feature_set=feature_set, grouping=grouping, value=value, **group_score))
        LOG.info("%s test macro F1 %.4f (validation %.4f)", stem, score["macro_f1"], validation_f1)

    # Same rows for every model. A majority baseline is fit only once.
    dummy = DummyClassifier(strategy="prior").fit(np.zeros((len(train), 1)), y[train])
    record("majority", "none", dummy, np.zeros((len(test), 1)), np.zeros((len(val), 1)))
    for feature_set in tqdm(args.feature_sets, desc="Feature sets"):
        LOG.info("Training feature set: %s", feature_set)
        raw = model_input(df, feature_set)
        transform = preprocessor(FEATURE_SETS[feature_set])
        Xtrain = transform.fit_transform(raw.iloc[train]).astype("float32")
        Xval = transform.transform(raw.iloc[val]).astype("float32")
        Xtest = transform.transform(raw.iloc[test]).astype("float32")
        features = transform.get_feature_names_out().tolist()
        dump_json(out / f"{feature_set}_transformed_features.json", features)
        # Fixed logistic baseline, with identical data and preprocessing.
        linear = LogisticRegression(max_iter=1000, class_weight="balanced" if args.class_weight == "balanced" else None, random_state=args.seed)
        linear.fit(Xtrain, y[train])
        record("logistic", feature_set, linear, Xtest, Xval)
        joblib.dump({"transform": transform, "model": linear, "feature_set": feature_set,
                     "thresholds": cuts, "classes": CLASSES}, out / f"{feature_set}_logistic.joblib")
        best, best_f1, best_params = None, -1., None
        for trial, candidate in enumerate(tqdm(selected_candidates, desc=feature_set, leave=False)):
            depth, rate, child, subsample = candidates[candidate]
            params = dict(max_depth=depth, learning_rate=rate, min_child_weight=child, subsample=subsample)
            model = xgb.XGBClassifier(objective="multi:softprob", num_class=3, tree_method="hist", device=args.device,
                                      n_estimators=args.trees, early_stopping_rounds=args.early_stopping,
                                      eval_metric="mlogloss", random_state=args.seed, n_jobs=args.threads,
                                      colsample_bytree=.9, reg_lambda=5., max_bin=256, **params)
            started = time.perf_counter()
            model.fit(Xtrain, y[train], sample_weight=weights, eval_set=[(Xval, y[val])], verbose=False)
            actual_device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
            if args.device.startswith("cuda") and not actual_device.startswith("cuda"):
                raise RuntimeError("XGBoost fell back to CPU. Run aborted instead of reporting GPU training.")
            score = float(f1_score(y[val], model.predict(Xval), labels=[0, 1, 2], average="macro", zero_division=0))
            trials.append(dict(feature_set=feature_set, trial=trial, validation_macro_f1=score,
                               best_iteration=model.best_iteration, seconds=time.perf_counter() - started,
                               actual_device=actual_device, **params))
            if score > best_f1:
                best, best_f1, best_params = model, score, params
            pd.DataFrame(trials).to_csv(out / "validation_trials.csv", index=False)
        # Selection uses validation only, and the held-out test is evaluated once per selected model.
        dump_json(out / f"{feature_set}_selected.json", dict(validation_macro_f1=best_f1, parameters=best_params,
                                                          best_iteration=best.best_iteration))
        best.save_model(out / f"{feature_set}_xgboost.ubj")
        joblib.dump({"transform": transform, "model": best, "feature_set": feature_set,
                     "thresholds": cuts, "classes": CLASSES}, out / f"{feature_set}_xgboost.joblib")
        record("xgboost", feature_set, best, Xtest, Xval)
        pd.DataFrame({"feature": features, "gain_importance": best.feature_importances_}).sort_values("gain_importance", ascending=False).to_csv(out / f"{feature_set}_gain_importance.csv", index=False)
        if args.permutation_rows:
            sample = np.random.default_rng(args.seed).choice(len(val), min(args.permutation_rows, len(val)), replace=False)
            # Permute raw columns before transformation: missing indicators stay with their source feature.
            full_pipeline = make_pipeline(transform, best)
            importance = permutation_importance(full_pipeline, raw.iloc[val].iloc[sample], y[val][sample],
                                                n_repeats=3, random_state=args.seed, scoring="f1_macro", n_jobs=1)
            pd.DataFrame({"feature": raw.columns, "validation_f1_drop_mean": importance.importances_mean,
                          "validation_f1_drop_std": importance.importances_std}).sort_values("validation_f1_drop_mean", ascending=False).to_csv(out / f"{feature_set}_permutation_importance.csv", index=False)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        pd.DataFrame(subgroups).to_csv(out / "subgroup_metrics.csv", index=False)
    prediction_table.to_csv(out / "test_predictions.csv.gz", index=False)
    best_validation = max(validation_scores, key=lambda x: x[0])
    dump_json(out / "selected_on_validation.json", dict(validation_macro_f1=best_validation[0],
                                                       feature_set=best_validation[1], model=best_validation[2]))
    # Paired feature ablations: all comparisons use the exact same test rows.
    table = pd.DataFrame(rows)
    comparisons = []
    for before, after in [("structural", "description"), ("description", "copilot"),
                          ("description", "review"), ("review", "review_copilot"), ("copilot", "review_copilot")]:
        for model_name in ["xgboost", "logistic"]:
            lookup = table.loc[table.model.eq(model_name)].set_index("feature_set")
            if before in lookup.index and after in lookup.index:
                comparisons.append(dict(model=model_name, before=before, after=after,
                                        test_macro_f1_difference=float(lookup.loc[after, "macro_f1"] - lookup.loc[before, "macro_f1"]),
                                        interpretation="Predictive difference; no causal inference or significance claim"))
    pd.DataFrame(comparisons, columns=["model", "before", "after", "test_macro_f1_difference", "interpretation"]).to_csv(out / "feature_comparisons.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot = table.loc[table.model.ne("majority")].pivot(index="feature_set", columns="model", values="macro_f1")
    ax = plot.plot.bar(figsize=(8, 4.5), rot=20)
    baseline = table.loc[table.model.eq("majority"), "macro_f1"].iloc[0]
    ax.axhline(baseline, color="gray", linestyle="--", label="Majority baseline")
    ax.set(ylabel="Test macro F1", xlabel="Feature set", ylim=(0, 1))
    ax.legend()
    ax.figure.tight_layout()
    ax.figure.savefig(out / "model_comparison.png", dpi=180)
    ax.figure.savefig(out / "model_comparison.pdf")
    plt.close(ax.figure)
    LOG.info("Finished. Results and fitted models: %s", out)


if __name__ == "__main__":
    main()
