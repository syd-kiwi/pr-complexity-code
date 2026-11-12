#!/usr/bin/env python3
"""Enhanced experiments: diagnostics, actual text, ordinal models, and regression.

Default: compare models using training/validation only. Test labels are reserved
unless --evaluate-test or a separate --external-test is explicitly supplied.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from common import (CLASSES, FEATURE_SETS, KEYS, TARGET, class_thresholds, dump_json,
                    fingerprint, fresh_output, labels, setup_logging, versions, LOG)
from modeling import (OrdinalModel, TextEncoder, check_external_overlap, check_prepared,
                      classification_metrics, duration_predictions, partitions, regression_metrics)
from train import check_gpu, model_input, preprocessor


def score_predictions(task, minutes, y, model, X, cuts):
    if task == "regression":
        log_pred = model.predict(X)
        durations = duration_predictions(log_pred)
        pred = labels(durations, cuts)
        return {**classification_metrics(y, pred), **regression_metrics(minutes, log_pred)}, pred, None, durations
    probabilities = model.predict_proba(X)
    pred = model.predict(X) if task == "ordinal" else probabilities.argmax(axis=1)
    return classification_metrics(y, pred, probabilities), pred, probabilities, None


def save_history(models, task, stem, out, plot=False):
    rows = []
    for stage, model in enumerate(models):
        history = model.evals_result()
        for dataset, metrics in history.items():
            for metric, values in metrics.items():
                rows.extend(dict(stage=stage, split="training_sample" if dataset == "validation_0" else "validation",
                                 metric=metric, iteration=i + 1, value=v) for i, v in enumerate(values))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / f"{stem}_learning_curves.csv", index=False)
    if plot:
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), squeeze=False)
        for stage, ax in enumerate(axes[0]):
            data = frame.loc[frame.stage.eq(stage)]
            for (split, metric), group in data.groupby(["split", "metric"]):
                ax.plot(group.iteration, group.value, label=f"{split}: {metric}")
            ax.axvline(models[stage].best_iteration + 1, color="gray", linestyle="--", label="Selected iteration")
            ax.set(xlabel="Boosting round", ylabel="Loss", title=f"{task}, stage {stage + 1}")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{stem}_learning_curves.png", dpi=150)
        plt.close(fig)


def fit_candidate(task, params, args, Xtrain, Xdiag, Xval, ytrain, ydiag, yval, log_train, log_diag, log_val, weights):
    base = dict(tree_method="hist", device=args.device, n_estimators=args.trees,
                early_stopping_rounds=args.early_stopping, n_jobs=args.threads,
                random_state=args.seed, colsample_bytree=.9, reg_lambda=5., max_bin=256, **params)
    if task == "regression":
        model = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="mae", **base)
        # Duration regression is unweighted; classification weights do not apply.
        model.fit(Xtrain, log_train, eval_set=[(Xdiag, log_diag), (Xval, log_val)], verbose=False)
        models = [model]
    elif task == "multiclass":
        model = xgb.XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", **base)
        model.fit(Xtrain, ytrain, sample_weight=weights,
                  eval_set=[(Xdiag, ydiag), (Xval, yval)], verbose=False)
        models = [model]
    else:
        models = []
        for threshold in [0, 1]:
            model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **base)
            model.fit(Xtrain, (ytrain > threshold).astype(int), sample_weight=weights,
                      eval_set=[(Xdiag, (ydiag > threshold).astype(int)), (Xval, (yval > threshold).astype(int))], verbose=False)
            models.append(model)
        model = OrdinalModel(models)
    devices = [json.loads(m.get_booster().save_config())["learner"]["generic_param"]["device"] for m in models]
    if args.device.startswith("cuda") and any(not d.startswith("cuda") for d in devices):
        raise RuntimeError("GPU request fell back to CPU; stopping the run")
    return model, models, devices


def diagnostic_tables(df, y, pred, cuts, stem, split, out, durations=None):
    table = df[KEYS + ["had_copilot_seat", "is_open_source"]].copy()
    table["true_class"], table["predicted_class"] = y, pred
    table["correct"] = (y == pred).astype(int)
    table["absolute_class_error"] = np.abs(y - pred)
    table["severe_error"] = (np.abs(y - pred) == 2).astype(int)
    if durations is not None:
        table["absolute_error_minutes"] = np.abs(df[TARGET].to_numpy() - durations)
    agg = {"n": ("correct", "size"), "accuracy": ("correct", "mean"),
           "mean_absolute_class_error": ("absolute_class_error", "mean"), "severe_error_rate": ("severe_error", "mean")}
    if durations is not None:
        agg["mae_minutes"] = ("absolute_error_minutes", "mean")
    table.groupby("repository_id", observed=True).agg(**agg).to_csv(out / f"{stem}_{split}_by_repository.csv")
    for group in ["had_copilot_seat", "is_open_source"]:
        table.groupby(group, dropna=False, observed=True).agg(**agg).to_csv(out / f"{stem}_{split}_by_{group}.csv")
    pd.DataFrame(confusion_matrix(y, pred, labels=[0, 1, 2]), index=CLASSES, columns=CLASSES).to_csv(out / f"{stem}_{split}_confusion.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared Parquet; use --include-text in preparation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["repository", "merge-time", "known-repository"], default="repository")
    parser.add_argument("--split-file", help="Reuse original splits.csv.gz; exact PR keys must match")
    parser.add_argument("--feature-sets", nargs="+", choices=list(FEATURE_SETS), default=["description", "copilot", "review", "review_copilot"])
    parser.add_argument("--text-methods", nargs="+", choices=["none", "tfidf", "embeddings"], default=["none", "tfidf"])
    parser.add_argument("--tasks", nargs="+", choices=["multiclass", "ordinal", "regression"], default=["multiclass", "ordinal", "regression"])
    parser.add_argument("--thresholds", nargs=2, type=float)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--trees", type=int, default=1600)
    parser.add_argument("--early-stopping", type=int, default=75)
    parser.add_argument("--diagnostic-rows", type=int, default=50000, help="Training rows used for curves/scores; 0 means all training rows")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--text-dimensions", type=int, default=128)
    parser.add_argument("--vocabulary", type=int, default=30000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-text-chars", type=int, default=12000)
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--encoder-revision")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--skip-linear", action="store_true", help="Skip logistic and ridge baselines; majority/median baselines remain")
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--evaluate-test", action="store_true", help="Also evaluate reserved internal test; exploratory if previously inspected")
    test_group.add_argument("--external-test", help="Separate prepared Parquet with nonoverlapping PR keys")
    args = parser.parse_args()
    positive = [args.threads, args.trials, args.trees, args.early_stopping, args.text_dimensions,
                args.vocabulary, args.min_df, args.max_text_chars, args.embedding_batch_size]
    if min(positive) < 1 or args.trials > 36 or args.diagnostic_rows < 0:
        parser.error("Positive parameters required, trials <= 36, diagnostic-rows >= 0")
    for values in [args.tasks, args.feature_sets, args.text_methods]:
        if len(set(values)) != len(values):
            parser.error("tasks, feature-sets, and text-methods cannot contain duplicates")
    if not (args.device in ["cpu", "cuda"] or (args.device.startswith("cuda:") and args.device[5:].isdigit())):
        parser.error("device must be cpu, cuda, or cuda:N")
    if args.split_file and args.split == "known-repository":
        parser.error("known-repository generates a separate protocol; omit --split-file")
    setup_logging()
    check_gpu(args.device, args.threads)
    df = pd.read_parquet(args.input).reset_index(drop=True)
    check_prepared(df, any(m != "none" for m in args.text_methods))
    parts = partitions(df, args.split, args.seed, args.split_file)
    train, val = parts["train"], parts["validation"]
    cuts = class_thresholds(df.iloc[train][TARGET], args.thresholds)
    y = labels(df[TARGET], cuts)
    if set(y[train]) != {0, 1, 2} or set(y[val]) != {0, 1, 2}:
        raise ValueError("Training and validation must each contain all 3 classes")
    rng = np.random.default_rng(args.seed)
    diag = train if args.diagnostic_rows == 0 else np.sort(rng.choice(train, min(args.diagnostic_rows, len(train)), replace=False))
    frames = {"training_sample": df.iloc[diag], "validation": df.iloc[val]}
    if args.evaluate_test:
        frames["test"] = df.iloc[parts["test"]]
    if args.external_test:
        external = pd.read_parquet(args.external_test).reset_index(drop=True)
        check_prepared(external, any(m != "none" for m in args.text_methods))
        check_external_overlap(df, external)
        frames["external_test"] = external
    out = fresh_output(args.output)
    manifest = {"arguments": vars(args), "versions": versions(), "source": fingerprint(args.input),
                "thresholds_minutes": cuts.tolist(), "classes": CLASSES,
                "feature_timing": "unverified_final_snapshot", "test_evaluation": "external" if args.external_test else "exploratory_internal" if args.evaluate_test else "not_evaluated",
                "selection_metrics": {"multiclass": "maximize validation macro F1", "ordinal": "minimize validation mean absolute class error", "regression": "minimize validation log MAE"},
                "training_diagnostics": {"rows": len(diag), "all_training_rows": len(train), "sampling": "seeded uniform without replacement"},
                "partitions": {name: {"rows": len(ix), "repositories": int(df.iloc[ix].repository_id.nunique()),
                                       "class_counts": np.bincount(y[ix], minlength=3).tolist()} for name, ix in parts.items()},
                "rows_excluded_by_split_protocol": len(df) - sum(len(ix) for ix in parts.values()),
                "limits": ["Merged PRs only. No causal Copilot effect, AI authorship, or acceptance prediction.",
                           "Final snapshots and merge-day splits are retrospective, not opening-time evaluations.",
                           "The original test has already informed development; reusing it is exploratory.",
                           "A new seed from the same dataset does not guarantee a fresh untouched final test.",
                           "No comparison with prior paper accuracy is valid without matching tasks, data, and splits."]}
    if args.split_file:
        manifest["split_source"] = fingerprint(args.split_file)
    if args.external_test:
        manifest["external_source"] = fingerprint(args.external_test)
        manifest["external_test_note"] = "PR nonoverlap is checked. The researcher must also establish independent collection and lack of prior tuning."
    dump_json(out / "experiment_manifest.json", manifest)
    membership = df[KEYS].copy()
    membership["split"] = "excluded"
    for name, ix in parts.items():
        membership.loc[ix, "split"] = name
    membership["class"] = y
    membership.to_csv(out / "splits.csv.gz", index=False)
    selection, rows, trial_rows = {}, [], []
    grid = list(itertools.product([3, 5, 7], [.03, .08], [1, 5, 15], [.8, 1.]))
    candidates = np.random.default_rng(args.seed).choice(len(grid), args.trials, replace=False)
    log_train = np.log1p(df.iloc[train][TARGET].to_numpy())
    weights = (len(train) / (3 * np.bincount(y[train], minlength=3)))[y[train]] if args.class_weight == "balanced" else None

    def record(stem, task, feature_set, text_method, model, matrices, model_name):
        scores = {}
        for split, frame in frames.items():
            split_y = labels(frame[TARGET], cuts)
            result, pred, probabilities, durations = score_predictions(task, frame[TARGET].to_numpy(), split_y, model, matrices[split], cuts)
            rows.append(dict(task=task, feature_set=feature_set, text_method=text_method, model=model_name, split=split, **result))
            scores[split] = result
            if split != "training_sample":
                diagnostic_tables(frame, split_y, pred, cuts, stem, split, out, durations)
                prediction = frame[KEYS].copy()
                prediction["true_class"], prediction["predicted_class"] = split_y, pred
                if probabilities is not None:
                    for j, name in enumerate(CLASSES):
                        prediction[f"p_{name}"] = probabilities[:, j]
                if durations is not None:
                    prediction["predicted_minutes"] = durations
                prediction.to_csv(out / f"{stem}_{split}_predictions.csv.gz", index=False)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        return scores

    # Simple baselines establish the scale of each task on the identical partitions.
    prior = np.bincount(y[train], minlength=3) / len(train)
    for split, frame in frames.items():
        true = labels(frame[TARGET], cuts)
        for name, value in [("majority", int(prior.argmax())), ("median_class", int(np.median(y[train])))]:
            rows.append(dict(task="classification_baseline", feature_set="none", text_method="none", model=name, split=split,
                             **classification_metrics(true, np.full(len(frame), value))))
        duration_score = regression_metrics(frame[TARGET].to_numpy(), np.full(len(frame), np.median(log_train)))
        rows.append(dict(task="regression_baseline", feature_set="none", text_method="none", model="training_median", split=split, **duration_score))
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

    for text_method in tqdm(args.text_methods, desc="Representations"):
        text_encoder, encoded = None, {}
        if text_method != "none":
            text_encoder = TextEncoder(text_method, args.vocabulary, args.text_dimensions, args.min_df,
                                       args.max_text_chars, args.seed, args.encoder, args.encoder_revision,
                                       args.device, args.embedding_batch_size)
            LOG.info("Fitting %s representation on %s training PRs", text_method, len(train))
            encoded["train"] = text_encoder.fit_transform(df.iloc[train])
            for split, frame in frames.items():
                encoded[split] = text_encoder.transform(frame)
            text_encoder.release_gpu()
            joblib.dump(text_encoder, out / f"text_encoder_{text_method}.joblib")
            dump_json(out / f"text_encoder_{text_method}.json", text_encoder.metadata())
        for feature_set in tqdm(args.feature_sets, desc=text_method, leave=False):
            raw = model_input(df.iloc[train], feature_set)
            transform = preprocessor(FEATURE_SETS[feature_set])
            Xtrain = transform.fit_transform(raw).astype("float32")
            matrices = {split: transform.transform(model_input(frame, feature_set)).astype("float32") for split, frame in frames.items()}
            features = transform.get_feature_names_out().tolist()
            if text_method != "none":
                Xtrain = np.column_stack([Xtrain, encoded["train"]])
                matrices = {split: np.column_stack([matrix, encoded[split]]) for split, matrix in matrices.items()}
                features += [f"{text_method}_{i}" for i in range(text_encoder.output_dimensions_)]
            Xdiag, Xval = matrices["training_sample"], matrices["validation"]
            for task in args.tasks:
                stem = f"{text_method}_{feature_set}_{task}"
                LOG.info("Fitting %s", stem)
                if not args.skip_linear and task != "ordinal":
                    baseline = (Ridge(alpha=10.) if task == "regression" else
                                LogisticRegression(max_iter=1000, class_weight="balanced" if args.class_weight == "balanced" else None, random_state=args.seed))
                    baseline.fit(Xtrain, log_train if task == "regression" else y[train])
                    name = "ridge" if task == "regression" else "logistic"
                    scores = record(f"{stem}_{name}", task, feature_set, text_method, baseline, matrices, name)
                    joblib.dump(dict(model=baseline, transform=transform, feature_set=feature_set, text_method=text_method,
                                     task=task, cuts=cuts, text_encoder_file=f"text_encoder_{text_method}.joblib" if text_encoder else None), out / f"{stem}_{name}.joblib")
                    criterion = scores["validation"]["log_mae" if task == "regression" else "macro_f1"]
                    utility = -criterion if task == "regression" else criterion
                    if task not in selection or utility > selection[task]["utility"]:
                        selection[task] = dict(utility=utility, bundle=f"{stem}_{name}.joblib", validation_score=criterion)
                best, best_models, best_utility, best_info = None, None, -np.inf, None
                for trial, candidate in enumerate(tqdm(candidates, desc=stem, leave=False)):
                    depth, rate, child, subsample = grid[candidate]
                    params = dict(max_depth=depth, learning_rate=rate, min_child_weight=child, subsample=subsample)
                    start = time.perf_counter()
                    model, models, devices = fit_candidate(task, params, args, Xtrain, Xdiag, Xval, y[train], y[diag], y[val],
                                                          log_train, np.log1p(df.iloc[diag][TARGET]), np.log1p(df.iloc[val][TARGET]), weights)
                    train_score, _, _, _ = score_predictions(task, df.iloc[diag][TARGET].to_numpy(), y[diag], model, Xdiag, cuts)
                    val_score, _, _, _ = score_predictions(task, df.iloc[val][TARGET].to_numpy(), y[val], model, Xval, cuts)
                    key = "log_mae" if task == "regression" else "mean_absolute_class_error" if task == "ordinal" else "macro_f1"
                    utility = val_score[key] if task == "multiclass" else -val_score[key]
                    trial_rows.append(dict(task=task, feature_set=feature_set, text_method=text_method, trial=trial,
                                           selection_metric=key, training_score=train_score[key], validation_score=val_score[key],
                                           training_macro_f1=train_score["macro_f1"], validation_macro_f1=val_score["macro_f1"],
                                           training_rows=len(diag), training_score_is_sample=len(diag) != len(train),
                                           best_iterations=json.dumps([m.best_iteration for m in models]),
                                           actual_device=json.dumps(devices), seconds=time.perf_counter() - start, **params))
                    save_history(models, task, f"{stem}_trial{trial}", out)
                    if utility > best_utility:
                        best, best_models, best_utility = model, models, utility
                        best_info = dict(trial=trial, validation_score=val_score[key], selection_metric=key, parameters=params)
                    pd.DataFrame(trial_rows).to_csv(out / "validation_trials.csv", index=False)
                bundle = dict(model=best, transform=transform, feature_set=feature_set, text_method=text_method,
                              task=task, cuts=cuts, feature_names=features,
                              text_encoder_file=f"text_encoder_{text_method}.joblib" if text_encoder else None)
                name = f"{stem}_xgboost"
                joblib.dump(bundle, out / f"{name}.joblib")
                for stage, model in enumerate(best_models):
                    model.save_model(out / f"{name}_stage{stage}.ubj")
                    pd.DataFrame({"feature": features, "gain_importance": model.feature_importances_}).sort_values(
                        "gain_importance", ascending=False).to_csv(out / f"{name}_stage{stage}_gain_importance.csv", index=False)
                dump_json(out / f"{name}_selected.json", best_info)
                save_history(best_models, task, name, out, plot=True)
                record(name, task, feature_set, text_method, best, matrices, "xgboost")
                if task not in selection or best_utility > selection[task]["utility"]:
                    selection[task] = dict(utility=float(best_utility), bundle=f"{name}.joblib", **best_info)
            del matrices, Xtrain
        del encoded
    dump_json(out / "selected_on_validation.json", selection)
    table = pd.DataFrame(rows)
    gaps = []
    for key, group in table.loc[table.model.isin(["xgboost", "logistic", "ridge"])].groupby(["task", "feature_set", "text_method", "model"]):
        lookup = group.set_index("split")
        if "validation" in lookup.index and "training_sample" in lookup.index:
            gap = dict(zip(["task", "feature_set", "text_method", "model"], key),
                       training_macro_f1=float(lookup.loc["training_sample", "macro_f1"]),
                       validation_macro_f1=float(lookup.loc["validation", "macro_f1"]),
                       training_minus_validation_macro_f1=float(lookup.loc["training_sample", "macro_f1"] - lookup.loc["validation", "macro_f1"]))
            if key[0] == "regression":
                gap.update(training_log_mae=float(lookup.loc["training_sample", "log_mae"]),
                           validation_log_mae=float(lookup.loc["validation", "log_mae"]),
                           validation_minus_training_log_mae=float(lookup.loc["validation", "log_mae"] - lookup.loc["training_sample", "log_mae"]))
            if key[0] == "ordinal":
                gap.update(training_mean_absolute_class_error=float(lookup.loc["training_sample", "mean_absolute_class_error"]),
                           validation_mean_absolute_class_error=float(lookup.loc["validation", "mean_absolute_class_error"]))
            gaps.append(gap)
    pd.DataFrame(gaps).to_csv(out / "generalization_gaps.csv", index=False)
    comparisons = []
    for (task, split, model), group in table.loc[table.model.isin(["xgboost", "logistic", "ridge"])].groupby(["task", "split", "model"]):
        index = group.set_index(["text_method", "feature_set"])
        pairs = []
        for method in args.text_methods:
            pairs.extend([((method, before), (method, after)) for before, after in
                          [("structural", "description"), ("description", "copilot"), ("description", "review"),
                           ("review", "review_copilot"), ("copilot", "review_copilot")]])
        for feature in args.feature_sets:
            pairs.extend([((before, feature), (after, feature)) for before, after in
                          [("none", "tfidf"), ("none", "embeddings"), ("tfidf", "embeddings")]])
        for before, after in pairs:
            if before in index.index and after in index.index:
                comparisons.append(dict(task=task, split=split, model=model,
                                        before="/".join(before), after="/".join(after),
                                        macro_f1_difference=float(index.loc[after, "macro_f1"] - index.loc[before, "macro_f1"]),
                                        absolute_class_error_difference=float(index.loc[after, "mean_absolute_class_error"] - index.loc[before, "mean_absolute_class_error"]),
                                        interpretation="Same PRs within this protocol; descriptive difference, no significance or causality claim"))
    pd.DataFrame(comparisons, columns=["task", "split", "model", "before", "after", "macro_f1_difference",
                                      "absolute_class_error_difference", "interpretation"]).to_csv(out / "feature_comparisons.csv", index=False)
    for task in args.tasks:
        plot = table.loc[table.task.eq(task) & table.split.eq("validation")].copy()
        plot["experiment"] = plot.text_method + "/" + plot.feature_set + "/" + plot.model
        metric = "log_mae" if task == "regression" else "mean_absolute_class_error" if task == "ordinal" else "macro_f1"
        fig, ax = plt.subplots(figsize=(9, max(4, .25 * len(plot))))
        ax.barh(plot.experiment, plot[metric])
        ax.set(xlabel=f"Validation {metric}", ylabel="")
        fig.tight_layout()
        fig.savefig(out / f"{task}_validation_comparison.png", dpi=150)
        plt.close(fig)
    LOG.info("Enhanced experiments completed: %s. Test evaluation: %s", out, manifest["test_evaluation"])


if __name__ == "__main__":
    main()
