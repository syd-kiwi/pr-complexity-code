#!/usr/bin/env python3
"""Evaluate one fixed, trusted enhanced model on separate prepared PR data.

No fitting or tuning occurs here. Keep the text_encoder_*.joblib file alongside
its model bundle when moving an experiment directory.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import KEYS, TARGET, dump_json, fingerprint, fresh_output, labels
from experiment import diagnostic_tables, score_predictions
from modeling import check_external_overlap, check_prepared
from train import model_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trusted enhanced *.joblib bundle")
    parser.add_argument("--input", required=True, help="Separate prepared Parquet with outcome")
    parser.add_argument("--development-input", required=True, help="Prepared data used during development, for PR overlap checks")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", help="Device for optional text encoder inference")
    args = parser.parse_args()
    path = Path(args.model)
    bundle = joblib.load(path)
    df = pd.read_parquet(args.input).reset_index(drop=True)
    development = pd.read_parquet(args.development_input, columns=KEYS)
    check_prepared(df, bundle["text_method"] != "none")
    check_external_overlap(development, df)
    X = bundle["transform"].transform(model_input(df, bundle["feature_set"])).astype("float32")
    if bundle["text_encoder_file"]:
        encoder = joblib.load(path.parent / bundle["text_encoder_file"])
        encoder.device = args.device
        X = np.column_stack([X, encoder.transform(df)])
        encoder.release_gpu()
    y = labels(df[TARGET], bundle["cuts"])
    result, pred, proba, durations = score_predictions(bundle["task"], df[TARGET].to_numpy(), y, bundle["model"], X, bundle["cuts"])
    out = fresh_output(args.output)
    dump_json(out / "evaluation.json", {"metrics": result, "model": fingerprint(path), "source": fingerprint(args.input),
                                       "development_source": fingerprint(args.development_input),
                                       "thresholds_minutes": bundle["cuts"].tolist(),
                                       "evaluation_note": "No refitting. PR key nonoverlap verified. Independent sampling and lack of prior tuning require researcher verification."})
    diagnostic_tables(df, y, pred, bundle["cuts"], "fixed_model", "external", out, durations)
    prediction = df[KEYS].copy()
    prediction["true_class"], prediction["predicted_class"] = y, pred
    if proba is not None:
        for j, name in enumerate(["fast", "medium", "slow"]):
            prediction[f"p_{name}"] = proba[:, j]
    if durations is not None:
        prediction["predicted_minutes"] = durations
    prediction.to_csv(out / "predictions.csv.gz", index=False)
    print(f"Fixed-model evaluation complete: {out}")


if __name__ == "__main__":
    main()
