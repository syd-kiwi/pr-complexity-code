"""Synthetic tests only. No source PR data or research results are included."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import (STRUCTURAL, REVIEW, TEXT, boolean, class_thresholds,
                    clean_data, labels, split_data)


def synthetic(n=900):
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"repository_id": [f"r{i % 30}" for i in range(n)],
                       "pull_request_id": [f"p{i}" for i in range(n)],
                       "minutes_to_merge": np.expm1(rng.uniform(0, 9, n)),
                       "is_open_source": True, "had_copilot_seat": rng.integers(0, 2, n),
                       "languages_list": ['["Python", "JavaScript"]'] * n,
                       "merge_day": pd.Timestamp("2026-03-01") + pd.to_timedelta(np.arange(n) % 60, unit="D")})
    for c in STRUCTURAL + REVIEW + TEXT:
        df[c] = rng.integers(0, 100, n)
    for c in ["is_code_pr", "has_title", "has_body", "short_body"]:
        df[c] = rng.integers(0, 2, n)
    return df


class PipelineTests(unittest.TestCase):
    def test_boolean_unknown_is_not_false(self):
        out = boolean(pd.Series(["False", "true", "0", "1", None], name="flag"))
        self.assertEqual(out.iloc[:4].tolist(), [0, 1, 0, 1])
        self.assertTrue(pd.isna(out.iloc[4]))
        with self.assertRaisesRegex(ValueError, "Invalid boolean"):
            boolean(pd.Series(["maybe"], name="flag"))

    def test_cleaning_scope_keys_targets_and_duplicates(self):
        source = synthetic(30)
        source.loc[0, "minutes_to_merge"] = -1
        source.loc[1, "repository_id"] = None
        source.loc[2, "is_open_source"] = False
        source.loc[3, "had_copilot_seat"] = np.nan
        source = pd.concat([source, source.iloc[[4]]], ignore_index=True)
        df, audit = clean_data(source, "open-source")
        self.assertEqual(len(df), 27)
        self.assertEqual(audit["identical_duplicate_rows_removed"], 1)
        self.assertEqual(audit["missing_copilot_rows"], 1)
        self.assertFalse(df.is_open_source.eq(0).any())
        self.assertTrue(df.minutes_to_merge.ge(0).all())

    def test_conflicting_snapshots_fail(self):
        source = synthetic(20)
        conflict = source.iloc[[0]].copy()
        conflict["files_touched"] += 1
        with self.assertRaisesRegex(ValueError, "Conflicting records"):
            clean_data(pd.concat([source, conflict]), "all")

    def test_group_split_has_no_repository_overlap(self):
        df, _ = clean_data(synthetic(), "all")
        parts = split_data(df, "repository", 42)
        groups = [set(df.iloc[ix].repository_id) for ix in parts.values()]
        self.assertEqual(sum(map(len, parts.values())), len(df))
        for i in range(3):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])
        again = split_data(df, "repository", 42)
        for name in parts:
            np.testing.assert_array_equal(parts[name], again[name])

    def test_merge_split_keeps_days_together(self):
        df, _ = clean_data(synthetic(), "all")
        parts = split_data(df, "merge-time", 42)
        self.assertLess(df.iloc[parts["train"]].merge_day.max(), df.iloc[parts["validation"]].merge_day.min())
        self.assertLess(df.iloc[parts["validation"]].merge_day.max(), df.iloc[parts["test"]].merge_day.min())

    def test_class_boundaries_and_train_only_thresholds(self):
        cuts = class_thresholds(pd.Series([0, 10, 20, 30, 40, 50, 60]))
        np.testing.assert_allclose(cuts, [20, 40])
        np.testing.assert_array_equal(labels([0, 20, 21, 40, 41, 10000], cuts), [0, 0, 1, 1, 2, 2])
        with self.assertRaisesRegex(ValueError, "Tied"):
            class_thresholds(pd.Series([0, 0, 0]))

    def test_csv_parquet_and_full_cpu_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            source = synthetic()
            csv = temp / "synthetic.csv"
            source.to_csv(csv, index=False)
            command = [sys.executable, str(HERE / "run_pipeline.py"), "--input", str(csv),
                       "--output", str(temp / "run"), "--device", "cpu", "--trials", "1", "--trees", "8",
                       "--threads", "1", "--permutation-rows", "20"]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            out = temp / "run" / "repository_seed42"
            metrics = pd.read_csv(out / "metrics.csv")
            self.assertEqual(len(metrics), 11)  # 5 sets x 2 models + majority
            self.assertTrue(metrics.macro_f1.between(0, 1).all())
            self.assertEqual(len(pd.read_csv(out / "feature_comparisons.csv")), 10)
            manifest = json.loads((out / "run_manifest.json").read_text())
            self.assertNotIn("complexity_score", sum(manifest["feature_sets"].values(), []))
            self.assertEqual(manifest["feature_timing"], "unverified_final_snapshot")
            self.assertTrue((temp / "run" / "rq1" / "within_repository_effect.json").exists())
            self.assertTrue((out / "copilot_xgboost.ubj").exists())
            # Verify saved preprocessing/model reproduce test probabilities.
            import joblib
            from train import model_input
            bundle = joblib.load(out / "copilot_xgboost.joblib")
            prepared = pd.read_parquet(temp / "run" / "prepared" / "prepared.parquet")
            split = split_data(prepared, "repository", 42)
            X = bundle["transform"].transform(model_input(prepared.iloc[split["test"]], "copilot"))
            probabilities = bundle["model"].predict_proba(X)
            predictions = pd.read_csv(out / "test_predictions.csv.gz")
            np.testing.assert_allclose(probabilities, predictions[[f"copilot_xgboost_p_{c}" for c in ["fast", "medium", "slow"]]], rtol=1e-5)
            # Parquet input exercises array-valued languages_list.
            source["languages_list"] = [["Python"] for _ in range(len(source))]
            parquet = temp / "synthetic.parquet"
            source.to_parquet(parquet)
            result = subprocess.run([sys.executable, str(HERE / "prepare.py"), "--input", str(parquet),
                                     "--output", str(temp / "from_parquet")], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # Existing output is protected.
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not empty", result.stderr)


if __name__ == "__main__":
    unittest.main()
