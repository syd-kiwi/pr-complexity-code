"""Enhanced experiment tests use synthetic PRs, never research results."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import clean_data, labels
from modeling import (TextEncoder, check_external_overlap, classification_metrics,
                      ordinal_probabilities, ordered_prediction, partitions)
from test_pipeline import synthetic


class FakeEncoder:
    max_seq_length = 256

    def __init__(self, name, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        return np.asarray([[len(t), len(t.split()), 1.] for t in texts], dtype="float32")


class ResearchTests(unittest.TestCase):
    def test_ordinal_crossings_and_ordered_predictions(self):
        probabilities = ordinal_probabilities(np.array([.2, .8, .1]), np.array([.7, .3, .05]))
        self.assertTrue((probabilities >= 0).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_allclose(probabilities[0], [.55, 0, .45])
        np.testing.assert_array_equal(ordered_prediction(np.array([[.6, .3, .1], [.1, .6, .3], [.1, .2, .7]])), [0, 1, 2])
        near = classification_metrics([0, 2], [1, 1])
        severe = classification_metrics([0, 2], [2, 0])
        self.assertLess(near["mean_absolute_class_error"], severe["mean_absolute_class_error"])
        self.assertEqual(severe["severe_error_rate"], 1.)

    def test_text_vocabulary_is_training_only_and_reloadable(self):
        train = pd.DataFrame({"title": ["fix parser bug", "add parser tests", "update build scripts", "improve parser tests"], "body": ["unit tests"] * 4})
        validation = pd.DataFrame({"title": ["validationonlytoken parser"], "body": ["anotherunseenword"]})
        encoder = TextEncoder("tfidf", max_features=100, dimensions=2, min_df=1)
        encoded_train = encoder.fit_transform(train)
        self.assertEqual(encoded_train.shape, (4, 2))
        encoded = encoder.transform(validation)
        self.assertNotIn("validationonlytoken", encoder.vectorizer.vocabulary_)
        self.assertNotIn("anotherunseenword", encoder.vectorizer.vocabulary_)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "encoder.joblib"
            joblib.dump(encoder, path)
            reloaded = joblib.load(path)
            np.testing.assert_allclose(encoded, reloaded.transform(validation))

    def test_optional_embeddings_without_network(self):
        train = pd.DataFrame({"title": ["fix", "add tests", "refactor parser"], "body": ["", "details", "many more details"]})
        with patch.dict(sys.modules, {"sentence_transformers": SimpleNamespace(SentenceTransformer=FakeEncoder)}):
            encoder = TextEncoder("embeddings", batch_size=2, chunk_size=2)
            fitted = encoder.fit_transform(train)
            np.testing.assert_allclose(fitted, encoder.transform(train), atol=1e-6)
            self.assertEqual(encoder.metadata()["max_sequence_tokens"], 256)
            self.assertIsNone(encoder.__getstate__()["_encoder"])

    def test_split_reuse_is_by_key_not_row_order(self):
        df, _ = clean_data(synthetic(), "all")
        original = partitions(df, "repository", 42)
        with tempfile.TemporaryDirectory() as temp:
            split = df[["repository_id", "pull_request_id"]].copy()
            for name, ix in original.items():
                split.loc[ix, "split"] = name
            path = Path(temp) / "splits.csv"
            split.sample(frac=1, random_state=3).to_csv(path, index=False)
            reused = partitions(df, "repository", 99, path)
            for name in original:
                self.assertEqual(set(reused[name]), set(original[name]))
            changed = split.copy()
            changed.loc[0, "pull_request_id"] = "unmatched"
            changed.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exactly match"):
                partitions(df, "repository", 42, path)

    def test_known_repository_protocol_and_external_overlap(self):
        df, _ = clean_data(synthetic(), "all")
        # Every repo must have observations on both sides of merge-day cutoffs.
        df["merge_day"] = pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(np.arange(len(df)) // 30, unit="D")
        parts = partitions(df, "known-repository", 42)
        known = set(df.iloc[parts["train"]].repository_id)
        for name in ["validation", "test"]:
            self.assertTrue(set(df.iloc[parts[name]].repository_id).issubset(known))
        self.assertLess(df.iloc[parts["train"]].merge_day.max(), df.iloc[parts["validation"]].merge_day.min())
        with self.assertRaisesRegex(ValueError, "already in development"):
            check_external_overlap(df, df.iloc[:5])
        external = df.iloc[:5].copy()
        external["pull_request_id"] = "new_" + external.pull_request_id
        check_external_overlap(df, external)

    def test_enhanced_cpu_end_to_end_and_fixed_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            source = synthetic(600)
            source["title"] = [f"Fix parser issue {i % 9}" if i % 2 else f"Update dependency version {i % 7}" for i in range(len(source))]
            source["body"] = ["Add regression tests for bug fix" if i % 2 else "Maintenance package dependency update" for i in range(len(source))]
            source.to_csv(temp / "source.csv", index=False)
            command = [sys.executable, str(HERE / "run_research.py"), "--input", str(temp / "source.csv"),
                       "--output", str(temp / "run"), "--device", "cpu", "--threads", "1", "--trials", "1", "--trees", "5",
                       "--early-stopping", "2", "--feature-sets", "description", "review", "--text-methods", "none", "tfidf",
                       "--text-dimensions", "3", "--min-df", "1", "--vocabulary", "100", "--diagnostic-rows", "0"]
            run = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            out = temp / "run" / "models"
            metrics = pd.read_csv(out / "metrics.csv")
            self.assertEqual(set(metrics.split), {"training_sample", "validation"})
            self.assertEqual(set(metrics.loc[metrics.model.eq("xgboost"), "task"]), {"multiclass", "ordinal", "regression"})
            self.assertTrue(metrics.macro_f1.dropna().between(0, 1).all())
            manifest = json.loads((out / "experiment_manifest.json").read_text())
            self.assertEqual(manifest["test_evaluation"], "not_evaluated")
            self.assertTrue((out / "generalization_gaps.csv").exists())
            comparisons = pd.read_csv(out / "feature_comparisons.csv")
            self.assertFalse(comparisons.empty)
            self.assertIn("none/description", set(comparisons.before))
            self.assertIn("tfidf/description", set(comparisons.after))
            history = pd.read_csv(out / "tfidf_review_ordinal_xgboost_learning_curves.csv")
            self.assertEqual(set(history.stage), {0, 1})
            self.assertEqual(set(history["split"]), {"training_sample", "validation"})
            selected = json.loads((out / "selected_on_validation.json").read_text())
            self.assertEqual(set(selected), {"multiclass", "ordinal", "regression"})
            self.assertTrue((out / "tfidf_review_multiclass_xgboost_stage0_gain_importance.csv").exists())
            # The saved model can score independently supplied PRs without fitting.
            prepared = pd.read_parquet(temp / "run" / "prepared" / "prepared.parquet")
            external = prepared.iloc[:30].copy()
            external["pull_request_id"] = "new_" + external.pull_request_id
            external.to_parquet(temp / "external.parquet", index=False)
            model = out / "tfidf_review_ordinal_xgboost.joblib"
            evaluate = [sys.executable, str(HERE / "evaluate_saved.py"), "--model", str(model),
                        "--input", str(temp / "external.parquet"), "--development-input", str(temp / "run" / "prepared" / "prepared.parquet"),
                        "--output", str(temp / "evaluation")]
            run = subprocess.run(evaluate, text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            bundle = joblib.load(model)
            encoder = joblib.load(out / bundle["text_encoder_file"])
            from train import model_input
            matrix = np.column_stack([bundle["transform"].transform(model_input(external, "review")), encoder.transform(external)])
            expected = bundle["model"].predict_proba(matrix)
            actual = pd.read_csv(temp / "evaluation" / "predictions.csv.gz")
            np.testing.assert_allclose(expected, actual[["p_fast", "p_medium", "p_slow"]], rtol=1e-5)
            # Overlap must fail before creating evaluation output.
            external = prepared.iloc[:30]
            external.to_parquet(temp / "external.parquet", index=False)
            run = subprocess.run(evaluate[:-1] + [str(temp / "overlap_evaluation")], text=True, capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("already in development", run.stderr)


if __name__ == "__main__":
    unittest.main()
