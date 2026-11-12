"""Text encoders, ordered predictions, metrics, and reproducible split utilities.

These implementations borrow ideas from prior work; they are not DeepPull or
Nudge replications. All fitted representations use training data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, log_loss, mean_absolute_error, median_absolute_error,
                             recall_score)
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from common import CLASSES, KEYS, TARGET, split_data


def finite(value):
    return float(value) if np.isfinite(value) else None


def classification_metrics(y, pred, proba=None):
    y, pred = np.asarray(y), np.asarray(pred)
    result = dict(n=len(y), accuracy=float(accuracy_score(y, pred)),
                  balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                  macro_f1=float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
                  mean_absolute_class_error=float(np.abs(y - pred).mean()),
                  severe_error_rate=float((np.abs(y - pred) == 2).mean()),
                  quadratic_weighted_kappa=finite(cohen_kappa_score(y, pred, labels=[0, 1, 2], weights="quadratic")))
    recalls = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    result.update({f"recall_{cls}": float(recalls[j]) for j, cls in enumerate(CLASSES)})
    if proba is not None:
        result["log_loss"] = float(log_loss(y, proba, labels=[0, 1, 2]))
    return result


def duration_predictions(log_predictions):
    # Bound pathological predictions before expm1 without looking at test targets.
    return np.expm1(np.clip(log_predictions, 0., 30.))


def regression_metrics(minutes, log_predictions):
    durations = duration_predictions(log_predictions)
    log_target = np.log1p(minutes)
    return dict(n=len(minutes), mae_minutes=float(mean_absolute_error(minutes, durations)),
                median_ae_minutes=float(median_absolute_error(minutes, durations)),
                log_mae=float(mean_absolute_error(log_target, np.clip(log_predictions, 0., 30.))),
                log_rmse=float(np.sqrt(np.mean((log_target - np.clip(log_predictions, 0., 30.)) ** 2))))


def ordinal_probabilities(prob_above_fast, prob_above_medium):
    """Project two independent cumulative estimates onto q1 >= q2.

    Crossing estimates are pooled to their arithmetic mean (L2 projection).
    """
    first, second = np.asarray(prob_above_fast).copy(), np.asarray(prob_above_medium).copy()
    crossed = first < second
    average = (first[crossed] + second[crossed]) / 2
    first[crossed], second[crossed] = average, average
    return np.column_stack([1 - first, first - second, second]).clip(0, 1)


def ordered_prediction(proba):
    # Posterior median minimizes expected absolute class error.
    return (np.cumsum(proba, axis=1)[:, :2] < .5).sum(axis=1).astype("int32")


class OrdinalModel:
    """Two cumulative XGBoost binary classifiers with monotonic probabilities."""
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        return ordinal_probabilities(self.models[0].predict_proba(X)[:, 1],
                                     self.models[1].predict_proba(X)[:, 1])

    def predict(self, X):
        return ordered_prediction(self.predict_proba(X))


class TextEncoder:
    """Training-only TF-IDF/SVD or frozen optional sentence embeddings.

    Text encoding is chunked for transformation. No row IDs, timestamps, or
    targets are concatenated to text. Long descriptions are explicitly truncated.
    """
    def __init__(self, method="tfidf", max_features=30000, dimensions=128, min_df=3,
                 max_chars=12000, seed=42, encoder_name="sentence-transformers/all-MiniLM-L6-v2",
                 encoder_revision=None, device="cpu", batch_size=128, chunk_size=10000):
        self.method = method
        self.max_features = max_features
        self.dimensions = dimensions
        self.min_df = min_df
        self.max_chars = max_chars
        self.seed = seed
        self.encoder_name = encoder_name
        self.encoder_revision = encoder_revision
        self.device = device
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.vectorizer = None
        self.svd = None
        self.scaler = None
        self._encoder = None

    def texts(self, df):
        missing = set(["title", "body"]) - set(df.columns)
        if missing:
            raise ValueError(f"Missing text columns {sorted(missing)}. Prepare with --include-text.")
        # Keep title and body distinct. Do not add a prefix to empty documents.
        return (df.title.fillna("").astype(str) + "\n" + df.body.fillna("").astype(str)).str.slice(0, self.max_chars).tolist()

    def _load_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install paper2/requirements-embeddings.txt to use embeddings") from exc
            self._encoder = SentenceTransformer(self.encoder_name, revision=self.encoder_revision,
                                                device=self.device, trust_remote_code=False)
        return self._encoder

    def fit(self, df, return_training=False):
        texts = self.texts(df)
        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=self.max_features, min_df=self.min_df,
                                              ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)
            matrix = self.vectorizer.fit_transform(texts)
            if matrix.shape[1] < 2:
                raise ValueError("Training text vocabulary has fewer than 2 terms. Check text/min-df.")
            n_components = min(self.dimensions, matrix.shape[1] - 1, matrix.shape[0] - 1)
            if n_components < 1:
                raise ValueError("Too few training documents for SVD")
            self.svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
            encoded = self.svd.fit_transform(matrix).astype("float32")
        elif self.method == "embeddings":
            encoded = self._load_encoder().encode(texts, batch_size=self.batch_size, show_progress_bar=True,
                                                  convert_to_numpy=True, normalize_embeddings=True)
            self.max_seq_length_ = int(self._encoder.max_seq_length)
        else:
            raise ValueError(f"Unknown text method: {self.method}")
        self.scaler = StandardScaler().fit(encoded)
        self.output_dimensions_ = encoded.shape[1]
        if return_training:
            return self.scaler.transform(encoded).astype("float32")
        return self

    def fit_transform(self, df):
        return self.fit(df, return_training=True)

    def transform(self, df):
        output = np.empty((len(df), self.output_dimensions_), dtype="float32")
        for start in tqdm(range(0, len(df), self.chunk_size), desc=f"Encoding {self.method}", leave=False):
            stop = min(start + self.chunk_size, len(df))
            texts = self.texts(df.iloc[start:stop])
            if self.method == "tfidf":
                encoded = self.svd.transform(self.vectorizer.transform(texts))
            else:
                encoded = self._load_encoder().encode(texts, batch_size=self.batch_size, show_progress_bar=False,
                                                      convert_to_numpy=True, normalize_embeddings=True)
            output[start:stop] = self.scaler.transform(encoded).astype("float32")
        return output

    def metadata(self):
        result = {"method": self.method, "dimensions": self.output_dimensions_, "max_chars": self.max_chars,
                  "fit_scope": "training PRs only; frozen pretrained encoder when embeddings are selected"}
        if self.method == "tfidf":
            result.update(vocabulary_size=len(self.vectorizer.vocabulary_),
                          explained_variance_ratio_sum=float(self.svd.explained_variance_ratio_.sum()))
        else:
            result.update(encoder=self.encoder_name, requested_revision=self.encoder_revision,
                          max_sequence_tokens=self.max_seq_length_,
                          reproducibility_note="Pin --encoder-revision to an immutable model commit for archival runs.")
        return result

    def release_gpu(self):
        if self._encoder is not None:
            self._encoder.to("cpu")
            self._encoder = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_encoder"] = None
        return state


def partitions(df, strategy, seed, split_file=None):
    if split_file:
        supplied = pd.read_csv(split_file, dtype={c: "string" for c in KEYS})
        if not set(KEYS + ["split"]).issubset(supplied.columns):
            raise ValueError("Split file must contain repository_id, pull_request_id, split")
        if supplied[KEYS].isna().any().any() or supplied.duplicated(KEYS).any():
            raise ValueError("Split file contains missing or duplicate PR identifiers")
        if not supplied["split"].isin(["train", "validation", "test"]).all():
            raise ValueError("Split file contains unknown partition labels")
        left = pd.MultiIndex.from_frame(df[KEYS].astype("string"))
        right = pd.MultiIndex.from_frame(supplied[KEYS])
        if len(left) != len(right) or not left.isin(right).all():
            raise ValueError("Split file PR keys do not exactly match prepared data. Reuse the same scope/cleaning.")
        aligned = supplied.set_index(KEYS)["split"].reindex(left).to_numpy()
        result = {name: np.flatnonzero(aligned == name) for name in ["train", "validation", "test"]}
        if any(len(ix) == 0 for ix in result.values()):
            raise ValueError("Split file contains an empty partition")
        if strategy == "repository":
            groups = [set(df.iloc[ix].repository_id) for ix in result.values()]
            if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
                raise ValueError("Split file violates repository separation")
        elif strategy == "merge-time":
            if "merge_day" not in df or df.merge_day.isna().any():
                raise ValueError("Merge-period split file requires valid merge_day values")
            for before, after in [("train", "validation"), ("validation", "test")]:
                if not df.iloc[result[before]].merge_day.max() < df.iloc[result[after]].merge_day.min():
                    raise ValueError("Split file violates chronological merge-day separation")
        return result
    if strategy != "known-repository":
        return split_data(df, strategy, seed)
    # Hold out later merge periods within repositories represented in training.
    result = split_data(df, "merge-time", seed)
    known = set(df.iloc[result["train"]].repository_id)
    for name in ["validation", "test"]:
        ix = result[name]
        result[name] = ix[df.iloc[ix].repository_id.isin(known).to_numpy()]
    if any(len(ix) == 0 for ix in result.values()):
        raise ValueError("No known-repository rows remain in a partition")
    return result


def check_prepared(df, text_required=False):
    if df.empty or df[KEYS].isna().any().any() or df.duplicated(KEYS).any():
        raise ValueError("Prepared data must contain unique, nonmissing PR keys")
    if not np.isfinite(df[TARGET]).all() or not df[TARGET].ge(0).all():
        raise ValueError("Prepared targets must be finite and nonnegative")
    if text_required and not {"title", "body"}.issubset(df.columns):
        raise ValueError("Prepare this input with --include-text before text experiments")


def check_external_overlap(df, external):
    overlap = pd.MultiIndex.from_frame(df[KEYS].astype("string")).intersection(
        pd.MultiIndex.from_frame(external[KEYS].astype("string")))
    if len(overlap):
        raise ValueError(f"External evaluation contains {len(overlap)} PR keys already in development data")
