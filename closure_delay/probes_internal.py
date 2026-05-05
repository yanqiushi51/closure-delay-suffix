"""Linear probes trained on hidden states for direction and readiness.

DirectionProbe: binary (0=continue, 1=finalize)
ReadinessProbe: binary (0=not-ready, 1=ready-to-submit)
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score


DIRECTION_NAMES = ["continue", "finalize"]


def _prepare_data(
    states: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(states).all(axis=1)
    return states[valid], labels[valid], positions[valid]


class DirectionProbe:
    """Linear binary probe: 0=continue, 1=finalize.

    C_dir = p(finalize | hidden_state)
    High C_dir -> model is approaching final-answer mode.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.clf = LogisticRegression(
            solver="lbfgs",
            C=C,
            max_iter=max_iter,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.clf.score(self.scaler.transform(X), y))

    @staticmethod
    def c_dir(probs: np.ndarray) -> np.ndarray:
        """C_dir = probability of finalize class."""
        return probs[:, 1]

    @staticmethod
    def e_dir(probs: np.ndarray) -> np.ndarray:
        """Exploration pressure: 1 - p_finalize."""
        return 1.0 - probs[:, 1]


class ReadinessProbe:
    """Linear binary probe for readiness-to-finalize classification."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.clf = LogisticRegression(
            solver="lbfgs",
            C=C,
            max_iter=max_iter,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.clf.score(self.scaler.transform(X), y))

    @staticmethod
    def c_conf(probs: np.ndarray) -> np.ndarray:
        """C_conf = probability of class 1 (ready)."""
        return probs[:, 1]


def train_and_evaluate_direction(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    C: float = 1.0,
) -> Dict:
    if len(X_train) < 5 or len(np.unique(y_train)) < 2:
        return {"accuracy": None, "balanced_accuracy": None, "probe": None, "n_train": len(X_train), "n_test": len(X_test)}
    probe = DirectionProbe(C=C)
    probe.fit(X_train, y_train)
    acc = probe.score(X_test, y_test)
    bal_acc = balanced_accuracy_score(y_test, probe.predict(X_test))
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "probe": probe,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def train_and_evaluate_readiness(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    C: float = 1.0,
) -> Dict:
    if len(X_train) < 5 or len(np.unique(y_train)) < 2:
        return {"accuracy": None, "balanced_accuracy": None, "probe": None, "n_train": len(X_train), "n_test": len(X_test)}
    probe = ReadinessProbe(C=C)
    probe.fit(X_train, y_train)
    acc = probe.score(X_test, y_test)
    bal_acc = balanced_accuracy_score(y_test, probe.predict(X_test))
    pred_proba = probe.predict_proba(X_test)
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "probe": probe,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "c_conf_mean": float(np.mean(ReadinessProbe.c_conf(pred_proba))),
    }


def fraction_curve(
    positions_list: Sequence[Sequence[int]],
    response_lengths: Sequence[int],
    c_values_list: Sequence[np.ndarray],
    fractions: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
) -> Dict:
    buckets = {f: [] for f in fractions}
    for positions, resp_len, values in zip(positions_list, response_lengths, c_values_list):
        if resp_len == 0:
            continue
        f_indices = [int(round(f * resp_len)) for f in fractions]
        for f_idx, f_val in zip(f_indices, fractions):
            if len(positions) == 0:
                continue
            diffs = np.abs(np.array(positions) - f_idx)
            closest_idx = int(np.argmin(diffs))
            if diffs[closest_idx] <= resp_len * 0.1:
                buckets[f_val].append(float(values[closest_idx]))
    fractions_out = sorted(buckets)
    means = [float(np.mean(buckets[f])) if buckets[f] else np.nan for f in fractions_out]
    stds = [float(np.std(buckets[f])) if buckets[f] else np.nan for f in fractions_out]
    counts = [len(buckets[f]) for f in fractions_out]
    return {"fractions": fractions_out, "means": means, "stds": stds, "counts": counts}
