"""The intent model as it runs: arrays and arithmetic, nothing else.

Training produces a standardisation, one weight vector and bias per class, and
a temperature. This module evaluates exactly that with numpy. Nothing is pickled
and no machine-learning library is imported, so a model saved today does not
depend on which versions of anything are installed next year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FeatureSpec, Featurizer


def log_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


@dataclass
class IntentModel:
    classes: tuple[str, ...]
    spec: FeatureSpec
    mean: np.ndarray  # [D]
    scale: np.ndarray  # [D]
    weights: np.ndarray  # [K, D]
    bias: np.ndarray  # [K]
    temperature: float = 1.0
    n_bins: int = 128
    n_channels: int = 4

    def __post_init__(self) -> None:
        self.classes = tuple(self.classes)
        self._featurizer = Featurizer(self.spec, self.n_bins, self.n_channels)
        if self.weights.shape != (len(self.classes), self._featurizer.size):
            raise ValueError(
                f"weights of shape {self.weights.shape} do not fit {len(self.classes)} classes and {self._featurizer.size} features"
            )

    @property
    def featurizer(self) -> Featurizer:
        return self._featurizer

    def scores(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.scale) @ self.weights.T + self.bias

    def log_proba_features(self, features: np.ndarray) -> np.ndarray:
        return log_softmax(self.scores(features) / self.temperature)

    def log_proba(self, cosp: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Log posterior ``[n, K]`` for frames ``[n, F, P]`` relative to ``reference``."""
        return self.log_proba_features(self._featurizer(cosp, reference))

    def proba(self, cosp: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return np.exp(self.log_proba(cosp, reference))

    # ------------------------------------------------------------------ storage
    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "mean": self.mean.astype(np.float64),
            "scale": self.scale.astype(np.float64),
            "weights": self.weights.astype(np.float64),
            "bias": self.bias.astype(np.float64),
            "temperature": np.float64(self.temperature),
        }

    @classmethod
    def from_arrays(cls, arrays, classes, spec: FeatureSpec, n_bins: int, n_channels: int) -> IntentModel:
        return cls(
            tuple(classes),
            spec,
            np.asarray(arrays["mean"]),
            np.asarray(arrays["scale"]),
            np.asarray(arrays["weights"]),
            np.asarray(arrays["bias"]),
            float(arrays["temperature"]),
            n_bins,
            n_channels,
        )
