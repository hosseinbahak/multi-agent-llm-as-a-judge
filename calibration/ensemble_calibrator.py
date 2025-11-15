from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

ArrayLike = Union[np.ndarray, list, float]

@dataclass
class EnsembleCalibrator:
    """
    Two-stage ensemble:
      1) GradientBoostingClassifier to learn a strong scorer from features.
      2) IsotonicRegression to monotically calibrate the scores to probabilities.
    """
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 2

    _gb: Optional[GradientBoostingClassifier] = None
    _iso: Optional[IsotonicRegression] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleCalibrator":
        gb = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=1.0,
        )
        gb.fit(X, y)
        scores = gb.decision_function(X).astype(float)

        # Isotonic expects inputs in [min, max] but not necessarily probabilities.
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(scores, y)

        self._gb = gb
        self._iso = iso
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._gb is None or self._iso is None:
            raise RuntimeError("Model not fitted. Call fit(X, y) first.")
        scores = self._gb.decision_function(X).astype(float)
        probs = self._iso.transform(scores)
        return np.clip(probs, 0.0, 1.0)

    # --- to satisfy the abstract base class contract ---
    def calibrate(self, X: ArrayLike) -> np.ndarray:
        X_arr = np.asarray(X)
        if X_arr.ndim == 0:
            X_arr = X_arr.reshape(1, -1)
        elif X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return self.predict(X_arr)

    def save(self, path: Union[str, Path]) -> None:
        if self._gb is None or self._iso is None:
            raise RuntimeError("Model not fitted. Nothing to save.")
        joblib.dump(
            {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "gb": self._gb,
                "iso": self._iso,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EnsembleCalibrator":
        blob = joblib.load(str(path))
        obj = cls(
            n_estimators=blob.get("n_estimators", 200),
            learning_rate=blob.get("learning_rate", 0.05),
            max_depth=blob.get("max_depth", 2),
        )
        obj._gb = blob["gb"]
        obj._iso = blob["iso"]
        return obj
