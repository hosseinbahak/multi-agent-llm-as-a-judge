from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ArrayLike = Union[np.ndarray, list, float]

@dataclass
class RegressionCalibrator:
    """
    A simple probabilistic calibrator using logistic regression over the
    engineered features. Implements `calibrate` to satisfy the abstract base.
    """
    C: float = 1.0
    max_iter: int = 200
    _pipe: Optional[Pipeline] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionCalibrator":
        self._pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs")),
            ]
        )
        self._pipe.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("Model not fitted. Call fit(X, y) first.")
        # Return calibrated probabilities of the positive class
        return self._pipe.predict_proba(X)[:, 1]

    # --- to satisfy the abstract base class contract ---
    def calibrate(self, X: ArrayLike) -> np.ndarray:
        """
        Accepts either a feature matrix or a single example. If a 1D list/array
        or scalar is passed, it will be reshaped appropriately.
        """
        X_arr = np.asarray(X)
        if X_arr.ndim == 0:
            X_arr = X_arr.reshape(1, -1)  # scalar -> (1, 1)
        elif X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)  # (n_features,) -> (1, n_features)
        return self.predict(X_arr)

    def save(self, path: Union[str, Path]) -> None:
        if self._pipe is None:
            raise RuntimeError("Model not fitted. Nothing to save.")
        joblib.dump(
            {
                "C": self.C,
                "max_iter": self.max_iter,
                "pipeline": self._pipe,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RegressionCalibrator":
        blob = joblib.load(str(path))
        obj = cls(C=blob.get("C", 1.0), max_iter=blob.get("max_iter", 200))
        obj._pipe = blob["pipeline"]
        return obj
