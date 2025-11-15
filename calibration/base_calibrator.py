# multi_agent_llm_judge/calibration/base_calibrator.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Any

class BaseCalibrator(ABC):
    """Abstract base class for all calibration models."""
    
    @abstractmethod
    async def calibrate(self, raw_confidence: float, features: Any) -> float:
        """
        Calibrate a confidence score using the features.
        
        Args:
            raw_confidence: The raw confidence score to calibrate
            features: The calibration features
            
        Returns:
            The calibrated confidence score
        """
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the calibration model.

        Args:
            X: Feature vectors for training data.
            y: True labels (0 or 1) for training data.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict calibrated probabilities for new data.

        Args:
            X: Feature vectors for which to predict probabilities.

        Returns:
            An array of calibrated probabilities.
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the trained model to a file."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> Any:
        """Load a trained model from a file."""
        pass
