# multi_agent_llm_judge/utils/metrics.py
import numpy as np

def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculates the Brier score, a measure of probabilistic forecast accuracy.
    Lower is better.
    
    Args:
        y_true: Array of true binary labels (0 or 1).
        y_prob: Array of predicted probabilities for the positive class.
    
    Returns:
        The Brier score.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return np.mean((y_prob - y_true) ** 2)

def calculate_expected_calibration_error(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    n_bins: int = 15
) -> float:
    """
    Calculates the Expected Calibration Error (ECE).
    This measures the difference between expected and actual accuracy. Lower is better.
    
    Args:
        y_true: Array of true binary labels (0 or 1).
        y_prob: Array of predicted probabilities for the positive class.
        n_bins: The number of bins to partition the probabilities into.
    
    Returns:
        The ECE value.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    # Create bin boundaries
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = len(y_true)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find indices of samples in the current bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            # Calculate accuracy in the bin
            accuracy_in_bin = np.mean(y_true[in_bin])
            # Calculate average confidence in the bin
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

