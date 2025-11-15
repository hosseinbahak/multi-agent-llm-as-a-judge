#!/usr/bin/env python3
"""
Train calibration models for the multi-agent LLM judge system.
Updated to work with the actual evaluation file structure.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from loguru import logger
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multi_agent_llm_judge.calibration.regression_calibrator import RegressionCalibrator
from multi_agent_llm_judge.calibration.ensemble_calibrator import EnsembleCalibrator

def extract_features_from_evaluation(eval_data: Dict) -> Optional[Dict]:
    """
    Extract features from an evaluation record based on the actual structure.
    
    The evaluation files contain:
    - jury_says_correct: bool (the prediction)
    - jury_confidence: float (the confidence)
    - ground_truth: int (0 or 1, the actual answer)
    - jury_decision: dict with consensus_level, vote_distribution, etc.
    - calibration_features: dict with various feature values
    - metrics: dict with performance metrics
    """
    try:
        # Extract basic features
        features = {
            'jury_confidence': eval_data.get('jury_confidence', 0.5),
            'ground_truth': eval_data.get('ground_truth', 0),
            'jury_says_correct': eval_data.get('jury_says_correct', False),
        }
        
        # Extract jury decision features
        if 'jury_decision' in eval_data:
            jury = eval_data['jury_decision']
            features.update({
                'consensus_level': jury.get('consensus_level', 0.0),
                'weighted_confidence': jury.get('weighted_confidence', 0.5),
                'vote_correct': jury.get('vote_distribution', {}).get('correct', 0),
                'vote_incorrect': jury.get('vote_distribution', {}).get('incorrect', 0),
                'vote_uncertain': jury.get('vote_distribution', {}).get('uncertain', 0),
                'num_key_agreements': len(jury.get('key_agreements', [])),
                'num_key_disagreements': len(jury.get('key_disagreements', [])),
            })
        
        # Extract calibration features if available
        if 'calibration_features' in eval_data:
            cal_features = eval_data['calibration_features']
            features.update({
                f'cal_{k}': v for k, v in cal_features.items()
                if isinstance(v, (int, float))
            })
        
        # Extract metrics
        if 'metrics' in eval_data:
            metrics = eval_data['metrics']
            features.update({
                'num_rounds': metrics.get('num_rounds', 1),
                'num_agents': metrics.get('num_agents', 1),
                'processing_time_ms': metrics.get('processing_time_ms', 0),
                'total_tokens': metrics.get('total_tokens', 0),
                'num_models': len(metrics.get('models_used', [])),
            })
        
        # Calculate vote ratios
        total_votes = features.get('vote_correct', 0) + features.get('vote_incorrect', 0) + features.get('vote_uncertain', 0)
        if total_votes > 0:
            features['vote_correct_ratio'] = features.get('vote_correct', 0) / total_votes
            features['vote_incorrect_ratio'] = features.get('vote_incorrect', 0) / total_votes
            features['vote_uncertain_ratio'] = features.get('vote_uncertain', 0) / total_votes
        else:
            features['vote_correct_ratio'] = 0.0
            features['vote_incorrect_ratio'] = 0.0
            features['vote_uncertain_ratio'] = 0.0
        
        return features
        
    except Exception as e:
        logger.warning(f"Failed to extract features: {e}")
        return None

def load_evaluation_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Load evaluation data and extract features and labels.
    
    Returns:
        X: Feature matrix
        y: Ground truth labels
        raw_data: List of raw evaluation dictionaries
    """
    eval_files = list(data_dir.glob("eval_*.json"))
    logger.info(f"Found {len(eval_files)} evaluation files")
    
    all_features = []
    all_labels = []
    raw_data = []
    
    for eval_file in eval_files:
        try:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
            
            features = extract_features_from_evaluation(eval_data)
            if features is not None:
                # Extract label (ground truth)
                label = features['ground_truth']
                
                # Remove non-feature fields
                feature_dict = {k: v for k, v in features.items() 
                               if k not in ['ground_truth', 'jury_says_correct']}
                
                all_features.append(feature_dict)
                all_labels.append(label)
                raw_data.append(eval_data)
            
        except Exception as e:
            logger.warning(f"Failed to process {eval_file}: {e}")
            continue
    
    if not all_features:
        raise ValueError("No valid training data found")
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(all_features)
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    # Convert to numpy arrays
    X = df.values
    y = np.array(all_labels)
    
    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    logger.info(f"Label distribution: {np.sum(y==1)} correct, {np.sum(y==0)} incorrect")
    
    return X, y, raw_data

def evaluate_calibration(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                        model_name: str, output_dir: Path) -> Dict:
    """Evaluate calibration quality and save plots."""
    # Calculate metrics
    brier = brier_score_loss(y_true, y_pred_proba)
    
    # ROC AUC - handle case where all labels are the same
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = 0.5  # Default for undefined AUC
        
    logloss = log_loss(y_true, y_pred_proba)
    
    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10, strategy='uniform'
    )
    
    # Expected Calibration Error (ECE)
    ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    
    # Plot calibration curve
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label=f'{model_name} (Brier={brier:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histogram of predictions
    plt.subplot(1, 2, 2)
    plt.hist(y_pred_proba[y_true == 0], bins=20, alpha=0.5, label='Incorrect', density=True)
    plt.hist(y_pred_proba[y_true == 1], bins=20, alpha=0.5, label='Correct', density=True)
    plt.xlabel('Predicted probability')
    plt.ylabel('Density')
    plt.title('Distribution of Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_calibration.png', dpi=150)
    plt.close()
    
    metrics = {
        'brier_score': float(brier),
        'auc_roc': float(auc),
        'log_loss': float(logloss),
        'ece': float(ece),
    }
    
    return metrics

def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "calibration_data" / "evaluations"
    output_dir = project_root / "calibration_models"
    output_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logger.add(output_dir / "training.log")
    
    try:
        # Load data
        logger.info("Loading evaluation data...")
        X, y, raw_data = load_evaluation_data(data_dir)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train regression calibrator
        logger.info("Training regression calibrator...")
        reg_calibrator = RegressionCalibrator()
        reg_calibrator.fit(X_train, y_train)
        
        # Evaluate
        reg_pred = reg_calibrator.predict(X_test)
        reg_metrics = evaluate_calibration(y_test, reg_pred, "regression", output_dir)
        logger.info(f"Regression calibrator metrics: {reg_metrics}")
        
        # Save model
        reg_calibrator.save(output_dir / "regression_calibrator.pkl")
        
        # Train ensemble calibrator
        logger.info("Training ensemble calibrator...")
        ens_calibrator = EnsembleCalibrator()
        ens_calibrator.fit(X_train, y_train)
        
        # Evaluate
        ens_pred = ens_calibrator.predict(X_test)
        ens_metrics = evaluate_calibration(y_test, ens_pred, "ensemble", output_dir)
        logger.info(f"Ensemble calibrator metrics: {ens_metrics}")
        
        # Save model
        ens_calibrator.save(output_dir / "ensemble_calibrator.pkl")
        
        # Compare uncalibrated vs calibrated
        # Extract original confidences for test set
        test_indices = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
        )[1]
        
        original_confidences = []
        for idx in test_indices:
            eval_data = raw_data[idx]
            # Get the original jury confidence
            conf = eval_data.get('jury_confidence', 0.5)
            # If jury_says_correct is False, we need to flip the confidence
            if not eval_data.get('jury_says_correct', True):
                conf = 1 - conf
            original_confidences.append(conf)
        
        original_confidences = np.array(original_confidences)
        
        # Evaluate original calibration
        orig_metrics = evaluate_calibration(y_test, original_confidences, "original", output_dir)
        logger.info(f"Original (uncalibrated) metrics: {orig_metrics}")
        
        # Save results summary
        results = {
            'original': orig_metrics,
            'regression': reg_metrics,
            'ensemble': ens_metrics,
            'data_stats': {
                'total_samples': len(X),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'num_features': X.shape[1],
                'positive_rate_train': float(np.mean(y_train)),
                'positive_rate_test': float(np.mean(y_test)),
            }
        }
        
        with open(output_dir / "calibration_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Models saved to {output_dir}")
        
        # Print improvement summary
        print("\n" + "="*50)
        print("CALIBRATION TRAINING SUMMARY")
        print("="*50)
        print(f"Original Brier Score: {orig_metrics['brier_score']:.4f}")
        print(f"Regression Brier Score: {reg_metrics['brier_score']:.4f} "
              f"(improvement: {orig_metrics['brier_score'] - reg_metrics['brier_score']:.4f})")
        print(f"Ensemble Brier Score: {ens_metrics['brier_score']:.4f} "
              f"(improvement: {orig_metrics['brier_score'] - ens_metrics['brier_score']:.4f})")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()

# python -m multi_agent_llm_judge.calibration.train_calibration --data-dir /home/zeus/Projects/hb/multi_agent_llm_judge/calibration_data/evaluations/ --out ./calibration_model.pkl





