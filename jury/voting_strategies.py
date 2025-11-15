# multi_agent_llm_judge/jury/voting_strategies.py
from abc import ABC, abstractmethod
from typing import List, Protocol
import numpy as np

from ..core.data_models import JuryVote
from ..config.schemas import JuryConfig

# Define a "duck type" for Juror for type hinting without circular imports
class JurorProtocol(Protocol):
    juror_id: int
    historical_accuracy: float

class VotingStrategy(ABC):
    """Abstract base class for jury vote aggregation strategies."""

    @abstractmethod
    def aggregate(self, votes: List[JuryVote], jurors: List[JurorProtocol], config: JuryConfig) -> dict:
        """
        Aggregates votes to produce a final decision.

        Returns:
            A dictionary containing at least 'majority_verdict' and 'weighted_confidence'.
        """
        pass

class WeightedPerformanceStrategy(VotingStrategy):
    """
    A voting strategy that weights juror votes based on performance metrics.
    """
    def aggregate(self, votes: List[JuryVote], jurors: List[JurorProtocol], config: JuryConfig) -> dict:
        """Aggregates votes using a weighted majority system."""
        if not votes:
            return {
                "majority_verdict": None, 
                "weighted_confidence": 0.5, 
                "consensus_level": 0.0
            }

        weights = np.array([self._compute_juror_weight(vote, jurors, config) for vote in votes])
        verdicts = np.array([1 if vote.verdict else -1 for vote in votes])
        confidences = np.array([vote.confidence for vote in votes])

        # Normalize weights
        total_weight = np.sum(weights)
        normalized_weights = weights / total_weight if total_weight > 0 else weights

        weighted_verdict_sum = np.sum(verdicts * normalized_weights)
        majority_verdict = True if weighted_verdict_sum >= 0 else False

        # Calculate weighted confidence for the majority verdict
        majority_indices = np.where(verdicts == (1 if majority_verdict else -1))[0]
        
        if len(majority_indices) > 0:
            majority_weights = normalized_weights[majority_indices]
            majority_confidences = confidences[majority_indices]
            sum_majority_weights = np.sum(majority_weights)
            weighted_confidence = np.sum(majority_confidences * majority_weights) / sum_majority_weights if sum_majority_weights > 0 else 0.5
        else: # Should not happen if votes exist, but as a fallback
            weighted_confidence = np.average(confidences, weights=normalized_weights)

        # Consensus level
        total_weight_true = np.sum(normalized_weights[verdicts == 1])
        total_weight_false = np.sum(normalized_weights[verdicts == -1])
        consensus_level = abs(total_weight_true - total_weight_false)

        return {
            "majority_verdict": majority_verdict,
            "weighted_confidence": float(weighted_confidence),
            "consensus_level": float(consensus_level)
        }

    def _compute_juror_weight(self, vote: JuryVote, jurors: List[JurorProtocol], config: JuryConfig) -> float:
        """Computes a juror's weight based on performance metrics."""
        juror = next((j for j in jurors if j.juror_id == vote.juror_id), None)
        if not juror:
            return 0.1 # Default low weight if juror not found

        # Coherence score based on rationale length (proxy for detail)
        coherence_score = min(len(vote.rationale.split()) / 150.0, 1.0)
        
        # Evidence match score
        evidence_match = 1.0 if vote.key_agreements or vote.key_disagreements else 0.5
        
        raw_weight = (
            config.historical_accuracy_weight * juror.historical_accuracy +
            config.confidence_weight * vote.confidence +
            config.coherence_weight * coherence_score +
            config.evidence_match_weight * evidence_match
        )
        return max(raw_weight, 0.01) # Ensure weight is non-zero
