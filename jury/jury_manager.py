# multi_agent_llm_judge/jury/jury_manager.py
import asyncio
import numpy as np
from typing import List, Optional, Dict
from loguru import logger

from .juror import Juror
from ..core.data_models import JuryDecision, JurorVote, Verdict
from ..core.model_manager import ModelManager
from ..config.schemas import JuryConfig
from ..core.exceptions import AllAgentsFailedError

class JuryManager:
    """Manages the jury deliberation process."""

    def __init__(self, config: JuryConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        
        # ایجاد jurors بر اساس تنظیمات مدل‌ها
        self.jurors = []
        juror_configs = config.get_juror_configs()
        
        for i, (model, weight, temperature, max_tokens) in enumerate(juror_configs):
            juror = Juror(
                juror_id=i,
                config=config,
                model_manager=model_manager,
                model=model,
                weight=weight,
                temperature=temperature,
                max_tokens=max_tokens
            )
            self.jurors.append(juror)
            
        logger.info(f"Created {len(self.jurors)} jurors with diverse models:")
        model_counts = {}
        for juror in self.jurors:
            model_counts[juror.model] = model_counts.get(juror.model, 0) + 1
        for model, count in model_counts.items():
            logger.info(f"  - {count} jurors using {model}")

    async def conduct_trial(
        self,
        question: str,
        answer: str,
        context: Optional[str],
        aggregated_analysis: str,
        budget_remaining: Optional[float]
    ) -> JuryDecision:
        """
        Orchestrate the jury trial: have all jurors vote and then aggregate results.
        """
        logger.info(f"Conducting jury trial with {len(self.jurors)} jurors.")

        # Jurors vote in parallel
        vote_tasks = [
            juror.vote(question, answer, context, aggregated_analysis, budget_remaining)
            for juror in self.jurors
        ]

        results = await asyncio.gather(*vote_tasks, return_exceptions=True)

        # Filter out failed votes
        valid_votes: List[JurorVote] = [res for res in results if isinstance(res, JurorVote)]
        failed_votes = len(results) - len(valid_votes)

        if failed_votes > 0:
            logger.warning(f"{failed_votes}/{len(self.jurors)} jurors failed to vote.")

        if not valid_votes:
            raise AllAgentsFailedError("All jurors failed to produce a vote.")

        return self._aggregate_votes(valid_votes)

    def _aggregate_votes(self, votes: List[JurorVote]) -> JuryDecision:
        """
        Aggregate votes using a weighted majority system.
        """
        if not votes:
            return JuryDecision(
                votes=[],
                majority_verdict=Verdict.UNCERTAIN,
                vote_distribution={"correct": 0, "incorrect": 0, "uncertain": 0},
                aggregate_confidence=0.5,
                weighted_confidence=0.5,
                consensus_level=0.0
            )

        # محاسبه وزن نهایی هر juror (ترکیب وزن مدل و عملکرد)
        weights = np.array([self._compute_final_weight(vote, idx) for idx, vote in enumerate(votes)])

        # Map verdict enum to numeric values
        verdict_values = []
        for vote in votes:
            if vote.verdict == Verdict.CORRECT:
                verdict_values.append(1)
            elif vote.verdict == Verdict.INCORRECT:
                verdict_values.append(-1)
            else:  # UNCERTAIN
                verdict_values.append(0)

        verdicts = np.array(verdict_values)
        confidences = np.array([vote.confidence for vote in votes])

        # Weighted sum of verdicts determines the outcome
        weighted_verdict_sum = np.sum(verdicts * weights)

        # Determine majority verdict based on weighted sum
        if weighted_verdict_sum > 0.1:
            majority_verdict = Verdict.CORRECT
        elif weighted_verdict_sum < -0.1:
            majority_verdict = Verdict.INCORRECT
        else:
            majority_verdict = Verdict.UNCERTAIN

        # Vote distribution by model
        vote_distribution = {
            "correct": int(np.sum(verdicts == 1)),
            "incorrect": int(np.sum(verdicts == -1)),
            "uncertain": int(np.sum(verdicts == 0))
        }
        
        # توزیع رأی بر اساس مدل
        model_distribution = {}
        for vote in votes:
            model = getattr(vote, 'model_used', 'unknown')
            if model not in model_distribution:
                model_distribution[model] = {"correct": 0, "incorrect": 0, "uncertain": 0}
            
            if vote.verdict == Verdict.CORRECT:
                model_distribution[model]["correct"] += 1
            elif vote.verdict == Verdict.INCORRECT:
                model_distribution[model]["incorrect"] += 1
            else:
                model_distribution[model]["uncertain"] += 1

        # Calculate aggregate confidence
        aggregate_confidence = float(np.mean(confidences))

        # Calculate weighted confidence
        if majority_verdict == Verdict.CORRECT:
            majority_indices = np.where(verdicts == 1)[0]
        elif majority_verdict == Verdict.INCORRECT:
            majority_indices = np.where(verdicts == -1)[0]
        else:
            majority_indices = np.where(verdicts == 0)[0]

        if len(majority_indices) > 0:
            majority_weights = weights[majority_indices]
            majority_confidences = confidences[majority_indices]
            weighted_confidence = np.sum(majority_confidences * majority_weights) / np.sum(majority_weights)
        else:
            weighted_confidence = np.sum(confidences * weights) / np.sum(weights)

        # Consensus level
        total_weight_correct = np.sum(weights[np.where(verdicts == 1)])
        total_weight_incorrect = np.sum(weights[np.where(verdicts == -1)])
        total_weight_uncertain = np.sum(weights[np.where(verdicts == 0)])
        total_weight = total_weight_correct + total_weight_incorrect + total_weight_uncertain

        if total_weight > 0:
            if majority_verdict == Verdict.CORRECT:
                consensus_level = total_weight_correct / total_weight
            elif majority_verdict == Verdict.INCORRECT:
                consensus_level = total_weight_incorrect / total_weight
            else:
                consensus_level = total_weight_uncertain / total_weight
        else:
            consensus_level = 0.0

        # Summarize rationales
        summary = self._summarize_rationales(votes, majority_verdict)

        # Log model performance
        logger.info("Model voting distribution:")
        for model, dist in model_distribution.items():
            logger.info(f"  {model}: {dist}")

        return JuryDecision(
            votes=votes,
            majority_verdict=majority_verdict,
            vote_distribution=vote_distribution,
            model_distribution=model_distribution,  # اضافه کردن توزیع مدل
            aggregate_confidence=aggregate_confidence,
            weighted_confidence=float(weighted_confidence),
            consensus_level=float(consensus_level),
            key_agreements=summary['agreements'],
            key_disagreements=summary['disagreements']
        )

    def _compute_final_weight(self, vote: JurorVote, vote_idx: int) -> float:
        """محاسبه وزن نهایی با ترکیب وزن مدل و عملکرد juror."""
        juror = self.jurors[vote_idx]
        
        # وزن پایه از تنظیمات مدل
        model_weight = juror.weight
        
        # محاسبه وزن عملکردی (مانند قبل)
        performance_weight = self._compute_juror_weight(vote)
        
        # ترکیب وزن‌ها بسته به استراتژی
        if self.config.model_distribution_strategy == "performance_based":
            # عملکرد اهمیت بیشتری دارد
            final_weight = model_weight * 0.3 + performance_weight * 0.7
        else:
            # وزن مدل اهمیت بیشتری دارد
            final_weight = model_weight * 0.7 + performance_weight * 0.3
            
        return final_weight

    def _compute_juror_weight(self, vote: JurorVote) -> float:
        """Computes a juror's weight based on performance metrics."""
        # Find the corresponding juror
        juror = None
        for j in self.jurors:
            if str(j.juror_id) == vote.juror_id:
                juror = j
                break
                
        if not juror:
            return 1.0

        # Coherence score based on rationale length
        coherence_score = min(len(vote.rationale.split()) / 150.0, 1.0)

        # Evidence match
        evidence_match = 1.0 if vote.key_factors or vote.dissenting_points else 0.5

        raw_weight = (
            self.config.historical_accuracy_weight * juror.historical_accuracy +
            self.config.confidence_weight * vote.confidence +
            self.config.coherence_weight * coherence_score +
            self.config.evidence_match_weight * evidence_match
        )
        return raw_weight

    def _summarize_rationales(self, votes: List[JurorVote], majority_verdict: Verdict) -> Dict[str, List[str]]:
        """Extract key agreements and disagreements from rationales."""
        # جمع‌آوری نقاط کلیدی از juror هایی که با رأی اکثریت موافق هستند
        agreements = []
        disagreements = []
        
        for vote in votes:
            if vote.verdict == majority_verdict:
                agreements.extend(vote.key_factors or [])
                disagreements.extend(vote.dissenting_points or [])
        
        # حذف موارد تکراری و مرتب‌سازی بر اساس فراوانی
        from collections import Counter
        
        agreement_counts = Counter(agreements)
        disagreement_counts = Counter(disagreements)
        
        # برگرداندن موارد پرتکرار
        top_agreements = [item for item, _ in agreement_counts.most_common(3)]
        top_disagreements = [item for item, _ in disagreement_counts.most_common(2)]
        
        return {
            'agreements': top_agreements,
            'disagreements': top_disagreements
        }
