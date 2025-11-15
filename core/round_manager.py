# multi_agent_llm_judge/core/round_manager.py
"""Main round management and orchestration."""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from loguru import logger

from .data_models import (
    EvaluationRequest, EvaluationResult, FinalJudgment,
    RoundSummary, AgentAnalysis, CalibrationFeatures, Evidence,
    AgentType, Verdict
)
from .agent_manager import AgentManager
from .model_manager import ModelManager
from .cache_manager import CacheManager
from ..jury.jury_manager import JuryManager
from ..calibration.ensemble_calibrator import EnsembleCalibrator
from ..calibration.feature_extractor import FeatureExtractor
from ..config.schemas import RoundTableConfig, AgentConfig
from .exceptions import BudgetExceededError
from ..agents.base_agent import BaseAgent

class RoundManager:
    """Orchestrates the entire evaluation process."""

    def __init__(
        self,
        config: RoundTableConfig,
        model_manager: ModelManager,
        cache_manager: CacheManager
    ):
        self.config = config
        self.model_manager = model_manager
        self.cache_manager = cache_manager

        # Initialize components
        self.agent_manager = AgentManager(
            model_manager=model_manager,
            cache_manager=cache_manager,
            max_concurrent=getattr(config.execution, 'parallel_limit', 5)
        )

        self.jury_manager = JuryManager(
            config=config.jury,
            model_manager=model_manager
        )

        # Initialize calibrator without config parameter
        self.calibrator = EnsembleCalibrator()
        
        # Try to load existing calibration model if specified
        if config.calibration.enabled and config.calibration.model_path:
            try:
                self.calibrator = EnsembleCalibrator.load(config.calibration.model_path)
                logger.info(f"Loaded calibration model from {config.calibration.model_path}")
            except Exception as e:
                logger.warning(f"Could not load calibration model: {e}. Using new instance.")
                self.calibrator = EnsembleCalibrator()
        
        # Get all possible agent types from the AgentType enum
        agent_types = [agent_type.value for agent_type in AgentType]
        self.feature_extractor = FeatureExtractor(agent_types)

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """
        Perform complete evaluation of the answer.

        Args:
            request: Evaluation request

        Returns:
            Complete evaluation result
        """
        start_time = datetime.now()

        try:
            # Check cache first
            cache_key = self.cache_manager.generate_key(
                request.question,
                request.answer,
                request.context
            )

            cached_result = await self.cache_manager.get(f"eval:{cache_key}")
            if cached_result:
                logger.info("Returning cached evaluation result")
                return cached_result

            # Initialize tracking
            round_summaries: List[RoundSummary] = []
            total_cost = 0.0
            total_tokens = 0
            models_used = set()

            # Create agents
            agents = await self._create_agents(request)

            # Execute rounds
            for round_num in range(1, self.config.execution.num_rounds + 1):
                logger.info(f"Starting round {round_num}/{self.config.execution.num_rounds}")

                # Check budget
                if request.max_cost and total_cost >= request.max_cost:
                    raise BudgetExceededError(
                        f"Budget exceeded: ${total_cost:.4f} >= ${request.max_cost:.4f}"
                    )

                # Execute round
                round_summary = await self._execute_round(
                    agents=agents,
                    request=request,
                    previous_summaries=round_summaries,
                    round_number=round_num,
                    budget_remaining=(
                        request.max_cost - total_cost
                        if request.max_cost else None
                    )
                )

                round_summaries.append(round_summary)
                total_cost += round_summary.total_cost
                total_tokens += round_summary.total_tokens

                # Track models used
                for analysis in round_summary.agent_analyses:
                    models_used.add(analysis.model_used)

                # Early stopping check
                if self.config.execution.early_stopping:
                    if round_summary.average_confidence >= self.config.execution.early_stopping_confidence:
                        logger.info(
                            f"Early stopping: confidence {round_summary.average_confidence:.2%} "
                            f">= {self.config.execution.early_stopping_confidence:.2%}"
                        )
                        break

            # Aggregate all analyses
            all_analyses = []
            for summary in round_summaries:
                all_analyses.extend(summary.agent_analyses)

            # Prepare aggregated analysis for jury
            aggregated_analysis = self._aggregate_analyses(round_summaries)

            # Conduct jury deliberation
            logger.info("Starting jury deliberation")
            jury_decision = await self.jury_manager.conduct_trial(
                question=request.question,
                answer=request.answer,
                context=request.context,
                aggregated_analysis=aggregated_analysis,
                budget_remaining=(
                    request.max_cost - total_cost
                    if request.max_cost else None
                )
            )

            total_cost += jury_decision.total_cost

            # For now, skip complex feature extraction since calibrator is not fitted
            calibrated_confidence = jury_decision.weighted_confidence
            
            # Create calibration features
            evidence_count = sum(len(a.evidence) for s in round_summaries for a in s.agent_analyses if a.evidence)
            
            calibration_features = CalibrationFeatures(
                base_confidence=jury_decision.weighted_confidence,
                agent_agreement=jury_decision.consensus_level,
                evidence_strength=min(evidence_count / 10.0, 1.0),  # Normalize
                reasoning_coherence=0.8,  # Placeholder
                model_diversity=len(models_used) / 10.0,  # Normalize
                consensus_strength=jury_decision.consensus_level,
                token_efficiency=0.7,  # Placeholder
                round_consistency=0.8  # Placeholder
            )

            # Create final judgment
            judgment = FinalJudgment(
                question=request.question,
                answer=request.answer,
                context=request.context,
                is_correct=jury_decision.majority_verdict,
                raw_confidence=jury_decision.weighted_confidence,
                calibrated_confidence=calibrated_confidence,
                executive_summary=self._create_executive_summary(
                    verdict=jury_decision.majority_verdict,
                    confidence=calibrated_confidence,
                    key_factors=jury_decision.key_agreements[:3]
                ),
                detailed_rationale=aggregated_analysis,
                evidence_refs=self._extract_evidence(all_analyses),
                round_summaries=round_summaries,
                jury_decision=jury_decision,
                calibration_features=calibration_features,
                models_used=list(models_used),
                total_tokens=total_tokens,
                total_cost=total_cost,
                processing_time_ms=int(
                    (datetime.now() - start_time).total_seconds() * 1000
                )
            )

            # Create result
            result = EvaluationResult(
                request=request,
                judgment=judgment,
                success=True,
                recommendations=self._generate_recommendations(judgment)
            )

            # Cache result
            await self.cache_manager.set(
                f"eval:{cache_key}",
                result,
                ttl=self.config.cache.ttl
            )

            logger.info(
                f"Evaluation complete: verdict={judgment.is_correct}, "
                f"confidence={calibrated_confidence:.2%}, "
                f"cost=${total_cost:.4f}, "
                f"time={judgment.processing_time_ms}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

            # Create error result
            return EvaluationResult(
                request=request,
                judgment=None,
                success=False,
                error=str(e),
                warnings=["Evaluation failed due to an error"]
            )

    async def _create_agents(self, request: EvaluationRequest) -> List[BaseAgent]:
        """Create agent instances based on configuration."""
        agents = []

        for agent_config in self.config.agents:
            # Skip disabled agents
            if not agent_config.enabled:
                continue

            # Skip excluded agents
            if request.required_agents and agent_config.type not in request.required_agents:
                continue

            try:
                agent = await self.agent_manager.create_agent(agent_config)
                agents.append(agent)
                logger.debug(f"Created agent: {agent.name}")

            except Exception as e:
                logger.error(f"Failed to create agent {agent_config.type}: {e}")

        return agents

    async def _execute_round(
        self,
        agents: List[BaseAgent],
        request: EvaluationRequest,
        previous_summaries: List[RoundSummary],
        round_number: int,
        budget_remaining: Optional[float]
    ) -> RoundSummary:
        """Execute a single round of evaluation."""
        start_time = datetime.now()

        # Get previous analyses for context
        previous_analyses = []
        if previous_summaries:
            for summary in previous_summaries:
                previous_analyses.extend(summary.agent_analyses)

        # Execute agents
        if self.config.execution.parallel_agents:
            analyses = await self.agent_manager.execute_agents_parallel(
                agents=agents,
                request=request,
                previous_analyses=previous_analyses,
                round_number=round_number
            )
        else:
            analyses = await self.agent_manager.execute_agents_sequential(
                agents=agents,
                request=request,
                previous_analyses=previous_analyses,
                round_number=round_number
            )

        # Calculate round statistics
        total_cost = sum(a.cost for a in analyses)
        total_tokens = sum(a.tokens_used for a in analyses)
        confidences = [a.confidence for a in analyses]
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        confidence_variance = (
            sum((c - average_confidence) ** 2 for c in confidences) / len(confidences)
            if confidences else 0.0
        )

        # Extract insights
        aggregated_insights = self._extract_insights(analyses)
        consensus_points = self._find_consensus(analyses)
        disagreement_points = self._find_disagreements(analyses)

        # Create round summary
        summary = RoundSummary(
            round_number=round_number,
            agent_analyses=analyses,
            aggregated_analysis=self._create_round_analysis(
                analyses, consensus_points, disagreement_points
            ),
            aggregated_insights=aggregated_insights,
            consensus_points=consensus_points,
            disagreement_points=disagreement_points,
            average_confidence=average_confidence,
            confidence_variance=confidence_variance,
            total_tokens=total_tokens,
            total_cost=total_cost,
            processing_time_ms=int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
        )

        return summary

    def _aggregate_analyses(self, round_summaries: List[RoundSummary]) -> str:
        """Create comprehensive aggregated analysis."""
        sections = []

        # Overview
        total_agents = sum(len(s.agent_analyses) for s in round_summaries)
        final_confidence = round_summaries[-1].average_confidence if round_summaries else 0.5

        sections.append(
            f"AGGREGATED ANALYSIS\n"
            f"==================\n\n"
            f"Total Rounds: {len(round_summaries)}\n"
            f"Total Agent Analyses: {total_agents}\n"
            f"Final Average Confidence: {final_confidence:.2%}\n"
        )

        # Key insights across all rounds
        all_insights = []
        for summary in round_summaries:
            all_insights.extend(summary.aggregated_insights)

        if all_insights:
            sections.append("\nKEY INSIGHTS:")
            for i, insight in enumerate(all_insights[:10], 1):
                sections.append(f"{i}. {insight}")

        # Consensus evolution
        sections.append("\n\nCONSENSUS EVOLUTION:")
        for summary in round_summaries:
            sections.append(
                f"\nRound {summary.round_number}:"
            )
            for point in summary.consensus_points[:3]:
                sections.append(f"  • {point}")

        # Major disagreements
        all_disagreements = []
        for summary in round_summaries:
            all_disagreements.extend(summary.disagreement_points)

        if all_disagreements:
            sections.append("\n\nMAJOR DISAGREEMENTS:")
            for i, disagreement in enumerate(set(all_disagreements[:5]), 1):
                sections.append(f"{i}. {disagreement}")

        # Individual agent summaries
        sections.append("\n\nAGENT VERDICTS:")
        for summary in round_summaries:
            for analysis in summary.agent_analyses:
                verdict_text = analysis.verdict.value if analysis.verdict else "Unknown"
                sections.append(
                    f"  • {analysis.agent_name}: {verdict_text} "
                    f"(confidence: {analysis.confidence:.2%})"
                )

        return "\n".join(sections)

    def _extract_insights(self, analyses: List[AgentAnalysis]) -> List[str]:
        """Extract key insights from analyses."""
        insights = []

        for analysis in analyses:
            # Extract from reasoning steps
            for step in analysis.reasoning_steps[:2]:
                if len(step) > 20:  # Filter meaningful steps
                    insights.append(step)

            # Extract from evidence
            for evidence in analysis.evidence[:1]:
                if evidence.relevance_score > 0.7:
                    insights.append(f"{evidence.source}: {evidence.content}")

        # Deduplicate similar insights
        unique_insights = []
        for insight in insights:
            if not any(self._similar(insight, existing) for existing in unique_insights):
                unique_insights.append(insight)

        return unique_insights[:10]

    def _find_consensus(self, analyses: List[AgentAnalysis]) -> List[str]:
        """Find consensus points among analyses."""
        # Simple consensus: points mentioned by multiple agents
        all_points = []

        for analysis in analyses:
            all_points.extend(analysis.reasoning_steps)
            all_points.extend([e.content for e in analysis.evidence])

        # Count occurrences (simplified - could use embeddings)
        point_counts = {}
        for point in all_points:
            found = False
            for existing in point_counts:
                if self._similar(point, existing):
                    point_counts[existing] += 1
                    found = True
                    break
            if not found and len(point) > 20:
                point_counts[point] = 1

        # Return points mentioned by multiple agents
        threshold = len(analyses) * 0.4  # 40% agreement
        consensus = [
            point for point, count in point_counts.items()
            if count >= threshold
        ]

        return sorted(consensus, key=lambda x: point_counts[x], reverse=True)[:5]

    def _find_disagreements(self, analyses: List[AgentAnalysis]) -> List[str]:
        """Find disagreement points among analyses."""
        disagreements = []

        # Check verdict disagreements
        verdicts = [a.verdict for a in analyses if a.verdict is not None]
        if verdicts and not all(v == verdicts[0] for v in verdicts):
            disagreements.append("Agents disagree on the final verdict")

        # Check confidence variance
        confidences = [a.confidence for a in analyses]
        if confidences:
            variance = sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)
            if variance > 0.1:
                disagreements.append(f"High confidence variance: {variance:.3f}")

        # Check assumption conflicts
        all_assumptions = []
        for analysis in analyses:
            all_assumptions.extend(analysis.assumptions)

        # Simplified conflict detection
        if len(set(all_assumptions)) > len(analyses) * 2:
            disagreements.append("Multiple conflicting assumptions identified")

        return disagreements[:5]

    def _similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are similar (simplified)."""
        # Simple similarity check - in production use embeddings
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union > 0.5

    def _create_round_analysis(
        self,
        analyses: List[AgentAnalysis],
        consensus: List[str],
        disagreements: List[str]
    ) -> str:
        """Create analysis summary for a round."""
        sections = []

        # Agent summary
        sections.append(f"Analyzed by {len(analyses)} agents:")
        for analysis in analyses:
            verdict_text = analysis.verdict.value if analysis.verdict else "Unknown"
            sections.append(
                f"  - {analysis.agent_name}: {verdict_text} "
                f"({analysis.confidence:.2%} confidence)"
            )

        # Consensus
        if consensus:
            sections.append("\nConsensus Points:")
            for point in consensus[:3]:
                sections.append(f"  • {point}")

        # Disagreements
        if disagreements:
            sections.append("\nDisagreements:")
            for point in disagreements[:3]:
                sections.append(f"  • {point}")

        return "\n".join(sections)

    def _extract_evidence(self, analyses: List[AgentAnalysis]) -> List[Evidence]:
        """Extract best evidence from all analyses."""
        all_evidence = []

        for analysis in analyses:
            all_evidence.extend(analysis.evidence)

        # Sort by relevance and deduplicate
        seen = set()
        unique_evidence = []

        for evidence in sorted(all_evidence, key=lambda e: e.relevance_score, reverse=True):
            key = (evidence.source, evidence.content[:50])
            if key not in seen:
                seen.add(key)
                unique_evidence.append(evidence)

        return unique_evidence[:10]

    def _create_executive_summary(
        self,
        verdict: Verdict,
        confidence: float,
        key_factors: List[str]
    ) -> str:
        """Create executive summary."""
        verdict_text = "correct" if verdict == Verdict.CORRECT else "incorrect" if verdict == Verdict.INCORRECT else "uncertain"
        confidence_level = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "low"

        summary = (
            f"The answer is {verdict_text} with {confidence_level} confidence ({confidence:.1%}). "
        )

        if key_factors:
            summary += f"Key factors: {'; '.join(key_factors[:2])}."

        return summary

    def _generate_recommendations(self, judgment: FinalJudgment) -> List[str]:
        """Generate recommendations based on judgment."""
        recommendations = []

        # Low confidence recommendation
        if judgment.calibrated_confidence < 0.6:
            recommendations.append(
                "Consider gathering more evidence or context for higher confidence"
            )

        # High disagreement recommendation
        if judgment.jury_decision.consensus_level < 0.6:
            recommendations.append(
                "Significant disagreement among evaluators - consider human review"
            )

        # Evidence recommendation
        if len(judgment.evidence_refs) < 3:
            recommendations.append(
                "Limited evidence found - consider providing more context"
            )

        return recommendations
