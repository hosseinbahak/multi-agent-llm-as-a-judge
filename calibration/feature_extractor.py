# multi_agent_llm_judge/calibration/feature_extractor.py
import numpy as np
from typing import List

from ..core.data_models import FinalJudgment, AgentAnalysis, Verdict

class FeatureExtractor:
    """Extracts a numerical feature vector from a FinalJudgment object."""

    def __init__(self, agent_types: List[str]):
        """
        Initialize with the list of all possible agent types to ensure consistent feature vector length.
        """
        self.agent_type_map = {agent_type: i for i, agent_type in enumerate(agent_types)}

    def extract(self, judgment: FinalJudgment) -> np.ndarray:
        """
        Converts a FinalJudgment into a NumPy feature vector.
        """
        features = []

        # Jury features
        features.append(judgment.jury_decision.weighted_confidence)
        features.append(judgment.jury_decision.consensus_level)
        features.append(len(judgment.jury_decision.votes))

        # Round features
        features.append(judgment.num_rounds_executed)

        # Agent analysis features
        agent_analyses = [analysis for round_result in judgment.round_results for analysis in round_result.agent_analyses]
        
        num_agents = len(agent_analyses)
        num_correct_votes = sum(1 for a in agent_analyses if a.verdict == Verdict.CORRECT)
        num_incorrect_votes = num_agents - num_correct_votes
        
        features.append(num_agents)
        features.append(num_correct_votes)
        features.append(num_incorrect_votes)
        features.append(num_correct_votes / num_agents if num_agents > 0 else 0)

        # Agent confidence stats
        if num_agents > 0:
            confidences = [a.confidence for a in agent_analyses]
            features.append(np.mean(confidences))
            features.append(np.std(confidences))
        else:
            features.extend([0.0, 0.0])

        # One-hot encoding for agent presence
        agent_presence = [0] * len(self.agent_type_map)
        for analysis in agent_analyses:
            agent_type_str = analysis.agent_type.value
            if agent_type_str in self.agent_type_map:
                agent_presence[self.agent_type_map[agent_type_str]] = 1
        features.extend(agent_presence)
        
        # Evidence and artifacts count
        num_evidence = sum(len(a.evidence) for a in agent_analyses if a.evidence)
        num_assumptions = sum(len(a.assumptions) for a in agent_analyses if a.assumptions)
        num_limitations = sum(len(a.limitations) for a in agent_analyses if a.limitations)
        
        features.append(num_evidence)
        features.append(num_assumptions)
        features.append(num_limitations)

        return np.array(features, dtype=float)
