# multi_agent_llm_judge/providers/model_registry.py
"""Model registry for tracking available models and their capabilities."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..core.data_models import AgentType


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    id: str
    name: str
    provider: str
    context_length: int
    max_output_tokens: int
    supports_streaming: bool = True
    supports_function_calling: bool = True
    supports_system_message: bool = True
    supports_vision: bool = False
    pricing: Optional[Dict[str, float]] = None
    capabilities: Optional[Dict[AgentType, float]] = None
    aliases: Optional[List[str]] = None

    def get_capability_score(self, agent_type: AgentType) -> float:
        """Get capability score for a specific agent type."""
        if not self.capabilities:
            return 0.5  # Default neutral score
        return self.capabilities.get(agent_type, 0.5)


class ModelRegistry:
    """Registry for managing available models."""

    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize default models."""
        # GPT-4 variants
        self.models["gpt-4o"] = ModelConfig(
            id="gpt-4o",
            name="GPT-4 Optimized",
            provider="openai",
            context_length=128000,
            max_output_tokens=16384,
            supports_vision=True,
            pricing={
                "input_per_1k": 0.005,
                "output_per_1k": 0.015
            },
            capabilities={
                AgentType.CHAIN_OF_THOUGHT: 0.95,
                AgentType.ADVERSARY: 0.9,
                AgentType.CHALLENGER: 0.9,
                AgentType.INNOVATOR: 0.85,
                AgentType.SYNTHESIZER: 0.9,
                AgentType.META_QA: 0.95,
                AgentType.ASSUMPTION_GRAPHER: 0.85,
                AgentType.BIAS_AUDITOR: 0.9,
                AgentType.RETRIEVAL_VERIFIER: 0.85,
            },
            aliases=["gpt-4-optimized", "gpt4o"]
        )

        self.models["gpt-4o-mini"] = ModelConfig(
            id="gpt-4o-mini",
            name="GPT-4 Optimized Mini",
            provider="openai",
            context_length=128000,
            max_output_tokens=16384,
            supports_vision=True,
            pricing={
                "input_per_1k": 0.00015,
                "output_per_1k": 0.0006
            },
            capabilities={
                AgentType.CHAIN_OF_THOUGHT: 0.85,
                AgentType.ADVERSARY: 0.8,
                AgentType.CHALLENGER: 0.8,
                AgentType.INNOVATOR: 0.75,
                AgentType.SYNTHESIZER: 0.8,
                AgentType.META_QA: 0.85,
                AgentType.ASSUMPTION_GRAPHER: 0.75,
                AgentType.BIAS_AUDITOR: 0.8,
                AgentType.RETRIEVAL_VERIFIER: 0.75,
            },
            aliases=["gpt-4o-mini-2024-07-18"]
        )

        # Claude variants
        self.models["claude-3-5-sonnet"] = ModelConfig(
            id="anthropic/claude-3.5-sonnet",
            name="Claude 3.5 Sonnet",
            provider="anthropic",
            context_length=200000,
            max_output_tokens=8192,
            supports_vision=True,
            pricing={
                "input_per_1k": 0.003,
                "output_per_1k": 0.015
            },
            capabilities={
                AgentType.CHAIN_OF_THOUGHT: 0.95,
                AgentType.ADVERSARY: 0.85,
                AgentType.CHALLENGER: 0.9,
                AgentType.INNOVATOR: 0.9,
                AgentType.SYNTHESIZER: 0.95,
                AgentType.META_QA: 0.9,
                AgentType.ASSUMPTION_GRAPHER: 0.85,
                AgentType.BIAS_AUDITOR: 0.85,
                AgentType.RETRIEVAL_VERIFIER: 0.8,
            },
            aliases=["claude-3-5-sonnet-20241022"]
        )

    def register_model(self, model_config: ModelConfig):
        """Register a new model."""
        self.models[model_config.id] = model_config
        if model_config.aliases:
            for alias in model_config.aliases:
                self.models[alias] = model_config

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration by ID."""
        return self.models.get(model_id)

    def get_all_models(self) -> List[ModelConfig]:
        """Get all registered models."""
        # Return unique models (not aliases)
        seen = set()
        result = []
        for model in self.models.values():
            if model.id not in seen:
                seen.add(model.id)
                result.append(model)
        return result

    def get_models_for_agent(
        self, 
        agent_type: AgentType,
        min_capability_score: float = 0.7
    ) -> List[ModelConfig]:
        """Get models suitable for a specific agent type."""
        suitable_models = []
        for model in self.get_all_models():
            if model.get_capability_score(agent_type) >= min_capability_score:
                suitable_models.append(model)
        return sorted(
            suitable_models, 
            key=lambda m: m.get_capability_score(agent_type),
            reverse=True
        )


    def get_recommended_models(
        self,
        agent_type: AgentType,
        max_models: int = 3
    ) -> List[str]:
        """Get recommended model IDs for an agent type."""
        suitable_models = self.get_models_for_agent(agent_type)
        # Return model IDs only
        return [model.id for model in suitable_models[:max_models]]

MODEL_REGISTRY = ModelRegistry()
