# core/model_manager.py
"""Model management and selection logic."""

import asyncio
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
from datetime import datetime, timedelta
from loguru import logger

from ..providers.base_provider import BaseProvider
from ..providers.model_registry import ModelRegistry
from .data_models import ModelConfig, AgentType, ModelSize
from .exceptions import ModelNotAvailableError, BudgetExceededError


class ModelStats:
    """Track model performance statistics."""
    
    def __init__(self):
        self.total_calls: int = 0
        self.successful_calls: int = 0
        self.failed_calls: int = 0
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.total_latency_ms: int = 0
        self.last_error: Optional[str] = None
        self.last_used: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls
    
    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_latency_ms / self.successful_calls


class ModelManager:
    """Manages model selection and usage tracking."""
    
    def __init__(self, provider: BaseProvider):
        self.provider = provider
        self.registry = ModelRegistry()
        self.model_stats: Dict[str, ModelStats] = defaultdict(ModelStats)
        self.available_models: Set[str] = set()
        self.model_pricing: Dict[str, Dict[str, float]] = {}
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize model manager by fetching available models."""
        if self._initialized:
            return
            
        async with self._lock:
            if self._initialized:
                return
                
            try:
                # Get available models from provider
                models = await self.provider.get_available_models()
                self.available_models = {m["id"] for m in models}
                
                # Get pricing information
                self.model_pricing = await self.provider.get_model_pricing()
                
                logger.info(f"Initialized with {len(self.available_models)} available models")
                self._initialized = True
                
            except Exception as e:
                logger.error(f"Failed to initialize model manager: {e}")
                raise
    
    async def select_model_for_agent(
        self,
        agent_type: AgentType,
        budget_remaining: Optional[float] = None,
        excluded_models: Optional[List[str]] = None,
        prefer_quality: bool = True
    ) -> str:
        """
        Select the best model for a specific agent type.
        
        Args:
            agent_type: Type of agent requesting a model
            budget_remaining: Remaining budget for evaluation
            excluded_models: Models to exclude from selection
            prefer_quality: Prefer quality over cost
            
        Returns:
            Model ID to use
        """
        if not self._initialized:
            await self.initialize()
        
        excluded = set(excluded_models or [])
        
        # Get recommended models for this agent type
        recommended = self.registry.get_recommended_models(agent_type)
        
        # Filter by availability and exclusions
        candidates = []
        for model_config in recommended:
            if model_config.id in self.available_models and model_config.id not in excluded:
                candidates.append(model_config)
        
        if not candidates:
            # Fallback to any available model
            for model_id in self.available_models:
                if model_id not in excluded:
                    config = self.registry.get_model_config(model_id)
                    if config:
                        candidates.append(config)
        
        if not candidates:
            raise ModelNotAvailableError("No available models found")
        
        # Score and sort candidates
        scored_candidates = []
        for model in candidates:
            score = self._score_model(
                model,
                agent_type,
                budget_remaining,
                prefer_quality
            )
            scored_candidates.append((score, model))
        
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Select best model
        selected = scored_candidates[0][1]
        
        logger.debug(
            f"Selected {selected.id} for {agent_type} "
            f"(score: {scored_candidates[0][0]:.2f})"
        )
        
        return selected.id
    
    def _score_model(
        self,
        model: ModelConfig,
        agent_type: AgentType,
        budget_remaining: Optional[float],
        prefer_quality: bool
    ) -> float:
        """Score a model for selection."""
        score = 0.0
        
        # Base quality score
        score += model.quality_score * (2.0 if prefer_quality else 1.0)
        
        # Speed score
        score += model.speed_score * (1.0 if prefer_quality else 1.5)
        
        # Recommendation bonus
        if agent_type in model.recommended_for:
            score += 0.5
        
        # Cost consideration
        if budget_remaining is not None:
            cost_per_1k = (model.cost_per_1k_prompt + model.cost_per_1k_completion) / 2
            if cost_per_1k > 0:
                # Estimate tokens for this evaluation
                estimated_tokens = 2000  # Conservative estimate
                estimated_cost = (estimated_tokens / 1000) * cost_per_1k
                
                if estimated_cost > budget_remaining:
                    return 0.0  # Can't afford this model
                
                # Cost efficiency score
                budget_ratio = estimated_cost / budget_remaining
                cost_score = 1.0 - min(budget_ratio * 2, 1.0)
                score += cost_score * (1.5 if not prefer_quality else 0.5)
        
        # Historical performance
        stats = self.model_stats[model.id]
        if stats.total_calls > 0:
            score += stats.success_rate * 0.3
            
            # Penalize high latency
            if stats.average_latency_ms > 10000:
                score *= 0.8
        
        return score
    
    async def estimate_cost(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Estimate cost for a model usage."""
        pricing = self.model_pricing.get(model_id, {})
        
        prompt_cost = (prompt_tokens / 1000) * pricing.get("prompt", 0)
        completion_cost = (completion_tokens / 1000) * pricing.get("completion", 0)
        
        return prompt_cost + completion_cost
    
    def update_stats(
        self,
        model_id: str,
        success: bool,
        tokens: int,
        cost: float,
        latency_ms: int,
        error: Optional[str] = None
    ):
        """Update model statistics."""
        stats = self.model_stats[model_id]
        stats.total_calls += 1
        
        if success:
            stats.successful_calls += 1
            stats.total_tokens += tokens
            stats.total_cost += cost
            stats.total_latency_ms += latency_ms
        else:
            stats.failed_calls += 1
            stats.last_error = error
        
        stats.last_used = datetime.now()
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a model."""
        return self.registry.get_model_config(model_id)
    
    def get_stats_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all model statistics."""
        summary = {}
        
        for model_id, stats in self.model_stats.items():
            summary[model_id] = {
                "total_calls": stats.total_calls,
                "success_rate": stats.success_rate,
                "average_latency_ms": stats.average_latency_ms,
                "total_cost": stats.total_cost,
                "total_tokens": stats.total_tokens,
                "last_used": stats.last_used.isoformat() if stats.last_used else None,
                "last_error": stats.last_error
            }
        
        return summary
