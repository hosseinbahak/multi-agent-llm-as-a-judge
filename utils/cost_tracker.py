# multi_agent_llm_judge/utils/cost_tracker.py
"""Cost tracking utilities for monitoring LLM usage."""

from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime
from loguru import logger

@dataclass
class ModelCost:
    """Cost per token for a specific model."""
    input_cost_per_1k: float  # Cost per 1000 input tokens
    output_cost_per_1k: float  # Cost per 1000 output tokens

class CostTracker:
    """Tracks costs for LLM API calls."""
    
    # Default costs per 1000 tokens (you should update these based on actual pricing)
    DEFAULT_COSTS = {
        "gpt-4": ModelCost(0.03, 0.06),
        "gpt-4-turbo": ModelCost(0.01, 0.03),
        "gpt-3.5-turbo": ModelCost(0.0005, 0.0015),
        "claude-3-opus": ModelCost(0.015, 0.075),
        "claude-3-sonnet": ModelCost(0.003, 0.015),
        "claude-3-haiku": ModelCost(0.00025, 0.00125),
        "claude-2.1": ModelCost(0.008, 0.024),
        "claude-2": ModelCost(0.008, 0.024),
        "claude-instant": ModelCost(0.00163, 0.00551),
    }
    
    def __init__(self, custom_costs: Optional[Dict[str, ModelCost]] = None):
        """
        Initialize cost tracker with optional custom costs.
        
        Args:
            custom_costs: Dictionary mapping model names to ModelCost instances
        """
        self.costs = self.DEFAULT_COSTS.copy()
        if custom_costs:
            self.costs.update(custom_costs)
        
        self.usage_history = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
    
    def track_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Track token usage and calculate cost.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            metadata: Optional metadata to store with usage
            
        Returns:
            Cost for this usage
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        usage_record = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "metadata": metadata or {}
        }
        
        self.usage_history.append(usage_record)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        
        logger.debug(
            f"Tracked usage - Model: {model}, "
            f"Input: {input_tokens}, Output: {output_tokens}, "
            f"Cost: ${cost:.4f}"
        )
        
        return cost
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for given token usage.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Total cost
        """
        # Try to find exact model match first
        if model in self.costs:
            model_cost = self.costs[model]
        else:
            # Try to find partial match (e.g., "gpt-4-0613" matches "gpt-4")
            model_cost = None
            for cost_model, cost in self.costs.items():
                if cost_model in model:
                    model_cost = cost
                    break
            
            if not model_cost:
                logger.warning(f"Unknown model '{model}', using default GPT-3.5 pricing")
                model_cost = self.costs.get("gpt-3.5-turbo", ModelCost(0.001, 0.002))
        
        input_cost = (input_tokens / 1000) * model_cost.input_cost_per_1k
        output_cost = (output_tokens / 1000) * model_cost.output_cost_per_1k
        
        return input_cost + output_cost
    
    def get_summary(self) -> Dict:
        """
        Get summary of all tracked usage.
        
        Returns:
            Dictionary with usage summary
        """
        model_breakdown = {}
        for record in self.usage_history:
            model = record["model"]
            if model not in model_breakdown:
                model_breakdown[model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0
                }
            
            breakdown = model_breakdown[model]
            breakdown["calls"] += 1
            breakdown["input_tokens"] += record["input_tokens"]
            breakdown["output_tokens"] += record["output_tokens"]
            breakdown["cost"] += record["cost"]
        
        return {
            "total_calls": len(self.usage_history),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "model_breakdown": model_breakdown,
            "average_cost_per_call": self.total_cost / len(self.usage_history) if self.usage_history else 0
        }
    
    def reset(self):
        """Reset all tracking data."""
        self.usage_history.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        logger.info("Cost tracker reset")
    
    def get_remaining_budget(self, budget: float) -> float:
        """
        Calculate remaining budget.
        
        Args:
            budget: Total budget
            
        Returns:
            Remaining budget
        """
        return max(0, budget - self.total_cost)
