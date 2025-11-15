# multi_agent_llm_judge/providers/__init__.py
"""LLM Provider implementations"""

from typing import Optional
from .base_provider import BaseProvider
from .openrouter import OpenRouterClient
from .model_registry import MODEL_REGISTRY

def get_provider(provider_name: str, **kwargs) -> BaseProvider:
    """Factory function to get provider instance.
    
    Args:
        provider_name: Name of the provider ('openai', 'anthropic', 'openrouter')
        **kwargs: Additional arguments for the provider
        
    Returns:
        Provider instance
        
    Raises:
        ValueError: If provider is not supported
    """
    provider_name = provider_name.lower()
    
    if provider_name == "openrouter":
        return OpenRouterClient(**kwargs)
    elif provider_name in ["openai", "anthropic"]:
        # For OpenAI and Anthropic, we use OpenRouter
        return OpenRouterClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

__all__ = [
    "BaseProvider",
    "OpenRouterClient",
    "MODEL_REGISTRY",
    "get_provider",
]
