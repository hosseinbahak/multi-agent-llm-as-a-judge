# multi_agent_llm_judge/providers/base_provider.py
"""Base provider interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ModelResponse:
    """Response from model API."""
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Dict[str, Any]
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.usage.get('total_tokens', 0)
    
    @property
    def prompt_tokens(self) -> int:
        """Get prompt tokens used."""
        return self.usage.get('prompt_tokens', 0)
    
    @property
    def completion_tokens(self) -> int:
        """Get completion tokens used."""
        return self.usage.get('completion_tokens', 0)

class BaseProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a chat completion request.
        
        Args:
            model: Model identifier
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Provider response dictionary
        """
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        pass
    
    @abstractmethod
    async def get_model_pricing(self) -> Dict[str, Dict[str, float]]:
        """Get pricing information for models."""
        pass
    
    async def close(self):
        """Close any open connections."""
        pass
