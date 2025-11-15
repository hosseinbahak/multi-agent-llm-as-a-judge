# multi_agent_llm_judge/providers/ollama_adapter.py
from typing import Dict, Any, List, Optional
from multi_agent_llm_judge.providers.ollama import OllamaClient
from loguru import logger

class OllamaProviderAdapter:
    """Adapter to make OllamaClient compatible with the existing provider interface."""
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        self.client = OllamaClient(base_url=base_url, timeout=timeout)
        self.model_mapping = {
            # Map OpenRouter model names to Ollama model names
            "anthropic/claude-3.5-sonnet": "llama3.2:latest",  # Example mapping
            "openai/gpt-4o": "llama3.2:latest",
            "meta-llama/llama-3.1-70b-instruct": "llama3.1:70b",
            # Add more mappings as needed
        }
        
    async def initialize(self):
        """Initialize the Ollama client."""
        await self.client.initialize()
        
        # List available models
        available_models = await self.client.list_models()
        logger.info(f"Available Ollama models: {available_models}")
        
    async def close(self):
        """Close the Ollama client."""
        await self.client.close()
        
    def _map_model_name(self, openrouter_model: str) -> str:
        """Map OpenRouter model names to Ollama model names."""
        if openrouter_model in self.model_mapping:
            return self.model_mapping[openrouter_model]
        
        # Try to extract a simple model name
        parts = openrouter_model.split('/')
        if len(parts) > 1:
            model_name = parts[-1].lower()
            # Check if we have this model
            return model_name
        
        # Default to a good local model
        logger.warning(f"Unknown model {openrouter_model}, defaulting to llama3.2:latest")
        return "llama3.2:latest"
    
    async def generate(self,
                      model: str,
                      messages: List[Dict[str, str]],
                      temperature: float = 0.7,
                      max_tokens: Optional[int] = None,
                      **kwargs) -> Dict[str, Any]:
        """Generate a completion using Ollama."""
        # Map model name
        ollama_model = self._map_model_name(model)
        
        try:
            # Use chat API for messages format
            result = await self.client.chat(
                model=ollama_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Convert to expected format
            return {
                "id": f"ollama-{ollama_model}",
                "model": model,  # Keep original model name for tracking
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": result["content"]
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                },
                "cost": 0.0,  # Local models are free
                "response_time_ms": result.get("total_duration", 0) // 1_000_000  # Convert nanoseconds to ms
            }
            
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            raise
    
    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost (always 0 for local models)."""
        return 0.0
