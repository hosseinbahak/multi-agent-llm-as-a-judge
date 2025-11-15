# multi_agent_llm_judge/providers/ollama.py
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
from loguru import logger
import json

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize the HTTP session."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def list_models(self) -> List[str]:
        """List available models in Ollama."""
        await self.initialize()
        
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                response.raise_for_status()
                data = await response.json()
                return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []
    
    async def generate(self, 
                      model: str,
                      prompt: str,
                      system: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: Optional[int] = None,
                      stream: bool = False) -> Dict[str, Any]:
        """Generate completion using Ollama."""
        await self.initialize()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            }
        }
        
        if system:
            payload["system"] = system
            
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()
                
                return {
                    "content": result.get("response", ""),
                    "model": model,
                    "done": result.get("done", True),
                    "total_duration": result.get("total_duration", 0),
                    "load_duration": result.get("load_duration", 0),
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "eval_count": result.get("eval_count", 0),
                    "eval_duration": result.get("eval_duration", 0),
                }
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            raise
    
    async def chat(self,
                   model: str,
                   messages: List[Dict[str, str]],
                   temperature: float = 0.7,
                   max_tokens: Optional[int] = None,
                   stream: bool = False) -> Dict[str, Any]:
        """Chat completion using Ollama."""
        await self.initialize()
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()
                
                return {
                    "content": result.get("message", {}).get("content", ""),
                    "model": model,
                    "done": result.get("done", True),
                    "total_duration": result.get("total_duration", 0),
                    "load_duration": result.get("load_duration", 0),
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "eval_count": result.get("eval_count", 0),
                    "eval_duration": result.get("eval_duration", 0),
                }
        except Exception as e:
            logger.error(f"Error in Ollama chat: {e}")
            raise
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Estimate cost for Ollama (always 0 since it's local)."""
        return 0.0
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation."""
        # Rough estimation: ~4 characters per token
        return len(text) // 4
