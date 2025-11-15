# multi_agent_llm_judge/providers/openrouter.py
"""OpenRouter API client implementation."""

import os
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import httpx
from loguru import logger
import backoff
from collections import defaultdict

from .base_provider import BaseProvider
from ..core.exceptions import ModelNotAvailableError

class OpenRouterClient(BaseProvider):
    """Client for OpenRouter API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required")

        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/multi-agent-llm-judge",
            "X-Title": "Multi-Agent LLM Judge"
        }

        self.rate_limits = defaultdict(lambda: {"requests": 0, "reset_time": datetime.now()})
        self.rate_limit_lock = asyncio.Lock()

        self.client = httpx.AsyncClient(timeout=120.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    @backoff.on_exception(
        backoff.expo,
        (httpx.TimeoutException, httpx.NetworkError),
        max_tries=3
    )
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Make chat completion request."""
        # Check rate limit
        await self._check_rate_limit(model)

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=data
        )

        response.raise_for_status()
        return response.json()

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available models."""
        response = await self.client.get(
            f"{self.base_url}/models",
            headers=self.headers
        )

        response.raise_for_status()
        return response.json().get("data", [])

    async def get_model_pricing(self) -> Dict[str, Dict[str, float]]:
        """Get model pricing."""
        models = await self.get_available_models()

        pricing = {}
        for model in models:
            if "pricing" in model:
                pricing[model["id"]] = {
                    "prompt": float(model["pricing"].get("prompt", 0)),
                    "completion": float(model["pricing"].get("completion", 0))
                }

        return pricing

    async def check_model_availability(self, model_id: str) -> bool:
        """Check if model is available."""
        models = await self.get_available_models()
        return any(m["id"] == model_id for m in models)

    async def _check_rate_limit(self, model: str):
        """Simple rate limiting."""
        async with self.rate_limit_lock:
            limits = self.rate_limits[model]
            now = datetime.now()

            if now >= limits["reset_time"]:
                limits["requests"] = 0
                limits["reset_time"] = now + timedelta(minutes=1)

            if limits["requests"] >= 60:
                wait_time = (limits["reset_time"] - now).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    limits["requests"] = 0

            limits["requests"] += 1
