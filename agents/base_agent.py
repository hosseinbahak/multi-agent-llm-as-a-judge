# agents/base_agent.py
"""Base class for all evaluation agents."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import re
from datetime import datetime
from loguru import logger

from ..core.data_models import AgentAnalysis, Evidence, AgentType
from ..core.model_manager import ModelManager
from ..core.cache_manager import CacheManager
from ..config.schemas import AgentConfig


class BaseAgent(ABC):
    """Abstract base class for evaluation agents."""
    
    # Must be set by subclasses
    agent_type: AgentType = None
    default_system_prompt: str = None
    
    def __init__(
        self,
        config: AgentConfig,
        model_manager: ModelManager,
        cache_manager: CacheManager
    ):
        self.config = config
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        
        self.name = f"{self.__class__.__name__}"
        self.system_prompt = config.custom_prompt or self.default_system_prompt
    
    async def analyze(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
        previous_analyses: Optional[List[AgentAnalysis]] = None,
        round_number: int = 1
    ) -> AgentAnalysis:
        """
        Analyze the answer to the question.
        
        Args:
            question: The question being evaluated
            answer: The answer to evaluate
            context: Optional additional context
            previous_analyses: Analyses from previous rounds/agents
            round_number: Current round number
            
        Returns:
            AgentAnalysis with the agent's findings
        """
        start_time = datetime.now()
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            question, answer, context, round_number
        )
        
        # Check cache
        if self.config.retry_attempts > 0:  # Only cache if retries enabled
            cached = await self.cache_manager.get(f"agent:{cache_key}")
            if cached:
                logger.debug(f"{self.name} returning cached analysis")
                return cached
        
        # Select model
        model_id = await self._select_model()
        
        # Prepare messages
        messages = self._prepare_messages(
            question, answer, context, previous_analyses, round_number
        )
        
        # Call model with retries
        response_text, tokens, cost = await self._call_model(model_id, messages)
        
        # Parse response
        analysis = self._parse_response(
            response_text,
            model_id,
            tokens,
            cost,
            int((datetime.now() - start_time).total_seconds() * 1000)
        )
        
        # Cache result
        if self.config.retry_attempts > 0:
            await self.cache_manager.set(
                f"agent:{cache_key}",
                analysis,
                ttl=3600  # 1 hour cache
            )
        
        return analysis
    
    # multi_agent_llm_judge/agents/base_agent.py
    # Update the _select_model method to handle missing imports:

    async def _select_model(self) -> str:
        """Select model for this agent."""
        try:
            # Try to get the preferred model
            if hasattr(self.model_manager, 'get_model'):
                model = self.model_manager.get_model(self.config.model)
                if model:
                    return model.id
            
            # If no specific model or not found, use the configured model ID directly
            if self.config.model:
                return self.config.model
                
            # Fall back to default model
            logger.warning(f"No model specified for {self.name}, using default")
            return "openai/gpt-4o-mini"
            
        except Exception as e:
            logger.error(f"Failed to select model: {e}")
            # Return a default model as last resort
            return "openai/gpt-4o-mini"

    def _prepare_messages(
        self,
        question: str,
        answer: str,
        context: Optional[str],
        previous_analyses: Optional[List[AgentAnalysis]],
        round_number: int
    ) -> List[Dict[str, str]]:
        """Prepare messages for the LLM."""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Build user prompt
        user_prompt = self._build_prompt(
            question=question,
            answer=answer,
            context=context,
            previous_analyses=previous_analyses,
            round_number=round_number
        )
        
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    async def _call_model(
        self,
        model_id: str,
        messages: List[Dict[str, str]]
    ) -> Tuple[str, int, float]:
        """Call the model and get response."""
        for attempt in range(self.config.retry_attempts):
            try:
                # Call model through provider
                response = await self.model_manager.provider.chat_completion(
                    model=model_id,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                # Extract response text
                response_text = response["choices"][0]["message"]["content"]
                
                # Extract usage statistics
                usage = response.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
                
                # Estimate cost
                cost = await self.model_manager.estimate_cost(
                    model_id=model_id,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0)
                )
                
                # Update model stats
                self.model_manager.update_stats(
                    model_id=model_id,
                    success=True,
                    tokens=total_tokens,
                    cost=cost,
                    latency_ms=0  # TODO: Track actual latency
                )
                
                return response_text, total_tokens, cost
                
            except Exception as e:
                logger.warning(
                    f"{self.name} attempt {attempt + 1} failed: {e}"
                )
                
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Try fallback model on last attempt
                    if model_id != "openai/gpt-4o-mini":
                        logger.info(f"Trying fallback model for {self.name}")
                        return await self._call_model(
                            "openai/gpt-4o-mini",
                            messages
                        )
                    raise
        
        raise Exception("All retry attempts failed")
    
    def _generate_cache_key(
        self,
        question: str,
        answer: str,
        context: Optional[str],
        round_number: int
    ) -> str:
        """Generate cache key for this analysis."""
        components = [
            self.agent_type.value,
            question[:50],  # First 50 chars
            answer[:50],
            context[:50] if context else "no_context",
            str(round_number)
        ]
        
        return self.cache_manager.generate_key(*components)
    
    @abstractmethod
    def _build_prompt(
        self,
        question: str,
        answer: str,
        context: Optional[str],
        previous_analyses: Optional[List[AgentAnalysis]],
        round_number: int
    ) -> str:
        """Build the prompt for the model."""
        pass
    
    @abstractmethod
    def _parse_response(
        self,
        response_text: str,
        model_id: str,
        tokens_used: int,
        cost: float,
        processing_time_ms: int
    ) -> AgentAnalysis:
        """Parse model response into AgentAnalysis."""
        pass
    
    def _extract_confidence(self, response_text: str) -> float:
        """Extract confidence score from response."""
        # Look for explicit confidence mentions
        patterns = [
            r"confidence[:\s]+(\d+(?:\.\d+)?)\s*%",
            r"(\d+(?:\.\d+)?)\s*%\s*confident",
            r"confidence\s*score[:\s]+(\d+(?:\.\d+)?)",
            r"certainty[:\s]+(\d+(?:\.\d+)?)\s*%"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                confidence = float(match.group(1))
                # Convert percentage to decimal if needed
                if confidence > 1:
                    confidence /= 100
                return min(max(confidence, 0.0), 1.0)
        
        # Default confidence based on response keywords
        response_lower = response_text.lower()
        if any(word in response_lower for word in ["definitely", "certainly", "absolutely"]):
            return 0.9
        elif any(word in response_lower for word in ["likely", "probably", "appears"]):
            return 0.7
        elif any(word in response_lower for word in ["possibly", "might", "could"]):
            return 0.5
        elif any(word in response_lower for word in ["unlikely", "doubtful", "questionable"]):
            return 0.3
        else:
            return 0.6  # Default moderate confidence
    
    def _extract_verdict(self, response_text: str) -> Optional[bool]:
        """Extract verdict from response."""
        response_lower = response_text.lower()
        
        # Look for explicit verdict statements
        correct_indicators = [
            "the answer is correct",
            "this is correct",
            "answer: correct",
            "verdict: correct",
            "assessment: correct",
            "conclusion: correct",
            "the answer accurately",
            "the response correctly"
        ]
        
        incorrect_indicators = [
            "the answer is incorrect",
            "this is incorrect", 
            "answer: incorrect",
            "verdict: incorrect",
            "assessment: incorrect",
            "conclusion: incorrect",
            "the answer is wrong",
            "the response fails"
        ]
        
        # Count indicators
        correct_count = sum(1 for indicator in correct_indicators if indicator in response_lower)
        incorrect_count = sum(1 for indicator in incorrect_indicators if indicator in response_lower)
        
        if correct_count > incorrect_count:
            return True
        elif incorrect_count > correct_count:
            return False
        else:
            return None  # Uncertain
