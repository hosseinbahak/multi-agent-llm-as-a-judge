# multi_agent_llm_judge/jury/juror.py
import random
import json
from typing import Dict, List, Optional
from loguru import logger

from ..core.data_models import JurorVote, Evidence, Verdict
from ..core.model_manager import ModelManager
from ..core.exceptions import LLMProviderError, ParsingError
from ..config.schemas import JuryConfig
from ..utils.parsing import extract_json_from_response, extract_confidence_from_text

class Juror:
    """Represents an individual juror in the jury."""

    def __init__(
        self,
        juror_id: int,
        config: JuryConfig,
        model_manager: ModelManager,
        model: str,  # مدل خاص این juror
        weight: float = 1.0,  # وزن این juror
        temperature: Optional[float] = None,  # دمای خاص
        max_tokens: Optional[int] = None  # حداکثر توکن خاص
    ):
        self.juror_id = juror_id
        self.config = config
        self.model_manager = model_manager
        self.model = model
        self.weight = weight
        self.temperature = temperature or config.temperature
        self.max_tokens = max_tokens or config.max_tokens
        self.historical_accuracy = 0.75  # Start with a neutral default
        self.name = f"Juror-{self.juror_id}-{model.split('/')[-1]}"

    async def vote(
        self,
        question: str,
        answer: str,
        context: Optional[str],
        aggregated_analysis: str,
        budget_remaining: Optional[float]
    ) -> JurorVote:
        """
        Cast a vote based on the provided evidence and analysis.
        """
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_user_prompt(question, answer, context, aggregated_analysis)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        last_error = None
        for attempt in range(self.config.retry_attempts):
            try:
                if budget_remaining is not None and budget_remaining <= 0:
                     raise LLMProviderError("Jury budget depleted.")

                response = await self.model_manager.provider.chat_completion(
                    model=self.model,  # استفاده از مدل خاص این juror
                    messages=messages,
                    temperature=self.temperature,  # استفاده از دمای خاص
                    max_tokens=self.max_tokens,  # استفاده از max_tokens خاص
                    response_format={"type": "json_object"}
                )

                response_text = response["choices"][0]["message"]["content"]

                # Parse response and create vote
                vote = self._parse_response(response_text)
                vote.model_used = self.model  # ثبت مدل استفاده شده
                vote.weight = self.weight  # ثبت وزن
                return vote

            except Exception as e:
                last_error = e
                logger.warning(f"{self.name} attempt {attempt + 1} failed: {e}")

        raise LLMProviderError(f"{self.name} failed after all retries.") from last_error

    def _parse_response(self, response_text: str) -> JurorVote:
        """Parse the JSON response from the LLM into a JuryVote object."""
        try:
            data = json.loads(response_text)

            # Use utility if confidence is not in JSON
            if 'confidence' not in data:
                 data['confidence'] = extract_confidence_from_text(data.get('rationale', ''))

            # Convert boolean verdict to Verdict enum
            verdict_bool = data.get('verdict', False)
            verdict = Verdict.CORRECT if verdict_bool else Verdict.INCORRECT

            return JurorVote(
                juror_id=str(self.juror_id),
                verdict=verdict,
                confidence=float(data.get('confidence', 0.5)),
                rationale=data.get('rationale', "No rationale provided."),
                key_factors=data.get('key_agreements', []),
                dissenting_points=data.get('key_disagreements', []),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse juror response: {response_text}")
            raise ParsingError(f"Could not parse juror JSON response. Error: {e}") from e

    def update_historical_accuracy(self, correct: bool):
        """Update the juror's historical accuracy using an exponential moving average."""
        alpha = 0.1  # Learning rate
        self.historical_accuracy = (alpha * (1 if correct else 0)) + (1 - alpha) * self.historical_accuracy

    def _get_system_prompt(self) -> str:
        """Select a system prompt variant based on the model."""
        # پرامپت‌های متفاوت برای مدل‌های مختلف
        model_specific_prompts = {
            "gpt-4": [
                "You are an expert juror with deep analytical capabilities. Your role is to provide a thorough and nuanced evaluation of an AI's answer based on multi-agent analysis. Leverage your advanced reasoning to deliver a comprehensive JSON judgment.",
                "As a sophisticated judicial analyst, synthesize complex findings from various AI agents to render a definitive verdict. Your analysis should demonstrate exceptional depth and critical thinking. Output in strict JSON format."
            ],
            "gpt-4o": [
                "You are a meticulous and unbiased juror. Your role is to evaluate an AI's answer to a question based on a provided analysis from multiple agents. You must deliver a structured JSON judgment with a clear verdict, confidence score, and a detailed rationale.",
                "As a critical juror, your task is to synthesize the findings of various analytical agents and render a final judgment. Focus on logical consistency, factual accuracy, and the strength of the evidence presented. Output your decision in a strict JSON format."
            ],
            "gpt-3.5-turbo": [
                "You are an efficient and fair juror. Review the analysis from multiple agents about an AI's answer and provide a clear verdict. Be concise but thorough. Output your judgment as a JSON object.",
                "As a practical juror, evaluate the correctness of an answer based on agent analyses. Focus on the most important points and provide a balanced judgment in JSON format."
            ]
        }
        
        # انتخاب پرامپت بر اساس مدل
        for model_key, prompts in model_specific_prompts.items():
            if model_key in self.model:
                return random.choice(prompts)
        
        # پرامپت پیش‌فرض
        default_prompts = [
            "You are an impartial judge. Review the consolidated analysis from a panel of AI agents regarding an answer's correctness. Your verdict must be objective and supported by a step-by-step rationale. Provide your output as a JSON object."
        ]
        return random.choice(default_prompts)

    def _build_user_prompt(
        self, question: str, answer: str, context: Optional[str], aggregated_analysis: str
    ) -> str:
        """Construct the user prompt for the juror."""

        prompt = f"""**TASK: EVALUATE THE ANSWER**

You must act as an impartial juror and determine if the provided 'Answer' is a correct and satisfactory response to the 'Question'.

**Model Context:** You are using {self.model} with weight {self.weight} in the jury deliberation.

**1. Core Task:**
- **Question:** "{question}"
- **Answer to Evaluate:** "{answer}"
"""

        if context:
            prompt += f"""- **Context:** "{context}"
"""

        prompt += f"""
**2. Aggregated Analysis from a Panel of AI Agents:**
This analysis summarizes the findings of multiple specialized agents who have already examined the answer. Review it carefully.
---
{aggregated_analysis}
---

**3. Your Judgment:**
Based on all the information, provide your final judgment in a JSON object. Your JSON response MUST include the following keys:
- `verdict`: (boolean) `true` if the answer is correct, `false` otherwise.
- `confidence`: (float) Your confidence in the verdict, from 0.0 (no confidence) to 1.0 (absolute certainty).
- `rationale`: (string) A detailed, step-by-step explanation for your decision. Synthesize the agent analysis, highlighting the most critical points that led to your verdict.
- `key_agreements`: (list of strings) The top 2-3 most important points where agents reached a consensus that you agree with.
- `key_disagreements`: (list of strings) Any significant disagreements among agents that influenced your decision.

Produce ONLY the JSON object and nothing else.
"""
        return prompt
