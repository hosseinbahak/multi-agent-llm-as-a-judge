# multi_agent_llm_judge/utils/parsing.py
"""Utilities for parsing LLM responses."""

import re
import json
from typing import Optional, Dict, Any
from loguru import logger

from ..core.data_models import Verdict

def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from a response string.
    Handles cases where the JSON might be wrapped in markdown code blocks.
    """
    # Try to parse the entire response as JSON first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in code blocks
    json_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}'
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.MULTILINE)
        for match in matches:
            try:
                # Try to parse each match as JSON
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    
    logger.warning(f"Could not extract JSON from response: {response[:200]}...")
    return None

def extract_confidence_from_text(text: str) -> float:
    """
    Extract confidence score from text.
    Looks for patterns like "confidence: 0.8", "90% confident", etc.
    """
    # Patterns to match confidence expressions
    patterns = [
        r'confidence[:\s]+([0-9.]+)',
        r'([0-9.]+)\s*(?:%|percent)\s*confident',
        r'confidence\s*score[:\s]+([0-9.]+)',
        r'certainty[:\s]+([0-9.]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            # If it's a percentage (> 1), normalize to 0-1
            if value > 1:
                value = value / 100.0
            return min(max(value, 0.0), 1.0)
    
    # Default confidence if none found
    return 0.5

def extract_verdict_from_text(text: str) -> Verdict:
    """
    Extract verdict from text.
    Looks for keywords indicating correct/incorrect.
    """
    text_lower = text.lower()
    
    # Keywords indicating correctness
    correct_keywords = [
        'correct', 'accurate', 'right', 'valid', 'true', 
        'acceptable', 'satisfactory', 'good', 'pass'
    ]
    
    # Keywords indicating incorrectness
    incorrect_keywords = [
        'incorrect', 'wrong', 'false', 'invalid', 'inaccurate',
        'unacceptable', 'unsatisfactory', 'fail', 'bad'
    ]
    
    # Count occurrences
    correct_count = sum(keyword in text_lower for keyword in correct_keywords)
    incorrect_count = sum(keyword in text_lower for keyword in incorrect_keywords)
    
    # Check for explicit verdict statements
    if re.search(r'\bverdict[:\s]+correct\b', text_lower):
        return Verdict.CORRECT
    if re.search(r'\bverdict[:\s]+incorrect\b', text_lower):
        return Verdict.INCORRECT
    
    # Return based on keyword counts
    if correct_count > incorrect_count:
        return Verdict.CORRECT
    elif incorrect_count > correct_count:
        return Verdict.INCORRECT
    else:
        # Default to uncertain if no clear indication
        return Verdict.UNCERTAIN
