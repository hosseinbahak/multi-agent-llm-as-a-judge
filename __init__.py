# multi_agent_llm_judge/__init__.py
"""Multi-Agent LLM Judge System"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .core.round_manager import RoundManager
from .core.data_models import EvaluationRequest, EvaluationResult
from .config.schemas import RoundTableConfig

__all__ = [
    "RoundManager",
    "EvaluationRequest", 
    "EvaluationResult",
    "RoundTableConfig"
]
