# multi_agent_llm_judge/core/exceptions.py
"""Custom exceptions for the multi-agent LLM judge system."""

class MultiAgentJudgeError(Exception):
    """Base exception for all multi-agent judge errors."""
    pass

class ConfigurationError(MultiAgentJudgeError):
    """Raised when there's a configuration error."""
    pass

class ModelError(MultiAgentJudgeError):
    """Base exception for model-related errors."""
    pass

class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""
    pass

class ModelNotAvailableError(ModelError):
    """Raised when a requested model is not available."""
    pass

class ProviderError(MultiAgentJudgeError):
    """Base exception for provider-related errors."""
    pass

class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider is not found."""
    pass

class ProviderAPIError(ProviderError):
    """Raised when there's an API error from a provider."""
    pass

class LLMProviderError(ProviderError):
    """Raised when there's an error with the LLM provider."""
    pass

class AgentError(MultiAgentJudgeError):
    """Base exception for agent-related errors."""
    pass

class AgentExecutionError(AgentError):
    """Raised when an agent fails to execute properly."""
    pass

class AgentTimeoutError(AgentError):
    """Raised when an agent execution times out."""
    pass

class AgentNotFoundError(AgentError):
    """Raised when a requested agent is not found."""
    pass

class AllAgentsFailedError(AgentError):
    """Raised when all agents fail to produce a result."""
    pass

class CacheError(MultiAgentJudgeError):
    """Base exception for cache-related errors."""
    pass

class CacheReadError(CacheError):
    """Raised when there's an error reading from cache."""
    pass

class CacheWriteError(CacheError):
    """Raised when there's an error writing to cache."""
    pass

class ValidationError(MultiAgentJudgeError):
    """Raised when validation fails."""
    pass

class ParsingError(ValidationError):
    """Raised when parsing fails."""
    pass

class CalibrationError(MultiAgentJudgeError):
    """Base exception for calibration-related errors."""
    pass

class InsufficientDataError(CalibrationError):
    """Raised when there's insufficient data for calibration."""
    pass

class CalibrationNotFittedError(CalibrationError):
    """Raised when trying to use a calibrator that hasn't been fitted."""
    pass

class RoundError(MultiAgentJudgeError):
    """Base exception for round-related errors."""
    pass

class MaxRoundsExceededError(RoundError):
    """Raised when the maximum number of rounds is exceeded."""
    pass

class ConsensusError(RoundError):
    """Raised when there's an error reaching consensus."""
    pass

class JuryError(MultiAgentJudgeError):
    """Base exception for jury-related errors."""
    pass

class InsufficientVotesError(JuryError):
    """Raised when there are insufficient votes for a decision."""
    pass

class VotingStrategyError(JuryError):
    """Raised when there's an error with the voting strategy."""
    pass

class AsyncError(MultiAgentJudgeError):
    """Base exception for async-related errors."""
    pass

class AsyncTimeoutError(AsyncError):
    """Raised when an async operation times out."""
    pass

class AsyncCancellationError(AsyncError):
    """Raised when an async operation is cancelled."""
    pass

class BudgetExceededError(MultiAgentJudgeError):
    """Raised when the budget is exceeded."""
    pass
