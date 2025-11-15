# multi_agent_llm_judge/core/data_models.py
"""Core data models for the multi-agent judge system."""

from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator, ConfigDict
from enum import Enum
import uuid

class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    META = "meta"
    MISTRAL = "mistral"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"

class ModelSize(str, Enum):
    """Model size categories."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"

class AgentType(str, Enum):
    """Types of agents in the system."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ADVERSARY = "adversary"
    INNOVATOR = "innovator"
    SYNTHESIZER = "synthesizer"
    CHALLENGER = "challenger"
    RETRIEVAL_VERIFIER = "retrieval_verifier"
    META_QA = "meta_qa"
    ASSUMPTION_GRAPHER = "assumption_grapher"
    BIAS_AUDITOR = "bias_auditor"

class Verdict(str, Enum):
    """Verdict options for agent analyses."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"

class Evidence(BaseModel):
    """Evidence supporting an analysis."""
    source: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)

class AgentAnalysis(BaseModel):
    """Analysis from a single agent."""
    model_config = ConfigDict(protected_namespaces=())
    
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType
    agent_name: str
    model_used: str

    analysis: str
    verdict: Optional[Verdict] = None
    confidence: float = Field(ge=0.0, le=1.0)

    evidence: List[Evidence] = []
    reasoning_steps: List[str] = []
    assumptions: List[str] = []
    limitations: List[str] = []

    tokens_used: int = 0
    cost: float = 0.0
    processing_time_ms: int = 0

    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = {}


class JurorVote(BaseModel):
    """Individual juror's vote."""
    model_config = ConfigDict(protected_namespaces=()) 

    juror_id: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str

    key_factors: List[str] = []
    dissenting_points: List[str] = []

    model_used: Optional[str] = None  
    weight: float = 1.0
    cost: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class RoundSummary(BaseModel):
    """Summary of a single round of evaluation."""
    round_number: int
    agent_analyses: List[AgentAnalysis]
    
    aggregated_analysis: str
    aggregated_insights: List[str]
    consensus_points: List[str]
    disagreement_points: List[str]
    
    average_confidence: float = Field(ge=0.0, le=1.0)
    confidence_variance: float = Field(ge=0.0)
    
    total_tokens: int = 0
    total_cost: float = 0.0
    processing_time_ms: int = 0


class JuryDecision(BaseModel):
    """Aggregated jury decision."""
    model_config = ConfigDict(protected_namespaces=()) 

    votes: List[JurorVote]

    majority_verdict: Verdict
    vote_distribution: Dict[str, int]
    model_distribution: Optional[Dict[str, Dict[str, int]]] = None  

    aggregate_confidence: float = Field(ge=0.0, le=1.0)
    weighted_confidence: float = Field(ge=0.0, le=1.0)
    consensus_level: float = Field(ge=0.0, le=1.0)

    key_agreements: List[str] = []
    key_disagreements: List[str] = []
    minority_report: Optional[str] = None

    total_cost: float = 0.0


class CalibrationFeatures(BaseModel):
    """Features used for confidence calibration."""
    model_config = ConfigDict(protected_namespaces=())
    
    base_confidence: float = Field(ge=0.0, le=1.0)
    agent_agreement: float = Field(ge=0.0, le=1.0)
    evidence_strength: float = Field(ge=0.0, le=1.0)
    reasoning_coherence: float = Field(ge=0.0, le=1.0)
    model_diversity: float = Field(ge=0.0, le=1.0)
    consensus_strength: float = Field(ge=0.0, le=1.0)
    token_efficiency: float = Field(ge=0.0, le=1.0)
    round_consistency: float = Field(ge=0.0, le=1.0)


class FinalJudgment(BaseModel):
    """Final judgment after all rounds and calibration."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    question: str
    answer: str
    context: Optional[str] = None
    
    is_correct: Verdict
    raw_confidence: float = Field(ge=0.0, le=1.0)
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    
    executive_summary: str
    detailed_rationale: str
    
    evidence_refs: List[Evidence] = []
    round_summaries: List[RoundSummary] = []
    jury_decision: JuryDecision
    calibration_features: CalibrationFeatures
    
    models_used: List[str] = []
    total_tokens: int = 0
    total_cost: float = 0.0
    processing_time_ms: int = 0
    
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = {}


class EvaluationRequest(BaseModel):
    """Request for evaluation."""
    question: str
    answer: str
    context: Optional[str] = None
    
    max_rounds: Optional[int] = None
    max_cost: Optional[float] = None
    min_confidence: Optional[float] = None
    
    required_agents: List[AgentType] = []
    excluded_models: List[str] = []
    
    metadata: Dict[str, Any] = {}


class EvaluationResult(BaseModel):
    """Complete evaluation result."""
    request: EvaluationRequest
    judgment: Optional[FinalJudgment] = None  # Make judgment optional
    
    success: bool = True
    error: Optional[str] = None
    warnings: List[str] = []
    recommendations: List[str] = []


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    id: str
    provider: ModelProvider
    name: str
    size: ModelSize
    
    context_length: int
    cost_per_1k_prompt: float
    cost_per_1k_completion: float
    
    capabilities: List[str] = []
    recommended_for: List[AgentType] = []
    
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 150000
    
    quality_score: float = Field(default=0.8, ge=0.0, le=1.0)
    speed_score: float = Field(default=0.8, ge=0.0, le=1.0)
    
    metadata: Dict[str, Any] = {}
