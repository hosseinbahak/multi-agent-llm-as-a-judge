# multi_agent_llm_judge/config/schemas.py
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict

from ..core.data_models import AgentType

class ModelProviderConfig(BaseModel):
    """Configuration for the LLM provider."""
    provider: str = "openai"
    api_key: Optional[str] = None

class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    provider_id: str
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    context_window: int = 8192

class JurorModelConfig(BaseModel):
    """Configuration for a specific juror model."""
    model: str
    weight: float = 1.0  
    count: int = 5
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None  

class JuryConfig(BaseModel):
    """Configuration for the jury."""
    model_config = ConfigDict(protected_namespaces=())

    model: Optional[str] = None  
    models: Optional[List[JurorModelConfig]] = None 
    
    num_jurors: Optional[int] = None 
    temperature: float = 0.8 
    max_tokens: int = 1500  
    retry_attempts: int = 2
    
    # Weights for juror score calculation
    historical_accuracy_weight: float = 0.4
    confidence_weight: float = 0.2
    coherence_weight: float = 0.2
    evidence_match_weight: float = 0.2
    
    # تنظیمات توزیع مدل‌ها
    model_distribution_strategy: str = "fixed"  # "fixed", "weighted_random", "performance_based"
    
    @field_validator('models', mode='after')
    def validate_models(cls, v, values):
        """اطمینان از اینکه حداقل یک روش تعریف شده است."""
        if v is None and values.data.get('model') is None:
            raise ValueError("Either 'model' or 'models' must be specified")
        return v
    
    def get_juror_configs(self) -> List[tuple[str, float, float, int]]:
        """برگرداندن لیست تنظیمات juror ها."""
        if self.models:
            configs = []
            for model_config in self.models:
                temp = model_config.temperature or self.temperature
                tokens = model_config.max_tokens or self.max_tokens
                for _ in range(model_config.count):
                    configs.append((
                        model_config.model,
                        model_config.weight,
                        temp,
                        tokens
                    ))
            return configs
        else:
            # حالت تک مدل
            return [(self.model, 1.0, self.temperature, self.max_tokens)] * (self.num_jurors or 5)

class AgentConfig(BaseModel):
    """Configuration for an agent."""
    name: str
    type: AgentType
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    max_retries: int = 3
    retry_attempts: int = 3
    system_prompt: Optional[str] = None
    custom_prompt: Optional[str] = None
    capabilities: List[str] = []
    max_cost_per_token: float = 0.01
    enabled: bool = True

class ExecutionConfig(BaseModel):
    """Configuration for execution flow."""
    num_rounds: int = 2
    parallel_agents: bool = True
    parallel_jurors: bool = True
    parallel_limit: int = 5
    early_stopping: bool = True
    early_stopping_confidence: float = 0.95

class CalibrationConfig(BaseModel):
    """Configuration for the calibration component."""
    model_config = {"protected_namespaces": ()}

    enabled: bool = True
    method: str = "logistic"
    model_path: Optional[str] = "calibration_model.pkl"

class CacheConfig(BaseModel):
    """Configuration for caching."""
    enabled: bool = True
    ttl: int = 3600
    backend: str = "memory"
    max_size: int = 10000
    cache_dir: str = "./cache"

class RoundTableConfig(BaseModel):
    """Root configuration for the entire system."""
    models: Dict[str, ModelConfig] = {}
    provider: ModelProviderConfig = ModelProviderConfig()
    agents: List[AgentConfig] = []
    jury: JuryConfig = JuryConfig()
    execution: ExecutionConfig = ExecutionConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    cache: CacheConfig = CacheConfig()

    @classmethod
    def default(cls):
        """Get default configuration."""
        return cls(
            models={},
            provider=ModelProviderConfig(),
            agents=[
                AgentConfig(
                    name="ChainOfThoughtAgent",
                    type=AgentType.CHAIN_OF_THOUGHT,
                    model="openai/gpt-4o-mini",
                    enabled=True,
                    retry_attempts=3
                ),
                AgentConfig(
                    name="AdversaryAgent",
                    type=AgentType.ADVERSARY,
                    model="openai/gpt-4o",
                    enabled=True,
                    retry_attempts=3
                ),
                AgentConfig(
                    name="ChallengerAgent",
                    type=AgentType.CHALLENGER,
                    model="openai/gpt-4-turbo",
                    enabled=True,
                    retry_attempts=3
                ),
                AgentConfig(
                    name="SynthesizerAgent",
                    type=AgentType.SYNTHESIZER,
                    model="openai/gpt-4o-mini",
                    enabled=True,
                    retry_attempts=3
                ),
                AgentConfig(
                    name="RetrievalVerifierAgent",
                    type=AgentType.RETRIEVAL_VERIFIER,
                    model="openai/gpt-4o-mini",
                    enabled=True,
                    retry_attempts=3
                ),
                AgentConfig(
                    name="BiasAuditorAgent",
                    type=AgentType.BIAS_AUDITOR,
                    model="openai/gpt-4o-mini",
                    enabled=True,
                    retry_attempts=3
                ),
                AgentConfig(
                    name="MetaQAAgent",
                    type=AgentType.META_QA,
                    model="openai/gpt-4o-mini",
                    enabled=True,
                    retry_attempts=3
                ),
            ],
            jury=JuryConfig(model="openai/gpt-4o-mini"),
            execution=ExecutionConfig(),
            calibration=CalibrationConfig(),
            cache=CacheConfig()
        )
