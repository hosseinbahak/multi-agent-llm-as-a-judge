# multi_agent_llm_judge/utils/validators.py
from ..config.schemas import RoundTableConfig
from ..core.exceptions import ConfigurationError

def validate_config_dependencies(config: RoundTableConfig):
    """
    Performs post-load validation of the configuration to check for logical
    dependencies and inconsistencies.

    Args:
        config: The fully loaded RoundTableConfig object.

    Raises:
        ConfigurationError: If a validation check fails.
    """
    # 1. Check if all referenced models are defined
    defined_models = set()
    for provider in config.models.providers:
        for model in provider.models:
            defined_models.add(model.id)
            
    for agent_config in config.agents.active_agents:
        if agent_config.model_id not in defined_models:
            raise ConfigurationError(
                f"Agent '{agent_config.name}' requires model '{agent_config.model_id}', which is not defined in models.yaml."
            )
            
    if config.jury.model_id not in defined_models:
        raise ConfigurationError(
            f"Jury requires model '{config.jury.model_id}', which is not defined in models.yaml."
        )

    # 2. Check for logical jury configuration
    if config.jury.num_jurors <= 0:
        raise ConfigurationError("Jury 'num_jurors' must be a positive integer.")
    if config.jury.num_jurors % 2 == 0:
        raise ConfigurationError("Jury 'num_jurors' should be an odd number to prevent ties.")

    # 3. Check execution parameters
    if config.execution.max_rounds <= 0:
         raise ConfigurationError("Execution 'max_rounds' must be a positive integer.")

    # 4. Check that there is at least one active agent
    if not config.agents.active_agents:
        raise ConfigurationError("There are no active agents defined in agents.yaml. At least one agent is required.")

    # If all checks pass
    return True