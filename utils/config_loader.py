# multi_agent_llm_judge/utils/config_loader.py
"""Configuration loading utilities."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from ..config.schemas import RoundTableConfig

def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """Load a single YAML file."""
    if file_path.exists():
        with open(file_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}

def load_config_file(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or directory."""
    if config_path:
        path = Path(config_path)
    else:
        # Look for config in multiple locations
        possible_paths = [
            Path.cwd() / "config",
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent / "config",
            Path(__file__).parent.parent / "config.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                break
        else:
            logger.warning("No config found, using defaults")
            return {}

    # If it's a directory, load all YAML files and merge them
    if path.is_dir():
        logger.info(f"Loading configuration from directory {path}")
        config_dict = {}
        
        # Define the order and mapping of config files
        config_files = {
            "system.yaml": None,  # Root level config
            "models.yaml": "models",
            "agents.yaml": "agents",
            "jury.yaml": "jury",
            "execution.yaml": "execution",
            "calibration.yaml": "calibration",
            "cache.yaml": "cache"
        }
        
        # First check defaults subdirectory
        defaults_dir = path / "defaults"
        if defaults_dir.exists():
            for filename, key in config_files.items():
                file_path = defaults_dir / filename
                if file_path.exists():
                    logger.debug(f"Loading {file_path}")
                    file_config = load_yaml_file(file_path)
                    
                    if key is None:
                        # Root level config (system.yaml)
                        config_dict.update(file_config)
                    elif key in file_config:
                        config_dict[key] = file_config[key]
                    else:
                        # If the file doesn't have the expected key, use the whole content
                        config_dict[key] = file_config
        
        # Then load files from main config directory (override defaults)
        for filename, key in config_files.items():
            file_path = path / filename
            if file_path.exists():
                logger.debug(f"Loading {file_path} (overriding defaults)")
                file_config = load_yaml_file(file_path)
                
                if key is None:
                    config_dict.update(file_config)
                elif key in file_config:
                    config_dict[key] = file_config[key]
                else:
                    config_dict[key] = file_config
        
        # Load main config.yaml if exists (highest priority)
        main_config = path / "config.yaml"
        if main_config.exists():
            logger.debug(f"Loading main config from {main_config}")
            config_dict.update(load_yaml_file(main_config))
            
        return config_dict
        
    elif path.exists() and path.suffix in ['.yaml', '.yml']:
        # Single file
        logger.info(f"Loading configuration from {path}")
        return load_yaml_file(path)
    else:
        logger.warning(f"Config path {path} not found, using defaults")
        return {}

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
            
    return result

def load_config(config_path: Optional[str] = None) -> RoundTableConfig:
    """Load and validate configuration."""
    # Load from file or directory
    config_dict = load_config_file(config_path)

    # Use default configuration if empty
    if not config_dict:
        logger.info("Using default configuration")
        return RoundTableConfig.default()

    try:
        # Get default config
        default_config = RoundTableConfig.default()
        default_dict = default_config.model_dump()
        
        # Merge with loaded config
        final_config = merge_configs(default_dict, config_dict)
        
        # Validate and create config object
        config = RoundTableConfig(**final_config)
        logger.success("Configuration loaded and validated successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to validate configuration: {e}")
        logger.info("Falling back to default configuration")
        return RoundTableConfig.default()

def load_app_config(config_path: Optional[str] = None) -> RoundTableConfig:
    """Alias for load_config for backward compatibility."""
    return load_config(config_path)
