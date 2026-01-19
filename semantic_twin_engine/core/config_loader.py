"""
Configuration loader for the Semantic Twin Engine.

This module provides Pydantic models for type-safe configuration loading
from YAML files and environment variables.
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class EntityConfig(BaseModel):
    """Configuration for the corporate entity being analyzed.
    
    Attributes:
        name: The official name of the corporate entity.
        legacy_identity: Keywords/phrases representing current perception.
        strategic_target: Keywords/phrases representing desired perception.
    """
    
    name: str = Field(..., min_length=1, description="Corporate entity name")
    legacy_identity: list[str] = Field(
        default_factory=list,
        description="Keywords representing current perception"
    )
    strategic_target: list[str] = Field(
        default_factory=list,
        description="Keywords representing target perception"
    )
    
    @field_validator("legacy_identity", "strategic_target")
    @classmethod
    def validate_non_empty_strings(cls, v: list[str]) -> list[str]:
        """Ensure all items in the list are non-empty strings."""
        return [item.strip() for item in v if item.strip()]


class AnchorConfig(BaseModel):
    """Configuration for reference anchors used in analysis.
    
    Anchors are reference points (e.g., competitors, industry leaders)
    used to contextualize the entity's position in the latent space.
    
    Attributes:
        entities: List of anchor entity names.
        concepts: List of anchor concept keywords.
    """
    
    entities: list[str] = Field(
        default_factory=list,
        description="Reference entity names for comparison"
    )
    concepts: list[str] = Field(
        default_factory=list,
        description="Reference concepts for analysis"
    )


class ProbeConfig(BaseModel):
    """Configuration for individual probe modules.
    
    Attributes:
        enabled: Whether this probe should be executed.
        params: Probe-specific parameters.
    """
    
    enabled: bool = Field(default=True, description="Whether probe is active")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Probe-specific parameters"
    )


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI API integration.
    
    Attributes:
        model: The model to use for embeddings/completions.
        embedding_model: Specific model for embeddings.
        temperature: Temperature setting (0 for deterministic).
        seed: Random seed for reproducibility.
        max_retries: Maximum API retry attempts.
    """
    
    model: str = Field(default="gpt-4o", description="Model for completions")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model for embeddings"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature (0 for deterministic)"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    max_retries: int = Field(default=3, ge=1, description="Max API retry attempts")


class OutputConfig(BaseModel):
    """Configuration for output handling.
    
    Attributes:
        directory: Path to output directory.
        format: Output format (json, yaml).
        include_metadata: Whether to include execution metadata.
    """
    
    directory: Path = Field(
        default=Path("./output"),
        description="Output directory path"
    )
    format: str = Field(default="json", pattern="^(json|yaml)$")
    include_metadata: bool = Field(
        default=True,
        description="Include execution metadata in output"
    )


class Settings(BaseModel):
    """Root configuration model for the Semantic Twin Engine.
    
    This is the main configuration object loaded from settings.yaml.
    All variable data (entity names, anchors, etc.) comes from this config.
    
    Attributes:
        entity: Configuration for the target entity.
        anchors: Reference anchor configuration.
        probes: Per-probe configuration.
        openai: OpenAI API configuration.
        output: Output handling configuration.
    """
    
    entity: EntityConfig
    anchors: AnchorConfig = Field(default_factory=AnchorConfig)
    probes: dict[str, ProbeConfig] = Field(default_factory=dict)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_settings(config_path: Path | None = None) -> Settings:
    """Load and validate settings from YAML configuration file.
    
    This function loads environment variables from .env file, then loads
    and validates the settings.yaml configuration.
    
    Args:
        config_path: Optional path to settings.yaml. Defaults to
            ./config/settings.yaml relative to project root.
    
    Returns:
        Validated Settings object.
    
    Raises:
        ConfigurationError: If the configuration file cannot be loaded
            or validation fails.
    """
    # Load environment variables
    load_dotenv()
    
    # Determine config path
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            details={"path": str(config_path)}
        )
    
    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            "Failed to parse YAML configuration",
            details={"path": str(config_path), "error": str(e)}
        ) from e
    
    if raw_config is None:
        raise ConfigurationError(
            "Configuration file is empty",
            details={"path": str(config_path)}
        )
    
    try:
        settings = Settings(**raw_config)
        logger.info("Configuration loaded successfully from %s", config_path)
        return settings
    except Exception as e:
        raise ConfigurationError(
            "Configuration validation failed",
            details={"path": str(config_path), "error": str(e)}
        ) from e
