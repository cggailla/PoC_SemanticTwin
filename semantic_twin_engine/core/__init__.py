"""
Core module for the Semantic Twin Engine.

This module contains the infrastructure layer:
- interfaces: Abstract Base Classes (The Contract)
- config_loader: Pydantic models for configuration
- data_manager: Handles I/O and result aggregation
- orchestrator: Main pipeline runner
- exceptions: Custom exceptions
- embedding_store: Cached embedding storage
"""

from core.interfaces import ProbeInterface
from core.config_loader import Settings, EntityConfig, ProbeConfig
from core.orchestrator import Orchestrator
from core.data_manager import DataManager
from core.embedding_store import EmbeddingStore, get_store
from core.exceptions import (
    SemanticTwinError,
    ConfigurationError,
    ProbeExecutionError,
    APIError,
)

__all__ = [
    "ProbeInterface",
    "Settings",
    "EntityConfig",
    "ProbeConfig",
    "Orchestrator",
    "DataManager",
    "EmbeddingStore",
    "get_store",
    "SemanticTwinError",
    "ConfigurationError",
    "ProbeExecutionError",
    "APIError",
]
