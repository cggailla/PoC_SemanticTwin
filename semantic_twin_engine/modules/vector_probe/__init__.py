"""
Vector Probe module for the Semantic Twin Engine.

This module analyzes embeddings and geometric relationships in the
LLM's latent space to measure semantic distance between entity
representations.
"""

from modules.vector_probe.logic import VectorProbe
from modules.vector_probe.models import DimensionConfig, DimensionResult, VectorProbeOutput

__all__ = ["VectorProbe", "DimensionConfig", "DimensionResult", "VectorProbeOutput"]
