"""
Vector Probe: Embeddings & Geometry Analysis.

This probe measures semantic positioning of entities on configurable
dimensions using centroid-based drift score calculation.
"""

import logging
from typing import Any

import numpy as np

from core.exceptions import ProbeExecutionError
from core.interfaces import ProbeContext, ProbeResult
from modules.base_probe import BaseProbe
from modules.vector_probe.models import DimensionConfig, DimensionResult, VectorProbeOutput

logger = logging.getLogger(__name__)


class VectorProbe(BaseProbe):
    """Probe for embedding-based semantic positioning.
    
    This probe:
    1. Loads dimension configurations from context
    2. Computes centroids for each dimension's anchor poles
    3. Embeds the entity using contextual prompts
    4. Calculates drift scores positioning entity between poles
    
    Attributes:
        name: "vector_probe"
    """
    
    @property
    def name(self) -> str:
        """Return the probe identifier."""
        return "vector_probe"
    
    def _load_dimensions(self, context: ProbeContext) -> list[DimensionConfig]:
        """Load dimension configurations from probe config.
        
        Args:
            context: Probe context containing configuration.
        
        Returns:
            List of DimensionConfig objects.
        
        Raises:
            ProbeExecutionError: If dimensions config is invalid.
        """
        dimensions_raw = context.probe_config.get("dimensions", {})
        
        if not dimensions_raw:
            raise ProbeExecutionError(
                "No dimensions configured for vector_probe",
                probe_name=self.name
            )
        
        dimensions = []
        for dim_name, dim_config in dimensions_raw.items():
            try:
                dimension = DimensionConfig(
                    name=dim_name,
                    anchor_a=dim_config.get("anchor_a", []),
                    anchor_b=dim_config.get("anchor_b", []),
                    contextual_prompt=dim_config.get("contextual_prompt", ""),
                )
                dimensions.append(dimension)
                logger.debug("Loaded dimension: %s", dim_name)
            except Exception as e:
                logger.warning("Failed to load dimension %s: %s", dim_name, e)
        
        return dimensions
    
    def _analyze_dimension(
        self,
        dimension: DimensionConfig,
        entity_name: str,
    ) -> DimensionResult:
        """Analyze entity position on a single dimension.
        
        Args:
            dimension: The dimension configuration.
            entity_name: Name of the entity to analyze.
        
        Returns:
            DimensionResult with drift score and distances.
        """
        logger.info(
            "Analyzing dimension '%s' with %d/%d anchor terms",
            dimension.name,
            len(dimension.anchor_a),
            len(dimension.anchor_b)
        )
        
        # Compute centroids for both poles (uses cache)
        centroid_a = self.get_centroid(
            dimension.anchor_a,
            dimension=dimension.name,
            pole="legacy"
        )
        logger.debug("Got legacy centroid (anchor_a)")
        
        centroid_b = self.get_centroid(
            dimension.anchor_b,
            dimension=dimension.name,
            pole="strategy"
        )
        logger.debug("Got strategy centroid (anchor_b)")
        
        # Embed entity using contextual prompt
        contextual_prompt = dimension.contextual_prompt.format(entity=entity_name)
        entity_embeddings = self.get_embeddings([contextual_prompt])
        entity_vector = entity_embeddings[0]
        logger.debug("Computed entity vector from contextual prompt")
        
        # Calculate drift score
        drift_data = self.calculate_drift_score(
            entity_vector=entity_vector,
            centroid_a=centroid_a,
            centroid_b=centroid_b,
        )
        
        return DimensionResult(
            dimension_name=dimension.name,
            drift_score=drift_data["drift_score"],
            distance_to_legacy=drift_data["distance_to_legacy"],
            distance_to_strategy=drift_data["distance_to_strategy"],
            centroid_separation=drift_data["centroid_separation"],
        )
    
    def run(self, context: ProbeContext) -> ProbeResult:
        """Execute the vector analysis on all configured dimensions.
        
        Args:
            context: Probe context with entity data and dimension configs.
        
        Returns:
            ProbeResult with dimension analysis results.
        """
        try:
            # Load dimension configurations
            dimensions = self._load_dimensions(context)
            
            if not dimensions:
                return self._create_result(
                    success=False,
                    data={},
                    errors=["No valid dimensions found in configuration"],
                )
            
            # Analyze each dimension
            output = VectorProbeOutput(entity_name=context.entity_name)
            
            for dimension in dimensions:
                try:
                    result = self._analyze_dimension(dimension, context.entity_name)
                    output.dimensions[dimension.name] = result
                    logger.info(
                        "Dimension '%s': drift_score=%.4f",
                        dimension.name,
                        result.drift_score
                    )
                except Exception as e:
                    logger.error("Failed to analyze dimension '%s': %s", dimension.name, e)
            
            # Generate summary
            if output.dimensions:
                avg_drift = sum(
                    d.drift_score for d in output.dimensions.values()
                ) / len(output.dimensions)
                output.summary = (
                    f"Entity '{context.entity_name}' analyzed on "
                    f"{len(output.dimensions)} dimension(s). "
                    f"Average drift score: {avg_drift:.4f}"
                )
            
            return self._create_result(
                success=True,
                data=output.model_dump(),
                metadata={
                    "dimensions_analyzed": len(output.dimensions),
                    "entity": context.entity_name,
                },
            )
            
        except ProbeExecutionError:
            raise
        except Exception as e:
            return self._handle_error(e, "Vector analysis failed")
