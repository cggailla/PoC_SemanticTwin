"""
Pydantic models for the Vector Probe module.

These models define the data contracts for dimension configuration
and analysis results.
"""

from pydantic import BaseModel, Field


class DimensionConfig(BaseModel):
    """Configuration for a single semantic dimension.
    
    Each dimension defines two anchors (poles) and a contextual prompt
    template for positioning the entity on that axis.
    
    Attributes:
        name: Human-readable dimension name.
        anchor_a: List of terms representing the legacy/source pole.
        anchor_b: List of terms representing the strategy/target pole.
        contextual_prompt: Template for entity embedding (use {entity} placeholder).
    """
    
    name: str = Field(..., description="Dimension identifier")
    anchor_a: list[str] = Field(..., description="Legacy pole terms")
    anchor_b: list[str] = Field(..., description="Strategy pole terms")
    contextual_prompt: str = Field(
        ...,
        description="Prompt template with {entity} placeholder"
    )


class DimensionResult(BaseModel):
    """Result of a single dimension analysis.
    
    Contains the drift score and distance measurements.
    
    Attributes:
        dimension_name: Identifier of the analyzed dimension.
        drift_score: Position on axis centered at 0 (negative = Legacy, positive = Strategy).
        distance_to_legacy: Cosine distance to anchor_a centroid.
        distance_to_strategy: Cosine distance to anchor_b centroid.
        centroid_separation: Cosine distance between the two centroids.
        interpretation: Human-readable interpretation of the score.
    """
    
    dimension_name: str
    drift_score: float = Field(..., description="Centered at 0: negative = Legacy, positive = Strategy")
    distance_to_legacy: float = Field(..., ge=0.0)
    distance_to_strategy: float = Field(..., ge=0.0)
    centroid_separation: float = Field(default=0.0, ge=0.0)
    interpretation: str = ""
    
    def model_post_init(self, __context) -> None:
        """Generate interpretation based on drift score."""
        if not self.interpretation:
            if self.drift_score < -0.5:
                self.interpretation = "Strongly Legacy"
            elif self.drift_score < -0.15:
                self.interpretation = "Leaning Legacy"
            elif self.drift_score <= 0.15:
                self.interpretation = "Neutral"
            elif self.drift_score <= 0.5:
                self.interpretation = "Leaning Strategy"
            else:
                self.interpretation = "Strongly Strategy"


class VectorProbeOutput(BaseModel):
    """Complete output of the Vector Probe.
    
    Aggregates results from all analyzed dimensions.
    
    Attributes:
        entity_name: The analyzed entity.
        dimensions: Dictionary of dimension results.
        summary: Overall positioning summary.
    """
    
    entity_name: str
    dimensions: dict[str, DimensionResult] = Field(default_factory=dict)
    summary: str = ""
