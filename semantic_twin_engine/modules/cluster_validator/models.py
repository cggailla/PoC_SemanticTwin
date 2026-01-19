"""
Pydantic models for the Cluster Validator module.

Defines data contracts for cluster validation input/output.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ClusterValidatorConfig(BaseModel):
    """Configuration for cluster validation thresholds.
    
    Attributes:
        z_score_threshold: Z-score above which a word is considered an outlier.
        min_density: Minimum acceptable cluster density (0-1).
        min_separation: Minimum acceptable distance between centroids.
        max_outliers: Maximum number of outliers before warning.
    """
    z_score_threshold: float = Field(default=2.0, ge=1.0, le=4.0)
    min_density: float = Field(default=0.80, ge=0.0, le=1.0)
    min_separation: float = Field(default=0.25, ge=0.0, le=1.0)
    max_outliers: int = Field(default=3, ge=0)


class ClusterMetrics(BaseModel):
    """Metrics for a single cluster (legacy or strategy).
    
    Attributes:
        density: How tightly packed words are around centroid (0-1, higher is better).
        variance: Standard deviation of distances to centroid (lower is better).
        word_count: Number of words in the cluster.
        mean_distance: Average cosine distance to centroid.
    """
    density: float = Field(description="Cluster cohesion (1 - mean_distance)")
    variance: float = Field(description="Std dev of distances")
    word_count: int = Field(description="Number of anchor terms")
    mean_distance: float = Field(description="Average distance to centroid")


class OutlierInfo(BaseModel):
    """Information about a detected outlier word.
    
    Attributes:
        word: The outlier term.
        cluster: Which cluster it belongs to ("legacy" or "strategy").
        z_score: How many standard deviations from mean.
        distance: Cosine distance to centroid.
        reason: Human-readable explanation.
    """
    word: str
    cluster: Literal["legacy", "strategy"]
    z_score: float
    distance: float
    reason: str


class ClusterValidationReport(BaseModel):
    """Complete validation report for a dimension.
    
    Attributes:
        dimension: Name of the validated dimension.
        status: Traffic light status (VALID, WARNING, BROKEN).
        metrics: Per-cluster metrics (legacy and strategy).
        axis_separation: Cosine distance between centroids.
        outliers_detected: List of outlier words with details.
        recommendation: Actionable suggestion for improvement.
    """
    dimension: str
    status: Literal["VALID", "WARNING", "BROKEN"]
    metrics: dict[str, ClusterMetrics] = Field(
        description="Metrics per cluster: 'legacy' and 'strategy'"
    )
    axis_separation: float = Field(
        description="Distance between legacy and strategy centroids"
    )
    outliers_detected: list[OutlierInfo] = Field(default_factory=list)
    recommendation: str = Field(
        description="Actionable suggestion based on analysis"
    )
    
    def to_emoji_status(self) -> str:
        """Return status with emoji prefix."""
        emojis = {"VALID": "🟢", "WARNING": "🟠", "BROKEN": "🔴"}
        return f"{emojis.get(self.status, '⚪')} {self.status}"
