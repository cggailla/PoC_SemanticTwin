"""
Cluster Validator logic for the Semantic Twin Engine.

Validates semantic anchor quality using density, variance, and separation metrics.
"""

import logging
from typing import Any

import numpy as np

from core.interfaces import ProbeContext, ProbeResult
from modules.base_probe import BaseProbe
from modules.cluster_validator.models import (
    ClusterValidatorConfig,
    ClusterMetrics,
    OutlierInfo,
    ClusterValidationReport,
)

logger = logging.getLogger(__name__)


class ClusterValidator(BaseProbe):
    """Validates the quality of semantic anchor clusters.
    
    This module audits anchor definitions BEFORE entity projection to ensure:
    - Clusters are dense (words are semantically coherent)
    - Clusters are separated (legacy and strategy are distinct)
    - No outliers are polluting the centroids
    
    Traffic Light Classification:
        🟢 VALID: High density, high separation, few outliers
        🟠 WARNING: High variance or too many outliers
        🔴 BROKEN: Centroids too close (concept overlap)
    """
    
    @property
    def name(self) -> str:
        return "cluster_validator"
    
    def run(self, context: ProbeContext) -> ProbeResult:
        """Execute cluster validation on all configured dimensions.
        
        Args:
            context: Probe execution context with config.
        
        Returns:
            ProbeResult containing validation reports for each dimension.
        """
        try:
            # Load config
            config = ClusterValidatorConfig(
                **context.probe_config.get("thresholds", {})
            )
            
            dimensions_config = context.probe_config.get("dimensions", {})
            
            reports: list[ClusterValidationReport] = []
            
            for dim_name, dim_config in dimensions_config.items():
                try:
                    report = self.validate_dimension(
                        dimension_name=dim_name,
                        anchor_a=dim_config.get("anchor_a", []),
                        anchor_b=dim_config.get("anchor_b", []),
                        config=config,
                    )
                    reports.append(report)
                    logger.info(
                        "Validated dimension '%s': %s",
                        dim_name,
                        report.to_emoji_status()
                    )
                except Exception as e:
                    logger.error("Failed to validate dimension '%s': %s", dim_name, e)
            
            # Aggregate status
            statuses = [r.status for r in reports]
            overall = (
                "BROKEN" if "BROKEN" in statuses else
                "WARNING" if "WARNING" in statuses else
                "VALID"
            )
            
            return self._create_result(
                success=True,
                data={
                    "overall_status": overall,
                    "dimensions": [r.model_dump() for r in reports],
                },
                metadata={"dimensions_validated": len(reports)},
            )
            
        except Exception as e:
            return self._handle_error(e, "Cluster validation failed")
    
    def validate_dimension(
        self,
        dimension_name: str,
        anchor_a: list[str],
        anchor_b: list[str],
        config: ClusterValidatorConfig | None = None,
    ) -> ClusterValidationReport:
        """Validate a single dimension's anchor clusters.
        
        Args:
            dimension_name: Name of the dimension.
            anchor_a: Legacy/anchor A terms.
            anchor_b: Strategy/anchor B terms.
            config: Validation thresholds.
        
        Returns:
            ClusterValidationReport with metrics and recommendations.
        """
        config = config or ClusterValidatorConfig()
        
        logger.info(
            "Validating dimension '%s' with %d/%d terms",
            dimension_name,
            len(anchor_a),
            len(anchor_b)
        )
        
        # Get embeddings (uses cache)
        embeddings_a = np.array(self.get_embeddings(anchor_a))
        embeddings_b = np.array(self.get_embeddings(anchor_b))
        
        # Compute centroids
        centroid_a = np.mean(embeddings_a, axis=0)
        centroid_b = np.mean(embeddings_b, axis=0)
        
        # Compute metrics per cluster
        metrics_a = self._compute_cluster_metrics(
            embeddings_a, centroid_a, "legacy"
        )
        metrics_b = self._compute_cluster_metrics(
            embeddings_b, centroid_b, "strategy"
        )
        
        # Compute axis separation
        axis_separation = float(self.cosine_distance(centroid_a, centroid_b))
        
        # Detect outliers
        outliers = []
        outliers.extend(self._detect_outliers(
            terms=anchor_a,
            embeddings=embeddings_a,
            centroid=centroid_a,
            cluster="legacy",
            mean_dist=metrics_a.mean_distance,
            std_dist=metrics_a.variance,
            threshold=config.z_score_threshold,
        ))
        outliers.extend(self._detect_outliers(
            terms=anchor_b,
            embeddings=embeddings_b,
            centroid=centroid_b,
            cluster="strategy",
            mean_dist=metrics_b.mean_distance,
            std_dist=metrics_b.variance,
            threshold=config.z_score_threshold,
        ))
        
        # Classify status
        status = self._classify_status(
            density_a=metrics_a.density,
            density_b=metrics_b.density,
            variance_a=metrics_a.variance,
            variance_b=metrics_b.variance,
            separation=axis_separation,
            outlier_count=len(outliers),
            config=config,
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            status=status,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            separation=axis_separation,
            outliers=outliers,
            config=config,
        )
        
        return ClusterValidationReport(
            dimension=dimension_name,
            status=status,
            metrics={
                "legacy": metrics_a,
                "strategy": metrics_b,
            },
            axis_separation=round(axis_separation, 4),
            outliers_detected=outliers,
            recommendation=recommendation,
        )
    
    def _compute_cluster_metrics(
        self,
        embeddings: np.ndarray,
        centroid: np.ndarray,
        cluster_name: str,
    ) -> ClusterMetrics:
        """Compute density and variance metrics for a cluster.
        
        Args:
            embeddings: Array of word embeddings.
            centroid: Cluster centroid vector.
            cluster_name: Name for logging.
        
        Returns:
            ClusterMetrics with density, variance, etc.
        """
        # Calculate cosine distances to centroid
        distances = np.array([
            self.cosine_distance(emb, centroid)
            for emb in embeddings
        ])
        
        mean_distance = float(np.mean(distances))
        variance = float(np.std(distances))
        density = 1.0 - mean_distance  # Higher density = more compact
        
        logger.debug(
            "Cluster '%s': density=%.3f, variance=%.3f",
            cluster_name,
            density,
            variance
        )
        
        return ClusterMetrics(
            density=round(density, 4),
            variance=round(variance, 4),
            word_count=len(embeddings),
            mean_distance=round(mean_distance, 4),
        )
    
    def _detect_outliers(
        self,
        terms: list[str],
        embeddings: np.ndarray,
        centroid: np.ndarray,
        cluster: str,
        mean_dist: float,
        std_dist: float,
        threshold: float,
    ) -> list[OutlierInfo]:
        """Detect outlier words using Z-score analysis.
        
        Args:
            terms: List of anchor terms.
            embeddings: Corresponding embeddings.
            centroid: Cluster centroid.
            cluster: Cluster name ("legacy" or "strategy").
            mean_dist: Mean distance in cluster.
            std_dist: Standard deviation of distances.
            threshold: Z-score threshold for outlier detection.
        
        Returns:
            List of detected outliers.
        """
        outliers = []
        
        if std_dist == 0:
            return outliers  # No variance, no outliers
        
        for term, embedding in zip(terms, embeddings):
            distance = self.cosine_distance(embedding, centroid)
            z_score = (distance - mean_dist) / std_dist
            
            if z_score > threshold:
                reason = self._explain_outlier(term, z_score, distance, cluster)
                outliers.append(OutlierInfo(
                    word=term,
                    cluster=cluster,
                    z_score=round(z_score, 2),
                    distance=round(distance, 4),
                    reason=reason,
                ))
                logger.debug("Outlier detected: '%s' (z=%.2f)", term, z_score)
        
        return outliers
    
    def _explain_outlier(
        self,
        term: str,
        z_score: float,
        distance: float,
        cluster: str,
    ) -> str:
        """Generate human-readable explanation for outlier.
        
        Args:
            term: The outlier term.
            z_score: Statistical measure.
            distance: Distance from centroid.
            cluster: Which cluster.
        
        Returns:
            Explanation string.
        """
        if z_score > 3.0:
            return f"Extreme outlier (Z-score {z_score:.1f}). Likely a different concept entirely."
        elif z_score > 2.5:
            return f"Strong outlier (Z-score {z_score:.1f}). Pulls the {cluster} cluster away from core meaning."
        else:
            return f"Borderline outlier (Z-score {z_score:.1f}). Consider removing for tighter cluster."
    
    def _classify_status(
        self,
        density_a: float,
        density_b: float,
        variance_a: float,
        variance_b: float,
        separation: float,
        outlier_count: int,
        config: ClusterValidatorConfig,
    ) -> str:
        """Classify dimension into traffic light status.
        
        Args:
            density_a: Legacy cluster density.
            density_b: Strategy cluster density.
            variance_a: Legacy cluster variance.
            variance_b: Strategy cluster variance.
            separation: Distance between centroids.
            outlier_count: Number of outliers detected.
            config: Thresholds from config.
        
        Returns:
            Status string: "VALID", "WARNING", or "BROKEN".
        """
        # BROKEN: Centroids too close (concept overlap)
        if separation < config.min_separation:
            return "BROKEN"
        
        # WARNING: High variance or too many outliers
        max_variance = 0.15
        if (variance_a > max_variance or variance_b > max_variance or 
            outlier_count >= config.max_outliers):
            return "WARNING"
        
        # Check density
        if density_a < config.min_density or density_b < config.min_density:
            return "WARNING"
        
        return "VALID"
    
    def _generate_recommendation(
        self,
        status: str,
        metrics_a: ClusterMetrics,
        metrics_b: ClusterMetrics,
        separation: float,
        outliers: list[OutlierInfo],
        config: ClusterValidatorConfig,
    ) -> str:
        """Generate actionable recommendation.
        
        Args:
            status: Current status.
            metrics_a: Legacy metrics.
            metrics_b: Strategy metrics.
            separation: Axis separation.
            outliers: Detected outliers.
            config: Thresholds.
        
        Returns:
            Recommendation string.
        """
        if status == "VALID":
            return "Anchors are well-defined. No action needed."
        
        if status == "BROKEN":
            return (
                f"CRITICAL: Centroid separation ({separation:.3f}) is below "
                f"threshold ({config.min_separation}). The two concepts overlap. "
                "Consider redefining anchors with more distinct terms."
            )
        
        # WARNING status
        parts = []
        
        if outliers:
            outlier_words = [o.word for o in outliers[:5]]
            parts.append(f"Remove outliers: {', '.join(repr(w) for w in outlier_words)}")
        
        if metrics_a.variance > 0.15:
            parts.append("Tighten legacy cluster (high variance)")
        
        if metrics_b.variance > 0.15:
            parts.append("Tighten strategy cluster (high variance)")
        
        if metrics_a.density < config.min_density:
            parts.append(f"Improve legacy density ({metrics_a.density:.2f} < {config.min_density})")
        
        if metrics_b.density < config.min_density:
            parts.append(f"Improve strategy density ({metrics_b.density:.2f} < {config.min_density})")
        
        return " | ".join(parts) if parts else "Review anchor terms for coherence."
