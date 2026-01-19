"""
Base Probe class for the Semantic Twin Engine.

All analytical probes must inherit from BaseProbe, which provides
common functionality and enforces the ProbeInterface contract.
"""

import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from core.embedding_store import EmbeddingStore, get_store
from core.exceptions import APIError, ProbeExecutionError
from core.interfaces import ProbeContext, ProbeInterface, ProbeResult

logger = logging.getLogger(__name__)


class BaseProbe(ProbeInterface):
    """Base class for all probe modules.
    
    This class provides common functionality shared by all probes:
    - Cached embedding retrieval via EmbeddingStore
    - Cosine similarity calculation
    - Centroid computation with caching
    - Standardized error handling
    
    Subclasses must implement:
    - name property: Unique probe identifier
    - run() method: Main analysis logic
    
    Example:
        class MyProbe(BaseProbe):
            @property
            def name(self) -> str:
                return "my_probe"
            
            def run(self, context: ProbeContext) -> ProbeResult:
                embeddings = self.get_embeddings(["text1", "text2"])
                similarity = self.cosine_similarity(embeddings[0], embeddings[1])
                return ProbeResult(...)
    """
    
    # Shared embedding store across all probes
    _store: EmbeddingStore | None = None
    
    def __init__(self, cache_dir: Path | str | None = None) -> None:
        """Initialize the base probe with embedding store.
        
        Args:
            cache_dir: Optional custom cache directory. Uses ./cache by default.
        """
        self._embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL", 
            "text-embedding-3-small"
        )
        
        # Use shared store or create new one
        if cache_dir:
            self._store = EmbeddingStore(cache_dir=cache_dir)
        elif BaseProbe._store is None:
            BaseProbe._store = get_store()
    
    @property
    def store(self) -> EmbeddingStore:
        """Get the embedding store instance.
        
        Returns:
            EmbeddingStore instance (shared or instance-specific).
        """
        if self._store is not None:
            return self._store
        if BaseProbe._store is None:
            BaseProbe._store = get_store()
        return BaseProbe._store
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier for this probe."""
        pass
    
    @abstractmethod
    def run(self, context: ProbeContext) -> ProbeResult:
        """Execute the probe analysis."""
        pass
    
    def get_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Get embeddings for a list of texts using cached store.
        
        This method uses the EmbeddingStore for caching, avoiding
        redundant API calls for previously seen texts.
        
        Args:
            texts: List of text strings to embed.
        
        Returns:
            List of numpy arrays containing embeddings.
        
        Raises:
            APIError: If the API call fails after retries.
        """
        return self.store.get_embeddings(texts)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text using cached store.
        
        Args:
            text: Text string to embed.
        
        Returns:
            Numpy array containing the embedding.
        """
        return self.store.get_embedding(text)
    
    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec_a: First vector.
            vec_b: Second vector.
        
        Returns:
            Cosine similarity score between -1 and 1.
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    @staticmethod
    def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine distance between two vectors.
        
        Cosine distance = 1 - cosine_similarity
        
        Args:
            vec_a: First vector.
            vec_b: Second vector.
        
        Returns:
            Cosine distance score between 0 and 2.
        """
        return 1.0 - BaseProbe.cosine_similarity(vec_a, vec_b)
    
    def get_centroid(
        self, 
        texts: list[str],
        dimension: str | None = None,
        pole: str | None = None,
    ) -> np.ndarray:
        """Compute the centroid (mean vector) for a list of texts.
        
        Uses cached centroids when dimension and pole are provided.
        
        Args:
            texts: List of text strings to embed and average.
            dimension: Optional dimension name for caching.
            pole: Optional pole name ('legacy' or 'strategy') for caching.
        
        Returns:
            Numpy array representing the centroid vector.
        
        Raises:
            APIError: If embedding retrieval fails.
            ValueError: If texts list is empty.
        """
        if not texts:
            raise ValueError("Cannot compute centroid of empty list")
        
        # Use cached centroid if dimension and pole provided
        if dimension and pole:
            return self.store.get_centroid(dimension, pole, texts)
        
        # Otherwise compute directly (still uses embedding cache)
        embeddings = self.get_embeddings(texts)
        centroid = np.mean(embeddings, axis=0)
        
        logger.debug(
            "Computed centroid from %d texts (dim=%d)",
            len(texts),
            len(centroid)
        )
        
        return centroid
    
    def calculate_drift_score(
        self,
        entity_vector: np.ndarray,
        centroid_a: np.ndarray,
        centroid_b: np.ndarray,
    ) -> dict[str, float]:
        """Calculate drift score using projection on the A→B axis.
        
        Projects the entity vector onto the axis defined by the two centroids,
        centered at the midpoint between them:
        - Negative = Closer to Legacy (A)
        - 0.0 = Exactly at midpoint between both centroids
        - Positive = Closer to Strategy (B)
        
        The projection uses the formula:
            midpoint = (centroid_a + centroid_b) / 2
            axis_vector = centroid_b - centroid_a
            projection = dot(entity - midpoint, axis_vector) / (||axis_vector|| / 2)
        
        The score is normalized so that:
            -1.0 = At Legacy centroid
            +1.0 = At Strategy centroid
        
        Args:
            entity_vector: The entity's embedding vector.
            centroid_a: The legacy/anchor A centroid.
            centroid_b: The strategy/anchor B centroid.
        
        Returns:
            Dictionary with:
                - drift_score: float centered at 0 (negative = Legacy, positive = Strategy)
                - centroid_separation: cosine distance between centroids
                - distance_to_legacy: cosine distance to centroid A
                - distance_to_strategy: cosine distance to centroid B
        """
        # Define the axis vector from A to B
        axis_vector = centroid_b - centroid_a
        
        # Calculate midpoint between centroids
        midpoint = (centroid_a + centroid_b) / 2
        
        # Project entity onto axis relative to midpoint
        entity_relative_to_midpoint = entity_vector - midpoint
        axis_length_sq = np.dot(axis_vector, axis_vector)
        
        if axis_length_sq == 0:
            # Centroids are identical - no axis
            drift_score = 0.0
        else:
            # Project and normalize: scale so centroid_a = -1, centroid_b = +1
            projection = np.dot(entity_relative_to_midpoint, axis_vector) / axis_length_sq
            # projection is now relative to midpoint, scale by 2 to get -1 to +1 range
            drift_score = float(projection * 2)
        
        # Also compute cosine distances for reference
        dist_legacy = self.cosine_distance(entity_vector, centroid_a)
        dist_strategy = self.cosine_distance(entity_vector, centroid_b)
        centroid_sep = self.cosine_distance(centroid_a, centroid_b)
        
        return {
            "drift_score": round(drift_score, 4),
            "centroid_separation": round(centroid_sep, 4),
            "distance_to_legacy": round(dist_legacy, 4),
            "distance_to_strategy": round(dist_strategy, 4),
        }
    
    def save_cache(self) -> None:
        """Persist the embedding cache to disk.
        
        Call this after completing probe execution to ensure
        newly computed embeddings are saved.
        """
        self.store.save()
    
    def _create_result(
        self,
        success: bool,
        data: dict[str, Any],
        errors: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProbeResult:
        """Helper to create a standardized ProbeResult.
        
        Args:
            success: Whether the probe executed successfully.
            data: The analysis results.
            errors: Optional list of error messages.
            metadata: Optional execution metadata.
        
        Returns:
            Configured ProbeResult instance.
        """
        return ProbeResult(
            probe_name=self.name,
            success=success,
            data=data,
            errors=errors or [],
            metadata=metadata or {},
        )
    
    def _handle_error(self, error: Exception, context: str = "") -> ProbeResult:
        """Handle an error and return a failure result.
        
        Args:
            error: The exception that occurred.
            context: Optional context about where the error occurred.
        
        Returns:
            ProbeResult indicating failure.
        """
        error_msg = f"{context}: {str(error)}" if context else str(error)
        logger.error("[%s] %s", self.name, error_msg)
        
        return self._create_result(
            success=False,
            data={},
            errors=[error_msg],
        )
