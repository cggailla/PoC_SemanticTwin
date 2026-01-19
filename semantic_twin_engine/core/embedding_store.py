"""
Embedding Store for the Semantic Twin Engine.

Provides persistent caching of embeddings to avoid redundant API calls.
Uses NumPy .npz format for efficient storage and SHA256 hashing for keys.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from core.exceptions import APIError

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Persistent cache for embeddings using NumPy .npz files.
    
    This store provides:
    - Content-addressed caching using SHA256 hashes
    - Automatic persistence to disk
    - Model-aware invalidation
    - Centroid pre-computation caching
    
    Cache Structure:
        cache/
        ├── embeddings.npz    # Raw embeddings {hash: vector}
        ├── centroids.npz     # Pre-computed centroids {dim_pole: vector}
        └── metadata.json     # Model version, timestamps
    
    Example:
        store = EmbeddingStore(cache_dir="./cache")
        
        # Get cached or compute new embedding
        embedding = store.get_embedding("combustion")
        
        # Get cached centroid
        centroid = store.get_centroid("product_physics", "legacy", terms)
    """
    
    def __init__(
        self,
        cache_dir: Path | str = "./cache",
        model: str | None = None,
    ) -> None:
        """Initialize the embedding store.
        
        Args:
            cache_dir: Directory for cache files.
            model: Embedding model name (default from env or text-embedding-3-small).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = model or os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small"
        )
        
        self._embeddings_path = self.cache_dir / "embeddings.npz"
        self._centroids_path = self.cache_dir / "centroids.npz"
        self._metadata_path = self.cache_dir / "metadata.json"
        
        self._client: OpenAI | None = None
        self._embeddings_cache: dict[str, np.ndarray] = {}
        self._centroids_cache: dict[str, np.ndarray] = {}
        self._dirty_embeddings = False
        self._dirty_centroids = False
        
        self._load_cache()
    
    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise APIError(
                    "OPENAI_API_KEY environment variable not set",
                    service="OpenAI"
                )
            self._client = OpenAI(api_key=api_key)
        return self._client
    
    def _compute_hash(self, text: str) -> str:
        """Compute SHA256 hash for cache key.
        
        The hash includes the model name to ensure cache invalidation
        when the model changes.
        
        Args:
            text: The text to hash.
        
        Returns:
            Hex digest of the hash.
        """
        content = f"{self.model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        # Load embeddings
        if self._embeddings_path.exists():
            try:
                data = np.load(self._embeddings_path, allow_pickle=True)
                self._embeddings_cache = dict(data.items())
                logger.info(
                    "Loaded %d embeddings from cache",
                    len(self._embeddings_cache)
                )
            except Exception as e:
                logger.warning("Failed to load embeddings cache: %s", e)
                self._embeddings_cache = {}
        
        # Load centroids
        if self._centroids_path.exists():
            try:
                data = np.load(self._centroids_path, allow_pickle=True)
                self._centroids_cache = dict(data.items())
                logger.info(
                    "Loaded %d centroids from cache",
                    len(self._centroids_cache)
                )
            except Exception as e:
                logger.warning("Failed to load centroids cache: %s", e)
                self._centroids_cache = {}
        
        # Validate model version
        if self._metadata_path.exists():
            try:
                with open(self._metadata_path, "r") as f:
                    metadata = json.load(f)
                cached_model = metadata.get("model")
                if cached_model and cached_model != self.model:
                    logger.warning(
                        "Model changed (%s -> %s), invalidating cache",
                        cached_model,
                        self.model
                    )
                    self._embeddings_cache = {}
                    self._centroids_cache = {}
            except Exception as e:
                logger.warning("Failed to load metadata: %s", e)
    
    def save(self) -> None:
        """Persist cache to disk."""
        # Save embeddings
        if self._dirty_embeddings and self._embeddings_cache:
            np.savez(self._embeddings_path, **self._embeddings_cache)
            self._dirty_embeddings = False
            logger.debug("Saved %d embeddings to cache", len(self._embeddings_cache))
        
        # Save centroids
        if self._dirty_centroids and self._centroids_cache:
            np.savez(self._centroids_path, **self._centroids_cache)
            self._dirty_centroids = False
            logger.debug("Saved %d centroids to cache", len(self._centroids_cache))
        
        # Save metadata
        metadata = {
            "model": self.model,
            "updated_at": datetime.now().isoformat(),
            "embeddings_count": len(self._embeddings_cache),
            "centroids_count": len(self._centroids_cache),
        }
        with open(self._metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _fetch_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Fetch embeddings from OpenAI API.
        
        Args:
            texts: List of texts to embed.
        
        Returns:
            List of embedding vectors.
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [np.array(item.embedding) for item in response.data]
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text, using cache if available.
        
        Args:
            text: The text to embed.
        
        Returns:
            Embedding vector as numpy array.
        """
        cache_key = self._compute_hash(text)
        
        if cache_key in self._embeddings_cache:
            logger.debug("Cache HIT for: %s...", text[:30])
            return self._embeddings_cache[cache_key]
        
        logger.debug("Cache MISS for: %s...", text[:30])
        embeddings = self._fetch_embeddings([text])
        self._embeddings_cache[cache_key] = embeddings[0]
        self._dirty_embeddings = True
        
        return embeddings[0]
    
    def get_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Get embeddings for multiple texts, using cache where available.
        
        This method checks the cache for each text individually and only
        fetches missing embeddings from the API.
        
        Args:
            texts: List of texts to embed.
        
        Returns:
            List of embedding vectors.
        """
        results: list[np.ndarray | None] = [None] * len(texts)
        texts_to_fetch: list[tuple[int, str]] = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._compute_hash(text)
            if cache_key in self._embeddings_cache:
                results[i] = self._embeddings_cache[cache_key]
            else:
                texts_to_fetch.append((i, text))
        
        cache_hits = len(texts) - len(texts_to_fetch)
        if cache_hits > 0:
            logger.info(
                "Embedding cache: %d hits, %d misses",
                cache_hits,
                len(texts_to_fetch)
            )
        
        # Fetch missing embeddings
        if texts_to_fetch:
            missing_texts = [t for _, t in texts_to_fetch]
            new_embeddings = self._fetch_embeddings(missing_texts)
            
            for (i, text), embedding in zip(texts_to_fetch, new_embeddings):
                cache_key = self._compute_hash(text)
                self._embeddings_cache[cache_key] = embedding
                results[i] = embedding
                self._dirty_embeddings = True
        
        return results  # type: ignore
    
    def get_centroid(
        self,
        dimension: str,
        pole: str,
        terms: list[str],
    ) -> np.ndarray:
        """Get centroid for a dimension pole, using cache if available.
        
        Args:
            dimension: Dimension name (e.g., "product_physics").
            pole: Pole name ("legacy" or "strategy").
            terms: List of anchor terms for this pole.
        
        Returns:
            Centroid vector as numpy array.
        """
        cache_key = f"{dimension}_{pole}"
        
        if cache_key in self._centroids_cache:
            logger.debug("Centroid cache HIT for: %s", cache_key)
            return self._centroids_cache[cache_key]
        
        logger.debug("Centroid cache MISS for: %s", cache_key)
        
        # Get embeddings for all terms (will use embedding cache)
        embeddings = self.get_embeddings(terms)
        centroid = np.mean(embeddings, axis=0)
        
        self._centroids_cache[cache_key] = centroid
        self._dirty_centroids = True
        
        return centroid
    
    def invalidate_dimension(self, dimension: str) -> None:
        """Invalidate cached centroids for a dimension.
        
        Call this when anchor terms for a dimension change.
        
        Args:
            dimension: Dimension name to invalidate.
        """
        keys_to_remove = [
            k for k in self._centroids_cache
            if k.startswith(f"{dimension}_")
        ]
        
        for key in keys_to_remove:
            del self._centroids_cache[key]
            logger.info("Invalidated centroid: %s", key)
        
        if keys_to_remove:
            self._dirty_centroids = True
    
    def invalidate_all(self) -> None:
        """Invalidate entire cache.
        
        Call this when the embedding model changes.
        """
        self._embeddings_cache = {}
        self._centroids_cache = {}
        self._dirty_embeddings = True
        self._dirty_centroids = True
        logger.info("Invalidated all cache")
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats.
        """
        return {
            "embeddings_count": len(self._embeddings_cache),
            "centroids_count": len(self._centroids_cache),
            "model": self.model,
            "cache_dir": str(self.cache_dir),
        }
    
    def __enter__(self) -> "EmbeddingStore":
        """Context manager entry."""
        return self
    
    def __exit__(self, *args) -> None:
        """Context manager exit - auto-save."""
        self.save()


# Global instance for convenience
_global_store: EmbeddingStore | None = None


def get_store(cache_dir: Path | str = "./cache") -> EmbeddingStore:
    """Get or create global embedding store instance.
    
    Args:
        cache_dir: Directory for cache files.
    
    Returns:
        EmbeddingStore instance.
    """
    global _global_store
    if _global_store is None:
        _global_store = EmbeddingStore(cache_dir=cache_dir)
    return _global_store
