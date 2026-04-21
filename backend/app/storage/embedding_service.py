"""
EmbeddingService — embedding via OpenAI-compatible API (Azure OpenAI or Ollama).

When AZURE_API_VERSION / LLM_EMBEDDING_API_VERSION are set, uses AzureOpenAI client.
Otherwise falls back to Ollama /api/embed endpoint for local models.
"""

import time
import logging
from typing import List, Optional

import requests
from openai import OpenAI, AzureOpenAI

from ..config import Config

logger = logging.getLogger('mirofish.embedding')


class EmbeddingService:
    """Generate embeddings via Azure OpenAI or local Ollama."""

    # Dimensions requested from the model — must match Neo4j vector index.
    DIMENSIONS = 768

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        azure_embed_version = Config.LLM_EMBEDDING_API_VERSION or Config.AZURE_API_VERSION
        use_azure = bool(azure_embed_version)

        if use_azure:
            self.model = model or Config.LLM_EMBEDDING_MODEL_NAME or Config.EMBEDDING_MODEL
            endpoint = base_url or Config.LLM_BASE_URL
            self._client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=Config.LLM_API_KEY,
                api_version=azure_embed_version,
            )
            self._use_openai_client = True
        else:
            self.model = model or Config.EMBEDDING_MODEL
            ollama_base = (base_url or Config.EMBEDDING_BASE_URL).rstrip('/')
            self._embed_url = f"{ollama_base}/api/embed"
            self._use_openai_client = False

        self.max_retries = max_retries
        self.timeout = timeout

        # Simple in-memory cache (text -> embedding vector)
        self._cache: dict[str, List[float]] = {}
        self._cache_max_size = 2000

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            768-dimensional float vector

        Raises:
            EmbeddingError: If Ollama request fails after retries
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        text = text.strip()

        # Check cache
        if text in self._cache:
            return self._cache[text]

        vectors = self._request_embeddings([text])
        vector = vectors[0]

        # Cache result
        self._cache_put(text, vector)

        return vector

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Processes in batches to avoid overwhelming Ollama.

        Args:
            texts: List of input texts
            batch_size: Number of texts per request

        Returns:
            List of embedding vectors (same order as input)
        """
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check cache first
        for i, text in enumerate(texts):
            text = text.strip() if text else ""
            if text in self._cache:
                results[i] = self._cache[text]
            elif text:
                uncached_indices.append(i)
                uncached_texts.append(text)
            else:
                # Empty text — zero vector
                results[i] = [0.0] * self.DIMENSIONS

        # Batch-embed uncached texts
        if uncached_texts:
            all_vectors: List[List[float]] = []
            for start in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[start:start + batch_size]
                vectors = self._request_embeddings(batch)
                all_vectors.extend(vectors)

            # Place results and cache
            for idx, vec, text in zip(uncached_indices, all_vectors, uncached_texts):
                results[idx] = vec
                self._cache_put(text, vec)

        return results  # type: ignore

    def _request_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Request embeddings from Azure OpenAI or Ollama, with retry.

        Returns:
            List of embedding vectors (768 dimensions)
        """
        if self._use_openai_client:
            return self._request_embeddings_openai(texts)
        return self._request_embeddings_ollama(texts)

    def _request_embeddings_openai(self, texts: List[str]) -> List[List[float]]:
        """Use OpenAI/Azure client (supports dimensions parameter)."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.embeddings.create(
                    input=texts,
                    model=self.model,
                    dimensions=self.DIMENSIONS,
                )
                vectors = [item.embedding for item in response.data]
                if len(vectors) != len(texts):
                    raise EmbeddingError(
                        f"Expected {len(texts)} embeddings, got {len(vectors)}"
                    )
                return vectors
            except Exception as e:
                last_error = e
                logger.warning(f"Embedding request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.info(f"Retrying in {wait}s...")
                    time.sleep(wait)

        raise EmbeddingError(
            f"Embedding failed after {self.max_retries} retries: {last_error}"
        )

    def _request_embeddings_ollama(self, texts: List[str]) -> List[List[float]]:
        """Use Ollama /api/embed endpoint (legacy local model path)."""
        payload = {"model": self.model, "input": texts}
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self._embed_url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("embeddings", [])
                if len(embeddings) != len(texts):
                    raise EmbeddingError(
                        f"Expected {len(texts)} embeddings, got {len(embeddings)}"
                    )
                return embeddings
            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.warning(f"Ollama connection failed (attempt {attempt + 1}/{self.max_retries}): {e}")
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(f"Ollama request timed out (attempt {attempt + 1}/{self.max_retries})")
            except requests.exceptions.HTTPError as e:
                last_error = e
                logger.error(f"Ollama HTTP error: {e.response.status_code} - {e.response.text}")
                if e.response.status_code < 500:
                    raise EmbeddingError(f"Ollama embedding failed: {e}") from e
            except (KeyError, ValueError) as e:
                raise EmbeddingError(f"Invalid Ollama response: {e}") from e
            if attempt < self.max_retries - 1:
                wait = 2 ** attempt
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)

        raise EmbeddingError(
            f"Ollama embedding failed after {self.max_retries} retries: {last_error}"
        )

    def _cache_put(self, text: str, vector: List[float]) -> None:
        """Add to cache, evicting oldest entries if full."""
        if len(self._cache) >= self._cache_max_size:
            # Remove ~10% of oldest entries
            keys_to_remove = list(self._cache.keys())[:self._cache_max_size // 10]
            for key in keys_to_remove:
                del self._cache[key]
        self._cache[text] = vector

    def health_check(self) -> bool:
        """Check if Ollama embedding endpoint is reachable."""
        try:
            vec = self.embed("health check")
            return len(vec) > 0
        except Exception:
            return False


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass
