"""Embedding providers.

The rest of the pipeline only depends on the ``EmbeddingProvider``
protocol, so the real Google provider can be swapped for the deterministic
fake (used in tests and when no API key is configured) without touching
indexing or retrieval code.
"""
import hashlib
import random
from typing import Protocol

from server.config import settings
from server.logger import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in input order."""


class FakeEmbeddingProvider:
    """Deterministic, offline embeddings for tests and key-less demos.

    Vectors are seeded from the text itself, so the same text always maps
    to the same vector and similar texts are not expected to be close --
    this provider proves the plumbing, not retrieval quality. Live
    retrieval quality requires the Google provider (see README).
    """

    def __init__(self, dimension: int = 16):
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            rng = random.Random(seed)
            vectors.append([rng.random() for _ in range(self.dimension)])
        return vectors


class GoogleEmbeddingProvider:
    """Google Generative AI embeddings (free tier).

    The ``google-genai`` import is lazy so the rest of the app (and the
    test suite) works without the package or an API key installed.
    """

    def __init__(self, api_key: str = settings.google_api_key,
                 model: str = settings.embedding_model):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")
        self.api_key = api_key
        # The SDK takes "gemini-embedding-001"; tolerate the "models/" prefix.
        self.model = model.removeprefix("models/")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed; run: pip install -r requirements.txt"
            ) from exc
        self._client = genai.Client(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.models.embed_content(
            model=self.model, contents=texts
        )
        return [list(embedding.values) for embedding in response.embeddings]


def get_embedding_provider() -> EmbeddingProvider:
    """Google provider when a key is configured, fake provider otherwise."""
    if settings.google_api_key:
        logger.info("Using Google embeddings (%s)", settings.embedding_model)
        return GoogleEmbeddingProvider()
    logger.warning(
        "GOOGLE_API_KEY not set: using FakeEmbeddingProvider. "
        "Live embeddings await Amine's free Google AI Studio key."
    )
    return FakeEmbeddingProvider()
