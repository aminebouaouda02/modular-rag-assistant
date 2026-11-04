"""Vector stores.

The pipeline only depends on the ``VectorStore`` protocol, so the real
Pinecone store can be swapped for the in-memory store (tests and key-less
demos) without touching indexing or retrieval code.
"""
import math
from dataclasses import dataclass
from typing import Protocol

from server.config import settings
from server.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict
    text: str = ""


class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors: list[list[float]],
               metadatas: list[dict], texts: list[str]) -> None:
        ...

    def query(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    """Offline store for tests and key-less demos (data lives in RAM)."""

    def __init__(self):
        self._vectors: dict[str, list[float]] = {}
        self._metadatas: dict[str, dict] = {}
        self._texts: dict[str, str] = {}

    @property
    def count(self) -> int:
        return len(self._vectors)

    def upsert(self, ids, vectors, metadatas, texts):
        for id_, vector, metadata, text in zip(ids, vectors, metadatas, texts):
            self._vectors[id_] = list(vector)
            self._metadatas[id_] = dict(metadata)
            self._texts[id_] = text

    def query(self, vector, top_k=5):
        ranked = sorted(
            (
                SearchResult(
                    id=id_,
                    score=_cosine(vector, stored),
                    metadata=self._metadatas[id_],
                    text=self._texts[id_],
                )
                for id_, stored in self._vectors.items()
            ),
            key=lambda r: r.score,
            reverse=True,
        )
        return ranked[:top_k]


class PineconeVectorStore:
    """Pinecone Starter vector store (free tier).

    The ``pinecone`` import and index creation are lazy: the index is
    created on first upsert, once the embedding dimension is known.
    """

    def __init__(self, api_key: str = settings.pinecone_api_key,
                 index_name: str = settings.pinecone_index_name):
        if not api_key:
            raise ValueError("PINECONE_API_KEY is not set")
        self.index_name = index_name
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise RuntimeError(
                "pinecone is not installed; run: pip install -r requirements.txt"
            ) from exc
        self._ServerlessSpec = ServerlessSpec
        self._client = Pinecone(api_key=api_key)
        self._index = None

    def _ensure_index(self, dimension: int):
        if self._index is not None:
            return
        existing = [idx["name"] for idx in self._client.list_indexes()]
        if self.index_name not in existing:
            logger.info("Creating Pinecone index %s (dim=%d)", self.index_name, dimension)
            self._client.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=self._ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self._index = self._client.Index(self.index_name)

    def upsert(self, ids, vectors, metadatas, texts):
        if not ids:
            return
        self._ensure_index(len(vectors[0]))
        records = [
            {
                "id": id_,
                "values": vector,
                "metadata": {**metadata, "text": text},
            }
            for id_, vector, metadata, text in zip(ids, vectors, metadatas, texts)
        ]
        self._index.upsert(vectors=records)

    def query(self, vector, top_k=5):
        self._ensure_index(len(vector))
        response = self._index.query(
            vector=vector, top_k=top_k, include_metadata=True
        )
        results = []
        for match in response["matches"]:
            metadata = dict(match.get("metadata", {}))
            text = metadata.pop("text", "")
            results.append(SearchResult(
                id=match["id"], score=float(match["score"]),
                metadata=metadata, text=text,
            ))
        return results


def get_vector_store() -> VectorStore:
    """Pinecone store when a key is configured, in-memory store otherwise."""
    if settings.pinecone_api_key:
        logger.info("Using Pinecone index %s", settings.pinecone_index_name)
        return PineconeVectorStore()
    logger.warning(
        "PINECONE_API_KEY not set: using InMemoryVectorStore. "
        "Live indexing awaits Amine's free Pinecone key."
    )
    return InMemoryVectorStore()
