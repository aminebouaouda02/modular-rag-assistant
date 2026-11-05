"""Retrieval: embed a question and fetch the most similar chunks."""
from server.modules.embeddings import EmbeddingProvider
from server.modules.vector_store import SearchResult, VectorStore
from server.logger import get_logger

logger = get_logger(__name__)


def retrieve(question: str, embedder: EmbeddingProvider,
             store: VectorStore, top_k: int = 5) -> list[SearchResult]:
    """Return the top-k chunks most similar to the question."""
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    query_vector = embedder.embed([question])[0]
    results = store.query(query_vector, top_k=top_k)
    logger.info("Retrieved %d chunks for question %.60r", len(results), question)
    return results
