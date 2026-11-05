"""RAG orchestration: retrieve chunks, then generate a grounded answer.

Every answer comes with its sources (file + page + score + excerpt), so
the caller can show exactly which passages the answer is based on. When
nothing was retrieved, the LLM is not called at all: the answer is the
explicit "don't know" sentence.
"""
from dataclasses import asdict, dataclass

from server.modules.embeddings import EmbeddingProvider
from server.modules.generation import (
    DONT_KNOW,
    SYSTEM_INSTRUCTION,
    LLMProvider,
)
from server.modules.retrieval import retrieve
from server.modules.vector_store import SearchResult, VectorStore
from server.logger import get_logger

logger = get_logger(__name__)

EXCERPT_LENGTH = 200


@dataclass
class Source:
    source: str
    page: int
    score: float
    excerpt: str


def build_grounded_prompt(question: str, chunks: list[SearchResult]) -> str:
    """Prompt the LLM with the retrieved chunks and nothing else."""
    lines = [SYSTEM_INSTRUCTION, "", "Context:"]
    for chunk in chunks:
        lines.append(f"[Source: {chunk.metadata.get('source', '?')}, "
                     f"p. {chunk.metadata.get('page', '?')}]")
        lines.append(chunk.text)
        lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


def answer_question(question: str, embedder: EmbeddingProvider,
                    store: VectorStore, llm: LLMProvider,
                    top_k: int = 5) -> dict:
    """Full ask step: retrieve, generate, return answer + sources."""
    results = retrieve(question, embedder, store, top_k=top_k)
    if not results:
        logger.info("No chunks retrieved for %.60r; answering without LLM", question)
        return {"answer": DONT_KNOW, "sources": []}
    prompt = build_grounded_prompt(question, results)
    answer = llm.generate(prompt)
    sources = [
        asdict(Source(
            source=str(result.metadata.get("source", "?")),
            page=int(result.metadata.get("page", 0)),
            score=round(float(result.score), 4),
            excerpt=result.text[:EXCERPT_LENGTH],
        ))
        for result in results
    ]
    logger.info("Answered %.60r from %d sources", question, len(sources))
    return {"answer": answer, "sources": sources}
