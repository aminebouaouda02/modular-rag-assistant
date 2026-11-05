"""Indexing: turn ingested chunks into vectors stored in the vector store."""
import re
from pathlib import Path

from server.modules.embeddings import EmbeddingProvider
from server.modules.pdf_handlers import Chunk, extract_and_chunk
from server.modules.vector_store import VectorStore
from server.logger import get_logger

logger = get_logger(__name__)


def chunk_id(chunk: Chunk) -> str:
    """Stable, Pinecone-safe id: <source-stem>-p<page>-c<index>."""
    stem = Path(chunk.source).stem[:50]
    stem = re.sub(r"[^A-Za-z0-9_-]", "-", stem).strip("-") or "doc"
    return f"{stem}-p{chunk.page}-c{chunk.chunk_index}"


def index_chunks(chunks: list[Chunk], embedder: EmbeddingProvider,
                 store: VectorStore) -> int:
    """Embed chunks and upsert them; returns the number indexed."""
    if not chunks:
        return 0
    vectors = embedder.embed([c.text for c in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("embedder returned a different number of vectors than chunks")
    store.upsert(
        ids=[chunk_id(c) for c in chunks],
        vectors=vectors,
        metadatas=[
            {"source": c.source, "page": c.page, "chunk_index": c.chunk_index}
            for c in chunks
        ],
        texts=[c.text for c in chunks],
    )
    logger.info("Indexed %d chunks", len(chunks))
    return len(chunks)


def index_pdf(pdf_path: str | Path, embedder: EmbeddingProvider,
              store: VectorStore, source_name: str | None = None) -> dict:
    """Full upload step for one PDF: ingest, chunk, embed, index."""
    chunks = extract_and_chunk(pdf_path, source_name=source_name)
    indexed = index_chunks(chunks, embedder, store)
    return {
        "filename": source_name or Path(pdf_path).name,
        "pages": len({c.page for c in chunks}),
        "chunks_indexed": indexed,
    }
