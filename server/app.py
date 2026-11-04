"""FastAPI application factory.

Providers default to the real Google/Pinecone implementations when API
keys are configured, and to the offline fake/in-memory providers otherwise,
so the app runs (and is testable) before Amine creates his free keys.
Run with: uvicorn server.app:app --reload
"""
from fastapi import FastAPI

from server.modules.embeddings import EmbeddingProvider, get_embedding_provider
from server.modules.vector_store import VectorStore, get_vector_store
from server.routes.upload import make_upload_router


def create_app(embedder: EmbeddingProvider | None = None,
               store: VectorStore | None = None) -> FastAPI:
    app = FastAPI(title="Modular RAG Assistant")
    app.include_router(make_upload_router(
        embedder or get_embedding_provider(),
        store or get_vector_store(),
    ))

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
