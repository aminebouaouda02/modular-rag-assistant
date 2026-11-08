"""FastAPI application factory.

Providers default to the real Google/Pinecone/Groq implementations when
API keys are configured, and to the offline fake/in-memory providers
otherwise, so the app runs (and is testable) before Amine creates his
free keys.
Run with: uvicorn server.app:app --reload
"""
from fastapi import FastAPI

from server.modules.embeddings import EmbeddingProvider, get_embedding_provider
from server.modules.generation import LLMProvider, get_llm_provider
from server.modules.vector_store import VectorStore, get_vector_store
from server.routes.ask import make_ask_router
from server.routes.chat import make_chat_router
from server.routes.upload import make_upload_router


def create_app(embedder: EmbeddingProvider | None = None,
               store: VectorStore | None = None,
               llm: LLMProvider | None = None) -> FastAPI:
    app = FastAPI(title="Modular RAG Assistant")
    embedder = embedder or get_embedding_provider()
    store = store or get_vector_store()
    llm = llm or get_llm_provider()
    app.include_router(make_chat_router())
    app.include_router(make_upload_router(embedder, store))
    app.include_router(make_ask_router(embedder, store, llm))

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
