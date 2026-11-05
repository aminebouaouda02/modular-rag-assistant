"""POST /ask: answer a question from the indexed documents, with sources."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from server.logger import get_logger
from server.modules.embeddings import EmbeddingProvider
from server.modules.generation import LLMProvider
from server.modules.rag import answer_question
from server.modules.vector_store import VectorStore

logger = get_logger(__name__)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer")
    top_k: int = Field(5, ge=1, le=10, description="Chunks to retrieve")


class SourceResponse(BaseModel):
    source: str
    page: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


def make_ask_router(embedder: EmbeddingProvider, store: VectorStore,
                    llm: LLMProvider) -> APIRouter:
    router = APIRouter()

    @router.post("/ask", response_model=AskResponse)
    async def ask(request: AskRequest) -> dict:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question must not be empty")
        try:
            return answer_question(
                request.question, embedder, store, llm, top_k=request.top_k
            )
        except Exception as exc:  # embedding / store / LLM failure
            logger.exception("Ask failed for %.60r", request.question)
            raise HTTPException(status_code=502, detail=f"Ask failed: {exc}") from exc

    return router
