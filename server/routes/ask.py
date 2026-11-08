"""POST /ask: answer a question from the indexed documents, with sources."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from server.logger import get_logger
from server.modules.embeddings import EmbeddingProvider
from server.modules.generation import LLMProvider
from server.modules.rag import answer_question
from server.modules.vector_store import VectorStore

logger = get_logger(__name__)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, description="One earlier chat turn")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer")
    top_k: int = Field(5, ge=1, le=10, description="Chunks to retrieve")
    history: list[ChatMessage] | None = Field(
        None, description="Earlier conversation turns, for follow-up questions"
    )


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
                request.question, embedder, store, llm, top_k=request.top_k,
                history=[m.model_dump() for m in request.history]
                if request.history else None,
            )
        except Exception as exc:  # embedding / store / LLM failure
            logger.exception("Ask failed for %.60r", request.question)
            raise HTTPException(status_code=502, detail=f"Ask failed: {exc}") from exc

    return router
