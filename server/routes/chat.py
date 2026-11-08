"""GET /: serve the chat interface (single self-contained HTML page)."""
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

INDEX_PATH = Path(__file__).resolve().parent.parent / "static" / "index.html"


def make_chat_router() -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=FileResponse)
    async def chat_page() -> FileResponse:
        return FileResponse(INDEX_PATH, media_type="text/html")

    return router
