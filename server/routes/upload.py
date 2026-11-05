"""POST /upload: ingest a PDF, embed its chunks, index them."""
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from server.logger import get_logger
from server.modules.embeddings import EmbeddingProvider
from server.modules.indexing import index_pdf
from server.modules.vector_store import VectorStore

logger = get_logger(__name__)


def make_upload_router(embedder: EmbeddingProvider, store: VectorStore) -> APIRouter:
    router = APIRouter()

    @router.post("/upload")
    async def upload_pdf(file: UploadFile = File(...)) -> dict:
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")
        tmp_path = None
        try:
            data = await file.read()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            result = index_pdf(tmp_path, embedder, store, source_name=file.filename)
            if result["chunks_indexed"] == 0:
                raise HTTPException(
                    status_code=400, detail="No text could be extracted from this PDF"
                )
            # Report the uploaded name, not the temp file name.
            result["filename"] = file.filename
            logger.info("Uploaded %s: %d chunks indexed",
                        file.filename, result["chunks_indexed"])
            return result
        except HTTPException:
            raise
        except Exception as exc:  # unreadable/corrupt PDF, embedding/store failure
            logger.exception("Upload failed for %s", file.filename)
            raise HTTPException(status_code=400, detail=f"Upload failed: {exc}") from exc
        finally:
            if tmp_path and Path(tmp_path).exists():
                os.unlink(tmp_path)

    return router
