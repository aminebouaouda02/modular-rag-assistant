"""PDF ingestion: extract text page by page, then split it into chunks.

Each chunk keeps its source file and page number so answers can cite
exactly where a passage came from.
"""
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from server.config import settings
from server.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PageText:
    source: str
    page: int  # 1-based
    text: str


@dataclass
class Chunk:
    source: str
    page: int  # page where the chunk starts
    chunk_index: int
    text: str


def load_pdf_pages(pdf_path: str | Path) -> list[PageText]:
    """Extract the text of every page of a PDF."""
    path = Path(pdf_path)
    pages: list[PageText] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append(PageText(source=path.name, page=i, text=text))
    logger.info("Loaded %d non-empty pages from %s", len(pages), path.name)
    return pages


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence breaks."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Try to end on a clean boundary inside the second half of the window.
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > chunk_size // 2:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - chunk_overlap
    return chunks


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[Chunk]:
    """Chunk every page, keeping source + page metadata on each chunk."""
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        for piece in chunk_text(page.text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(source=page.source, page=page.page, chunk_index=index, text=piece)
            )
            index += 1
    logger.info("Produced %d chunks from %d pages", len(chunks), len(pages))
    return chunks


def extract_and_chunk(pdf_path: str | Path) -> list[Chunk]:
    """Full ingestion step for one PDF: load pages, then chunk them."""
    return chunk_pages(load_pdf_pages(pdf_path))
