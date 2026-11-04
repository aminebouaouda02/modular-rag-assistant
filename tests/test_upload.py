import fitz  # PyMuPDF
from fastapi.testclient import TestClient

from server.app import create_app
from server.modules.embeddings import FakeEmbeddingProvider
from server.modules.vector_store import InMemoryVectorStore


def _make_pdf(tmp_path, pages_text):
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(pdf_path)
    doc.close()
    return pdf_path.read_bytes()


def _client():
    store = InMemoryVectorStore()
    app = create_app(embedder=FakeEmbeddingProvider(), store=store)
    return TestClient(app), store


def test_health():
    client, _ = _client()
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_pdf_indexes_chunks(tmp_path):
    client, store = _client()
    pdf_bytes = _make_pdf(tmp_path, [
        "Retrieval augmented generation grounds answers in documents. " * 10,
        "Vector search finds the most similar chunks. " * 10,
    ])
    response = client.post(
        "/upload",
        files={"file": ("cours.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["filename"] == "cours.pdf"
    assert body["pages"] == 2
    assert body["chunks_indexed"] > 0
    assert store.count == body["chunks_indexed"]


def test_upload_rejects_non_pdf():
    client, _ = _client()
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
