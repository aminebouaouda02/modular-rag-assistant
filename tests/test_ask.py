import fitz  # PyMuPDF
from fastapi.testclient import TestClient

from server.app import create_app
from server.modules.embeddings import FakeEmbeddingProvider
from server.modules.generation import DONT_KNOW, FakeLLMProvider
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
    app = create_app(
        embedder=FakeEmbeddingProvider(), store=store, llm=FakeLLMProvider()
    )
    return TestClient(app), store


def test_ask_after_upload_answers_from_document(tmp_path):
    client, _ = _client()
    pdf_bytes = _make_pdf(tmp_path, [
        "Apache Kafka streams events between services. " * 10,
        "PostgreSQL stores relational data reliably. " * 10,
    ])
    upload = client.post(
        "/upload",
        files={"file": ("cours.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload.status_code == 200, upload.text

    response = client.post(
        "/ask",
        json={"question": "Apache Kafka streams events between services. " * 10},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "Kafka streams events between services" in body["answer"]
    assert "[cours.pdf, p. 1]" in body["answer"]
    assert body["sources"], "answer must cite its sources"
    # Fake embeddings rank chunks arbitrarily, so check the set, not the order.
    assert all(s["source"] == "cours.pdf" for s in body["sources"])
    assert any(s["page"] == 1 for s in body["sources"])
    assert all("excerpt" in s for s in body["sources"])


def test_ask_with_no_documents_says_dont_know():
    client, _ = _client()
    response = client.post("/ask", json={"question": "What is Kafka?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == DONT_KNOW
    assert body["sources"] == []


def test_ask_rejects_empty_question():
    client, _ = _client()
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


def test_ask_rejects_missing_question():
    client, _ = _client()
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_rejects_top_k_out_of_range():
    client, _ = _client()
    assert client.post("/ask", json={"question": "x", "top_k": 0}).status_code == 422
    assert client.post("/ask", json={"question": "x", "top_k": 11}).status_code == 422
