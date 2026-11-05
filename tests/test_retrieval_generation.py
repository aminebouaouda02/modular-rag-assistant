import pytest

from server.modules.embeddings import FakeEmbeddingProvider
from server.modules.generation import DONT_KNOW, FakeLLMProvider
from server.modules.rag import answer_question, build_grounded_prompt
from server.modules.retrieval import retrieve
from server.modules.vector_store import InMemoryVectorStore, SearchResult


def _store_with_chunks():
    store = InMemoryVectorStore()
    embedder = FakeEmbeddingProvider()
    texts = [
        "Retrieval augmented generation grounds answers in documents.",
        "Vector search finds the most similar chunks for a question.",
    ]
    vectors = embedder.embed(texts)
    store.upsert(
        ids=["doc-p1-c0", "doc-p1-c1"],
        vectors=vectors,
        metadatas=[
            {"source": "cours.pdf", "page": 1, "chunk_index": 0},
            {"source": "cours.pdf", "page": 1, "chunk_index": 1},
        ],
        texts=texts,
    )
    return store, embedder


def test_retrieve_returns_best_match_first():
    store, embedder = _store_with_chunks()
    results = retrieve(
        "Retrieval augmented generation grounds answers in documents.",
        embedder, store, top_k=1,
    )
    assert len(results) == 1
    assert results[0].text.startswith("Retrieval augmented generation")
    assert results[0].metadata["source"] == "cours.pdf"
    assert results[0].metadata["page"] == 1


def test_retrieve_rejects_empty_question():
    store, embedder = _store_with_chunks()
    with pytest.raises(ValueError):
        retrieve("   ", embedder, store)


def test_build_grounded_prompt_contains_context_and_citation_labels():
    chunks = [SearchResult(
        id="doc-p1-c0", score=0.9,
        metadata={"source": "cours.pdf", "page": 3},
        text="Spark processes data in memory.",
    )]
    prompt = build_grounded_prompt("What does Spark do?", chunks)
    assert "What does Spark do?" in prompt
    assert "Spark processes data in memory." in prompt
    assert "[Source: cours.pdf, p. 3]" in prompt
    assert "ONLY the context" in prompt


def test_fake_llm_answers_from_context_with_citations():
    chunks = [SearchResult(
        id="doc-p1-c0", score=0.9,
        metadata={"source": "cours.pdf", "page": 2},
        text="Kafka streams events between services. It scales horizontally.",
    )]
    prompt = build_grounded_prompt("What is Kafka?", chunks)
    answer = FakeLLMProvider().generate(prompt)
    assert "Kafka streams events between services." in answer
    assert "[cours.pdf, p. 2]" in answer


def test_fake_llm_says_dont_know_without_context():
    prompt = build_grounded_prompt("What is Kafka?", [])
    assert FakeLLMProvider().generate(prompt) == DONT_KNOW


def test_answer_question_returns_answer_and_sources():
    store, embedder = _store_with_chunks()
    result = answer_question(
        "Retrieval augmented generation grounds answers in documents.",
        embedder, store, FakeLLMProvider(), top_k=2,
    )
    assert "Retrieval augmented generation grounds answers" in result["answer"]
    assert len(result["sources"]) == 2
    source = result["sources"][0]
    assert source["source"] == "cours.pdf"
    assert source["page"] == 1
    assert 0.0 <= source["score"] <= 1.0
    assert source["excerpt"]


def test_answer_question_empty_store_skips_llm():
    class ExplodingLLM:
        def generate(self, prompt):
            raise AssertionError("LLM must not be called with no chunks")

    result = answer_question(
        "Anything?", FakeEmbeddingProvider(), InMemoryVectorStore(),
        ExplodingLLM(),
    )
    assert result["answer"] == DONT_KNOW
    assert result["sources"] == []
