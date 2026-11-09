"""Tests for the evaluation module (J5).

Retrieval quality is exercised with a deterministic keyword-overlap
embedder defined here (test-only): the production fake embeddings are
hash-seeded random vectors, so they prove plumbing, not retrieval.
The metrics themselves are provider-independent and are also tested
with stub answer functions.
"""
import re

from server.modules.evaluation import (
    EVAL_DIR,
    EvalItem,
    build_corpus_pdf,
    evaluate,
    format_report,
    load_corpus,
    load_eval_set,
    summarize,
)
from server.modules.generation import DONT_KNOW, FakeLLMProvider
from server.modules.indexing import index_pdf
from server.modules.rag import answer_question
from server.modules.vector_store import InMemoryVectorStore


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class KeywordEmbeddingProvider:
    """Deterministic test embedder: vectors are token counts over a fixed
    vocabulary, so a question is closest to the chunk sharing its words."""

    def __init__(self, texts: list[str]):
        vocab = sorted({token for text in texts for token in _tokens(text)})
        self._index = {token: i for i, token in enumerate(vocab)}
        self.dimension = len(vocab)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimension
            for token in _tokens(text):
                if token in self._index:
                    vector[self._index[token]] += 1.0
            vectors.append(vector)
        return vectors


def _indexed_eval_store(tmp_path):
    """Index the real eval corpus PDF with the keyword embedder."""
    corpus = load_corpus()
    pdf_path = build_corpus_pdf(corpus, tmp_path / corpus["filename"])
    texts = [entry["text"] for entry in corpus["pages"]]
    embedder = KeywordEmbeddingProvider(texts)
    store = InMemoryVectorStore()
    indexed = index_pdf(pdf_path, embedder, store, source_name=corpus["filename"])
    assert indexed["chunks_indexed"] == len(corpus["pages"])
    return store, embedder


def test_load_eval_set_has_answerable_and_unanswerable():
    items = load_eval_set()
    assert len(items) == 8
    assert sum(item.answerable for item in items) == 6
    assert sum(not item.answerable for item in items) == 2
    assert all(item.expected_source == "eval-corpus.pdf"
               for item in items if item.answerable)
    assert all(item.expected_page is not None
               for item in items if item.answerable)


def test_evaluate_retrieval_and_answers_with_keyword_embedder(tmp_path):
    store, embedder = _indexed_eval_store(tmp_path)
    items = load_eval_set()
    results = evaluate(
        lambda q: answer_question(q, embedder, store, FakeLLMProvider(), top_k=3),
        items, top_k=3,
    )
    summary = summarize(results, top_k=3)
    assert summary["total"] == 8
    assert summary["retrieval_hit_rate_at_k"] == 1.0
    assert summary["answer_accuracy"] == 1.0
    assert summary["citation_rate"] == 1.0


def test_evaluate_abstention_on_empty_store():
    store = InMemoryVectorStore()
    embedder = KeywordEmbeddingProvider(["kafka streaming platform"])
    items = [item for item in load_eval_set() if not item.answerable]
    results = evaluate(
        lambda q: answer_question(q, embedder, store, FakeLLMProvider(), top_k=3),
        items, top_k=3,
    )
    summary = summarize(results, top_k=3)
    assert summary["abstention_rate"] == 1.0
    assert all(r.abstained for r in results)
    assert all(r.answer == DONT_KNOW for r in results)


def test_evaluate_scores_a_wrong_answer_as_miss():
    item = EvalItem(
        question="What is Apache Kafka?",
        expected_keywords=("kafka", "event streaming"),
        expected_source="eval-corpus.pdf", expected_page=1, answerable=True,
    )

    def wrong_answer_fn(question):
        return {"answer": "Docker runs containers. [eval-corpus.pdf, p. 3]",
                "sources": [{"source": "eval-corpus.pdf", "page": 3}]}

    (result,) = evaluate(wrong_answer_fn, [item], top_k=3)
    assert result.retrieval_hit is False
    assert result.answer_match is False
    assert result.cited is True  # it cites, just the wrong page
    assert result.abstained is False


def test_evaluate_marks_dont_know_as_abstention():
    item = EvalItem(question="What is the capital of France?", answerable=False)
    (result,) = evaluate(
        lambda q: {"answer": DONT_KNOW, "sources": []}, [item], top_k=3)
    assert result.abstained is True
    assert result.retrieval_hit is None
    assert result.answer_match is None


def test_summarize_rates_and_format_report():
    items = load_eval_set()[:3]
    stubbed = [
        {"answer": "Apache Kafka is a distributed event streaming platform. "
                   "[eval-corpus.pdf, p. 1]",
         "sources": [{"source": "eval-corpus.pdf", "page": 1}]},
        {"answer": "Something unrelated, no citation.",
         "sources": [{"source": "eval-corpus.pdf", "page": 4}]},
        {"answer": DONT_KNOW, "sources": []},
    ]
    results = evaluate(
        lambda q, it=iter(stubbed): next(it), items, top_k=3)
    summary = summarize(results, top_k=3)
    assert summary["retrieval_hit_rate_at_k"] == 0.333
    assert summary["answer_accuracy"] == 0.333
    assert summary["citation_rate"] == 0.333
    report = format_report(summary, results, "test providers")
    assert "Retrieval hit@3" in report
    assert "33%" in report
    assert "What is Apache Kafka?" in report


def test_build_corpus_pdf_matches_corpus_json(tmp_path):
    corpus = load_corpus()
    pdf_path = build_corpus_pdf(corpus, tmp_path / "eval-corpus.pdf")
    assert pdf_path.exists()
    import fitz

    with fitz.open(pdf_path) as doc:
        assert len(doc) == 6
        assert "Kafka" in doc[0].get_text()
        assert "FastAPI" in doc[5].get_text()
    assert EVAL_DIR.name == "eval"
