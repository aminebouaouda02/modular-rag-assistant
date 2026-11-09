"""Run the evaluation set end to end and print the Markdown report.

Builds the eval corpus PDF from ``data/eval/corpus.json``, indexes it with
the configured providers, asks every question in ``data/eval/qa.json``
through the real pipeline (``answer_question``) and scores the answers.
The report is printed to stdout and the machine-readable results are
written to ``data/eval/results.json``.

Run from the project root: ``python scripts/run_eval.py [--top-k 3]``
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.config import settings  # noqa: E402
from server.modules.embeddings import get_embedding_provider  # noqa: E402
from server.modules.evaluation import (  # noqa: E402
    EVAL_DIR,
    build_corpus_pdf,
    evaluate,
    format_report,
    load_corpus,
    load_eval_set,
    results_to_json,
    summarize,
)
from server.modules.generation import get_llm_provider  # noqa: E402
from server.modules.indexing import index_pdf  # noqa: E402
from server.modules.rag import answer_question  # noqa: E402
from server.modules.vector_store import get_vector_store  # noqa: E402


def provider_label() -> str:
    """Name the providers actually used, so a baseline is never mistaken
    for a live measurement."""
    if settings.groq_api_key and settings.google_api_key and settings.pinecone_api_key:
        return "live (Groq + Google embeddings + Pinecone)"
    return "offline fakes (no API keys configured)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation set.")
    parser.add_argument("--top-k", type=int, default=3,
                        help="chunks retrieved per question (default: 3)")
    parser.add_argument("--results", type=Path,
                        default=EVAL_DIR / "results.json",
                        help="where to write the JSON results")
    args = parser.parse_args()

    embedder = get_embedding_provider()
    store = get_vector_store()
    llm = get_llm_provider()
    label = provider_label()

    corpus = load_corpus()
    items = load_eval_set()
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = build_corpus_pdf(corpus, Path(tmp) / corpus["filename"])
        indexed = index_pdf(pdf_path, embedder, store,
                            source_name=corpus["filename"])
        print(f"Indexed eval corpus: {indexed['chunks_indexed']} chunks "
              f"from {indexed['pages']} pages.\n")

        results = evaluate(
            lambda q: answer_question(q, embedder, store, llm, top_k=args.top_k),
            items, top_k=args.top_k,
        )

    summary = summarize(results, top_k=args.top_k)
    report = format_report(summary, results, label)
    print(report)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(results_to_json(summary, results, label), f, indent=2)
    print(f"Results written to {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
