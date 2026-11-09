"""Evaluation: score the RAG pipeline against a fixed question set.

The set lives in ``data/eval/qa.json`` (6 answerable questions over a small
known corpus, 2 unanswerable ones) and the corpus itself in
``data/eval/corpus.json``. Metrics:

- retrieval hit@k: the expected (source, page) is among the returned sources
- answer accuracy: the answer contains every expected keyword
- citation rate: the answer cites at least one source as ``[file, p. N]``
- abstention rate: unanswerable questions answered with the "don't know"
  sentence instead of a guess

``evaluate()`` only needs an ``answer_fn(question) -> {"answer", "sources"}``,
so it scores any pipeline — the real one (``answer_question``), a test
double, or a future provider — without depending on how answers are built.
"""
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from server.logger import get_logger
from server.modules.generation import DONT_KNOW

logger = get_logger(__name__)

CITATION_RE = re.compile(r"\[[^\]]+, p\. \d+\]")

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "eval"


@dataclass(frozen=True)
class EvalItem:
    question: str
    expected_keywords: tuple[str, ...] = ()
    expected_source: str | None = None
    expected_page: int | None = None
    answerable: bool = True


@dataclass
class EvalResult:
    item: EvalItem
    answer: str
    sources: list[dict]
    retrieval_hit: bool | None  # None = not applicable (unanswerable)
    answer_match: bool | None  # None = not applicable (unanswerable)
    cited: bool
    abstained: bool


def load_eval_set(path: str | Path = EVAL_DIR / "qa.json") -> list[EvalItem]:
    """Load the question set from its JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = [
        EvalItem(
            question=item["question"],
            expected_keywords=tuple(item.get("expected_keywords", [])),
            expected_source=item.get("expected_source"),
            expected_page=item.get("expected_page"),
            answerable=item.get("answerable", True),
        )
        for item in data["items"]
    ]
    logger.info("Loaded %d eval questions from %s", len(items), path)
    return items


def load_corpus(path: str | Path = EVAL_DIR / "corpus.json") -> dict:
    """Load the eval corpus definition (filename + one text per page)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_corpus_pdf(corpus: dict, output_path: str | Path) -> Path:
    """Write the eval corpus as a one-page-per-entry PDF (PyMuPDF)."""
    import fitz  # PyMuPDF

    output = Path(output_path)
    doc = fitz.open()
    for entry in corpus["pages"]:
        page = doc.new_page()
        # Textbox wraps the text; insert_text would clip it at the page edge.
        page.insert_textbox(
            fitz.Rect(72, 72, 540, 720), entry["text"], fontsize=11
        )
    doc.save(output)
    logger.info("Built eval corpus PDF (%d pages) at %s", len(corpus["pages"]), output)
    return output


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _keywords_match(answer: str, keywords: tuple[str, ...]) -> bool:
    normalised = _normalise(answer)
    return all(_normalise(keyword) in normalised for keyword in keywords)


def evaluate(answer_fn, items: list[EvalItem], top_k: int = 3) -> list[EvalResult]:
    """Run every question through ``answer_fn`` and score the answers.

    ``answer_fn`` takes a question string and returns
    ``{"answer": str, "sources": [{"source": ..., "page": ...}, ...]}``,
    the same shape ``answer_question`` returns.
    """
    results: list[EvalResult] = []
    for item in items:
        response = answer_fn(item.question)
        answer = response.get("answer", "")
        sources = response.get("sources", [])
        abstained = answer.strip() == DONT_KNOW
        cited = bool(CITATION_RE.search(answer))
        if item.answerable:
            retrieval_hit = any(
                s.get("source") == item.expected_source
                and s.get("page") == item.expected_page
                for s in sources
            )
            answer_match = _keywords_match(answer, item.expected_keywords)
        else:
            retrieval_hit = None
            answer_match = None
        results.append(EvalResult(
            item=item, answer=answer, sources=sources,
            retrieval_hit=retrieval_hit, answer_match=answer_match,
            cited=cited, abstained=abstained,
        ))
        logger.info(
            "Eval %.60r: hit=%s match=%s cited=%s abstained=%s",
            item.question, retrieval_hit, answer_match, cited, abstained,
        )
    return results


def summarize(results: list[EvalResult], top_k: int = 3) -> dict:
    """Aggregate per-question results into the headline metrics."""
    answerable = [r for r in results if r.item.answerable]
    unanswerable = [r for r in results if not r.item.answerable]

    def rate(values: list[bool | None]) -> float | None:
        known = [v for v in values if v is not None]
        return round(sum(known) / len(known), 3) if known else None

    return {
        "total": len(results),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "top_k": top_k,
        "retrieval_hits": sum(r.retrieval_hit is True for r in answerable),
        "retrieval_hit_rate_at_k": rate([r.retrieval_hit for r in answerable]),
        "answers_correct": sum(r.answer_match is True for r in answerable),
        "answer_accuracy": rate([r.answer_match for r in answerable]),
        "citations": sum(r.cited for r in answerable),
        "citation_rate": rate([r.cited for r in answerable]),
        "abstentions": sum(r.abstained for r in unanswerable),
        "abstention_rate": rate([r.abstained for r in unanswerable]),
    }


def _percent(rate: float | None) -> str:
    return f"{rate * 100:.0f}%" if rate is not None else "n/a"


def format_report(summary: dict, results: list[EvalResult],
                  provider_label: str) -> str:
    """Render the metrics as a Markdown report (for stdout and the README)."""
    lines = [
        "# Evaluation results",
        "",
        f"Providers: {provider_label}. "
        f"{summary['total']} questions "
        f"({summary['answerable']} answerable, {summary['unanswerable']} "
        f"unanswerable), top_k={summary['top_k']}.",
        "",
        "| Metric | Result |",
        "|--------|--------|",
        f"| Retrieval hit@{summary['top_k']} | "
        f"{_percent(summary['retrieval_hit_rate_at_k'])} "
        f"({summary['retrieval_hits']}/{summary['answerable']}) |",
        f"| Answer accuracy | {_percent(summary['answer_accuracy'])} "
        f"({summary['answers_correct']}/{summary['answerable']}) |",
        f"| Citation rate | {_percent(summary['citation_rate'])} "
        f"({summary['citations']}/{summary['answerable']}) |",
        f"| Abstention rate (unanswerable) | "
        f"{_percent(summary['abstention_rate'])} "
        f"({summary['abstentions']}/{summary['unanswerable']}) |",
        "",
        "## Per-question results",
        "",
        "| Question | Retrieval hit | Answer match | Cited | Abstained |",
        "|----------|--------------|--------------|-------|-----------|",
    ]
    for r in results:
        def mark(value: bool | None) -> str:
            return "n/a" if value is None else ("yes" if value else "no")

        lines.append(
            f"| {r.item.question} | {mark(r.retrieval_hit)} | "
            f"{mark(r.answer_match)} | {mark(r.cited)} | {mark(r.abstained)} |"
        )
    lines.append("")
    return "\n".join(lines)


def results_to_json(summary: dict, results: list[EvalResult],
                    provider_label: str) -> dict:
    """Machine-readable form of a run (written to data/eval/results.json)."""
    return {
        "providers": provider_label,
        "summary": summary,
        "questions": [
            {
                "question": r.item.question,
                "answerable": r.item.answerable,
                "expected_source": r.item.expected_source,
                "expected_page": r.item.expected_page,
                "answer": r.answer,
                "sources": r.sources,
                "retrieval_hit": r.retrieval_hit,
                "answer_match": r.answer_match,
                "cited": r.cited,
                "abstained": r.abstained,
            }
            for r in results
        ],
    }


__all__ = [
    "CITATION_RE",
    "EvalItem",
    "EvalResult",
    "build_corpus_pdf",
    "evaluate",
    "format_report",
    "load_corpus",
    "load_eval_set",
    "results_to_json",
    "summarize",
]
