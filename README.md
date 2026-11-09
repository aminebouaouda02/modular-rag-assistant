# Modular RAG Assistant

A modular **Retrieval-Augmented Generation** assistant: upload PDF documents,
ask questions, and get answers **grounded in your documents, with sources
(file + page)** instead of the model's raw memory.

Built incrementally, one working slice per day.

## Architecture

```
PDF upload ──► extraction (per page) ──► chunking (overlap, metadata)
           ──► embeddings (Google Generative AI) ──► Pinecone vector store

Question ──► query embedding ──► similarity search ──► top chunks
         ──► LLM (Groq) prompted with the chunks only ──► answer + sources
```

## Tech stack

| Component    | Technology                              |
|--------------|-----------------------------------------|
| LLM          | Groq API (free tier)                    |
| Embeddings   | Google Generative AI (free tier)        |
| Vector store | Pinecone (free Starter plan)            |
| Backend      | FastAPI                                 |
| PDF parsing  | PyMuPDF                                 |
| Tests        | pytest                                  |

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your own API keys, never commit .env
pytest
```

Without API keys the app still runs and is testable: it falls back to
deterministic fake embeddings, an in-memory vector store and a fake LLM
that answers strictly from the retrieved chunks. Live Google / Pinecone /
Groq validation requires the free keys in `.env` (J2–J5 were validated
with the offline providers, since no keys are configured yet).

## Chat interface

```bash
uvicorn server.app:app --reload
# open http://localhost:8000
```

The page at `/` lets you upload a PDF and chat with your documents in the
browser: each answer shows its sources as `[file, p. N]` chips, and the
conversation history is sent with every question so follow-ups can refer
back to earlier exchanges (`POST /ask` accepts an optional `history` list
of `{"role": "user"|"assistant", "content": ...}` turns).

## Evaluation

`scripts/run_eval.py` scores the pipeline against a fixed set of 8 questions
over a small known corpus (6 pages: Kafka, Spark, Docker, Pinecone, Groq,
FastAPI — see `data/eval/corpus.json` and `data/eval/qa.json`):

```bash
python scripts/run_eval.py            # prints the report, writes data/eval/results.json
python scripts/run_eval.py --top-k 5  # retrieve 5 chunks per question
```

Metrics: retrieval hit@k (the expected source page is among the returned
sources), answer accuracy (the answer contains every expected keyword),
citation rate (the answer cites a source as `[file, p. N]`), abstention
rate (unanswerable questions answered "I don't know based on the provided
documents.").

Measured 2026-11-08 with the offline fake providers (no API keys configured
— a deterministic baseline, not a quality claim):

| Metric | Result |
|--------|--------|
| Retrieval hit@3 | 33% (2/6) |
| Answer accuracy | 33% (2/6) |
| Citation rate | 100% (6/6) |
| Abstention rate (unanswerable) | 0% (0/2) |

What the numbers say: the harness itself works — the test suite scores 100%
on every metric with a deterministic keyword embedder — but the fake
embeddings are hash-seeded random vectors, so retrieval is near chance (33%
is the 3-of-6 chance level for top_k=3), and answers follow whatever was
retrieved. The 0% abstention rate also exposes a real gap: `/ask` has no
similarity threshold, so once any document is indexed it always answers
from the top-k chunks, even for questions unrelated to the corpus. Re-run
the eval after the free API keys are set in `.env` for live
Google/Pinecone/Groq numbers; a similarity threshold is a candidate fix
for a later slice.

## API

```bash
# Upload a PDF (extracts, chunks, embeds, indexes)
curl -F "file=@cours.pdf" http://localhost:8000/upload
# {"filename": "cours.pdf", "pages": 12, "chunks_indexed": 48}

# Ask a question grounded in the uploaded documents
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Kafka do?", "top_k": 5}'
# {"answer": "Based on the provided documents: ... [cours.pdf, p. 3]",
#  "sources": [{"source": "cours.pdf", "page": 3, "score": 0.87,
#               "excerpt": "..."}]}
```

With no documents indexed, `/ask` answers
"I don't know based on the provided documents." instead of guessing.

## Roadmap

- [x] Project scaffold, configuration, logging
- [x] PDF ingestion: per-page extraction + chunking with source/page metadata
- [x] Embeddings + Pinecone indexing (`POST /upload`)
- [x] Retrieval + grounded generation (`POST /ask`)
- [x] Chat interface (web page at `/`, multi-turn `history` on `/ask`)
- [x] Evaluation set with published metrics
- [ ] GitHub Actions CI + Docker + final polish

## Design choices

- **Modular**: ingestion, indexing, retrieval and generation are separate
  modules with explicit inputs/outputs, so each provider (LLM, embeddings,
  vector store) can be swapped without touching the rest.
- **API-first**: free tiers of Groq / Google AI / Pinecone, so the project
  costs 0 EUR to build and demo.
- **Traceable**: every chunk keeps its source file and page number, and
  answers cite them.
