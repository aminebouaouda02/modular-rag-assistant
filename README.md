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
deterministic fake embeddings and an in-memory vector store. Live Google /
Pinecone validation requires the free keys in `.env` (J2 was validated with
the offline providers, since no keys are configured yet).

## Roadmap

- [x] Project scaffold, configuration, logging
- [x] PDF ingestion: per-page extraction + chunking with source/page metadata
- [x] Embeddings + Pinecone indexing (`POST /upload`)
- [ ] Retrieval + grounded generation (`POST /ask`)
- [ ] Chat interface
- [ ] Evaluation set with published metrics
- [ ] Tests, CI, Docker

## Design choices

- **Modular**: ingestion, indexing, retrieval and generation are separate
  modules with explicit inputs/outputs, so each provider (LLM, embeddings,
  vector store) can be swapped without touching the rest.
- **API-first**: free tiers of Groq / Google AI / Pinecone, so the project
  costs 0 EUR to build and demo.
- **Traceable**: every chunk keeps its source file and page number, and
  answers cite them.
