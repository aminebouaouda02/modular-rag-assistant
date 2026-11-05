"""LLM providers for grounded generation.

The rest of the pipeline only depends on the ``LLMProvider`` protocol, so
the real Groq provider can be swapped for the deterministic fake (used in
tests and when no API key is configured) without touching retrieval or
the /ask route. The fake answers strictly from the context embedded in
the prompt, so it exercises the grounding path offline; live answer
quality requires the Groq provider (see README).
"""
import re
from typing import Protocol

from server.config import settings
from server.logger import get_logger

logger = get_logger(__name__)

DONT_KNOW = "I don't know based on the provided documents."

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Answer the question using ONLY the "
    "context below. If the context does not contain the answer, reply "
    f"exactly: \"{DONT_KNOW}\" Do not use knowledge outside the context. "
    "Cite every factual claim with its source as [source, p. N]."
)


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        """Return the model's completion for the prompt."""


class FakeLLMProvider:
    """Deterministic, offline LLM for tests and key-less demos.

    It reads the context blocks out of the prompt and answers by quoting
    their first sentences with citations -- it never invents content
    beyond the retrieved chunks. With no context blocks it answers
    DONT_KNOW, mirroring the instruction given to the real model.
    """

    _BLOCK_RE = re.compile(
        r"\[Source: (?P<source>.+?), p\. (?P<page>\d+)\]\n(?P<text>.*?)(?=\n\[Source: |\nQuestion: )",
        re.DOTALL,
    )

    def generate(self, prompt: str) -> str:
        blocks = [
            (m.group("source"), int(m.group("page")), m.group("text").strip())
            for m in self._BLOCK_RE.finditer(prompt)
        ]
        if not blocks:
            return DONT_KNOW
        sentences = []
        for source, page, text in blocks:
            first = text.split(". ")[0].strip()
            if first and not first.endswith("."):
                first += "."
            sentences.append(f"{first} [{source}, p. {page}]")
        return "Based on the provided documents: " + " ".join(sentences)


class GroqLLMProvider:
    """Groq chat completions (free tier).

    The ``groq`` import is lazy so the rest of the app (and the test
    suite) works without the package or an API key installed.
    """

    def __init__(self, api_key: str = settings.groq_api_key,
                 model: str = settings.groq_model):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        self.api_key = api_key
        self.model = model
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError(
                "groq is not installed; run: pip install -r requirements.txt"
            ) from exc
        self._client = Groq(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return (response.choices[0].message.content or "").strip()


def get_llm_provider() -> LLMProvider:
    """Groq provider when a key is configured, fake provider otherwise."""
    if settings.groq_api_key:
        logger.info("Using Groq LLM (%s)", settings.groq_model)
        return GroqLLMProvider()
    logger.warning(
        "GROQ_API_KEY not set: using FakeLLMProvider. "
        "Live generation awaits Amine's free Groq key."
    )
    return FakeLLMProvider()
