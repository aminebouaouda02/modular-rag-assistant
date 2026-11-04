import pytest

from server.modules.pdf_handlers import PageText, chunk_pages, chunk_text


def test_chunk_text_empty():
    assert chunk_text("", 100, 20) == []
    assert chunk_text("   ", 100, 20) == []


def test_chunk_text_short_text_single_chunk():
    assert chunk_text("Hello world.", 100, 20) == ["Hello world."]


def test_chunk_text_respects_size_and_overlaps():
    text = ". ".join(f"Sentence number {i}" for i in range(60)) + "."
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 3
    assert all(len(c) <= 200 for c in chunks)
    # Consecutive chunks share content because of the overlap.
    assert chunks[0][-20:] in chunks[1] or chunks[1][:20] in chunks[0]


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, chunk_overlap=10)


def test_chunk_pages_keeps_source_and_page():
    pages = [
        PageText(source="cours.pdf", page=1, text="Alpha beta gamma. " * 30),
        PageText(source="cours.pdf", page=2, text="Delta epsilon zeta. " * 30),
    ]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=40)
    assert chunks
    assert all(c.source == "cours.pdf" for c in chunks)
    assert {c.page for c in chunks} == {1, 2}
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
