from server.modules.embeddings import FakeEmbeddingProvider
from server.modules.indexing import chunk_id, index_chunks
from server.modules.pdf_handlers import Chunk
from server.modules.vector_store import InMemoryVectorStore


def _chunk(source="cours.pdf", page=1, index=0, text="Hello world."):
    return Chunk(source=source, page=page, chunk_index=index, text=text)


def test_fake_embeddings_are_deterministic_and_sized():
    embedder = FakeEmbeddingProvider(dimension=16)
    a = embedder.embed(["same text", "same text"])
    assert a[0] == a[1]
    assert len(a[0]) == 16
    assert embedder.embed(["other text"])[0] != a[0]


def test_chunk_id_is_stable_and_unique():
    c1 = _chunk(index=0)
    c2 = _chunk(index=1)
    assert chunk_id(c1) == chunk_id(_chunk(index=0))
    assert chunk_id(c1) != chunk_id(c2)
    assert chunk_id(c1) == "cours-p1-c0"


def test_index_chunks_stores_and_retrieves():
    store = InMemoryVectorStore()
    embedder = FakeEmbeddingProvider()
    chunks = [_chunk(index=0, text="Alpha beta gamma."),
              _chunk(page=2, index=1, text="Delta epsilon zeta.")]
    assert index_chunks(chunks, embedder, store) == 2
    assert store.count == 2
    # Querying with the exact embedding of chunk 0 returns it first.
    results = store.query(embedder.embed(["Alpha beta gamma."])[0], top_k=1)
    assert len(results) == 1
    assert results[0].metadata["source"] == "cours.pdf"
    assert results[0].metadata["page"] == 1
    assert results[0].text == "Alpha beta gamma."


def test_index_chunks_empty_is_noop():
    store = InMemoryVectorStore()
    assert index_chunks([], FakeEmbeddingProvider(), store) == 0
    assert store.count == 0


def test_reindexing_same_chunk_overwrites_not_duplicates():
    store = InMemoryVectorStore()
    embedder = FakeEmbeddingProvider()
    chunk = _chunk(index=0, text="Version one.")
    index_chunks([chunk], embedder, store)
    index_chunks([_chunk(index=0, text="Version two.")], embedder, store)
    assert store.count == 1
