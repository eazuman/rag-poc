from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main


class FakeVectorStore:
    """In-memory stand-in for SingleDocVectorStore so tests avoid real Chroma."""

    def __init__(self) -> None:
        self._chunks: list[str] = []

    def reset(self) -> None:
        self._chunks = []

    def add_chunks(self, doc_id, chunks, embeddings, source_filename) -> None:
        self._chunks = list(chunks)

    def count(self) -> int:
        return len(self._chunks)

    def peek(self, limit=10, include_embeddings=False):
        return [{"chunk_id": f"c-{i}", "text": t, "metadata": {}} for i, t in enumerate(self._chunks[:limit])]


@pytest.fixture
def client(monkeypatch):
    fake_store = FakeVectorStore()
    monkeypatch.setattr(main, "vector_store", fake_store)
    # Deterministic fake embeddings; dimension is irrelevant for the fake store.
    monkeypatch.setattr(main.embedder, "embed_texts", lambda texts: [[0.0] * 8 for _ in texts])
    return TestClient(main.app)


def test_health_starts_empty(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "chunks_indexed": 0}


def test_ingest_then_health_reports_chunks(client):
    files = {"file": ("sample.txt", b"hello world, this is a test document.", "text/plain")}
    resp = client.post("/ingest", files=files)
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "success"
    assert body["chunks_created"] >= 1
    assert body["doc_id"].startswith("sample.txt-")

    health = client.get("/health")
    assert health.json()["chunks_indexed"] >= 1


def test_ingest_rejects_non_txt(client):
    files = {"file": ("data.csv", b"a,b,c", "text/csv")}
    resp = client.post("/ingest", files=files)
    assert resp.status_code == 400


def test_ingest_rejects_empty_file(client):
    files = {"file": ("sample.txt", b"", "text/plain")}
    resp = client.post("/ingest", files=files)
    assert resp.status_code == 400


def test_reset_clears_store(client):
    files = {"file": ("sample.txt", b"some indexed content here", "text/plain")}
    client.post("/ingest", files=files)

    resp = client.delete("/reset")
    assert resp.status_code == 200
    assert client.get("/health").json()["chunks_indexed"] == 0
