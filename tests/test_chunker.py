from __future__ import annotations

import pytest

from app.chunker import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_is_single_chunk():
    chunks = chunk_text("hello world", chunk_size=500, overlap=50)
    assert chunks == ["hello world"]


def test_long_text_splits_into_multiple_chunks():
    text = "A" * 1300
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    # step = 450 -> starts at 0, 450, 900, 1350(stop) => 3 chunks
    assert len(chunks) == 3
    assert all(len(c) <= 500 for c in chunks)


def test_overlap_preserves_context_between_chunks():
    text = "0123456789" * 60  # 600 chars
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    # Tail of first chunk should reappear at the head of the second chunk.
    assert chunks[0][-20:] == chunks[1][:20]


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=0)


def test_overlap_not_smaller_than_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, overlap=100)
