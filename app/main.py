from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import APIError, AuthenticationError, RateLimitError
from pydantic import BaseModel, Field

from app.chunker import chunk_text
from app.embedder import OpenAIEmbedder
from app.rag import RAGService
from app.vector_store import SingleDocVectorStore

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

embedder = OpenAIEmbedder(model="text-embedding-3-small")
vector_store = SingleDocVectorStore(path=str(CHROMA_DIR))
rag_service = RAGService(embedder=embedder, vector_store=vector_store, llm_model="gpt-4o-mini")

app = FastAPI(title="RAG POC API", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


def make_doc_id(filename: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    safe_name = Path(filename).name.replace(" ", "_")
    return f"{safe_name}-{digest}"


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    """Chunk, embed, and index a single .txt file (replaces previous content)."""
    filename = file.filename or "uploaded.txt"
    if not filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 encoded text",
        ) from exc

    chunks = chunk_text(text=text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text content found to index")

    doc_id = make_doc_id(filename=filename, text=text)
    save_path = UPLOAD_DIR / f"{doc_id}.txt"
    save_path.write_bytes(raw_bytes)

    try:
        embeddings = embedder.embed_texts(chunks)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail="OpenAI API authentication failed. Check OPENAI_API_KEY.",
        ) from exc
    except RateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail=(
                "OpenAI quota/rate limit reached while creating embeddings. "
                "Check billing/quota and try again."
            ),
        ) from exc
    except APIError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API error while creating embeddings: {exc}",
        ) from exc

    # Single-document POC behavior: replace existing indexed content on each ingest.
    vector_store.reset()
    vector_store.add_chunks(
        doc_id=doc_id,
        chunks=chunks,
        embeddings=embeddings,
        source_filename=filename,
    )

    return {
        "status": "success",
        "doc_id": doc_id,
        "chunks_created": len(chunks),
        "characters_processed": len(text),
    }


@app.post("/ask")
async def ask(payload: AskRequest) -> dict:
    """Answer a question using top-K retrieved chunks from the indexed document."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        return rag_service.answer_question(question=question, top_k=TOP_K)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail="OpenAI API authentication failed. Check OPENAI_API_KEY.",
        ) from exc
    except RateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail=(
                "OpenAI quota/rate limit reached while generating answer. "
                "Check billing/quota and try again."
            ),
        ) from exc
    except APIError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API error while generating answer: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}") from exc


@app.get("/health")
async def health() -> dict:
    """Return basic service health and the number of indexed chunks."""
    return {"status": "ok", "chunks_indexed": vector_store.count()}


@app.get("/debug/chunks")
async def debug_chunks(
    limit: int = 10,
    snippet_length: int = 200,
    include_embeddings: bool = False,
    embedding_preview: int = 8,
) -> dict:
    """Inspect stored chunks (and optionally embeddings) for debugging."""
    items = vector_store.peek(limit=limit, include_embeddings=include_embeddings)
    for item in items:
        text = item.get("text") or ""
        if snippet_length > 0 and len(text) > snippet_length:
            item["text"] = text[:snippet_length] + "..."

        if include_embeddings and "embedding" in item and embedding_preview > 0:
            vector = item["embedding"]
            if len(vector) > embedding_preview:
                item["embedding"] = vector[:embedding_preview]
                item["embedding_truncated"] = True
    return {"chunks_indexed": vector_store.count(), "returned": len(items), "chunks": items}


@app.delete("/reset")
async def reset() -> dict:
    """Clear the vector store, removing all indexed chunks."""
    vector_store.reset()
    return {"status": "success", "message": "Vector store reset"}
