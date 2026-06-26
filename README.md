# RAG POC (Single Document, Minimal)

A simple, local Retrieval-Augmented Generation (RAG) proof of concept that runs from terminal.

This project demonstrates the core RAG workflow end to end:

1. Ingest text via API
2. Chunk and embed content
3. Store embeddings in a local vector database
4. Retrieve relevant chunks for a user question
5. Augment the prompt with retrieved context and generate an answer

## What This POC Covers

- Upload one `.txt` file through an API endpoint
- Store chunked embeddings in persistent ChromaDB
- Query with a question and get an answer grounded in retrieved chunks
- Return source chunks and metadata for transparency
- Keep one active document at a time (new ingest replaces previous index)

## Tech Stack

- Language: Python 3.10+
- API framework: FastAPI + Uvicorn
- Vector database: ChromaDB (persistent local storage)
- Embeddings: OpenAI `text-embedding-3-small`
- Answer generation: OpenAI `gpt-4o-mini`
- Multipart handling: `python-multipart`

No heavy orchestration framework is used (no LangChain / LlamaIndex). The pipeline is implemented directly in app modules for clarity.

## RAG Flow in This Repo

1. `POST /ingest`
- Read uploaded `.txt`
- Chunk text with overlap
- Create embeddings for chunks
- Reset existing collection (single-document behavior)
- Store chunks + vectors + metadata in Chroma

2. `POST /ask`
- Embed the question
- Retrieve top-k nearest chunks from Chroma
- Build context from retrieved chunks
- Send grounded prompt to the LLM
- Return final answer and source chunks

## Prerequisites

- macOS/Linux terminal (or Windows with equivalent commands)
- `uv` installed
- An OpenAI API key with available quota

## OpenAI API Key Requirement

You need an OpenAI API key to run this POC as currently implemented, because both embedding generation and final answer generation use OpenAI APIs.

Set key in `.env` (recommended):

```bash
echo 'OPENAI_API_KEY=sk-...' > .env
```

If `.env` is missing, `run.py` prompts for a key once and saves it automatically.

## Setup

1. Install `uv` (if needed)

```bash
brew install uv
```

2. Sync dependencies

```bash
uv sync
```

3. Configure API key (required for testing)

```bash
echo 'OPENAI_API_KEY=sk-...' > .env
```

4. Start the API server

```bash
uv run python run.py
```

Server runs at `http://localhost:8000` and docs are at `http://localhost:8000/docs`.

## Step-by-Step Test Using Sample Text

Use `samples/sample.txt` to verify ingest, retrieval, and answer generation.

1. Start server

```bash
uv run python run.py
```

2. Health check (should be empty initially)

```bash
curl http://localhost:8000/health
```

Expected shape:

```json
{"status":"ok","chunks_indexed":0}
```

3. Ingest sample file

```bash
curl -F "file=@samples/sample.txt" http://localhost:8000/ingest
```

Expected shape:

```json
{
  "status": "success",
  "doc_id": "sample.txt-...",
  "chunks_created": 1,
  "characters_processed": 100
}
```

4. Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?"}'
```

Expected shape:

```json
{
  "answer": "...",
  "sources": [
    {
      "chunk_id": "...",
      "text": "...",
      "score": 0.12,
      "metadata": {"chunk_index": 0, "doc_id": "...", "source_file": "sample.txt"}
    }
  ]
}
```

5. Confirm index count updated

```bash
curl http://localhost:8000/health
```

`chunks_indexed` should be greater than 0.

6. Optional reset for clean reruns

```bash
curl -X DELETE http://localhost:8000/reset
```

## API Endpoints

### `POST /ingest`

Upload and index one text file.

### `POST /ask`

Ask a grounded question against indexed content.

### `GET /health`

Basic health + indexed chunk count.

### `GET /debug/chunks`

Inspect stored chunks (dev/debug convenience). Supports query params:

- `limit` — max chunks to return (default 10)
- `snippet_length` — trim chunk text length (0 = full text)
- `include_embeddings` — set `true` to include embedding vectors
- `embedding_preview` — how many embedding numbers to show (0 = full vector)

```bash
curl "http://localhost:8000/debug/chunks?limit=5&include_embeddings=true&embedding_preview=8"
```

### `DELETE /reset`

Clear current vector collection.

## Notes

- This is intentionally a small POC, not a production system.
- For this POC, each ingest replaces previous indexed content.
- If you switch embedding model/provider later, reset and re-ingest documents.
- `/debug/chunks` is for learning/inspection — remove or protect it before any real deployment.
