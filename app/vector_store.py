from __future__ import annotations

from typing import Any

import chromadb


class SingleDocVectorStore:
    """Persistent Chroma store that keeps one active document at a time."""

    def __init__(self, path: str, collection_name: str = "single_doc") -> None:
        self._client = chromadb.PersistentClient(path=path)
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def reset(self) -> None:
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(name=self._collection_name)

    def add_chunks(
        self,
        doc_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        source_filename: str,
    ) -> None:
        ids = [f"{doc_id}-{idx}" for idx in range(len(chunks))]
        metadatas = [
            {
                "doc_id": doc_id,
                "chunk_index": idx,
                "source_file": source_filename,
            }
            for idx in range(len(chunks))
        ]
        self._collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], top_k: int = 3) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: list[dict[str, Any]] = []
        for idx, chunk_id in enumerate(ids):
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            score = max(0.0, 1.0 - distance)
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": documents[idx] if idx < len(documents) else "",
                    "score": round(score, 4),
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                }
            )

        return hits

    def peek(self, limit: int = 10, include_embeddings: bool = False) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []

        include = ["documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")

        result = self._collection.get(
            limit=limit,
            include=include,
        )

        ids = result.get("ids") or []
        documents = result.get("documents")
        metadatas = result.get("metadatas")
        embeddings = result.get("embeddings")

        documents = documents if documents is not None else []
        metadatas = metadatas if metadatas is not None else []
        embeddings = embeddings if embeddings is not None else []

        items: list[dict[str, Any]] = []
        for idx, chunk_id in enumerate(ids):
            item: dict[str, Any] = {
                "chunk_id": chunk_id,
                "text": documents[idx] if idx < len(documents) else "",
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
            }
            if include_embeddings and idx < len(embeddings):
                vector = list(embeddings[idx])
                item["embedding_dim"] = len(vector)
                item["embedding"] = vector
            items.append(item)

        return items

    def count(self) -> int:
        return self._collection.count()
