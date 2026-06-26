from __future__ import annotations

import os

from openai import OpenAI


class OpenAIEmbedder:
    """Small wrapper around OpenAI embeddings API."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.model = model
        self._client = OpenAI(api_key=api_key)

    @property
    def client(self) -> OpenAI:
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []
