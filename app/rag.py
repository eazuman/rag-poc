from __future__ import annotations

from app.embedder import OpenAIEmbedder
from app.vector_store import SingleDocVectorStore


class RAGService:
    """Retrieve relevant chunks and ask the chat model for a grounded answer."""

    def __init__(
        self,
        embedder: OpenAIEmbedder,
        vector_store: SingleDocVectorStore,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_model = llm_model

    def answer_question(self, question: str, top_k: int = 3) -> dict:
        if self.vector_store.count() == 0:
            return {
                "answer": "No document is indexed yet. Upload a .txt file first.",
                "sources": [],
            }

        query_embedding = self.embedder.embed_query(question)
        hits = self.vector_store.query(query_embedding=query_embedding, top_k=top_k)
        if not hits:
            return {
                "answer": "I could not find relevant context in the indexed document.",
                "sources": [],
            }

        context = "\n\n".join(
            [f"Chunk {idx + 1}:\n{hit['text']}" for idx, hit in enumerate(hits)]
        )

        system_prompt = (
            "You are a helpful RAG assistant. "
            "Answer the question using the information in the provided context. "
            "The answer may need to be inferred or synthesized from relevant facts, "
            "even if the wording does not exactly match the question. "
            "If the context contains no relevant information to answer the question, "
            "reply: 'I do not know based on the provided document.' "
            "Do not invent facts that are not supported by the context."
        )
        user_prompt = f"Question: {question}\n\nContext:\n{context}"

        response = self.embedder.client.chat.completions.create(
            model=self.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = (response.choices[0].message.content or "").strip()
        return {"answer": answer, "sources": hits}
