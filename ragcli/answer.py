import os

import anthropic

from ragcli.embeddings import embed_query
from ragcli.store import VectorStore

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided document excerpts.

Rules:
- Only answer using the provided context. If the context doesn't contain the answer, say so.
- Cite which source file the information came from.
- Be concise and direct."""


def build_context(hits: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        source = hit["source"]
        text = hit["text"]
        parts.append(f"[{i}] Source: {source}\n{text}")
    return "\n\n---\n\n".join(parts)


def ask_question(
    question: str,
    store: VectorStore,
    top_k: int = 5,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Export it with:\n"
            "  export ANTHROPIC_API_KEY=your-key-here"
        )

    query_embedding = embed_query(question)
    hits = store.query(query_embedding, top_k=top_k)

    if not hits:
        return "No relevant documents found. Try ingesting some documents first."

    context = build_context(hits)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text
