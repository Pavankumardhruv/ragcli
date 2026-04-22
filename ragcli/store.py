from pathlib import Path

import chromadb
from chromadb.config import Settings

from ragcli.loader import Chunk

DEFAULT_STORE_DIR = Path.home() / ".ragcli" / "store"
COLLECTION_NAME = "documents"


class VectorStore:
    def __init__(self, persist_dir: Path = DEFAULT_STORE_DIR):
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        ids = [f"{c.source}::chunk{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [{"source": c.source, "chunk_index": c.chunk_index} for c in chunks]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def query(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i],
            })
        return hits

    def count(self) -> int:
        return self._collection.count()

    def sources(self) -> list[str]:
        all_meta = self._collection.get(include=["metadatas"])
        return sorted({m["source"] for m in all_meta["metadatas"]})
