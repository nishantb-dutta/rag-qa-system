"""
Retrieval module.

Embeds the user query with the same sentence-transformer model used during
ingestion and performs a cosine-similarity search against the ChromaDB
collection.
"""

from typing import Any, Dict

import chromadb
from sentence_transformers import SentenceTransformer

import config

# Cache model + client across calls so we don't reload on every query.
_model: SentenceTransformer | None = None
_collection: Any = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        _collection = client.get_collection(config.COLLECTION_NAME)
    return _collection


def reset_collection_cache():
    """Force the retriever to reconnect (useful after re-ingestion)."""
    global _collection
    _collection = None


def retrieve(query: str, top_k: int = config.TOP_K) -> Dict[str, Any]:
    """
    Retrieve the *top_k* most relevant chunks for *query*.

    Returns the raw ChromaDB query result dict with keys:
        ids, documents, metadatas, distances
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results
