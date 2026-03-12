"""
Retrieval module.

Embeds the user query with the same sentence-transformer model used during
ingestion and performs a cosine-similarity search against the ChromaDB
collection.
"""

from typing import Any, Dict

import chromadb
from openai import OpenAI

import config

# Cache model + client across calls so we don't reload on every query.
_openai_client: OpenAI | None = None
_collection: Any = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


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
    client = _get_client()
    collection = _get_collection()

    response = client.embeddings.create(
        input=[query],
        model=config.EMBEDDING_MODEL
    )
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results
