"""
Document ingestion pipeline.

Loads documents from the `documents/` folder, splits them into overlapping
chunks, embeds each chunk with sentence-transformers, and upserts everything
into a ChromaDB collection.

Usage
-----
    python ingest.py                     # index the default documents/ folder
    python ingest.py /path/to/folder     # index a custom folder
"""

import sys
from pathlib import Path
from typing import List, Tuple

import chromadb
from openai import OpenAI

import config


# ── Document loading ─────────────────────────────────────────────────────────

def _load_text_file(path: Path) -> str:
    """Read a plain-text or markdown file."""
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf_file(path: Path) -> str:
    """Extract text from every page of a PDF using pdfplumber."""
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


LOADERS = {
    ".txt": _load_text_file,
    ".md":  _load_text_file,
    ".pdf": _load_pdf_file,
}


def load_documents(directory: Path) -> List[Tuple[str, str]]:
    """
    Recursively load all supported files from *directory*.

    Returns a list of (filename, full_text) tuples.
    """
    docs: List[Tuple[str, str]] = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_dir():
            continue
        loader = LOADERS.get(file_path.suffix.lower())
        if loader is not None:
            try:
                text = loader(file_path)
                if text.strip():
                    docs.append((file_path.name, text))
            except Exception as e:
                print(f"[ERROR] Error loading {file_path}: {e}")
                # We continue to next file instead of raising to be more resilient
    return docs


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[str]:
    """
    Split *text* into overlapping windows of *chunk_size* characters.
    """
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Ingestion orchestrator ───────────────────────────────────────────────────

def ingest(directory: Path | None = None) -> int:
    """
    End-to-end ingestion: load → chunk → embed → store.

    Returns the total number of chunks inserted.
    """
    directory = directory or config.DOCUMENTS_DIR

    # 1. Load documents
    print(f"[LOAD] Loading documents from {directory} ...")
    docs = load_documents(directory)
    if not docs:
        print("[WARN] No supported documents found.")
        return 0
    print(f"    Found {len(docs)} document(s).")

    # 2. Chunk
    all_chunks: List[str] = []
    all_ids: List[str] = []
    all_meta: List[dict] = []

    for filename, text in docs:
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{filename}::chunk_{idx}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_meta.append({"source": filename, "chunk_index": idx})

    print(f"[CHUNK] Created {len(all_chunks)} chunk(s).")

    # 3. Embed
    print(f"[EMBED] Embedding with OpenAI API ({config.EMBEDDING_MODEL}) ...")
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    embeddings = []
    # OpenAI allows up to 2048 in a batch for standard tier, we use 100 to be safe
    EMBED_BATCH = 100  
    print(f"    Requesting embeddings in batches of {EMBED_BATCH}...")
    for i in range(0, len(all_chunks), EMBED_BATCH):
        batch = all_chunks[i : i + EMBED_BATCH]
        response = client.embeddings.create(
            input=batch,
            model=config.EMBEDDING_MODEL
        )
        batch_emb = [data.embedding for data in response.data]
        embeddings.extend(batch_emb)

    # 4. Upsert into ChromaDB
    print(f"[STORE] Storing in ChromaDB at {config.CHROMA_PERSIST_DIR} ...")
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

    # Delete existing collection to allow clean re-ingestion
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception as e:
        print(f"    [INFO] Could not delete collection (might be in use or not exist): {e}")

    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB has a batch-size limit; upsert in batches of 500
    BATCH = 500
    print(f"[STORE] Upserting {len(all_chunks)} chunks in batches of {BATCH}...")
    for i in range(0, len(all_chunks), BATCH):
        print(f"    Upserting batch {i//BATCH + 1}...")
        collection.upsert(
            ids=all_ids[i : i + BATCH],
            documents=all_chunks[i : i + BATCH],
            embeddings=embeddings[i : i + BATCH],
            metadatas=all_meta[i : i + BATCH],
        )

    print(f"[DONE] Ingestion complete - {collection.count()} chunks in collection '{config.COLLECTION_NAME}'.")
    return collection.count()


# ── CLI entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    ingest(folder)
