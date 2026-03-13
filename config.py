"""
Centralised configuration for the RAG Q&A system.
All tuneable parameters live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", str(BASE_DIR / "documents")))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_store"))

# ── ChromaDB ─────────────────────────────────────────────────────────────────
COLLECTION_NAME = "rag_docs"

# ── Embedding model (OpenAI API) ──────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"

# ── LLM  (Google Gemini — free tier: 15 RPM / 1 M tokens per day) ───────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = "gemini-2.5-flash"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between consecutive chunks

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = 5              # number of chunks to retrieve per query
