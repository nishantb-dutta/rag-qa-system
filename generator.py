"""
Grounded answer generation using Google Gemini (free tier).

Takes retrieved context chunks and a user question, constructs a strict
grounding prompt, and calls Gemini to produce an answer that cites its sources.
"""

import time
from typing import Any, Dict, List

from google import genai

import config



# ── Gemini client ────────────────────────────────────────────────────────────

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not config.GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/apikey "
                "and add it to your .env file."
            )
        _client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _client


def reset_client():
    """Force re-creation of the client (useful after updating the API key)."""
    global _client
    _client = None


# ── Prompt engineering ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a helpful, precise Q&A assistant.

RULES:
1. Answer the user's question using ONLY the provided context passages.
2. If the context does not contain enough information, say:
   "I don't have enough information in the provided documents to answer this."
3. Cite your sources by mentioning the document name in square brackets,
   e.g. [report.pdf].
4. Keep your answer concise and well-structured.
"""


def _build_context_block(retrieval_results: Dict[str, Any]) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines: list[str] = []
    documents = retrieval_results.get("documents", [[]])[0]
    metadatas = retrieval_results.get("metadatas", [[]])[0]

    for i, (text, meta) in enumerate(zip(documents, metadatas), start=1):
        source = meta.get("source", "unknown")
        lines.append(f"--- Passage {i} (source: {source}) ---")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────────────────

def generate_answer(question: str, retrieval_results: Dict[str, Any]) -> str:
    """
    Generate a grounded answer for *question* using *retrieval_results*.
    Tries multiple free-tier models if the primary one fails.
    """
    context_block = _build_context_block(retrieval_results)

    user_message = (
        f"### Context\n{context_block}\n\n"
        f"### Question\n{question}"
    )

    client = _get_client()
    try:
        response = client.models.generate_content(
            model=config.LLM_MODEL,
            contents=[
                {"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + user_message}]},
            ],
        )
        return response.text or "(Gemini returned an empty response.)"

    except Exception as exc:
        err_str = str(exc)
        if "API_KEY_INVALID" in err_str or "API key expired" in err_str:
            raise RuntimeError(
                f"API key is invalid or expired. "
                "Please generate a new key at https://aistudio.google.com/apikey "
                "and update your .env file, then restart the app."
            )
        
        # Raise the original error (like 429 RESOURCE_EXHAUSTED) directly
        raise RuntimeError(f"Gemini API Error: {err_str}")
