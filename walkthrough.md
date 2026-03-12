# RAG Q&A System — Walkthrough

## What Was Built

A complete **Retrieval-Augmented Generation** Q&A system in `c:\Users\nisha\Antigravity001\`:

| File | Purpose |
|------|---------|
| [config.py](file:///c:/Users/nisha/Antigravity001/config.py) | Central settings (models, paths, chunking params) |
| [ingest.py](file:///c:/Users/nisha/Antigravity001/ingest.py) | Load docs -> chunk -> embed -> store in ChromaDB |
| [retriever.py](file:///c:/Users/nisha/Antigravity001/retriever.py) | Cosine-similarity search over ChromaDB |
| [generator.py](file:///c:/Users/nisha/Antigravity001/generator.py) | Grounded answer generation via Gemini free tier |
| [app.py](file:///c:/Users/nisha/Antigravity001/app.py) | Gradio web UI (ingest button, Q&A, source viewer) |
| [test_pipeline.py](file:///c:/Users/nisha/Antigravity001/test_pipeline.py) | End-to-end smoke test |

### Free-Tier Models Used
- **Embeddings**: `all-MiniLM-L6-v2` (sentence-transformers, runs 100% locally)
- **LLM**: Google Gemini `gemini-2.0-flash` (free tier: 15 req/min, 1M tokens/day)
- **Vector DB**: ChromaDB (local, open-source, zero-cost)

## Running App

![RAG Q&A System UI](C:/Users/nisha/.gemini/antigravity/brain/33f39d6d-f8b8-4509-aacb-d90242c290e7/rag_system_ui_1773313056996.png)

## Verification Results

- **Ingestion**: Passed - 5 chunks indexed from `sample.txt`
- **Retrieval**: Passed - 5 relevant chunks returned for test query
- **Generation**: Correctly requires API key (free key from https://aistudio.google.com/apikey)

## Final Setup & Finishing Guide

Follow these exact steps to resolve API key issues and finish your product:

### 1. Resolve API Key (Quota Issues)
The `429 RESOURCE_EXHAUSTED` error means your current API key has no remaining free-tier quota (this often happens if a key is flagged for unusual activity or shared/leaked).

1.  **Generate a FRESH key**: Go to [Google AI Studio](https://aistudio.google.com/apikey).
2.  **Delete any old keys** first to avoid confusion.
3.  **Create a NEW key**.
4.  **Update `.env`**:
    - Open `c:\Users\nisha\Antigravity001\.env`.
    - Replace the `GOOGLE_API_KEY=` value with your brand new key.
    - **Save the file (Ctrl+S)**.

### 2. Restart the Application
The environment variable is only loaded when Python starts. You **must** restart for the new key to take effect.

1.  Stop the current process (press **Ctrl+C** in the terminal).
2.  Start it again:
    ```powershell
    & "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" app.py
    ```

### 3. Verify in the Browser
1.  Open **http://127.0.0.1:7860**.
2.  **Ingest Documents**: Click the button once to ensure everything is indexed.
3.  **Ask a Question**: Type a question that is answered in your documents (e.g., about badminton scores if using the PDF).

### Troubleshooting Common Errors
- **404 NOT_FOUND**: I have removed the buggy model fallback logic, so you should no longer see 404s for the model. Ensure `config.py` lists `LLM_MODEL = "gemini-2.0-flash"`.
- **429 RESOURCE_EXHAUSTED**: If this persists with a *new* key, wait 60 seconds (rate limiting) or try a different Google account for a key.
- **"No response"**: The UI might take 5-10 seconds to generate an answer. Check your terminal for logs.

### GitHub
Your code is pushed to: [nishantb-dutta/rag-qa-system](https://github.com/nishantb-dutta/rag-qa-system).
> [!NOTE]
> `.env` and `chroma_store/` are excluded from GitHub for security and efficiency.
