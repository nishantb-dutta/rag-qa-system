"""
Gradio-based web UI for the RAG Q&A system.

Run with:
    python app.py
"""

import shutil
from pathlib import Path
from typing import List

import gradio as gr


# ── Callback functions (lazy-import heavy modules) ───────────────────────────

def upload_files(files: List[str]) -> str:
    """Copy uploaded files into the documents/ folder."""
    if not files:
        return "No files selected."

    from config import DOCUMENTS_DIR
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for filepath in files:
        src = Path(filepath)
        dest = DOCUMENTS_DIR / src.name
        shutil.copy2(str(src), str(dest))
        saved.append(src.name)

    names = ", ".join(saved)
    return f"[OK] Uploaded {len(saved)} file(s): {names}. Now click 'Ingest Documents' to index them."


def run_ingest():
    """Re-index everything in the documents/ folder."""
    try:
        from ingest import ingest
        from retriever import reset_collection_cache
        count = ingest()
        reset_collection_cache()
        return f"[OK] Ingestion complete - {count} chunks indexed."
    except Exception as exc:
        return f"[ERROR] Ingestion failed: {exc}"


def ask_question(question: str):
    """Retrieve context and generate an answer."""
    if not question.strip():
        return "Please enter a question.", ""

    try:
        from retriever import retrieve
        results = retrieve(question)
    except Exception as exc:
        return f"[ERROR] Retrieval error: {exc}", ""

    # Build the sources panel
    sources_parts: list[str] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, (text, meta, dist) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        source = meta.get("source", "unknown")
        similarity = 1 - dist  # cosine distance -> similarity
        sources_parts.append(
            f"**[{i}] {source}** (similarity: {similarity:.2%})\n\n"
            f"> {text[:300]}{'...' if len(text) > 300 else ''}\n"
        )

    sources_md = "\n---\n".join(sources_parts) if sources_parts else "No sources found."

    # Generate answer
    try:
        from generator import generate_answer
        answer = generate_answer(question, results)
    except Exception as exc:
        answer = f"[ERROR] Generation error: {exc}"

    return answer, sources_md


# ── Gradio UI ────────────────────────────────────────────────────────────────

DESCRIPTION = """\
# RAG Q&A System

**Upload documents** below (PDF, TXT, Markdown), click **Ingest Documents** to index them,
then ask any question. The system retrieves relevant passages and uses
**Google Gemini** to generate a grounded answer.
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(title="RAG Q&A System") as demo:
    gr.Markdown(DESCRIPTION)

    # ── Upload & Ingest section ──────────────────────────────────────────
    with gr.Row():
        file_upload = gr.File(
            label="Upload Documents",
            file_count="multiple",
            file_types=[".pdf", ".txt", ".md"],
            scale=2,
        )
        upload_status = gr.Textbox(
            label="Upload Status",
            interactive=False,
            scale=2,
            placeholder="Drag & drop files here or click to browse...",
        )

    file_upload.change(fn=upload_files, inputs=file_upload, outputs=upload_status)

    with gr.Row():
        ingest_btn = gr.Button("Ingest Documents", variant="secondary", scale=1)
        ingest_status = gr.Textbox(
            label="Ingestion Status",
            interactive=False,
            scale=3,
            placeholder="Upload files above, then click 'Ingest Documents'...",
        )

    ingest_btn.click(fn=run_ingest, outputs=ingest_status)

    gr.Markdown("---")

    # ── Q&A section ──────────────────────────────────────────────────────
    question_box = gr.Textbox(
        label="Your Question",
        placeholder="e.g. What are the benefits of RAG?",
        lines=2,
    )
    ask_btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        answer_box = gr.Markdown(label="Answer")

    with gr.Accordion("Retrieved Sources", open=False):
        sources_box = gr.Markdown()

    ask_btn.click(fn=ask_question, inputs=question_box, outputs=[answer_box, sources_box])
    question_box.submit(fn=ask_question, inputs=question_box, outputs=[answer_box, sources_box])


if __name__ == "__main__":
    demo.launch(theme=THEME)
   #demo.launch(share=True)
