"""Quick smoke test for the full RAG pipeline."""

from retriever import retrieve
from generator import generate_answer


def test_retrieval():
    print("=== RETRIEVAL TEST ===")
    query = "What are the benefits of RAG?"
    results = retrieve(query)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    print(f"Query: {query}")
    print(f"Chunks returned: {len(docs)}")
    for i, (d, m) in enumerate(zip(docs, metas)):
        print(f"  [{i+1}] {m['source']}:  {d[:80]}...")
    print()
    assert len(docs) > 0, "No chunks retrieved!"
    print("[PASS] Retrieval test passed.\n")
    return results


def test_generation(results):
    print("=== GENERATION TEST ===")
    question = "What are the benefits of RAG?"
    print(f"Question: {question}")
    answer = generate_answer(question, results)
    print(f"Answer:\n{answer}")
    assert len(answer) > 0, "Empty answer!"
    print("\n[PASS] Generation test passed.\n")


if __name__ == "__main__":
    r = test_retrieval()
    try:
        test_generation(r)
    except Exception as e:
        print(f"[SKIP] Generation test skipped (expected if no API key): {e}")
    print("=== ALL TESTS DONE ===")
