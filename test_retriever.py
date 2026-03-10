from src.retriever import get_retriever

retriever = get_retriever()

query = "What is the Artemis program?"
results = retriever.retrieve(query, top_k=5)

print(f"Retrieved {len(results)} chunks\n")

for i, r in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print("Title:", r.get("title"))
    print("Source:", r.get("source"))
    print("Page:", r.get("page"))
    print("Score:", r.get("score"))
    print("Text preview:", r.get("text", "")[:500])
    print()