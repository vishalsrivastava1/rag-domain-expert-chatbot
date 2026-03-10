from src.rag_chain import get_chain

chain = get_chain()

query = "What is the Artemis program?"
result = chain.answer(query)

print("ANSWER:\n")
print(result["answer"])
print("\nEXPANDED QUERY:\n")
print(result["expanded_query"])
print("\nNUM CHUNKS:", result["num_chunks"])
print("\nSOURCES:")
for s in result["sources"]:
    print(f"[Source {s['num']}] {s['title']} | page {s['page']} | score={s['score']}")