import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag.retriever import search_documents


query = "What are the course objectives?"


results = search_documents(
    query,
    top_k=3
)


print("================================")
print("RETRIEVAL TEST")
print("================================")

print("Query:", query)
print("Results:", len(results))


for result in results:

    print("\n----------------------------")
    print("Score:", result.score)
    print("Chunk ID:", result.payload["chunk_id"])
    print("Page:", result.payload["page_number"])
    print("Method:", result.payload["extraction_method"])

    print("\nText:")
    print(result.payload["text"][:500])