import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag.retriever import search_documents
from rag.generator import generate_answer


query = "What is HS3151?"


# 1. Retrieve relevant chunks
results = search_documents(
    query,
    top_k=3
)


# 2. Prepare context
context_parts = []

for result in results:

    text = result.payload["text"]
    page = result.payload["page_number"]

    context_parts.append(
        f"[Page {page}]\n{text}"
    )


context = "\n\n".join(context_parts)


# 3. Generate answer
answer = generate_answer(
    query,
    context
)


# 4. Display answer
print("================================")
print("RAG TEST")
print("================================")

print("\nQuestion:")
print(query)

print("\nAnswer:")
print(answer)


# 5. Display sources
print("\n================================")
print("Sources")
print("================================")

source_pages = sorted(
    set(
        result.payload["page_number"]
        for result in results
    )
)

for page in source_pages:
    print(f"Page {page}")