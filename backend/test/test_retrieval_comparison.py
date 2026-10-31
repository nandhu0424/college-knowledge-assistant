
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import create_chunks

from rag.bm25_retriever import create_bm25_index, search_bm25
from rag.retriever import search_documents


# 1. Load and process the PDF
pdf_path = "data/sample/HS3151 Syllabus.pdf"

pages = extract_text_from_pdf(pdf_path)

for page in pages:
    page["text"] = clean_text(page["text"])


# 2. Create chunks
chunks = create_chunks(
    pages,
    chunk_size=1000,
    overlap=150
)


# 3. Build BM25 index
bm25 = create_bm25_index(chunks)


# 4. Define the same query for both methods
query = "What is HS3151?"


# 5. Semantic search
semantic_results = search_documents(
    query,
    top_k=3
)


# 6. BM25 keyword search
bm25_results = search_bm25(
    query=query,
    chunks=chunks,
    bm25=bm25,
    top_k=3
)


# 7. Display semantic results
print("\n================================")
print("SEMANTIC SEARCH RESULTS")
print("================================")

for rank, result in enumerate(semantic_results, start=1):

    print(f"\nRank: {rank}")
    print("Chunk ID:", result.payload["chunk_id"])
    print("Page:", result.payload["page_number"])
    print("Score:", result.score)


# 8. Display BM25 results
print("\n================================")
print("BM25 SEARCH RESULTS")
print("================================")

for rank, result in enumerate(bm25_results, start=1):

    print(f"\nRank: {rank}")
    print("Chunk ID:", result["chunk_id"])
    print("Page:", result["page_number"])
    print("Score:", result["score"])


# 9. Compare retrieved chunk IDs
semantic_ids = [
    result.payload["chunk_id"]
    for result in semantic_results
]

bm25_ids = [
    result["chunk_id"]
    for result in bm25_results
]

print("\n================================")
print("RETRIEVAL COMPARISON")
print("================================")

print("Semantic chunks:", semantic_ids)
print("BM25 chunks:", bm25_ids)

print(
    "Common chunks:",
    sorted(set(semantic_ids) & set(bm25_ids))
)
