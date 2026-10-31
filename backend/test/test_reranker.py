
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import create_chunks

from rag.retriever import search_documents
from rag.bm25_retriever import create_bm25_index, search_bm25
from rag.hybrid_retriever import reciprocal_rank_fusion
from rag.reranker import rerank_documents


# 1. Load the sample PDF
pdf_path = "data/sample/HS3151 Syllabus.pdf"

pages = extract_text_from_pdf(pdf_path)


# 2. Clean extracted text
for page in pages:
    page["text"] = clean_text(page["text"])


# 3. Create chunks
chunks = create_chunks(
    pages,
    chunk_size=1000,
    overlap=150
)


# 4. Build BM25 index
bm25 = create_bm25_index(chunks)


# 5. Define query
query = "What is HS3151?"


# 6. Semantic search
semantic_results = search_documents(
    query,
    top_k=3
)


# 7. BM25 search
bm25_results = search_bm25(
    query=query,
    chunks=chunks,
    bm25=bm25,
    top_k=3
)


# 8. Hybrid search using RRF
hybrid_results = reciprocal_rank_fusion(
    semantic_results=semantic_results,
    bm25_results=bm25_results,
    k=60,
    top_k=5
)


# 9. Display hybrid results
print("\n================================")
print("BEFORE RERANKING - HYBRID SEARCH")
print("================================")

for rank, result in enumerate(hybrid_results, start=1):

    print("\nRank:", rank)
    print("Chunk ID:", result["chunk_id"])
    print("RRF Score:", result["rrf_score"])


# 10. Rerank hybrid results
reranked_results = rerank_documents(
    query=query,
    documents=hybrid_results,
    top_k=3
)


# 11. Display reranked results
print("\n================================")
print("AFTER RERANKING - CROSS ENCODER")
print("================================")

for rank, result in enumerate(reranked_results, start=1):

    print("\nRank:", rank)
    print("Chunk ID:", result["id"])
    print("Relevance Score:", result["score"])
    print("Page:", result["meta"]["page_number"])

    print("\nText:")
    print(result["text"][:300])


print("\n================================")
print("RERANKING TEST COMPLETED")
print("================================")
