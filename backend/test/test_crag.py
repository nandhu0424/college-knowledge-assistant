
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
from rag.crag import evaluate_retrieval
from rag.corrective_retriever import perform_corrective_retrieval


# 1. Load sample PDF
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

print("\n================================")
print("ALL DOCUMENT CHUNKS")
print("================================")

for chunk in chunks:
    print("\n--------------------------------")
    print("Chunk ID:", chunk["chunk_id"])
    print("Page:", chunk["page_number"])
    print("Text:")
    print(chunk["text"])

print("\nTotal Pages:", len(pages))
print("Total Chunks:", len(chunks))


# 4. Build BM25 index
bm25 = create_bm25_index(chunks)


# 5. Define query
query = "What topics are covered in Unit V of HS3151?"


# 6. Semantic search
semantic_results = search_documents(query, top_k=9)

print("\nSEMANTIC SEARCH:")
for result in semantic_results:
    print(
        f"Chunk ID: {result.payload['chunk_id']} | "
        f"Score: {result.score:.4f}"
    )

# 7. BM25 search
bm25_results = search_bm25(
    query=query,
    chunks=chunks,
    bm25=bm25,
    top_k=3
)

print("\n========== SEMANTIC SEARCH ==========")

for result in semantic_results:
    print("Chunk ID:", result.payload["chunk_id"])

print("\n========== BM25 SEARCH ==========")

for result in bm25_results:
    print("Chunk ID:", result["chunk_id"])

# 8. Hybrid search using RRF
hybrid_results = reciprocal_rank_fusion(
    semantic_results=semantic_results,
    bm25_results=bm25_results,
    k=60,
    top_k=5
)

print("\nHYBRID RETRIEVAL RESULTS")

for document in hybrid_results:
    print(
        "Chunk ID:", document["chunk_id"],
        "| Page:", document["page_number"]
    )

# 9. Cross-Encoder reranking
reranked_results = rerank_documents(
    query=query,
    documents=hybrid_results,
    top_k=5
)


# 10. Initial CRAG evaluation
initial_crag_result = evaluate_retrieval(
    query=query,
    documents=reranked_results
)


# 11. Corrective retrieval
final_result = perform_corrective_retrieval(
    query=query,
    initial_documents=reranked_results,
    initial_crag_result=initial_crag_result,
    chunks=chunks,
    bm25=bm25,
    top_k=3
)


# 12. Display retrieval information
print("\n========================================")
print("CRAG CORRECTIVE RETRIEVAL RESULTS")
print("========================================")

print("\nUser Query:")
print(query)

print("\nInitial Retrieval Status:")
print(initial_crag_result["retrieval_status"])

print("\nCorrective Retrieval Attempted:")
print(final_result["corrective_attempted"])

if final_result["corrective_attempted"]:

    print("\nRewritten Query:")
    print(final_result["rewritten_query"])

    print("\nCorrected Retrieval Status:")
    print(final_result["corrected_status"])


# 13. Display final retrieval status
print("\nFinal Retrieval Status:")
print(final_result["final_status"])


# 14. Display actual final retrieved documents
print("\n========================================")
print("FINAL RETRIEVED DOCUMENTS")
print("========================================")

for document in final_result["final_documents"]:

    print("\n----------------------------------------")

    print(
        "Chunk ID:",
        document.get("id", document.get("chunk_id"))
    )

    print(
        "Page Number:",
        document.get("meta", {}).get(
            "page_number",
            document.get("page_number", "N/A")
        )
    )

    print("\nDocument Text:")
    print(document["text"])


# 15. Display final CRAG evaluations
print("\n========================================")
print("FINAL CRAG DOCUMENT EVALUATIONS")
print("========================================")

for item in final_result["final_evaluation"]["evaluations"]:

    print("\n----------------------------------------")

    print("Chunk ID:", item["chunk_id"])
    print("Label:", item["label"])
    print("Reason:", item["reason"])


print("\n========================================")
print("CRAG TEST COMPLETED")
print("========================================")
