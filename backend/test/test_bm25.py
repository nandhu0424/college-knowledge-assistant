
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import create_chunks
from rag.bm25_retriever import create_bm25_index, search_bm25


# 1. Load the PDF
pdf_path = "data/sample/HS3151 Syllabus.pdf"

pages = extract_text_from_pdf(pdf_path)


# 2. Clean extracted text
for page in pages:
    page["text"] = clean_text(page["text"])


# 3. Create document chunks
chunks = create_chunks(
    pages,
    chunk_size=1000,
    overlap=150
)


# 4. Build BM25 index
bm25 = create_bm25_index(chunks)


# 5. Search using keywords
query = "What is HS3151?"

results = search_bm25(
    query=query,
    chunks=chunks,
    bm25=bm25,
    top_k=3
)


# 6. Display results
print("================================")
print("BM25 RETRIEVAL TEST")
print("================================")

print("Query:", query)
print("Total chunks:", len(chunks))
print("Retrieved results:", len(results))


for result in results:

    print("\n----------------------------")
    print("Score:", result["score"])
    print("Chunk ID:", result["chunk_id"])
    print("Page:", result["page_number"])

    print("\nText:")
    print(result["text"][:500])
