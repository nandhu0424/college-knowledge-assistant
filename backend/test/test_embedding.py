import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import create_chunks
from embeddings.nomic import create_embedding


pdf_path = "data/sample/HS3151 Syllabus.pdf"

# 1. Extract PDF
pages = extract_text_from_pdf(pdf_path)

# 2. Clean text
for page in pages:
    page["text"] = clean_text(page["text"])

# 3. Create chunks
chunks = create_chunks(
    pages,
    chunk_size=1000,
    overlap=150
)

# 4. Take the first chunk
text = chunks[0]["text"]

# 5. Create embedding
embedding = create_embedding(text)

print("\n================================")
print("EMBEDDING TEST")
print("================================")

print("Chunk ID:", chunks[0]["chunk_id"])
print("Page:", chunks[0]["page_number"])
print("Embedding dimension:", len(embedding))

print("\nFirst 10 values:")
print(embedding[:10])