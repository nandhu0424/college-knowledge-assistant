import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import create_chunks
from embeddings.nomic import create_embedding
from vectorstore.qdrant import insert_chunks


pdf_path = "data/sample/HS3151 Syllabus.pdf"


# 1. Extract PDF text
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


# 4. Create embeddings
embeddings = []

for chunk in chunks:

    embedding = create_embedding(chunk["text"])

    embeddings.append(embedding)

    print(
        f"Embedding created for chunk "
        f"{chunk['chunk_id']}"
    )


# 5. Insert into Qdrant
insert_chunks(chunks, embeddings)