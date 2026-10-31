import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import create_chunks


pdf_path = "data/sample/HS3151 Syllabus.pdf"


# 1. Extract PDF text
pages = extract_text_from_pdf(pdf_path)


# 2. Clean the text
for page in pages:
    page["text"] = clean_text(page["text"])


# 3. Create chunks
chunks = create_chunks(
    pages,
    chunk_size=1000,
    overlap=150
)


print("\n================================")
print("TOTAL CHUNKS:", len(chunks))
print("================================")


# Display first 3 chunks
for chunk in chunks[:3]:

    print("\n--------------------------------")
    print("CHUNK ID:", chunk["chunk_id"])
    print("PAGE:", chunk["page_number"])
    print("METHOD:", chunk["extraction_method"])
    print("--------------------------------")

    print(chunk["text"])