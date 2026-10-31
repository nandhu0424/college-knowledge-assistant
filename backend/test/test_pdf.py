import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from ingestion.pdf_loader import extract_text_from_pdf

pdf_path = "data/sample/HS3151 Syllabus.pdf"

pages = extract_text_from_pdf(pdf_path)

for page in pages:

    print("\n==============================")
    print("PAGE:", page["page_number"])
    print("METHOD:", page["extraction_method"])
    print("==============================")

    print(page["text"][:1500])