import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.cleaner import clean_text


pdf_path = "data/sample/HS3151 Syllabus.pdf"

pages = extract_text_from_pdf(pdf_path)

for page in pages:

    cleaned_text = clean_text(page["text"])

    print("\n==============================")
    print("PAGE:", page["page_number"])
    print("METHOD:", page["extraction_method"])
    print("==============================")

    print(cleaned_text)