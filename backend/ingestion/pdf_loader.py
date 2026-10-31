import pymupdf
import pytesseract
from PIL import Image
from io import BytesIO
import re


# Tesseract location
pytesseract.pytesseract.tesseract_cmd = (
    r"D:\Program Files\Tesseract-OCR\tesseract.exe"
)


def is_text_good(text):
    """
    Check whether extracted PDF text looks readable.
    """

    if not text or len(text.strip()) < 50:
        return False

    # Count alphabetic characters
    letters = sum(char.isalpha() for char in text)

    # Count strange/suspicious characters
    strange = sum(
        not (char.isalnum() or char.isspace() or char in ".,;:!?-'\"()%/&")
        for char in text
    )

    total = len(text)

    letter_ratio = letters / total
    strange_ratio = strange / total

    # Basic quality check
    if letter_ratio < 0.35:
        return False

    if strange_ratio > 0.15:
        return False

    return True


def clean_text(text):
    """
    Basic text cleaning.
    """

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def extract_text_from_pdf(file_path):

    document = pymupdf.open(file_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        # --------------------------------
        # STEP 1: Normal PDF extraction
        # --------------------------------

        text = page.get_text().strip()

        extraction_method = "text"

        # --------------------------------
        # STEP 2: Check text quality
        # --------------------------------

        if not is_text_good(text):

            print(
                f"Page {page_number}: "
                f"Poor text extraction → Using OCR"
            )

            # Render PDF page as image
            pix = page.get_pixmap(dpi=300)

            image = Image.open(
                BytesIO(pix.tobytes("png"))
            )

            # --------------------------------
            # STEP 3: OCR
            # --------------------------------

            text = pytesseract.image_to_string(image)

            extraction_method = "ocr"

        else:

            print(
                f"Page {page_number}: "
                f"Text extraction successful"
            )

        # --------------------------------
        # STEP 4: Clean text
        # --------------------------------

        text = clean_text(text)

        pages.append({
            "page_number": page_number,
            "text": text,
            "extraction_method": extraction_method
        })

    document.close()

    return pages