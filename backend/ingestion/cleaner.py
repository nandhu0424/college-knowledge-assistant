import re
import unicodedata


def clean_text(text):
    # 1. Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # 2. Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 3. Join words split by hyphen + newline
    # Example:
    # communi-
    # cation
    # → communication
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 4. Normalize spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # 5. Remove spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # 6. Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 7. Remove leading/trailing spaces from each line
    lines = []

    for line in text.split("\n"):
        line = line.strip()

        if line:
            lines.append(line)

    # 8. Reconstruct cleaned text
    text = "\n".join(lines)

    return text.strip()