def create_chunks(pages, chunk_size=1000, overlap=150):
    """
    Create overlapping chunks without splitting words
    and without getting stuck in an infinite loop.
    """

    chunks = []

    for page in pages:

        text = page["text"]
        page_number = page["page_number"]
        extraction_method = page["extraction_method"]

        words = text.split()

        start = 0

        while start < len(words):

            current_words = []
            current_length = 0

            # Build the current chunk
            for i in range(start, len(words)):

                word = words[i]

                word_length = len(word)

                if current_words:
                    word_length += 1

                if current_length + word_length > chunk_size:
                    break

                current_words.append(word)
                current_length += word_length

            # Safety check
            if not current_words:
                current_words.append(words[start])

            chunk_text = " ".join(current_words)

            chunks.append({
                "chunk_id": len(chunks) + 1,
                "page_number": page_number,
                "text": chunk_text,
                "extraction_method": extraction_method
            })

            # -----------------------------
            # Calculate overlap
            # -----------------------------

            overlap_length = 0
            overlap_words = 0

            for word in reversed(current_words):

                word_length = len(word) + 1

                if overlap_length + word_length > overlap:
                    break

                overlap_length += word_length
                overlap_words += 1

            # ---------------------------------
            # IMPORTANT: Always move forward
            # ---------------------------------

            next_start = start + len(current_words) - overlap_words

            if next_start <= start:
                next_start = start + len(current_words)

            start = next_start

    return chunks