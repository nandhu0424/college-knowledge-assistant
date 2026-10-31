
from rank_bm25 import BM25Okapi
import re


def tokenize(text):
    """
    Convert text into lowercase words.
    """
    return re.findall(r"\b\w+\b", text.lower())


def create_bm25_index(chunks):
    """
    Create a BM25 index using document chunks.
    """

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    return bm25


def search_bm25(query, chunks, bm25, top_k=3):
    """
    Retrieve relevant chunks using BM25 keyword search.
    """

    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:

        if scores[index] <= 0:
            continue

        results.append({
            "chunk_id": chunks[index]["chunk_id"],
            "page_number": chunks[index]["page_number"],
            "text": chunks[index]["text"],
            "score": float(scores[index])
        })

    return results
