
from flashrank import Ranker, RerankRequest


# Initialize the reranking model
ranker = Ranker(
    model_name="ms-marco-TinyBERT-L-2-v2"
)


def rerank_documents(query, documents, top_k=3):
    """
    Rerank retrieved documents using FlashRank.
    """

    passages = []

    for document in documents:

        passages.append({
            "id": document["chunk_id"],
            "text": document["text"],
            "meta": {
                "page_number": document["page_number"]
            }
        })

    request = RerankRequest(
        query=query,
        passages=passages
    )

    results = ranker.rerank(request)

    return results[:top_k]
