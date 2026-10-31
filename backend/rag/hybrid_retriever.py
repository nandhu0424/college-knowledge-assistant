
def reciprocal_rank_fusion(
    semantic_results,
    bm25_results,
    k=60,
    top_k=5
):
    """
    Combine semantic and BM25 rankings using RRF.
    """

    fused_scores = {}
    chunk_data = {}

    # Process semantic search results
    for rank, result in enumerate(semantic_results, start=1):

        chunk_id = result.payload["chunk_id"]

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

        chunk_data[chunk_id] = {
            "chunk_id": chunk_id,
            "page_number": result.payload["page_number"],
            "text": result.payload["text"]
        }

    # Process BM25 search results
    for rank, result in enumerate(bm25_results, start=1):

        chunk_id = result["chunk_id"]

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

        chunk_data[chunk_id] = {
            "chunk_id": chunk_id,
            "page_number": result["page_number"],
            "text": result["text"]
        }

    # Sort chunks by combined RRF score
    ranked_chunks = sorted(
        fused_scores,
        key=lambda chunk_id: fused_scores[chunk_id],
        reverse=True
    )

    # Prepare final results
    final_results = []

    for chunk_id in ranked_chunks[:top_k]:

        result = chunk_data[chunk_id].copy()

        result["rrf_score"] = fused_scores[chunk_id]

        final_results.append(result)

    return final_results
