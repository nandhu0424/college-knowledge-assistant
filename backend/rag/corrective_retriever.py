
import ollama

from rag.retriever import search_documents
from rag.bm25_retriever import search_bm25
from rag.hybrid_retriever import reciprocal_rank_fusion
from rag.reranker import rerank_documents
from rag.crag import evaluate_retrieval


MODEL_NAME = "qwen2.5:3b-instruct"


def rewrite_query(query, evaluations):
    """
    Rewrite the user's query using CRAG evaluation feedback.
    """

    feedback = "\n".join(
        [
            f"Chunk {item['chunk_id']}: "
            f"{item['label']} - {item['reason']}"
            for item in evaluations
        ]
    )

    prompt = f"""
You are a query rewriting assistant for a college
knowledge management system.

Original user question:
{query}

Initial document evaluation:
{feedback}

Rewrite the original question to improve document retrieval.

Rules:
1. Preserve important course codes and identifiers.
2. Preserve the original meaning.
3. Use relevant keywords from the evaluation feedback.
4. Do not answer the question.
5. Return only the rewritten search query.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    rewritten_query = response["message"]["content"].strip()

    # Remove surrounding quotation marks if present
    rewritten_query = rewritten_query.strip('"').strip("'")

    if not rewritten_query:
        return query

    return rewritten_query


def perform_corrective_retrieval(
    query,
    initial_documents,
    initial_crag_result,
    chunks,
    bm25,
    top_k=3
):
    """
    Perform corrective retrieval when initial retrieval
    is ambiguous or incorrect.
    """

    initial_status = initial_crag_result["retrieval_status"]

    # If the initial retrieval is already correct,
    # no corrective retrieval is required.
    if initial_status == "correct":
        return {
            "corrective_attempted": False,
            "original_query": query,
            "rewritten_query": None,
            "initial_status": initial_status,
            "corrected_status": None,
            "final_status": initial_status,
            "final_documents": initial_documents,
            "final_evaluation": initial_crag_result
        }

    print("\nCRAG detected insufficient retrieval quality.")
    print("Starting corrective retrieval...")

    # Step 1: Rewrite the original query
    corrected_query = rewrite_query(
        query=query,
        evaluations=initial_crag_result["evaluations"]
    )

    print("\nOriginal Query:", query)
    print("Rewritten Query:", corrected_query)

    # Step 2: Perform semantic retrieval again
    semantic_results = search_documents(
        corrected_query,
        top_k=top_k
    )

    # Step 3: Perform BM25 retrieval again
    bm25_results = search_bm25(
        query=corrected_query,
        chunks=chunks,
        bm25=bm25,
        top_k=top_k
    )

    # Step 4: Combine results using RRF
    hybrid_results = reciprocal_rank_fusion(
        semantic_results=semantic_results,
        bm25_results=bm25_results,
        k=60,
        top_k=5
    )

    # Step 5: Rerank the corrected results
    if hybrid_results:
        corrected_documents = rerank_documents(
            query=corrected_query,
            documents=hybrid_results,
            top_k=top_k
        )
    else:
        corrected_documents = []

    # Step 6: Evaluate corrected retrieval
    corrected_evaluation = evaluate_retrieval(
        query=query,
        documents=corrected_documents
    )

    corrected_status = corrected_evaluation["retrieval_status"]

    # Step 7: Compare initial and corrected retrieval
    status_priority = {
        "incorrect": 0,
        "ambiguous": 1,
        "correct": 2
    }

    if status_priority[corrected_status] > status_priority[initial_status]:
        final_documents = corrected_documents
        final_evaluation = corrected_evaluation
        final_status = corrected_status
    else:
        final_documents = initial_documents
        final_evaluation = initial_crag_result
        final_status = initial_status

    return {
        "corrective_attempted": True,
        "original_query": query,
        "rewritten_query": corrected_query,
        "initial_status": initial_status,
        "corrected_status": corrected_status,
        "final_status": final_status,
        "final_documents": final_documents,
        "final_evaluation": final_evaluation
    }
