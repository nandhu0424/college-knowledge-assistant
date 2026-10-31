from embeddings.nomic import create_embedding
from vectorstore.qdrant import client, COLLECTION_NAME


def search_documents(query, top_k=3):

    # Convert user query into an embedding
    query_embedding = create_embedding(query)

    # Search Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True
    )

    return results.points