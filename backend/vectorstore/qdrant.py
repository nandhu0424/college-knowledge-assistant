from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


COLLECTION_NAME = "college_documents"

client = QdrantClient(
    host="localhost",
    port=6333
)


def create_collection():
    existing_collections = [
        collection.name
        for collection in client.get_collections().collections
    ]

    if COLLECTION_NAME not in existing_collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE
            )
        )

        print(f"Collection '{COLLECTION_NAME}' created successfully!")

    else:
        print(f"Collection '{COLLECTION_NAME}' already exists!")


def insert_chunks(chunks, embeddings):

    points = []

    for chunk, embedding in zip(chunks, embeddings):

        point = PointStruct(
            id=chunk["chunk_id"],
            vector=embedding,
            payload={
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "extraction_method": chunk["extraction_method"],
                "text": chunk["text"]
            }
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"{len(points)} chunks inserted into Qdrant.")