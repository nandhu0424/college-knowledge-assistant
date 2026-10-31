from qdrant_client import QdrantClient

client = QdrantClient(
    host="localhost",
    port=6333
)

collections = client.get_collections()

print("================================")
print("QDRANT CONNECTION TEST")
print("================================")
print("Connected to Qdrant successfully!")
print("Collections:", collections)