import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from vectorstore.qdrant import client, COLLECTION_NAME


collection_info = client.get_collection(
    collection_name=COLLECTION_NAME
)

print("================================")
print("QDRANT COLLECTION")
print("================================")
print("Collection:", COLLECTION_NAME)
print("Points:", collection_info.points_count)
print("Vector size:", collection_info.config.params.vectors.size)
print("Distance:", collection_info.config.params.vectors.distance)