import ollama


MODEL_NAME = "nomic-embed-text"


def create_embedding(text):
    response = ollama.embed(
        model=MODEL_NAME,
        input=text
    )

    return response["embeddings"][0]