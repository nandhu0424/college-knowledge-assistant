import ollama


MODEL_NAME = "qwen2.5:3b-instruct"


def generate_answer(query, context):

    prompt = f"""
You are a College Knowledge Assistant.

Answer the user's question using ONLY the information
provided in the context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not available in the context,
   say that you could not find the answer.
4. Give a clear and concise answer.
5. Do not mention the retrieval process.

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]