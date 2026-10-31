
import json
import re
import ollama


MODEL_NAME = "qwen2.5:3b-instruct"


def normalize_text(text):
    """Normalize whitespace and case for evidence verification."""
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_identifiers(text):
    """
    Extract alphanumeric identifiers containing
    both letters and numbers.

    Examples: HS3151, CS8601, MA3151
    """
    return re.findall(
        r"\b[A-Z]{1,}[A-Z0-9]*\d+[A-Z0-9]*\b",
        text.upper()
    )


def evaluate_document(query, document):
    """
    Evaluate whether one document actually answers
    the user's question.
    """

    chunk_id = document.get(
        "chunk_id",
        document.get("id")
    )

    document_text = document.get("text", "")

    query_identifiers = extract_identifiers(query)
    document_identifiers = extract_identifiers(document_text)

    matched_identifiers = [
        identifier
        for identifier in query_identifiers
        if identifier in document_identifiers
    ]

    identifier_context = (
        ", ".join(matched_identifiers)
        if matched_identifiers
        else "No exact identifier match"
    )

    prompt = f"""
You are a careful relevance evaluator for a
college knowledge assistant.

USER QUESTION:
{query}

IDENTIFIER MATCH INFORMATION:
{identifier_context}

DOCUMENT:
{document_text}

Evaluate whether this document answers the
specific user question.

Use these labels:

correct:
The document explicitly contains information
that answers the specific question.

ambiguous:
The document contains related information,
but it is insufficient to confidently answer
the specific question.

incorrect:
The document has no meaningful relevance
to the specific question.

IMPORTANT RULES:

1. Evaluate the complete question, not just
   matching keywords or course codes.

2. A course code appearing in a document does
   NOT automatically make it correct.

3. If the question asks about a specific unit,
   check whether the document contains content
   belonging to that unit.

4. Do not assume that a list of topics belongs
   to a particular unit unless the document
   provides supporting evidence.

5. Do not invent information.

6. Evaluate only the supplied document.

7. If the document does not answer the question,
   do not label it correct.

Return valid JSON only:

{{
    "label": "correct",
    "reason": "Explain the relevance briefly",
    "evidence_quote": "An exact quotation copied from the document"
}}

For ambiguous or incorrect documents, evidence_quote
may be an empty string.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={
            "temperature": 0
        }
    )

    try:
        result = json.loads(
            response["message"]["content"]
        )

        label = result.get("label", "").lower()

        if label not in [
            "correct",
            "ambiguous",
            "incorrect"
        ]:
            raise ValueError("Invalid CRAG label")

        reason = result.get(
            "reason",
            "No reason provided."
        )

        evidence_quote = result.get(
            "evidence_quote",
            ""
        )

        # A correct classification must contain
        # a verifiable quotation from the document.
        if label == "correct":
            if (
                not evidence_quote
                or normalize_text(evidence_quote)
                not in normalize_text(document_text)
            ):
                label = "ambiguous"
                reason = (
                    "The evaluator did not provide "
                    "verifiable supporting evidence. "
                    + reason
                )

    except (json.JSONDecodeError, ValueError):
        label = "ambiguous"
        reason = "Could not reliably evaluate the document."
        evidence_quote = ""

    return {
        "chunk_id": chunk_id,
        "label": label,
        "reason": reason,
        "evidence_quote": evidence_quote
    }


def evaluate_retrieval(query, documents):
    """
    Evaluate all retrieved documents and determine
    the overall retrieval quality.
    """

    if not documents:
        return {
            "retrieval_status": "incorrect",
            "evaluations": []
        }

    evaluations = []

    for document in documents:
        evaluation = evaluate_document(
            query,
            document
        )

        evaluations.append(evaluation)

    labels = [
        item["label"]
        for item in evaluations
    ]

    if "correct" in labels:
        status = "correct"

    elif "ambiguous" in labels:
        status = "ambiguous"

    else:
        status = "incorrect"

    return {
        "retrieval_status": status,
        "evaluations": evaluations
    }
