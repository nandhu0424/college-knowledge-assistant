"""Configuration settings for the RAG backend."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"

EMBEDDING_MODEL = "nomic-embed-text"
VECTORSTORE_COLLECTION = "college_documents"
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = ""

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "llama3.1"
