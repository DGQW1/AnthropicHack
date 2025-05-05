"""
Configuration settings for the RAG application
"""

import os
from pathlib import Path

# API Keys (read from environment variables)
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# File paths and storage
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", BASE_DIR / "data"))
STORAGE_DIR = DATA_DIR / "storage"
FAISS_DIR = DATA_DIR / "faiss_store"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True, parents=True)
STORAGE_DIR.mkdir(exist_ok=True, parents=True)
FAISS_DIR.mkdir(exist_ok=True, parents=True)

# Default folders to process
DEFAULT_CONTENT_FOLDERS = [
    "270materials/270handout",
    "270materials/270questions",
    "270materials/270slides"
]

# Processing parameters
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
MAX_TEXT_LENGTH = 20000  # For truncating long documents when sending to Claude

# Model settings
EMBEDDING_MODEL = "voyage-3"
LLM_MODEL = "claude-3-5-haiku-20241022"
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 200
SUMMARY_MAX_TOKENS = 200
CONCEPT_MAX_TOKENS = 50