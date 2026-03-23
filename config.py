"""
Centralised configuration for the RAG pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "QUALITY ASSURANCE AND REGULATORY AFFAIRS FOR THE BIOSCIENCES.pdf"
DATA_DIR = BASE_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# ── OpenAI ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# ── Embedding ────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 1536  # text-embedding-3-small output dimension

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 2000       # characters (~500 tokens)
CHUNK_OVERLAP = 200     # characters of overlap between chunks

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = 5               # number of chunks to retrieve per query

# ── Image extraction ─────────────────────────────────────────────────────────
MIN_IMAGE_SIZE = 5_000  # minimum image byte size to consider (skip tiny icons)
