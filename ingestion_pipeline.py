"""
Ingestion Pipeline
──────────────────
Reads the extracted page JSON files, chunks the text, generates embeddings
with OpenAI, and builds + persists a FAISS index.

Output:
  • data/faiss_index/index.faiss
  • data/faiss_index/metadata.pkl
"""

import json
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EXTRACTED_DIR,
    FAISS_INDEX_DIR,
    OPENAI_API_KEY,
)

client = OpenAI(api_key=OPENAI_API_KEY)


# ── chunking ─────────────────────────────────────────────────────────────────


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split *text* into overlapping chunks using a simple recursive approach.
    Tries to split on paragraph breaks → sentence ends → spaces.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " "]
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Find the best split point
        split_pos = end
        for sep in separators:
            pos = text.rfind(sep, start, end)
            if pos > start:
                split_pos = pos + len(sep)
                break

        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward, accounting for overlap
        start = max(start + 1, split_pos - overlap)

    return chunks


def _build_page_text(page_data: dict) -> str:
    """Combine page text and image captions into a single text block."""
    parts = []
    if page_data.get("text"):
        parts.append(page_data["text"])
    for caption in page_data.get("image_captions", []):
        parts.append(f"[Figure] {caption}")
    return "\n\n".join(parts)





def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI embeddings API."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def _embed_all(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """Embed all texts in batches and return a numpy array."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"    Embedding batch {i // batch_size + 1} ({len(batch)} chunks) …")
        embeddings = _embed_batch(batch)
        all_embeddings.extend(embeddings)
    return np.array(all_embeddings, dtype="float32")





def run_ingestion() -> None:
    """Run the full ingestion pipeline: load → chunk → embed → index."""
    if not EXTRACTED_DIR.exists():
        print("Extracted data not found. Run extraction_pipeline.py first.")
        sys.exit(1)

    json_files = sorted(EXTRACTED_DIR.glob("page_*.json"))
    if not json_files:
        print("No page JSON files found in extracted directory.")
        sys.exit(1)

    print(f"📂  Found {len(json_files)} extracted page files")

    # ── Step 1: Chunk ────────────────────────────────────────────────────────
    all_chunks: list[dict] = []  # {text, page_number, chunk_id}
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as fp:
            page_data = json.load(fp)

        page_text = _build_page_text(page_data)
        if not page_text.strip():
            continue

        chunks = _chunk_text(page_text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "text": chunk,
                    "page_number": page_data["page_number"],
                    "chunk_id": f"p{page_data['page_number']}_c{i}",
                }
            )

    print(f"Created {len(all_chunks)} chunks from {len(json_files)} pages")

    if not all_chunks:
        print("No chunks produced — nothing to index.")
        sys.exit(1)

    # ── Step 2: Embed ────────────────────────────────────────────────────────
    print("Generating embeddings …")
    start = time.time()
    texts = [c["text"] for c in all_chunks]
    embeddings = _embed_all(texts)
    elapsed = time.time() - start
    print(f"    Done in {elapsed:.1f}s  (shape: {embeddings.shape})")

    # ── Step 3: Build FAISS index ────────────────────────────────────────────
    print("Building FAISS index …")
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    print(f"    Index size: {index.ntotal} vectors")

    # ── Step 4: Persist ──────────────────────────────────────────────────────
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = FAISS_INDEX_DIR / "index.faiss"
    faiss.write_index(index, str(index_path))

    metadata_path = FAISS_INDEX_DIR / "metadata.pkl"
    with open(metadata_path, "wb") as fp:
        pickle.dump(all_chunks, fp)

    print(f"\n Ingestion complete")
    print(f"    Index  → {index_path}")
    print(f"    Meta   → {metadata_path}")


if __name__ == "__main__":
    run_ingestion()
