# Design Document — RAG Pipeline for QA & Regulatory Affairs

## 1. Architecture

### 1.1 System Overview

The system follows a **two-phase pipeline architecture** with a clear separation between offline data preparation and online query serving:

```
┌──────────────────── OFFLINE (run once) ────────────────────┐
│                                                             │
│  PDF File                                                   │
│    │                                                        │
│    ▼                                                        │
│  Extraction Pipeline (extraction_pipeline.py)               │
│    • PyMuPDF parses text per page                           │
│    • Extracts embedded images                               │
│    • GPT-4o-mini (vision) generates image captions          │
│    • Output: JSON per page → data/extracted/                │
│    │                                                        │
│    ▼                                                        │
│  Ingestion Pipeline (ingestion_pipeline.py)                 │
│    • Loads page JSONs                                       │
│    • Chunks text (recursive, ~2000 chars, 200 overlap)      │
│    • Embeds chunks via text-embedding-3-small (1536-dim)    │
│    • Builds FAISS IndexFlatL2                               │
│    • Persists index + metadata → data/faiss_index/          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌──────────────────── ONLINE (per query) ────────────────────┐
│                                                             │
│  User Question                                              │
│    │                                                        │
│    ▼                                                        │
│  Query Pipeline (query_pipeline.py)                         │
│    1. Embed query with text-embedding-3-small               │
│    2. FAISS search → top-5 chunks                           │
│    3. Build prompt (system + context + question)            │
│    4. GPT-4o-mini generates answer with [Page N] citations  │
│    5. Return answer + citation metadata                     │
│    │                                                        │
│    ▼                                                        │
│  FastAPI (app.py) → JSON response or Web UI                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Component Responsibilities

| Component | Technology | Role |
|---|---|---|
| PDF parsing | PyMuPDF (fitz) | Text extraction, image extraction |
| Image captioning | GPT-4o-mini (vision) | Converts figures to searchable text |
| Chunking | Custom recursive splitter | Splits text into semantically coherent chunks |
| Embedding | OpenAI text-embedding-3-small | Converts text → 1536-dim vectors |
| Vector store | FAISS (IndexFlatL2) | Stores and searches embedding vectors |
| Answer generation | GPT-4o-mini | Generates cited answers from retrieved context |
| API layer | FastAPI + Uvicorn | HTTP interface with web UI |

---

## 2. RAG Design

### 2.1 PDF Chunking Strategy

**Approach:** Recursive character splitting with paragraph-aware boundaries.

- **Chunk size:** ~2000 characters (~500 tokens). This balances between:
  - Too small: loses context, fragments sentences
  - Too large: dilutes relevance, wastes LLM context window
- **Overlap:** 200 characters. Ensures information at chunk boundaries is not lost.
- **Split hierarchy:** The splitter tries `\n\n` (paragraph) → `\n` (line) → `. ` (sentence) → ` ` (word) to find natural break points.
- **Image handling:** Image captions are appended to page text with a `[Figure]` prefix, then chunked together with the surrounding text. This keeps figures contextually linked to their source page.

**Why not fixed-token chunking?** Character-based splitting with natural-boundary heuristics produces more coherent chunks than blind token counting, while being simpler than full semantic chunking.

### 2.2 Embedding Model

**Model:** `text-embedding-3-small` (OpenAI)

| Factor | Reasoning |
|---|---|
| Dimension | 1536 — good balance of quality vs. storage cost |
| Quality | Strong performance on retrieval benchmarks (MTEB) |
| Cost | ~$0.02 per 1M tokens — very cost-effective for ~100 pages |
| Consistency | Same model embeds both documents and queries, avoiding distribution mismatch |

### 2.3 FAISS Configuration

| Parameter | Value | Reasoning |
|---|---|---|
| Index type | `IndexFlatL2` | Exact nearest-neighbour search; optimal for small corpora (<10K vectors) |
| Distance metric | L2 (Euclidean) | Default for OpenAI embeddings; equivalent ranking to cosine with normalised vectors |
| Dimensionality | 1536 | Matches text-embedding-3-small output |
| Persistence | `faiss.write_index` / `faiss.read_index` | Simple file-based persistence to disk |

For ~100 pages producing ~200-400 chunks, `IndexFlatL2` provides sub-millisecond exact search. Approximate indices (IVF, HNSW) would add complexity without benefit at this scale.

### 2.4 Query Pipeline Flow

```
User question: "What is GMP?"
        │
        ▼
   ┌─────────────────┐
   │ Embed query      │  text-embedding-3-small → [1536-dim vector]
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │ FAISS search     │  IndexFlatL2.search(query_vec, k=5) → top-5 chunks
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │ Build prompt     │  System msg + "Page 12: chunk text..." + question
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │ LLM call         │  GPT-4o-mini (temp=0.2, max_tokens=1024)
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │ Return answer    │  Answer text + [Page 12, Page 15, ...] citations
   └─────────────────┘
```

**Citation mechanism:** The system prompt instructs the LLM to cite every claim as `[Page N]`. The API response also includes structured citation metadata (page number, chunk ID, L2 distance score).

---

## 3. Scaling Thought Experiment

### 3.1 Scaling to Many Large Documents & Thousands of Users

To evolve this prototype for production use with many documents and high concurrency:

#### Vector Store Scaling

| Challenge | Solution |
|---|---|
| Millions of vectors | Switch from `IndexFlatL2` to `IndexIVFPQ` or `IndexHNSWFlat` for sub-linear search |
| Disk-based index | Use `IndexIVFPQ` with `OnDiskInvertedLists` for memory-efficient storage |
| Multi-document search | Add document-level metadata filtering (pre-filter by doc ID before FAISS search) |
| Index updates | Implement incremental index building; use `IndexIDMap` to support add/remove |

#### Embedding Scaling

| Challenge | Solution |
|---|---|
| Batch throughput | Parallelize embedding calls; use batch sizes of 500-2000 |
| Cost at scale | Consider self-hosted embedding models (e.g., `sentence-transformers` on GPU) |
| Latency | Cache query embeddings for repeated questions |

#### LLM Scaling

| Challenge | Solution |
|---|---|
| API rate limits | Implement request queuing with exponential backoff |
| Latency | Use streaming responses to reduce perceived latency |
| Cost | Cache LLM responses for identical queries; use cheaper models for simple questions |

#### Infrastructure

| Challenge | Solution |
|---|---|
| Concurrency | Deploy behind a load balancer with multiple FastAPI workers (Gunicorn) |
| Caching | Redis cache for embeddings + LLM responses (TTL-based expiry) |
| Async | Use async OpenAI client for non-blocking I/O in the web server |

### 3.2 Achieving P95 Latency < 2 Seconds

**Latency budget breakdown (target: < 2s P95):**

| Stage | Current | Optimised Target | Technique |
|---|---|---|---|
| Query embedding | ~200ms | ~100ms | Batch / cache common queries |
| FAISS search | <1ms | <10ms (at 10M vectors) | IVF index with nprobe=16 |
| LLM generation | ~1.5s | ~1s | Streaming, prompt compression |
| Network + overhead | ~50ms | ~50ms | Keep-alive connections |
| **Total** | **~1.75s** | **~1.16s** | |

**Key trade-offs:**
- **IVF nprobe vs. accuracy:** Lower `nprobe` = faster but less accurate retrieval. At nprobe=16 with 256 centroids, recall@10 stays above 95%.
- **Smaller chunks vs. more calls:** Smaller chunks improve retrieval precision but require more LLM context. Optimal chunk size depends on document structure.
- **Caching vs. freshness:** Aggressive caching improves latency but may serve stale answers if the index is updated.

### 3.3 Additional Scaling Considerations

- **Sharding:** Partition the FAISS index by document category or date range. Route queries to relevant shards.
- **Reranking:** Add a cross-encoder reranking step after FAISS retrieval to improve precision (adds ~100ms).
- **Hybrid search:** Combine dense (FAISS) and sparse (BM25) retrieval for better recall on keyword-heavy queries.

---

## 4. Limitations and Next Steps

### Current Limitations

1. **No table/structured-data extraction.** PyMuPDF extracts table text as raw characters, losing row/column structure. This degrades answers about tabular regulatory data.
2. **Fixed chunking strategy.** The recursive character splitter does not understand document section boundaries (chapters, headings). Chunks may split mid-section.
3. **No reranking.** FAISS returns chunks by L2 distance alone. A cross-encoder reranker would significantly improve retrieval quality.
4. **Single-document scope.** The system assumes a single PDF. Multi-document support would require document-level metadata and filtering.
5. **No conversation memory.** Each query is independent — no follow-up question support.
6. **Flat L2 index.** Exact search is fine for <1K vectors but does not scale to millions.

### Recommended Next Steps (Priority Order)

1. **Structured extraction:** Use a layout-aware PDF parser (e.g., `unstructured`, `docling`) to preserve table structure and section headings.
2. **Section-aware chunking:** Parse document structure (TOC, headings) and chunk within section boundaries. Add section titles as metadata.
3. **Cross-encoder reranking:** After FAISS retrieval, rerank with a model like `cross-encoder/ms-marco-MiniLM-L-6-v2` to improve precision.
4. **Hybrid retrieval:** Combine FAISS (dense) with BM25 (sparse) search using Reciprocal Rank Fusion for better recall.
5. **Conversation memory:** Add a conversation buffer to support follow-up questions and multi-turn dialogue.
6. **Evaluation framework:** Implement automated evaluation with ground-truth Q&A pairs to measure retrieval recall and answer accuracy.
7. **Observability:** Add structured logging, latency metrics, and tracing (e.g., LangSmith or OpenTelemetry) for debugging and monitoring.
