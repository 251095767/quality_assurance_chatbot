# RAG Pipeline — Quality Assurance & Regulatory Affairs

A minimal **Retrieval-Augmented Generation (RAG)** service that ingests a biosciences QA textbook PDF, builds a FAISS vector index, and answers user questions with page-number citations.

---

## Architecture Overview

```
PDF  ──►  Extraction Pipeline  ──►  Ingestion Pipeline  ──►  FAISS Index
                                                                 │
                                          User Query  ──►  Query Pipeline
                                                              │
                                                         GPT-4o-mini  ──►  Answer + Citations
```

| Component | File | Purpose |
|---|---|---|
| Configuration | `config.py` | Paths, model names, chunking params |
| Extraction | `extraction_pipeline.py` | PDF → text + image captions (per page JSON) |
| Ingestion | `ingestion_pipeline.py` | Chunks → embeddings → FAISS index |
| Query | `query_pipeline.py` | Embed query → retrieve → LLM → answer |
| API / UI | `app.py` | FastAPI server with web interface |

---

## Prerequisites

- **Python 3.10+**
- **OpenAI API key** with access to `gpt-4o-mini` and `text-embedding-3-small`

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root (or export the variable):

```
OPENAI_API_KEY=sk-your-api-key-here
```

### 4. Place the PDF

Ensure the PDF file is in the project root:

```
new_project/
  └── QUALITY ASSURANCE AND REGULATORY AFFAIRS FOR THE BIOSCIENCES.pdf
```

---

## Running the Pipelines

### Step 1 — Extract (PDF → JSON)

```bash
python extraction_pipeline.py
```

This parses every page from the PDF, extracts text, and generates captions for images using GPT-4o-mini vision. Output is saved as individual JSON files in `data/extracted/`.

### Step 2 — Ingest (JSON → FAISS Index)

```bash
python ingestion_pipeline.py
```

This reads the extracted JSON, chunks the text (~2000 character chunks with 200 char overlap), embeds each chunk with `text-embedding-3-small`, and builds a FAISS `IndexFlatL2` index. Output is saved to `data/faiss_index/`.

### Step 3 — Query (start the service)

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser to use the web UI, or call the API directly:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is GMP?"}'
```

You can also query from the CLI without starting the server:

```bash
python query_pipeline.py "What is GMP?"
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/query` | Query the RAG pipeline |
| `GET` | `/health` | Health check |

### POST /query

**Request:**
```json
{
  "question": "What are the key principles of GMP?"
}
```

**Response:**
```json
{
  "answer": "GMP (Good Manufacturing Practice) ...",
  "citations": [
    {"page_number": 12, "chunk_id": "p12_c0", "score": 0.42},
    {"page_number": 15, "chunk_id": "p15_c1", "score": 0.58}
  ]
}
```

---

## Project Structure

```
new_project/
├── app.py                   # FastAPI service + web UI
├── config.py                # Centralised configuration
├── extraction_pipeline.py   # PDF parsing + image captioning
├── ingestion_pipeline.py    # Chunking + embedding + FAISS indexing
├── query_pipeline.py        # RAG retrieval + LLM generation
├── requirements.txt         # Python dependencies
├── design_doc.md            # Design document (2-4 pages)
├── .env                     # API key (not committed)
├── .gitignore               # Git ignore rules
└── data/                    # Generated at runtime
    ├── extracted/            # Per-page JSON files
    └── faiss_index/          # FAISS index + metadata
```

---

## Notes

- The PDF file is **not** included in the repository (excluded via `.gitignore`).
- The `data/` directory is generated at runtime and also excluded from version control.
- The FAISS index uses `IndexFlatL2` (exact search) which is sufficient for ~100 pages.
