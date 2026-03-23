"""
FastAPI Service
───────────────
Provides:
  • GET  /        → Minimal web UI for querying
  • POST /query   → JSON API  { question } → { answer, citations }
  • GET  /health  → Health check
  • GET  /chunks  → API to list all stored chunks
  • GET  /viewer  → Web UI to browse FAISS chunks
"""

import pickle
from pathlib import Path
from typing import Optional

import faiss
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from config import FAISS_INDEX_DIR
from query_pipeline import query

app = FastAPI(
    title="RAG Pipeline — QA & Regulatory Affairs",
    description="FAISS-based RAG service for the biosciences QA textbook.",
    version="1.0.0",
)


# ── Health ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Query API ────────────────────────────────────────────────────────────────


@app.post("/query")
async def query_endpoint(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()
    if not question:
        return JSONResponse(
            status_code=400,
            content={"error": "Question is required."},
        )
    try:
        result = query(question)
        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "is_relevant": result["is_relevant"],
            "guardrail_reason": result["guardrail_reason"],
        }
    except FileNotFoundError as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ── Chunks API ───────────────────────────────────────────────────────────────


@app.get("/chunks")
async def list_chunks(page: Optional[int] = None, search: Optional[str] = None):
    """Return all chunks stored in the FAISS index with metadata."""
    metadata_path = FAISS_INDEX_DIR / "metadata.pkl"
    index_path = FAISS_INDEX_DIR / "index.faiss"

    if not metadata_path.exists() or not index_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "FAISS index not found. Run ingestion_pipeline.py first."},
        )

    with open(metadata_path, "rb") as fp:
        metadata = pickle.load(fp)

    index = faiss.read_index(str(index_path))

    chunks = []
    for i, m in enumerate(metadata):
        chunk = {
            "index": i,
            "chunk_id": m["chunk_id"],
            "page_number": m["page_number"],
            "text": m["text"],
            "text_length": len(m["text"]),
        }
        chunks.append(chunk)

    # Filter by page number
    if page is not None:
        chunks = [c for c in chunks if c["page_number"] == page]

    # Filter by search text
    if search:
        search_lower = search.lower()
        chunks = [c for c in chunks if search_lower in c["text"].lower()]

    # Gather stats
    all_pages = sorted(set(m["page_number"] for m in metadata))

    return {
        "total_chunks": len(metadata),
        "total_vectors": index.ntotal,
        "embedding_dim": index.d,
        "filtered_count": len(chunks),
        "pages": all_pages,
        "chunks": chunks,
    }


# ── Web UI ───────────────────────────────────────────────────────────────────

HTML_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>RAG Pipeline — QA &amp; Regulatory Affairs</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d2e;
    --surface-alt: #232740;
    --primary: #6c63ff;
    --primary-glow: rgba(108,99,255,.25);
    --accent: #00d4aa;
    --text: #e2e4f0;
    --text-muted: #9a9cb8;
    --border: #2e3250;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin:0; padding:0; }

  body {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  /* Header */
  .header {
    width: 100%;
    padding: 2rem 1.5rem 1.5rem;
    text-align: center;
    background: linear-gradient(180deg, rgba(108,99,255,.08) 0%, transparent 100%);
  }
  .header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .header p {
    color: var(--text-muted);
    margin-top: .4rem;
    font-size: .95rem;
  }

  /* Main */
  .container {
    width: 100%;
    max-width: 800px;
    padding: 0 1.5rem 3rem;
    flex: 1;
  }

  /* Query form */
  .query-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,.3);
  }
  .query-box label {
    font-weight: 600;
    font-size: .9rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: .05em;
    display: block;
    margin-bottom: .5rem;
  }
  .input-row {
    display: flex;
    gap: .75rem;
  }
  .query-box textarea {
    flex: 1;
    background: var(--surface-alt);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    font-family: var(--font);
    font-size: 1rem;
    padding: .75rem 1rem;
    resize: vertical;
    min-height: 56px;
    transition: border-color .2s;
  }
  .query-box textarea:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px var(--primary-glow);
  }
  .btn {
    background: linear-gradient(135deg, var(--primary), #8b5cf6);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: .75rem 1.5rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: transform .15s, box-shadow .2s;
    white-space: nowrap;
  }
  .btn:hover { transform: translateY(-1px); box-shadow: 0 4px 16px var(--primary-glow); }
  .btn:active { transform: translateY(0); }
  .btn:disabled { opacity: .5; cursor: not-allowed; transform: none; }

  /* Spinner */
  .spinner {
    display: none;
    text-align: center;
    padding: 2rem;
    color: var(--text-muted);
  }
  .spinner.active { display: block; }
  .spinner::before {
    content: '';
    display: inline-block;
    width: 28px; height: 28px;
    border: 3px solid var(--border);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin .7s linear infinite;
    vertical-align: middle;
    margin-right: .6rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Result */
  .result {
    display: none;
    margin-top: 1.5rem;
  }
  .result.active { display: block; }
  .result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,.3);
  }
  .result-card h2 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: .75rem;
    text-transform: uppercase;
    letter-spacing: .05em;
  }
  .answer {
    line-height: 1.7;
    white-space: pre-wrap;
    font-size: .97rem;
  }
  .citations {
    margin-top: 1.25rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
  }
  .citations h3 {
    font-size: .85rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: .05em;
    margin-bottom: .5rem;
  }
  .citation-pill {
    display: inline-block;
    background: var(--surface-alt);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: .3rem .7rem;
    font-size: .85rem;
    margin: .2rem .3rem .2rem 0;
    color: var(--primary);
    font-weight: 500;
  }
  .error-msg {
    color: #ff6b6b;
    background: rgba(255,107,107,.1);
    border: 1px solid rgba(255,107,107,.25);
    border-radius: 10px;
    padding: 1rem;
    margin-top: 1.5rem;
  }
</style>
</head>
<body>

<div class="header">
  <h1>📚 RAG Pipeline</h1>
  <p>Quality Assurance &amp; Regulatory Affairs for the Biosciences</p>
  <div style="margin-top:.75rem"><a href="/viewer" style="color:var(--primary);text-decoration:none;font-size:.9rem;font-weight:500;">🔍 View FAISS Chunks</a></div>
</div>

<div class="container">
  <div class="query-box">
    <label for="question">Ask a question</label>
    <div class="input-row">
      <textarea id="question" rows="2" placeholder="e.g. What is GMP and why is it important?"></textarea>
      <button class="btn" id="askBtn" onclick="askQuestion()">Ask</button>
    </div>
  </div>

  <div class="spinner" id="spinner">Searching and generating answer …</div>

  <div class="result" id="result">
    <div class="result-card">
      <div id="guardrailStatus" style="margin-bottom:.75rem"></div>
      <h2>Answer</h2>
      <div class="answer" id="answerText"></div>
      <div class="citations" id="citations"></div>
    </div>
  </div>

  <div id="errorBox"></div>
</div>

<script>
async function askQuestion() {
  const q = document.getElementById('question').value.trim();
  if (!q) return;

  const btn = document.getElementById('askBtn');
  const spinner = document.getElementById('spinner');
  const result = document.getElementById('result');
  const errorBox = document.getElementById('errorBox');

  btn.disabled = true;
  spinner.classList.add('active');
  result.classList.remove('active');
  errorBox.innerHTML = '';

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || 'Unknown error');
    }

    document.getElementById('answerText').textContent = data.answer;

    // Guardrail status
    const gsDiv = document.getElementById('guardrailStatus');
    if (data.is_relevant) {
      gsDiv.innerHTML = '<span style="display:inline-block;background:rgba(0,212,170,.15);color:#00d4aa;border:1px solid rgba(0,212,170,.3);border-radius:8px;padding:.3rem .8rem;font-size:.8rem;font-weight:600;">✅ Guardrail: Relevant</span>';
    } else {
      gsDiv.innerHTML = '<span style="display:inline-block;background:rgba(255,107,107,.12);color:#ff6b6b;border:1px solid rgba(255,107,107,.3);border-radius:8px;padding:.3rem .8rem;font-size:.8rem;font-weight:600;">🚫 Guardrail: Rejected</span>' +
        (data.guardrail_reason ? '<span style="color:#9a9cb8;font-size:.8rem;margin-left:.5rem;">' + data.guardrail_reason + '</span>' : '');
    }

    const citDiv = document.getElementById('citations');
    if (data.is_relevant && data.citations && data.citations.length > 0) {
      citDiv.innerHTML = '<h3>Source Pages</h3>';
      data.citations.forEach(c => {
        const pill = document.createElement('span');
        pill.className = 'citation-pill';
        pill.textContent = 'Page ' + c.page_number;
        citDiv.appendChild(pill);
      });
    } else {
      citDiv.innerHTML = '';
    }

    result.classList.add('active');
  } catch (err) {
    errorBox.innerHTML = '<div class="error-msg">⚠ ' + err.message + '</div>';
  } finally {
    btn.disabled = false;
    spinner.classList.remove('active');
  }
}

// Enter key support
document.getElementById('question').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    askQuestion();
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_PAGE


# ── Chunk Viewer UI ──────────────────────────────────────────────────────────

CHUNK_VIEWER_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>FAISS Chunk Viewer</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d2e;
    --surface-alt: #232740;
    --primary: #6c63ff;
    --primary-glow: rgba(108,99,255,.25);
    --accent: #00d4aa;
    --text: #e2e4f0;
    --text-muted: #9a9cb8;
    --border: #2e3250;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin:0; padding:0; }
  body {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  .header {
    width: 100%; padding: 2rem 1.5rem 1rem;
    text-align: center;
    background: linear-gradient(180deg, rgba(108,99,255,.08) 0%, transparent 100%);
  }
  .header h1 {
    font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  }
  .header a { color: var(--primary); text-decoration: none; font-size:.9rem; }

  .container { max-width: 1000px; margin: 0 auto; padding: 1rem 1.5rem 3rem; }

  /* Stats bar */
  .stats {
    display: flex; gap: 1rem; flex-wrap: wrap;
    margin-bottom: 1.25rem;
  }
  .stat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: .75rem 1.25rem; flex: 1; min-width: 140px;
    text-align: center;
  }
  .stat-card .val { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
  .stat-card .lbl { font-size: .75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing:.05em; margin-top:.2rem; }

  /* Filters */
  .filters {
    display: flex; gap: .75rem; flex-wrap: wrap; margin-bottom: 1.25rem;
  }
  .filters select, .filters input {
    background: var(--surface-alt); border: 1px solid var(--border);
    border-radius: 10px; color: var(--text); font-family: var(--font);
    font-size: .9rem; padding: .6rem 1rem;
    transition: border-color .2s;
  }
  .filters select:focus, .filters input:focus {
    outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-glow);
  }
  .filters input { flex: 1; min-width: 200px; }
  .result-count { color: var(--text-muted); font-size: .85rem; margin-bottom: 1rem; }

  /* Chunk cards */
  .chunk-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.25rem; margin-bottom: .75rem;
    transition: border-color .2s;
  }
  .chunk-card:hover { border-color: var(--primary); }
  .chunk-meta {
    display: flex; gap: .75rem; flex-wrap: wrap; margin-bottom: .6rem;
  }
  .meta-pill {
    display: inline-flex; align-items: center; gap: .3rem;
    background: var(--surface-alt); border: 1px solid var(--border);
    border-radius: 6px; padding: .2rem .6rem;
    font-size: .78rem; font-weight: 500;
  }
  .meta-pill.page { color: var(--primary); }
  .meta-pill.id { color: var(--accent); }
  .meta-pill.len { color: var(--text-muted); }
  .meta-pill.idx { color: #f59e0b; }
  .chunk-text {
    font-size: .88rem; line-height: 1.65; color: var(--text);
    white-space: pre-wrap; word-break: break-word;
    max-height: 200px; overflow-y: auto;
    background: var(--surface-alt); border-radius: 8px;
    padding: .75rem 1rem;
  }
  .chunk-text::-webkit-scrollbar { width: 6px; }
  .chunk-text::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  .expand-btn {
    background: none; border: none; color: var(--primary); cursor: pointer;
    font-size: .8rem; font-weight: 500; margin-top: .4rem; padding: .2rem 0;
  }
  .expand-btn:hover { text-decoration: underline; }

  .loading { text-align: center; padding: 3rem; color: var(--text-muted); }
  .error-msg {
    color: #ff6b6b; background: rgba(255,107,107,.1);
    border: 1px solid rgba(255,107,107,.25); border-radius: 10px;
    padding: 1rem; margin: 1rem 0;
  }
</style>
</head>
<body>

<div class="header">
  <h1>🔍 FAISS Chunk Viewer</h1>
  <div style="margin-top:.5rem"><a href="/">← Back to Query</a></div>
</div>

<div class="container">
  <div class="stats" id="stats"></div>

  <div class="filters">
    <select id="pageFilter">
      <option value="">All Pages</option>
    </select>
    <input type="text" id="searchFilter" placeholder="Search chunk text…" />
  </div>

  <div class="result-count" id="resultCount"></div>
  <div id="chunkList"><div class="loading">Loading chunks…</div></div>
  <div id="errorBox"></div>
</div>

<script>
let allData = null;
let debounceTimer = null;

async function loadChunks(page, search) {
  const params = new URLSearchParams();
  if (page) params.set('page', page);
  if (search) params.set('search', search);

  try {
    const res = await fetch('/chunks?' + params.toString());
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    return data;
  } catch (err) {
    document.getElementById('errorBox').innerHTML =
      '<div class="error-msg">⚠ ' + err.message + '</div>';
    return null;
  }
}

function renderStats(data) {
  document.getElementById('stats').innerHTML = `
    <div class="stat-card"><div class="val">${data.total_chunks}</div><div class="lbl">Total Chunks</div></div>
    <div class="stat-card"><div class="val">${data.total_vectors}</div><div class="lbl">Vectors</div></div>
    <div class="stat-card"><div class="val">${data.embedding_dim}</div><div class="lbl">Dimensions</div></div>
    <div class="stat-card"><div class="val">${data.pages.length}</div><div class="lbl">Pages</div></div>
  `;
}

function renderPageFilter(pages) {
  const sel = document.getElementById('pageFilter');
  sel.innerHTML = '<option value="">All Pages</option>';
  pages.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p; opt.textContent = 'Page ' + p;
    sel.appendChild(opt);
  });
}

function renderChunks(chunks) {
  const list = document.getElementById('chunkList');
  document.getElementById('resultCount').textContent =
    chunks.length + ' chunk' + (chunks.length !== 1 ? 's' : '') + ' shown';

  if (chunks.length === 0) {
    list.innerHTML = '<div class="loading">No chunks match your filters.</div>';
    return;
  }

  list.innerHTML = chunks.map(c => `
    <div class="chunk-card">
      <div class="chunk-meta">
        <span class="meta-pill idx">Index #${c.index}</span>
        <span class="meta-pill page">Page ${c.page_number}</span>
        <span class="meta-pill id">${c.chunk_id}</span>
        <span class="meta-pill len">${c.text_length} chars</span>
      </div>
      <div class="chunk-text" id="ct-${c.index}">${escapeHtml(c.text)}</div>
      <button class="expand-btn" onclick="toggleExpand(${c.index})">Expand</button>
    </div>
  `).join('');
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function toggleExpand(idx) {
  const el = document.getElementById('ct-' + idx);
  if (el.style.maxHeight === 'none') {
    el.style.maxHeight = '200px';
    el.nextElementSibling.textContent = 'Expand';
  } else {
    el.style.maxHeight = 'none';
    el.nextElementSibling.textContent = 'Collapse';
  }
}

async function refresh() {
  const page = document.getElementById('pageFilter').value;
  const search = document.getElementById('searchFilter').value.trim();
  const data = await loadChunks(page, search);
  if (!data) return;

  if (!allData) {
    allData = data;
    renderStats(data);
    renderPageFilter(data.pages);
  }
  renderChunks(data.chunks);
}

document.getElementById('pageFilter').addEventListener('change', refresh);
document.getElementById('searchFilter').addEventListener('input', () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(refresh, 300);
});

refresh();
</script>
</body>
</html>
"""


@app.get("/viewer", response_class=HTMLResponse)
async def chunk_viewer():
    return CHUNK_VIEWER_PAGE
