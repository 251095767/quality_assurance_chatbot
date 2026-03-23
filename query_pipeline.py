"""
Query Pipeline (LangGraph)
──────────────────────────
A node-based workflow using LangGraph with three stages:

  1. Guardrail Node  → LLM checks if the question is relevant to QA/Regulatory Affairs
  2. Retrieval Node  → Embed query → FAISS search → top-k chunks
  3. Answer Node     → LLM generates a cited answer from retrieved context

If the guardrail rejects the question, the pipeline short-circuits and returns
a polite refusal without hitting FAISS or the answer LLM.

Can be used as a library  (``query(question)``)  or run from the CLI.
"""

import operator
import pickle
import sys
from typing import Annotated, Any, Literal, TypedDict

import faiss
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    LLM_MODEL,
    OPENAI_API_KEY,
    TOP_K,
)
from openai import OpenAI

_oai_client = OpenAI(api_key=OPENAI_API_KEY)

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.1,
    max_tokens=1024,
)

guardrail_llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.0,
    max_tokens=50,
)






def _embed_query(text: str) -> np.ndarray:
    """Embed a single query string."""
    response = _oai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    return np.array([response.data[0].embedding], dtype="float32")




_index_cache: dict = {}


def _load_index():
    if "index" not in _index_cache:
        index_path = FAISS_INDEX_DIR / "index.faiss"
        metadata_path = FAISS_INDEX_DIR / "metadata.pkl"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "FAISS index not found. Run ingestion_pipeline.py first."
            )

        _index_cache["index"] = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as fp:
            _index_cache["metadata"] = pickle.load(fp)

        print(
            f"📦  Loaded FAISS index ({_index_cache['index'].ntotal} vectors) "
            f"and metadata ({len(_index_cache['metadata'])} chunks)"
        )

    return _index_cache["index"], _index_cache["metadata"]




class PipelineState(TypedDict):
    """Shared state passed through the graph nodes."""
    question: str
    is_relevant: bool
    guardrail_reason: str
    retrieved_chunks: list[dict]
    answer: str
    citations: list[dict]


GUARDRAIL_SYSTEM_PROMPT = """\
You are a strict relevance classifier for a Quality Assurance (QA) and \
Regulatory Affairs knowledge base for the biosciences / pharmaceutical industry.

Your job is to decide whether a user question is RELEVANT or IRRELEVANT to \
the following topics:
  • Quality Assurance (QA), Quality Control (QC)
  • Good Manufacturing Practice (GMP), Good Laboratory Practice (GLP), \
Good Clinical Practice (GCP)
  • Regulatory affairs, FDA, EMA, ICH guidelines
  • Pharmaceutical / biotech manufacturing, validation, compliance
  • Clinical trials, drug development, biologics, medical devices
  • Standard Operating Procedures (SOPs), CAPA, audits, inspections
  • Any bioscience or life-science regulatory topic

Rules:
  • If the question is clearly about one of these topics → respond RELEVANT
  • If the question is ambiguous but could plausibly relate → respond RELEVANT
  • If the question is completely unrelated (e.g. cooking, sports, coding) → respond IRRELEVANT

Respond with EXACTLY one word: RELEVANT or IRRELEVANT
Then on the next line, a very brief reason (≤ 15 words).
"""


def guardrail_node(state: PipelineState) -> dict:
    """
    Guardrail Node — calls the LLM to classify whether the user's question
    is relevant to the QA / Regulatory Affairs domain.
    """
    question = state["question"]

    messages = [
        SystemMessage(content=GUARDRAIL_SYSTEM_PROMPT),
        HumanMessage(content=f"User question: {question}"),
    ]

    response = guardrail_llm.invoke(messages)
    reply = response.content.strip()

    # Parse response
    lines = reply.split("\n", 1)
    verdict = lines[0].strip().upper()
    reason = lines[1].strip() if len(lines) > 1 else ""

    is_relevant = verdict == "RELEVANT"

    return {
        "is_relevant": is_relevant,
        "guardrail_reason": reason,
    }




def retrieval_node(state: PipelineState) -> dict:
    """
    Retrieval Node — embeds the query and searches FAISS for top-k chunks.
    """
    question = state["question"]

    # Embed query
    query_emb = _embed_query(question)

    # Search FAISS
    index, metadata = _load_index()
    distances, indices = index.search(query_emb, TOP_K)

    chunks = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = metadata[idx].copy()
        chunk["score"] = float(dist)
        chunks.append(chunk)

    return {"retrieved_chunks": chunks}



ANSWER_SYSTEM_PROMPT = """\
You are a helpful expert assistant on Quality Assurance and Regulatory Affairs \
for the biosciences industry. Answer the user's question using ONLY the context \
provided below. If the context does not contain enough information to answer, \
say so clearly.

IMPORTANT:
• Cite page numbers for every claim using the format [Page X].
• If information comes from multiple pages, cite all relevant pages.
• Structure your answer clearly with paragraphs or bullet points where appropriate.
"""


def answer_formulation_node(state: PipelineState) -> dict:
    """
    Answer Formulation Node — builds a prompt from retrieved chunks and
    calls GPT-4o-mini to generate a cited answer.
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]

    # Build context block
    context_parts = []
    for c in chunks:
        context_parts.append(
            f"--- Page {c['page_number']} (chunk {c['chunk_id']}) ---\n{c['text']}"
        )
    context_block = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Context:\n{context_block}\n\n"
                f"Question: {question}\n\n"
                "Please provide a detailed answer with page-number citations."
            )
        ),
    ]

    response = llm.invoke(messages)
    answer = response.content.strip()

    # Compile citations (deduplicated page numbers, ordered)
    seen_pages = set()
    citations = []
    for c in chunks:
        if c["page_number"] not in seen_pages:
            seen_pages.add(c["page_number"])
            citations.append(
                {
                    "page_number": c["page_number"],
                    "chunk_id": c["chunk_id"],
                    "score": c["score"],
                }
            )

    return {
        "answer": answer,
        "citations": citations,
    }



def rejected_node(state: PipelineState) -> dict:
    """Return a polite refusal when the guardrail rejects the question."""
    reason = state.get("guardrail_reason", "")
    return {
        "answer": (
            "I'm sorry, but your question doesn't appear to be related to "
            "Quality Assurance or Regulatory Affairs for the biosciences. "
            "I can only answer questions within that domain.\n\n"
            f"Reason: {reason}" if reason else
            "I'm sorry, but your question doesn't appear to be related to "
            "Quality Assurance or Regulatory Affairs for the biosciences. "
            "I can only answer questions within that domain."
        ),
        "citations": [],
        "retrieved_chunks": [],
    }



def route_after_guardrail(state: PipelineState) -> Literal["retrieval", "rejected"]:
    """Route to retrieval if relevant, else to rejected."""
    if state.get("is_relevant", False):
        return "retrieval"
    return "rejected"



def build_graph():
    """Construct and compile the LangGraph pipeline."""
    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("answer_formulation", answer_formulation_node)
    workflow.add_node("rejected", rejected_node)

    # Set entry point
    workflow.set_entry_point("guardrail")

    # Conditional edge: guardrail → retrieval  OR  guardrail → rejected
    workflow.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "retrieval": "retrieval",
            "rejected": "rejected",
        },
    )

    # Linear edges
    workflow.add_edge("retrieval", "answer_formulation")
    workflow.add_edge("answer_formulation", END)
    workflow.add_edge("rejected", END)

    return workflow.compile()


# Compile once at module level
_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph




def query(question: str) -> dict:
    """
    End-to-end RAG query using LangGraph pipeline.

    Flow:
        question → guardrail → (relevant?) → retrieval → answer_formulation → result
                                (irrelevant?) → rejected → result

    Returns:
        {
            "answer": str,
            "citations": [...],
            "is_relevant": bool,
            "guardrail_reason": str,
            "context_chunks": [...]
        }
    """
    graph = _get_graph()

    initial_state: PipelineState = {
        "question": question,
        "is_relevant": False,
        "guardrail_reason": "",
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
    }

    result = graph.invoke(initial_state)

    return {
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "is_relevant": result.get("is_relevant", False),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "context_chunks": result.get("retrieved_chunks", []),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python query_pipeline.py "Your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"❓  Question: {question}\n")

    result = query(question)

    if not result["is_relevant"]:
        print(f"    Reason: {result['guardrail_reason']}")
        print(f"\n{result['answer']}")
        sys.exit(0)
    print(result["answer"])
    for c in result["citations"]:
        print(f"    • Page {c['page_number']}  (chunk {c['chunk_id']}, L2 dist: {c['score']:.4f})")
