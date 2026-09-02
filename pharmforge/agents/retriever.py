"""Retriever agent — wraps hybrid RAG."""
from __future__ import annotations

from pharmforge.agents.types import RetrieverOutput, PlannerOutput
from pharmforge.rag import query_rag

def retrieve(planner: PlannerOutput, top_k: int = 5) -> RetrieverOutput:
    # Build enriched query from planner
    q = planner.original_query
    if planner.keywords:
        q = q + " " + " ".join(planner.keywords[:5])
    if planner.target_smiles:
        q = q + " SMILES:" + planner.target_smiles[:60]
    docs = query_rag(q, k=top_k)
    note = f"hybrid RAG (vector+FTS+RRF) over {', '.join(planner.keywords[:3])}" if planner.keywords else "hybrid RAG"
    return RetrieverOutput(docs=docs, query=q, recall_note=note)
