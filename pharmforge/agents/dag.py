"""Agentic DAG orchestrator — planner → retriever → chemist → critic → reporter."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from pharmforge.agents.chemist import chemist as run_chemist
from pharmforge.agents.critic import criticize
from pharmforge.agents.planner import plan
from pharmforge.agents.reporter import report
from pharmforge.agents.retriever import retrieve
from pharmforge.agents.types import DagTrace

# Feedback
from pharmforge.feedback.store import append_trace
from pharmforge.observability.metrics import record_query
from pharmforge.observability.tracing import trace_span


def run_dag(query: str, top_k: int = 5, target_smiles: Optional[str] = None, include_codegen: bool = False) -> DagTrace:
    start = time.perf_counter()
    with trace_span("pharmforge.dag", attributes={"query": query[:100]}):
        with trace_span("planner"):
            planner_out = plan(query, target_smiles=target_smiles)
            # Honor explicit include_codegen
            if include_codegen:
                planner_out.needs_codegen = True
                if "Generate runnable analysis script" not in planner_out.subtasks:
                    planner_out.subtasks.append("Generate runnable analysis script")
        with trace_span("retriever"):
            retriever_out = retrieve(planner_out, top_k=top_k)
        with trace_span("chemist"):
            chemist_out = run_chemist(planner_out, retriever_out)
        with trace_span("critic"):
            critic_out = criticize(chemist_out, retriever_out)

        code_path = None
        code_ok = False
        if planner_out.needs_codegen or include_codegen:
            with trace_span("codegen"):
                from pharmforge.codegen.generator import generate_script
                from pharmforge.codegen.sandbox import validate_script
                try:
                    # Use first validated SMILES or first doc SMILES
                    smiles_list = chemist_out.raw_smiles_validated[:3] if chemist_out.raw_smiles_validated else []
                    if not smiles_list and retriever_out.docs:
                        smiles_list = [d.smiles for d in retriever_out.docs[:2] if d.smiles]
                    code = generate_script(query, smiles_list, retriever_out.docs[:2])
                    # Write to temp
                    tmp_dir = Path("data/generated")
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    # sanitize query for filename
                    safe = "".join(c if c.isalnum() else "_" for c in query[:30])
                    code_path_obj = tmp_dir / f"gen_{safe}.py"
                    code_path_obj.write_text(code, encoding="utf-8")
                    code_path = str(code_path_obj)
                    result = validate_script(code_path_obj)
                    code_ok = result["success"]
                except Exception as e:
                    code_path = str(e) if code_path is None else code_path
                    code_ok = False

        with trace_span("reporter"):
            reporter_out = report(planner_out, retriever_out, chemist_out, critic_out, code_path=code_path, code_ok=code_ok)

    latency_ms = (time.perf_counter() - start) * 1000
    trace = DagTrace(
        query=query,
        planner=planner_out,
        retriever=retriever_out,
        chemist=chemist_out,
        critic=critic_out,
        reporter=reporter_out,
        latency_ms=round(latency_ms, 2),
        total_docs=len(retriever_out.docs),
    )
    # Record metrics + feedback
    try:
        record_query(latency_ms, passed=critic_out.passed)
        append_trace(trace)
    except Exception:
        pass
    return trace
