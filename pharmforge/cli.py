"""CLI entry points."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def cmd_query(args):
    from pharmforge.agents.dag import run_dag
    trace = run_dag(args.query, top_k=args.top_k, target_smiles=args.smiles, include_codegen=args.codegen)
    print(trace.reporter.markdown)
    if args.json:
        print("\n--- TRACE JSON ---")
        print(trace.model_dump_json(indent=2))

def cmd_ingest(args):
    from pharmforge.rag import ingest
    n = ingest(force=args.force)
    print(f"Ingested {n} molecules")

def cmd_serve(args):
    import uvicorn
    uvicorn.run("pharmforge.api.main:app", host=args.host, port=args.port, reload=False)

def cmd_mcp(args):
    from pharmforge.mcp.server import run_stdio
    run_stdio()

def cmd_feedback_export(args):
    from pharmforge.feedback.export import export_to_file
    p = export_to_file(args.output, min_qed=args.min_qed)
    print(f"Exported to {p}")
    print(p.read_text(encoding="utf-8")[:2000])

def cmd_eval(args):
    # delegate to scripts/eval.py
    from scripts.eval import main as eval_main
    eval_main()

def main():
    parser = argparse.ArgumentParser(prog="pharmforge", description="PharmForge — Scientific Agentic Platform")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("query", help="Run agentic DAG query")
    p.add_argument("query", help="Natural language query")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--smiles", default=None)
    p.add_argument("--codegen", action="store_true", help="Include code generation")
    p.add_argument("--json", action="store_true", help="Dump trace JSON")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("ingest", help="Ingest molecules into RAG")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("serve", help="Run FastAPI server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("mcp", help="Run MCP server over stdio")
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser("feedback-export", help="Export feedback training data")
    p.add_argument("--output", default="data/training_export.jsonl")
    p.add_argument("--min-qed", type=float, default=0.0)
    p.set_defaults(func=cmd_feedback_export)

    p = sub.add_parser("eval", help="Run eval harness")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
