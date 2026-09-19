"""FastAPI — POST /query, POST /codegen, GET /molecule/{id}, GET /health, GET /metrics"""
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from pharmforge.agents.dag import run_dag
from pharmforge.agents.types import QueryRequest
from pharmforge.chem import (
    fingerprint_similarity,
    generate_conformers,
    predict_properties,
    validate_smiles,
)
from pharmforge.codegen.generator import generate_script
from pharmforge.codegen.sandbox import validate_code_string
from pharmforge.data.loader import load_molecules
from pharmforge.feedback.export import export_high_quality
from pharmforge.observability.metrics import PROM_AVAILABLE
from pharmforge.rag import ingest, query_rag

app = FastAPI(title="PharmForge", version="0.1.0", description="Scientific Agentic Integration Platform")

# Ensure ingestion on startup
@app.on_event("startup")
def _startup():
    try:
        ingest()
    except Exception as e:
        print(f"Ingest failed on startup: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0", "rdkit_available": True}

@app.get("/metrics")
def metrics():
    if PROM_AVAILABLE:
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
            return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)
        except Exception:
            pass
    return PlainTextResponse("# metrics unavailable\n")

@app.post("/query")
def query(req: QueryRequest):
    trace = run_dag(req.query, top_k=req.top_k, target_smiles=req.target_smiles, include_codegen=req.include_codegen)
    return {
        "query": trace.query,
        "answer": trace.reporter.markdown,
        "provenance": trace.reporter.provenance,
        "latency_ms": trace.latency_ms,
        "critic_passed": trace.critic.passed,
        "hallucination_risk": trace.critic.hallucination_risk,
        "trace": trace.model_dump(),
    }

class CodegenRequest(BaseModel):
    query: str
    smiles: List[str] = []

@app.post("/codegen")
def codegen(req: CodegenRequest):
    code = generate_script(req.query, req.smiles)
    result = validate_code_string(code)
    return {"code": code, "validation": result}

@app.get("/molecule/{mol_id}")
def get_molecule(mol_id: str):
    mols = load_molecules()
    for m in mols:
        if m.id == mol_id:
            props = predict_properties(m.smiles)
            return {"molecule": m.model_dump(), "properties": props.model_dump()}
    raise HTTPException(status_code=404, detail="Molecule not found")

@app.get("/molecules")
def list_molecules(limit: int = 20, offset: int = 0, q: Optional[str] = None):
    mols = load_molecules()
    if q:
        ql = q.lower()
        mols = [m for m in mols if ql in m.name.lower() or ql in m.description.lower() or ql in m.smiles.lower() or ql in m.id.lower()]
    total = len(mols)
    sliced = mols[offset: offset+limit]
    return {"total": total, "molecules": [m.model_dump() for m in sliced]}

@app.post("/chem/validate")
def chem_validate(smiles: str):
    ok, err = validate_smiles(smiles)
    return {"smiles": smiles, "valid": ok, "error": err}

@app.post("/chem/properties")
def chem_properties(smiles: str):
    return predict_properties(smiles).model_dump()

@app.post("/chem/similarity")
def chem_similarity(smiles1: str, smiles2: str):
    return {"smiles1": smiles1, "smiles2": smiles2, "tanimoto": fingerprint_similarity(smiles1, smiles2)}

@app.post("/chem/conformers")
def chem_conformers(smiles: str, num_confs: int = 5):
    return generate_conformers(smiles, num_confs=num_confs)

@app.get("/rag/search")
def rag_search(q: str, k: int = 5):
    docs = query_rag(q, k=k)
    return {"query": q, "docs": [d.model_dump() for d in docs]}

@app.get("/feedback/export")
def feedback_export(min_qed: float = 0.0, require_passed: bool = False):
    examples = export_high_quality(min_qed=min_qed, require_passed=require_passed)
    return {"count": len(examples), "examples": examples[:20]}
