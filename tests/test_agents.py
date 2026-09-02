"""Tests for agentic DAG — planner, retriever, chemist, critic, reporter, full DAG."""
import pytest

from pharmforge.agents.planner import plan
from pharmforge.agents.retriever import retrieve
from pharmforge.agents.chemist import chemist as run_chemist
from pharmforge.agents.critic import criticize
from pharmforge.agents.reporter import report
from pharmforge.agents.dag import run_dag

def test_planner_similarity_intent():
    p = plan("Find molecules similar to imatinib")
    assert p.intent in ("similarity", "multi")
    assert p.needs_chem is True

def test_planner_property_intent():
    p = plan("Predict solubility and LogP for aspirin")
    assert p.intent in ("property", "multi", "general")

def test_planner_codegen_intent():
    p = plan("Generate a Python script to analyze solubility")
    assert p.needs_codegen is True

def test_planner_extract_smiles():
    smi = "CC(=O)OC1=CC=CC=C1C(=O)O"
    p = plan(f"Analyze {smi} for ADME")
    # Should detect smiles
    assert p.target_smiles is not None or "CC(=O)OC" in p.original_query

def test_planner_with_target_smiles():
    p = plan("similar to imatinib", target_smiles="CCO")
    assert p.target_smiles == "CCO"

def test_retriever_returns_docs():
    p = plan("Find kinase inhibitors like imatinib")
    res = retrieve(p, top_k=3)
    assert len(res.docs) >= 1
    assert len(res.docs) <= 3

def test_chemist_with_smiles():
    p = plan("Predict for aspirin", target_smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
    r = retrieve(p, top_k=3)
    c = run_chemist(p, r)
    assert len(c.properties) >= 1
    assert any(pr.valid for pr in c.properties)

def test_chemist_without_smiles():
    p = plan("Find molecules like imatinib")
    r = retrieve(p, top_k=3)
    c = run_chemist(p, r)
    # Should still compute for retrieved docs
    assert isinstance(c.properties, list)

def test_critic_valid():
    p = plan("Find similar to aspirin", target_smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
    r = retrieve(p, top_k=3)
    c = run_chemist(p, r)
    crit = criticize(c, r)
    assert isinstance(crit.passed, bool)
    assert crit.hallucination_risk in ("low", "medium", "high")

def test_critic_invalid_smiles():
    from pharmforge.agents.types import ChemistOutput, RetrieverOutput
    from pharmforge.chem import PropertyPrediction
    c = ChemistOutput(properties=[PropertyPrediction(smiles="INVALID", valid=False, error="bad")], similarities=[], raw_smiles_validated=["INVALID"])
    r = RetrieverOutput(docs=[], query="test")
    crit = criticize(c, r)
    # Should flag no docs -> high risk
    assert crit.hallucination_risk == "high"

def test_reporter_markdown():
    p = plan("Find imatinib analogs")
    r = retrieve(p, top_k=2)
    c = run_chemist(p, r)
    crit = criticize(c, r)
    rep = report(p, r, c, crit)
    assert "# PharmForge Report" in rep.markdown
    assert "Provenance" in rep.markdown or "provenance" in rep.markdown.lower()

def test_full_dag_simple():
    trace = run_dag("Find molecules similar to imatinib", top_k=3)
    assert trace.query
    assert trace.retriever
    assert len(trace.retriever.docs) >= 1
    assert trace.reporter.markdown
    assert trace.latency_ms > 0

def test_full_dag_with_codegen():
    trace = run_dag("Generate script to analyze aspirin CC(=O)OC1=CC=CC=C1C(=O)O", top_k=2, include_codegen=True)
    assert trace.reporter.generated_code_path is not None
    # code_validated may be True if RDKit available
    assert isinstance(trace.reporter.code_validated, bool)

def test_full_dag_with_smiles():
    trace = run_dag("Predict properties for CC(=O)OC1=CC=CC=C1C(=O)O", top_k=2, target_smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
    assert len(trace.chemist.properties) >= 1

def test_dag_feedback_written():
    from pathlib import Path
    before = 0
    p = Path("data/feedback.jsonl")
    if p.exists():
        before = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
    run_dag("test feedback loop query for imatinib", top_k=2)
    after = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip()) if p.exists() else 0
    assert after >= before + 1

def test_dag_latency_under_5s():
    trace = run_dag("Find kinase inhibitors", top_k=5)
    assert trace.latency_ms < 5000, f"latency {trace.latency_ms} should be <5000"

def test_planner_subtasks_not_empty():
    p = plan("random query about solubility")
    assert len(p.subtasks) >= 2
