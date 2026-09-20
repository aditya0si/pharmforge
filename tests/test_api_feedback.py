"""Tests for feedback loop and API."""
from pathlib import Path

from fastapi.testclient import TestClient

from pharmforge.agents.dag import run_dag
from pharmforge.api.main import app
from pharmforge.feedback.export import export_high_quality, export_to_file
from pharmforge.feedback.store import count_traces, load_traces

client = TestClient(app)

def test_feedback_append_and_load():
    before = count_traces()
    run_dag("feedback test query aspirin", top_k=2)
    after = count_traces()
    assert after >= before + 1
    traces = load_traces(limit=5)
    assert len(traces) >= 1
    assert "_ts" in traces[-1]
    assert "_label" in traces[-1]

def test_export_high_quality():
    # Ensure at least one trace
    run_dag("export test imatinib analog", top_k=2)
    examples = export_high_quality(min_qed=0.0, require_passed=False)
    assert isinstance(examples, list)
    # Should have at least one
    assert len(examples) >= 1
    assert "query" in examples[0]

def test_export_to_file(tmp_path: Path = None):
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "export.jsonl"
    run_dag("export file test caffeine", top_k=2)
    p = export_to_file(tmp, min_qed=0.0)
    assert p.exists()
    assert p.read_text(encoding="utf-8").strip() != ""

def test_api_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_api_query():
    resp = client.post("/query", json={"query": "Find imatinib analogs", "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "trace" in data
    assert data["latency_ms"] > 0

def test_api_query_with_codegen():
    resp = client.post("/query", json={"query": "Generate script for aspirin", "top_k": 2, "include_codegen": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["trace"]["reporter"]["generated_code_path"] is not None

def test_api_molecule_get():
    resp = client.get("/molecule/PF0001")
    assert resp.status_code == 200
    assert resp.json()["molecule"]["id"] == "PF0001"

def test_api_molecule_not_found():
    resp = client.get("/molecule/DOESNOTEXIST")
    assert resp.status_code == 404

def test_api_molecules_list():
    resp = client.get("/molecules?limit=5")
    assert resp.status_code == 200
    assert resp.json()["total"] >= 5
    assert len(resp.json()["molecules"]) == 5

def test_api_molecules_search():
    resp = client.get("/molecules?q=imatinib")
    assert resp.status_code == 200
    assert any("imatinib" in m["name"].lower() for m in resp.json()["molecules"])

def test_api_chem_validate():
    resp = client.post("/chem/validate?smiles=CCO")
    assert resp.status_code == 200
    assert resp.json()["valid"] is True

def test_api_chem_properties():
    resp = client.post("/chem/properties?smiles=CC(=O)OC1=CC=CC=C1C(=O)O")
    assert resp.status_code == 200
    assert resp.json()["valid"] is True
    assert resp.json()["mw"] is not None

def test_api_chem_similarity():
    resp = client.post("/chem/similarity?smiles1=CCO&smiles2=CCO")
    assert resp.status_code == 200
    assert abs(resp.json()["tanimoto"] - 1.0) < 0.05

def test_api_rag_search():
    resp = client.get("/rag/search?q=kinase&k=3")
    assert resp.status_code == 200
    assert len(resp.json()["docs"]) >= 1

def test_api_codegen():
    resp = client.post("/codegen", json={"query": "analyze aspirin", "smiles": ["CCO"]})
    assert resp.status_code == 200
    assert "code" in resp.json()
    assert "validation" in resp.json()

def test_api_feedback_export():
    run_dag("api feedback export test", top_k=2)
    resp = client.get("/feedback/export?min_qed=0&require_passed=false")
    assert resp.status_code == 200
    assert "count" in resp.json()

def test_api_metrics():
    resp = client.get("/metrics")
    assert resp.status_code == 200

def test_mcp_tools():
    from pharmforge.mcp.server import handle_tool
    res = handle_tool("chem.validate", {"smiles": "CCO"})
    assert res["valid"] is True
    res2 = handle_tool("chem.properties", {"smiles": "CCO"})
    assert "mw" in res2 or "valid" in res2
    res3 = handle_tool("chem.search", {"query": "imatinib", "k": 2})
    assert isinstance(res3, list)
    assert len(res3) >= 1
    res4 = handle_tool("chem.similarity", {"smiles1": "CCO", "smiles2": "CCO"})
    assert "tanimoto" in res4

def test_api_query_validation():
    # query too short should 422
    resp = client.post("/query", json={"query": "ab"})
    assert resp.status_code == 422
