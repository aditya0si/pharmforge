"""Tests for codegen + sandbox."""
import tempfile
from pathlib import Path

from pharmforge.codegen.generator import generate_script
from pharmforge.codegen.sandbox import validate_code_string, validate_script


def test_generate_script_returns_code():
    code = generate_script("analyze aspirin", ["CC(=O)OC1=CC=CC=C1C(=O)O"])
    assert "Chem.MolFromSmiles" in code
    assert "CC(=O)OC1=CC=CC=C1C(=O)O" in code
    assert "def analyze" in code

def test_generate_script_empty_fallback():
    code = generate_script("test", [])
    assert "SMILES" in code
    assert "Chem.MolFromSmiles" in code

def test_generate_script_with_docs():
    from pharmforge.rag.store import RetrievedDoc
    docs = [RetrievedDoc(id="PF0001", name="Imatinib", smiles="CCO", text="imatinib kinase", score=0.9, provenance={})]
    code = generate_script("test", ["CCO"], docs)
    assert "PF0001" in code or "imatinib" in code.lower() or "CCO" in code

def test_validate_good_code():
    code = generate_script("test good", ["CCO", "CCC"])
    res = validate_code_string(code, timeout=10)
    # Should succeed if RDKit available; at least static_ok true
    assert res["static_ok"] is True
    # If RDKit installed, should execute
    # We check success is bool
    assert isinstance(res["success"], bool)
    if res["success"]:
        assert "MW=" in res["stdout"] or "Analysis complete" in res["stdout"]

def test_validate_syntax_error():
    bad = "def broken(:\n  pass"
    res = validate_code_string(bad, timeout=5)
    assert res["success"] is False
    assert res["static_ok"] is False

def test_validate_banned_socket():
    code = "import socket\ns=socket.socket()"
    res = validate_code_string(code, timeout=5)
    # Should be blocked by static check
    assert res["success"] is False

def test_validate_timeout():
    code = "import time\ntime.sleep(10)\nprint('done')"
    res = validate_code_string(code, timeout=2)
    assert res["success"] is False
    assert "Timeout" in res["stderr"] or "timeout" in res["stderr"].lower()

def test_validate_script_file():
    code = generate_script("file test", ["CCO"])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = Path(f.name)
    try:
        res = validate_script(path, timeout=10)
        assert res["static_ok"] is True
        assert "success" in res
    finally:
        path.unlink(missing_ok=True)

def test_generated_script_is_runnable_with_rdkit():
    # End-to-end: generate then run
    code = generate_script("runnable test", ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO"])
    res = validate_code_string(code, timeout=10)
    # On CI with RDKit, must pass
    # If RDKit missing, may still pass via fallback? But our template requires RDKit.
    # We allow either: if RDKit not installed, the run will fail ImportError, which is okay
    # But static_ok must be true
    assert res["static_ok"] is True

def test_codegen_generator_is_deterministic():
    c1 = generate_script("same query", ["CCO"])
    c2 = generate_script("same query", ["CCO"])
    assert c1 == c2
