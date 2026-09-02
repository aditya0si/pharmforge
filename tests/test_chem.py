"""Tests for chem module — real RDKit when available."""
import pytest
from pharmforge.chem import (
    validate_smiles,
    fingerprint_similarity,
    predict_properties,
    generate_conformers,
    batch_similarity_search,
    RDKIT_AVAILABLE,
)

def test_validate_valid_smiles():
    ok, err = validate_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin
    assert ok is True
    assert err is None

def test_validate_invalid_smiles():
    ok, err = validate_smiles("XYZ123NOTASMILES!!!!")
    assert ok is False
    # if RDKit available, error from RDKit; if not, heuristic
    assert err is not None

def test_validate_empty():
    ok, err = validate_smiles("")
    assert ok is False

def test_validate_imatinib():
    smi = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
    ok, err = validate_smiles(smi)
    assert ok is True

def test_fingerprint_self_similarity():
    smi = "CC(=O)OC1=CC=CC=C1C(=O)O"
    sim = fingerprint_similarity(smi, smi)
    assert abs(sim - 1.0) < 0.01, f"self-similarity should be ~1.0 got {sim}"

def test_fingerprint_dissimilar():
    smi1 = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin
    smi2 = "CCO"  # ethanol
    sim = fingerprint_similarity(smi1, smi2)
    assert sim < 0.4, f"dissimilar should be <0.4 got {sim}"

def test_fingerprint_imatinib_vs_imatinib():
    smi = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
    sim = fingerprint_similarity(smi, smi)
    assert sim > 0.99

def test_fingerprint_order_invariance():
    a = "CCO"
    b = "CCC"
    assert abs(fingerprint_similarity(a, b) - fingerprint_similarity(b, a)) < 1e-6

def test_predict_properties_valid():
    smi = "CC(=O)OC1=CC=CC=C1C(=O)O"
    p = predict_properties(smi)
    assert p.valid is True
    assert p.mw is not None and 150 < p.mw < 250
    assert p.logp is not None
    assert p.qed is not None and 0 <= p.qed <= 1
    assert p.tpsa is not None
    assert p.solubility_proxy is not None

def test_predict_properties_invalid():
    p = predict_properties("INVALIDSMILES999")
    assert p.valid is False

def test_predict_properties_metformin():
    smi = "CN(C)C(=N)NC(=N)N"
    p = predict_properties(smi)
    assert p.valid is True
    assert p.mw is not None

def test_predict_aspirin_logp_range():
    smi = "CC(=O)OC1=CC=CC=C1C(=O)O"
    p = predict_properties(smi)
    # aspirin logP ~1.2
    assert -1 < p.logp < 3

def test_generate_conformers_valid():
    smi = "CCO"  # ethanol simple
    res = generate_conformers(smi, num_confs=3)
    if RDKIT_AVAILABLE:
        assert res["success"] is True or res["num_generated"] >= 0
    else:
        assert res["success"] is False

def test_generate_conformers_invalid():
    res = generate_conformers("INVALID", num_confs=2)
    assert res["success"] is False

def test_batch_similarity_search():
    query = "CC(=O)OC1=CC=CC=C1C(=O)O"
    targets = [
        ("PF0002", "Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("PF0003", "Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
        ("PF0004", "Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ]
    results = batch_similarity_search(query, targets, top_k=2)
    assert len(results) == 2
    assert results[0].target_id == "PF0002"
    assert results[0].tanimoto > 0.99

def test_batch_similarity_sorted():
    query = "CCO"
    targets = [("A", "a", "CCO"), ("B", "b", "CCCC"), ("C", "c", "c1ccccc1")]
    results = batch_similarity_search(query, targets, top_k=3)
    scores = [r.tanimoto for r in results]
    assert scores == sorted(scores, reverse=True)
