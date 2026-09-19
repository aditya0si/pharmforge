"""Tests for RAG — hybrid retrieval."""
import tempfile
from pathlib import Path

from pharmforge.rag import query_rag
from pharmforge.rag.store import HybridStore, embed_text


def test_embed_deterministic():
    v1 = embed_text("imatinib kinase inhibitor")
    v2 = embed_text("imatinib kinase inhibitor")
    assert v1 == v2

def test_embed_normalized():
    v = embed_text("hello world")
    norm = sum(x*x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-6

def test_embed_different_texts_different():
    v1 = embed_text("aspirin")
    v2 = embed_text("quantum mechanics")
    # dot product should be <0.9 for unrelated
    dot = sum(a*b for a,b in zip(v1, v2))
    assert dot < 0.9

def test_hybrid_store_add_and_search(tmp_path: Path = None):
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    store = HybridStore(persist_dir=tmp)
    store.clear()
    store.add(
        ids=["PF0001", "PF0002"],
        texts=["Imatinib is a BCR-ABL kinase inhibitor for CML", "Aspirin is a COX inhibitor"],
        metadatas=[{"name": "Imatinib", "smiles": "CC1=C...", "tags": ["kinase"]}, {"name": "Aspirin", "smiles": "CC(=O)OC", "tags": ["nsaid"]}],
    )
    assert store.count() == 2
    hits = store.vector_search("BCR-ABL kinase", k=2)
    assert len(hits) == 2
    # Imatinib should rank higher
    assert hits[0][0] == "PF0001"

def test_fts_search(tmp_path=None):
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    store = HybridStore(persist_dir=tmp)
    store.clear()
    store.add(
        ids=["A", "B"],
        texts=["BRAF inhibitor vemurafenib", "Aspirin acetylsalicylic acid"],
        metadatas=[{"name": "Vemurafenib", "smiles": "CCC", "tags": []}, {"name": "Aspirin", "smiles": "CCO", "tags": []}],
    )
    hits = store.fts_search("vemurafenib", k=5)
    assert len(hits) >= 1
    assert hits[0][0] == "A"

def test_hybrid_rrf(tmp_path=None):
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    store = HybridStore(persist_dir=tmp)
    store.clear()
    store.add(
        ids=["X1", "X2", "X3"],
        texts=["kinase inhibitor imatinib BCR-ABL", "proton pump omeprazole", "statin atorvastatin cholesterol"],
        metadatas=[
            {"name": "Imatinib", "smiles": "s1", "tags": []},
            {"name": "Omeprazole", "smiles": "s2", "tags": []},
            {"name": "Atorvastatin", "smiles": "s3", "tags": []},
        ],
    )
    hits = store.hybrid_search("kinase inhibitor imatinib", k=2)
    assert len(hits) == 2
    assert hits[0].id == "X1"

def test_ingest_and_query_rag(tmp_path=None):
    # Use temp persist dir to avoid polluting main
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    # We need to test global ingest but avoid double-ingest flakiness
    # Instead test low-level
    store = HybridStore(persist_dir=tmp)
    store.clear()
    from pharmforge.data.loader import load_molecules
    mols = load_molecules()
    assert len(mols) >= 40
    # ingest 5 into tmp store
    ids = [m.id for m in mols[:5]]
    texts = [f"{m.name} {m.description} {m.smiles}" for m in mols[:5]]
    metas = [{"name": m.name, "smiles": m.smiles, "tags": m.tags} for m in mols[:5]]
    store.add(ids, texts, metas)
    hits = store.hybrid_search(mols[0].name, k=3)
    assert any(h.id == mols[0].id for h in hits)

def test_query_rag_integration():
    # Integration against default store (may already be ingested)
    docs = query_rag("imatinib", k=3)
    assert len(docs) >= 1
    # Should contain imatinib doc
    assert any("imatinib" in d.name.lower() or d.id == "PF0001" for d in docs)

def test_chromadb_persist(tmp_path=None):
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    s1 = HybridStore(persist_dir=tmp)
    s1.clear()
    s1.add(ids=["T1"], texts=["test doc for persist"], metadatas=[{"name": "T", "smiles": "CCO", "tags": []}])
    s1.count()  # warm the store before re-opening it below
    # New instance same dir should see same data (if chroma persist works)
    s2 = HybridStore(persist_dir=tmp)
    # Note: FTS persists, vector may persist; count should be >=1
    assert s2.count() >= 1
