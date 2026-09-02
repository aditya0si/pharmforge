"""RAG ingestion and query interface."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pharmforge.data.loader import load_molecules
from pharmforge.rag.store import HybridStore, RetrievedDoc

_store: HybridStore | None = None

def get_store(persist_dir: Path | None = None) -> HybridStore:
    global _store
    if _store is not None and persist_dir is None:
        return _store
    s = HybridStore(persist_dir=persist_dir)
    if persist_dir is None:
        _store = s
    return s


def ingest(force: bool = False, persist_dir: Path | None = None) -> int:
    """Ingest molecules into HybridStore. Returns count."""
    store = get_store(persist_dir=persist_dir)
    if store.count() > 0 and not force:
        return store.count()
    if force:
        store.clear()
    mols = load_molecules()
    ids, texts, metas, embs = [], [], [], []
    from pharmforge.rag.store import embed_text
    for m in mols:
        doc = (
            f"ID: {m.id} | Name: {m.name} | SMILES: {m.smiles} | "
            f"Formula: {m.formula or ''} | MW: {m.mw} | "
            f"Description: {m.description} | Assay: {m.assay_notes or ''} | Tags: {','.join(m.tags)}"
        )
        ids.append(m.id)
        texts.append(doc)
        metas.append({"name": m.name, "smiles": m.smiles, "formula": m.formula or "", "mw": m.mw or 0, "tags": m.tags, "id": m.id})
    # Batch add (Chroma benefits)
    B = 200
    for i in range(0, len(ids), B):
        batch_ids = ids[i:i+B]
        batch_texts = texts[i:i+B]
        batch_metas = metas[i:i+B]
        batch_embs = [embed_text(t) for t in batch_texts]
        store.add(batch_ids, batch_texts, batch_metas, embeddings=batch_embs)
    return store.count()


def query_rag(query: str, k: int = 5, persist_dir: Path | None = None) -> List[RetrievedDoc]:
    store = get_store(persist_dir=persist_dir)
    if store.count() == 0:
        ingest(persist_dir=persist_dir)
    return store.hybrid_search(query, k=k)
