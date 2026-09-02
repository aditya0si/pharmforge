"""Hybrid RAG: Chroma vector store + FTS (sqlite) + RRF."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import math
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


# --- Lightweight embedding fallback (hashing trick) ---
def embed_text(text: str, dim: int = 384) -> List[float]:
    """Deterministic hashing embedding — no model download required.
    Produces L2-normalized vector. Fast, offline, better than random.
    """
    vec = [0.0] * dim
    # Tokenize
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    if not tokens:
        tokens = [text.lower()]
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        for i in range(3):  # 3 hashes per token for denser
            idx = (h >> (i * 12)) % dim
            sign = 1 if ((h >> (i * 8)) & 1) else -1
            vec[idx] += sign * (1.0 + (h % 7) / 7)
    # L2 normalize
    norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x / norm for x in vec]


class RetrievedDoc(BaseModel):
    id: str
    name: str
    smiles: str
    text: str
    score: float
    provenance: dict
    source: str = "vector"  # vector | fts | hybrid


class HybridStore:
    """Hybrid store: Chroma (vector) + SQLite FTS5 (keyword) + RRF."""

    def __init__(self, persist_dir: Optional[Path] = None, collection_name: str = "pharmforge"):
        self.persist_dir = persist_dir or Path(__file__).parent / ".chroma"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._fts_path = self.persist_dir / "fts.db"
        self._init_fts()

        if CHROMA_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(path=str(self.persist_dir), settings=Settings(anonymized_telemetry=False))
                self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
            except Exception as e:
                print(f"[HybridStore] Chroma init failed: {e} — using fallback")
                self.client = None
                self.collection = None
        else:
            self.client = None
            self.collection = None

        # In-memory fallback store when Chroma unavailable
        self._mem_docs: List[dict] = []
        self._mem_embs: List[List[float]] = []

    def _init_fts(self):
        self._fts_conn = sqlite3.connect(str(self._fts_path), check_same_thread=False)
        self._fts_conn.execute("PRAGMA journal_mode=WAL;")
        self._fts_conn.execute("""
            CREATE TABLE IF NOT EXISTS docs (
                id TEXT PRIMARY KEY,
                name TEXT,
                smiles TEXT,
                text TEXT,
                tags TEXT
            )
        """)
        # FTS5 virtual table
        try:
            self._fts_conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                    id, name, text, tags, content='docs', content_rowid='rowid'
                )
            """)
            # Triggers
            self._fts_conn.executescript("""
                CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
                    INSERT INTO docs_fts(rowid, id, name, text, tags) VALUES (new.rowid, new.id, new.name, new.text, new.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
                    INSERT INTO docs_fts(docs_fts, rowid, id, name, text, tags) VALUES('delete', old.rowid, old.id, old.name, old.text, old.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON docs BEGIN
                    INSERT INTO docs_fts(docs_fts, rowid, id, name, text, tags) VALUES('delete', old.rowid, old.id, old.name, old.text, old.tags);
                    INSERT INTO docs_fts(rowid, id, name, text, tags) VALUES (new.rowid, new.id, new.name, new.text, new.tags);
                END;
            """)
        except sqlite3.OperationalError as e:
            # FTS5 not available
            print(f"[HybridStore] FTS5 unavailable: {e}")
        self._fts_conn.commit()

    def clear(self):
        if self.collection is not None:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
            except Exception:
                pass
        self._fts_conn.execute("DELETE FROM docs")
        self._fts_conn.execute("DELETE FROM docs_fts")
        self._fts_conn.commit()
        self._mem_docs = []
        self._mem_embs = []

    def _sanitize_meta_for_chroma(self, meta: dict) -> dict:
        out = {}
        for k, v in meta.items():
            if isinstance(v, list):
                if not v:
                    out[k] = ""
                else:
                    # join or stringify
                    out[k] = ",".join(str(x) for x in v)
            elif v is None:
                out[k] = ""
            else:
                out[k] = v
        return out

    def add(self, ids: List[str], texts: List[str], metadatas: List[dict], embeddings: Optional[List[List[float]]] = None):
        if embeddings is None:
            embeddings = [embed_text(t) for t in texts]
        # Chroma
        if self.collection is not None:
            try:
                sanitized = [self._sanitize_meta_for_chroma(m) for m in metadatas]
                self.collection.add(ids=ids, documents=texts, metadatas=sanitized, embeddings=embeddings)
            except Exception as e:
                print(f"[HybridStore] Chroma add failed: {e}")
        # FTS
        for _id, txt, meta in zip(ids, texts, metadatas):
            try:
                self._fts_conn.execute(
                    "INSERT OR REPLACE INTO docs (id, name, smiles, text, tags) VALUES (?,?,?,?,?)",
                    (_id, meta.get("name", ""), meta.get("smiles", ""), txt, ",".join(meta.get("tags", []))),
                )
            except Exception as e:
                print(f"[HybridStore] FTS insert failed: {e}")
        self._fts_conn.commit()
        # Mem fallback
        for _id, txt, meta, emb in zip(ids, texts, metadatas, embeddings):
            self._mem_docs.append({"id": _id, "text": txt, "meta": meta})
            self._mem_embs.append(emb)

    def vector_search(self, query: str, k: int = 10) -> List[Tuple[str, float, dict]]:
        """Returns list of (id, distance/score, meta)."""
        q_emb = embed_text(query)
        if self.collection is not None:
            try:
                res = self.collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])
                ids = res.get("ids", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[]])[0]
                docs = res.get("documents", [[]])[0]
                out = []
                for _id, meta, dist, doc in zip(ids, metas, dists, docs):
                    # Chroma cosine distance: 0=identical, 2=opposite. Convert to similarity 1-dist/2 or 1-dist
                    score = 1.0 - dist
                    out.append((_id, score, {**meta, "_doc": doc}))
                return out
            except Exception as e:
                print(f"[HybridStore] vector_search chroma failed: {e}")
        # Fallback brute-force cosine
        scores: List[Tuple[str, float, dict]] = []
        for doc, emb in zip(self._mem_docs, self._mem_embs):
            # cosine
            dot = sum(a*b for a,b in zip(q_emb, emb))
            # embs are normalized, so dot is cosine
            scores.append((doc["id"], dot, {**doc["meta"], "_doc": doc["text"]}))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def fts_search(self, query: str, k: int = 10) -> List[Tuple[str, float, dict]]:
        # FTS5 BM25
        # Sanitize query for FTS5
        # Keep alphanumeric, split, OR
        tokens = re.findall(r"[a-zA-Z0-9\-]+", query)
        if not tokens:
            return []
        # Build FTS query: tokens joined with OR, quote each
        fts_q = " OR ".join(f'"{t}"' for t in tokens[:10])
        try:
            cur = self._fts_conn.execute(
                """
                SELECT docs.id, docs.name, docs.smiles, docs.text, rank
                FROM docs_fts
                JOIN docs ON docs.rowid = docs_fts.rowid
                WHERE docs_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_q, k),
            )
            rows = cur.fetchall()
            # rank is negative bm25; more negative = better
            out = []
            for _id, name, smiles, text, rank in rows:
                score = -rank if rank is not None else 0.0
                # Normalize: FTS rank magnitude varies; map to 0-1 via tanh
                norm_score = math.tanh(abs(score) / 5.0)
                out.append((_id, norm_score, {"name": name, "smiles": smiles, "_doc": text}))
            return out
        except sqlite3.OperationalError as e:
            print(f"[HybridStore] FTS search failed: {e}")
            # Fallback substring
            q = query.lower()
            out = []
            for doc in self._mem_docs:
                txt = doc["text"].lower()
                if q in txt or any(t.lower() in txt for t in tokens):
                    score = txt.count(q) * 0.1 + sum(txt.count(t.lower())*0.05 for t in tokens)
                    out.append((doc["id"], min(score, 1.0), {**doc["meta"], "_doc": doc["text"]}))
            out.sort(key=lambda x: -x[1])
            return out[:k]
        except Exception as e:
            print(f"[HybridStore] FTS unknown error: {e}")
            return []

    def hybrid_search(self, query: str, k: int = 5, rrf_k: int = 60) -> List[RetrievedDoc]:
        """Reciprocal Rank Fusion of vector + FTS."""
        vec_hits = self.vector_search(query, k=50)
        fts_hits = self.fts_search(query, k=50)

        # Build rank maps
        vec_rank = {hid: i+1 for i, (hid, _, _) in enumerate(vec_hits)}
        fts_rank = {hid: i+1 for i, (hid, _, _) in enumerate(fts_hits)}
        all_ids = set(vec_rank) | set(fts_rank)

        # Build lookup for metadata/doc
        meta_map: dict[str, dict] = {}
        for hid, _, meta in vec_hits:
            meta_map[hid] = meta
        for hid, _, meta in fts_hits:
            if hid not in meta_map:
                meta_map[hid] = meta
            else:
                # merge
                meta_map[hid] = {**meta_map[hid], **meta}

        # RRF scores with exact-name boost for single-token queries (e.g., imatinib -> PF0001)
        q_lower = query.lower().strip()
        # extract first token for boost decision
        q_tokens = re.findall(r"[a-zA-Z0-9\-]+", q_lower)
        single_token = q_tokens[0] if len(q_tokens)==1 else None
        rrf: List[Tuple[str, float]] = []
        for hid in all_ids:
            s = 0.0
            if hid in vec_rank:
                s += 1.0 / (rrf_k + vec_rank[hid])
            if hid in fts_rank:
                s += 1.0 / (rrf_k + fts_rank[hid])
            # Boost exact name match — handles both single-token "imatinib" and multi-token "Find molecules similar to imatinib"
            meta = meta_map.get(hid, {})
            name_lower = str(meta.get("name","")).lower()
            if name_lower:
                # Strong boost if any query token exactly equals the molecule name (for multi-token queries like "Find molecules similar to imatinib")
                if name_lower in q_tokens:
                    # name is single word and appears as token in query
                    s += 0.08
                elif single_token and name_lower == single_token:
                    s += 0.08
                elif single_token and (name_lower.startswith(single_token + " ") or single_token in name_lower):
                    # fallback for single token partial
                    if name_lower.startswith(single_token + " "):
                        s += 0.03
                    elif single_token in name_lower:
                        s += 0.015 * (1.0 / (1.0 + len(name_lower.split()) - 1))
                elif single_token:
                    # no boost
                    pass
                else:
                    # multi-token query: small boost if name token appears in query
                    if q_lower and name_lower in q_lower:
                        # shouldn't happen for multi-word name, but keep
                        s += 0.01
            rrf.append((hid, s))
        rrf.sort(key=lambda x: -x[1])

        # Also need scores for provenance: combine vector + FTS scores
        vec_score_map = {hid: s for hid, s, _ in vec_hits}
        fts_score_map = {hid: s for hid, s, _ in fts_hits}

        results: List[RetrievedDoc] = []
        for hid, rrf_score in rrf[:k]:
            meta = meta_map.get(hid, {})
            # blended display score
            blend = 0.5 * vec_score_map.get(hid, 0) + 0.5 * fts_score_map.get(hid, 0)
            # prefer RRF magnitude but also include blend
            doc_text = meta.get("_doc", "")
            results.append(RetrievedDoc(
                id=hid,
                name=meta.get("name", hid),
                smiles=meta.get("smiles", ""),
                text=doc_text[:800],
                score=round(float(blend * 0.5 + rrf_score * 10), 4),
                provenance={"rrf": round(rrf_score, 5), "vector": round(vec_score_map.get(hid, 0), 4), "fts": round(fts_score_map.get(hid, 0), 4)},
                source="hybrid",
            ))
        return results

    def count(self) -> int:
        if self.collection is not None:
            try:
                return self.collection.count()
            except Exception:
                pass
        return len(self._mem_docs)
