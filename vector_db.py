"""Production-ready FAISS vector database for contract clauses and chunks."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

EMBEDDING_MODEL = "text-embedding-004"
DEFAULT_VECTOR_DIMS = 64


def chunk_text(text, size=300):
    """Split text into fixed-size word chunks."""
    words = (text or "").split()
    if not words:
        return []
    out = []
    for i in range(0, len(words), size):
        out.append(" ".join(words[i : i + size]))
    return out


def _safe_float_list(values) -> List[float]:
    out: List[float] = []
    for v in values or []:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


class ClauseVectorDB:
    """FAISS store with incremental insert, update/delete, filters, and ANN tuning."""

    def __init__(
        self,
        persist_path: str = "clause_vectors.json",
        embedding_model: str = EMBEDDING_MODEL,
        index_type: str = "hnsw",
        hnsw_m: int = 32,
        hnsw_ef_search: int = 80,
        ivf_nlist: int = 100,
        ivf_nprobe: int = 16,
        query_cache_size: int = 512,
    ):
        self.persist_path = Path(persist_path)
        self.sqlite_path = self.persist_path.with_suffix(".db")
        self.embedding_model = embedding_model
        self.vector_dims = DEFAULT_VECTOR_DIMS
        self.index_type = index_type.lower()
        self.hnsw_m = max(8, int(hnsw_m))
        self.hnsw_ef_search = max(16, int(hnsw_ef_search))
        self.ivf_nlist = max(8, int(ivf_nlist))
        self.ivf_nprobe = max(1, int(ivf_nprobe))

        self.index: faiss.Index = self._create_index()
        self.documents: Dict[str, Dict] = {}
        self.docid_to_faiss: Dict[str, int] = {}
        self.faiss_to_docid: Dict[int, str] = {}
        self.next_faiss_id = 1
        self._loaded = False
        self._lock = threading.RLock()
        self.query_cache_size = max(16, int(query_cache_size))
        self._query_embedding_cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._connection = self._create_connection()

        self._init_sqlite()

    def _normalize(self, vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return vec
        return [x / norm for x in vec]

    def _fallback_embedding(self, text: str, dims: int = DEFAULT_VECTOR_DIMS) -> List[float]:
        vec = [0.0] * dims
        words = (text or "").lower().split()
        if not words:
            return vec
        for token in words:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], "big") % dims
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vec[idx] += sign
        return self._normalize(vec)

    def _embed_with_model(self, client, text: str) -> Optional[List[float]]:
        try:
            response = client.models.embed_content(
                model=self.embedding_model,
                contents=text,
            )
            embeddings = getattr(response, "embeddings", None)
            if embeddings and getattr(embeddings[0], "values", None):
                return self._normalize(_safe_float_list(embeddings[0].values))
        except Exception:
            return None
        return None

    def embed_text(self, client, text: str) -> List[float]:
        if not client:
            return self._fallback_embedding(text, self.vector_dims)
        model_vec = self._embed_with_model(client, text)
        if model_vec:
            return model_vec
        return self._fallback_embedding(text, self.vector_dims)

    def _cached_query_embedding(self, client, text: str) -> List[float]:
        key = (text or "").strip()
        if not key:
            return self.embed_text(client, key)
        with self._lock:
            cached = self._query_embedding_cache.get(key)
            if cached is not None:
                self._query_embedding_cache.move_to_end(key)
                return cached
        vec = self.embed_text(client, key)
        with self._lock:
            self._query_embedding_cache[key] = vec
            self._query_embedding_cache.move_to_end(key)
            while len(self._query_embedding_cache) > self.query_cache_size:
                self._query_embedding_cache.popitem(last=False)
        return vec

    def _create_index(self) -> faiss.Index:
        if self.index_type == "hnsw":
            base = faiss.IndexHNSWFlat(self.vector_dims, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            base.hnsw.efSearch = self.hnsw_ef_search
            return faiss.IndexIDMap2(base)
        if self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.vector_dims)
            base = faiss.IndexIVFFlat(quantizer, self.vector_dims, self.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
            base.nprobe = self.ivf_nprobe
            return faiss.IndexIDMap2(base)
        base = faiss.IndexFlatIP(self.vector_dims)
        return faiss.IndexIDMap2(base)

    def _ensure_index_dims(self, dims: int):
        if dims <= 0:
            raise ValueError("Invalid embedding dimension.")
        if dims != self.vector_dims:
            # Protect against production mismatch between old persisted vectors and new model dims.
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.vector_dims}, got {dims}. "
                "Use a separate DB path when changing embedding model/dimensions."
            )

    def _init_sqlite(self):
        con = self._connection
        with self._lock:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    faiss_id INTEGER UNIQUE NOT NULL,
                    text_content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    vector_blob BLOB NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            con.commit()

    def _create_connection(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA temp_store=MEMORY;")
        return con

    def _vector_to_blob(self, vector: List[float]) -> bytes:
        arr = np.asarray(vector, dtype=np.float32)
        return arr.tobytes()

    def _blob_to_vector(self, blob: bytes) -> List[float]:
        if not blob:
            return []
        arr = np.frombuffer(blob, dtype=np.float32)
        return arr.tolist()

    def _save_meta(self):
        payload = {
            "embedding_model": self.embedding_model,
            "vector_dims": self.vector_dims,
            "index_type": self.index_type,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_search": self.hnsw_ef_search,
            "ivf_nlist": self.ivf_nlist,
            "ivf_nprobe": self.ivf_nprobe,
            "next_faiss_id": self.next_faiss_id,
        }
        con = self._connection
        with self._lock:
            con.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
                ("config", json.dumps(payload, ensure_ascii=False)),
            )
            con.commit()

    def _load_meta(self):
        con = self._connection
        with self._lock:
            row = con.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
        if not row:
            return
        cfg = json.loads(row[0])
        stored_dims = int(cfg.get("vector_dims", self.vector_dims))
        stored_model = cfg.get("embedding_model", self.embedding_model)
        if stored_dims != self.vector_dims or stored_model != self.embedding_model:
            raise ValueError(
                "Stored vector DB config differs from runtime embedding config. "
                "Use a different persist path for a new embedding model/dimension."
            )
        self.index_type = cfg.get("index_type", self.index_type)
        self.hnsw_m = int(cfg.get("hnsw_m", self.hnsw_m))
        self.hnsw_ef_search = int(cfg.get("hnsw_ef_search", self.hnsw_ef_search))
        self.ivf_nlist = int(cfg.get("ivf_nlist", self.ivf_nlist))
        self.ivf_nprobe = int(cfg.get("ivf_nprobe", self.ivf_nprobe))
        self.next_faiss_id = int(cfg.get("next_faiss_id", self.next_faiss_id))

    def _train_if_needed(self, vectors: np.ndarray):
        base = self.index.index if isinstance(self.index, faiss.IndexIDMap2) else self.index
        if isinstance(base, faiss.IndexIVFFlat) and not base.is_trained:
            if vectors.shape[0] >= self.ivf_nlist:
                base.train(vectors)

    def _rebuild_from_documents(self):
        self.index = self._create_index()
        if not self.documents:
            return
        items = sorted(self.documents.items(), key=lambda x: self.docid_to_faiss.get(x[0], 0))
        vectors = []
        ids = []
        for doc_id, doc in items:
            vec = _safe_float_list(doc.get("vector"))
            if len(vec) != self.vector_dims:
                continue
            fid = self.docid_to_faiss.get(doc_id)
            if fid is None:
                continue
            vectors.append(vec)
            ids.append(fid)
        if not vectors:
            return
        v_arr = np.asarray(vectors, dtype=np.float32)
        i_arr = np.asarray(ids, dtype=np.int64)
        self._train_if_needed(v_arr)
        base = self.index.index if isinstance(self.index, faiss.IndexIDMap2) else self.index
        if isinstance(base, faiss.IndexIVFFlat) and not base.is_trained:
            return
        self.index.add_with_ids(v_arr, i_arr)

    def _token_overlap_score(self, query: str, text: str) -> float:
        q = set((query or "").lower().split())
        t = set((text or "").lower().split())
        if not q or not t:
            return 0.0
        return len(q & t) / max(len(q), 1)

    def _hybrid_score(self, query: str, vector_score: float, doc_text: str) -> float:
        lexical = self._token_overlap_score(query, doc_text)
        return 0.8 * vector_score + 0.2 * lexical

    def _rerank(self, client, query: str, rows: List[Dict], mode: str = "none") -> List[Dict]:
        if mode != "none":
            return rows
        return rows

    def _matches_filters(self, metadata: Dict, metadata_filter: Optional[Dict]) -> bool:
        if not metadata_filter:
            return True
        for key, expected in metadata_filter.items():
            if metadata.get(key) != expected:
                return False
        return True

    def _persist_document(self, doc: Dict, faiss_id: int):
        con = self._connection
        with self._lock:
            con.execute(
                """
                INSERT OR REPLACE INTO documents(doc_id, faiss_id, text_content, metadata_json, vector_blob, active)
                VALUES(?, ?, ?, ?, ?, 1)
                """,
                (
                    doc["id"],
                    int(faiss_id),
                    doc["text"],
                    json.dumps(doc["metadata"], ensure_ascii=False),
                    self._vector_to_blob(doc["vector"]),
                ),
            )
            con.commit()

    def _set_document_active(self, doc_id: str, active: bool):
        con = self._connection
        with self._lock:
            con.execute(
                "UPDATE documents SET active=? WHERE doc_id = ?",
                (1 if active else 0, doc_id),
            )
            con.commit()

    def _allocate_faiss_id(self) -> int:
        fid = int(self.next_faiss_id)
        self.next_faiss_id += 1
        return fid

    def insert(
        self,
        client,
        text: str,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        result = self.bulk_insert(
            client,
            [{"text": text, "metadata": metadata or {}, "doc_id": doc_id}],
        )
        return result[0]

    def bulk_insert(self, client, docs: List[Dict], batch_size: int = 128) -> List[str]:
        inserted_ids: List[str] = []
        if not docs:
            return inserted_ids

        with self._lock:
            pending_vectors: List[List[float]] = []
            pending_faiss_ids: List[int] = []
            pending_persist: List[tuple] = []

            for payload in docs:
                text = payload.get("text", "")
                metadata = payload.get("metadata", {}) or {}
                item_id = payload.get("doc_id") or str(uuid.uuid4())

                if item_id in self.documents:
                    self.delete(item_id, soft=True)

                vector = self.embed_text(client, text)
                self._ensure_index_dims(len(vector))
                faiss_id = self._allocate_faiss_id()
                doc = {
                    "id": item_id,
                    "text": text,
                    "metadata": metadata,
                    "vector": vector,
                }
                self.documents[item_id] = doc
                self.docid_to_faiss[item_id] = faiss_id
                self.faiss_to_docid[faiss_id] = item_id
                inserted_ids.append(item_id)

                pending_vectors.append(vector)
                pending_faiss_ids.append(faiss_id)
                pending_persist.append((doc, faiss_id))

                if len(pending_vectors) >= batch_size:
                    self._flush_batch(pending_vectors, pending_faiss_ids, pending_persist)
                    pending_vectors, pending_faiss_ids, pending_persist = [], [], []

            if pending_vectors:
                self._flush_batch(pending_vectors, pending_faiss_ids, pending_persist)

            self._save_meta()
        return inserted_ids

    def _flush_batch(self, vectors: List[List[float]], faiss_ids: List[int], rows: List[tuple]):
        v_arr = np.asarray(vectors, dtype=np.float32)
        ids_arr = np.asarray(faiss_ids, dtype=np.int64)
        self._train_if_needed(v_arr)
        base = self.index.index if isinstance(self.index, faiss.IndexIDMap2) else self.index
        if isinstance(base, faiss.IndexIVFFlat) and not base.is_trained:
            self._rebuild_from_documents()
        else:
            self.index.add_with_ids(v_arr, ids_arr)
        self._persist_rows(rows)

    def _persist_rows(self, rows: List[tuple]):
        con = self._connection
        with self._lock:
            con.executemany(
                """
                INSERT OR REPLACE INTO documents(doc_id, faiss_id, text_content, metadata_json, vector_blob, active)
                VALUES(?, ?, ?, ?, ?, 1)
                """,
                [
                    (
                        doc["id"],
                        int(faiss_id),
                        doc["text"],
                        json.dumps(doc["metadata"], ensure_ascii=False),
                        self._vector_to_blob(doc["vector"]),
                    )
                    for doc, faiss_id in rows
                ],
            )
            con.commit()

    def delete(self, doc_id: str, soft: bool = True) -> bool:
        if doc_id not in self.documents:
            return False
        faiss_id = self.docid_to_faiss.get(doc_id)
        self.documents.pop(doc_id, None)
        self.docid_to_faiss.pop(doc_id, None)
        if faiss_id is not None:
            self.faiss_to_docid.pop(faiss_id, None)
            if not soft:
                remove_ids = np.asarray([faiss_id], dtype=np.int64)
                try:
                    self.index.remove_ids(remove_ids)
                except RuntimeError:
                    self._rebuild_from_documents()
        self._set_document_active(doc_id, False)
        self._save_meta()
        return True

    def update(
        self,
        client,
        doc_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        current = self.documents.get(doc_id)
        if not current:
            return False
        new_text = text if text is not None else current.get("text", "")
        new_meta = metadata if metadata is not None else current.get("metadata", {})
        self.delete(doc_id, soft=True)
        self.insert(client, text=new_text, metadata=new_meta, doc_id=doc_id)
        return True

    def clear(self):
        self.documents = {}
        self.docid_to_faiss = {}
        self.faiss_to_docid = {}
        self.next_faiss_id = 1
        self.index = self._create_index()
        con = self._connection
        with self._lock:
            con.execute("DELETE FROM documents")
            con.commit()
        self._save_meta()

    def add_clauses(self, client, clauses: List[Dict], chunk_size: int = 300):
        """Build store from clauses using chunked documents."""
        self.clear()
        for clause in clauses:
            clause_text = clause.get("clause_text", "")
            clause_title = clause.get("clause_title", "")
            clause_number = str(clause.get("clause_number", "")).strip()
            chunks = chunk_text(clause_text, size=chunk_size) or [clause_text]
            for idx, chunk in enumerate(chunks):
                metadata = {
                    "clause_number": clause_number,
                    "clause_title": clause_title,
                    "chunk_index": idx,
                    "chunk_count": len(chunks),
                    "source": "contract_clause",
                    "clause": clause,
                }
                self.insert(client, text=chunk, metadata=metadata, doc_id=f"{clause_number or 'clause'}:{idx}")

    def save(self):
        # Data is persisted incrementally in SQLite on each write.
        self._save_meta()

    def load(self) -> bool:
        self._load_meta()
        self.documents = {}
        self.docid_to_faiss = {}
        self.faiss_to_docid = {}
        con = self._connection
        with self._lock:
            rows = con.execute(
                "SELECT doc_id, faiss_id, text_content, metadata_json, vector_blob, active FROM documents"
            ).fetchall()
        for doc_id, faiss_id, text_content, metadata_json, vector_blob, active in rows:
            if int(active) != 1:
                continue
            metadata = json.loads(metadata_json or "{}")
            vector = self._blob_to_vector(vector_blob)
            self.documents[str(doc_id)] = {
                "id": str(doc_id),
                "text": text_content or "",
                "metadata": metadata,
                "vector": vector,
            }
            fid = int(faiss_id)
            self.docid_to_faiss[str(doc_id)] = fid
            self.faiss_to_docid[fid] = str(doc_id)
            self.next_faiss_id = max(self.next_faiss_id, fid + 1)
        self._rebuild_from_documents()
        self._loaded = True
        return True

    def search(
        self,
        client,
        query: str,
        top_k: int = 3,
        metadata_filter: Optional[Dict] = None,
        rerank_mode: str = "none",
    ) -> List[Dict]:
        if not self.documents and not self._loaded:
            if not self.load():
                return []
        if not self.documents:
            return []

        query_vec = self._cached_query_embedding(client, query)
        if len(query_vec) != self.vector_dims:
            return []
        q = np.asarray([query_vec], dtype=np.float32)

        base_index = self.index.index if isinstance(self.index, faiss.IndexIDMap2) else self.index
        if isinstance(base_index, faiss.IndexHNSWFlat):
            base_index.hnsw.efSearch = max(self.hnsw_ef_search, top_k * 8)
            scan_k = min(base_index.hnsw.efSearch, max(len(self.documents), 1))
        elif isinstance(base_index, faiss.IndexIVFFlat):
            base_index.nprobe = max(self.ivf_nprobe, min(self.ivf_nlist, max(4, top_k // 2)))
            scan_k = min(max(top_k * base_index.nprobe, top_k), max(len(self.documents), 1))
        else:
            scan_k = min(max(top_k * 8, top_k), max(len(self.documents), 1))
        scores, ids = self.index.search(q, scan_k)
        rows = []
        for score, faiss_id in zip(scores[0], ids[0]):
            if faiss_id < 0:
                continue
            doc_id = self.faiss_to_docid.get(int(faiss_id))
            if not doc_id:
                continue
            doc = self.documents.get(doc_id, {})
            metadata = doc.get("metadata", {}) or {}
            if not self._matches_filters(metadata, metadata_filter):
                continue
            text = doc.get("text", "")
            hybrid_score = self._hybrid_score(query, float(score), text)
            clause_obj = metadata.get("clause") or {
                "clause_number": metadata.get("clause_number"),
                "clause_title": metadata.get("clause_title"),
                "clause_text": text,
            }
            rows.append(
                {
                    "id": doc_id,
                    "score": hybrid_score,
                    "vector_score": float(score),
                    "text": text,
                    "metadata": metadata,
                    "clause": clause_obj,
                }
            )

        rows.sort(key=lambda x: x["score"], reverse=True)
        rows = self._rerank(client, query, rows, mode=rerank_mode)
        return rows[:top_k]

    async def asearch(
        self,
        client,
        query: str,
        top_k: int = 3,
        metadata_filter: Optional[Dict] = None,
        rerank_mode: str = "none",
    ) -> List[Dict]:
        import asyncio

        return await asyncio.to_thread(
            self.search,
            client,
            query,
            top_k,
            metadata_filter,
            rerank_mode,
        )
