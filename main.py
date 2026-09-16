import json
import math
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import heapq
from typing import Callable, Dict, List, Optional, Tuple, Any

from fastapi import FastAPI, HTTPException, Query, Path, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

DIMS = 16  # Demo vector dimensions

# =====================================================================
#  DATA TYPES
# =====================================================================

class VectorItem:
    def __init__(self, id: int, metadata: str, category: str, emb: List[float]):
        self.id = id
        self.metadata = metadata
        self.category = category
        self.emb = emb

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "metadata": self.metadata,
            "category": self.category,
            "embedding": self.emb,
        }


class DocItem:
    def __init__(self, id: int, title: str, text: str, emb: List[float]):
        self.id = id
        self.title = title
        self.text = text
        self.emb = emb

    def to_dict(self) -> Dict[str, Any]:
        preview = self.text[:120]
        if len(self.text) > 120:
            preview += "…"
        return {
            "id": self.id,
            "title": self.title,
            "preview": preview,
            "words": len(self.text.split()),
        }


DistFn = Callable[[List[float], List[float]], float]

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: List[float], b: List[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


def cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    denom = math.sqrt(na) * math.sqrt(nb)
    sim = dot / denom
    return 1.0 - max(min(sim, 1.0), -1.0)


def manhattan(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(name: str) -> DistFn:
    if name == "cosine":
        return cosine
    if name == "manhattan":
        return manhattan
    return euclidean


# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem) -> None:
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist_fn: DistFn) -> List[Tuple[float, int]]:
        res = [(dist_fn(q, v.emb), v.id) for v in self.items]
        res.sort(key=lambda x: x[0])
        return res[:k]

    def remove(self, id: int) -> None:
        self.items = [v for v in self.items if v.id != id]


# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    def __init__(self, item: VectorItem):
        self.item = item
        self.left: Optional[KDNode] = None
        self.right: Optional[KDNode] = None


class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root: Optional[KDNode] = None

    def _ins(self, node: Optional[KDNode], v: VectorItem, d: int) -> KDNode:
        if node is None:
            return KDNode(v)
        ax = d % self.dims
        if v.emb[ax] < node.item.emb[ax]:
            node.left = self._ins(node.left, v, d + 1)
        else:
            node.right = self._ins(node.right, v, d + 1)
        return node

    def insert(self, v: VectorItem) -> None:
        self.root = self._ins(self.root, v, 0)

    def _knn(
        self,
        node: Optional[KDNode],
        q: List[float],
        k: int,
        d: int,
        dist_fn: DistFn,
        heap: List[Tuple[float, int]],
    ) -> None:
        if node is None:
            return
        dn = dist_fn(q, node.item.emb)
        # Store (-dn, id) so Python's min-heap acts as max-heap of size k
        if len(heap) < k:
            heapq.heappush(heap, (-dn, node.item.id))
        elif dn < -heap[0][0]:
            heapq.heapreplace(heap, (-dn, node.item.id))

        ax = d % self.dims
        diff = q[ax] - node.item.emb[ax]
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, q, k, d + 1, dist_fn, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, d + 1, dist_fn, heap)

    def knn(self, q: List[float], k: int, dist_fn: DistFn) -> List[Tuple[float, int]]:
        heap: List[Tuple[float, int]] = []
        self._knn(self.root, q, k, 0, dist_fn, heap)
        # Extract and sort by actual distance ascending
        res = [(-d, item_id) for d, item_id in heap]
        res.sort(key=lambda x: x[0])
        return res

    def rebuild(self, items: List[VectorItem]) -> None:
        self.root = None
        for v in items:
            self.insert(v)


# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSWNode:
    def __init__(self, item: VectorItem, max_lyr: int):
        self.item = item
        self.max_lyr = max_lyr
        self.nbrs: List[List[int]] = [[] for _ in range(max_lyr + 1)]


class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200):
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m))
        self.G: Dict[int, HNSWNode] = {}
        self.top_layer = -1
        self.entry_pt = -1
        self.rng = random.Random(42)

    def _rand_level(self) -> int:
        u = self.rng.random()
        if u <= 0.0:
            u = 1e-7
        return int(math.floor(-math.log(u) * self.mL))

    def _search_layer(
        self,
        q: List[float],
        ep: int,
        ef: int,
        lyr: int,
        dist_fn: DistFn,
    ) -> List[Tuple[float, int]]:
        if ep not in self.G:
            return []

        vis = {ep}
        d0 = dist_fn(q, self.G[ep].item.emb)

        # Min-heap for candidate exploration: (distance, node_id)
        cands: List[Tuple[float, int]] = [(d0, ep)]
        # Max-heap for nearest found set of size <= ef: (-distance, node_id)
        found: List[Tuple[float, int]] = [(-d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            farthest_found_dist = -found[0][0]

            if len(found) >= ef and cd > farthest_found_dist:
                break

            node = self.G.get(cid)
            if node is None or lyr >= len(node.nbrs):
                continue

            for nid in node.nbrs[lyr]:
                if nid in vis or nid not in self.G:
                    continue
                vis.add(nid)
                nd = dist_fn(q, self.G[nid].item.emb)
                farthest_found_dist = -found[0][0]

                if len(found) < ef or nd < farthest_found_dist:
                    heapq.heappush(cands, (nd, nid))
                    if len(found) < ef:
                        heapq.heappush(found, (-nd, nid))
                    else:
                        heapq.heapreplace(found, (-nd, nid))

        res = [(-d, nid) for d, nid in found]
        res.sort(key=lambda x: x[0])
        return res

    def _select_nbrs(
        self, cands: List[Tuple[float, int]], max_m: int
    ) -> List[int]:
        return [cid for _, cid in cands[:max_m]]

    def insert(self, item: VectorItem, dist_fn: DistFn) -> None:
        item_id = item.id
        lvl = self._rand_level()
        self.G[item_id] = HNSWNode(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt = item_id
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            node = self.G.get(ep)
            if node and lc < len(node.nbrs):
                w = self._search_layer(item.emb, ep, 1, lc, dist_fn)
                if w:
                    ep = w[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            w = self._search_layer(item.emb, ep, self.ef_build, lc, dist_fn)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(w, max_m)
            self.G[item_id].nbrs[lc] = list(sel)

            for nid in sel:
                nbr_node = self.G.get(nid)
                if not nbr_node:
                    continue
                while len(nbr_node.nbrs) <= lc:
                    nbr_node.nbrs.append([])
                conn = nbr_node.nbrs[lc]
                conn.append(item_id)
                if len(conn) > max_m:
                    ds = [
                        (dist_fn(nbr_node.item.emb, self.G[c].item.emb), c)
                        for c in conn
                        if c in self.G
                    ]
                    ds.sort(key=lambda x: x[0])
                    nbr_node.nbrs[lc] = [c for _, c in ds[:max_m]]

            if w:
                ep = w[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = item_id

    def knn(
        self, q: List[float], k: int, ef: int, dist_fn: DistFn
    ) -> List[Tuple[float, int]]:
        if self.entry_pt == -1 or not self.G:
            return []

        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            node = self.G.get(ep)
            if node and lc < len(node.nbrs):
                w = self._search_layer(q, ep, 1, lc, dist_fn)
                if w:
                    ep = w[0][1]

        w = self._search_layer(q, ep, max(ef, k), 0, dist_fn)
        return w[:k]

    def remove(self, item_id: int) -> None:
        if item_id not in self.G:
            return

        for nid, nd in self.G.items():
            if nid == item_id:
                continue
            for layer in nd.nbrs:
                if item_id in layer:
                    layer.remove(item_id)

        if self.entry_pt == item_id:
            self.entry_pt = -1
            for nid in self.G:
                if nid != item_id:
                    self.entry_pt = nid
                    break

        del self.G[item_id]

    def get_info(self) -> Dict[str, Any]:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []

        for nid, nd in self.G.items():
            nodes.append({
                "id": nid,
                "metadata": nd.item.metadata,
                "category": nd.item.category,
                "maxLyr": nd.max_lyr,
            })
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for neighbor_id in nd.nbrs[lc]:
                        if nid < neighbor_id:
                            edges_per_layer[lc] += 1
                            edges.append({
                                "src": nid,
                                "dst": neighbor_id,
                                "lyr": lc,
                            })

        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
        }

    def size(self) -> int:
        return len(self.G)


# =====================================================================
#  VECTOR DATABASE (16D demo vector index)
# =====================================================================

class VectorDB:
    def __init__(self, dims: int = DIMS, chroma_client: Optional[Any] = None):
        self.dims = dims
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.mu = threading.Lock()
        self.next_id = 1
        self.chroma_client = chroma_client
        self.chroma_col = None
        if chroma_client is not None:
            try:
                self.chroma_col = chroma_client.get_or_create_collection(
                    name="demo_vectors",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                print(f"ChromaDB demo_vectors collection initialization note: {e}")

    def insert(
        self,
        meta: str,
        cat: str,
        emb: List[float],
        dist_fn: DistFn,
    ) -> int:
        with self.mu:
            v = VectorItem(self.next_id, meta, cat, emb)
            self.next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist_fn)
            if self.chroma_col is not None:
                try:
                    self.chroma_col.upsert(
                        ids=[str(v.id)],
                        embeddings=[v.emb],
                        metadatas=[{"metadata": v.metadata, "category": v.category}],
                    )
                except Exception as e:
                    print(f"ChromaDB insert note: {e}")
            return v.id

    def remove(self, item_id: int) -> bool:
        with self.mu:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.bf.remove(item_id)
            self.hnsw.remove(item_id)
            self.kdt.rebuild(list(self.store.values()))
            if self.chroma_col is not None:
                try:
                    self.chroma_col.delete(ids=[str(item_id)])
                except Exception:
                    pass
            return True

    def search(
        self,
        q: List[float],
        k: int,
        metric: str,
        algo: str,
    ) -> Dict[str, Any]:
        with self.mu:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()

            if algo == "chromadb" and self.chroma_col is not None and len(self.store) > 0:
                n_res = min(k, len(self.store))
                res = self.chroma_col.query(
                    query_embeddings=[q],
                    n_results=n_res,
                    include=["metadatas", "distances", "embeddings"]
                )
                us = int((time.perf_counter() - t0) * 1_000_000)
                hits = []
                if res and res.get("ids") and len(res["ids"]) > 0:
                    for i, item_id_str in enumerate(res["ids"][0]):
                        item_id = int(item_id_str) if item_id_str.isdigit() else 0
                        dist = float(res["distances"][0][i]) if res.get("distances") else 0.0
                        meta_dict = res["metadatas"][0][i] if res.get("metadatas") else {}
                        emb_raw = res["embeddings"][0][i] if res.get("embeddings") is not None else (self.store[item_id].emb if item_id in self.store else [])
                        if hasattr(emb_raw, "tolist"):
                            emb = emb_raw.tolist()
                        elif isinstance(emb_raw, (list, tuple)):
                            emb = [float(x) for x in emb_raw]
                        else:
                            emb = []
                        hits.append({
                            "id": item_id,
                            "metadata": meta_dict.get("metadata", self.store[item_id].metadata if item_id in self.store else ""),
                            "category": meta_dict.get("category", self.store[item_id].category if item_id in self.store else "cs"),
                            "distance": round(dist, 6),
                            "embedding": emb,
                        })
                return {
                    "results": hits,
                    "latencyUs": us,
                    "algo": algo,
                    "metric": metric,
                }

            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)

            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, item_id in raw:
                if item_id in self.store:
                    item = self.store[item_id]
                    hits.append({
                        "id": item.id,
                        "metadata": item.metadata,
                        "category": item.category,
                        "distance": round(d, 6),
                        "embedding": item.emb,
                    })

            return {
                "results": hits,
                "latencyUs": us,
                "algo": algo,
                "metric": metric,
            }

    def benchmark(
        self, q: List[float], k: int, metric: str
    ) -> Dict[str, Any]:
        with self.mu:
            dfn = get_dist_fn(metric)

            def time_fn(fn: Callable[[], Any]) -> int:
                t0 = time.perf_counter()
                fn()
                return int((time.perf_counter() - t0) * 1_000_000)

            bf_us = time_fn(lambda: self.bf.knn(q, k, dfn))
            kd_us = time_fn(lambda: self.kdt.knn(q, k, dfn))
            hnsw_us = time_fn(lambda: self.hnsw.knn(q, k, 50, dfn))
            chroma_us = 0
            if self.chroma_col is not None and len(self.store) > 0:
                try:
                    chroma_us = time_fn(lambda: self.chroma_col.query(query_embeddings=[q], n_results=min(k, len(self.store))))
                except Exception:
                    chroma_us = 0

            return {
                "bruteforceUs": bf_us,
                "kdtreeUs": kd_us,
                "hnswUs": hnsw_us,
                "chromadbUs": chroma_us,
                "itemCount": len(self.store),
            }

    def all(self) -> List[Dict[str, Any]]:
        with self.mu:
            return [v.to_dict() for v in self.store.values()]

    def hnsw_info(self) -> Dict[str, Any]:
        with self.mu:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.mu:
            return len(self.store)


# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(
    text: str, chunk_words: int = 250, overlap_words: int = 30
) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = chunk_words - overlap_words
    if step <= 0:
        step = chunk_words

    for i in range(0, len(words), step):
        chunk_slice = words[i : i + chunk_words]
        chunks.append(" ".join(chunk_slice))
        if i + chunk_words >= len(words):
            break

    return chunks


# =====================================================================
#  OLLAMA CLIENT
# =====================================================================

class OllamaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 11434,
        embed_model: Optional[str] = None,
        gen_model: Optional[str] = None,
    ):
        raw = (
            base_url
            or host
            or os.environ.get("OLLAMA_HOST")
            or os.environ.get("OLLAMA_URL")
            or "http://127.0.0.1:11434"
        ).strip()
        self._base_url = self._normalize_url(raw, port)
        self.embed_model = embed_model or os.environ.get("OLLAMA_EMBED_MODEL") or "nomic-embed-text"
        self.gen_model = gen_model or os.environ.get("OLLAMA_GEN_MODEL") or "llama3.2"

    def _normalize_url(self, raw: str, default_port: int = 11434) -> str:
        url = raw.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"http://{url}"
        parsed = urllib.parse.urlparse(url)
        if not parsed.port and not parsed.netloc.endswith(":11434") and "." not in parsed.netloc.split(":")[0]:
            pass
        return url.rstrip("/")

    @property
    def base_url(self) -> str:
        return self._base_url

    def set_url(
        self,
        url: str,
        embed_model: Optional[str] = None,
        gen_model: Optional[str] = None,
    ) -> None:
        self._base_url = self._normalize_url(url)
        if embed_model:
            self.embed_model = embed_model.strip()
        if gen_model:
            self.gen_model = gen_model.strip()

    def list_models(self) -> List[str]:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception:
            pass
        return []

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def embed(self, text: str) -> List[float]:
        try:
            payload = json.dumps({
                "model": self.embed_model,
                "prompt": text,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("embedding", [])
        except Exception:
            pass
        return []

    def generate(self, prompt: str) -> str:
        try:
            payload = json.dumps({
                "model": self.gen_model,
                "prompt": prompt,
                "stream": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("response", "")
        except Exception:
            pass
        return f"ERROR: Ollama server unavailable at {self.base_url}."


# =====================================================================
#  DOCUMENT DATABASE (ChromaDB + HNSW for real Ollama embeddings)
# =====================================================================

class DocumentDB:
    def __init__(self, chroma_client: Optional[Any] = None):
        self.store: Dict[int, DocItem] = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.mu = threading.Lock()
        self.next_id = 1
        self.dims = 0
        self.chroma_client = chroma_client
        self.col = None
        if chroma_client is not None:
            try:
                self.col = chroma_client.get_or_create_collection(
                    name="rag_documents",
                    metadata={"hnsw:space": "cosine"}
                )
                existing = self.col.get(include=["metadatas", "embeddings", "documents"])
                if existing and existing.get("ids"):
                    for i, id_str in enumerate(existing["ids"]):
                        doc_id = int(id_str) if id_str.isdigit() else i + 1
                        metas = existing["metadatas"][i] if existing.get("metadatas") else {}
                        title = metas.get("title", f"Doc {doc_id}")
                        text = metas.get("text", existing["documents"][i] if existing.get("documents") else "")
                        emb_raw = existing["embeddings"][i] if existing.get("embeddings") is not None and len(existing["embeddings"]) > i else []
                        if hasattr(emb_raw, "tolist"):
                            emb = emb_raw.tolist()
                        elif isinstance(emb_raw, (list, tuple)):
                            emb = list(emb_raw)
                        else:
                            emb = []
                        item = DocItem(doc_id, title, text, emb)
                        self.store[doc_id] = item
                        if emb:
                            if self.dims == 0:
                                self.dims = len(emb)
                            vi = VectorItem(doc_id, title, "doc", emb)
                            self.hnsw.insert(vi, cosine)
                            self.bf.insert(vi)
                        if doc_id >= self.next_id:
                            self.next_id = doc_id + 1
            except Exception as e:
                print(f"ChromaDB rag_documents initialization note: {e}")

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self.mu:
            if self.dims == 0:
                self.dims = len(emb)
            doc_id = self.next_id
            self.next_id += 1
            item = DocItem(doc_id, title, text, emb)
            self.store[doc_id] = item
            vi = VectorItem(doc_id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)

            if self.col is not None:
                try:
                    self.col.upsert(
                        ids=[str(doc_id)],
                        embeddings=[emb],
                        documents=[text],
                        metadatas=[{"title": title, "text": text}],
                    )
                except Exception as e:
                    print(f"ChromaDB doc insert note: {e}")

            return doc_id

    def search(
        self, q: List[float], k: int, max_dist: Optional[float] = 1.2
    ) -> List[Tuple[float, DocItem]]:
        with self.mu:
            if not self.store:
                return []

            if self.col is not None and self.col.count() > 0:
                try:
                    n_res = min(k, len(self.store))
                    res = self.col.query(
                        query_embeddings=[q],
                        n_results=n_res,
                        include=["metadatas", "distances"]
                    )
                    out = []
                    if res and res.get("ids") and len(res["ids"]) > 0:
                        for i, doc_id_str in enumerate(res["ids"][0]):
                            doc_id = int(doc_id_str) if doc_id_str.isdigit() else None
                            dist = float(res["distances"][0][i]) if res.get("distances") else 0.0
                            if doc_id in self.store and (max_dist is None or dist <= max_dist):
                                out.append((dist, self.store[doc_id]))
                    return out
                except Exception as e:
                    print(f"ChromaDB search note: {e}")

            raw = (
                self.bf.knn(q, k, cosine)
                if len(self.store) < 10
                else self.hnsw.knn(q, k, 50, cosine)
            )
            out = []
            for d, item_id in raw:
                if item_id in self.store and (max_dist is None or d <= max_dist):
                    out.append((d, self.store[item_id]))
            return out

    def remove(self, doc_id: int) -> bool:
        with self.mu:
            if doc_id not in self.store:
                return False
            del self.store[doc_id]
            self.hnsw.remove(doc_id)
            self.bf.remove(doc_id)
            if self.col is not None:
                try:
                    self.col.delete(ids=[str(doc_id)])
                except Exception:
                    pass
            return True

    def all(self) -> List[Dict[str, Any]]:
        with self.mu:
            return [doc.to_dict() for doc in self.store.values()]

    def size(self) -> int:
        with self.mu:
            return len(self.store)

    def get_dims(self) -> int:
        with self.mu:
            return self.dims


# =====================================================================
#  LINEAR RAG PIPELINE (Deterministic Retrieval-Augmented Generation)
# =====================================================================

class LinearRAG:
    """
    High-Performance Deterministic Linear RAG Pipeline:
      1. Embed Query via nomic-embed-text (768D)
      2. Retrieve Top-K Nearest Document Chunks from VectorDB (ChromaDB / HNSW)
      3. Format Context Prompt with Grounding Rules
      4. Synthesize Grounded Answer using Local LLM (llama3.2)
    """

    def __init__(self, doc_db: DocumentDB, ollama: OllamaClient):
        self.doc_db = doc_db
        self.ollama = ollama

    def run(self, question: str, k: int = 3) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q_emb = self.ollama.embed(question)
        if not q_emb:
            return {
                "answer": "Ollama service unavailable. Please ensure Ollama is running at " + self.ollama.base_url,
                "error": "Ollama unavailable",
                "contexts": [],
                "docCount": self.doc_db.size(),
                "workflow": "Linear RAG",
                "steps": [],
            }

        hits = self.doc_db.search(q_emb, k)
        contexts = [
            {
                "id": item.id,
                "title": item.title,
                "text": item.text,
                "distance": round(dist, 4),
            }
            for dist, item in hits
        ]

        steps = [
            {
                "step": "retrieve",
                "node": "Vector Retriever",
                "action": "Vector Similarity Search",
                "detail": f"Retrieved {len(hits)} nearest context chunk(s) from VectorDB for query: '{question}'",
                "count": len(hits),
                "timestamp": round(time.time(), 3),
            }
        ]

        ctx_str = ""
        for i, (dist, item) in enumerate(hits):
            ctx_str += f"[{i+1}] Document: {item.title}\nContent: {item.text}\n\n"

        if not hits or not ctx_str.strip():
            ctx_str = "(No relevant documents found in knowledge base)\n\n"

        prompt = (
            "You are a helpful knowledge assistant. Use the following context (including document titles and content) to answer the user's question clearly and accurately.\n"
            "- Synthesize your answer directly from the facts given in the context.\n"
            "- If the context does not contain any relevant information to answer the question, say: 'I cannot answer this question based on the provided documents.'\n\n"
            f"Context:\n{ctx_str}"
            f"Question: {question}\n\n"
            "Answer:"
        )

        answer = self.ollama.generate(prompt)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        steps.append({
            "step": "generate",
            "node": "Grounded Generator",
            "action": "Linear Context Synthesis",
            "detail": f"Synthesized grounded response in {elapsed_ms}ms using {len(hits)} chunk(s) via {self.ollama.gen_model}.",
            "model": self.ollama.gen_model,
            "timestamp": round(time.time(), 3),
        })

        return {
            "answer": answer,
            "model": self.ollama.gen_model,
            "contexts": contexts,
            "docCount": self.doc_db.size(),
            "workflow": "Linear RAG",
            "steps": steps,
            "latency_ms": elapsed_ms,
        }


# =====================================================================
#  DEMO DATA (16D categorical vectors)
# =====================================================================

def load_demo(db: VectorDB) -> None:
    dist = get_dist_fn("cosine")
    demo_items = [
        # Computer Science (dims 0-3 dominant)
        ("Linked List: nodes connected by pointers", "cs", [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs", [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs", [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs", [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs", [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09]),

        # Mathematics (dims 4-7 dominant)
        ("Calculus: derivatives integrals and limits", "math", [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math", [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09]),
        ("Probability: distributions random variables Bayes theorem", "math", [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math", [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06]),
        ("Combinatorics: permutations combinations generating functions", "math", [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07]),

        # Food & Cooking (dims 8-11 dominant)
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food", [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food", [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food", [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food", [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food", [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06]),

        # Sports & Games (dims 12-15 dominant)
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports", [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports", [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports", [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports", [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports", [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80]),
    ]
    for meta, cat, emb in demo_items:
        db.insert(meta, cat, emb, dist)


# =====================================================================
#  PYDANTIC REQUEST SCHEMAS
# =====================================================================

class InsertVectorRequest(BaseModel):
    metadata: str = Field(..., description="Description or label of the vector")
    category: str = Field(..., description="Category: cs, math, food, sports, doc")
    embedding: List[float] = Field(..., description="16D float embedding array")


class InsertDocRequest(BaseModel):
    title: str = Field(..., description="Title or topic of the document")
    text: str = Field(..., description="Full text content of document")


class SearchDocRequest(BaseModel):
    question: str = Field(..., description="Question to search context for")
    k: int = Field(default=3, ge=1, le=20, description="Top-k chunks to retrieve")


class AskDocRequest(BaseModel):
    question: str = Field(..., description="Question for the RAG agent to answer")
    k: int = Field(default=3, ge=1, le=20, description="Number of context chunks to use")


class ConfigOllamaRequest(BaseModel):
    url: str = Field(default="", description="Ollama server URL")
    embedModel: Optional[str] = Field(default=None, description="Embedding model name")
    genModel: Optional[str] = Field(default=None, description="Generation LLM name")


# =====================================================================
#  FASTAPI APPLICATION FACTORY
# =====================================================================

def create_app(db_path: Optional[str] = None) -> FastAPI:
    if db_path is None:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_data")
    chroma_client = None
    if CHROMA_AVAILABLE:
        try:
            os.makedirs(db_path, exist_ok=True)
            chroma_client = chromadb.PersistentClient(path=db_path)
        except Exception as e:
            print(f"Warning: Could not initialize ChromaDB PersistentClient: {e}")

    db = VectorDB(DIMS, chroma_client=chroma_client)
    doc_db = DocumentDB(chroma_client=chroma_client)
    ollama = OllamaClient()
    linear_rag = LinearRAG(doc_db=doc_db, ollama=ollama)

    load_demo(db)

    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    html_content = "<h1>VectorDB UI Not Found</h1>"
    if os.path.exists(html_path):
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        except Exception:
            pass

    app = FastAPI(
        title="VectorDB & Linear RAG API",
        description="High-Performance Vector Database with HNSW, ChromaDB persistence, and Deterministic Linear RAG pipeline.",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach state handles
    app.state.db = db
    app.state.doc_db = doc_db
    app.state.ollama = ollama
    app.state.rag = linear_rag
    app.state.agentic_rag = linear_rag  # Backward compatibility
    app.state.html_content = html_content

    # ── Web UI ──
    @app.get("/", response_class=HTMLResponse, summary="Serve Web UI Visualizer")
    async def serve_index():
        return HTMLResponse(content=app.state.html_content)

    # ── Demo Vector Endpoints ──
    @app.get("/items", summary="List All Demo Vectors")
    async def list_items():
        return app.state.db.all()

    @app.get("/search", summary="Search Nearest Vectors")
    async def search_vectors(
        v: str = Query(..., description="Comma-separated 16D vector values"),
        k: int = Query(default=5, ge=1, le=50, description="Top-k nearest neighbors"),
        metric: str = Query(default="cosine", description="Distance metric (cosine, euclidean, manhattan)"),
        algo: str = Query(default="hnsw", description="Algorithm (chromadb, hnsw, kdtree, bruteforce)"),
    ):
        try:
            v_str = v.strip()
            if v_str.startswith("[") and v_str.endswith("]"):
                import json
                q = [float(x) for x in json.loads(v_str)]
            else:
                q = [float(x) for x in v_str.split(",") if x.strip()]
        except Exception:
            q = []

        if len(q) != DIMS:
            raise HTTPException(status_code=400, detail=f"need {DIMS}D vector")

        return app.state.db.search(q, k, metric, algo)

    @app.get("/benchmark", summary="Benchmark All Vector Search Engines")
    async def benchmark_engines(
        v: str = Query(..., description="Comma-separated 16D vector values"),
        k: int = Query(default=5, ge=1, le=50, description="Top-k results"),
        metric: str = Query(default="cosine", description="Distance metric"),
    ):
        try:
            v_str = v.strip()
            if v_str.startswith("[") and v_str.endswith("]"):
                import json
                q = [float(x) for x in json.loads(v_str)]
            else:
                q = [float(x) for x in v_str.split(",") if x.strip()]
        except Exception:
            q = []

        if len(q) != DIMS:
            raise HTTPException(status_code=400, detail=f"need {DIMS}D vector")

        return app.state.db.benchmark(q, k, metric)

    @app.get("/hnsw-info", summary="Get HNSW Graph Topology and Layers")
    async def get_hnsw_info():
        return app.state.db.hnsw_info()

    @app.get("/status", summary="Get System Status and Health")
    async def get_status():
        ollama_up = app.state.ollama.is_available()
        models = app.state.ollama.list_models() if ollama_up else []
        return {
            "ollamaAvailable": ollama_up,
            "ollamaUrl": app.state.ollama.base_url,
            "embedModel": app.state.ollama.embed_model,
            "genModel": app.state.ollama.gen_model,
            "models": models,
            "docCount": app.state.doc_db.size(),
            "docDims": app.state.doc_db.get_dims(),
            "demoDims": DIMS,
            "demoCount": app.state.db.size(),
            "chromaAvailable": CHROMA_AVAILABLE,
            "ragWorkflow": "Linear RAG",
            "vectorDbEngine": "ChromaDB (Persistent)" if CHROMA_AVAILABLE else "Pure-Python In-Memory",
            "backendFramework": "FastAPI (Async)",
            "persistencePath": "./chroma_data",
        }

    @app.get("/stats", summary="Get Vector Database Statistics")
    async def get_stats():
        return {
            "count": app.state.db.size(),
            "dims": DIMS,
            "algorithms": ["chromadb", "hnsw", "kdtree", "bruteforce"] if CHROMA_AVAILABLE else ["hnsw", "kdtree", "bruteforce"],
            "metrics": ["euclidean", "cosine", "manhattan"],
            "vectorDbEngine": "ChromaDB" if CHROMA_AVAILABLE else "Custom Python",
            "ragWorkflow": "Linear RAG",
            "backendFramework": "FastAPI",
            "persistencePath": "./chroma_data",
        }

    @app.post("/config/ollama", summary="Update Ollama Server Configuration")
    async def configure_ollama(req: ConfigOllamaRequest):
        url = req.url.strip()
        if url:
            app.state.ollama.set_url(url, req.embedModel, req.genModel)

        available = app.state.ollama.is_available()
        models = app.state.ollama.list_models() if available else []

        return {
            "ok": True,
            "available": available,
            "url": app.state.ollama.base_url,
            "embedModel": app.state.ollama.embed_model,
            "genModel": app.state.ollama.gen_model,
            "models": models,
        }

    @app.post("/insert", summary="Insert Custom 16D Vector")
    async def insert_vector(req: InsertVectorRequest):
        if not req.metadata or len(req.embedding) != DIMS:
            raise HTTPException(status_code=400, detail="invalid body or dimensions")

        dist_fn = get_dist_fn("cosine")
        new_id = app.state.db.insert(req.metadata, req.category, req.embedding, dist_fn)
        return {"id": new_id}

    @app.delete("/delete/{item_id}", summary="Delete 16D Vector by ID")
    async def delete_vector(item_id: int = Path(..., description="ID of vector to delete")):
        ok = app.state.db.remove(item_id)
        return {"ok": ok}

    # ── Document & RAG Endpoints ──
    @app.get("/doc/list", summary="List All Persisted Documents")
    async def list_documents():
        return app.state.doc_db.all()

    @app.post("/doc/insert", summary="Chunk, Embed, and Insert Document")
    async def insert_doc(req: InsertDocRequest):
        title = req.title.strip()
        text = req.text.strip()
        if not title or not text:
            raise HTTPException(status_code=400, detail="need title and text")

        chunks = chunk_text(text, 250, 30)
        ids = []

        for i, chunk in enumerate(chunks):
            chunk_title = f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title
            text_to_embed = f"Title: {title}\n{chunk}"
            emb = app.state.ollama.embed(text_to_embed)
            if not emb:
                raise HTTPException(
                    status_code=503,
                    detail="Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2",
                )

            chunk_id = app.state.doc_db.insert(chunk_title, chunk, emb)
            ids.append(chunk_id)

        return {
            "ids": ids,
            "chunks": len(chunks),
            "dims": app.state.doc_db.get_dims(),
        }

    @app.post("/doc/search", summary="Search Documents by Semantic Vector")
    async def search_doc(req: SearchDocRequest):
        question = req.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="need question")

        q_emb = app.state.ollama.embed(question)
        if not q_emb:
            raise HTTPException(status_code=503, detail="Ollama unavailable")

        hits = app.state.doc_db.search(q_emb, req.k)
        contexts = [
            {
                "id": item.id,
                "title": item.title,
                "distance": round(dist, 4),
            }
            for dist, item in hits
        ]
        return {"contexts": contexts}

    @app.post("/doc/ask", summary="Ask RAG Pipeline (Deterministic Linear RAG Flow)")
    async def ask_doc(req: AskDocRequest):
        question = req.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="need question")

        res = app.state.rag.run(question, req.k)
        if res.get("error"):
            raise HTTPException(status_code=503, detail=res["error"])

        return res

    @app.delete("/doc/delete/{doc_id}", summary="Delete Document by ID")
    async def delete_document(doc_id: int = Path(..., description="ID of document to delete")):
        ok = app.state.doc_db.remove(doc_id)
        return {"ok": ok}

    return app


# =====================================================================
#  SERVER ENTRY POINT
# =====================================================================

app = create_app()


def main() -> None:
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"

    ollama = app.state.ollama
    db = app.state.db

    ollama_up = ollama.is_available()
    print("=== VectorDB Engine (FastAPI + ChromaDB + HNSW + Linear RAG) ===")
    print(f"Backend Server: FastAPI + Uvicorn on http://localhost:{port}")
    print(f"Interactive API Docs: http://localhost:{port}/docs")
    print(f"Vector Database: {'ChromaDB (Persistent ./chroma_data)' if CHROMA_AVAILABLE else 'Pure Python (In-Memory)'}")
    print(f"{db.size()} demo vectors | {DIMS} dims | ChromaDB+HNSW+KD-Tree+BruteForce")
    print("RAG Engine: Linear RAG (Deterministic Vector Context Injection)")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

