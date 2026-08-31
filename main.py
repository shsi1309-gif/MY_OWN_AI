"""
VectorDB — Pure Python Vector Database from Scratch with HNSW, KD-Tree, Brute Force & RAG.
Zero external dependencies required (uses Python 3.8+ standard library).
"""

import http.server
import json
import math
import os
import random
import re
import socketserver
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import heapq
from typing import Callable, Dict, List, Optional, Tuple, Any

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
    def __init__(self, dims: int = DIMS):
        self.dims = dims
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.mu = threading.Lock()
        self.next_id = 1

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
            return v.id

    def remove(self, item_id: int) -> bool:
        with self.mu:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.bf.remove(item_id)
            self.hnsw.remove(item_id)
            self.kdt.rebuild(list(self.store.values()))
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

            return {
                "bruteforceUs": bf_us,
                "kdtreeUs": kd_us,
                "hnswUs": hnsw_us,
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
        host: str = "127.0.0.1",
        port: int = 11434,
        embed_model: str = "nomic-embed-text",
        gen_model: str = "llama3.2",
    ):
        self.host = host
        self.port = port
        self.embed_model = embed_model
        self.gen_model = gen_model

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
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
        return "ERROR: Ollama unavailable. Run: ollama serve"


# =====================================================================
#  DOCUMENT DATABASE (HNSW over real Ollama embeddings)
# =====================================================================

class DocumentDB:
    def __init__(self):
        self.store: Dict[int, DocItem] = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.mu = threading.Lock()
        self.next_id = 1
        self.dims = 0

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
            return doc_id

    def search(
        self, q: List[float], k: int, max_dist: float = 0.7
    ) -> List[Tuple[float, DocItem]]:
        with self.mu:
            if not self.store:
                return []
            raw = (
                self.bf.knn(q, k, cosine)
                if len(self.store) < 10
                else self.hnsw.knn(q, k, 50, cosine)
            )
            out = []
            for d, item_id in raw:
                if item_id in self.store and d <= max_dist:
                    out.append((d, self.store[item_id]))
            return out

    def remove(self, doc_id: int) -> bool:
        with self.mu:
            if doc_id not in self.store:
                return False
            del self.store[doc_id]
            self.hnsw.remove(doc_id)
            self.bf.remove(doc_id)
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
#  HTTP REQUEST HANDLER
# =====================================================================

class VectorDBHandler(http.server.BaseHTTPRequestHandler):
    db: VectorDB
    doc_db: DocumentDB
    ollama: OllamaClient
    html_content: bytes = b""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy standard HTTP access logs
        pass

    def send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def send_json(self, data: Any, status_code: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query_params = urllib.parse.parse_qs(parsed_url.query)

        # Serve index.html
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.html_content)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(self.html_content)
            return

        # GET /items
        if path == "/items":
            self.send_json(self.db.all())
            return

        # GET /search?v=...&k=5&metric=cosine&algo=hnsw
        if path == "/search":
            v_str = query_params.get("v", [""])[0]
            try:
                q = [float(x) for x in v_str.split(",") if x.strip()]
            except ValueError:
                q = []

            if len(q) != DIMS:
                self.send_json({"error": f"need {DIMS}D vector"}, 400)
                return

            try:
                k = int(query_params.get("k", [5])[0])
            except (ValueError, TypeError):
                k = 5

            metric = query_params.get("metric", ["cosine"])[0] or "cosine"
            algo = query_params.get("algo", ["hnsw"])[0] or "hnsw"

            res = self.db.search(q, k, metric, algo)
            self.send_json(res)
            return

        # GET /benchmark?v=...&k=5&metric=cosine
        if path == "/benchmark":
            v_str = query_params.get("v", [""])[0]
            try:
                q = [float(x) for x in v_str.split(",") if x.strip()]
            except ValueError:
                q = []

            if len(q) != DIMS:
                self.send_json({"error": f"need {DIMS}D vector"}, 400)
                return

            try:
                k = int(query_params.get("k", [5])[0])
            except (ValueError, TypeError):
                k = 5

            metric = query_params.get("metric", ["cosine"])[0] or "cosine"
            res = self.db.benchmark(q, k, metric)
            self.send_json(res)
            return

        # GET /hnsw-info
        if path == "/hnsw-info":
            self.send_json(self.db.hnsw_info())
            return

        # GET /status
        if path == "/status":
            ollama_up = self.ollama.is_available()
            self.send_json({
                "ollamaAvailable": ollama_up,
                "embedModel": self.ollama.embed_model,
                "genModel": self.ollama.gen_model,
                "docCount": self.doc_db.size(),
                "docDims": self.doc_db.get_dims(),
                "demoDims": DIMS,
                "demoCount": self.db.size(),
            })
            return

        # GET /stats
        if path == "/stats":
            self.send_json({
                "count": self.db.size(),
                "dims": DIMS,
                "algorithms": ["bruteforce", "kdtree", "hnsw"],
                "metrics": ["euclidean", "cosine", "manhattan"],
            })
            return

        # GET /doc/list
        if path == "/doc/list":
            self.send_json(self.doc_db.all())
            return

        self.send_json({"error": "Not Found"}, 404)

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            body_json = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            body_json = {}

        # POST /insert
        if path == "/insert":
            meta = body_json.get("metadata", "")
            cat = body_json.get("category", "")
            emb = body_json.get("embedding", [])

            if not meta or not isinstance(emb, list) or len(emb) != DIMS:
                self.send_json({"error": "invalid body"}, 400)
                return

            dist_fn = get_dist_fn("cosine")
            new_id = self.db.insert(meta, cat, emb, dist_fn)
            self.send_json({"id": new_id})
            return

        # POST /doc/insert
        if path == "/doc/insert":
            title = body_json.get("title", "").strip()
            text = body_json.get("text", "").strip()

            if not title or not text:
                self.send_json({"error": "need title and text"}, 400)
                return

            chunks = chunk_text(text, 250, 30)
            ids = []

            for i, chunk in enumerate(chunks):
                emb = self.ollama.embed(chunk)
                if not emb:
                    self.send_json(
                        {
                            "error": (
                                "Ollama unavailable. "
                                "Install from https://ollama.com then run: "
                                "ollama pull nomic-embed-text && ollama pull llama3.2"
                            )
                        },
                        503,
                    )
                    return

                chunk_title = (
                    f"{title} [{i+1}/{len(chunks)}]"
                    if len(chunks) > 1
                    else title
                )
                chunk_id = self.doc_db.insert(chunk_title, chunk, emb)
                ids.append(chunk_id)

            self.send_json({
                "ids": ids,
                "chunks": len(chunks),
                "dims": self.doc_db.get_dims(),
            })
            return

        # POST /doc/search
        if path == "/doc/search":
            question = body_json.get("question", "").strip()
            k = body_json.get("k", 3)
            try:
                k = int(k)
            except (ValueError, TypeError):
                k = 3

            if not question:
                self.send_json({"error": "need question"}, 400)
                return

            q_emb = self.ollama.embed(question)
            if not q_emb:
                self.send_json({"error": "Ollama unavailable"}, 503)
                return

            hits = self.doc_db.search(q_emb, k)
            contexts = [
                {
                    "id": item.id,
                    "title": item.title,
                    "distance": round(dist, 4),
                }
                for dist, item in hits
            ]
            self.send_json({"contexts": contexts})
            return

        # POST /doc/ask
        if path == "/doc/ask":
            question = body_json.get("question", "").strip()
            k = body_json.get("k", 3)
            try:
                k = int(k)
            except (ValueError, TypeError):
                k = 3

            if not question:
                self.send_json({"error": "need question"}, 400)
                return

            # Step 1: Embed question
            q_emb = self.ollama.embed(question)
            if not q_emb:
                self.send_json({"error": "Ollama unavailable"}, 503)
                return

            # Step 2: Retrieve top-k chunks
            hits = self.doc_db.search(q_emb, k)

            # Step 3: Build prompt
            ctx_str = ""
            for i, (dist, item) in enumerate(hits):
                ctx_str += f"[{i+1}] {item.title}:\n{item.text}\n\n"

            prompt = (
                "You are a helpful assistant. Answer the user's question directly. "
                "Use the provided context if it contains relevant information. "
                "If it doesn't, just use your own general knowledge. "
                "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
                "Just answer the question naturally.\n\n"
                f"Context:\n{ctx_str}"
                f"Question: {question}\n\n"
                "Answer:"
            )

            # Step 4: Generate answer
            answer = self.ollama.generate(prompt)

            # Step 5: Return response
            contexts = [
                {
                    "id": item.id,
                    "title": item.title,
                    "text": item.text,
                    "distance": round(dist, 4),
                }
                for dist, item in hits
            ]

            self.send_json({
                "answer": answer,
                "model": self.ollama.gen_model,
                "contexts": contexts,
                "docCount": self.doc_db.size(),
            })
            return

        self.send_json({"error": "Not Found"}, 404)

    def do_DELETE(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # DELETE /delete/<id>
        m_del = re.match(r"^/delete/(\d+)$", path)
        if m_del:
            item_id = int(m_del.group(1))
            ok = self.db.remove(item_id)
            self.send_json({"ok": ok})
            return

        # DELETE /doc/delete/<id>
        m_doc_del = re.match(r"^/doc/delete/(\d+)$", path)
        if m_doc_del:
            doc_id = int(m_doc_del.group(1))
            ok = self.doc_db.remove(doc_id)
            self.send_json({"ok": ok})
            return

        self.send_json({"error": "Not Found"}, 404)


# =====================================================================
#  SERVER ENTRY POINT
# =====================================================================

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def create_server(host: str = "0.0.0.0", port: int = 8080) -> ThreadedHTTPServer:
    db = VectorDB(DIMS)
    doc_db = DocumentDB()
    ollama = OllamaClient()

    load_demo(db)

    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    html_content = b"<h1>VectorDB UI Not Found</h1>"
    if os.path.exists(html_path):
        with open(html_path, "rb") as f:
            html_content = f.read()

    VectorDBHandler.db = db
    VectorDBHandler.doc_db = doc_db
    VectorDBHandler.ollama = ollama
    VectorDBHandler.html_content = html_content

    return ThreadedHTTPServer((host, port), VectorDBHandler)


def main() -> None:
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"

    server = create_server(host, port)
    ollama = VectorDBHandler.ollama
    db = VectorDBHandler.db

    ollama_up = ollama.is_available()
    print("=== VectorDB Engine (Python) ===")
    print(f"http://localhost:{port}")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
