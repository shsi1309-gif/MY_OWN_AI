"""
Unit and integration tests for Python VectorDB.
Tests all search algorithms (HNSW, KD-Tree, BruteForce), distance metrics,
chunking, document store, and HTTP REST endpoints.
"""

import json
import math
import threading
import time
import unittest
import urllib.request
import urllib.parse
from main import (
    DIMS,
    VectorItem,
    DocItem,
    euclidean,
    cosine,
    manhattan,
    get_dist_fn,
    BruteForce,
    KDTree,
    HNSW,
    VectorDB,
    DocumentDB,
    chunk_text,
    create_server,
    load_demo,
)


class TestDistanceMetrics(unittest.TestCase):
    def test_euclidean(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 6.0, 3.0]
        # sqrt((3)^2 + (4)^2 + 0) = 5.0
        self.assertAlmostEqual(euclidean(a, b), 5.0, places=5)

    def test_cosine(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        # orthogonal vectors -> dot=0, cosine distance = 1.0 - 0 = 1.0
        self.assertAlmostEqual(cosine(a, b), 1.0, places=5)

        c = [2.0, 0.0, 0.0]
        # parallel vectors -> dot=2, len(a)=1, len(c)=2, cos_sim=1.0, dist=0.0
        self.assertAlmostEqual(cosine(a, c), 0.0, places=5)

    def test_manhattan(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 6.0, 3.0]
        # |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7.0
        self.assertAlmostEqual(manhattan(a, b), 7.0, places=5)


class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.items = [
            VectorItem(1, "Item A", "cat1", [1.0, 0.0, 0.0, 0.0]),
            VectorItem(2, "Item B", "cat1", [0.9, 0.1, 0.0, 0.0]),
            VectorItem(3, "Item C", "cat2", [0.0, 1.0, 0.0, 0.0]),
            VectorItem(4, "Item D", "cat2", [0.0, 0.9, 0.1, 0.0]),
            VectorItem(5, "Item E", "cat3", [0.0, 0.0, 1.0, 0.0]),
        ]
        self.query = [0.95, 0.05, 0.0, 0.0]

    def test_bruteforce(self):
        bf = BruteForce()
        for item in self.items:
            bf.insert(item)

        results = bf.knn(self.query, 2, cosine)
        self.assertEqual(len(results), 2)
        # Item 1 and Item 2 should be the closest to query
        ids = [item_id for _, item_id in results]
        self.assertEqual(ids, [1, 2])

    def test_kdtree(self):
        kdt = KDTree(4)
        for item in self.items:
            kdt.insert(item)

        results = kdt.knn(self.query, 2, cosine)
        self.assertEqual(len(results), 2)
        ids = [item_id for _, item_id in results]
        self.assertEqual(ids, [1, 2])

    def test_hnsw(self):
        hnsw = HNSW(m=4, ef_build=50)
        for item in self.items:
            hnsw.insert(item, cosine)

        results = hnsw.knn(self.query, 2, ef=20, dist_fn=cosine)
        self.assertEqual(len(results), 2)
        ids = [item_id for _, item_id in results]
        self.assertEqual(ids, [1, 2])

        info = hnsw.get_info()
        self.assertEqual(info["nodeCount"], 5)
        self.assertGreaterEqual(info["topLayer"], 0)

    def test_hnsw_removal(self):
        hnsw = HNSW(m=4, ef_build=50)
        for item in self.items:
            hnsw.insert(item, cosine)

        self.assertEqual(hnsw.size(), 5)
        hnsw.remove(1)
        self.assertEqual(hnsw.size(), 4)

        results = hnsw.knn(self.query, 1, ef=20, dist_fn=cosine)
        self.assertEqual(results[0][1], 2)


class TestVectorDB(unittest.TestCase):
    def test_demo_vectors_and_search(self):
        db = VectorDB(DIMS)
        load_demo(db)
        self.assertEqual(db.size(), 20)

        # CS query vector
        cs_query = [0.9, 0.8, 0.7, 0.6, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

        # Test search with HNSW, KDTree, BruteForce
        for algo in ["hnsw", "kdtree", "bruteforce"]:
            res = db.search(cs_query, 3, "cosine", algo)
            self.assertEqual(len(res["results"]), 3)
            # Top results should belong to 'cs' category
            self.assertEqual(res["results"][0]["category"], "cs")

        # Test benchmark
        bench = db.benchmark(cs_query, 5, "cosine")
        self.assertEqual(bench["itemCount"], 20)
        self.assertIn("hnswUs", bench)
        self.assertIn("kdtreeUs", bench)
        self.assertIn("bruteforceUs", bench)

        # Test insert and remove
        new_id = db.insert("Test item", "test_cat", [0.1] * DIMS, cosine)
        self.assertEqual(db.size(), 21)
        ok = db.remove(new_id)
        self.assertTrue(ok)
        self.assertEqual(db.size(), 20)


class TestTextChunkerAndDocumentDB(unittest.TestCase):
    def test_chunk_text(self):
        words = ["word" + str(i) for i in range(550)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_words=250, overlap_words=30)
        self.assertGreaterEqual(len(chunks), 3)

    def test_document_db(self):
        doc_db = DocumentDB()
        emb1 = [0.9, 0.1, 0.0, 0.0]
        emb2 = [0.0, 0.0, 0.9, 0.1]

        id1 = doc_db.insert("Doc 1", "Algorithms content", emb1)
        id2 = doc_db.insert("Doc 2", "Cooking content", emb2)

        self.assertEqual(doc_db.size(), 2)
        self.assertEqual(doc_db.get_dims(), 4)

        # Search for algorithms
        hits = doc_db.search([0.85, 0.15, 0.0, 0.0], k=1, max_dist=0.7)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][1].id, id1)

        # Delete
        self.assertTrue(doc_db.remove(id1))
        self.assertEqual(doc_db.size(), 1)


class TestHttpServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_port = 8899
        cls.server = create_server("127.0.0.1", cls.test_port)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path):
        req = urllib.request.Request(f"http://127.0.0.1:{self.test_port}{path}")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.headers, resp.read()

    def _post(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.test_port}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.headers, resp.read()

    def _delete(self, path):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.test_port}{path}", method="DELETE"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.headers, resp.read()

    def test_get_index_html(self):
        status, headers, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"VectorDB", body)

    def test_get_items(self):
        status, _, body = self._get("/items")
        self.assertEqual(status, 200)
        items = json.loads(body.decode("utf-8"))
        self.assertEqual(len(items), 20)

    def test_get_stats(self):
        status, _, body = self._get("/stats")
        self.assertEqual(status, 200)
        stats = json.loads(body.decode("utf-8"))
        self.assertEqual(stats["count"], 20)
        self.assertEqual(stats["dims"], 16)

    def test_get_status(self):
        status, _, body = self._get("/status")
        self.assertEqual(status, 200)
        res = json.loads(body.decode("utf-8"))
        self.assertIn("ollamaAvailable", res)
        self.assertEqual(res["demoCount"], 20)

    def test_get_search(self):
        v = ",".join(["0.1"] * 16)
        status, _, body = self._get(f"/search?v={v}&k=3&metric=cosine&algo=hnsw")
        self.assertEqual(status, 200)
        res = json.loads(body.decode("utf-8"))
        self.assertEqual(len(res["results"]), 3)

    def test_get_benchmark(self):
        v = ",".join(["0.1"] * 16)
        status, _, body = self._get(f"/benchmark?v={v}&k=3&metric=cosine")
        self.assertEqual(status, 200)
        res = json.loads(body.decode("utf-8"))
        self.assertIn("hnswUs", res)

    def test_get_hnsw_info(self):
        status, _, body = self._get("/hnsw-info")
        self.assertEqual(status, 200)
        res = json.loads(body.decode("utf-8"))
        self.assertIn("nodes", res)
        self.assertIn("edges", res)

    def test_insert_and_delete_vector(self):
        # Insert
        payload = {
            "metadata": "Custom item",
            "category": "cs",
            "embedding": [0.5] * 16,
        }
        status, _, body = self._post("/insert", payload)
        self.assertEqual(status, 200)
        res = json.loads(body.decode("utf-8"))
        item_id = res["id"]

        # Delete
        status, _, body = self._delete(f"/delete/{item_id}")
        self.assertEqual(status, 200)
        res = json.loads(body.decode("utf-8"))
        self.assertTrue(res["ok"])


if __name__ == "__main__":
    unittest.main()
