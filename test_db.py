"""
Unit and integration tests for VectorDB with ChromaDB + HNSW.
Tests all search algorithms (ChromaDB, HNSW, KD-Tree, BruteForce), distance metrics,
chunking, document store, persistence, and HTTP REST endpoints.
"""

import json
import math
import os
import shutil
import threading
import time
import unittest
import urllib.request
import urllib.parse

from fastapi.testclient import TestClient

from main import (
    DIMS,
    CHROMA_AVAILABLE,
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
    OllamaClient,
    LinearRAG,
    chunk_text,
    create_app,
    load_demo,
)

if CHROMA_AVAILABLE:
    import chromadb


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
        algos = ["hnsw", "kdtree", "bruteforce"]
        for algo in algos:
            res = db.search(cs_query, 3, "cosine", algo)
            self.assertEqual(len(res["results"]), 3)
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


import tempfile

class TestChromaDBIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="chroma_unit_")
        if CHROMA_AVAILABLE:
            self.client = chromadb.PersistentClient(path=self.test_dir)
        else:
            self.client = None

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_chroma_vector_db_search_and_bench(self):
        if not CHROMA_AVAILABLE:
            self.skipTest("chromadb not installed")

        db = VectorDB(dims=DIMS, chroma_client=self.client)
        load_demo(db)
        self.assertEqual(db.size(), 20)

        cs_query = [0.9, 0.8, 0.7, 0.6, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        res = db.search(cs_query, 3, "cosine", "chromadb")
        self.assertEqual(len(res["results"]), 3)
        self.assertEqual(res["results"][0]["category"], "cs")
        self.assertIn("latencyUs", res)

        bench = db.benchmark(cs_query, 5, "cosine")
        self.assertIn("chromadbUs", bench)
        self.assertGreaterEqual(bench["chromadbUs"], 0)

    def test_chroma_document_persistence(self):
        if not CHROMA_AVAILABLE:
            self.skipTest("chromadb not installed")

        doc_db1 = DocumentDB(chroma_client=self.client)
        emb1 = [0.9, 0.1, 0.0, 0.0]
        emb2 = [0.0, 0.0, 0.9, 0.1]

        id1 = doc_db1.insert("Doc Algorithms", "Algorithms content", emb1)
        id2 = doc_db1.insert("Doc Cooking", "Cooking content", emb2)
        self.assertEqual(doc_db1.size(), 2)

        # Search with doc_db1
        hits = doc_db1.search([0.85, 0.15, 0.0, 0.0], k=1, max_dist=0.7)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][1].id, id1)

        # Simulate server restart by creating doc_db2 connected to same persistent storage
        doc_db2 = DocumentDB(chroma_client=self.client)
        self.assertEqual(doc_db2.size(), 2)
        hits2 = doc_db2.search([0.85, 0.15, 0.0, 0.0], k=1, max_dist=0.7)
        self.assertEqual(len(hits2), 1)
        self.assertEqual(hits2[0][1].id, id1)


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


class TestFastAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.test_data = tempfile.mkdtemp(prefix="chroma_fastapi_")
        cls.app = create_app(db_path=cls.test_data)
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_data):
            shutil.rmtree(cls.test_data, ignore_errors=True)

    def test_get_index_html(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("VectorDB", resp.text)
        self.assertIn("FASTAPI", resp.text)

    def test_swagger_docs(self):
        resp = self.client.get("/docs")
        self.assertEqual(resp.status_code, 200)
        resp_schema = self.client.get("/openapi.json")
        self.assertEqual(resp_schema.status_code, 200)
        self.assertIn("VectorDB & Linear RAG API", resp_schema.json()["info"]["title"])

    def test_get_items(self):
        resp = self.client.get("/items")
        self.assertEqual(resp.status_code, 200)
        items = resp.json()
        self.assertEqual(len(items), 20)

    def test_get_stats(self):
        resp = self.client.get("/stats")
        self.assertEqual(resp.status_code, 200)
        stats = resp.json()
        self.assertEqual(stats["count"], 20)
        self.assertEqual(stats["dims"], 16)
        self.assertEqual(stats["backendFramework"], "FastAPI")

    def test_get_status(self):
        resp = self.client.get("/status")
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertIn("ollamaAvailable", res)
        self.assertEqual(res["demoCount"], 20)
        self.assertIn("FastAPI", res["backendFramework"])

    def test_get_search_hnsw_and_chroma(self):
        v = ",".join(["0.1"] * 16)
        resp = self.client.get(f"/search?v={v}&k=3&metric=cosine&algo=hnsw")
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertEqual(len(res["results"]), 3)

        if CHROMA_AVAILABLE:
            resp_c = self.client.get(f"/search?v={v}&k=3&metric=cosine&algo=chromadb")
            self.assertEqual(resp_c.status_code, 200)
            res_c = resp_c.json()
            self.assertEqual(len(res_c["results"]), 3)
            self.assertEqual(res_c["algo"], "chromadb")

    def test_get_benchmark(self):
        v = ",".join(["0.1"] * 16)
        resp = self.client.get(f"/benchmark?v={v}&k=3&metric=cosine")
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertIn("hnswUs", res)

    def test_get_hnsw_info(self):
        resp = self.client.get("/hnsw-info")
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertIn("nodes", res)
        self.assertIn("edges", res)

    def test_insert_and_delete_vector(self):
        payload = {
            "metadata": "Custom FastAPI item",
            "category": "cs",
            "embedding": [0.5] * 16,
        }
        resp = self.client.post("/insert", json=payload)
        self.assertEqual(resp.status_code, 200)
        item_id = resp.json()["id"]

        del_resp = self.client.delete(f"/delete/{item_id}")
        self.assertEqual(del_resp.status_code, 200)
        self.assertTrue(del_resp.json()["ok"])

    def test_status_endpoint_reports_langgraph(self):
        resp = self.client.get("/status")
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertIn("ragWorkflow", res)
        self.assertEqual(res["ragWorkflow"], "Linear RAG")
        self.assertIn("vectorDbEngine", res)


class TestLinearRAG(unittest.TestCase):
    def setUp(self):
        class MockOllamaClient:
            def __init__(self):
                self.embed_model = "nomic-embed-text"
                self.gen_model = "llama3.2"
                self.base_url = "http://mock-ollama:11434"

            def embed(self, text: str):
                t = text.lower()
                if "python" in t or "programming" in t or "code" in t:
                    return [0.9, 0.8, 0.1, 0.1]
                if "pizza" in t or "recipe" in t or "food" in t:
                    return [0.1, 0.1, 0.9, 0.8]
                return [0.5, 0.5, 0.5, 0.5]

            def generate(self, prompt: str):
                return "Python is a high-level, general-purpose programming language."

        self.mock_ollama = MockOllamaClient()
        self.doc_db = DocumentDB()
        self.doc_db.insert(
            "Python Overview",
            "Python is a high-level interpreted programming language created by Guido van Rossum.",
            [0.9, 0.8, 0.1, 0.1]
        )
        self.doc_db.insert(
            "Pizza Recipe",
            "Neapolitan pizza dough requires flour, water, yeast, and salt with San Marzano tomatoes.",
            [0.1, 0.1, 0.9, 0.8]
        )

    def test_linear_rag_initialization(self):
        rag = LinearRAG(self.doc_db, self.mock_ollama)  # type: ignore
        self.assertIsNotNone(rag.doc_db)
        self.assertIsNotNone(rag.ollama)

    def test_linear_rag_execution(self):
        rag = LinearRAG(self.doc_db, self.mock_ollama)  # type: ignore
        res = rag.run("Tell me about Python programming", k=2)
        self.assertIn("answer", res)
        self.assertEqual(res.get("workflow"), "Linear RAG")
        self.assertIn("steps", res)
        self.assertGreaterEqual(len(res["steps"]), 2)

        # Check step actions
        step_names = [s.get("step") for s in res["steps"]]
        self.assertIn("retrieve", step_names)
        self.assertIn("generate", step_names)

        # Contexts returned
        self.assertGreaterEqual(len(res["contexts"]), 1)
        self.assertEqual(res["contexts"][0]["title"], "Python Overview")


if __name__ == "__main__":
    unittest.main()
