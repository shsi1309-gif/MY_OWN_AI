# MY AI — Vector Database & Linear RAG Platform

An enterprise-grade, high-performance Vector Database and Deterministic Retrieval-Augmented Generation (RAG) platform powered by **FastAPI**, **ChromaDB**, **HNSW**, **Ollama**, and a **60 FPS HTML5 Canvas Visualizer**.

---

## 🏛️ 10-Point Technology Stack & Architecture Trade-Offs

The platform is engineered around 10 deliberate design decisions optimized for privacy, real-time performance, and deterministic reliability:

| # | Functional Area | Selected Technology | Primary Alternative | The Core Trade-off Made |
|:---:|---|---|---|---|
| **1** | **Web Framework** | **FastAPI (Async)** | Node.js / Flask | **Native Python AI Ecosystem & OpenAPI Docs**: High concurrency async I/O with automatic Swagger UI (`/docs`) vs. compiled runtime speed. |
| **2** | **Orchestration** | **Linear RAG Pipeline** | Cyclic Agentic Graphs | **Deterministic Speed & Single-Shot Latency**: Direct vector retrieval and context synthesis with zero multi-hop loop overhead vs. cyclic query rewrites. |
| **3** | **Search Algorithm** | **HNSW** | KD-Tree / Flat Index | **$O(\log N)$ Search Scalability**: Multi-layer navigable small-world graphs for sub-millisecond retrieval vs. higher memory footprint. |
| **4** | **Vector Storage** | **ChromaDB** | Pinecone / pgvector | **Zero-Config Embedded Persistence**: Local disk storage in `./chroma_data` with zero SaaS vendor lock-in vs. distributed cloud clustering. |
| **5** | **Embedding Model** | **nomic-embed-text (768D)** | OpenAI Embeddings | **8k Context & 100% On-Prem Privacy**: Zero token fees and strict privacy vs. managed cloud batch throughput. |
| **6** | **Generative LLM** | **llama3.2 (Ollama)** | GPT-4o / Claude 3.5 | **Local Edge Privacy & Zero Operating Cost**: Air-gapped deployment with local model weights vs. massive general world knowledge. |
| **7** | **Distance Metric** | **Cosine Distance** | Euclidean ($L_2$) | **Document-Length Invariance**: Normalizes varying text chunk lengths vs. raw unnormalized Euclidean distance computation. |
| **8** | **Text Chunking** | **Sliding Window (250/30)** | Fixed-sentence Split | **Semantic Boundary Preservation**: Overlapping 30-word windows prevent loss of context across splits vs. 15% duplicated vector storage. |
| **9** | **Projection** | **PCA (Power Iteration)** | t-SNE / UMAP | **$O(D)$ Real-Time Projection**: Allows live ad-hoc query vector projection in sub-milliseconds without recalculating non-linear manifolds. |
| **10** | **Frontend** | **Vanilla HTML5 Canvas** | React / Streamlit | **Direct 60 FPS GPU Rendering & Zero Build Step**: Direct `requestAnimationFrame` canvas rendering with zero compilation lag vs. declarative UI components. |

---

## 🧩 System Architecture

```mermaid
graph TD
    Client["User Browser / Client"] -->|HTTP / WebSocket| FastAPI["FastAPI + Uvicorn Server (Port 8080)"]
    FastAPI -->|/docs & /redoc| Swagger["OpenAPI Swagger UI & ReDoc"]
    FastAPI -->|GET /| UI["Vanilla HTML5 Canvas (60 FPS Power Iteration PCA)"]
    
    subgraph "Storage & Indexing Engine"
        FastAPI --> Chroma["ChromaDB (Persistent in ./chroma_data)"]
        FastAPI --> HNSW["HNSW Graph Indexer (Multi-layer Skip List)"]
        FastAPI --> KDT["KD-Tree (Space Partitioning)"]
        FastAPI --> BF["BruteForce (Exact O(N) Benchmark)"]
    end
    
    subgraph "Linear RAG Pipeline"
        FastAPI --> Embed["1. Embed Query (nomic-embed-text 768D)"]
        Embed --> Retrieve["2. Retrieve Top-K Context Chunks (ChromaDB / HNSW)"]
        Retrieve --> Prompt["3. Inject Grounded Context into Prompt"]
        Prompt --> Gen["4. Synthesize Answer (llama3.2)"]
        Gen --> Out["5. Verified Grounded Output"]
    end
    
    subgraph "Local / Remote AI Infrastructure"
        FastAPI -.->|Embeddings (768D)| OllamaEmbed["Ollama: nomic-embed-text"]
        FastAPI -.->|Generation (3B)| OllamaLLM["Ollama: llama3.2"]
    end
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.8+** (Python 3.10–3.14 fully supported)
- Optional: **Ollama** (for local embeddings and generation from [ollama.com](https://ollama.com))

### 2. Installation
```bash
# Clone repository and enter directory
cd myai

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start Server
```bash
./start.sh
```
Or directly with Python:
```bash
python3 main.py
```

### 4. Open in Browser
- **Interactive Web Visualizer:** [http://localhost:8080](http://localhost:8080)
- **Interactive OpenAPI Swagger Docs:** [http://localhost:8080/docs](http://localhost:8080/docs)
- **ReDoc API Documentation:** [http://localhost:8080/redoc](http://localhost:8080/redoc)

---

## 🧪 Automated Test Suite

Run the automated test suite (verifying HNSW, KD-Tree, BruteForce, ChromaDB persistence, Cosine/L2/L1 metrics, Linear RAG pipeline, and FastAPI endpoints):

```bash
./.venv/bin/python3 test_db.py
```

Expected output:
```text
........................
----------------------------------------------------------------------
Ran 24 tests in 0.173s

OK
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves 60 FPS HTML5 Canvas Visualizer SPA |
| `GET` | `/docs` | Interactive OpenAPI Swagger UI explorer |
| `GET` | `/status` | Telemetry: ChromaDB, Linear RAG, Ollama, vector counts |
| `GET` | `/items` | List all 16D demo vector embeddings |
| `GET` | `/search` | Search nearest neighbors (`v`, `k`, `metric`, `algo`) |
| `GET` | `/benchmark` | Benchmark ChromaDB vs HNSW vs KD-Tree vs BruteForce |
| `GET` | `/hnsw-info` | Inspect HNSW graph topology, nodes, and layer distribution |
| `POST` | `/insert` | Insert custom vector with metadata |
| `DELETE` | `/delete/{id}` | Delete vector from database by ID |
| `GET` | `/doc/list` | List all ingested knowledge documents |
| `POST` | `/doc/insert` | Chunk, embed with `nomic-embed-text`, and store in ChromaDB |
| `POST` | `/doc/ask` | Execute Deterministic Linear RAG pipeline |
| `DELETE` | `/doc/delete/{id}` | Delete document and associated chunks |

---

## 🛡️ License
MIT License. Built with ❤️ for high-performance Vector Search & AI retrieval workflows.
