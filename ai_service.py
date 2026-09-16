"""
AI/ML Core Microservice
Runs Python FastAPI with ChromaDB, HNSW, KD-Tree, BruteForce, LangGraph Agentic RAG, and Ollama.
Default Port: 8000
"""

import os
import uvicorn
from main import create_app, CHROMA_AVAILABLE, LANGGRAPH_AVAILABLE, DIMS

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"

    ollama = app.state.ollama
    db = app.state.db

    ollama_up = ollama.is_available()
    print("=== Python AI/ML Microservice (FastAPI + ChromaDB + HNSW + LangGraph) ===")
    print(f"🧠 AI Service running at http://localhost:{port}")
    print(f"📖 Microservice Docs at http://localhost:{port}/docs")
    print(f"Vector Database: {'ChromaDB (Persistent ./chroma_data)' if CHROMA_AVAILABLE else 'Pure Python (In-Memory)'}")
    print(f"{db.size()} demo vectors | {DIMS} dims | ChromaDB+HNSW+KD-Tree+BruteForce")
    print(f"RAG Engine: {'LangGraph (Self-Correcting CRAG)' if LANGGRAPH_AVAILABLE else 'Linear RAG (Fallback)'}")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    uvicorn.run(app, host=host, port=port)
