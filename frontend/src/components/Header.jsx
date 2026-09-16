import React from 'react';
import { ExternalLink, Database, Cpu, Server, Sparkles, Activity } from 'lucide-react';

export default function Header({ systemHealth, ollamaStatus, vectorCount, dimension = 16 }) {
  const isPythonOk = systemHealth?.status === 'ok' || systemHealth?.python_ai === 'ok';
  const isOllamaOk = ollamaStatus?.available;

  return (
    <header>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Sparkles size={20} color="#00d9ff" />
        <h1>MY AI</h1>
      </div>

      <span className="badge tier">
        <Server size={11} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
        3-TIER ENTERPRISE
      </span>

      <span className="badge ok" title="Node.js Express API Gateway on Port 8080">
        <Activity size={11} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
        Node.js: 8080
      </span>

      <span className={`badge ${isPythonOk ? 'ok' : 'err'}`} title="Python FastAPI + LangGraph AI Service on Port 8000">
        <Cpu size={11} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
        Python AI: 8000 {isPythonOk ? '●' : '○'}
      </span>

      <span className={`badge ${isOllamaOk ? 'ok' : 'err'}`} title={`Ollama Embeddings & LLM Service on Port 11434 (${ollamaStatus?.model || 'nomic-embed-text'})`}>
        <Database size={11} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
        Ollama: 11434 {isOllamaOk ? '●' : '○'}
      </span>

      <span className="badge hl" title="Self-Correcting Corrective RAG via LangGraph StateGraph">
        LangGraph CRAG
      </span>

      <span className="badge hl" title="ChromaDB Persistent Vector Store">
        ChromaDB
      </span>

      <div className="stats-label" style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <span>
          <strong style={{ color: 'var(--cs)' }}>{vectorCount ?? '...'}</strong> vectors ({dimension}D)
        </span>
        <a
          href="/docs"
          target="_blank"
          rel="noreferrer"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
            color: 'var(--accent)',
            textDecoration: 'none',
            fontSize: '11px',
            fontWeight: 600,
            padding: '3px 8px',
            background: 'rgba(108, 99, 255, 0.12)',
            borderRadius: '4px',
            border: '1px solid rgba(108, 99, 255, 0.3)'
          }}
          title="Explore interactive OpenAPI Swagger Documentation"
        >
          API Docs <ExternalLink size={12} />
        </a>
      </div>
    </header>
  );
}
