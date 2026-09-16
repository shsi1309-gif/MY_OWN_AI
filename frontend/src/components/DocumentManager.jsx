import React, { useState } from 'react';
import { FileText, Plus, Trash2, CheckCircle2, AlertCircle, RefreshCw, Upload, Sparkles } from 'lucide-react';

const SAMPLE_TEMPLATES = {
  finance: {
    title: 'Financial Portfolio Theory & Valuation',
    category: 'math',
    content: `Modern Portfolio Theory (MPT) was introduced by Harry Markowitz. It maximizes expected portfolio return for a given level of risk by selecting proportion of diverse assets. The Capital Asset Pricing Model (CAPM) calculates expected return based on systematic risk (Beta). Discounted Cash Flow (DCF) values a business by projecting future free cash flows and discounting them back using Weighted Average Cost of Capital (WACC).`
  },
  ai: {
    title: 'Self-Correcting RAG & LangGraph Architecture',
    category: 'cs',
    content: `Corrective RAG (CRAG) evaluates retrieved documents using an agentic grader. If retrieved context is irrelevant, the query is rewritten and alternative embeddings are queried. Generation is grounded strictly on verified citations. A final hallucination check guardrail ensures the answer contains zero unverified claims before returning to the user.`
  },
  microservices: {
    title: 'Distributed Systems & Gateway Routing',
    category: 'cs',
    content: `An API Gateway serves as a single entry point for client applications. In our 3-tier architecture, Node.js Express acts as the reverse proxy gateway on port 8080 handling authentication, CORS, rate limiting, and Swagger API docs, while delegating high-throughput vector math to the Python AI service on port 8000.`
  }
};

export default function DocumentManager({
  documents = [],
  ollamaStatus = null,
  onIngestDocument,
  onDeleteDocument,
  onCheckOllama,
  isIngesting = false
}) {
  const [title, setTitle] = useState('');
  const [category, setCategory] = useState('doc');
  const [content, setContent] = useState('');
  const [chunkSize, setChunkSize] = useState(250);
  const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434');

  const estimatedChunks = Math.max(1, Math.ceil(content.length / (chunkSize || 200)));

  const handleLoadTemplate = (key) => {
    const t = SAMPLE_TEMPLATES[key];
    if (t) {
      setTitle(t.title);
      setCategory(t.category);
      setContent(t.content);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!title.trim() || !content.trim()) return;

    onIngestDocument({
      title: title.trim(),
      category: category.toLowerCase(),
      content: content.trim(),
      chunk_size: chunkSize
    });

    setTitle('');
    setContent('');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Ollama Service Connection Status */}
      <div>
        <div className="sec" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>Ollama Neural Embedding Provider</span>
          <button
            type="button"
            onClick={() => onCheckOllama && onCheckOllama(ollamaUrl)}
            className="btn-s"
            style={{ width: 'auto', padding: '2px 8px', fontSize: '9px' }}
          >
            <RefreshCw size={10} /> Test
          </button>
        </div>
        <div className={`ollama-status ${ollamaStatus?.available ? 'ok' : 'err'}`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600 }}>
            {ollamaStatus?.available ? (
              <>
                <CheckCircle2 size={13} color="var(--green)" />
                <span style={{ color: 'var(--green)' }}>Connected to Ollama</span>
              </>
            ) : (
              <>
                <AlertCircle size={13} color="var(--red)" />
                <span style={{ color: 'var(--red)' }}>Ollama Offline (Synthetic 16D Mode)</span>
              </>
            )}
          </div>
          <div style={{ fontSize: '10px', color: 'var(--muted)', marginTop: '4px' }}>
            Model: <strong>{ollamaStatus?.model || 'nomic-embed-text / llama3.2'}</strong> • Endpoint: {ollamaUrl}
          </div>
        </div>
      </div>

      {/* Quick Template Fillers */}
      <div>
        <div className="sec">Quick Knowledge Packs</div>
        <div style={{ display: 'flex', gap: '5px' }}>
          <button
            type="button"
            className="algo-btn"
            style={{ fontSize: '9.5px' }}
            onClick={() => handleLoadTemplate('finance')}
          >
            📊 Finance MPT
          </button>
          <button
            type="button"
            className="algo-btn"
            style={{ fontSize: '9.5px' }}
            onClick={() => handleLoadTemplate('ai')}
          >
            🧠 LangGraph RAG
          </button>
          <button
            type="button"
            className="algo-btn"
            style={{ fontSize: '9.5px' }}
            onClick={() => handleLoadTemplate('microservices')}
          >
            🌐 3-Tier Gateway
          </button>
        </div>
      </div>

      {/* Ingest Document Form */}
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <div className="sec">Ingest Knowledge Document</div>

        <div>
          <input
            type="text"
            placeholder="Document Title (e.g. Quantum Computing Primer)"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            required
          />
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <div style={{ flex: 1 }}>
            <select value={category} onChange={(e) => setCategory(e.target.value)}>
              <option value="cs">Computer Science (CS)</option>
              <option value="math">Mathematics & Finance</option>
              <option value="food">Culinary & Cooking</option>
              <option value="sports">Athletics & Sports</option>
              <option value="doc">General Knowledge</option>
            </select>
          </div>
          <div style={{ width: '120px' }}>
            <input
              type="number"
              min="50"
              max="1000"
              step="50"
              placeholder="Chunk Size"
              value={chunkSize}
              onChange={(e) => setChunkSize(Number(e.target.value))}
              title="Chunk size in characters"
            />
          </div>
        </div>

        <div>
          <textarea
            placeholder="Paste raw text, articles, or documentation here. Text will be automatically chunked, embedded, and indexed into ChromaDB and HNSW..."
            value={content}
            onChange={(e) => setContent(e.target.value)}
            required
            rows={5}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--muted)', marginTop: '4px' }}>
            <span>{content.length} characters</span>
            <span>~{estimatedChunks} chunks will be created</span>
          </div>
        </div>

        <button
          type="submit"
          className="btn-g"
          disabled={isIngesting || !content.trim() || !title.trim()}
        >
          <Upload size={13} />
          {isIngesting ? 'Embedding & Indexing...' : 'Ingest into ChromaDB & HNSW'}
        </button>
      </form>

      {/* Document Library */}
      <div>
        <div className="sec">Knowledge Library ({documents.length} Docs)</div>
        {documents.length === 0 ? (
          <div style={{ fontSize: '11px', color: 'var(--muted)', textAlign: 'center', padding: '16px' }}>
            No documents ingested yet. Add one above or select a Quick Knowledge Pack.
          </div>
        ) : (
          <div>
            {documents.map((doc, idx) => (
              <div key={doc.id || idx} className="dcard">
                <div className="dcard-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <FileText size={13} color="var(--accent)" />
                  <span style={{ flex: 1 }}>{doc.title}</span>
                  <span
                    className="rcat"
                    style={{
                      fontSize: '9px',
                      background: 'rgba(108, 99, 255, 0.15)',
                      color: 'var(--accent)'
                    }}
                  >
                    {doc.category || 'DOC'}
                  </span>
                </div>
                <div className="dcard-preview">
                  {doc.preview || doc.content?.slice(0, 120) || 'No preview available'}...
                </div>
                <div className="dcard-foot">
                  <span className="dcard-words">
                    {doc.chunk_count || doc.chunks?.length || 1} chunks • {doc.content ? doc.content.split(/\s+/).length : '0'} words
                  </span>
                  {onDeleteDocument && (
                    <button
                      type="button"
                      className="del"
                      onClick={() => onDeleteDocument(doc.id)}
                      title="Delete document and remove from index"
                    >
                      <Trash2 size={10} style={{ marginRight: '3px' }} />
                      Delete
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
