import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import VectorVisualizer from './components/VectorVisualizer';
import SearchPanel from './components/SearchPanel';
import DocumentManager from './components/DocumentManager';
import AgenticRagChat from './components/AgenticRagChat';
import { Search, Database, MessageSquare, Plus, RefreshCw, Cpu, CheckCircle2, ShieldCheck, Box } from 'lucide-react';

const API_BASE = ''; // Same-origin relative URLs handled by Express Gateway (port 8080) or Vite Proxy

export default function App() {
  const [activeTab, setActiveTab] = useState(0); // 0: Search, 1: Documents, 2: RAG Chat
  const [vectors, setVectors] = useState([]);
  const [queryPoint, setQueryPoint] = useState(null);
  const [searchResults, setSearchResults] = useState([]);
  const [searchStats, setSearchStats] = useState(null);
  const [benchmarks, setBenchmarks] = useState(null);
  const [hnswLayers, setHnswLayers] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [systemHealth, setSystemHealth] = useState(null);
  const [ollamaStatus, setOllamaStatus] = useState(null);

  const [isSearching, setIsSearching] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [isThinking, setIsThinking] = useState(false);

  // Fetch initial vectors from API
  const fetchVectors = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/items`);
      if (resp.ok) {
        const data = await resp.json();
        setVectors(Array.isArray(data) ? data : data.items || []);
      }
    } catch (err) {
      console.warn('Error fetching vectors:', err);
    }
  }, []);

  // Fetch HNSW topology
  const fetchHnswInfo = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/hnsw-info`);
      if (resp.ok) {
        const data = await resp.json();
        if (data && Array.isArray(data.nodesPerLayer)) {
          setHnswLayers(data.nodesPerLayer.map((count, layer) => ({ layer, count })));
        } else if (data && data.layers) {
          setHnswLayers(data.layers);
        }
      }
    } catch (_) {}
  }, []);

  // Fetch Document library
  const fetchDocuments = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/doc/list`);
      if (resp.ok) {
        const data = await resp.json();
        setDocuments(Array.isArray(data) ? data : []);
      }
    } catch (err) {
      console.warn('Error fetching documents:', err);
    }
  }, []);

  // Fetch System Status & Ollama
  const fetchHealth = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/status`);
      if (resp.ok) {
        const data = await resp.json();
        setSystemHealth(data);
        setOllamaStatus({
          available: !!data.ollamaAvailable,
          model: data.ollamaEmbedModel || 'nomic-embed-text'
        });
      }
    } catch (_) {
      setSystemHealth({ status: 'err', python_ai: 'offline' });
    }
  }, []);

  useEffect(() => {
    fetchVectors();
    fetchHnswInfo();
    fetchDocuments();
    fetchHealth();
  }, [fetchVectors, fetchHnswInfo, fetchDocuments, fetchHealth]);

  // Execute Vector Search
  const handleSearch = async ({ vector, k, algorithm, metric }) => {
    setIsSearching(true);
    setQueryPoint(vector);

    try {
      const vStr = encodeURIComponent(Array.isArray(vector) ? vector.join(',') : vector);
      const url = `${API_BASE}/search?v=${vStr}&k=${k}&metric=${metric}&algo=${algorithm}`;
      const resp = await fetch(url);
      if (resp.ok) {
        const data = await resp.json();
        setSearchResults(data.results || data.hits || []);
        const latUs = data.latencyUs || data.latency_us;
        setSearchStats({
          latency_us: latUs,
          latency_ms: latUs ? (latUs / 1000).toFixed(2) : '0.15',
          algorithm: data.algo || data.algorithm || algorithm,
          metric: data.metric || metric,
          visited_nodes: data.visited_nodes || (data.results ? data.results.length : 0)
        });
      }

      // Also trigger background comparative benchmark
      try {
        const benchUrl = `${API_BASE}/benchmark?v=${vStr}&k=${k}&metric=${metric}`;
        const benchResp = await fetch(benchUrl);
        if (benchResp.ok) {
          const benchData = await benchResp.json();
          if (benchData.bruteforceUs !== undefined) {
            setBenchmarks({
              bruteforce: benchData.bruteforceUs,
              kdtree: benchData.kdtreeUs,
              hnsw: benchData.hnswUs,
              chromadb: benchData.chromadbUs || 0
            });
          } else {
            setBenchmarks(benchData.latencies_us || benchData);
          }
        }
      } catch (_) {}
    } catch (err) {
      console.error('Search error:', err);
    } finally {
      setIsSearching(false);
    }
  };

  // Delete Vector
  const handleDeleteVector = async (id) => {
    try {
      const resp = await fetch(`${API_BASE}/delete/${id}`, { method: 'DELETE' });
      if (resp.ok) {
        setVectors(prev => prev.filter(v => v.id !== id));
        setSearchResults(prev => prev.filter(r => r.id !== id));
      }
    } catch (err) {
      console.error('Delete vector error:', err);
    }
  };

  // Ingest Document into ChromaDB / HNSW
  const handleIngestDocument = async ({ title, category, content, chunk_size }) => {
    setIsIngesting(true);
    try {
      const resp = await fetch(`${API_BASE}/doc/insert`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, text: content, chunk_size, category })
      });

      if (resp.ok) {
        await fetchDocuments();
        await fetchHealth();
      } else {
        const errData = await resp.json().catch(() => ({}));
        alert(errData.detail || 'Failed to ingest document');
      }
    } catch (err) {
      console.error('Ingest document error:', err);
      alert('Ingest document failed: ' + err.message);
    } finally {
      setIsIngesting(false);
    }
  };

  // Delete Document
  const handleDeleteDocument = async (id) => {
    try {
      const resp = await fetch(`${API_BASE}/doc/delete/${id}`, { method: 'DELETE' });
      if (resp.ok) {
        setDocuments(prev => prev.filter(d => d.id !== id));
        await fetchHealth();
      }
    } catch (err) {
      console.error('Delete doc error:', err);
    }
  };

  // Ask LangGraph Agentic RAG
  const handleAskRag = async ({ question, k }) => {
    setIsThinking(true);
    try {
      const resp = await fetch(`${API_BASE}/doc/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, k })
      });

      if (resp.ok) {
        const data = await resp.json();
        setChatHistory(prev => [
          ...prev,
          {
            question,
            answer: data.answer || 'No response generated.',
            grounded: data.grounded ?? true,
            trace: data.steps || data.trace || [],
            documents: data.contexts || data.filtered_documents || []
          }
        ]);
      } else {
        const err = await resp.json().catch(() => ({}));
        setChatHistory(prev => [
          ...prev,
          {
            question,
            answer: `❌ Error: ${err.detail || err.error || 'Failed to connect to LangGraph service. Ensure Ollama/Python AI is online.'}`,
            grounded: false,
            trace: [],
            documents: []
          }
        ]);
      }
    } catch (err) {
      setChatHistory(prev => [
        ...prev,
        {
          question,
          answer: `❌ Network Error: ${err.message}`,
          grounded: false,
          trace: [],
          documents: []
        }
      ]);
    } finally {
      setIsThinking(false);
    }
  };

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Top Header */}
      <Header
        systemHealth={systemHealth}
        ollamaStatus={ollamaStatus}
        vectorCount={vectors.length}
        dimension={16}
      />

      {/* Main 3-Column Layout */}
      <div className="layout">
        {/* Left Panel: Stats, Index Types, Quick Seed */}
        <aside className="left-panel">
          <div>
            <div className="sec">Architecture State</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '11px', color: 'var(--muted)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Gateway:</span>
                <strong style={{ color: 'var(--green)' }}>Node.js (8080)</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>AI Service:</span>
                <strong style={{ color: 'var(--cs)' }}>FastAPI (8000)</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Frontend:</span>
                <strong style={{ color: 'var(--accent)' }}>React + Canvas</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Index Engine:</span>
                <strong style={{ color: 'var(--math)' }}>ChromaDB + HNSW</strong>
              </div>
            </div>
          </div>

          <div>
            <div className="sec">Cluster Color Legend</div>
            <div className="legend">
              <div className="leg-row">
                <span className="dot" style={{ background: 'var(--cs)' }} />
                <span>CS / Algorithms</span>
              </div>
              <div className="leg-row">
                <span className="dot" style={{ background: 'var(--math)' }} />
                <span>Mathematics / Calc</span>
              </div>
              <div className="leg-row">
                <span className="dot" style={{ background: 'var(--food)' }} />
                <span>Culinary / Food</span>
              </div>
              <div className="leg-row">
                <span className="dot" style={{ background: 'var(--sports)' }} />
                <span>Athletics / Sports</span>
              </div>
              <div className="leg-row">
                <span className="dot" style={{ background: 'var(--accent)' }} />
                <span>Document Chunks</span>
              </div>
            </div>
          </div>

          <div>
            <div className="sec">Database Control</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <button
                type="button"
                className="btn-s"
                onClick={() => {
                  fetchVectors();
                  fetchHnswInfo();
                  fetchDocuments();
                }}
                title="Refresh vectors and status"
              >
                <RefreshCw size={12} /> Sync Database
              </button>
            </div>
          </div>

          <div style={{ marginTop: 'auto', borderTop: '1px solid var(--border)', paddingTop: '12px', fontSize: '10px', color: 'var(--muted)', lineHeight: 1.5 }}>
            ⚡ <strong>60 FPS Canvas:</strong> Renders multi-dimensional vectors projected via client-side Power Iteration PCA with zero DOM reflow.
          </div>
        </aside>

        {/* Center Panel: 60 FPS HTML5 Canvas Visualizer */}
        <main style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          <VectorVisualizer
            vectors={vectors}
            queryPoint={queryPoint}
            searchResults={searchResults}
            onSelectVector={(v) => setQueryPoint(v.vector)}
          />
        </main>

        {/* Right Panel: Tabbed Navigation */}
        <aside className="right-panel">
          {/* Tabs Header */}
          <div className="tabs">
            <div
              className={`tab ${activeTab === 0 ? 'on' : ''}`}
              onClick={() => setActiveTab(0)}
            >
              <Search size={11} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
              16D Search
            </div>
            <div
              className={`tab ${activeTab === 1 ? 'on' : ''}`}
              onClick={() => setActiveTab(1)}
            >
              <Database size={11} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
              Documents
            </div>
            <div
              className={`tab ${activeTab === 2 ? 'on' : ''}`}
              onClick={() => setActiveTab(2)}
            >
              <MessageSquare size={11} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
              LangGraph RAG
            </div>
          </div>

          {/* Active Tab Content */}
          <div className="tab-content">
            {activeTab === 0 && (
              <SearchPanel
                onSearch={handleSearch}
                searchResults={searchResults}
                searchStats={searchStats}
                benchmarks={benchmarks}
                hnswLayers={hnswLayers}
                onDeleteVector={handleDeleteVector}
                isSearching={isSearching}
              />
            )}

            {activeTab === 1 && (
              <DocumentManager
                documents={documents}
                ollamaStatus={ollamaStatus}
                onIngestDocument={handleIngestDocument}
                onDeleteDocument={handleDeleteDocument}
                onCheckOllama={fetchHealth}
                isIngesting={isIngesting}
              />
            )}

            {activeTab === 2 && (
              <AgenticRagChat
                onAskRag={handleAskRag}
                chatHistory={chatHistory}
                isThinking={isThinking}
                onClearHistory={() => setChatHistory([])}
              />
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
