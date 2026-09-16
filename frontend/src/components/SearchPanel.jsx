import React, { useState } from 'react';
import { Search, Zap, Layers, BarChart2, Trash2, Sliders } from 'lucide-react';

const PRESETS = {
  cs: {
    name: 'CS / Algorithms',
    vec: [0.95, 0.90, 0.85, 0.80, 0.1, 0.05, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  },
  math: {
    name: 'Mathematics / Calc',
    vec: [0.05, 0.0, 0.05, 0.0, 0.95, 0.90, 0.85, 0.80, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  },
  food: {
    name: 'Culinary / Food',
    vec: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.90, 0.85, 0.80, 0.0, 0.0, 0.0, 0.0]
  },
  sports: {
    name: 'Athletics / Sports',
    vec: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.90, 0.85, 0.80]
  }
};

export default function SearchPanel({
  onSearch,
  searchResults = [],
  searchStats = null,
  benchmarks = null,
  hnswLayers = null,
  onDeleteVector = null,
  isSearching = false
}) {
  const [algorithm, setAlgorithm] = useState('chroma'); // 'chroma', 'hnsw', 'kdtree', 'bruteforce'
  const [metric, setMetric] = useState('cosine'); // 'cosine', 'euclidean', 'manhattan'
  const [topK, setTopK] = useState(5);
  const [customVector, setCustomVector] = useState(PRESETS.cs.vec);
  const [activePreset, setActivePreset] = useState('cs');

  const handleSelectPreset = (key) => {
    setActivePreset(key);
    if (key === 'random') {
      const rand = Array.from({ length: 16 }, () => Number(Math.random().toFixed(2)));
      setCustomVector(rand);
    } else if (PRESETS[key]) {
      setCustomVector(PRESETS[key].vec);
    }
  };

  const handleRunSearch = () => {
    onSearch({
      vector: customVector,
      k: topK,
      algorithm,
      metric
    });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Algorithm Selector */}
      <div>
        <div className="sec">Search Engine Engine</div>
        <div className="algo-row">
          <button
            type="button"
            className={`algo-btn ${algorithm === 'chroma' ? 'on' : ''}`}
            onClick={() => setAlgorithm('chroma')}
          >
            ChromaDB
          </button>
          <button
            type="button"
            className={`algo-btn ${algorithm === 'hnsw' ? 'on' : ''}`}
            onClick={() => setAlgorithm('hnsw')}
          >
            HNSW
          </button>
          <button
            type="button"
            className={`algo-btn ${algorithm === 'kdtree' ? 'on' : ''}`}
            onClick={() => setAlgorithm('kdtree')}
          >
            KD-Tree
          </button>
          <button
            type="button"
            className={`algo-btn ${algorithm === 'bruteforce' ? 'on' : ''}`}
            onClick={() => setAlgorithm('bruteforce')}
          >
            BruteForce
          </button>
        </div>
      </div>

      {/* Metric & Top-K */}
      <div style={{ display: 'flex', gap: '10px' }}>
        <div style={{ flex: 1 }}>
          <div className="sec">Metric</div>
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>
            <option value="cosine">Cosine Sim</option>
            <option value="euclidean">Euclidean L2</option>
            <option value="manhattan">Manhattan L1</option>
          </select>
        </div>
        <div style={{ width: '90px' }}>
          <div className="sec">Top-K: {topK}</div>
          <input
            type="range"
            min="1"
            max="15"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            style={{ width: '100%', marginTop: '6px' }}
          />
        </div>
      </div>

      {/* Query Vector Preset */}
      <div>
        <div className="sec">Target Vector Presets (16D)</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginBottom: '8px' }}>
          {Object.entries(PRESETS).map(([key, p]) => (
            <button
              key={key}
              type="button"
              className={`algo-btn ${activePreset === key ? 'on' : ''}`}
              style={{ fontSize: '10px', padding: '6px 4px' }}
              onClick={() => handleSelectPreset(key)}
            >
              {p.name.split(' / ')[0]}
            </button>
          ))}
        </div>
        <button
          type="button"
          className={`algo-btn ${activePreset === 'random' ? 'on' : ''}`}
          style={{ width: '100%', fontSize: '10px', padding: '6px' }}
          onClick={() => handleSelectPreset('random')}
        >
          🎲 Generate Random 16D Vector
        </button>
      </div>

      {/* Search Button */}
      <button
        type="button"
        className="btn-p"
        onClick={handleRunSearch}
        disabled={isSearching}
      >
        <Search size={14} />
        {isSearching ? 'Computing...' : `Run 16D Search (${algorithm.toUpperCase()})`}
      </button>

      {/* Latency & Stats */}
      {searchStats && (
        <div style={{ background: 'var(--card-subtle)', border: '1px solid var(--border)', borderRadius: '8px', padding: '12px' }}>
          <div className="sec">Search Telemetry</div>
          <div className="lat-big">
            {searchStats.latency_us ? `${(searchStats.latency_us / 1000).toFixed(2)} ms` : `${searchStats.latency_ms || 0.12} ms`}
          </div>
          <div className="lat-sub">
            {searchStats.algorithm?.toUpperCase()} Index • {searchStats.metric} • {searchStats.visited_nodes ?? searchResults.length} vectors scanned
          </div>
        </div>
      )}

      {/* Comparative Latency Benchmark */}
      {benchmarks && (
        <div>
          <div className="sec" style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <BarChart2 size={12} /> Multi-Engine Benchmark (μs)
          </div>
          <div className="bench">
            {Object.entries(benchmarks).map(([algo, lat]) => {
              const maxLat = Math.max(...Object.values(benchmarks), 1);
              const pct = Math.min(100, Math.max(8, (lat / maxLat) * 100));
              const isFastest = lat === Math.min(...Object.values(benchmarks));
              return (
                <div key={algo} className="brow">
                  <div className="blabel">
                    <span style={{ color: isFastest ? 'var(--green)' : 'var(--text)' }}>
                      {algo.toUpperCase()} {isFastest && '★'}
                    </span>
                    <span style={{ color: 'var(--muted)' }}>{lat.toFixed(1)} μs</span>
                  </div>
                  <div className="btrack">
                    <div
                      className="bfill"
                      style={{
                        width: `${pct}%`,
                        background: isFastest ? 'var(--green)' : 'var(--accent)'
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* HNSW Layer Distribution */}
      {hnswLayers && (
        <div>
          <div className="sec" style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <Layers size={12} /> HNSW Hierarchical Graph Layers
          </div>
          <div className="layers">
            {hnswLayers.map((layer, idx) => (
              <div key={idx} className="lrow">
                <span className="lnum">L{idx}</span>
                <div className="ltrack">
                  <div className="lfill" style={{ width: `${Math.min(100, (layer.count / (hnswLayers[0]?.count || 1)) * 100)}%` }} />
                </div>
                <span className="lcount">{layer.count} nodes</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Search Results */}
      <div>
        <div className="sec">Nearest Matches ({searchResults.length})</div>
        {searchResults.length === 0 ? (
          <div style={{ fontSize: '11px', color: 'var(--muted)', textAlign: 'center', padding: '16px' }}>
            Click "Run 16D Search" to discover closest vectors.
          </div>
        ) : (
          <div className="results">
            {searchResults.map((res, index) => {
              const catColor = {
                cs: 'var(--cs)',
                math: 'var(--math)',
                food: 'var(--food)',
                sports: 'var(--sports)',
                doc: 'var(--accent)'
              }[res.category] || 'var(--text)';

              return (
                <div key={res.id || index} className="rcard">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span className="rrank">RANK #{index + 1}</span>
                    <span
                      className="rcat"
                      style={{
                        background: `${catColor}18`,
                        color: catColor,
                        border: `1px solid ${catColor}44`
                      }}
                    >
                      {res.category?.toUpperCase() || 'DATA'}
                    </span>
                  </div>
                  <div className="rmeta">
                    {typeof res.metadata === 'string'
                      ? res.metadata
                      : (res.metadata?.text || res.text || res.label || `Vector #${res.id}`)}
                  </div>
                  <div className="rfoot">
                    <span className="rdist">
                      Dist: <strong>{(res.distance ?? res.score ?? 0).toFixed(4)}</strong>
                    </span>
                    {onDeleteVector && (
                      <button
                        type="button"
                        className="del"
                        onClick={() => onDeleteVector(res.id)}
                        title="Delete vector"
                      >
                        <Trash2 size={10} style={{ marginRight: '3px' }} />
                        Del
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
