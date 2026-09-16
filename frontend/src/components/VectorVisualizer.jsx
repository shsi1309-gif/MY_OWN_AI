import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { Maximize2, RefreshCw, ZoomIn, ZoomOut, Compass } from 'lucide-react';

const CATEGORY_COLORS = {
  cs: '#00d9ff',
  math: '#b388ff',
  food: '#ffb74d',
  sports: '#69f0ae',
  doc: '#6c63ff',
  default: '#cdd6f4'
};

// Power iteration PCA for 2D projection
function computePCA2D(vectors) {
  if (!vectors || vectors.length < 2) {
    return vectors.map((_, i) => [i * 10, i * 10]);
  }
  const n = vectors.length;
  const d = vectors[0].length;

  // Mean center
  const mean = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      mean[j] += vectors[i][j];
    }
  }
  for (let j = 0; j < d; j++) mean[j] /= n;

  const X = vectors.map(v => v.map((val, j) => val - mean[j]));

  // Power iteration for PC1
  let pc1 = new Array(d).fill(0).map((_, i) => Math.sin(i + 1));
  let norm1 = Math.sqrt(pc1.reduce((s, v) => s + v * v, 0)) || 1;
  pc1 = pc1.map(v => v / norm1);

  for (let iter = 0; iter < 15; iter++) {
    const next = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      const dot = X[i].reduce((s, v, idx) => s + v * pc1[idx], 0);
      for (let j = 0; j < d; j++) next[j] += dot * X[i][j];
    }
    const norm = Math.sqrt(next.reduce((s, v) => s + v * v, 0)) || 1;
    pc1 = next.map(v => v / norm);
  }

  // Power iteration for PC2 (orthogonal to PC1)
  let pc2 = new Array(d).fill(0).map((_, i) => Math.cos(i + 1));
  for (let iter = 0; iter < 15; iter++) {
    const dot1 = pc2.reduce((s, v, idx) => s + v * pc1[idx], 0);
    pc2 = pc2.map((v, idx) => v - dot1 * pc1[idx]);

    const next = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      const dot = X[i].reduce((s, v, idx) => s + v * pc2[idx], 0);
      for (let j = 0; j < d; j++) next[j] += dot * X[i][j];
    }
    const dotNext1 = next.reduce((s, v, idx) => s + v * pc1[idx], 0);
    const ortho = next.map((v, idx) => v - dotNext1 * pc1[idx]);
    const norm = Math.sqrt(ortho.reduce((s, v) => s + v * v, 0)) || 1;
    pc2 = ortho.map(v => v / norm);
  }

  return X.map(v => [
    v.reduce((s, val, idx) => s + val * pc1[idx], 0),
    v.reduce((s, val, idx) => s + val * pc2[idx], 0)
  ]);
}

export default function VectorVisualizer({
  vectors = [],
  queryPoint = null,
  searchResults = [],
  onSelectVector = null
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  // Transform state: pan & zoom
  const [transform, setTransform] = useState({ x: 0, y: 0, k: 1 });
  const [hoveredNode, setHoveredNode] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0 });
  const [fps, setFps] = useState(60);

  // 2D Projection calculations
  const projectedPoints = useMemo(() => {
    if (!vectors || vectors.length === 0) return [];
    const rawVecs = vectors.map(v => v.vector || v.embedding || []);
    const validRaw = rawVecs.filter(v => v.length > 0);
    if (validRaw.length === 0) return [];

    const projected2D = computePCA2D(rawVecs);

    // Find bounds for normalization
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    projected2D.forEach(([x, y]) => {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    });

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    return vectors.map((item, idx) => {
      const [px, py] = projected2D[idx] || [0, 0];
      // Normalize to [-1, 1] range
      const nx = ((px - minX) / rangeX - 0.5) * 2;
      const ny = ((py - minY) / rangeY - 0.5) * 2;
      return {
        ...item,
        x: nx,
        y: ny,
        color: CATEGORY_COLORS[item.category] || CATEGORY_COLORS.default
      };
    });
  }, [vectors]);

  // Project active query point into PCA space if available
  const projectedQuery = useMemo(() => {
    if (!queryPoint || !vectors.length) return null;
    // Approximate query projection using centroid of nearest match or simple offset
    const targetMatch = searchResults[0];
    if (targetMatch) {
      const node = projectedPoints.find(p => p.id === targetMatch.id);
      if (node) {
        return { x: node.x - 0.05, y: node.y - 0.05 };
      }
    }
    return { x: 0, y: 0 };
  }, [queryPoint, searchResults, projectedPoints]);

  // Render loop using requestAnimationFrame
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animId;
    let lastTime = performance.now();
    let frameCount = 0;

    const render = (time) => {
      frameCount++;
      if (time - lastTime >= 1000) {
        setFps(Math.round((frameCount * 1000) / (time - lastTime)));
        frameCount = 0;
        lastTime = time;
      }

      const { width, height } = canvas;
      if (width === 0 || height === 0) {
        animId = requestAnimationFrame(render);
        return;
      }

      ctx.clearRect(0, 0, width, height);

      // Background radial gradient
      const bgGrad = ctx.createRadialGradient(
        width / 2, height / 2, 50,
        width / 2, height / 2, Math.max(width, height) / 1.5
      );
      bgGrad.addColorStop(0, '#0a0a16');
      bgGrad.addColorStop(1, '#05050a');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, width, height);

      // Coordinate axes & subtle grid
      ctx.save();
      const centerX = width / 2 + transform.x;
      const centerY = height / 2 + transform.y;
      const scale = Math.min(width, height) * 0.38 * transform.k;

      ctx.strokeStyle = 'rgba(21, 21, 40, 0.8)';
      ctx.lineWidth = 1;
      const gridSize = 40 * transform.k;
      const startX = (centerX % gridSize) - gridSize;
      const startY = (centerY % gridSize) - gridSize;

      ctx.beginPath();
      for (let x = startX; x < width + gridSize; x += gridSize) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
      }
      for (let y = startY; y < height + gridSize; y += gridSize) {
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
      }
      ctx.stroke();

      // Center crosshair axes
      ctx.strokeStyle = 'rgba(108, 99, 255, 0.18)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(centerX, 0);
      ctx.lineTo(centerX, height);
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();

      // Search laser beams from query point to top results
      if (projectedQuery && searchResults && searchResults.length > 0) {
        const qScreenX = centerX + projectedQuery.x * scale;
        const qScreenY = centerY + projectedQuery.y * scale;

        searchResults.forEach((res, rank) => {
          const target = projectedPoints.find(p => p.id === res.id);
          if (target) {
            const tScreenX = centerX + target.x * scale;
            const tScreenY = centerY + target.y * scale;

            // Pulsing beam
            ctx.save();
            const beamPulse = (Math.sin(time / 200 + rank) + 1) / 2;
            ctx.strokeStyle = rank === 0 ? 'rgba(0, 217, 255, 0.8)' : 'rgba(179, 136, 255, 0.45)';
            ctx.lineWidth = rank === 0 ? 2 + beamPulse : 1.2;
            ctx.setLineDash([4, 4]);
            ctx.lineDashOffset = -time / 40;

            ctx.beginPath();
            ctx.moveTo(qScreenX, qScreenY);
            ctx.lineTo(tScreenX, tScreenY);
            ctx.stroke();
            ctx.restore();

            // Highlight ring around matched node
            ctx.save();
            ctx.beginPath();
            ctx.arc(tScreenX, tScreenY, 12 + beamPulse * 4, 0, Math.PI * 2);
            ctx.strokeStyle = target.color;
            ctx.lineWidth = 1.5;
            ctx.stroke();
            ctx.restore();
          }
        });

        // Render query point target diamond
        ctx.save();
        ctx.translate(qScreenX, qScreenY);
        ctx.rotate(Math.PI / 4);
        const pulse = 1 + Math.sin(time / 250) * 0.2;
        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = '#00d9ff';
        ctx.shadowBlur = 15;
        ctx.fillRect(-6 * pulse, -6 * pulse, 12 * pulse, 12 * pulse);
        ctx.restore();
      }

      // Render vector nodes
      projectedPoints.forEach((node) => {
        const screenX = centerX + node.x * scale;
        const screenY = centerY + node.y * scale;
        const isHovered = hoveredNode && hoveredNode.id === node.id;
        const radius = isHovered ? 8 : 5.5;

        // Glow halo
        ctx.save();
        ctx.beginPath();
        ctx.arc(screenX, screenY, radius + 4, 0, Math.PI * 2);
        ctx.fillStyle = `${node.color}22`;
        ctx.fill();

        // Node core
        ctx.beginPath();
        ctx.arc(screenX, screenY, radius, 0, Math.PI * 2);
        ctx.fillStyle = node.color;
        if (isHovered) {
          ctx.shadowColor = node.color;
          ctx.shadowBlur = 18;
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
        ctx.fill();
        ctx.restore();

        // Label if hovered
        if (isHovered) {
          ctx.save();
          ctx.font = '10px "Fira Code", monospace';
          ctx.fillStyle = '#ffffff';
          ctx.shadowColor = '#000000';
          ctx.shadowBlur = 4;
          ctx.fillText(`[${node.category}] #${node.id}`, screenX + 10, screenY - 10);
          ctx.restore();
        }
      });

      ctx.restore();
      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [projectedPoints, transform, hoveredNode, projectedQuery, searchResults]);

  // Resize canvas to match container
  const handleResize = useCallback(() => {
    if (!containerRef.current || !canvasRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvasRef.current.width = rect.width * dpr;
    canvasRef.current.height = rect.height * dpr;
    canvasRef.current.style.width = `${rect.width}px`;
    canvasRef.current.style.height = `${rect.height}px`;
    const ctx = canvasRef.current.getContext('2d');
    ctx.scale(dpr, dpr);
  }, []);

  useEffect(() => {
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [handleResize]);

  // Mouse interaction: Hover tooltip & Drag Pan
  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (isDragging) {
      setTransform(prev => ({
        ...prev,
        x: prev.x + (mouseX - dragStartRef.current.x),
        y: prev.y + (mouseY - dragStartRef.current.y)
      }));
      dragStartRef.current = { x: mouseX, y: mouseY };
      return;
    }

    const width = rect.width;
    const height = rect.height;
    const centerX = width / 2 + transform.x;
    const centerY = height / 2 + transform.y;
    const scale = Math.min(width, height) * 0.38 * transform.k;

    let found = null;
    for (const node of projectedPoints) {
      const screenX = centerX + node.x * scale;
      const screenY = centerY + node.y * scale;
      const dist = Math.hypot(mouseX - screenX, mouseY - screenY);
      if (dist < 12) {
        found = node;
        break;
      }
    }

    if (found) {
      setHoveredNode(found);
      setTooltipPos({ x: e.clientX + 14, y: e.clientY + 14 });
    } else {
      setHoveredNode(null);
    }
  };

  const handleMouseDown = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    dragStartRef.current = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
    setIsDragging(true);
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleWheel = (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
    setTransform(prev => ({
      ...prev,
      k: Math.max(0.4, Math.min(6, prev.k * zoomFactor))
    }));
  };

  const handleResetView = () => setTransform({ x: 0, y: 0, k: 1 });

  return (
    <div
      ref={containerRef}
      className="center-panel"
      style={{ position: 'relative', width: '100%', height: '100%', userSelect: 'none' }}
    >
      <canvas
        ref={canvasRef}
        id="scatter"
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => { setIsDragging(false); setHoveredNode(null); }}
        onWheel={handleWheel}
        style={{ cursor: isDragging ? 'grabbing' : hoveredNode ? 'pointer' : 'grab' }}
      />

      {/* Floating Canvas Controls Overlay */}
      <div
        style={{
          position: 'absolute',
          top: 14,
          left: 14,
          background: 'rgba(12, 12, 24, 0.85)',
          backdropFilter: 'blur(8px)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          padding: '6px 12px',
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          fontSize: 11,
          color: 'var(--muted)',
          zIndex: 10
        }}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: 5, color: 'var(--cs)', fontWeight: 600 }}>
          <Compass size={13} /> 2D PCA Projection (60 FPS)
        </span>
        <span>•</span>
        <span>{projectedPoints.length} Points</span>
        <span>•</span>
        <span style={{ color: 'var(--green)' }}>{fps} FPS</span>
      </div>

      <div
        style={{
          position: 'absolute',
          bottom: 14,
          right: 14,
          display: 'flex',
          gap: 6,
          zIndex: 10
        }}
      >
        <button
          onClick={() => setTransform(p => ({ ...p, k: Math.min(6, p.k * 1.25) }))}
          className="btn-s"
          style={{ width: 32, height: 32, padding: 0 }}
          title="Zoom In"
        >
          <ZoomIn size={14} />
        </button>
        <button
          onClick={() => setTransform(p => ({ ...p, k: Math.max(0.4, p.k * 0.8) }))}
          className="btn-s"
          style={{ width: 32, height: 32, padding: 0 }}
          title="Zoom Out"
        >
          <ZoomOut size={14} />
        </button>
        <button
          onClick={handleResetView}
          className="btn-s"
          style={{ width: 32, height: 32, padding: 0 }}
          title="Reset View"
        >
          <RefreshCw size={14} />
        </button>
      </div>

      {/* Hover Tooltip */}
      {hoveredNode && (
        <div
          id="tip"
          style={{
            left: `${tooltipPos.x}px`,
            top: `${tooltipPos.y}px`,
            display: 'block'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ fontWeight: 700, color: hoveredNode.color }}>#{hoveredNode.id}</span>
            <span
              style={{
                fontSize: 9,
                padding: '1px 6px',
                borderRadius: 4,
                background: `${hoveredNode.color}22`,
                color: hoveredNode.color
              }}
            >
              {hoveredNode.category?.toUpperCase()}
            </span>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text)', marginBottom: 4 }}>
            {typeof hoveredNode.metadata === 'string'
              ? hoveredNode.metadata
              : (hoveredNode.metadata?.text || hoveredNode.text || hoveredNode.label || 'No description')}
          </div>
          {(hoveredNode.emb || hoveredNode.vector) && (
            <div style={{ fontSize: 9, color: 'var(--muted)', fontFamily: 'monospace' }}>
              [{(hoveredNode.emb || hoveredNode.vector).slice(0, 4).map(v => Number(v).toFixed(2)).join(', ')}...]
            </div>
          )}
        </div>
      )}
    </div>
  );
}
