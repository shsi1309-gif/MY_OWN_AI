/**
 * Node.js + Express API Gateway for 3-Tier AI Platform
 * Coordinates client traffic, serves Swagger UI at /docs, and proxies AI requests to Python FastAPI AI service.
 */

import express from 'express';
import cors from 'cors';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import swaggerUi from 'swagger-ui-express';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = parseInt(process.env.PORT || '8080', 10);
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://127.0.0.1:8000';

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// =====================================================================
//  SWAGGER OPENAPI SPECIFICATION
// =====================================================================

const swaggerDocument = {
  openapi: '3.0.0',
  info: {
    title: 'VectorDB & LangGraph 3-Tier Platform API',
    version: '2.0.0',
    description: 'Node.js Express API Gateway proxying to Python FastAPI + LangGraph AI Microservice.',
  },
  servers: [{ url: `http://localhost:${PORT}` }],
  paths: {
    '/api/status': {
      get: {
        summary: 'System health, LangGraph & ChromaDB status',
        responses: { 200: { description: 'Status info' } },
      },
    },
    '/api/items': {
      get: {
        summary: 'List all demo 16D vectors',
        responses: { 200: { description: 'Vector items list' } },
      },
    },
    '/api/search': {
      get: {
        summary: 'Search nearest vectors',
        parameters: [
          { name: 'v', in: 'query', required: true, schema: { type: 'string' } },
          { name: 'k', in: 'query', schema: { type: 'integer', default: 5 } },
          { name: 'metric', in: 'query', schema: { type: 'string', default: 'cosine' } },
          { name: 'algo', in: 'query', schema: { type: 'string', default: 'hnsw' } },
        ],
        responses: { 200: { description: 'Search hits' } },
      },
    },
    '/api/benchmark': {
      get: {
        summary: 'Benchmark all vector algorithms side-by-side',
        parameters: [
          { name: 'v', in: 'query', required: true, schema: { type: 'string' } },
          { name: 'k', in: 'query', schema: { type: 'integer', default: 5 } },
          { name: 'metric', in: 'query', schema: { type: 'string', default: 'cosine' } },
        ],
        responses: { 200: { description: 'Latency comparison' } },
      },
    },
    '/api/hnsw-info': {
      get: {
        summary: 'Get HNSW graph layers and node connectivity',
        responses: { 200: { description: 'Graph topology' } },
      },
    },
    '/api/doc/insert': {
      post: {
        summary: 'Chunk, embed and insert document into ChromaDB',
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                properties: {
                  title: { type: 'string' },
                  text: { type: 'string' },
                },
                required: ['title', 'text'],
              },
            },
          },
        },
        responses: { 200: { description: 'Inserted chunk IDs' } },
      },
    },
    '/api/doc/ask': {
      post: {
        summary: 'Ask question with LangGraph Self-Correcting Agentic RAG',
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                properties: {
                  question: { type: 'string' },
                  k: { type: 'integer', default: 3 },
                },
                required: ['question'],
              },
            },
          },
        },
        responses: { 200: { description: 'Synthesized answer + execution steps' } },
      },
    },
  },
};

app.use('/docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));

// =====================================================================
//  PROXY HELPER TO PYTHON AI/ML SERVICE
// =====================================================================

async function forwardToAI(req, res, targetPath) {
  try {
    const url = `${AI_SERVICE_URL}${targetPath}`;
    const options = {
      method: req.method,
      headers: {
        'Content-Type': 'application/json',
      },
    };

    if (req.method !== 'GET' && req.method !== 'HEAD' && req.body) {
      options.body = JSON.stringify(req.body);
    }

    const aiResp = await fetch(url, options);
    const contentType = aiResp.headers.get('content-type') || '';

    if (contentType.includes('application/json')) {
      const data = await aiResp.json();
      return res.status(aiResp.status).json(data);
    } else {
      const text = await aiResp.text();
      return res.status(aiResp.status).send(text);
    }
  } catch (error) {
    console.error(`Error proxying to AI Service at ${AI_SERVICE_URL}${targetPath}:`, error.message);
    return res.status(503).json({
      error: `Python AI Engine unavailable at ${AI_SERVICE_URL}. Is ai_service.py running?`,
      detail: error.message,
    });
  }
}

// =====================================================================
//  ROUTING (Supports both /api/* and root paths)
// =====================================================================

const routeMap = [
  { method: 'GET', path: '/items' },
  { method: 'GET', path: '/search' },
  { method: 'GET', path: '/benchmark' },
  { method: 'GET', path: '/hnsw-info' },
  { method: 'GET', path: '/stats' },
  { method: 'GET', path: '/doc/list' },
  { method: 'POST', path: '/config/ollama' },
  { method: 'POST', path: '/insert' },
  { method: 'POST', path: '/doc/insert' },
  { method: 'POST', path: '/doc/search' },
  { method: 'POST', path: '/doc/ask' },
  { method: 'DELETE', path: '/delete/:id' },
  { method: 'DELETE', path: '/doc/delete/:id' },
];

routeMap.forEach(({ method, path: p }) => {
  const handler = (req, res) => {
    // Preserve query parameters if present
    const queryString = req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : '';
    const cleanPath = req.path.startsWith('/api') ? req.path.replace(/^\/api/, '') : req.path;
    forwardToAI(req, res, `${cleanPath}${queryString}`);
  };

  if (method === 'GET') {
    app.get(p, handler);
    app.get(`/api${p}`, handler);
  } else if (method === 'POST') {
    app.post(p, handler);
    app.post(`/api${p}`, handler);
  } else if (method === 'DELETE') {
    app.delete(p, handler);
    app.delete(`/api${p}`, handler);
  }
});

// Custom Status endpoint returning Gateway + AI Service info
app.get(['/status', '/api/status'], async (req, res) => {
  try {
    const aiResp = await fetch(`${AI_SERVICE_URL}/status`);
    if (aiResp.ok) {
      const aiData = await aiResp.json();
      return res.json({
        ...aiData,
        gateway: 'Node.js Express (Port ' + PORT + ')',
        aiService: 'Python FastAPI (Port 8000)',
        architecture: '3-Tier Hybrid (React + Express + Python AI)',
      });
    }
  } catch (_) {}

  return res.json({
    gateway: 'Node.js Express (Online)',
    aiService: `Offline (Attempted ${AI_SERVICE_URL})`,
    langgraphAvailable: false,
    ollamaAvailable: false,
    chromaAvailable: false,
    demoCount: 0,
    docCount: 0,
  });
});

// =====================================================================
//  STATIC ASSETS & REACT FRONTEND SERVING
// =====================================================================

const distPath = path.join(__dirname, 'frontend', 'dist');
if (fs.existsSync(distPath)) {
  app.use(express.static(distPath));
  app.get('*', (req, res) => {
    if (!req.path.startsWith('/api') && !req.path.startsWith('/docs')) {
      res.sendFile(path.join(distPath, 'index.html'));
    }
  });
} else {
  // Fallback to serving the root index.html if frontend is not built yet
  app.get('/', (req, res) => {
    const rootHtml = path.join(__dirname, 'index.html');
    if (fs.existsSync(rootHtml)) {
      res.sendFile(rootHtml);
    } else {
      res.send('<h1>VectorDB Platform — React Frontend Building...</h1>');
    }
  });
}

// Start Server
if (process.env.NODE_ENV !== 'test') {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`=== VectorDB 3-Tier Platform (Node.js Gateway) ===`);
    console.log(`🌐 Public Gateway: http://localhost:${PORT}`);
    console.log(`📖 Swagger Docs:  http://localhost:${PORT}/docs`);
    console.log(`🧠 AI Microservice: ${AI_SERVICE_URL}`);
  });
}

export default app;
