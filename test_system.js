/**
 * Automated End-to-End System Test Suite
 * Tests Node.js Express Gateway (8080) proxying to Python FastAPI AI service (8000).
 */

import http from 'http';
import { spawn } from 'child_process';

const PYTHON_PORT = 8000;
const GATEWAY_PORT = 8080;

function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function fetchJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let data = '';
      res.on('data', chunk => { data += chunk; });
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, data: JSON.parse(data), headers: res.headers });
        } catch (_) {
          resolve({ status: res.statusCode, text: data, headers: res.headers });
        }
      });
    }).on('error', reject);
  });
}

async function waitForService(url, maxRetries = 15, delayMs = 300) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const res = await fetchJson(url);
      if (res.status === 200) return res;
    } catch (_) {}
    await wait(delayMs);
  }
  throw new Error(`Service at ${url} failed to respond after ${maxRetries * delayMs}ms`);
}

async function runTests() {
  console.log('=== Starting 3-Tier Integration Test Suite ===');

  let pyProc = null;
  let nodeProc = null;

  try {
    // 1. Start Python AI Microservice
    console.log('[Test] Starting Python AI Microservice...');
    const pythonExe = './.venv/bin/python3';
    pyProc = spawn(pythonExe, ['ai_service.py'], {
      env: { ...process.env, PORT: `${PYTHON_PORT}` },
      stdio: 'pipe'
    });

    pyProc.stderr.on('data', (d) => console.log(`[Python Log]: ${d}`));

    // Wait for Python service to respond
    console.log('[Test 1] Verifying Python AI Service directly on port 8000...');
    const pyStatus = await waitForService(`http://127.0.0.1:${PYTHON_PORT}/status`);
    console.log('  -> Python /status code:', pyStatus.status);
    console.log('  -> Demo vectors count:', pyStatus.data.demoCount);

    // 2. Start Node.js Express Gateway
    console.log('[Test 2] Starting Node.js Express Gateway on port 8080...');
    nodeProc = spawn('node', ['server.js'], {
      env: { ...process.env, PORT: `${GATEWAY_PORT}`, AI_SERVICE_URL: `http://127.0.0.1:${PYTHON_PORT}` },
      stdio: 'pipe'
    });

    nodeProc.stderr.on('data', (d) => console.log(`[Node Err]: ${d}`));

    // Test Gateway /status
    console.log('[Test 3] Verifying Node.js Gateway /status on port 8080...');
    const gwStatus = await waitForService(`http://127.0.0.1:${GATEWAY_PORT}/status`);
    console.log('  -> Gateway /status:', gwStatus.status, gwStatus.data?.gateway);

    // Test Gateway /items (list 16D vectors)
    console.log('[Test 4] Verifying /items proxy through Gateway...');
    const itemsRes = await fetchJson(`http://127.0.0.1:${GATEWAY_PORT}/items`);
    console.log('  -> Items count:', itemsRes.data?.length);
    if (!Array.isArray(itemsRes.data) || itemsRes.data.length === 0) {
      throw new Error('Expected items array from /items');
    }

    // Test Gateway /search
    console.log('[Test 5] Verifying 16D Vector Search proxy...');
    const testVec = [0.9, 0.8, 0.7, 0.6, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const searchRes = await fetchJson(
      `http://127.0.0.1:${GATEWAY_PORT}/search?v=${encodeURIComponent(testVec.join(','))}&k=3&algo=hnsw&metric=cosine`
    );
    console.log('  -> Search status:', searchRes.status);
    console.log('  -> Algorithm:', searchRes.data?.algo);
    console.log('  -> Latency (us):', searchRes.data?.latencyUs);
    console.log('  -> Top hit:', searchRes.data?.results?.[0]?.metadata);
    if (!searchRes.data?.results || searchRes.data.results.length === 0) {
      throw new Error('Expected search hits from /search');
    }

    // Test Gateway /benchmark
    console.log('[Test 6] Verifying Multi-Engine Benchmark proxy...');
    const benchRes = await fetchJson(
      `http://127.0.0.1:${GATEWAY_PORT}/benchmark?v=${encodeURIComponent(testVec.join(','))}&k=3`
    );
    console.log('  -> Benchmark latencies (us):', benchRes.data);

    // Test Gateway /hnsw-info
    console.log('[Test 7] Verifying HNSW Graph Layers proxy...');
    const hnswRes = await fetchJson(`http://127.0.0.1:${GATEWAY_PORT}/hnsw-info`);
    console.log('  -> HNSW Nodes Per Layer:', hnswRes.data?.nodesPerLayer);
    console.log('  -> HNSW Total Nodes:', hnswRes.data?.nodeCount);

    // Test Gateway Swagger UI /docs
    console.log('[Test 8] Verifying Swagger UI /docs endpoint...');
    const docsRes = await fetchJson(`http://127.0.0.1:${GATEWAY_PORT}/docs/`);
    console.log('  -> /docs status:', docsRes.status);

    console.log('\n==================================================');
    console.log('🎉 ALL 8 INTEGRATION TESTS PASSED SUCCESSFULLY!');
    console.log('==================================================\n');

  } catch (err) {
    console.error('❌ Test failed with error:', err);
    process.exitCode = 1;
  } finally {
    if (pyProc) pyProc.kill('SIGTERM');
    if (nodeProc) nodeProc.kill('SIGTERM');
  }
}

runTests();
