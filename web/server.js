#!/usr/bin/env node
/**
 * Node.js management & monitoring dashboard for llm-inference-benchmark.
 * Zero external npm dependencies; uses Node.js built-in `http` module only.
 * See web/SPEC.md for the API contract and design notes.
 */

'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');
const {spawn} = require('child_process');

/* ── Configuration ────────────────────────────────────────────────────────── */

const HOST = process.env.LLM_BENCHMARK_WEB_HOST || '127.0.0.1';
const PORT = parseInt(process.env.LLM_BENCHMARK_WEB_PORT, 10) || 3000;
const MAX_LOG_LINES = 300;
const MAX_RUN_EVENTS = 400;
const MAX_HTTP_BODY = 20 * 1024 * 1024;
const UTF8 = 'utf-8';

const WEB_DIR = __dirname;
const STATIC_DIR = path.join(WEB_DIR, 'public');
const RUNS_DIR = path.join(WEB_DIR, 'runs');
const REPO_ROOT = path.join(WEB_DIR, '..');
const BENCHMARK_SCRIPT = path.join(REPO_ROOT, 'benchmark', 'benchmark.py');
const PYTHON_BIN = path.join(REPO_ROOT, '.venv', 'bin', 'python');
const EXAMPLES_DIR = path.join(REPO_ROOT, 'examples');

/* ── State ────────────────────────────────────────────────────────────────── */

/** Map of run_id → RunRecord. Survives the server lifetime, not restarts. */
const runs = new Map();
let runCounter = 0;
/** Ring buffer of the last N ingest events (regardless of run). */
const globalEventLog = [];
/** Server-Sent Events subscribers. */
const sseClients = new Set();

/* ── Utilities ────────────────────────────────────────────────────────────── */

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.js':   'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg':  'image/svg+xml',
  '.ico':  'image/x-icon',
  '.png':  'image/png',
  '.woff2':'font/woff2',
};

/** HTTP error class used to send structured REST errors. */
class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function sendJson(res, status, body) {
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Access-Control-Allow-Origin': '*',
  });
  res.end(JSON.stringify(body));
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > MAX_HTTP_BODY) {
        reject(new HttpError(413, 'payload too large'));
        req.pause();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      if (chunks.length === 0) {
        resolve({});
        return;
      }
      const raw = Buffer.concat(chunks).toString(UTF8);
      try {
        resolve(JSON.parse(raw));
      } catch {
        reject(new HttpError(400, 'invalid JSON'));
      }
    });
    req.on('error', (err) => reject(err));
  });
}

function serveFile(res, filePath) {
  fs.readFile(filePath, (err, buf) => {
    if (err) {
      res.writeHead(404, {'Content-Type': 'text/plain; charset=utf-8'});
      res.end('Not found');
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Cache-Control': 'no-cache',
    });
    res.end(buf);
  });
}

function log(message) {
  console.log(`[${new Date().toISOString().slice(11, 19)}] ${message}`);
}

/* ── Run record & lifecycle ───────────────────────────────────────────────── */

function newRunId() {
  runCounter += 1;
  const stamp = new Date().toISOString().replace(/[-:TZ]/g, '').slice(0, 12);
  return `r${stamp}-${runCounter}`;
}

function makeRun(config) {
  const id = newRunId();
  const run = {
    id,
    name: (config && config.name) || id,
    status: 'queued',
    config: config || {},
    configPath: null,
    reportPath: null,
    pid: null,
    started_at: null,
    finished_at: null,
    exit_code: null,
    case_position: 0,
    case_total: Array.isArray(config.cases) ? config.cases.length : 0,
    current_case: null,
    stage: null,
    round: null,
    events: [],
    logs: [],
    final_report: null,
    _child: null,
    _token: id,
  };
  runs.set(id, run);
  return run;
}

function activeRun() {
  for (const run of runs.values()) {
    if (run.status === 'queued' || run.status === 'running') return run;
  }
  return null;
}

function findRunByToken(token) {
  for (const run of runs.values()) {
    if (run._token && run._token === token) return run;
  }
  return null;
}

function appendLog(run, stream, line) {
  if (!run || !line || !line.trim()) return;
  run.logs.push({stream, line, at: new Date().toISOString()});
  if (run.logs.length > MAX_LOG_LINES) {
    run.logs.splice(0, run.logs.length - MAX_LOG_LINES);
  }
}

function appendIngest(run, payload) {
  const event = Object.assign({receivedAt: new Date().toISOString()}, payload);
  if (run) {
    run.events.push(event);
    if (run.events.length > MAX_RUN_EVENTS) run.events.splice(0, run.events.length - MAX_RUN_EVENTS);
  }
  globalEventLog.push(Object.assign({run_id: run ? run.id : null}, event));
  if (globalEventLog.length > MAX_RUN_EVENTS) globalEventLog.splice(0, globalEventLog.length - MAX_RUN_EVENTS);
  return event;
}

function ensureRunsDir() {
  if (!fs.existsSync(RUNS_DIR)) fs.mkdirSync(RUNS_DIR, {recursive: true});
}

function startRun(run) {
  ensureRunsDir();
  run.configPath = path.join(RUNS_DIR, `${run.id}-config.json`);
  run.reportPath = path.join(RUNS_DIR, `${run.id}-report.json`);
  run._token = run.id;
  run.started_at = new Date().toISOString();
  run.status = 'running';

  try {
    fs.writeFileSync(run.configPath, JSON.stringify(run.config, null, 2), UTF8);
  } catch (err) {
    run.status = 'failed';
    run.finished_at = new Date().toISOString();
    appendLog(run, 'error', `Failed to write config: ${err.message}`);
    return run;
  }

  const childEnv = {...process.env, LLM_BENCHMARK_WEB_TOKEN: run._token};
  const args = [
    BENCHMARK_SCRIPT,
    '--config', run.configPath,
    '--report', run.reportPath,
    '--progress', 'off',
    '--web-port', String(PORT),
    '--web-host', HOST,
  ];

  try {
    run._child = spawn(PYTHON_BIN, args, {cwd: REPO_ROOT, env: childEnv});
  } catch (err) {
    run.status = 'failed';
    run.finished_at = new Date().toISOString();
    appendLog(run, 'error', `Spawn failed: ${err.message}`);
    return run;
  }

  run.pid = run._child.pid || null;

  run._child.stdout.on('data', (chunk) => {
    for (const line of String(chunk).split('\n')) {
      appendLog(run, 'stdout', line);
    }
    broadcast('log', {run_id: run.id});
  });
  run._child.stderr.on('data', (chunk) => {
    for (const line of String(chunk).split('\n')) {
      appendLog(run, 'stderr', line);
    }
    broadcast('log', {run_id: run.id});
  });
  run._child.on('exit', (code, signal) => {
    run._child = null;
    run.pid = null;
    run.exit_code = code;
    run.finished_at = new Date().toISOString();
    if (run.status === 'running' || run.status === 'queued') {
      if (signal === 'SIGTERM' || signal === 'SIGKILL') run.status = 'interrupted';
      else if (code === 0) run.status = 'passed';
      else run.status = 'failed';
    }
    loadFinalReport(run);
    log(` run ${run.id} exited code=${code} signal=${signal}`);
    broadcast('snapshot', snapshot());
  });
  run._child.on('error', (err) => {
    appendLog(run, 'error', `Spawn error: ${err.message}`);
    run.status = 'failed';
    run.finished_at = new Date().toISOString();
  });

  return run;
}

function loadFinalReport(run) {
  if (!run.reportPath || !fs.existsSync(run.reportPath)) return;
  try {
    run.final_report = JSON.parse(fs.readFileSync(run.reportPath, UTF8));
  } catch {
    run.final_report = null;
  }
}

function killRun(run) {
  const child = run && run._child;
  if (!child) return false;
  try {
    child.kill('SIGTERM');
  } catch {
    return false;
  }
  setTimeout(() => {
    try {
      if (run._child && run._child.exitCode === null) run._child.kill('SIGKILL');
    } catch {
      // already gone
    }
  }, 3000);
  return true;
}

/** Forget a finished run entirely; if active, signals it first. */
function forgetRun(run) {
  killRun(run);
  runs.delete(run.id);
  for (let i = globalEventLog.length - 1; i >= 0; i -= 1) {
    if (globalEventLog[i].run_id === run.id) {
      globalEventLog.splice(i, 1);
    }
  }
}

/** Fully clear local state; used by POST /api/reset. */
function markIdle() {
  for (const run of runs.values()) killRun(run);
  runs.clear();
  globalEventLog.splice(0, globalEventLog.length);
}

/* ── Ingest routing: map the bearer token back to the originating run ─────── */

function applyIngest(payload, token) {
  const run = token ? findRunByToken(token) : null;
  const event = appendIngest(run, payload);
  if (!run) return event;

  switch (payload && payload.event) {
    case 'case_started':
      run.status = 'running';
      run.current_case = {name: payload.case_name || null, scenario: payload.scenario || null};
      run.case_position = payload.position || 0;
      run.case_total = payload.total_cases || run.case_total;
      run.stage = null;
      break;
    case 'case_finished':
      run.current_case = null;
      run.stage = null;
      run.case_position = payload.completed_cases || run.case_position;
      break;
    case 'stage_started':
      run.stage = payload.stage || null;
      break;
    case 'stage_finished':
      run.stage = null;
      break;
    case 'round_started':
      run.round = {
        total_requests: payload.round_total || 0,
        concurrency: payload.concurrency || 0,
        completed: 0,
        succeeded: 0,
        failed: 0,
      };
      break;
    case 'request_finished':
      if (run.round) {
        run.round.completed = payload.completed || 0;
        run.round.succeeded = payload.succeeded || 0;
        run.round.failed = payload.failed || 0;
      }
      break;
    case 'round_finished':
      run.round = null;
      break;
    case 'final_results':
      run.final_report = payload.report || null;
      if (payload.report_location) run.reportPath = payload.report_location;
      if (run.final_report && run.final_report.suite && run.final_report.suite.name) {
        run.name = run.final_report.suite.name;
      }
      break;
    default:
      break;
  }
  return event;
}

/* ── Snapshot for SSE / HTTP ──────────────────────────────────────────────── */

function summarizeRun(run) {
  return {
    id: run.id,
    name: run.name,
    status: run.status,
    case_position: run.case_position,
    case_total: run.case_total,
    current_case: run.current_case,
    stage: run.stage,
    round: run.round,
    started_at: run.started_at,
    finished_at: run.finished_at,
    exit_code: run.exit_code,
    has_report: Boolean(run.final_report),
  };
}

function snapshot() {
  const active = activeRun();
  const runsList = [...runs.values()].map(summarizeRun).reverse();
  return {
    active_run_id: active ? active.id : null,
    active_run: active ? summarizeRun(active) : null,
    runs: runsList,
    events: globalEventLog.slice(-100),
    total_runs: runs.size,
  };
}

/* ── Example and report helpers ───────────────────────────────────────────── */

const REPORT_PATTERNS = [/^benchmark-report.*\.json$/, /^.*-report-.*\.json$/];

function isReportName(name) {
  if (typeof name !== 'string') return false;
  if (!name.endsWith('.json')) return false;
  if (name.includes('..') || name.includes('/') || name.includes('\\')) return false;
  return REPORT_PATTERNS.some((re) => re.test(name));
}

function findReportPath(name) {
  const candidates = [path.join(REPO_ROOT, name), path.join(RUNS_DIR, name)];
  for (const candidate of candidates) {
    try {
      fs.accessSync(candidate, fs.constants.R_OK);
      return candidate;
    } catch {
      continue;
    }
  }
  return null;
}

function listReports() {
  const seen = new Map();
  for (const dir of [REPO_ROOT, RUNS_DIR]) {
    let entries;
    try {
      entries = fs.readdirSync(dir);
    } catch {
      continue;
    }
    for (const name of entries) {
      if (!isReportName(name) || seen.has(name)) continue;
      const full = path.join(dir, name);
      try {
        const stat = fs.statSync(full);
        seen.set(name, {file: name, size: stat.size, modifiedAt: stat.mtime.toISOString()});
      } catch {
        // vanished between readdir and stat
      }
    }
  }
  return [...seen.values()].sort((a, b) => b.modifiedAt.localeCompare(a.modifiedAt));
}

function listExamples() {
  const out = [];
  try {
    const names = fs.readdirSync(EXAMPLES_DIR).filter((n) => n.endsWith('.json'));
    for (const name of names) {
      const full = path.join(EXAMPLES_DIR, name);
      try {
        const parsed = JSON.parse(fs.readFileSync(full, UTF8));
        out.push({
          file: name,
          name: parsed.name || name,
          description: parsed.description || '',
          cases: Array.isArray(parsed.cases) ? parsed.cases.length : 0,
          config: parsed,
        });
      } catch {
        // malformed example file; skip
      }
    }
  } catch {
    // examples directory may be absent in installed wheel
  }
  return out;
}

/* ── SSE broadcast ────────────────────────────────────────────────────────── */

function broadcast(eventName, data) {
  for (const client of sseClients) {
    try {
      client.res.write(`event: ${eventName}\n`);
      client.res.write(`data: ${JSON.stringify(data)}\n\n`);
    } catch {
      sseClients.delete(client);
    }
  }
}

/* ── HTTP routing ─────────────────────────────────────────────────────────── */

async function handleApi(req, res, pathname) {
  const method = req.method;
  const runMatch = pathname.match(/^\/api\/runs\/([^/]+)$/);

  // POST /ingest — lifecycle events pushed by the Python benchmark.
  if (method === 'POST' && pathname === '/ingest') {
    const expected = process.env.LLM_BENCHMARK_WEB_TOKEN;
    const authHeader = req.headers.authorization || '';
    const bearerToken = authHeader.replace(/^Bearer\s+/i, '') || null;
    if (expected && bearerToken !== expected) {
      sendJson(res, 403, {error: 'token mismatch'});
      return true;
    }
    let payload;
    try {
      payload = await readJsonBody(req);
    } catch (err) {
      sendJson(res, err.status || 400, {error: err.message});
      return true;
    }
    const event = applyIngest(payload || {}, bearerToken);
    broadcast('snapshot', snapshot());
    sendJson(res, 200, {ok: true, run_id: event.run_id || null});
    return true;
  }

  if (method === 'POST' && pathname === '/api/reset') {
    markIdle();
    broadcast('snapshot', snapshot());
    sendJson(res, 200, {ok: true});
    return true;
  }

  if (method === 'GET' && pathname === '/api/status') {
    sendJson(res, 200, snapshot());
    return true;
  }

  if (method === 'POST' && pathname === '/api/runs') {
    let body;
    try {
      body = await readJsonBody(req);
    } catch (err) {
      sendJson(res, err.status || 400, {error: err.message});
      return true;
    }
    const config = body && body.config;
    if (!config || typeof config !== 'object' || Array.isArray(config)) {
      sendJson(res, 400, {error: 'body must be a JSON object with a "config" field'});
      return true;
    }
    if (!Array.isArray(config.cases)) {
      sendJson(res, 400, {error: 'config.cases must be an array'});
      return true;
    }
    const run = makeRun(config);
    startRun(run);
    log(`▶ start run ${run.id}`);
    broadcast('snapshot', snapshot());
    sendJson(res, 201, {id: run.id, status: run.status});
    return true;
  }

  if (method === 'GET' && pathname === '/api/runs') {
    sendJson(res, 200, {runs: [...runs.values()].map(summarizeRun).reverse()});
    return true;
  }

  if (method === 'GET' && runMatch) {
    const id = decodeURIComponent(runMatch[1]);
    const run = runs.get(id);
    if (!run) {
      sendJson(res, 404, {error: 'run not found'});
      return true;
    }
    sendJson(res, 200, {
      ...run,
      _child: undefined,
      _token: undefined,
      events: run.events.slice(-100),
      logs: run.logs.slice(-100),
    });
    return true;
  }

  if (method === 'DELETE' && runMatch) {
    const id = decodeURIComponent(runMatch[1]);
    const run = runs.get(id);
    if (!run) {
      sendJson(res, 404, {error: 'run not found'});
      return true;
    }
    forgetRun(run);
    broadcast('snapshot', snapshot());
    sendJson(res, 200, {ok: true});
    return true;
  }

  if (method === 'GET' && pathname === '/api/examples') {
    sendJson(res, 200, {examples: listExamples()});
    return true;
  }

  if (method === 'GET' && pathname === '/api/reports') {
    sendJson(res, 200, {reports: listReports()});
    return true;
  }

  const reportMatch = pathname.match(/^\/api\/reports\/(.+)$/);
  if (method === 'GET' && reportMatch) {
    const name = decodeURIComponent(reportMatch[1]);
    const filePath = findReportPath(name);
    if (!filePath) {
      sendJson(res, 404, {error: 'report not found'});
      return true;
    }
    try {
      const parsed = JSON.parse(fs.readFileSync(filePath, UTF8));
      sendJson(res, 200, parsed);
    } catch {
      sendJson(res, 502, {error: 'report contains invalid JSON'});
    }
    return true;
  }

  sendJson(res, 404, {error: 'not found'});
  return true;
}

/* ── HTTP server wiring ───────────────────────────────────────────────────── */

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${HOST}:${PORT}`);
  const pathname = url.pathname;

  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    });
    res.end();
    return;
  }

  if (req.method === 'GET' && pathname === '/events') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'Access-Control-Allow-Origin': '*',
      'X-Accel-Buffering': 'no',
    });
    const client = {res};
    sseClients.add(client);
    res.write(`event: snapshot\ndata: ${JSON.stringify(snapshot())}\n\n`);
    const keepalive = setInterval(() => {
      try {
        res.write(': ping\n\n');
      } catch {
        clearInterval(keepalive);
      }
    }, 15000);
    req.on('close', () => {
      clearInterval(keepalive);
      sseClients.delete(client);
    });
    return;
  }

  if (req.method === 'GET' && (pathname === '/' || pathname === '/index.html')) {
    serveFile(res, path.join(STATIC_DIR, 'index.html'));
    return;
  }

  if (pathname.startsWith('/api/') || pathname === '/ingest') {
    await handleApi(req, res, pathname.replace(/\/+$/, ''));
    return;
  }

  if (req.method === 'GET') {
    const resolved = path.resolve(STATIC_DIR, '.' + pathname);
    if (!resolved.startsWith(STATIC_DIR)) {
      sendJson(res, 403, {error: 'forbidden'});
      return;
    }
    serveFile(res, resolved);
    return;
  }

  sendJson(res, 404, {error: 'not found'});
});

function checkEnvironment() {
  const problems = [];
  if (!fs.existsSync(PYTHON_BIN)) {
    problems.push(`未找到 Python 解释器 ${PYTHON_BIN}，请先在仓库根目录运行 uv venv 创建虚拟环境。`);
  }
  if (!fs.existsSync(BENCHMARK_SCRIPT)) {
    problems.push(`未找到 benchmark 入口 ${BENCHMARK_SCRIPT}，请确认 web/ 目录位于仓库根目录内。`);
  }
  for (const problem of problems) {
    log(`⚠️  ${problem}`);
  }
  return problems.length === 0;
}

if (require.main === module) {
  const isLoopback = HOST === 'localhost' || HOST.startsWith('127.') || HOST === '::1';
  if (!isLoopback) {
    log('⚠️  监听地址不是环回地址，请确认防火墙与暴露范围。');
  }
  checkEnvironment();
  ensureRunsDir();
  server.listen(PORT, HOST, () => {
    log(`🌐 Dashboard listening on http://${HOST}:${PORT}`);
    log(`   Ingest:  POST /ingest`);
    log(`   SSE:     GET /events`);
    log(`   API:     GET /api/...`);
  });
}

module.exports = {
  server,
  applyIngest,
  snapshot,
  listReports,
  listExamples,
  makeRun,
  markIdle,
  isReportName,
  findReportPath,
  summarizeRun,
};
