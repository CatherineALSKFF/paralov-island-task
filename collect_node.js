#!/usr/bin/env node
/**
 * Node.js Replay Collector — Runs independently of browser
 *
 * Usage:
 *   node collect_node.js --token <JWT> [--round <id>] [--target 10000] [--concurrency 20]
 *
 * Get token: In browser console on app.ainm.no, run: _M.getToken()
 * Or: DevTools → Application → Cookies → copy access_token value
 *
 * Data saves to: ./replay_data_<round_id>.json
 * Graceful shutdown: Ctrl+C saves immediately
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

// ═══════════════════════════════════════════════════════════════════════
// CONFIG
// ═══════════════════════════════════════════════════════════════════════
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5;

// Parse args
const args = {};
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, '');
  args[key] = process.argv[i + 1];
}

const TOKEN = args.token || process.env.ASTAR_TOKEN;
const TARGET = parseInt(args.target || '10000');
const CONCURRENCY = parseInt(args.concurrency || '15');
const ROUND_ID = args.round || null;

if (!TOKEN) {
  console.error('Usage: node collect_node.js --token <JWT> [--round <id>] [--target 10000] [--concurrency 15]');
  console.error('');
  console.error('Get token: browser console → _M.getToken()');
  console.error('Or: DevTools → Application → Cookies → access_token');
  process.exit(1);
}

// ═══════════════════════════════════════════════════════════════════════
// HTTP HELPERS
// ═══════════════════════════════════════════════════════════════════════
function apiGet(endpoint) {
  return new Promise((resolve, reject) => {
    const url = new URL(BASE + endpoint);
    const opts = {
      hostname: url.hostname,
      path: url.pathname,
      method: 'GET',
      headers: {
        'Authorization': 'Bearer ' + TOKEN,
        'Accept': 'application/json'
      }
    };
    const req = https.request(opts, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        try { resolve(JSON.parse(body)); }
        catch (e) { reject(new Error('Parse error: ' + body.slice(0, 200))); }
      });
    });
    req.on('error', reject);
    req.setTimeout(10000, () => { req.destroy(); reject(new Error('timeout')); });
    req.end();
  });
}

function apiPost(endpoint, data) {
  return new Promise((resolve, reject) => {
    const url = new URL(BASE + endpoint);
    const payload = JSON.stringify(data);
    const opts = {
      hostname: url.hostname,
      path: url.pathname,
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + TOKEN,
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Content-Length': Buffer.byteLength(payload)
      }
    };
    const req = https.request(opts, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        try { resolve(JSON.parse(body)); }
        catch (e) { reject(new Error('Parse error: ' + body.slice(0, 200))); }
      });
    });
    req.on('error', reject);
    req.setTimeout(15000, () => { req.destroy(); reject(new Error('timeout')); });
    req.write(payload);
    req.end();
  });
}

// ═══════════════════════════════════════════════════════════════════════
// TERRAIN MAPPING
// ═══════════════════════════════════════════════════════════════════════
function t2c(t) {
  return (t === 10 || t === 11 || t === 0) ? 0 : ((t >= 1 && t <= 5) ? t : 0);
}

// ═══════════════════════════════════════════════════════════════════════
// ACCUMULATOR
// ═══════════════════════════════════════════════════════════════════════
let acc = {};
let roundId = ROUND_ID;
let dataFile = null;

function initSeed(s) {
  if (acc[s]) return;
  acc[s] = { count: 0, grid: [] };
  for (let y = 0; y < H; y++) {
    acc[s].grid[y] = [];
    for (let x = 0; x < W; x++) acc[s].grid[y][x] = [0, 0, 0, 0, 0, 0];
  }
}

function addReplay(seed, data) {
  if (!data || !data.frames || data.frames.length < 2) return false;
  initSeed(seed);
  const f = data.frames[data.frames.length - 1];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      acc[seed].grid[y][x][t2c(f.grid[y][x])]++;
    }
  }
  acc[seed].count++;
  return true;
}

function saveData() {
  if (!dataFile) return;
  fs.writeFileSync(dataFile, JSON.stringify(acc));
}

function loadData() {
  if (dataFile && fs.existsSync(dataFile)) {
    try {
      acc = JSON.parse(fs.readFileSync(dataFile, 'utf8'));
      const cts = [];
      for (let s = 0; s < SEEDS; s++) cts.push(acc[s] ? acc[s].count : 0);
      console.log('Loaded existing data: [' + cts.join(',') + ']');
    } catch (e) {
      console.log('Could not load existing data, starting fresh');
      acc = {};
    }
  }
}

function getCounts() {
  return Array.from({length: SEEDS}, (_, s) => acc[s] ? acc[s].count : 0);
}

// ═══════════════════════════════════════════════════════════════════════
// COLLECTION ENGINE
// ═══════════════════════════════════════════════════════════════════════
let running = true;
let inflight = 0;
let totalCollected = 0;
let errors = 0;
let startTime = Date.now();

function nextSeed() {
  let minN = Infinity, minS = 0;
  for (let s = 0; s < SEEDS; s++) {
    const n = acc[s] ? acc[s].count : 0;
    if (n < minN) { minN = n; minS = s; }
  }
  return minS;
}

function needsMore() {
  for (let s = 0; s < SEEDS; s++) {
    if (!acc[s] || acc[s].count < TARGET) return true;
  }
  return false;
}

async function fetchOne() {
  if (!running || !needsMore()) return;
  const seed = nextSeed();
  inflight++;
  try {
    const data = await apiPost('/replay', { round_id: roundId, seed_index: seed });
    if (data && data.seed_index >= 0 && data.frames) {
      addReplay(data.seed_index, data);
      totalCollected++;
    } else {
      errors++;
      if (data && data.error) {
        console.error('API error:', data.error);
        if (data.error.includes('auth') || data.error.includes('token') || data.error.includes('unauthorized')) {
          console.error('AUTH FAILED — token may have expired. Stopping.');
          running = false;
        }
      }
    }
  } catch (e) {
    errors++;
    if (errors > 50) {
      console.error('Too many errors (' + errors + '), pausing 5s...');
      await new Promise(r => setTimeout(r, 5000));
      errors = 0;
    }
  }
  inflight--;
}

async function collect() {
  console.log('Starting collection: target=' + TARGET + '/seed, concurrency=' + CONCURRENCY);
  console.log('');

  const printInterval = setInterval(() => {
    const elapsed = (Date.now() - startTime) / 1000;
    const rate = totalCollected / elapsed;
    const cts = getCounts();
    const minCount = Math.min(...cts);
    const remaining = Math.max(0, TARGET - minCount);
    const eta = rate > 0 ? (remaining * SEEDS / rate / 60).toFixed(1) : '?';
    console.log(
      '[' + elapsed.toFixed(0) + 's] +' + totalCollected +
      ' [' + cts.join(',') + '] ' +
      rate.toFixed(1) + '/s, errors=' + errors +
      ', ETA ~' + eta + 'min'
    );
  }, 10000);

  const saveInterval = setInterval(saveData, 30000);

  // Main loop
  while (running && needsMore()) {
    // Fill up to concurrency
    while (inflight < CONCURRENCY && running && needsMore()) {
      fetchOne(); // fire and forget — async
    }
    // Small yield to let promises resolve
    await new Promise(r => setTimeout(r, 50));
  }

  // Wait for inflight to finish
  while (inflight > 0) {
    await new Promise(r => setTimeout(r, 100));
  }

  clearInterval(printInterval);
  clearInterval(saveInterval);
  saveData();

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
  const cts = getCounts();
  console.log('');
  console.log('DONE! Total: ' + totalCollected + ' in ' + elapsed + 's');
  console.log('Final counts: [' + cts.join(',') + ']');
  console.log('Saved to: ' + dataFile);
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════
async function main() {
  console.log('═══════════════════════════════════════════════════');
  console.log('  Astar Island Replay Collector (Node.js)');
  console.log('═══════════════════════════════════════════════════');
  console.log('');

  // Find active round
  if (!roundId) {
    console.log('Detecting active round...');
    const rounds = await apiGet('/rounds');
    const active = rounds.find(r => r.status === 'active');
    const scoring = rounds.find(r => r.status === 'scoring');
    const completed = rounds.filter(r => r.status === 'completed').sort((a, b) => b.round_number - a.round_number);
    const round = active || scoring || completed[0] || rounds[rounds.length - 1];
    roundId = round.id;
    console.log('Round ' + round.round_number + ' (' + round.status + ') — ' + roundId.slice(0, 8));
  } else {
    console.log('Using round: ' + roundId.slice(0, 8));
  }

  dataFile = path.join(__dirname, 'replay_data_' + roundId.slice(0, 8) + '.json');
  console.log('Data file: ' + dataFile);

  // Load existing data
  loadData();

  // Check if already at target
  if (!needsMore()) {
    console.log('Already at target! Counts: [' + getCounts().join(',') + ']');
    return;
  }

  // Graceful shutdown
  process.on('SIGINT', () => {
    console.log('\nCtrl+C received — saving and exiting...');
    running = false;
    saveData();
    const cts = getCounts();
    console.log('Saved: [' + cts.join(',') + '] to ' + dataFile);
    process.exit(0);
  });

  // Test auth with a single replay
  console.log('Testing auth...');
  try {
    const test = await apiPost('/replay', { round_id: roundId, seed_index: 0 });
    if (test && test.frames) {
      addReplay(0, test);
      totalCollected++;
      console.log('Auth OK! Got replay with ' + test.frames.length + ' frames');
    } else {
      console.error('Auth test failed:', JSON.stringify(test).slice(0, 200));
      console.error('Token may be invalid or expired. Get a fresh one from the browser.');
      process.exit(1);
    }
  } catch (e) {
    console.error('Auth test error:', e.message);
    process.exit(1);
  }

  console.log('');
  await collect();
}

main().catch(e => {
  console.error('Fatal error:', e);
  saveData();
  process.exit(1);
});
