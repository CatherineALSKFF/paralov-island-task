#!/usr/bin/env node
/**
 * Astar Island R3 Solver — Comprehensive prediction engine
 *
 * Two modes:
 *   1. REPLAY MODE (completed rounds): Aggregate replay final-frames → empirical distributions
 *   2. MC MODE (active rounds): Learn transitions from completed rounds → MC simulate target round
 *
 * Usage:
 *   node solver_r3.js train   --token <JWT> [--target 5000] [--concurrency 20]
 *   node solver_r3.js predict --token <JWT> --round <id> [--sims 1000]
 *   node solver_r3.js replay  --token <JWT> --round <id> [--target 5000]
 *   node solver_r3.js score   --token <JWT> --round <id>
 *   node solver_r3.js analyze --token <JWT>
 *
 * train:   Collect replays from ALL completed rounds, build transition model
 * predict: Run MC sims on a target round using learned model, submit predictions
 * replay:  Collect replays for a completed round (pure replay approach)
 * score:   Score predictions against GT (completed rounds only)
 * analyze: Show detailed error analysis
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

// ═══════════════════════════════════════════════════════════════════════
// CONFIG
// ═══════════════════════════════════════════════════════════════════════
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5;
const DATA_DIR = __dirname;

// Parse CLI args
const args = {};
const CMD = process.argv[2] || 'help';
for (let i = 3; i < process.argv.length; i += 2) {
  if (process.argv[i] && process.argv[i].startsWith('--')) {
    args[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
  }
}
const TOKEN = args.token || process.env.ASTAR_TOKEN;
const TARGET = parseInt(args.target || '5000');
const CONCURRENCY = parseInt(args.concurrency || '20');
const NSIMS = parseInt(args.sims || '1000');

// ═══════════════════════════════════════════════════════════════════════
// HTTP HELPERS
// ═══════════════════════════════════════════════════════════════════════
function apiRequest(method, endpoint, data) {
  return new Promise((resolve, reject) => {
    const url = new URL(BASE + endpoint);
    const payload = data ? JSON.stringify(data) : null;
    const opts = {
      hostname: url.hostname, path: url.pathname, method,
      headers: {
        'Authorization': 'Bearer ' + TOKEN,
        'Accept': 'application/json',
        ...(payload ? { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) } : {})
      }
    };
    const req = https.request(opts, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        try { resolve({ status: res.statusCode, data: JSON.parse(body) }); }
        catch (e) { reject(new Error(`Parse error (${res.statusCode}): ${body.slice(0, 200)}`)); }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('timeout')); });
    if (payload) req.write(payload);
    req.end();
  });
}

const api = {
  get: (ep) => apiRequest('GET', ep),
  post: (ep, data) => apiRequest('POST', ep, data),
};

// ═══════════════════════════════════════════════════════════════════════
// TERRAIN MAPPING
// ═══════════════════════════════════════════════════════════════════════
// Raw terrain codes: 0=empty, 1=settlement, 2=port, 3=ruin, 4=forest, 5=mountain, 10=ocean, 11=plains
// GT classes: 0=plains(0,10,11), 1=settlement, 2=port, 3=ruin, 4=forest, 5=mountain
function t2c(t) { return (t === 10 || t === 11 || t === 0) ? 0 : ((t >= 1 && t <= 5) ? t : 0); }
function isStatic(t) { return t === 10 || t === 5; } // ocean and mountain never change

// ═══════════════════════════════════════════════════════════════════════
// FEATURE EXTRACTION — The heart of the transition model
// ═══════════════════════════════════════════════════════════════════════

// Precompute neighbor offsets
const N8 = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
const N4 = [[-1,0],[1,0],[0,-1],[0,1]];

function extractFeatures(grid, y, x, year) {
  const terrain = grid[y][x];

  // 8-neighbor composition
  let nOcean = 0, nSettle = 0, nPort = 0, nForest = 0, nRuin = 0, nMtn = 0, nPlains = 0;
  for (const [dy, dx] of N8) {
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) { nOcean++; continue; } // out of bounds = ocean
    const t = grid[ny][nx];
    if (t === 10) nOcean++;
    else if (t === 1) nSettle++;
    else if (t === 2) nPort++;
    else if (t === 4) nForest++;
    else if (t === 3) nRuin++;
    else if (t === 5) nMtn++;
    else nPlains++; // 0, 11
  }

  // Distance to nearest ocean (Manhattan, max 5)
  let dOcean = 6;
  for (let r = 1; r <= 5 && dOcean > 5; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (Math.abs(dy) + Math.abs(dx) !== r) continue; // Manhattan distance exactly r
        const ny = y + dy, nx = x + dx;
        if (ny < 0 || ny >= H || nx < 0 || nx >= W) { dOcean = Math.min(dOcean, r); continue; }
        if (grid[ny][nx] === 10) dOcean = Math.min(dOcean, r);
      }
    }
  }

  // Settlements in radius 2 (Manhattan)
  let settleR2 = 0, forestR2 = 0;
  for (let dy = -2; dy <= 2; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      if (Math.abs(dy) + Math.abs(dx) > 2) continue;
      if (dy === 0 && dx === 0) continue;
      const ny = y + dy, nx = x + dx;
      if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
      const t = grid[ny][nx];
      if (t === 1 || t === 2) settleR2++;
      if (t === 4) forestR2++;
    }
  }

  // Settlements in radius 3
  let settleR3 = 0;
  for (let dy = -3; dy <= 3; dy++) {
    for (let dx = -3; dx <= 3; dx++) {
      if (Math.abs(dy) + Math.abs(dx) > 3) continue;
      if (dy === 0 && dx === 0) continue;
      const ny = y + dy, nx = x + dx;
      if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
      if (grid[ny][nx] === 1 || grid[ny][nx] === 2) settleR3++;
    }
  }

  // Phase (10-year bins)
  const phase = Math.min(4, Math.floor(year / 10));

  // Coastal flag (4-adjacent to ocean)
  const coastal = (nOcean > 0 || dOcean <= 1) ? 1 : 0;

  return {
    terrain, phase,
    nOcean, nSettle, nPort, nForest, nRuin, nMtn, nPlains,
    dOcean: Math.min(dOcean, 5),
    coastal,
    settleR2: Math.min(settleR2, 8),
    settleR3: Math.min(settleR3, 15),
    forestR2: Math.min(forestR2, 8),
    nSettleAll: nSettle + nPort, // combined settlement+port neighbors
  };
}

// Multi-level feature keys (from most to least specific)
function featureKeys(f) {
  const sbin2 = f.settleR2 === 0 ? 0 : f.settleR2 <= 2 ? 1 : f.settleR2 <= 4 ? 2 : 3;
  const sbin3 = f.settleR3 === 0 ? 0 : f.settleR3 <= 3 ? 1 : f.settleR3 <= 7 ? 2 : 3;
  const fbin = f.forestR2 === 0 ? 0 : f.forestR2 <= 3 ? 1 : 2;
  const nSA = Math.min(f.nSettleAll, 4);

  return [
    // Level 0: Most specific — terrain + phase + exact neighbor settlement count + coastal + ocean dist + settle density
    `L0_${f.terrain}_${f.phase}_${nSA}_${f.coastal}_${Math.min(f.dOcean, 3)}_${sbin2}_${fbin}`,
    // Level 1: terrain + phase + settlement neighbors + coastal + settle density
    `L1_${f.terrain}_${f.phase}_${Math.min(nSA, 3)}_${f.coastal}_${sbin2}`,
    // Level 2: terrain + phase + has settlement neighbor + coastal
    `L2_${f.terrain}_${f.phase}_${nSA > 0 ? 1 : 0}_${f.coastal}`,
    // Level 3: terrain + phase
    `L3_${f.terrain}_${f.phase}`,
    // Level 4: terrain only
    `L4_${f.terrain}`,
  ];
}

// ═══════════════════════════════════════════════════════════════════════
// TRANSITION MODEL
// ═══════════════════════════════════════════════════════════════════════
class TransitionModel {
  constructor() {
    this.tables = {};   // key → { outcomes: {terrainCode: count}, total: N }
    this.totalTransitions = 0;
    this.roundsProcessed = new Set();
  }

  // Record a single transition
  addTransition(keys, fromTerrain, toTerrain) {
    for (const key of keys) {
      if (!this.tables[key]) this.tables[key] = { outcomes: {}, total: 0 };
      const t = this.tables[key];
      t.outcomes[toTerrain] = (t.outcomes[toTerrain] || 0) + 1;
      t.total++;
    }
    this.totalTransitions++;
  }

  // Get transition probabilities with hierarchical fallback
  getProbs(keys, minSamples) {
    minSamples = minSamples || [50, 30, 20, 10, 5]; // min samples per level
    for (let level = 0; level < keys.length; level++) {
      const t = this.tables[keys[level]];
      const minN = minSamples[Math.min(level, minSamples.length - 1)];
      if (t && t.total >= minN) {
        const probs = {};
        for (const terrain in t.outcomes) {
          probs[parseInt(terrain)] = t.outcomes[terrain] / t.total;
        }
        return { probs, level, samples: t.total };
      }
    }
    return null; // no data at any level
  }

  // Process a single replay: extract all frame-to-frame transitions
  processReplay(replay) {
    if (!replay || !replay.frames || replay.frames.length < 2) return false;
    const frames = replay.frames;
    for (let f = 0; f < frames.length - 1; f++) {
      const curr = frames[f].grid;
      const next = frames[f + 1].grid;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          if (isStatic(curr[y][x])) continue; // skip ocean/mountain
          const features = extractFeatures(curr, y, x, f);
          const keys = featureKeys(features);
          this.addTransition(keys, curr[y][x], next[y][x]);
        }
      }
    }
    return true;
  }

  save(filepath) {
    const data = {
      tables: this.tables,
      totalTransitions: this.totalTransitions,
      roundsProcessed: [...this.roundsProcessed],
      keyCount: Object.keys(this.tables).length,
      timestamp: new Date().toISOString(),
    };
    fs.writeFileSync(filepath, JSON.stringify(data));
    console.log(`Model saved: ${Object.keys(this.tables).length} keys, ${this.totalTransitions} transitions → ${filepath}`);
  }

  load(filepath) {
    if (!fs.existsSync(filepath)) return false;
    try {
      const data = JSON.parse(fs.readFileSync(filepath, 'utf8'));
      this.tables = data.tables;
      this.totalTransitions = data.totalTransitions;
      this.roundsProcessed = new Set(data.roundsProcessed || []);
      console.log(`Model loaded: ${Object.keys(this.tables).length} keys, ${this.totalTransitions} transitions`);
      return true;
    } catch (e) {
      console.error('Failed to load model:', e.message);
      return false;
    }
  }

  stats() {
    const levels = {};
    for (const key in this.tables) {
      const level = key.split('_')[0];
      if (!levels[level]) levels[level] = { keys: 0, totalSamples: 0 };
      levels[level].keys++;
      levels[level].totalSamples += this.tables[key].total;
    }
    return { totalKeys: Object.keys(this.tables).length, totalTransitions: this.totalTransitions, levels };
  }
}

// ═══════════════════════════════════════════════════════════════════════
// MC SIMULATOR — Apply learned transitions to simulate forward
// ═══════════════════════════════════════════════════════════════════════
function mcSimulate(model, initialGrid, nSteps) {
  nSteps = nSteps || 50;

  // Deep copy the grid
  let grid = initialGrid.map(r => [...r]);

  for (let year = 0; year < nSteps; year++) {
    const newGrid = grid.map(r => [...r]);

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (isStatic(grid[y][x])) continue;

        const features = extractFeatures(grid, y, x, year);
        const keys = featureKeys(features);
        const result = model.getProbs(keys);

        if (!result) continue; // no data, keep current terrain

        // Sample from transition distribution
        const r = Math.random();
        let cum = 0;
        for (const terrain in result.probs) {
          cum += result.probs[terrain];
          if (r < cum) {
            newGrid[y][x] = parseInt(terrain);
            break;
          }
        }
      }
    }

    grid = newGrid;
  }

  return grid;
}

// Run multiple MC simulations and aggregate results
function runMC(model, initialGrid, nSims, onProgress) {
  const counts = [];
  for (let y = 0; y < H; y++) {
    counts[y] = [];
    for (let x = 0; x < W; x++) counts[y][x] = new Float32Array(6);
  }

  for (let sim = 0; sim < nSims; sim++) {
    const finalGrid = mcSimulate(model, initialGrid);

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        counts[y][x][t2c(finalGrid[y][x])]++;
      }
    }

    if (onProgress && (sim + 1) % 50 === 0) onProgress(sim + 1, nSims);
  }

  return { counts, nSims };
}

// ═══════════════════════════════════════════════════════════════════════
// REPLAY ACCUMULATOR — For pure replay-based predictions
// ═══════════════════════════════════════════════════════════════════════
class ReplayAccumulator {
  constructor() {
    this.seeds = {};
  }

  addReplay(seedIndex, replay) {
    if (!replay || !replay.frames || replay.frames.length < 2) return false;
    if (!this.seeds[seedIndex]) {
      this.seeds[seedIndex] = { count: 0, grid: [] };
      for (let y = 0; y < H; y++) {
        this.seeds[seedIndex].grid[y] = [];
        for (let x = 0; x < W; x++) this.seeds[seedIndex].grid[y][x] = new Float32Array(6);
      }
    }
    const f = replay.frames[replay.frames.length - 1]; // last frame = year 50
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        this.seeds[seedIndex].grid[y][x][t2c(f.grid[y][x])]++;
      }
    }
    this.seeds[seedIndex].count++;
    return true;
  }

  getCounts() {
    return Array.from({ length: SEEDS }, (_, s) => this.seeds[s] ? this.seeds[s].count : 0);
  }

  save(filepath) {
    // Convert Float32Arrays to regular arrays for JSON serialization
    const data = {};
    for (const s in this.seeds) {
      data[s] = { count: this.seeds[s].count, grid: [] };
      for (let y = 0; y < H; y++) {
        data[s].grid[y] = [];
        for (let x = 0; x < W; x++) {
          data[s].grid[y][x] = Array.from(this.seeds[s].grid[y][x]);
        }
      }
    }
    fs.writeFileSync(filepath, JSON.stringify(data));
  }

  load(filepath) {
    if (!fs.existsSync(filepath)) return false;
    try {
      const data = JSON.parse(fs.readFileSync(filepath, 'utf8'));
      for (const s in data) {
        this.seeds[s] = { count: data[s].count, grid: [] };
        for (let y = 0; y < H; y++) {
          this.seeds[s].grid[y] = [];
          for (let x = 0; x < W; x++) {
            this.seeds[s].grid[y][x] = new Float32Array(data[s].grid[y][x]);
          }
        }
      }
      console.log(`Accumulator loaded: [${this.getCounts().join(',')}]`);
      return true;
    } catch (e) {
      console.error('Failed to load accumulator:', e.message);
      return false;
    }
  }

  merge(other) {
    for (const s in other.seeds) {
      if (!this.seeds[s]) {
        this.seeds[s] = other.seeds[s];
      } else {
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            for (let c = 0; c < 6; c++) {
              this.seeds[s].grid[y][x][c] += other.seeds[s].grid[y][x][c];
            }
          }
        }
        this.seeds[s].count += other.seeds[s].count;
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════
// PREDICTION BUILDER — Per-cell adaptive Dirichlet smoothing
// ═══════════════════════════════════════════════════════════════════════
function buildPrediction(counts, N, initialGrid) {
  const pred = [];

  for (let y = 0; y < H; y++) {
    pred[y] = [];
    for (let x = 0; x < W; x++) {
      // Static cells: hard prediction, zero smoothing
      if (initialGrid) {
        const t = initialGrid[y][x];
        if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
        if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }
      }

      const c = counts[y][x];

      // Count distinct classes and compute entropy
      let nClasses = 0, maxCount = 0, totalCount = 0;
      for (let k = 0; k < 6; k++) {
        if (c[k] > 0) nClasses++;
        if (c[k] > maxCount) maxCount = c[k];
        totalCount += c[k];
      }

      if (totalCount === 0) {
        // No data — uniform (shouldn't happen with MC)
        pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
        continue;
      }

      // Empirical entropy
      let ent = 0;
      for (let k = 0; k < 6; k++) {
        if (c[k] > 0) {
          const p = c[k] / totalCount;
          ent -= p * Math.log(p);
        }
      }

      // Per-cell adaptive alpha
      // Base alpha decreases with more data: alpha = 0.15 * sqrt(150/N)
      const baseAlpha = Math.max(0.01, 0.15 * Math.sqrt(150 / N));
      let alpha;

      if (nClasses <= 1) {
        alpha = 0.0005; // essentially static — tiny smoothing
      } else if (nClasses === 2 && maxCount > totalCount * 0.97) {
        alpha = 0.001; // nearly static
      } else if (nClasses === 2 && maxCount > totalCount * 0.90) {
        alpha = 0.003;
      } else if (nClasses === 2) {
        alpha = Math.max(0.005, baseAlpha * 0.25);
      } else {
        // Dynamic cell: scale by entropy ratio
        const maxEnt = Math.log(6); // ~1.79
        const entRatio = Math.max(0.1, ent / maxEnt);
        alpha = baseAlpha * entRatio;
      }

      // Dirichlet smoothing
      const p = new Array(6);
      const total = totalCount + 6 * alpha;
      for (let k = 0; k < 6; k++) p[k] = (c[k] + alpha) / total;

      // Precision fix: round to 6 decimals, ensure sum = 1.0
      let sum = 0, maxIdx = 0, maxVal = 0;
      for (let k = 0; k < 6; k++) {
        p[k] = parseFloat(p[k].toFixed(6));
        sum += p[k];
        if (p[k] > maxVal) { maxVal = p[k]; maxIdx = k; }
      }
      p[maxIdx] = parseFloat((p[maxIdx] + (1.0 - sum)).toFixed(6));

      pred[y][x] = p;
    }
  }

  return pred;
}

// ═══════════════════════════════════════════════════════════════════════
// SCORER — Competition scoring formula
// ═══════════════════════════════════════════════════════════════════════
function score(pred, gt) {
  let totalKL = 0, totalEnt = 0, dynamicCells = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const g = gt[y][x];
      let ent = 0;
      for (let c = 0; c < 6; c++) {
        if (g[c] > 1e-6) ent -= g[c] * Math.log(g[c]);
      }
      if (ent < 0.01) continue; // skip static cells
      dynamicCells++;
      let kl = 0;
      for (let c = 0; c < 6; c++) {
        if (g[c] > 1e-6) kl += g[c] * Math.log(g[c] / Math.max(pred[y][x][c], 1e-15));
      }
      totalKL += Math.max(0, kl) * ent;
      totalEnt += ent;
    }
  }
  const wkl = totalEnt > 0 ? totalKL / totalEnt : 0;
  return {
    score: Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl))),
    wkl, dynamicCells, totalEnt
  };
}

// ═══════════════════════════════════════════════════════════════════════
// VALIDATION — Check prediction format
// ═══════════════════════════════════════════════════════════════════════
function validate(pred) {
  let errors = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let sum = 0;
      for (let c = 0; c < 6; c++) {
        if (pred[y][x][c] < 0) { errors++; }
        sum += pred[y][x][c];
      }
      if (Math.abs(sum - 1.0) > 0.01) { errors++; }
    }
  }
  return errors === 0;
}

// ═══════════════════════════════════════════════════════════════════════
// COLLECT & TRAIN — Fetch replays, build transition model
// ═══════════════════════════════════════════════════════════════════════
async function trainModel() {
  console.log('═══════════════════════════════════════════════════');
  console.log('  TRAIN: Collecting replays & building transition model');
  console.log('═══════════════════════════════════════════════════');

  if (!TOKEN) { console.error('--token required'); process.exit(1); }

  // Find completed rounds
  const { data: rounds } = await api.get('/rounds');
  const completed = rounds.filter(r => r.status === 'completed');
  console.log(`Found ${completed.length} completed rounds`);

  // Load existing model
  const model = new TransitionModel();
  const modelFile = path.join(DATA_DIR, 'transition_model.json');
  model.load(modelFile);

  // For each completed round, collect replays and process transitions
  for (const round of completed) {
    const rid = round.id;
    const ridShort = rid.slice(0, 8);
    console.log(`\nRound ${round.round_number} (${ridShort}): ${round.status}`);

    // Also maintain the replay accumulator for this round
    const accFile = path.join(DATA_DIR, `replay_data_${ridShort}.json`);
    const acc = new ReplayAccumulator();
    acc.load(accFile);
    const prevCounts = acc.getCounts();
    const minPrev = Math.min(...prevCounts);
    console.log(`  Existing replay data: [${prevCounts.join(',')}]`);

    // How many more replays do we need?
    const need = Math.max(0, TARGET - minPrev);
    if (need === 0 && model.roundsProcessed.has(rid)) {
      console.log(`  Already at target and model trained. Skipping.`);
      continue;
    }

    // Build replay queue (round-robin across seeds)
    const seedNeed = {};
    for (let s = 0; s < SEEDS; s++) {
      seedNeed[s] = Math.max(0, TARGET - (acc.seeds[s] ? acc.seeds[s].count : 0));
    }
    const totalNeed = Object.values(seedNeed).reduce((a, b) => a + b, 0);

    if (totalNeed === 0) {
      console.log(`  All seeds at target. Marking model as trained.`);
      model.roundsProcessed.add(rid);
      continue;
    }

    console.log(`  Need: ${totalNeed} more replays (target ${TARGET}/seed)`);

    // Test auth
    try {
      const test = await api.post('/replay', { round_id: rid, seed_index: 0 });
      if (!test.data || !test.data.frames) {
        console.error(`  Auth failed or replay unavailable:`, JSON.stringify(test.data).slice(0, 200));
        if (test.status === 403) { console.log('  Round replays not available (403). Skipping.'); continue; }
        process.exit(1);
      }
      // Process the test replay
      model.processReplay(test.data);
      acc.addReplay(test.data.seed_index, test.data);
    } catch (e) {
      console.error(`  Auth test error: ${e.message}`);
      process.exit(1);
    }

    // Parallel collection
    let inflight = 0, collected = 0, errors = 0;
    const startTime = Date.now();
    let running = true;

    // Graceful shutdown
    const sigHandler = () => {
      console.log('\nCtrl+C — saving...');
      running = false;
    };
    process.on('SIGINT', sigHandler);

    function nextSeed() {
      let minN = Infinity, minS = 0;
      for (let s = 0; s < SEEDS; s++) {
        const n = acc.seeds[s] ? acc.seeds[s].count : 0;
        if (n < TARGET && n < minN) { minN = n; minS = s; }
      }
      return minS;
    }

    function needsMore() {
      for (let s = 0; s < SEEDS; s++) {
        if (!acc.seeds[s] || acc.seeds[s].count < TARGET) return true;
      }
      return false;
    }

    async function fetchOne() {
      if (!running || !needsMore()) return;
      const seed = nextSeed();
      inflight++;
      try {
        const { data } = await api.post('/replay', { round_id: rid, seed_index: seed });
        if (data && data.frames && data.seed_index >= 0) {
          model.processReplay(data);
          acc.addReplay(data.seed_index, data);
          collected++;
        } else {
          errors++;
          if (data && data.detail && (data.detail.includes('auth') || data.detail.includes('token'))) {
            console.error('  AUTH FAILED — token expired. Stopping.');
            running = false;
          }
        }
      } catch (e) {
        errors++;
        if (errors > 100) {
          console.error(`  Too many errors (${errors}). Pausing 5s...`);
          await new Promise(r => setTimeout(r, 5000));
          errors = 0;
        }
      }
      inflight--;
    }

    // Progress printer
    const printInterval = setInterval(() => {
      const elapsed = (Date.now() - startTime) / 1000;
      const rate = collected / Math.max(elapsed, 1);
      const cts = acc.getCounts();
      const minCount = Math.min(...cts);
      const remaining = Math.max(0, TARGET - minCount);
      const eta = rate > 0 ? (remaining * SEEDS / rate / 60).toFixed(1) : '?';
      console.log(`  [${elapsed.toFixed(0)}s] +${collected} [${cts.join(',')}] ${rate.toFixed(1)}/s err=${errors} ETA~${eta}min`);
    }, 15000);

    // Save periodically
    const saveInterval = setInterval(() => {
      acc.save(accFile);
      model.save(modelFile);
    }, 60000);

    // Main collection loop
    while (running && needsMore()) {
      while (inflight < CONCURRENCY && running && needsMore()) {
        fetchOne();
      }
      await new Promise(r => setTimeout(r, 50));
    }

    // Wait for inflight
    while (inflight > 0) await new Promise(r => setTimeout(r, 100));

    clearInterval(printInterval);
    clearInterval(saveInterval);
    process.removeListener('SIGINT', sigHandler);

    // Final save
    acc.save(accFile);
    model.roundsProcessed.add(rid);
    model.save(modelFile);

    const cts = acc.getCounts();
    console.log(`  Done: +${collected} replays. Final: [${cts.join(',')}]`);
  }

  // Print model stats
  const stats = model.stats();
  console.log('\n═══ MODEL STATS ═══');
  console.log(`Total keys: ${stats.totalKeys}`);
  console.log(`Total transitions: ${stats.totalTransitions}`);
  for (const level in stats.levels) {
    console.log(`  ${level}: ${stats.levels[level].keys} keys, ${stats.levels[level].totalSamples} samples`);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// PREDICT — Run MC sims and submit
// ═══════════════════════════════════════════════════════════════════════
async function predict() {
  console.log('═══════════════════════════════════════════════════');
  console.log('  PREDICT: MC simulation + submit');
  console.log('═══════════════════════════════════════════════════');

  if (!TOKEN) { console.error('--token required'); process.exit(1); }

  // Load transition model
  const model = new TransitionModel();
  const modelFile = path.join(DATA_DIR, 'transition_model.json');
  if (!model.load(modelFile)) {
    console.error('No transition model found. Run "train" first.');
    process.exit(1);
  }

  // Find target round
  let roundId = args.round;
  let roundDetail;
  if (!roundId) {
    const { data: rounds } = await api.get('/rounds');
    const active = rounds.find(r => r.status === 'active');
    const scoring = rounds.find(r => r.status === 'scoring');
    const completed = rounds.filter(r => r.status === 'completed').sort((a, b) => b.round_number - a.round_number);
    const round = active || scoring || completed[0];
    roundId = round.id;
    console.log(`Auto-detected round ${round.round_number} (${round.status}): ${roundId.slice(0, 8)}`);
  }

  const { data: detail } = await api.get(`/rounds/${roundId}`);
  roundDetail = detail;
  console.log(`Round ${detail.round_number}: ${detail.map_width}x${detail.map_height}, ${detail.seeds_count} seeds`);

  // Check if we also have replay data for this round (completed = can use pure replay approach)
  const accFile = path.join(DATA_DIR, `replay_data_${roundId.slice(0, 8)}.json`);
  const acc = new ReplayAccumulator();
  const hasReplayData = acc.load(accFile);
  if (hasReplayData) {
    const cts = acc.getCounts();
    console.log(`Replay data available: [${cts.join(',')}]`);
  }

  // Run MC for each seed
  for (let seed = 0; seed < SEEDS; seed++) {
    console.log(`\n--- Seed ${seed} ---`);
    const initialGrid = detail.initial_states[seed].grid;

    let prediction;

    if (hasReplayData && acc.seeds[seed] && acc.seeds[seed].count >= 100) {
      // Pure replay approach (much better if available)
      const N = acc.seeds[seed].count;
      console.log(`  Using REPLAY data (N=${N}) — pure empirical approach`);
      prediction = buildPrediction(acc.seeds[seed].grid, N, initialGrid);
    } else {
      // MC simulation approach
      console.log(`  Running ${NSIMS} MC simulations...`);
      const startTime = Date.now();

      const { counts, nSims } = runMC(model, initialGrid, NSIMS, (done, total) => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
        process.stdout.write(`  MC: ${done}/${total} (${elapsed}s)\r`);
      });

      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`  MC complete: ${NSIMS} sims in ${elapsed}s`);

      // If we also have some replay data, blend MC + replay
      if (hasReplayData && acc.seeds[seed] && acc.seeds[seed].count >= 10) {
        const replayN = acc.seeds[seed].count;
        console.log(`  Blending MC (${NSIMS}) + Replay (${replayN})`);

        // Weight replay data more heavily (it's from the real simulator)
        const replayWeight = Math.min(0.8, replayN / (replayN + NSIMS * 0.1));
        const mcWeight = 1 - replayWeight;

        const blendedCounts = [];
        for (let y = 0; y < H; y++) {
          blendedCounts[y] = [];
          for (let x = 0; x < W; x++) {
            blendedCounts[y][x] = new Float32Array(6);
            for (let c = 0; c < 6; c++) {
              blendedCounts[y][x][c] =
                mcWeight * (counts[y][x][c] / NSIMS) * 1000 +
                replayWeight * (acc.seeds[seed].grid[y][x][c] / replayN) * 1000;
            }
          }
        }
        prediction = buildPrediction(blendedCounts, 1000, initialGrid);
      } else {
        prediction = buildPrediction(counts, NSIMS, initialGrid);
      }
    }

    // Validate
    if (!validate(prediction)) {
      console.error(`  VALIDATION FAILED for seed ${seed}!`);
      continue;
    }

    // Submit
    console.log(`  Submitting seed ${seed}...`);
    try {
      const { data: result, status } = await api.post('/submit', {
        round_id: roundId,
        seed_index: seed,
        prediction: prediction,
      });
      console.log(`  Seed ${seed}: HTTP ${status} — ${JSON.stringify(result).slice(0, 200)}`);
    } catch (e) {
      console.error(`  Submit error: ${e.message}`);
    }

    await new Promise(r => setTimeout(r, 500)); // rate limit
  }

  console.log('\n═══ ALL SEEDS SUBMITTED ═══');
}

// ═══════════════════════════════════════════════════════════════════════
// REPLAY — Pure replay collection for a specific round
// ═══════════════════════════════════════════════════════════════════════
async function collectReplays() {
  console.log('═══════════════════════════════════════════════════');
  console.log('  REPLAY: Collecting replays for empirical predictions');
  console.log('═══════════════════════════════════════════════════');

  if (!TOKEN) { console.error('--token required'); process.exit(1); }

  let roundId = args.round;
  if (!roundId) {
    const { data: rounds } = await api.get('/rounds');
    const completed = rounds.filter(r => r.status === 'completed').sort((a, b) => b.round_number - a.round_number);
    if (completed.length === 0) { console.error('No completed rounds'); process.exit(1); }
    roundId = completed[0].id;
  }

  const ridShort = roundId.slice(0, 8);
  console.log(`Round: ${ridShort}, target: ${TARGET}/seed, concurrency: ${CONCURRENCY}`);

  const accFile = path.join(DATA_DIR, `replay_data_${ridShort}.json`);
  const acc = new ReplayAccumulator();
  acc.load(accFile);

  // Also feed into transition model
  const model = new TransitionModel();
  const modelFile = path.join(DATA_DIR, 'transition_model.json');
  model.load(modelFile);

  // Test auth
  try {
    const test = await api.post('/replay', { round_id: roundId, seed_index: 0 });
    if (!test.data || !test.data.frames) {
      console.error(`Replay unavailable:`, JSON.stringify(test.data).slice(0, 200));
      process.exit(1);
    }
    model.processReplay(test.data);
    acc.addReplay(test.data.seed_index, test.data);
    console.log('Auth OK');
  } catch (e) {
    console.error('Auth error:', e.message);
    process.exit(1);
  }

  let inflight = 0, collected = 0, errors = 0, running = true;
  const startTime = Date.now();

  process.on('SIGINT', () => {
    console.log('\nSaving...');
    running = false;
    acc.save(accFile);
    model.save(modelFile);
    const cts = acc.getCounts();
    console.log(`Saved: [${cts.join(',')}]`);
    process.exit(0);
  });

  function nextSeed() {
    let minN = Infinity, minS = 0;
    for (let s = 0; s < SEEDS; s++) {
      const n = acc.seeds[s] ? acc.seeds[s].count : 0;
      if (n < TARGET && n < minN) { minN = n; minS = s; }
    }
    return minS;
  }

  function needsMore() {
    for (let s = 0; s < SEEDS; s++) {
      if (!acc.seeds[s] || acc.seeds[s].count < TARGET) return true;
    }
    return false;
  }

  async function fetchOne() {
    if (!running || !needsMore()) return;
    const seed = nextSeed();
    inflight++;
    try {
      const { data } = await api.post('/replay', { round_id: roundId, seed_index: seed });
      if (data && data.frames && data.seed_index >= 0) {
        model.processReplay(data);
        acc.addReplay(data.seed_index, data);
        collected++;
      } else {
        errors++;
      }
    } catch (e) {
      errors++;
    }
    inflight--;
  }

  const printInterval = setInterval(() => {
    const elapsed = (Date.now() - startTime) / 1000;
    const rate = collected / Math.max(elapsed, 1);
    const cts = acc.getCounts();
    console.log(`[${elapsed.toFixed(0)}s] +${collected} [${cts.join(',')}] ${rate.toFixed(1)}/s err=${errors}`);
  }, 10000);

  const saveInterval = setInterval(() => {
    acc.save(accFile);
    model.save(modelFile);
  }, 30000);

  while (running && needsMore()) {
    while (inflight < CONCURRENCY && running && needsMore()) {
      fetchOne();
    }
    await new Promise(r => setTimeout(r, 50));
  }

  while (inflight > 0) await new Promise(r => setTimeout(r, 100));
  clearInterval(printInterval);
  clearInterval(saveInterval);

  acc.save(accFile);
  model.save(modelFile);
  const cts = acc.getCounts();
  console.log(`\nDONE: [${cts.join(',')}] in ${((Date.now() - startTime) / 1000).toFixed(0)}s`);
}

// ═══════════════════════════════════════════════════════════════════════
// SCORE — Score predictions against ground truth
// ═══════════════════════════════════════════════════════════════════════
async function scoreAgainstGT() {
  console.log('═══════════════════════════════════════════════════');
  console.log('  SCORE: Evaluating predictions against GT');
  console.log('═══════════════════════════════════════════════════');

  if (!TOKEN) { console.error('--token required'); process.exit(1); }

  let roundId = args.round;
  if (!roundId) {
    const { data: rounds } = await api.get('/rounds');
    const completed = rounds.filter(r => r.status === 'completed').sort((a, b) => b.round_number - a.round_number);
    if (completed.length === 0) { console.error('No completed rounds'); process.exit(1); }
    roundId = completed[0].id;
  }

  const ridShort = roundId.slice(0, 8);
  const { data: detail } = await api.get(`/rounds/${roundId}`);
  console.log(`Round ${detail.round_number} (${ridShort})\n`);

  // Load replay data
  const accFile = path.join(DATA_DIR, `replay_data_${ridShort}.json`);
  const acc = new ReplayAccumulator();
  const hasReplay = acc.load(accFile);

  // Load transition model
  const model = new TransitionModel();
  const modelFile = path.join(DATA_DIR, 'transition_model.json');
  const hasModel = model.load(modelFile);

  const results = [];

  for (let seed = 0; seed < SEEDS; seed++) {
    // Fetch GT
    try {
      const { data: analysis, status } = await api.get(`/analysis/${roundId}/${seed}`);
      if (status !== 200) { console.log(`Seed ${seed}: No GT available (${status})`); continue; }
      const gt = analysis.ground_truth;

      const initialGrid = detail.initial_states[seed].grid;
      const seedResults = { seed };

      // Score REPLAY prediction
      if (hasReplay && acc.seeds[seed] && acc.seeds[seed].count > 0) {
        const N = acc.seeds[seed].count;
        const pred = buildPrediction(acc.seeds[seed].grid, N, initialGrid);
        const sc = score(pred, gt);
        seedResults.replay = { score: sc.score.toFixed(2), N, wkl: sc.wkl.toFixed(6) };
      }

      // Score MC prediction
      if (hasModel) {
        const nSims = Math.min(NSIMS, 200); // use fewer sims for quick scoring
        console.log(`  Seed ${seed}: Running ${nSims} MC sims for scoring...`);
        const { counts } = runMC(model, initialGrid, nSims);
        const pred = buildPrediction(counts, nSims, initialGrid);
        const sc = score(pred, gt);
        seedResults.mc = { score: sc.score.toFixed(2), nSims, wkl: sc.wkl.toFixed(6) };
      }

      results.push(seedResults);

      // Print
      let line = `Seed ${seed}:`;
      if (seedResults.replay) line += ` REPLAY=${seedResults.replay.score} (N=${seedResults.replay.N})`;
      if (seedResults.mc) line += ` MC=${seedResults.mc.score} (N=${seedResults.mc.nSims})`;
      console.log(line);
    } catch (e) {
      console.error(`Seed ${seed}: Error — ${e.message}`);
    }
  }

  // Averages
  if (results.length > 0) {
    const avgReplay = results.filter(r => r.replay).reduce((s, r) => s + parseFloat(r.replay.score), 0) / Math.max(1, results.filter(r => r.replay).length);
    const avgMC = results.filter(r => r.mc).reduce((s, r) => s + parseFloat(r.mc.score), 0) / Math.max(1, results.filter(r => r.mc).length);
    console.log(`\nAverages:`);
    if (avgReplay > 0) console.log(`  REPLAY: ${avgReplay.toFixed(2)}`);
    if (avgMC > 0) console.log(`  MC: ${avgMC.toFixed(2)}`);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// ANALYZE — Detailed error analysis
// ═══════════════════════════════════════════════════════════════════════
async function analyze() {
  console.log('═══════════════════════════════════════════════════');
  console.log('  ANALYZE: Detailed prediction error analysis');
  console.log('═══════════════════════════════════════════════════');

  if (!TOKEN) { console.error('--token required'); process.exit(1); }

  // Find all completed rounds
  const { data: rounds } = await api.get('/rounds');
  const completed = rounds.filter(r => r.status === 'completed');

  // Load model
  const model = new TransitionModel();
  const modelFile = path.join(DATA_DIR, 'transition_model.json');
  model.load(modelFile);
  const stats = model.stats();
  console.log(`\nModel: ${stats.totalKeys} keys, ${stats.totalTransitions} transitions`);
  for (const level in stats.levels) {
    console.log(`  ${level}: ${stats.levels[level].keys} keys, ${stats.levels[level].totalSamples} samples`);
  }

  for (const round of completed) {
    const rid = round.id;
    const ridShort = rid.slice(0, 8);
    const { data: detail } = await api.get(`/rounds/${rid}`);
    console.log(`\n═══ Round ${round.round_number} (${ridShort}) ═══`);

    // Load replay data
    const accFile = path.join(DATA_DIR, `replay_data_${ridShort}.json`);
    const acc = new ReplayAccumulator();
    acc.load(accFile);

    for (let seed = 0; seed < 1; seed++) { // analyze seed 0 only for brevity
      try {
        const { data: analysis, status } = await api.get(`/analysis/${rid}/${seed}`);
        if (status !== 200) continue;
        const gt = analysis.ground_truth;
        const initialGrid = detail.initial_states[seed].grid;

        // Build prediction (replay if available, else MC)
        let pred, method;
        if (acc.seeds && acc.seeds[seed] && acc.seeds[seed].count >= 50) {
          const N = acc.seeds[seed].count;
          pred = buildPrediction(acc.seeds[seed].grid, N, initialGrid);
          method = `replay(N=${N})`;
        } else {
          const { counts } = runMC(model, initialGrid, 200);
          pred = buildPrediction(counts, 200, initialGrid);
          method = 'mc(N=200)';
        }

        const sc = score(pred, gt);
        console.log(`  Seed ${seed} [${method}]: score=${sc.score.toFixed(2)}, wkl=${sc.wkl.toFixed(6)}, dynamic=${sc.dynamicCells}`);

        // Find worst cells
        const cellErrors = [];
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const g = gt[y][x];
            let ent = 0;
            for (let c = 0; c < 6; c++) if (g[c] > 1e-6) ent -= g[c] * Math.log(g[c]);
            if (ent < 0.01) continue;
            let kl = 0;
            for (let c = 0; c < 6; c++) {
              if (g[c] > 1e-6) kl += g[c] * Math.log(g[c] / Math.max(pred[y][x][c], 1e-15));
            }
            kl = Math.max(0, kl);
            cellErrors.push({ y, x, kl, ent, contrib: kl * ent, terrain: initialGrid[y][x] });
          }
        }
        cellErrors.sort((a, b) => b.contrib - a.contrib);

        console.log(`  Top 10 worst cells:`);
        const terrainNames = { 0: 'empty', 1: 'settle', 2: 'port', 3: 'ruin', 4: 'forest', 5: 'mtn', 10: 'ocean', 11: 'plains' };
        for (const ce of cellErrors.slice(0, 10)) {
          const gtStr = gt[ce.y][ce.x].map(v => (v * 100).toFixed(0) + '%').join(' ');
          const prStr = pred[ce.y][ce.x].map(v => (v * 100).toFixed(0) + '%').join(' ');
          console.log(`    (${ce.y},${ce.x}) ${terrainNames[ce.terrain] || ce.terrain}: KL=${ce.kl.toFixed(4)} ent=${ce.ent.toFixed(3)}`);
          console.log(`      GT:   [${gtStr}]`);
          console.log(`      Pred: [${prStr}]`);
        }

        // Error by terrain type
        const byTerrain = {};
        for (const ce of cellErrors) {
          const t = terrainNames[ce.terrain] || String(ce.terrain);
          if (!byTerrain[t]) byTerrain[t] = { count: 0, totalContrib: 0 };
          byTerrain[t].count++;
          byTerrain[t].totalContrib += ce.contrib;
        }
        console.log(`  Error by initial terrain:`);
        for (const [t, data] of Object.entries(byTerrain).sort((a, b) => b[1].totalContrib - a[1].totalContrib)) {
          console.log(`    ${t}: ${data.count} cells, total KL×ent=${data.totalContrib.toFixed(4)} (${(data.totalContrib / cellErrors.reduce((s, c) => s + c.contrib, 0) * 100).toFixed(1)}%)`);
        }
      } catch (e) {
        console.error(`  Seed ${seed}: ${e.message}`);
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════
// STATUS — Quick overview
// ═══════════════════════════════════════════════════════════════════════
async function showStatus() {
  console.log('═══════════════════════════════════════════════════');
  console.log('  Astar Island Solver R3 — Status');
  console.log('═══════════════════════════════════════════════════');

  // Check for data files
  const modelFile = path.join(DATA_DIR, 'transition_model.json');
  if (fs.existsSync(modelFile)) {
    const model = new TransitionModel();
    model.load(modelFile);
    const stats = model.stats();
    console.log(`\nTransition model: ${stats.totalKeys} keys, ${stats.totalTransitions} transitions`);
  } else {
    console.log('\nNo transition model found.');
  }

  // Check replay data files
  const files = fs.readdirSync(DATA_DIR).filter(f => f.startsWith('replay_data_'));
  for (const file of files) {
    const acc = new ReplayAccumulator();
    if (acc.load(path.join(DATA_DIR, file))) {
      console.log(`${file}: [${acc.getCounts().join(',')}]`);
    }
  }

  if (!TOKEN) {
    console.log('\nNo token provided. Get one from browser: DevTools → Application → Cookies → access_token');
    return;
  }

  // Check rounds
  try {
    const { data: rounds } = await api.get('/rounds');
    console.log('\nRounds:');
    for (const r of rounds) {
      console.log(`  R${r.round_number}: ${r.status} (${r.id.slice(0, 8)})`);
    }

    // Check leaderboard
    const { data: lb } = await api.get('/leaderboard');
    const us = lb.find(t => t.team_slug === 'tastebrettenes-venner');
    if (us) {
      console.log(`\nOur team: Rank ${us.rank}/${lb.length}, weighted=${us.weighted_score.toFixed(2)}, streak=${us.hot_streak_score.toFixed(2)}`);
    }
    console.log('\nTop 5:');
    for (const t of lb.slice(0, 5)) {
      console.log(`  #${t.rank}: ${t.team_name} — ${t.weighted_score.toFixed(2)}`);
    }
  } catch (e) {
    console.log(`API error: ${e.message}`);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════
async function main() {
  switch (CMD) {
    case 'train':
      await trainModel();
      break;
    case 'predict':
      await predict();
      break;
    case 'replay':
      await collectReplays();
      break;
    case 'score':
      await scoreAgainstGT();
      break;
    case 'analyze':
      await analyze();
      break;
    case 'status':
      await showStatus();
      break;
    default:
      console.log(`
Astar Island R3 Solver — Comprehensive prediction engine

Usage:
  node solver_r3.js <command> --token <JWT> [options]

Commands:
  train     Collect replays from completed rounds, build transition model
  predict   Run MC sims on target round, submit predictions
  replay    Collect replays for a completed round (pure replay approach)
  score     Score predictions against ground truth
  analyze   Detailed error analysis (worst cells, error by terrain)
  status    Show current data and leaderboard status

Options:
  --token <JWT>       Auth token (from browser cookies)
  --round <id>        Target round ID (auto-detects if omitted)
  --target <N>        Replays per seed (default: 5000)
  --concurrency <N>   Parallel requests (default: 20)
  --sims <N>          MC simulations per seed (default: 1000)

Get token:
  Browser → DevTools → Application → Cookies → copy 'access_token' value

Workflow:
  1. node solver_r3.js train --token <JWT>     # collect R1+R2 replays, build model
  2. node solver_r3.js predict --token <JWT>    # MC simulate + submit for active round
  3. node solver_r3.js score --token <JWT>      # check scores after round completes
  4. node solver_r3.js analyze --token <JWT>    # find where we're losing points
`);
  }
}

main().catch(e => {
  console.error('Fatal error:', e);
  process.exit(1);
});
