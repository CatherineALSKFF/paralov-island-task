#!/usr/bin/env node
/**
 * Ultimate Astar Island Solver v2
 *
 * Complete pipeline with LOO validation, parameter optimization,
 * massive replay data, and viewport data persistence.
 *
 * Modes:
 *   --mode validate  : Run LOO validation (requires replays on disk)
 *   --mode submit    : Submit for active round (full pipeline)
 *   --mode full      : Validate then submit (default)
 *
 * Usage: node ultimate_solver.js <JWT> [--mode validate|submit|full]
 */
const https = require('https');
const fs = require('fs');
const path = require('path');
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5, C = 6;
const TOKEN = process.argv[2] || '';
const DATA_DIR = path.join(__dirname, 'data');
const MODE = (() => { const i = process.argv.indexOf('--mode'); return i >= 0 ? process.argv[i+1] : 'full'; })();

if (!TOKEN) { console.log('Usage: node ultimate_solver.js <JWT> [--mode validate|submit|full]'); process.exit(1); }
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

function api(method, pth, body) {
  return new Promise((resolve, reject) => {
    const url = new URL(BASE + pth);
    const payload = body ? JSON.stringify(body) : null;
    const opts = { hostname: url.hostname, path: url.pathname + url.search, method,
      headers: { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' } };
    if (payload) opts.headers['Content-Length'] = Buffer.byteLength(payload);
    const req = https.request(opts, res => {
      let data = ''; res.on('data', c => data += c);
      res.on('end', () => { try { resolve({ ok: res.statusCode < 300, status: res.statusCode, data: JSON.parse(data) }); } catch { resolve({ ok: false, status: res.statusCode, data }); } });
    }); req.on('error', reject); if (payload) req.write(payload); req.end();
  });
}
const GET = p => api('GET', p), POST = (p, b) => api('POST', p, b);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const log = m => { const t = new Date().toISOString().slice(11, 19); console.log(`[${t}] ${m}`); };
function t2c(t) { return (t===10||t===11||t===0)?0:(t>=1&&t<=5)?t:0; }

// ═══ FEATURE FUNCTIONS ═══
// Enhanced features with mountain adjacency and edge detection
function cf(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS=0,co=0,fN=0,sR2=0,mN=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
    if(!dy&&!dx)continue;
    const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;
    const nt=g[ny][nx];
    if(nt===1||nt===2)nS++;
    if(nt===10)co=1;
    if(nt===4)fN++;
    if(nt===5)mN++;
  }
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){
    if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;
    const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;
    if(g[ny][nx]===1||g[ny][nx]===2)sR2++;
  }
  const sa=Math.min(nS,5);
  const sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3;
  const fb=fN<=1?0:fN<=3?1:2;
  const mb=mN>0?1:0; // mountain adjacency
  // Edge detection: cells within 1 of border
  const edge=(y<=0||y>=H-1||x<=0||x>=W-1)?1:0;

  // D0: most specific (terrain, settlements, coastal, R2, forest)
  // D0e: D0 + mountain adjacency
  // D1: less specific
  // D2: coarse
  // D3: very coarse
  // D4: just terrain type
  return [
    `D0_${t}_${sa}_${co}_${sb2}_${fb}`,
    `D0e_${t}_${sa}_${co}_${sb2}_${fb}_${mb}`,
    `D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,
    `D2_${t}_${sa>0?1:0}_${co}`,
    `D3_${t}_${co}`,
    `D4_${t}`
  ];
}

// Feature keys for model lookup (original D0-D4 indices)
const FEAT_INDICES = { D0: 0, D0e: 1, D1: 2, D2: 3, D3: 4, D4: 5 };

// ═══ DATA LOADING ═══
function loadReplays(roundName) {
  const f = path.join(DATA_DIR, `replays_${roundName}.json`);
  if (!fs.existsSync(f)) return null;
  return JSON.parse(fs.readFileSync(f, 'utf8'));
}
function loadInits(roundName) {
  const f = path.join(DATA_DIR, `inits_${roundName}.json`);
  if (!fs.existsSync(f)) return null;
  return JSON.parse(fs.readFileSync(f, 'utf8'));
}
function loadGT(roundName) {
  const f = path.join(DATA_DIR, `gt_${roundName}.json`);
  if (!fs.existsSync(f)) return null;
  return JSON.parse(fs.readFileSync(f, 'utf8'));
}
function saveViewportData(roundId, data) {
  const f = path.join(DATA_DIR, `viewport_${roundId.slice(0,8)}.json`);
  fs.writeFileSync(f, JSON.stringify(data));
  log(`  Viewport data saved to ${f}`);
}
function loadViewportData(roundId) {
  const f = path.join(DATA_DIR, `viewport_${roundId.slice(0,8)}.json`);
  if (!fs.existsSync(f)) return null;
  return JSON.parse(fs.readFileSync(f, 'utf8'));
}

// ═══ MODEL BUILDING ═══
function buildCrossRoundModel(replays, inits, roundNames, featIdx=0, alpha=0.05) {
  const m = {};
  for (const rn of roundNames) {
    if (!replays[rn] || !inits[rn]) continue;
    for (const rep of replays[rn]) {
      const initGrid = inits[rn][rep.si];
      if (!initGrid) continue;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cf(initGrid, y, x); if (!keys) continue;
        const k = keys[featIdx];
        const fc = t2c(rep.finalGrid[y][x]);
        if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
        m[k].n++; m[k].counts[fc]++;
      }
    }
  }
  for (const k of Object.keys(m)) {
    const total = m[k].n + C * alpha;
    m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / total);
  }
  return m;
}

function buildViewportFeatureModel(initGrid, observations, featIdx=0, alpha=0.1) {
  const m = {};
  for (const obs of observations) {
    for (let dy = 0; dy < obs.grid.length; dy++) {
      for (let dx = 0; dx < obs.grid[0].length; dx++) {
        const gy = obs.vy + dy, gx = obs.vx + dx;
        if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
        const keys = cf(initGrid, gy, gx); if (!keys) continue;
        const k = keys[featIdx];
        const fc = t2c(obs.grid[dy][dx]);
        if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
        m[k].n++; m[k].counts[fc]++;
      }
    }
  }
  for (const k of Object.keys(m)) {
    const total = m[k].n + C * alpha;
    m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / total);
  }
  return m;
}

// Per-cell model from viewport observations
function buildPerCellModel(initGrid, observations, alpha=0.5) {
  const cells = {}; // key: "y,x" → {n, counts}
  for (const obs of observations) {
    for (let dy = 0; dy < obs.grid.length; dy++) {
      for (let dx = 0; dx < obs.grid[0].length; dx++) {
        const gy = obs.vy + dy, gx = obs.vx + dx;
        if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
        const t = initGrid[gy][gx];
        if (t === 10 || t === 5) continue;
        const k = `${gy},${gx}`;
        const fc = t2c(obs.grid[dy][dx]);
        if (!cells[k]) cells[k] = { n: 0, counts: new Float64Array(C) };
        cells[k].n++; cells[k].counts[fc]++;
      }
    }
  }
  return cells;
}

function fuseModels(crossModel, viewportModel, crossWeight, featIdx=0) {
  const m = {};
  const allKeys = new Set([...Object.keys(crossModel), ...Object.keys(viewportModel)]);
  for (const k of allKeys) {
    const cm = crossModel[k]; const vm = viewportModel[k];
    if (cm && vm) {
      const priorAlpha = cm.a.map(p => p * crossWeight);
      const posterior = priorAlpha.map((a, c) => a + vm.counts[c]);
      let total = posterior.reduce((a,b)=>a+b, 0);
      m[k] = { n: cm.n + vm.n, a: posterior.map(v => v / total) };
    } else if (vm) { m[k] = { n: vm.n, a: vm.a.slice() }; }
    else { m[k] = { n: cm.n, a: cm.a.slice() }; }
  }
  return m;
}

// ═══ PREDICTION ═══
// Hierarchical feature prediction with configurable weights
function predict(grid, models, cfg) {
  // models is an array: [model_D0, model_D0e, model_D1, model_D2, model_D3, model_D4]
  // or a single model (backwards compatible)
  const isSingle = !Array.isArray(models);
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1,0,0,0,0,0]; continue; }
      if (t === 5) { pred[y][x] = [0,0,0,0,0,1]; continue; }
      const keys = cf(grid, y, x);
      if (!keys) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }

      const p = [0,0,0,0,0,0]; let wS = 0;
      if (isSingle) {
        // Single model: use keys as D0-D4 hierarchy
        for (let ki = 0; ki < Math.min(keys.length, cfg.ws.length); ki++) {
          const d = models[keys[ki]];
          if (d && d.n >= cfg.minN) {
            const w = cfg.ws[ki] * Math.pow(d.n, cfg.pow);
            for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w;
          }
        }
      } else {
        // Multiple models: each model uses its specific feature key
        for (let ki = 0; ki < Math.min(models.length, cfg.ws.length); ki++) {
          if (!models[ki]) continue;
          const d = models[ki][keys[ki]];
          if (d && d.n >= cfg.minN) {
            const w = cfg.ws[ki] * Math.pow(d.n, cfg.pow);
            for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w;
          }
        }
      }
      if (wS === 0) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      let s = 0;
      for (let c = 0; c < C; c++) { p[c] /= wS; if (p[c] < cfg.fl) p[c] = cfg.fl; s += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

// Per-cell correction: override feature predictions with direct cell observations
function applyPerCellCorrections(pred, grid, perCellModel, featurePred, cellWeight=5) {
  // For cells with direct observations, blend feature prediction with per-cell data
  const corrected = pred.map(row => row.map(p => p.slice()));
  for (const [key, cell] of Object.entries(perCellModel)) {
    const [y, x] = key.split(',').map(Number);
    if (grid[y][x] === 10 || grid[y][x] === 5) continue;
    if (cell.n < 3) continue; // Need minimum observations for correction

    // Bayesian: use feature prediction as prior, update with per-cell observations
    const prior = corrected[y][x];
    const posterior = new Array(C);
    const priorWeight = cellWeight; // How much to trust the feature model
    let total = 0;
    for (let c = 0; c < C; c++) {
      posterior[c] = prior[c] * priorWeight + cell.counts[c];
      total += posterior[c];
    }
    for (let c = 0; c < C; c++) posterior[c] /= total;
    corrected[y][x] = posterior;
  }
  return corrected;
}

// ═══ SCORING ═══
function computeScore(pred, gt) {
  // gt is a H×W×C probability tensor (ground truth)
  let totalEntropy = 0, totalWeightedKL = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const p = gt[y][x]; // true distribution
    const q = pred[y][x]; // our prediction
    // Compute entropy of true distribution
    let entropy = 0;
    for (let c = 0; c < C; c++) {
      if (p[c] > 0.001) entropy -= p[c] * Math.log(p[c]);
    }
    if (entropy < 0.01) continue; // Skip static cells
    // KL divergence: KL(p || q)
    let kl = 0;
    for (let c = 0; c < C; c++) {
      if (p[c] > 0.001) {
        const q_safe = Math.max(q[c], 1e-10);
        kl += p[c] * Math.log(p[c] / q_safe);
      }
    }
    totalEntropy += entropy;
    totalWeightedKL += entropy * kl;
  }
  if (totalEntropy === 0) return 100;
  const wkl = totalWeightedKL / totalEntropy;
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl)));
}

// ═══ LOO VALIDATION ═══
async function runLOOValidation(config) {
  log('\n═══ LOO Cross-Validation ═══');
  log(`Config: feat=${config.featIdx}, ws=[${config.ws.join(',')}], pow=${config.pow}, minN=${config.minN}, fl=${config.fl}, alpha=${config.alpha}`);
  if (config.crossWeight !== undefined) log(`  crossWeight=${config.crossWeight}`);

  // Load all available data
  const rounds = [];
  for (let r = 1; r <= 10; r++) {
    const rn = `R${r}`;
    const replays = loadReplays(rn);
    const inits = loadInits(rn);
    const gt = loadGT(rn);
    if (replays && inits && gt && replays.length > 0) {
      rounds.push({ rn, replays, inits, gt });
    }
  }

  if (rounds.length < 2) {
    log(`  Only ${rounds.length} rounds with data. Need at least 2 for LOO.`);
    return null;
  }

  log(`  Using ${rounds.length} rounds: ${rounds.map(r => r.rn).join(', ')}`);
  log(`  Replay counts: ${rounds.map(r => `${r.rn}=${r.replays.length}`).join(', ')}`);

  const scores = [];
  for (let holdout = 0; holdout < rounds.length; holdout++) {
    const testRound = rounds[holdout];
    const trainRounds = rounds.filter((_, i) => i !== holdout);

    // Build cross-round model from training rounds
    const replaysMap = {}, initsMap = {};
    for (const tr of trainRounds) {
      replaysMap[tr.rn] = tr.replays;
      initsMap[tr.rn] = tr.inits;
    }
    const model = buildCrossRoundModel(replaysMap, initsMap, trainRounds.map(r => r.rn), config.featIdx, config.alpha);

    // If we have simulated viewport data, fuse
    let finalModel = model;
    // (For LOO, we don't have viewport data for the held-out round)

    // Evaluate on held-out round
    const roundScores = [];
    for (let si = 0; si < SEEDS; si++) {
      if (!testRound.inits[si] || !testRound.gt[si]) continue;
      const p = predict(testRound.inits[si], finalModel, config);
      const score = computeScore(p, testRound.gt[si]);
      roundScores.push(score);
    }
    const avgScore = roundScores.reduce((a,b)=>a+b, 0) / roundScores.length;
    scores.push({ round: testRound.rn, score: avgScore, perSeed: roundScores });
    log(`  ${testRound.rn}: ${avgScore.toFixed(2)} [${roundScores.map(s => s.toFixed(1)).join(', ')}]`);
  }

  const overallLOO = scores.reduce((a,b) => a + b.score, 0) / scores.length;
  log(`  ────────────────────`);
  log(`  Overall LOO: ${overallLOO.toFixed(2)}`);
  return { overallLOO, scores };
}

// ═══ LOO WITH SIMULATED VIEWPORT ═══
async function runLOOWithViewport(config) {
  log('\n═══ LOO with Simulated Viewport ═══');
  log(`Config: crossWeight=${config.crossWeight}, cellWeight=${config.cellWeight || 'none'}`);

  const rounds = [];
  for (let r = 1; r <= 10; r++) {
    const rn = `R${r}`;
    const replays = loadReplays(rn);
    const inits = loadInits(rn);
    const gt = loadGT(rn);
    if (replays && inits && gt && replays.length > 0) {
      rounds.push({ rn, replays, inits, gt });
    }
  }

  if (rounds.length < 2) {
    log(`  Only ${rounds.length} rounds. Need ≥2.`);
    return null;
  }

  log(`  Using ${rounds.length} rounds: ${rounds.map(r => r.rn).join(', ')}`);

  const scores = [];
  for (let holdout = 0; holdout < rounds.length; holdout++) {
    const testRound = rounds[holdout];
    const trainRounds = rounds.filter((_, i) => i !== holdout);

    // Build cross-round model
    const replaysMap = {}, initsMap = {};
    for (const tr of trainRounds) {
      replaysMap[tr.rn] = tr.replays;
      initsMap[tr.rn] = tr.inits;
    }
    const crossModel = buildCrossRoundModel(replaysMap, initsMap, trainRounds.map(r => r.rn), config.featIdx || 0, config.alpha || 0.05);

    // Simulate viewport: use held-out round's OWN replays as "viewport observations"
    // Use N replays as viewport observations (simulating 50 viewport queries)
    // Each replay covers the full 40x40, but viewport covers 15x15 per query
    // With 50 queries × ~165 dynamic cells per viewport, we get ~8250 observations
    // Simulate this by using N=50 replays but only counting cells within viewport positions
    const viewportObs = [];
    const starts = [0, 13, 25];
    const N_VIEWPORT_SIMS = Math.min(50, testRound.replays.length);
    for (let i = 0; i < N_VIEWPORT_SIMS; i++) {
      const rep = testRound.replays[i];
      // Assign each replay to a viewport position
      const vpIdx = i % 9;
      const vy = starts[Math.floor(vpIdx / 3)];
      const vx = starts[vpIdx % 3];
      // Extract 15x15 region
      const grid = [];
      for (let dy = 0; dy < 15; dy++) {
        grid[dy] = [];
        for (let dx = 0; dx < 15; dx++) {
          const gy = vy + dy, gx = vx + dx;
          if (gy < H && gx < W) grid[dy][dx] = rep.finalGrid[gy][gx];
          else grid[dy][dx] = 10;
        }
      }
      viewportObs.push({ vy, vx, grid });
    }

    const initGrid = testRound.inits[0]; // Use seed 0's initial grid
    const viewportModel = buildViewportFeatureModel(initGrid, viewportObs, config.featIdx || 0, 0.1);

    // Fuse
    const fusedModel = fuseModels(crossModel, viewportModel, config.crossWeight);

    // Per-cell corrections
    let usePerCell = config.cellWeight !== undefined && config.cellWeight > 0;
    let perCellModel = null;
    if (usePerCell) {
      perCellModel = buildPerCellModel(initGrid, viewportObs, 0.5);
    }

    // Evaluate
    const roundScores = [];
    for (let si = 0; si < SEEDS; si++) {
      if (!testRound.inits[si] || !testRound.gt[si]) continue;
      let p = predict(testRound.inits[si], fusedModel, config);
      if (usePerCell && si === 0 && perCellModel) {
        p = applyPerCellCorrections(p, testRound.inits[si], perCellModel, p, config.cellWeight);
      }
      const score = computeScore(p, testRound.gt[si]);
      roundScores.push(score);
    }
    const avgScore = roundScores.reduce((a,b)=>a+b, 0) / roundScores.length;
    scores.push({ round: testRound.rn, score: avgScore, perSeed: roundScores });
    log(`  ${testRound.rn}: ${avgScore.toFixed(2)} [${roundScores.map(s => s.toFixed(1)).join(', ')}]`);
  }

  const overallLOO = scores.reduce((a,b) => a + b.score, 0) / scores.length;
  log(`  ────────────────────`);
  log(`  Overall LOO: ${overallLOO.toFixed(2)}`);
  return { overallLOO, scores };
}

// ═══ VIEWPORT PLANNING ═══
function planAllViewports() {
  const starts = [0, 13, 25];
  const viewports = [];
  for (let pass = 0; pass < 5; pass++) {
    for (const vy of starts) for (const vx of starts)
      viewports.push({ vy, vx });
  }
  // 5 offset positions
  const extras = [{vy:7,vx:7},{vy:7,vx:20},{vy:20,vx:7},{vy:20,vx:20},{vy:12,vx:12}];
  for (const pos of extras) viewports.push(pos);
  return viewports;
}

async function executeViewportQueries(roundId, seedIndex, viewports) {
  const observations = [];
  let queryCount = 0, failures = 0;
  for (const vp of viewports) {
    let success = false;
    for (let retry = 0; retry < 3 && !success; retry++) {
      try {
        const res = await POST('/simulate', {
          round_id: roundId, seed_index: seedIndex, year: 50,
          viewport: { x: vp.vx, y: vp.vy, width: 15, height: 15 }
        });
        queryCount++;
        if (res.ok && res.data.grid) {
          observations.push({ vy: vp.vy, vx: vp.vx, grid: res.data.grid });
          success = true;
        } else if (res.status === 429) {
          await sleep(1000);
        } else {
          log(`  VP (${vp.vy},${vp.vx}) failed: ${JSON.stringify(res.data).slice(0,100)}`);
          failures++; success = true;
        }
      } catch (e) { await sleep(500); }
    }
    await sleep(220);
    if (queryCount % 10 === 0) log(`  VP progress: ${queryCount}/${viewports.length}, ${observations.length} obs`);
  }
  return { observations, queryCount, failures };
}

// ═══ REPLAY COLLECTION (quick, for new rounds) ═══
async function collectReplays(roundId, count, concurrency = 10) {
  const results = []; let collected = 0, errors = 0;
  while (collected < count) {
    const batch = [];
    for (let i = 0; i < Math.min(concurrency, count - collected); i++) {
      const si = (collected + i) % SEEDS;
      batch.push((async () => {
        try {
          const res = await POST('/replay', { round_id: roundId, seed_index: si });
          if (!res.ok || !res.data.frames) { errors++; return null; }
          return { si, finalGrid: res.data.frames[res.data.frames.length - 1].grid };
        } catch { errors++; return null; }
      })());
    }
    const batchResults = await Promise.all(batch);
    for (const r of batchResults) { if (r) { results.push(r); collected++; } }
    await sleep(150);
  }
  return { results, errors };
}

// ═══ PARAMETER GRID SEARCH ═══
async function gridSearch() {
  log('\n═══ PARAMETER GRID SEARCH ═══');

  const baseConfig = { ws: [1, 0, 0.3, 0.15, 0.08, 0.02], pow: 0.5, minN: 2, fl: 0.00005 };
  const results = [];

  // Test different feature indices and alphas
  const configs = [
    // Baseline: D0 features
    { ...baseConfig, featIdx: 0, alpha: 0.05, label: 'D0 a=0.05' },
    { ...baseConfig, featIdx: 0, alpha: 0.02, label: 'D0 a=0.02' },
    { ...baseConfig, featIdx: 0, alpha: 0.1, label: 'D0 a=0.1' },
    // Enhanced D0 with mountain adjacency
    { ...baseConfig, featIdx: 1, alpha: 0.05, label: 'D0e a=0.05' },
    // Different weight schemes
    { ...baseConfig, ws: [1, 0, 0.5, 0.2, 0.1, 0.05], featIdx: 0, alpha: 0.05, label: 'D0 higher-fallback' },
    { ...baseConfig, ws: [1, 0, 0.1, 0.05, 0.02, 0.01], featIdx: 0, alpha: 0.05, label: 'D0 lower-fallback' },
    // Different pow
    { ...baseConfig, pow: 0.3, featIdx: 0, alpha: 0.05, label: 'D0 pow=0.3' },
    { ...baseConfig, pow: 0.7, featIdx: 0, alpha: 0.05, label: 'D0 pow=0.7' },
    // Different minN
    { ...baseConfig, minN: 1, featIdx: 0, alpha: 0.05, label: 'D0 minN=1' },
    { ...baseConfig, minN: 5, featIdx: 0, alpha: 0.05, label: 'D0 minN=5' },
    // Different floor
    { ...baseConfig, fl: 0.0001, featIdx: 0, alpha: 0.05, label: 'D0 fl=0.0001' },
    { ...baseConfig, fl: 0.00001, featIdx: 0, alpha: 0.05, label: 'D0 fl=0.00001' },
  ];

  for (const cfg of configs) {
    const result = await runLOOValidation(cfg);
    if (result) {
      results.push({ label: cfg.label, loo: result.overallLOO, config: cfg });
    }
  }

  // Sort by LOO score
  results.sort((a, b) => b.loo - a.loo);
  log('\n═══ GRID SEARCH RESULTS ═══');
  for (const r of results) {
    log(`  ${r.loo.toFixed(2)} : ${r.label}`);
  }

  return results;
}

// ═══ CROSS-WEIGHT SEARCH ═══
async function crossWeightSearch(bestConfig) {
  log('\n═══ CROSS-WEIGHT SEARCH (with simulated viewport) ═══');
  const crossWeights = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100];
  const results = [];

  for (const cw of crossWeights) {
    const cfg = { ...bestConfig, crossWeight: cw };
    const result = await runLOOWithViewport(cfg);
    if (result) {
      results.push({ cw, loo: result.overallLOO });
    }
  }

  results.sort((a, b) => b.loo - a.loo);
  log('\n═══ CROSS-WEIGHT RESULTS ═══');
  for (const r of results) {
    log(`  cw=${r.cw}: LOO=${r.loo.toFixed(2)}`);
  }

  return results;
}

// ═══ PER-CELL WEIGHT SEARCH ═══
async function cellWeightSearch(bestConfig, bestCW) {
  log('\n═══ PER-CELL WEIGHT SEARCH ═══');
  const cellWeights = [0, 2, 5, 10, 20, 50];
  const results = [];

  for (const cellW of cellWeights) {
    const cfg = { ...bestConfig, crossWeight: bestCW, cellWeight: cellW };
    const result = await runLOOWithViewport(cfg);
    if (result) {
      results.push({ cellWeight: cellW, loo: result.overallLOO });
    }
  }

  results.sort((a, b) => b.loo - a.loo);
  log('\n═══ CELL-WEIGHT RESULTS ═══');
  for (const r of results) {
    log(`  cellW=${r.cellWeight}: LOO=${r.loo.toFixed(2)}`);
  }

  return results;
}

// ═══ SUBMISSION ═══
async function submitForRound(roundId, config) {
  log('\n═══ SUBMITTING FOR ACTIVE ROUND ═══');

  // Load target round
  const { data: targetRound } = await GET('/rounds/' + roundId);
  const targetInits = targetRound.initial_states.map(is => is.grid);
  log(`  Round: ${roundId.slice(0,8)}... (${SEEDS} seeds)`);
  log(`  Closes: ${targetRound.closes_at}`);

  // Load all training data from disk
  const replaysMap = {}, initsMap = {};
  const trainRounds = [];
  for (let r = 1; r <= 10; r++) {
    const rn = `R${r}`;
    const replays = loadReplays(rn);
    const inits = loadInits(rn);
    if (replays && inits && replays.length > 0) {
      replaysMap[rn] = replays;
      initsMap[rn] = inits;
      trainRounds.push(rn);
    }
  }
  log(`  Training rounds: ${trainRounds.join(', ')} (${trainRounds.map(r => replaysMap[r].length).join(', ')} replays)`);

  // Build cross-round model
  const crossModel = buildCrossRoundModel(replaysMap, initsMap, trainRounds, config.featIdx || 0, config.alpha || 0.05);
  const d0Keys = Object.keys(crossModel).filter(k => !k.startsWith('D0e_')).length;
  log(`  Cross-round model: ${d0Keys} feature keys`);

  // Submit baseline
  log('\n  ── Submitting BASELINE ──');
  for (let si = 0; si < SEEDS; si++) {
    const p = predict(targetInits[si], crossModel, config);
    const res = await POST('/submit', { round_id: roundId, seed_index: si, prediction: p });
    log(`  Seed ${si}: ${res.ok ? 'ACCEPTED' : 'FAILED'} - ${JSON.stringify(res.data).slice(0, 80)}`);
    await sleep(600);
  }
  log('  ✅ Baseline submitted');

  // Check for existing viewport data
  let observations = null;
  const existingVP = loadViewportData(roundId);
  if (existingVP) {
    observations = existingVP;
    log(`\n  Loaded ${observations.length} viewport observations from disk`);
  } else {
    // Execute viewport queries
    log('\n  ── Executing 50 viewport queries on SEED 0 ──');
    const viewports = planAllViewports();
    log(`  Planned ${viewports.length} queries`);
    const vpResult = await executeViewportQueries(roundId, 0, viewports);
    observations = vpResult.observations;
    log(`  ✅ ${observations.length} observations (${vpResult.failures} failures)`);

    // SAVE IMMEDIATELY
    if (observations.length > 0) {
      saveViewportData(roundId, observations);
    }
  }

  if (observations && observations.length > 0) {
    // Build viewport feature model
    const viewportModel = buildViewportFeatureModel(targetInits[0], observations, config.featIdx || 0, 0.1);
    const vpKeys = Object.keys(viewportModel).length;
    let totalObs = 0;
    for (const v of Object.values(viewportModel)) totalObs += v.n;
    log(`  Viewport model: ${vpKeys} keys, ${totalObs} total obs, ${(totalObs/Math.max(vpKeys,1)).toFixed(1)} avg/key`);

    // Fuse
    const cw = config.crossWeight || 30;
    const fusedModel = fuseModels(crossModel, viewportModel, cw);
    log(`  Fused model (cw=${cw}): ${Object.keys(fusedModel).length} keys`);

    // Per-cell corrections
    let perCellModel = null;
    if (config.cellWeight && config.cellWeight > 0) {
      perCellModel = buildPerCellModel(targetInits[0], observations, 0.5);
      const cellCount = Object.keys(perCellModel).filter(k => perCellModel[k].n >= 3).length;
      log(`  Per-cell model: ${cellCount} cells with ≥3 observations`);
    }

    // Submit improved
    log('\n  ── Submitting IMPROVED predictions ──');
    for (let si = 0; si < SEEDS; si++) {
      let p = predict(targetInits[si], fusedModel, config);
      // Per-cell corrections only for seed 0 (same seed as viewport)
      if (perCellModel && si === 0) {
        p = applyPerCellCorrections(p, targetInits[si], perCellModel, p, config.cellWeight);
      }
      // Validate
      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const s = p[y][x].reduce((a,b)=>a+b,0);
        if (Math.abs(s-1) > 0.02) valid = false;
        for (let c = 0; c < C; c++) if (p[y][x][c] < 0) valid = false;
      }
      if (!valid) { log(`  Seed ${si}: VALIDATION FAILED!`); continue; }

      const res = await POST('/submit', { round_id: roundId, seed_index: si, prediction: p });
      log(`  Seed ${si}: ${res.ok ? 'ACCEPTED' : 'FAILED'} - ${JSON.stringify(res.data).slice(0, 80)}`);
      await sleep(600);
    }
    log('  ✅ Improved predictions submitted!');

    // Try alternative crossWeights
    log('\n  ── Testing alternative crossWeights ──');
    for (const altCW of [10, 20, 50]) {
      if (altCW === cw) continue;
      const altFused = fuseModels(crossModel, viewportModel, altCW);
      let maxDiff = 0;
      const p1 = predict(targetInits[0], fusedModel, config);
      const p2 = predict(targetInits[0], altFused, config);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
        for (let c = 0; c < C; c++)
          maxDiff = Math.max(maxDiff, Math.abs(p1[y][x][c] - p2[y][x][c]));
      log(`  cw=${altCW}: max diff from cw=${cw}: ${maxDiff.toFixed(4)}`);
    }
  }

  log('\n  ✅ SUBMISSION COMPLETE');
}

// ═══ MAIN ═══
async function main() {
  log('╔══════════════════════════════════════════════════╗');
  log('║  Ultimate Astar Island Solver v2                 ║');
  log(`║  Mode: ${MODE.padEnd(42)}║`);
  log('╚══════════════════════════════════════════════════╝');

  if (MODE === 'validate' || MODE === 'full') {
    // Phase 1: Grid search for best parameters
    const gridResults = await gridSearch();

    if (gridResults.length > 0) {
      const bestConfig = gridResults[0].config;
      log(`\n🏆 Best base config: ${gridResults[0].label} (LOO=${gridResults[0].loo.toFixed(2)})`);

      // Phase 2: Cross-weight search
      const cwResults = await crossWeightSearch(bestConfig);
      const bestCW = cwResults.length > 0 ? cwResults[0].cw : 30;
      log(`\n🏆 Best crossWeight: ${bestCW} (LOO=${cwResults.length > 0 ? cwResults[0].loo.toFixed(2) : '?'})`);

      // Phase 3: Per-cell weight search
      const cellResults = await cellWeightSearch(bestConfig, bestCW);
      const bestCellW = cellResults.length > 0 ? cellResults[0].cellWeight : 0;
      log(`\n🏆 Best cellWeight: ${bestCellW} (LOO=${cellResults.length > 0 ? cellResults[0].loo.toFixed(2) : '?'})`);

      // Save best config
      const finalConfig = {
        ...bestConfig,
        crossWeight: bestCW,
        cellWeight: bestCellW
      };
      fs.writeFileSync(path.join(DATA_DIR, 'best_config.json'), JSON.stringify(finalConfig, null, 2));
      log(`\nBest config saved to data/best_config.json`);

      if (MODE === 'full') {
        // Phase 4: Find and submit for active round
        const { data: rounds } = await GET('/rounds');
        const active = rounds.find(r => r.status === 'active');
        if (active) {
          log(`\nActive round found: R${active.round_number} (${active.id})`);
          await submitForRound(active.id, finalConfig);
        } else {
          log('\nNo active round. Run with --mode submit when a round is active.');
        }
      }
    }
  } else if (MODE === 'submit') {
    // Load best config or use defaults
    let config;
    const configFile = path.join(DATA_DIR, 'best_config.json');
    if (fs.existsSync(configFile)) {
      config = JSON.parse(fs.readFileSync(configFile, 'utf8'));
      log(`Loaded config from ${configFile}`);
    } else {
      config = { ws: [1, 0, 0.3, 0.15, 0.08, 0.02], pow: 0.5, minN: 2, fl: 0.00005, featIdx: 0, alpha: 0.05, crossWeight: 30, cellWeight: 0 };
      log('Using default config');
    }

    const { data: rounds } = await GET('/rounds');
    const active = rounds.find(r => r.status === 'active');
    if (active) {
      await submitForRound(active.id, config);
    } else {
      log('No active round found!');
      // Wait for next round
      log('Waiting for next round...');
      while (true) {
        try {
          const { data: r2 } = await GET('/rounds');
          const a = r2.find(r => r.status === 'active');
          if (a) {
            log(`\n🎯 Round R${a.round_number} is active! (${a.id})`);
            await submitForRound(a.id, config);
            break;
          }
        } catch (e) {
          log(`  Network error: ${e.message}`);
        }
        log('  Checking again in 30s...');
        await sleep(30000);
      }
    }
  }

  log('\n═══ DONE ═══');
}

main().catch(e => { console.error('Fatal:', e.message, e.stack); process.exit(1); });
