#!/usr/bin/env node
/**
 * R8+ Ultimate Pipeline
 *
 * KEY IMPROVEMENTS over r7_feature_viewport.js:
 * 1. Viewport data saved to disk IMMEDIATELY (never lose it again)
 * 2. Spread viewport queries across seeds (10 per seed × 5 = 50)
 *    → Feature model benefits all seeds + per-cell corrections for all
 * 3. Uses D1 features for cross-round (validated: 81.72 LOO vs D0's 81.49)
 * 4. Uses D0 features for viewport (more specific = better within-round)
 * 5. Mixed-level fusion: D1 cross-round prior + D0 viewport observations
 * 6. Loads cached replays from disk (collect_replays_massive.js)
 * 7. Per-cell Bayesian corrections for viewport-observed cells
 * 8. Automatic round detection with robust error handling
 * 9. Iterative resubmission with different crossWeights
 *
 * Usage: node r8_pipeline.js <JWT> [--round-id ID] [--no-wait]
 */
const https = require('https');
const fs = require('fs');
const path = require('path');
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5, C = 6;
const TOKEN = process.argv[2] || '';
const DATA_DIR = path.join(__dirname, 'data');
const ROUND_ARG = process.argv.findIndex(a => a === '--round-id');
const ROUND_ID_OVERRIDE = ROUND_ARG >= 0 ? process.argv[ROUND_ARG + 1] : null;
const NO_WAIT = process.argv.includes('--no-wait');

if (!TOKEN) { console.log('Usage: node r8_pipeline.js <JWT> [--round-id ID] [--no-wait]'); process.exit(1); }
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
// D0: specific (for viewport model)
// D1: coarser (for cross-round model, validated better LOO)
function cf(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS=0,co=0,fN=0,sR2=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
    if(!dy&&!dx)continue;
    const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;
    const nt=g[ny][nx];
    if(nt===1||nt===2)nS++;
    if(nt===10)co=1;
    if(nt===4)fN++;
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
  return {
    d0: `D0_${t}_${sa}_${co}_${sb2}_${fb}`,
    d1: `D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,
    d2: `D2_${t}_${sa>0?1:0}_${co}`,
    d3: `D3_${t}_${co}`,
    d4: `D4_${t}`
  };
}

// ═══ MODEL BUILDING ═══
function buildCrossRoundModel(replaysMap, initsMap, roundNames, level='d1', alpha=0.05) {
  const m = {};
  for (const rn of roundNames) {
    if (!replaysMap[rn] || !initsMap[rn]) continue;
    for (const rep of replaysMap[rn]) {
      const initGrid = initsMap[rn][rep.si];
      if (!initGrid) continue;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cf(initGrid, y, x); if (!keys) continue;
        const k = keys[level];
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

// Build from GT directly (more accurate when replays aren't available)
function buildCrossRoundModelFromGT(gtsMap, initsMap, roundNames, level='d1', alpha=0.05) {
  const m = {};
  for (const rn of roundNames) {
    if (!gtsMap[rn] || !initsMap[rn]) continue;
    for (let si = 0; si < SEEDS; si++) {
      if (!initsMap[rn][si] || !gtsMap[rn][si]) continue;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cf(initsMap[rn][si], y, x); if (!keys) continue;
        const k = keys[level];
        if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
        const p = gtsMap[rn][si][y][x];
        for (let c = 0; c < C; c++) m[k].counts[c] += p[c];
        m[k].n++;
      }
    }
  }
  for (const k of Object.keys(m)) {
    const total = Array.from(m[k].counts).reduce((a,b)=>a+b,0) + C * alpha;
    m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / total);
  }
  return m;
}

function buildViewportFeatureModel(initGrid, observations, level='d0', alpha=0.1) {
  const m = {};
  for (const obs of observations) {
    for (let dy = 0; dy < obs.grid.length; dy++) {
      for (let dx = 0; dx < obs.grid[0].length; dx++) {
        const gy = obs.vy + dy, gx = obs.vx + dx;
        if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
        const keys = cf(initGrid, gy, gx); if (!keys) continue;
        const k = keys[level];
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

function buildPerCellModel(initGrid, observations, alpha=0.5) {
  const cells = {};
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

// ═══ MIXED-LEVEL FUSION ═══
// Cross-round (D1) + Viewport (D0) → D0-level predictions
// For each D0 viewport key, find the matching D1 cross-round key as prior
function fuseMixedLevel(crossModelD1, viewportModelD0, initGrid, crossWeight) {
  const m = {};
  // First, copy all cross-round D1 entries
  for (const k of Object.keys(crossModelD1)) {
    m[k] = { n: crossModelD1[k].n, a: crossModelD1[k].a.slice(), level: 'd1' };
  }
  // For each D0 viewport key, create a D0-level entry by fusing D1 prior + D0 viewport
  for (const [d0key, vm] of Object.entries(viewportModelD0)) {
    // Extract terrain and features from D0 key to find D1 key
    // D0_t_sa_co_sb2_fb → D1_t_min(sa,3)_co_sb2
    const parts = d0key.split('_');
    const t = parts[1], sa = parseInt(parts[2]), co = parts[3], sb2 = parts[4];
    const d1key = `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`;

    const cm = crossModelD1[d1key];
    if (cm) {
      const priorAlpha = cm.a.map(p => p * crossWeight);
      const posterior = priorAlpha.map((a, c) => a + vm.counts[c]);
      const total = posterior.reduce((a,b)=>a+b, 0);
      m[d0key] = { n: vm.n, a: posterior.map(v => v / total), level: 'd0_fused' };
    } else {
      m[d0key] = { n: vm.n, a: vm.a.slice(), level: 'd0_viewport_only' };
    }
  }
  return m;
}

// Same-level fusion (D0+D0 or D1+D1) for comparison
function fuseSameLevel(crossModel, viewportModel, crossWeight) {
  const m = {};
  const allKeys = new Set([...Object.keys(crossModel), ...Object.keys(viewportModel)]);
  for (const k of allKeys) {
    const cm = crossModel[k]; const vm = viewportModel[k];
    if (cm && vm) {
      const priorAlpha = cm.a.map(p => p * crossWeight);
      const posterior = priorAlpha.map((a, c) => a + vm.counts[c]);
      const total = posterior.reduce((a,b)=>a+b, 0);
      m[k] = { n: cm.n + vm.n, a: posterior.map(v => v / total) };
    } else if (vm) { m[k] = { n: vm.n, a: vm.a.slice() }; }
    else { m[k] = { n: cm.n, a: cm.a.slice() }; }
  }
  return m;
}

// ═══ PREDICTION ═══
// Multi-level hierarchical prediction with temperature scaling
function predict(grid, model, fl=0.00005, temp=1.05) {
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1,0,0,0,0,0]; continue; }
      if (t === 5) { pred[y][x] = [0,0,0,0,0,1]; continue; }
      const keys = cf(grid, y, x);
      if (!keys) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }

      // Try D0 first (fused or viewport), then D1 (cross-round), then D2, D3, D4
      const levels = ['d0', 'd1', 'd2', 'd3', 'd4'];
      const ws = [1.0, 0.3, 0.15, 0.08, 0.02];
      const p = [0,0,0,0,0,0]; let wS = 0;

      for (let li = 0; li < levels.length; li++) {
        const d = model[keys[levels[li]]];
        if (d && d.n >= 1) {
          const w = ws[li] * Math.pow(d.n, 0.5);
          for (let c = 0; c < C; c++) p[c] += w * d.a[c];
          wS += w;
        }
      }

      if (wS === 0) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      let s = 0;
      for (let c = 0; c < C; c++) {
        p[c] = Math.pow(Math.max(p[c]/wS, 1e-10), 1/temp);
        if (p[c] < fl) p[c] = fl;
        s += p[c];
      }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

// Apply per-cell corrections
function applyPerCellCorrections(pred, grid, perCellModel, cellWeight=5) {
  const corrected = pred.map(row => row.map(p => [...p]));
  for (const [key, cell] of Object.entries(perCellModel)) {
    const [y, x] = key.split(',').map(Number);
    if (grid[y][x] === 10 || grid[y][x] === 5) continue;
    if (cell.n < 3) continue;
    const prior = corrected[y][x];
    const posterior = new Array(C);
    let total = 0;
    for (let c = 0; c < C; c++) {
      posterior[c] = prior[c] * cellWeight + cell.counts[c];
      total += posterior[c];
    }
    for (let c = 0; c < C; c++) posterior[c] /= total;
    corrected[y][x] = posterior;
  }
  return corrected;
}

// ═══ VIEWPORT PLANNING ═══
// Spread across seeds: 10 queries per seed
function planViewportsMultiSeed() {
  const starts = [0, 13, 25];
  const plans = {};

  for (let si = 0; si < SEEDS; si++) {
    const viewports = [];
    // 1 full pass = 9 positions
    for (const vy of starts) for (const vx of starts)
      viewports.push({ vy, vx });
    // 1 offset position
    const offsets = [
      { vy: 7, vx: 7 }, { vy: 7, vx: 20 },
      { vy: 20, vx: 7 }, { vy: 20, vx: 20 },
      { vy: 12, vx: 12 }
    ];
    viewports.push(offsets[si]); // Different offset for each seed
    plans[si] = viewports; // 10 queries per seed
  }

  return plans; // Total: 50 queries
}

// Alternative: concentrated on seed 0 (original approach)
function planViewportsSeed0() {
  const starts = [0, 13, 25];
  const viewports = [];
  for (let pass = 0; pass < 5; pass++) {
    for (const vy of starts) for (const vx of starts)
      viewports.push({ vy, vx });
  }
  const extras = [{vy:7,vx:7},{vy:7,vx:20},{vy:20,vx:7},{vy:20,vx:20},{vy:12,vx:12}];
  for (const pos of extras) viewports.push(pos);
  return { 0: viewports }; // All 50 on seed 0
}

async function executeViewportQueries(roundId, seedIndex, viewports, allObservations) {
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
          const obs = { vy: vp.vy, vx: vp.vx, grid: res.data.grid, si: seedIndex };
          allObservations.push(obs);
          // SAVE IMMEDIATELY after each observation
          saveViewportData(roundId, allObservations);
          success = true;
        } else if (res.status === 429) {
          await sleep(1000);
        } else if (res.data && res.data.detail === 'Query budget exhausted') {
          log(`  ⚠️ Budget exhausted at ${queryCount} queries`);
          return { queryCount, failures };
        } else {
          log(`  VP (${vp.vy},${vp.vx}) s${seedIndex} failed: ${JSON.stringify(res.data).slice(0,100)}`);
          failures++; success = true;
        }
      } catch (e) { await sleep(500); }
    }
    await sleep(220);
  }
  return { queryCount, failures };
}

function saveViewportData(roundId, observations) {
  const f = path.join(DATA_DIR, `viewport_${roundId.slice(0,8)}.json`);
  fs.writeFileSync(f, JSON.stringify(observations));
}
function loadViewportData(roundId) {
  const f = path.join(DATA_DIR, `viewport_${roundId.slice(0,8)}.json`);
  if (!fs.existsSync(f)) return null;
  return JSON.parse(fs.readFileSync(f, 'utf8'));
}

// ═══ REPLAY COLLECTION (quick, for newly completed rounds) ═══
async function collectReplaysQuick(roundId, roundName, count=200, concurrency=10) {
  const f = path.join(DATA_DIR, `replays_${roundName}.json`);
  let existing = [];
  if (fs.existsSync(f)) {
    existing = JSON.parse(fs.readFileSync(f, 'utf8'));
    if (existing.length >= count) return existing;
  }
  const needed = count - existing.length;
  log(`  Collecting ${needed} replays for ${roundName}...`);
  const results = [...existing]; let collected = 0, errors = 0;
  while (collected < needed) {
    const batch = [];
    for (let i = 0; i < Math.min(concurrency, needed - collected); i++) {
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
    if (collected % 50 < concurrency) {
      fs.writeFileSync(f, JSON.stringify(results));
      log(`    ${results.length}/${count} replays`);
    }
    await sleep(150);
  }
  fs.writeFileSync(f, JSON.stringify(results));
  return results;
}

// ═══ SCORING ═══
function computeScore(pred, gt) {
  let totalEntropy = 0, totalWeightedKL = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const p = gt[y][x], q = pred[y][x];
    let entropy = 0;
    for (let c = 0; c < C; c++) if (p[c] > 0.001) entropy -= p[c] * Math.log(p[c]);
    if (entropy < 0.01) continue;
    let kl = 0;
    for (let c = 0; c < C; c++) if (p[c] > 0.001) kl += p[c] * Math.log(p[c] / Math.max(q[c], 1e-10));
    totalEntropy += entropy;
    totalWeightedKL += entropy * kl;
  }
  if (totalEntropy === 0) return 100;
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * totalWeightedKL / totalEntropy)));
}

// ═══ MAIN PIPELINE ═══
async function main() {
  log('╔══════════════════════════════════════════════════════╗');
  log('║  R8+ Ultimate Pipeline                               ║');
  log('║  Multi-seed viewport + Mixed-level fusion + Per-cell  ║');
  log('╚══════════════════════════════════════════════════════╝');

  // ═══ Step 0: Find target round ═══
  // Already submitted rounds
  const SUBMITTED = new Set([
    'ae78003a-4efe-425a-881a-d16a39bca0ad', // R6
    '36e581f1-73f8-453f-ab98-cbe3052b701b', // R7
    // R8+ will be handled dynamically
  ]);

  let roundId = ROUND_ID_OVERRIDE;
  let roundNumber = '?', roundWeight = '?';

  if (!roundId) {
    log('\nLooking for target round...');
    while (true) {
      try {
        const { data: rounds } = await GET('/rounds');
        const active = rounds.find(r => r.status === 'active' && !SUBMITTED.has(r.id));
        if (active) {
          roundId = active.id;
          roundNumber = active.round_number;
          roundWeight = active.round_weight;
          log(`\n🎯 Target round: R${roundNumber} (${roundId})`);
          log(`   Closes: ${active.closes_at}`);
          log(`   Weight: ${roundWeight}`);
          break;
        }
        const currentActive = rounds.find(r => r.status === 'active');
        if (currentActive && SUBMITTED.has(currentActive.id)) {
          const closes = new Date(currentActive.closes_at);
          const minsLeft = Math.round((closes - Date.now()) / 60000);
          log(`  R${currentActive.round_number} already submitted (${minsLeft} min left). Waiting...`);
        } else {
          log('  No active round. Checking in 30s...');
        }
      } catch (e) {
        log(`  Network error: ${e.message}. Retrying in 30s...`);
      }
      if (NO_WAIT) { log('--no-wait: exiting.'); return; }
      await sleep(30000);
    }
  }

  // ═══ Step 1: Load all cached data ═══
  log('\n═══ Step 1: Loading cached data ═══');
  const replaysMap = {}, initsMap = {}, gtsMap = {};
  const trainRounds = [];
  const DEATH_ROUNDS = [3]; // Exclude death rounds

  for (let r = 1; r <= 20; r++) {
    if (DEATH_ROUNDS.includes(r)) continue;
    const rn = `R${r}`;
    const repFile = path.join(DATA_DIR, `replays_${rn}.json`);
    const initFile = path.join(DATA_DIR, `inits_${rn}.json`);
    const gtFile = path.join(DATA_DIR, `gt_${rn}.json`);
    if (fs.existsSync(repFile) && fs.existsSync(initFile)) {
      replaysMap[rn] = JSON.parse(fs.readFileSync(repFile, 'utf8'));
      initsMap[rn] = JSON.parse(fs.readFileSync(initFile, 'utf8'));
      if (fs.existsSync(gtFile)) gtsMap[rn] = JSON.parse(fs.readFileSync(gtFile, 'utf8'));
      trainRounds.push(rn);
    }
  }

  // Check for any new completed rounds not yet cached
  const { data: allRounds } = await GET('/rounds');
  for (const r of allRounds) {
    if (r.status !== 'completed' || r.id === roundId) continue;
    if (DEATH_ROUNDS.includes(r.round_number)) continue;
    const rn = `R${r.round_number}`;
    if (!trainRounds.includes(rn)) {
      log(`  New completed round: ${rn}. Loading data...`);
      // Load initial states
      const { data } = await GET('/rounds/' + r.id);
      initsMap[rn] = data.initial_states.map(is => is.grid);
      fs.writeFileSync(path.join(DATA_DIR, `inits_${rn}.json`), JSON.stringify(initsMap[rn]));

      // Load GT
      const gts = [];
      let gotAll = true;
      for (let si = 0; si < SEEDS; si++) {
        const res = await GET('/analysis/' + r.id + '/' + si);
        if (res.ok && res.data && res.data.ground_truth) gts[si] = res.data.ground_truth;
        else gotAll = false;
      }
      if (gotAll) {
        gtsMap[rn] = gts;
        fs.writeFileSync(path.join(DATA_DIR, `gt_${rn}.json`), JSON.stringify(gts));
      }

      // Collect replays (quick, 200)
      replaysMap[rn] = await collectReplaysQuick(r.id, rn, 200);
      trainRounds.push(rn);
    }
  }

  log(`  Training rounds: ${trainRounds.join(', ')}`);
  log(`  Replay counts: ${trainRounds.map(r => `${r}=${replaysMap[r].length}`).join(', ')}`);

  // Load target round
  const { data: targetRound } = await GET('/rounds/' + roundId);
  const targetInits = targetRound.initial_states.map(is => is.grid);
  log(`  Target round: R${roundNumber}, ${SEEDS} seeds`);

  // ═══ Step 2: Build cross-round models ═══
  log('\n═══ Step 2: Building cross-round models ═══');

  // D1-level model (validated best for cross-round, LOO=81.72)
  const crossModelD1 = buildCrossRoundModel(replaysMap, initsMap, trainRounds, 'd1', 0.05);
  const d1Keys = Object.keys(crossModelD1).length;
  log(`  D1 cross-round model: ${d1Keys} keys`);

  // D0-level model (for comparison/fallback)
  const crossModelD0 = buildCrossRoundModel(replaysMap, initsMap, trainRounds, 'd0', 0.05);
  const d0Keys = Object.keys(crossModelD0).length;
  log(`  D0 cross-round model: ${d0Keys} keys`);

  // Also build GT-based model if GT is available
  const roundsWithGT = trainRounds.filter(rn => gtsMap[rn]);
  let crossModelGT_D1 = null;
  if (roundsWithGT.length > 0) {
    crossModelGT_D1 = buildCrossRoundModelFromGT(gtsMap, initsMap, roundsWithGT, 'd1', 0.05);
    log(`  GT-based D1 model: ${Object.keys(crossModelGT_D1).length} keys (from ${roundsWithGT.length} rounds with GT)`);
  }

  // ═══ Step 3: Submit BASELINE ═══
  log('\n═══ Step 3: Submitting BASELINE predictions ═══');

  // Use combined model: merge replay-based and GT-based with GT weight=20 (LOO-optimized)
  const GTW = 20; // GT weight: LOO-validated as optimal
  const TEMP = 1.05; // Temperature: slight softening helps generalization
  const baselineModel = {};

  // Build multi-level model for hierarchical prediction
  for (const level of ['d0', 'd1', 'd2', 'd3', 'd4']) {
    const replayModel = buildCrossRoundModel(replaysMap, initsMap, trainRounds, level, 0.05);
    const gtModel = roundsWithGT.length > 0 ? buildCrossRoundModelFromGT(gtsMap, initsMap, roundsWithGT, level, 0.05) : {};
    const allKeys = new Set([...Object.keys(replayModel), ...Object.keys(gtModel)]);
    for (const k of allKeys) {
      const rm = replayModel[k], gm = gtModel[k];
      if (rm && gm) {
        const c = new Float64Array(C);
        for (let i = 0; i < C; i++) c[i] = rm.counts[i] + gm.counts[i] * GTW;
        const tot = Array.from(c).reduce((a,b)=>a+b,0) + C * 0.05;
        if (!baselineModel[k]) baselineModel[k] = { n: rm.n + gm.n * GTW, counts: c, a: Array.from(c).map(v => (v + 0.05) / tot) };
      } else if (gm) {
        const c = new Float64Array(C);
        for (let i = 0; i < C; i++) c[i] = gm.counts[i] * GTW;
        const tot = Array.from(c).reduce((a,b)=>a+b,0) + C * 0.05;
        if (!baselineModel[k]) baselineModel[k] = { n: gm.n * GTW, counts: c, a: Array.from(c).map(v => (v + 0.05) / tot) };
      } else if (rm) {
        if (!baselineModel[k]) baselineModel[k] = { n: rm.n, counts: rm.counts, a: rm.a.slice() };
      }
    }
  }
  log(`  Baseline model: ${Object.keys(baselineModel).length} keys (GTW=${GTW}, TEMP=${TEMP})`);

  for (let si = 0; si < SEEDS; si++) {
    const p = predict(targetInits[si], baselineModel);
    const res = await POST('/submit', { round_id: roundId, seed_index: si, prediction: p });
    log(`  Seed ${si}: ${res.ok ? 'ACCEPTED' : 'FAILED'} - ${JSON.stringify(res.data).slice(0, 80)}`);
    await sleep(600);
  }
  log('  ✅ Baseline submitted');

  // ═══ Step 4: Execute viewport queries ═══
  log('\n═══ Step 4: Executing viewport queries ═══');

  // Check for existing viewport data
  let allObservations = loadViewportData(roundId) || [];

  if (allObservations.length > 0) {
    log(`  Loaded ${allObservations.length} existing viewport observations from disk`);
  } else {
    // STRATEGY: Spread across seeds (10 per seed) for maximum benefit
    const vpPlans = planViewportsMultiSeed();
    log(`  Strategy: multi-seed (10 queries × ${SEEDS} seeds = 50 total)`);

    let totalQueries = 0, totalFailures = 0;
    for (let si = 0; si < SEEDS; si++) {
      log(`  Querying seed ${si} (${vpPlans[si].length} viewports)...`);
      const { queryCount, failures } = await executeViewportQueries(
        roundId, si, vpPlans[si], allObservations
      );
      totalQueries += queryCount;
      totalFailures += failures;
      log(`    Seed ${si}: ${queryCount} queries, ${failures} failures`);
    }

    log(`  ✅ Total: ${allObservations.length} observations from ${totalQueries} queries (${totalFailures} failures)`);
  }

  if (allObservations.length === 0) {
    log('  ⚠️ No viewport data! Keeping baseline submission.');
    return;
  }

  // ═══ Step 5: Build viewport models per seed ═══
  log('\n═══ Step 5: Building viewport feature models ═══');

  // Group observations by seed
  const obsBySeed = {};
  for (const obs of allObservations) {
    const si = obs.si !== undefined ? obs.si : 0;
    if (!obsBySeed[si]) obsBySeed[si] = [];
    obsBySeed[si].push(obs);
  }

  log(`  Observations by seed: ${Object.entries(obsBySeed).map(([si, obs]) => `s${si}=${obs.length}`).join(', ')}`);

  // Build feature model from ALL observations (applies to all seeds via features)
  const allViewportModelD0 = buildViewportFeatureModel(targetInits[0], allObservations, 'd0', 0.1);
  const vpD0Keys = Object.keys(allViewportModelD0).length;
  let totalVPObs = 0;
  for (const v of Object.values(allViewportModelD0)) totalVPObs += v.n;
  log(`  Combined D0 viewport model: ${vpD0Keys} keys, ${totalVPObs} obs, ${(totalVPObs/Math.max(vpD0Keys,1)).toFixed(1)} avg/key`);

  // Per-cell models per seed
  const perCellModels = {};
  for (const [si, obs] of Object.entries(obsBySeed)) {
    perCellModels[si] = buildPerCellModel(targetInits[parseInt(si)], obs, 0.5);
    const cellCount = Object.keys(perCellModels[si]).filter(k => perCellModels[si][k].n >= 3).length;
    log(`  Seed ${si} per-cell: ${cellCount} cells with ≥3 obs`);
  }

  // ═══ Step 6: Fuse and submit improved predictions ═══
  log('\n═══ Step 6: Fusing models and submitting ═══');

  // Use fused model with best crossWeight (validated: cw=30 with viewport data)
  // IMPORTANT: Submit with BEST model LAST (since last submission overwrites)
  const crossWeightsToTest = [50, 30, 20]; // Last = best = final submission
  const cellWeight = 5; // Per-cell correction weight

  for (const cw of crossWeightsToTest) {
    log(`\n  ── CrossWeight = ${cw} ──`);

    // Build full multi-level fused model:
    // D0: viewport D0 fused with cross-round baseline as prior
    // D1-D4: from baseline (GT-weighted cross-round)
    const fusedModel = {};

    // Start with all baseline keys (multi-level)
    for (const [k, v] of Object.entries(baselineModel)) {
      fusedModel[k] = { n: v.n, a: v.a.slice(), counts: v.counts ? new Float64Array(v.counts) : null };
    }

    // Fuse viewport D0 observations with baseline D0 prior
    for (const [d0key, vm] of Object.entries(allViewportModelD0)) {
      const bm = baselineModel[d0key]; // May have baseline D0 key
      if (bm) {
        const priorAlpha = bm.a.map(p => p * cw);
        const posterior = priorAlpha.map((a, c) => a + vm.counts[c]);
        const total = posterior.reduce((a,b)=>a+b, 0);
        fusedModel[d0key] = { n: bm.n + vm.n, a: posterior.map(v => v / total) };
      } else {
        // Find D1 key as fallback prior
        const parts = d0key.split('_');
        const t = parts[1], sa = parseInt(parts[2]), co = parts[3], sb2 = parts[4];
        const d1key = 'D1_' + t + '_' + Math.min(sa, 3) + '_' + co + '_' + sb2;
        const cm = baselineModel[d1key];
        if (cm) {
          const priorAlpha = cm.a.map(p => p * cw);
          const posterior = priorAlpha.map((a, c) => a + vm.counts[c]);
          const total = posterior.reduce((a,b)=>a+b, 0);
          fusedModel[d0key] = { n: vm.n, a: posterior.map(v => v / total) };
        } else {
          fusedModel[d0key] = { n: vm.n, a: vm.a.slice() };
        }
      }
    }

    const fusedKeys = Object.keys(fusedModel).length;
    log(`  Fused model: ${fusedKeys} keys`);

    for (let si = 0; si < SEEDS; si++) {
      let p = predict(targetInits[si], fusedModel, 0.00005, TEMP);

      // Per-cell corrections if we have data for this seed
      if (perCellModels[si]) {
        p = applyPerCellCorrections(p, targetInits[si], perCellModels[si], cellWeight);
      }

      // Validate
      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const s = p[y][x].reduce((a,b)=>a+b,0);
        if (Math.abs(s-1) > 0.02 || p[y][x].some(v => v < 0)) valid = false;
      }
      if (!valid) { log(`  Seed ${si}: VALIDATION FAILED!`); continue; }

      const res = await POST('/submit', { round_id: roundId, seed_index: si, prediction: p });
      log(`  Seed ${si}: ${res.ok ? 'ACCEPTED' : 'FAILED'} - ${JSON.stringify(res.data).slice(0, 80)}`);
      await sleep(600);
    }
    log(`  ✅ Submitted with cw=${cw}`);
  }

  log('\n╔══════════════════════════════════════════════════════╗');
  log(`║  ✅ R${roundNumber} PIPELINE COMPLETE                          ║`);
  log(`║  Round: ${roundId.slice(0,8)}... Weight: ${roundWeight}           ║`);
  log(`║  VP observations: ${allObservations.length}                            ║`);
  log(`║  Last submission: D0+D0 cw=30 + per-cell                ║`);
  log('╚══════════════════════════════════════════════════════╝');
}

main().catch(e => { console.error('Fatal:', e.message, e.stack); process.exit(1); });
