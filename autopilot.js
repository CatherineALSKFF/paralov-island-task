#!/usr/bin/env node
/**
 * AUTOPILOT — Runs indefinitely, handles ALL rounds automatically.
 *
 * 1. Immediately re-submits R7 seed 4 with best GT-based model
 * 2. Monitors for new rounds
 * 3. When a new round starts: baseline → viewport queries → fused resubmit
 * 4. After round completes: collect replays + GT for future rounds
 *
 * Usage: node autopilot.js <JWT>
 */
const https = require('https');
const fs = require('fs');
const path = require('path');
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5, C = 6;
const TOKEN = process.argv[2] || '';
const DATA_DIR = path.join(__dirname, 'data');

if (!TOKEN) { console.log('Usage: node autopilot.js <JWT>'); process.exit(1); }
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

// ═══ API ═══
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
function buildModelFromReplays(replaysMap, initsMap, roundNames, level='d1', alpha=0.05) {
  const m = {};
  for (const rn of roundNames) {
    if (!replaysMap[rn] || !initsMap[rn]) continue;
    for (const rep of replaysMap[rn]) {
      const initGrid = initsMap[rn][rep.si]; if (!initGrid) continue;
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

function buildModelFromGT(gtsMap, initsMap, roundNames, level='d1', alpha=0.05) {
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
        const si = obs.si !== undefined ? obs.si : 0;
        // Use the appropriate init grid for this seed
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

// ═══ FUSION ═══
function buildBestCrossModel(replaysMap, initsMap, gtsMap, trainRounds, level='d1') {
  // Merge replay-based and GT-based models for maximum accuracy
  const replayModel = buildModelFromReplays(replaysMap, initsMap, trainRounds, level, 0.05);
  const roundsWithGT = trainRounds.filter(rn => gtsMap[rn]);
  if (roundsWithGT.length === 0) return replayModel;

  const gtModel = buildModelFromGT(gtsMap, initsMap, roundsWithGT, level, 0.05);

  // Merge: GT counts are treated as high-quality evidence
  // Weight GT more heavily since it represents the true distribution
  const merged = {};
  const allKeys = new Set([...Object.keys(replayModel), ...Object.keys(gtModel)]);
  for (const k of allKeys) {
    const rm = replayModel[k];
    const gm = gtModel[k];
    if (rm && gm) {
      const counts = new Float64Array(C);
      // GT soft-counts are already excellent — weight them 20x (LOO-optimized)
      for (let c = 0; c < C; c++) counts[c] = rm.counts[c] + gm.counts[c] * 20;
      const total = Array.from(counts).reduce((a,b)=>a+b,0) + C * 0.05;
      merged[k] = { n: rm.n + gm.n * 20, counts, a: Array.from(counts).map(v => (v + 0.05) / total) };
    } else if (gm) {
      const counts = new Float64Array(C);
      for (let c = 0; c < C; c++) counts[c] = gm.counts[c] * 20;
      const total = Array.from(counts).reduce((a,b)=>a+b,0) + C * 0.05;
      merged[k] = { n: gm.n * 20, counts, a: Array.from(counts).map(v => (v + 0.05) / total) };
    } else {
      merged[k] = { n: rm.n, counts: rm.counts, a: rm.a.slice() };
    }
  }
  return merged;
}

function fuseMixedLevel(crossModelD1, viewportModelD0, crossWeight) {
  const m = {};
  // Copy all D1 cross-round entries
  for (const k of Object.keys(crossModelD1)) {
    m[k] = { n: crossModelD1[k].n, a: crossModelD1[k].a.slice() };
  }
  // For each D0 viewport key, fuse D1 prior + D0 viewport observations
  for (const [d0key, vm] of Object.entries(viewportModelD0)) {
    const parts = d0key.split('_');
    const t = parts[1], sa = parseInt(parts[2]), co = parts[3], sb2 = parts[4];
    const d1key = `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`;
    const cm = crossModelD1[d1key];
    if (cm) {
      const priorAlpha = cm.a.map(p => p * crossWeight);
      const posterior = priorAlpha.map((a, c) => a + vm.counts[c]);
      const total = posterior.reduce((a,b)=>a+b, 0);
      m[d0key] = { n: vm.n, a: posterior.map(v => v / total) };
    } else {
      m[d0key] = { n: vm.n, a: vm.a.slice() };
    }
  }
  return m;
}

// ═══ PREDICTION ═══
function predict(grid, model, fl=0.00005, temp=1.1) {
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1,0,0,0,0,0]; continue; }
      if (t === 5) { pred[y][x] = [0,0,0,0,0,1]; continue; }
      const keys = cf(grid, y, x);
      if (!keys) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
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

function applyPerCellCorrections(pred, grid, perCellModel) {
  const corrected = pred.map(row => row.map(p => [...p]));
  for (const [key, cell] of Object.entries(perCellModel)) {
    const [y, x] = key.split(',').map(Number);
    if (grid[y][x] === 10 || grid[y][x] === 5) continue;
    // Adaptive prior weight: more observations = less prior needed
    let pw;
    if (cell.n >= 5) pw = 2;
    else if (cell.n >= 3) pw = 4;
    else if (cell.n >= 2) pw = 7;
    else pw = 15; // 1 obs: 1/(15+1) = 6.25% influence
    const prior = corrected[y][x];
    const posterior = new Array(C);
    let total = 0;
    for (let c = 0; c < C; c++) {
      posterior[c] = prior[c] * pw + cell.counts[c];
      total += posterior[c];
    }
    if (total > 0) {
      for (let c = 0; c < C; c++) posterior[c] /= total;
      corrected[y][x] = posterior;
    }
  }
  return corrected;
}

// ═══ VIEWPORT PLANNING ═══
function planViewportsMultiSeed() {
  const starts = [0, 13, 25];
  const plans = {};
  for (let si = 0; si < SEEDS; si++) {
    const viewports = [];
    for (const vy of starts) for (const vx of starts)
      viewports.push({ vy, vx });
    const offsets = [
      { vy: 7, vx: 7 }, { vy: 7, vx: 20 },
      { vy: 20, vx: 7 }, { vy: 20, vx: 20 },
      { vy: 12, vx: 12 }
    ];
    viewports.push(offsets[si]);
    plans[si] = viewports;
  }
  return plans;
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
          saveViewportData(roundId, allObservations);
          success = true;
        } else if (res.status === 429) {
          await sleep(1500);
        } else if (res.data && (res.data.detail === 'Query budget exhausted' || (typeof res.data === 'string' && res.data.includes('budget')))) {
          log(`  ⚠️ Budget exhausted after ${queryCount} queries`);
          return { queryCount, failures };
        } else {
          log(`  VP (${vp.vy},${vp.vx}) s${seedIndex}: ${JSON.stringify(res.data).slice(0,100)}`);
          failures++; success = true;
        }
      } catch (e) { await sleep(500); }
    }
    await sleep(250);
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
  try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch { return null; }
}

// ═══ REPLAY + GT COLLECTION ═══
async function collectReplaysQuick(roundId, roundName, count=200, concurrency=10) {
  const f = path.join(DATA_DIR, `replays_${roundName}.json`);
  let existing = [];
  if (fs.existsSync(f)) {
    try { existing = JSON.parse(fs.readFileSync(f, 'utf8')); } catch {}
    if (existing.length >= count) return existing;
  }
  const needed = count - existing.length;
  log(`  Collecting ${needed} replays for ${roundName}...`);
  const results = [...existing]; let collected = 0, errors = 0, consec = 0;
  while (collected < needed) {
    const batch = [];
    for (let i = 0; i < Math.min(concurrency, needed - collected); i++) {
      const si = (collected + i) % SEEDS;
      batch.push((async () => {
        try {
          const res = await POST('/replay', { round_id: roundId, seed_index: si });
          if (!res.ok || !res.data.frames) { errors++; consec++; return null; }
          consec = 0;
          return { si, finalGrid: res.data.frames[res.data.frames.length - 1].grid };
        } catch { errors++; consec++; return null; }
      })());
    }
    const batchResults = await Promise.all(batch);
    for (const r of batchResults) { if (r) { results.push(r); collected++; } }
    if (collected % 50 < concurrency) {
      fs.writeFileSync(f, JSON.stringify(results));
      log(`    ${results.length}/${count} replays (${errors} errors)`);
    }
    if (consec > 20) { await sleep(5000); consec = 0; }
    await sleep(150);
  }
  fs.writeFileSync(f, JSON.stringify(results));
  log(`  ✅ ${results.length} replays saved for ${roundName}`);
  return results;
}

async function collectGT(roundId, roundName) {
  const f = path.join(DATA_DIR, `gt_${roundName}.json`);
  if (fs.existsSync(f)) return JSON.parse(fs.readFileSync(f, 'utf8'));

  log(`  Loading GT for ${roundName}...`);
  const gts = [];
  for (let si = 0; si < SEEDS; si++) {
    const res = await GET('/analysis/' + roundId + '/' + si);
    if (res.ok && res.data && res.data.ground_truth) {
      gts[si] = res.data.ground_truth;
    } else {
      log(`  ⚠️ No GT for ${roundName} seed ${si}`);
      return null;
    }
  }
  fs.writeFileSync(f, JSON.stringify(gts));
  log(`  ✅ GT saved for ${roundName}`);
  return gts;
}

async function collectInits(roundId, roundName) {
  const f = path.join(DATA_DIR, `inits_${roundName}.json`);
  if (fs.existsSync(f)) return JSON.parse(fs.readFileSync(f, 'utf8'));

  log(`  Loading inits for ${roundName}...`);
  const { data } = await GET('/rounds/' + roundId);
  const inits = data.initial_states.map(is => is.grid);
  fs.writeFileSync(f, JSON.stringify(inits));
  return inits;
}

// ═══ SCORING (for self-validation when GT is available) ═══
function computeScore(pred, gt) {
  let tE = 0, tWK = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const p = gt[y][x], q = pred[y][x];
    let e = 0; for (let c = 0; c < C; c++) if (p[c] > 0.001) e -= p[c] * Math.log(p[c]);
    if (e < 0.01) continue;
    let kl = 0; for (let c = 0; c < C; c++) if (p[c] > 0.001) kl += p[c] * Math.log(p[c] / Math.max(q[c], 1e-10));
    tE += e; tWK += e * kl;
  }
  if (tE === 0) return 100;
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * tWK / tE)));
}

// ═══ LOAD ALL CACHED DATA ═══
function loadAllCachedData() {
  const replaysMap = {}, initsMap = {}, gtsMap = {};
  const trainRounds = [];
  const DEATH_ROUNDS = [3];

  for (let r = 1; r <= 30; r++) {
    if (DEATH_ROUNDS.includes(r)) continue;
    const rn = `R${r}`;
    const repFile = path.join(DATA_DIR, `replays_${rn}.json`);
    const initFile = path.join(DATA_DIR, `inits_${rn}.json`);
    const gtFile = path.join(DATA_DIR, `gt_${rn}.json`);

    if (fs.existsSync(initFile)) {
      try { initsMap[rn] = JSON.parse(fs.readFileSync(initFile, 'utf8')); } catch {}
    }
    if (fs.existsSync(repFile)) {
      try { replaysMap[rn] = JSON.parse(fs.readFileSync(repFile, 'utf8')); } catch {}
    }
    if (fs.existsSync(gtFile)) {
      try { gtsMap[rn] = JSON.parse(fs.readFileSync(gtFile, 'utf8')); } catch {}
    }

    if (initsMap[rn] && (replaysMap[rn] || gtsMap[rn])) {
      trainRounds.push(rn);
    }
  }

  return { replaysMap, initsMap, gtsMap, trainRounds };
}

// ═══ FULL ROUND PIPELINE ═══
async function handleNewRound(roundId, roundNumber, roundWeight) {
  log(`\n${'═'.repeat(60)}`);
  log(`🎯 HANDLING R${roundNumber} (${roundId.slice(0,8)}...) weight=${roundWeight}`);
  log(`${'═'.repeat(60)}`);

  // Load all cached data
  const { replaysMap, initsMap, gtsMap, trainRounds } = loadAllCachedData();
  log(`Training data: ${trainRounds.join(', ')}`);
  log(`Replays: ${trainRounds.filter(r=>replaysMap[r]).map(r=>`${r}=${replaysMap[r].length}`).join(', ')}`);
  log(`GT: ${trainRounds.filter(r=>gtsMap[r]).join(', ')}`);

  // Load target round
  const { data: targetRound } = await GET('/rounds/' + roundId);
  const targetInits = targetRound.initial_states.map(is => is.grid);
  log(`Target: R${roundNumber}, ${SEEDS} seeds`);

  // Build best cross-round model (GT + replays merged)
  log('\n── Building cross-round models ──');
  const crossD1 = buildBestCrossModel(replaysMap, initsMap, gtsMap, trainRounds, 'd1');
  const crossD0 = buildBestCrossModel(replaysMap, initsMap, gtsMap, trainRounds, 'd0');
  log(`D1 model: ${Object.keys(crossD1).length} keys`);
  log(`D0 model: ${Object.keys(crossD0).length} keys`);

  // Add fallback levels
  const crossD2 = buildBestCrossModel(replaysMap, initsMap, gtsMap, trainRounds, 'd2');
  const crossD3 = buildBestCrossModel(replaysMap, initsMap, gtsMap, trainRounds, 'd3');
  const crossD4 = buildBestCrossModel(replaysMap, initsMap, gtsMap, trainRounds, 'd4');

  // Build baseline model with all levels
  const baselineModel = { ...crossD0 };
  for (const [k,v] of Object.entries(crossD1)) if (!baselineModel[k]) baselineModel[k] = v;
  for (const [k,v] of Object.entries(crossD2)) if (!baselineModel[k]) baselineModel[k] = v;
  for (const [k,v] of Object.entries(crossD3)) if (!baselineModel[k]) baselineModel[k] = v;
  for (const [k,v] of Object.entries(crossD4)) if (!baselineModel[k]) baselineModel[k] = v;

  // ── Submit baseline ──
  log('\n── Submitting BASELINE ──');
  for (let si = 0; si < SEEDS; si++) {
    const p = predict(targetInits[si], baselineModel);
    const res = await POST('/submit', { round_id: roundId, seed_index: si, prediction: p });
    log(`  Seed ${si}: ${res.ok ? '✅' : '❌'} ${JSON.stringify(res.data).slice(0, 80)}`);
    await sleep(600);
  }
  log('  Baseline submitted!');

  // ── Execute viewport queries ──
  log('\n── Executing viewport queries ──');
  let allObservations = loadViewportData(roundId) || [];

  if (allObservations.length > 0) {
    log(`  Loaded ${allObservations.length} existing viewport observations`);
  } else {
    const vpPlans = planViewportsMultiSeed();
    log(`  Strategy: 10 queries × ${SEEDS} seeds = 50 total`);

    let totalQ = 0, totalF = 0;
    for (let si = 0; si < SEEDS; si++) {
      log(`  Seed ${si}...`);
      const { queryCount, failures } = await executeViewportQueries(
        roundId, si, vpPlans[si], allObservations
      );
      totalQ += queryCount; totalF += failures;
      log(`    ${queryCount} queries, ${failures} failures, ${allObservations.filter(o=>o.si===si).length} obs`);
    }
    log(`  ✅ Total: ${allObservations.length} observations (${totalQ} queries, ${totalF} failures)`);
  }

  if (allObservations.length === 0) {
    log('  ⚠️ No viewport data! Keeping baseline.');
    return;
  }

  // ── Build viewport models ──
  log('\n── Building viewport models ──');
  const obsBySeed = {};
  for (const obs of allObservations) {
    const si = obs.si !== undefined ? obs.si : 0;
    if (!obsBySeed[si]) obsBySeed[si] = [];
    obsBySeed[si].push(obs);
  }
  log(`  By seed: ${Object.entries(obsBySeed).map(([s,o])=>`s${s}=${o.length}`).join(', ')}`);

  // Build D0 viewport model using CORRECT per-seed init grids
  const vpD0 = {};
  for (const obs of allObservations) {
    const si = obs.si !== undefined ? obs.si : 0;
    for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
      const gy = obs.vy + dy, gx = obs.vx + dx;
      if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
      const keys = cf(targetInits[si], gy, gx); if (!keys) continue;
      const k = keys.d0, fc = t2c(obs.grid[dy][dx]);
      if (!vpD0[k]) vpD0[k] = { n: 0, counts: new Float64Array(C) };
      vpD0[k].n++; vpD0[k].counts[fc]++;
    }
  }
  log(`  D0 viewport: ${Object.keys(vpD0).length} keys`);

  // Per-cell models per seed
  const perCellModels = {};
  for (const [si, obs] of Object.entries(obsBySeed)) {
    perCellModels[si] = buildPerCellModel(targetInits[parseInt(si)], obs, 0.5);
  }

  // ── Submit with D0-only VP fusion at cw=20 (non-final, will be overwritten by FINAL) ──
  log(`\n── Submitting with D0-VP cw=20 (intermediate) ──`);
  {
    const interModel = { ...baselineModel };
    for (const [k, vm] of Object.entries(vpD0)) {
      const bm = interModel[k];
      if (bm) {
        const pa = bm.a.map(p => p * 20);
        const post = pa.map((a, c) => a + vm.counts[c]);
        const tot = post.reduce((a,b)=>a+b, 0);
        interModel[k] = { n: bm.n + vm.n, a: post.map(v => v / tot) };
      } else {
        const parts = k.split('_');
        const t = parts[1], sa = parseInt(parts[2]), co = parts[3], sb2 = parts[4];
        const d1k = `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`;
        const cm = interModel[d1k];
        if (cm) {
          const pa = cm.a.map(p => p * 20);
          const post = pa.map((a, c) => a + vm.counts[c]);
          const tot = post.reduce((a,b)=>a+b, 0);
          interModel[k] = { n: vm.n + 20, a: post.map(v => v / tot) };
        } else {
          const tot = vm.n + C * 0.1;
          interModel[k] = { n: vm.n, a: Array.from(vm.counts).map(v => (v + 0.1) / tot) };
        }
      }
    }
    for (let si = 0; si < SEEDS; si++) {
      let p = predict(targetInits[si], interModel);
      // No per-cell yet on intermediate
      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const s = p[y][x].reduce((a,b)=>a+b,0);
        if (Math.abs(s-1) > 0.02 || p[y][x].some(v => v < 0)) valid = false;
      }
      if (!valid) { log(`  Seed ${si}: VALIDATION FAILED`); continue; }
      const res = await POST('/submit', { round_id: roundId, seed_index: si, prediction: p });
      log(`  Seed ${si}: ${res.ok ? '✅' : '❌'} ${JSON.stringify(res.data).slice(0, 60)}`);
      await sleep(600);
    }
  }

  // ── FINAL BEST: D0-only VP fusion (cw=20) + all per-cell adaptive ──
  log('\n── FINAL SUBMISSION: D0-VP(cw=20) + temp=1.1 + all per-cell adaptive ──');
  const CW = 20;
  const finalModel = { ...baselineModel }; // Start with full multi-level cross-round
  // Fuse D0 viewport observations into cross-round D0 keys
  for (const [k, vm] of Object.entries(vpD0)) {
    const bm = finalModel[k];
    if (bm) {
      const pa = bm.a.map(p => p * CW);
      const post = pa.map((a, c) => a + vm.counts[c]);
      const tot = post.reduce((a,b)=>a+b, 0);
      finalModel[k] = { n: bm.n + vm.n, a: post.map(v => v / tot) };
    } else {
      // No cross-round D0 key, try D1 as prior
      const parts = k.split('_');
      const t = parts[1], sa = parseInt(parts[2]), co = parts[3], sb2 = parts[4];
      const d1k = `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`;
      const cm = finalModel[d1k];
      if (cm) {
        const pa = cm.a.map(p => p * CW);
        const post = pa.map((a, c) => a + vm.counts[c]);
        const tot = post.reduce((a,b)=>a+b, 0);
        finalModel[k] = { n: vm.n + CW, a: post.map(v => v / tot) };
      } else {
        const tot = vm.n + C * 0.1;
        finalModel[k] = { n: vm.n, a: Array.from(vm.counts).map(v => (v + 0.1) / tot) };
      }
    }
  }
  log(`  VP D0 keys fused: ${Object.keys(vpD0).length}`);
  log(`  Final model: ${Object.keys(finalModel).length} keys`);

  for (let si = 0; si < SEEDS; si++) {
    let p = predict(targetInits[si], finalModel);
    if (perCellModels[si]) {
      p = applyPerCellCorrections(p, targetInits[si], perCellModels[si]);
    }
    // Validate
    let valid = true;
    for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
      const s = p[y][x].reduce((a,b)=>a+b,0);
      if (Math.abs(s-1) > 0.02 || p[y][x].some(v => v < 0)) valid = false;
    }
    if (!valid) { log(`  Seed ${si}: VALIDATION FAILED`); continue; }
    const res = await POST('/submit', { round_id: roundId, seed_index: si, prediction: p });
    log(`  Seed ${si}: ${res.ok ? '✅' : '❌'} ${JSON.stringify(res.data).slice(0, 60)}`);
    await sleep(600);
  }

  log(`\n🏆 R${roundNumber} COMPLETE — all submissions done`);
  log(`   Weight: ${roundWeight}, need 84.3+ for #1 (118.63/${roundWeight})`);
}

// ═══ POST-ROUND DATA COLLECTION ═══
async function collectPostRoundData(roundId, roundName) {
  log(`\n── Post-round data collection for ${roundName} ──`);

  await collectInits(roundId, roundName);
  await collectGT(roundId, roundName);
  await collectReplaysQuick(roundId, roundName, 500);

  log(`  ✅ ${roundName} data collected`);
}

// ═══ MAIN LOOP ═══
async function main() {
  log('╔══════════════════════════════════════════════════════╗');
  log('║  🤖 AUTOPILOT — Handling ALL rounds automatically   ║');
  log('║  Go to sleep, I got this!                           ║');
  log('╚══════════════════════════════════════════════════════╝');

  // Track which rounds we've handled
  const handledRounds = new Set();
  const collectedRounds = new Set();

  // ── Step 0: Improve R7 seed 4 submission ──
  log('\n═══ Step 0: Improving R7 submission ═══');

  try {
    const { replaysMap, initsMap, gtsMap, trainRounds } = loadAllCachedData();
    log(`Training data available: ${trainRounds.join(', ')}`);

    const R7_ID = '36e581f1-73f8-453f-ab98-cbe3052b701b';

    // Check if R7 is still active
    const { data: rounds } = await GET('/rounds');
    const r7 = rounds.find(r => r.id === R7_ID);

    if (r7 && r7.status === 'active') {
      const closes = new Date(r7.closes_at);
      const minsLeft = Math.round((closes - Date.now()) / 60000);
      log(`R7 still active (${minsLeft} min left). Re-submitting with best model...`);

      // Load R7 initial states
      const { data: r7Data } = await GET('/rounds/' + R7_ID);
      const r7Inits = r7Data.initial_states.map(is => is.grid);

      // Build best cross-round model from ALL available GT + replays
      const crossD1 = buildBestCrossModel(replaysMap, initsMap, gtsMap, trainRounds, 'd1');
      const crossD2 = buildBestCrossModel(replaysMap, initsMap, gtsMap, trainRounds, 'd2');
      const crossD3 = buildBestCrossModel(replaysMap, initsMap, gtsMap, trainRounds, 'd3');
      const crossD4 = buildBestCrossModel(replaysMap, initsMap, gtsMap, trainRounds, 'd4');

      log(`D1 model: ${Object.keys(crossD1).length} keys from ${trainRounds.length} rounds`);

      // Check for existing R7 viewport data (probably lost, but check)
      const r7Viewport = loadViewportData(R7_ID);

      if (r7Viewport && r7Viewport.length > 0) {
        log(`  Found ${r7Viewport.length} viewport observations! Using them.`);
        // Build viewport model and fuse
        const vpModelD0 = buildViewportFeatureModel(r7Inits[0], r7Viewport, 'd0', 0.1);
        const fusedModel = fuseMixedLevel(crossD1, vpModelD0, 30);
        for (const [k,v] of Object.entries(crossD2)) if (!fusedModel[k]) fusedModel[k] = v;
        for (const [k,v] of Object.entries(crossD3)) if (!fusedModel[k]) fusedModel[k] = v;
        for (const [k,v] of Object.entries(crossD4)) if (!fusedModel[k]) fusedModel[k] = v;

        const obsBySeed = {};
        for (const obs of r7Viewport) {
          const si = obs.si !== undefined ? obs.si : 0;
          if (!obsBySeed[si]) obsBySeed[si] = [];
          obsBySeed[si].push(obs);
        }

        for (let si = 0; si < SEEDS; si++) {
          let p = predict(r7Inits[si], fusedModel);
          if (obsBySeed[si]) {
            const pcm = buildPerCellModel(r7Inits[si], obsBySeed[si], 0.5);
            p = applyPerCellCorrections(p, r7Inits[si], pcm, 5);
          }
          const res = await POST('/submit', { round_id: R7_ID, seed_index: si, prediction: p });
          log(`  R7 seed ${si}: ${res.ok ? '✅' : '❌'} ${JSON.stringify(res.data).slice(0, 60)}`);
          await sleep(600);
        }
      } else {
        log(`  No R7 viewport data found. Re-submitting ALL seeds with improved GT cross-round model.`);
        // Seeds 0-3 had viewport-enhanced submissions from old pipeline
        // But that old pipeline used only 50 replays/round for cross-round
        // Our new GT-based model is MUCH better for the cross-round component
        // However, we lost the viewport data, so we can't fuse viewport + new cross-round
        //
        // Decision: Only re-submit seed 4 (which didn't get viewport data)
        // Seeds 0-3 keep their viewport-enhanced predictions

        // Build model with all levels
        const model = { ...crossD1 };
        for (const [k,v] of Object.entries(crossD2)) if (!model[k]) model[k] = v;
        for (const [k,v] of Object.entries(crossD3)) if (!model[k]) model[k] = v;
        for (const [k,v] of Object.entries(crossD4)) if (!model[k]) model[k] = v;

        // Submit seed 4 with improved model
        const si = 4;
        const p = predict(r7Inits[si], model);
        const res = await POST('/submit', { round_id: R7_ID, seed_index: si, prediction: p });
        log(`  R7 seed ${si}: ${res.ok ? '✅' : '❌'} ${JSON.stringify(res.data).slice(0, 60)}`);

        log(`  ✅ R7 seed 4 re-submitted with improved cross-round model`);
        log(`  Seeds 0-3: keeping viewport-enhanced submissions from earlier pipeline`);
      }

      handledRounds.add(R7_ID);
    } else if (r7 && r7.status === 'completed') {
      log(`R7 already completed. Collecting data...`);
      await collectPostRoundData(R7_ID, 'R7');
      collectedRounds.add(R7_ID);
      handledRounds.add(R7_ID);
    }
  } catch (e) {
    log(`R7 handling error: ${e.message}`);
  }

  // ── Main monitoring loop ──
  log('\n═══ Entering monitoring loop ═══');
  log('Will check for new rounds every 30 seconds...\n');

  while (true) {
    try {
      const { data: rounds } = await GET('/rounds');

      // Check for newly completed rounds that need data collection
      for (const r of rounds) {
        if (r.status === 'completed' && !collectedRounds.has(r.id) && r.round_number !== 3) {
          const rn = `R${r.round_number}`;
          const gtFile = path.join(DATA_DIR, `gt_${rn}.json`);
          const repFile = path.join(DATA_DIR, `replays_${rn}.json`);

          // Check if we already have GT + replays
          if (!fs.existsSync(gtFile) || !fs.existsSync(repFile)) {
            log(`\n📦 New completed round: ${rn}. Collecting data...`);
            await collectPostRoundData(r.id, rn);
          }
          collectedRounds.add(r.id);
        }
      }

      // Check for active round that needs handling
      const active = rounds.find(r => r.status === 'active' && !handledRounds.has(r.id));
      if (active) {
        const closes = new Date(active.closes_at);
        const minsLeft = Math.round((closes - Date.now()) / 60000);

        if (minsLeft > 5) { // Only start if we have enough time
          log(`\n🆕 NEW ROUND DETECTED: R${active.round_number} (${minsLeft} min left)`);
          await handleNewRound(active.id, active.round_number, active.round_weight);
          handledRounds.add(active.id);
        } else {
          log(`  R${active.round_number} only ${minsLeft} min left, too late to start pipeline`);
          handledRounds.add(active.id);
        }
      } else {
        const currentActive = rounds.find(r => r.status === 'active');
        if (currentActive) {
          const closes = new Date(currentActive.closes_at);
          const minsLeft = Math.round((closes - Date.now()) / 60000);
          process.stdout.write(`\r[${new Date().toISOString().slice(11,19)}] R${currentActive.round_number} active (${minsLeft}m left, already handled). Waiting...  `);
        } else {
          process.stdout.write(`\r[${new Date().toISOString().slice(11,19)}] No active round. Waiting for next round...  `);
        }
      }
    } catch (e) {
      log(`Loop error: ${e.message}. Retrying in 30s...`);
    }

    await sleep(30000);
  }
}

main().catch(e => { console.error('Fatal:', e.message, e.stack); process.exit(1); });
