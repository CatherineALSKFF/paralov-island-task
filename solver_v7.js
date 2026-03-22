#!/usr/bin/env node
/**
 * ═══════════════════════════════════════════════════════════════════
 * ASTAR ISLAND SOLVER v7 — The Proper One
 * ═══════════════════════════════════════════════════════════════════
 *
 * THREE-PHASE approach to score 100:
 *
 * Phase 1: GT-based cross-round model (~83 pts baseline)
 *   - Trains on completed growth rounds' GT probability distributions
 *   - Multi-level feature blending with conservative params
 *
 * Phase 2: Viewport observations from /simulate (~boost to 88-92+)
 *   - 45 queries: 9 viewports × 5 seeds covering full 40×40 grid
 *   - Each call = independent simulation = independent sample
 *   - Bayesian update: posterior = prior + viewport evidence
 *
 * Phase 3: Replay collection after round closes (~95-100)
 *   - Unlimited replays from /replay API
 *   - Each replay = full 40×40 grid at year 50
 *   - Aggregate hundreds of replays → near-perfect distributions
 *
 * Usage:
 *   node solver_v7.js --token <JWT>                     # auto-detect active round, full pipeline
 *   node solver_v7.js --token <JWT> --phase 1            # GT model only (fast, ~83 pts)
 *   node solver_v7.js --token <JWT> --phase 2            # GT + viewport (needs active round)
 *   node solver_v7.js --token <JWT> --phase 3 --round ID # Replay collection for completed round
 *   node solver_v7.js --token <JWT> --phase 3 --round ID --target 1000
 *   node solver_v7.js --token <JWT> --validate           # LOO cross-validation only
 *
 * Get token: DevTools → Application → Cookies → access_token
 * ═══════════════════════════════════════════════════════════════════
 */

const https = require('https');
const fs = require('fs');

const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5, C = 6;

// Conservative model params — validated on 5-round LOO at ~83 avg
const CFG = { ws: [1, 0.2, 0.1, 0.05, 0.01], pow: 0.5, minN: 2, fl: 0.0001 };

// ═══════════════════════════════════════════════════════════════════
// ARGS
// ═══════════════════════════════════════════════════════════════════
const args = {};
for (let i = 2; i < process.argv.length; i++) {
  if (process.argv[i].startsWith('--')) {
    const key = process.argv[i].slice(2);
    const val = process.argv[i + 1] && !process.argv[i + 1].startsWith('--') ? process.argv[++i] : 'true';
    args[key] = val;
  }
}
if (!args.token) {
  console.log('Usage: node solver_v7.js --token <JWT> [options]');
  console.log('Get token: DevTools → Application → Cookies → access_token');
  process.exit(1);
}
const TOKEN = args.token;
const PHASE = parseInt(args.phase || '2');

// ═══════════════════════════════════════════════════════════════════
// HTTP
// ═══════════════════════════════════════════════════════════════════
function api(method, path, body) {
  return new Promise((resolve, reject) => {
    const url = new URL(BASE + path);
    const payload = body ? JSON.stringify(body) : null;
    const opts = {
      hostname: url.hostname, path: url.pathname + url.search, method,
      headers: { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' }
    };
    if (payload) opts.headers['Content-Length'] = Buffer.byteLength(payload);
    const req = https.request(opts, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        try { resolve({ ok: res.statusCode >= 200 && res.statusCode < 300, status: res.statusCode, data: JSON.parse(data) }); }
        catch { resolve({ ok: false, status: res.statusCode, data }); }
      });
    });
    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}
const GET = path => api('GET', path);
const POST = (path, body) => api('POST', path, body);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const log = msg => { const t = new Date().toISOString().slice(11, 19); console.log(`[${t}] ${msg}`); };

// ═══════════════════════════════════════════════════════════════════
// TERRAIN CONVERSION
// ═══════════════════════════════════════════════════════════════════
function t2c(t) { return (t === 10 || t === 11 || t === 0) ? 0 : (t >= 1 && t <= 5) ? t : 0; }

// ═══════════════════════════════════════════════════════════════════
// FEATURE EXTRACTION (5-level hierarchy)
// ═══════════════════════════════════════════════════════════════════
function cellFeatures(grid, y, x) {
  const t = grid[y][x];
  if (t === 10 || t === 5) return null;
  let nS = 0, co = 0, fN = 0, sR2 = 0;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    if (!dy && !dx) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    const nt = grid[ny][nx];
    if (nt === 1 || nt === 2) nS++;
    if (nt === 10) co = 1;
    if (nt === 4) fN++;
  }
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    if (Math.abs(dy) <= 1 && Math.abs(dx) <= 1) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    if (grid[ny][nx] === 1 || grid[ny][nx] === 2) sR2++;
  }
  const sa = Math.min(nS, 5);
  const sb2 = sR2 === 0 ? 0 : sR2 <= 2 ? 1 : sR2 <= 4 ? 2 : 3;
  const fb = fN <= 1 ? 0 : fN <= 3 ? 1 : 2;
  return [
    `D0_${t}_${sa}_${co}_${sb2}_${fb}`,
    `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`,
    `D2_${t}_${sa > 0 ? 1 : 0}_${co}`,
    `D3_${t}_${co}`,
    `D4_${t}`
  ];
}

// ═══════════════════════════════════════════════════════════════════
// MODEL: Build from GT probability distributions
// ═══════════════════════════════════════════════════════════════════
function buildModel(inits, gts, roundNames) {
  const m = {};
  for (const rn of roundNames) {
    for (let si = 0; si < SEEDS; si++) {
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cellFeatures(inits[rn][si], y, x);
        if (!keys) continue;
        const g = gts[rn][si][y][x];
        for (let ki = 0; ki < keys.length; ki++) {
          const k = keys[ki];
          if (!m[k]) m[k] = { n: 0, s: new Float64Array(C) };
          m[k].n++;
          for (let c = 0; c < C; c++) m[k].s[c] += g[c];
        }
      }
    }
  }
  for (const k of Object.keys(m)) {
    m[k].a = Array.from(m[k].s).map(v => v / m[k].n);
    delete m[k].s;
  }
  return m;
}

// ═══════════════════════════════════════════════════════════════════
// PREDICT: Multi-level blending
// ═══════════════════════════════════════════════════════════════════
function predict(grid, model, cfg) {
  const { ws, pow, minN, fl } = cfg;
  const pred = [];
  for (let y = 0; y < H; y++) {
    pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
      if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }
      const keys = cellFeatures(grid, y, x);
      if (!keys) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }
      const p = [0, 0, 0, 0, 0, 0];
      let wS = 0;
      for (let ki = 0; ki < keys.length; ki++) {
        const d = model[keys[ki]];
        if (d && d.n >= minN) {
          const w = ws[ki] * Math.pow(d.n, pow);
          for (let c = 0; c < C; c++) p[c] += w * d.a[c];
          wS += w;
        }
      }
      if (wS === 0) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }
      let s = 0;
      for (let c = 0; c < C; c++) { p[c] /= wS; if (p[c] < fl) p[c] = fl; s += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

// ═══════════════════════════════════════════════════════════════════
// BAYESIAN UPDATE with viewport observations
// ═══════════════════════════════════════════════════════════════════
function bayesUpdate(pred, obs, pseudoCount) {
  const u = [];
  for (let y = 0; y < H; y++) {
    u[y] = [];
    for (let x = 0; x < W; x++) {
      const o = obs[y] && obs[y][x];
      if (!o || o.length === 0) { u[y][x] = pred[y][x]; continue; }
      const counts = [0, 0, 0, 0, 0, 0];
      for (const c of o) counts[c]++;
      const p = [];
      let s = 0;
      for (let c = 0; c < C; c++) {
        p[c] = pseudoCount * pred[y][x][c] + counts[c];
        s += p[c];
      }
      for (let c = 0; c < C; c++) p[c] /= s;
      u[y][x] = p;
    }
  }
  return u;
}

// ═══════════════════════════════════════════════════════════════════
// REPLAY-BASED PREDICTION (Phase 3)
// ═══════════════════════════════════════════════════════════════════
function replayPredict(counts, alpha) {
  const pred = [];
  for (let y = 0; y < H; y++) {
    pred[y] = [];
    for (let x = 0; x < W; x++) {
      const key = `${y}_${x}`;
      const c = counts[key] || [0, 0, 0, 0, 0, 0];
      const N = c.reduce((a, b) => a + b, 0);
      if (N === 0) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }
      const p = [];
      let s = 0;
      for (let i = 0; i < C; i++) { p[i] = (c[i] + alpha) / (N + C * alpha); s += p[i]; }
      for (let i = 0; i < C; i++) p[i] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

// ═══════════════════════════════════════════════════════════════════
// SCORING
// ═══════════════════════════════════════════════════════════════════
function score(pred, gt) {
  let tKL = 0, tE = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const g = gt[y][x];
    let e = 0;
    for (let c = 0; c < C; c++) if (g[c] > 1e-6) e -= g[c] * Math.log(g[c]);
    if (e < 0.01) continue;
    let kl = 0;
    for (let c = 0; c < C; c++) if (g[c] > 1e-6) kl += g[c] * Math.log(g[c] / Math.max(pred[y][x][c], 1e-15));
    tKL += Math.max(0, kl) * e; tE += e;
  }
  return tE > 0 ? 100 * Math.exp(-3 * tKL / tE) : 0;
}

// ═══════════════════════════════════════════════════════════════════
// VIEWPORT POSITIONS (9 viewports covering 40×40)
// ═══════════════════════════════════════════════════════════════════
function viewportPositions() {
  const starts = [0, 13, 25];
  const vps = [];
  for (const y of starts) for (const x of starts) vps.push({ x, y, width: 15, height: 15 });
  return vps;
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════
async function main() {
  log('🏝️  Astar Island Solver v7');

  // ─── Fetch rounds ───
  const { data: rounds } = await GET('/rounds');
  const active = rounds.find(r => r.status === 'active');
  const completed = rounds.filter(r => r.status === 'completed');
  log(`Rounds: ${completed.length} completed, active=${active ? 'R' + active.round_number : 'none'}`);

  if (active) {
    const mins = Math.floor((new Date(active.closes_at) - new Date()) / 60000);
    log(`R${active.round_number} closes in ${mins} min`);
  }

  // ─── Fetch all data ───
  log('Fetching init states + GT...');
  const inits = {}, gts = {};
  const allRounds = [...completed, ...(active ? [active] : [])];

  await Promise.all(allRounds.map(async r => {
    const rn = 'R' + r.round_number;
    const { data } = await GET('/rounds/' + r.id);
    inits[rn] = data.initial_states.map(is => is.grid);
  }));

  await Promise.all(completed.map(async r => {
    const rn = 'R' + r.round_number;
    gts[rn] = [];
    await Promise.all(Array.from({ length: SEEDS }, (_, si) =>
      GET('/analysis/' + r.id + '/' + si).then(res => { gts[rn][si] = res.data.ground_truth; })
    ));
  }));

  log('Data loaded: ' + Object.keys(inits).join(', '));

  // ─── Detect round regimes ───
  const growthRounds = [];
  for (const rn of Object.keys(gts)) {
    let avgS = 0;
    for (let si = 0; si < SEEDS; si++)
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
        avgS += gts[rn][si][y][x][1] + gts[rn][si][y][x][2];
    avgS /= SEEDS * H * W;
    const isDeath = avgS < 0.01;
    log(`  ${rn}: avgS=${avgS.toFixed(4)} ${isDeath ? '💀 DEATH' : '📈 GROWTH'}`);
    if (!isDeath) growthRounds.push(rn);
  }

  // ─── Validate-only mode ───
  if (args.validate) {
    log('\n═══ CROSS-VALIDATION ═══');
    for (const testRn of growthRounds) {
      const trainRns = growthRounds.filter(r => r !== testRn);
      const model = buildModel(inits, gts, trainRns);
      let total = 0;
      for (let si = 0; si < SEEDS; si++) {
        const p = predict(inits[testRn][si], model, CFG);
        total += score(p, gts[testRn][si]);
      }
      log(`  ${testRn}: ${(total / SEEDS).toFixed(2)} (train: ${trainRns.join('+')})`);
    }
    return;
  }

  // ─── Determine target round ───
  const targetRound = args.round ? rounds.find(r => r.id === args.round) : active;
  if (!targetRound) { log('No active round. Use --round <id> or wait.'); return; }
  const targetRn = 'R' + targetRound.round_number;
  const targetId = targetRound.id;
  log(`\n🎯 Target: ${targetRn} (${targetId})`);

  // ═══════════════════════════════════════════════════════════
  // PHASE 3: REPLAY COLLECTION (for completed rounds)
  // ═══════════════════════════════════════════════════════════
  if (PHASE >= 3 && targetRound.status === 'completed') {
    const target = parseInt(args.target || '500');
    const dataFile = `replay_counts_${targetId}.json`;
    log(`\n═══ PHASE 3: Collecting ${target} replays ═══`);

    // Load existing data
    let allCounts = {};
    let totalCollected = 0;
    if (fs.existsSync(dataFile)) {
      const saved = JSON.parse(fs.readFileSync(dataFile, 'utf-8'));
      allCounts = saved.counts;
      totalCollected = saved.total;
      log(`Loaded ${totalCollected} existing replays`);
    } else {
      for (let si = 0; si < SEEDS; si++) allCounts[si] = {};
    }

    const CONCURRENCY = parseInt(args.concurrency || '15');
    let errors = 0;

    while (totalCollected < target) {
      const batch = [];
      for (let i = 0; i < CONCURRENCY; i++) {
        const si = (totalCollected + i) % SEEDS;
        batch.push((async () => {
          try {
            const res = await POST('/replay', { round_id: targetId, seed_index: si });
            if (!res.ok || !res.data.frames) { errors++; return null; }
            return { si, grid: res.data.frames[res.data.frames.length - 1].grid };
          } catch { errors++; return null; }
        })());
      }
      const results = await Promise.all(batch);
      for (const r of results) {
        if (!r) continue;
        const counts = allCounts[r.si];
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const key = `${y}_${x}`;
          if (!counts[key]) counts[key] = [0, 0, 0, 0, 0, 0];
          counts[key][t2c(r.grid[y][x])]++;
        }
        totalCollected++;
      }

      if (totalCollected % 50 === 0 || totalCollected >= target) {
        fs.writeFileSync(dataFile, JSON.stringify({ counts: allCounts, total: totalCollected }));
        log(`Replays: ${totalCollected}/${target} (${errors} errors)`);
      }
      await sleep(150);
    }

    // Submit replay-based predictions
    log('Submitting replay predictions...');
    const alpha = Math.max(0.02, 0.15 * Math.sqrt(150 / totalCollected));
    log(`Dirichlet alpha=${alpha.toFixed(4)} (for ${totalCollected} replays)`);

    for (let si = 0; si < SEEDS; si++) {
      const p = replayPredict(allCounts[si], alpha);
      const res = await POST('/submit', { round_id: targetId, seed_index: si, prediction: p });
      log(`  Seed ${si}: ${res.data.status || JSON.stringify(res.data)}`);
      await sleep(600);
    }

    // Score against GT if available
    if (gts[targetRn]) {
      let total = 0;
      for (let si = 0; si < SEEDS; si++) {
        const p = replayPredict(allCounts[si], alpha);
        total += score(p, gts[targetRn][si]);
      }
      log(`Replay score: ${(total / SEEDS).toFixed(2)}`);
    }
    return;
  }

  // ═══════════════════════════════════════════════════════════
  // PHASE 1: GT MODEL BASELINE
  // ═══════════════════════════════════════════════════════════
  const trainRns = growthRounds.filter(rn => rn !== targetRn);
  log(`\n═══ PHASE 1: GT model (train: ${trainRns.join('+')}) ═══`);
  const model = buildModel(inits, gts, trainRns);
  log(`Model: ${Object.keys(model).length} feature keys`);

  // Submit Phase 1
  for (let si = 0; si < SEEDS; si++) {
    const p = predict(inits[targetRn][si], model, CFG);
    const res = await POST('/submit', { round_id: targetId, seed_index: si, prediction: p });
    log(`  Seed ${si}: ${res.data.status || JSON.stringify(res.data)}`);
    await sleep(600);
  }
  log('Phase 1 submitted (~83 expected)');

  if (PHASE < 2) return;

  // ═══════════════════════════════════════════════════════════
  // PHASE 2: VIEWPORT OBSERVATIONS (Bayesian update)
  // ═══════════════════════════════════════════════════════════
  if (targetRound.status !== 'active') {
    log('Round not active — skipping viewport observations');
    return;
  }

  const { data: budget } = await GET('/budget');
  const remaining = budget.queries_max - budget.queries_used;
  log(`\n═══ PHASE 2: Viewport observations (${remaining} queries available) ═══`);

  if (remaining < 10) { log('Not enough queries for viewport observations'); return; }

  const vps = viewportPositions();
  const queriesPerSeed = Math.min(Math.floor(remaining / SEEDS), vps.length);
  log(`Using ${queriesPerSeed} viewports per seed`);

  // Collect observations
  const r6obs = {};
  for (let si = 0; si < SEEDS; si++) {
    r6obs[si] = {};
    for (let vi = 0; vi < queriesPerSeed; vi++) {
      const vp = vps[vi];
      try {
        const res = await POST('/simulate', {
          round_id: targetId, seed_index: si, year: 50,
          viewport: vp
        });
        if (res.ok && res.data.grid) {
          for (let dy = 0; dy < vp.height; dy++) for (let dx = 0; dx < vp.width; dx++) {
            const gy = vp.y + dy, gx = vp.x + dx;
            if (gy >= H || gx >= W) continue;
            if (!r6obs[si][gy]) r6obs[si][gy] = {};
            if (!r6obs[si][gy][gx]) r6obs[si][gy][gx] = [];
            r6obs[si][gy][gx].push(t2c(res.data.grid[dy][dx]));
          }
        }
        await sleep(220);
      } catch (e) { log(`  Error: ${e.message}`); }
    }
    let cellCount = 0;
    for (const y of Object.keys(r6obs[si]))
      for (const x of Object.keys(r6obs[si][y])) cellCount++;
    log(`  Seed ${si}: ${cellCount} cells observed`);
  }

  // ─── Validate pseudoCount on R5 ───
  log('\nValidating pseudoCount on R5...');
  // Use replays to simulate viewport observations for R5
  const r5obs = {};
  for (let si = 0; si < SEEDS; si++) r5obs[si] = {};
  let nRep = 0;

  for (let rep = 0; rep < 9; rep++) {
    for (let si = 0; si < SEEDS; si++) {
      try {
        const res = await POST('/replay', { round_id: completed.find(r => r.round_number === (rounds.length > 5 ? rounds.length : 5)).id, seed_index: si });
        if (!res.ok || !res.data.frames) continue;
        const lastGrid = res.data.frames[res.data.frames.length - 1].grid;
        const starts = [0, 13, 25];
        const vy = starts[Math.floor(rep / 3)], vx = starts[rep % 3];
        for (let dy = 0; dy < 15; dy++) for (let dx = 0; dx < 15; dx++) {
          const y = vy + dy, x = vx + dx;
          if (y >= H || x >= W) continue;
          if (!r5obs[si][y]) r5obs[si][y] = {};
          if (!r5obs[si][y][x]) r5obs[si][y][x] = [];
          r5obs[si][y][x].push(t2c(lastGrid[y][x]));
        }
        nRep++;
      } catch {}
    }
    await sleep(200);
  }
  log(`Collected ${nRep} R5 validation replays`);

  // Find last completed growth round for validation
  const valRn = growthRounds[growthRounds.length - 1]; // latest completed growth round
  const valTrainRns = growthRounds.filter(rn => rn !== valRn && rn !== targetRn);
  const valModel = buildModel(inits, gts, valTrainRns);

  let baseScore = 0;
  for (let si = 0; si < SEEDS; si++) {
    const p = predict(inits[valRn][si], valModel, CFG);
    baseScore += score(p, gts[valRn][si]);
  }
  baseScore /= SEEDS;
  log(`${valRn} base (no obs): ${baseScore.toFixed(2)}`);

  let bestPC = 5, bestScore = 0;
  for (const pc of [1, 2, 3, 5, 7, 10, 15, 20]) {
    let total = 0;
    for (let si = 0; si < SEEDS; si++) {
      const p = predict(inits[valRn][si], valModel, CFG);
      const obsGrid = [];
      for (let y = 0; y < H; y++) {
        obsGrid[y] = [];
        for (let x = 0; x < W; x++)
          obsGrid[y][x] = r5obs[si][y] && r5obs[si][y][x] ? r5obs[si][y][x] : [];
      }
      total += score(bayesUpdate(p, obsGrid, pc), gts[valRn][si]);
    }
    const avg = total / SEEDS;
    log(`  pc=${pc}: ${avg.toFixed(2)} ${avg > baseScore ? '✅ improved!' : ''}`);
    if (avg > bestScore) { bestScore = avg; bestPC = pc; }
  }
  log(`Best pseudoCount: ${bestPC} (${bestScore.toFixed(2)} vs base ${baseScore.toFixed(2)})`);

  // ─── Resubmit with Bayesian-updated predictions ───
  if (bestScore > baseScore) {
    log(`\nResubmitting with viewport observations (pc=${bestPC})...`);
    for (let si = 0; si < SEEDS; si++) {
      const basePred = predict(inits[targetRn][si], model, CFG);
      const obsGrid = [];
      for (let y = 0; y < H; y++) {
        obsGrid[y] = [];
        for (let x = 0; x < W; x++)
          obsGrid[y][x] = r6obs[si][y] && r6obs[si][y][x] ? r6obs[si][y][x] : [];
      }
      const updated = bayesUpdate(basePred, obsGrid, bestPC);
      const res = await POST('/submit', { round_id: targetId, seed_index: si, prediction: updated });
      log(`  Seed ${si}: ${res.data.status || JSON.stringify(res.data)}`);
      await sleep(600);
    }
    log('Phase 2 submitted (viewport-enhanced)!');
  } else {
    log('Viewport observations did not improve — keeping Phase 1 predictions');
  }

  log('\n✅ Done! Waiting for scoring...');
}

main().catch(e => { console.error('Fatal:', e.message); process.exit(1); });
