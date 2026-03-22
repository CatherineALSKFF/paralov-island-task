#!/usr/bin/env node
/**
 * R8 FINAL SUBMISSION — Best possible model
 * Uses: GT-weighted multi-level model + D0 viewport fusion + temp scaling + per-cell
 *
 * Key improvements over autopilot's D1+D1 final:
 * 1. Full multi-level model (D0-D4) with gtW=20 GT weighting
 * 2. D0 viewport data INCLUDED in final (autopilot dropped it)
 * 3. Temperature scaling (1.05)
 * 4. Per-cell Bayesian corrections
 * 5. LOO validation before submitting
 */
const fs = require('fs');
const path = require('path');
const https = require('https');
const H = 40, W = 40, SEEDS = 5, C = 6;
const DD = path.join(__dirname, 'data');
const TOKEN = process.argv[2] || '';
const BASE = 'https://api.ainm.no/astar-island';
const R8 = 'c5cdf100-a876-4fb7-b5d8-757162c97989';

function api(m, p, b) {
  return new Promise((res, rej) => {
    const u = new URL(BASE + p); const pl = b ? JSON.stringify(b) : null;
    const o = { hostname: u.hostname, path: u.pathname + u.search, method: m,
      headers: { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' } };
    if (pl) o.headers['Content-Length'] = Buffer.byteLength(pl);
    const r = https.request(o, re => {
      let d = ''; re.on('data', c => d += c);
      re.on('end', () => { try { res({ ok: re.statusCode < 300, status: re.statusCode, data: JSON.parse(d) }); } catch { res({ ok: false, status: re.statusCode, data: d }); } });
    }); r.on('error', rej); if (pl) r.write(pl); r.end();
  });
}
const GET = p => api('GET', p), POST = (p, b) => api('POST', p, b);
const sleep = ms => new Promise(r => setTimeout(r, ms));

function t2c(t) { return (t === 10 || t === 11 || t === 0) ? 0 : (t >= 1 && t <= 5) ? t : 0; }

function cf(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS = 0, co = 0, fN = 0, sR2 = 0;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    if (dy === 0 && dx === 0) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    const nt = g[ny][nx];
    if (nt === 1 || nt === 2) nS++;
    if (nt === 10) co = 1;
    if (nt === 4) fN++;
  }
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    if (Math.abs(dy) <= 1 && Math.abs(dx) <= 1) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    if (g[ny][nx] === 1 || g[ny][nx] === 2) sR2++;
  }
  const sa = Math.min(nS, 5), sb2 = sR2 === 0 ? 0 : sR2 <= 2 ? 1 : sR2 <= 4 ? 2 : 3;
  const fb = fN <= 1 ? 0 : fN <= 3 ? 1 : 2;
  return {
    d0: `D0_${t}_${sa}_${co}_${sb2}_${fb}`,
    d1: `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`,
    d2: `D2_${t}_${sa > 0 ? 1 : 0}_${co}`,
    d3: `D3_${t}_${co}`,
    d4: `D4_${t}`
  };
}

// Build GT-based model
function buildGT(gts, inits, rounds, level, alpha) {
  const m = {};
  for (const rn of rounds) {
    if (!gts[rn] || !inits[rn]) continue;
    for (let si = 0; si < SEEDS; si++) {
      if (!inits[rn][si] || !gts[rn][si]) continue;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cf(inits[rn][si], y, x); if (!keys) continue;
        const k = keys[level];
        if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
        const p = gts[rn][si][y][x];
        for (let c = 0; c < C; c++) m[k].counts[c] += p[c];
        m[k].n++;
      }
    }
  }
  for (const k of Object.keys(m)) {
    const tot = Array.from(m[k].counts).reduce((a, b) => a + b, 0) + C * alpha;
    m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / tot);
  }
  return m;
}

// Build replay-based model
function buildRep(reps, inits, rounds, level, alpha) {
  const m = {};
  for (const rn of rounds) {
    if (!reps[rn] || !inits[rn]) continue;
    for (const rep of reps[rn]) {
      const g = inits[rn][rep.si]; if (!g) continue;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cf(g, y, x); if (!keys) continue;
        const k = keys[level];
        const fc = t2c(rep.finalGrid[y][x]);
        if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
        m[k].n++;
        m[k].counts[fc]++;
      }
    }
  }
  for (const k of Object.keys(m)) {
    const tot = m[k].n + C * alpha;
    m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / tot);
  }
  return m;
}

// Score function for LOO validation
function computeScore(pred, gt) {
  let tE = 0, tWK = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const p = gt[y][x], q = pred[y][x];
    let e = 0;
    for (let c = 0; c < C; c++) if (p[c] > 0.001) e -= p[c] * Math.log(p[c]);
    if (e < 0.01) continue;
    let kl = 0;
    for (let c = 0; c < C; c++) if (p[c] > 0.001) kl += p[c] * Math.log(p[c] / Math.max(q[c], 1e-10));
    tE += e; tWK += e * kl;
  }
  if (tE === 0) return 100;
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * tWK / tE)));
}

// Predict with temperature scaling
function predict(grid, model, temp = 1.05) {
  const pred = [];
  for (let y = 0; y < H; y++) {
    pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
      if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }
      const keys = cf(grid, y, x);
      if (!keys) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }
      const levels = ['d0', 'd1', 'd2', 'd3', 'd4'];
      const ws = [1.0, 0.3, 0.15, 0.08, 0.02];
      const p = [0, 0, 0, 0, 0, 0]; let wS = 0;
      for (let li = 0; li < levels.length; li++) {
        const d = model[keys[levels[li]]];
        if (d && d.n >= 1) {
          const w = ws[li] * Math.pow(d.n, 0.5);
          for (let c = 0; c < C; c++) p[c] += w * d.a[c];
          wS += w;
        }
      }
      if (wS === 0) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }
      let s = 0;
      for (let c = 0; c < C; c++) {
        p[c] = Math.pow(Math.max(p[c] / wS, 1e-10), 1 / temp);
        if (p[c] < 0.00005) p[c] = 0.00005;
        s += p[c];
      }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

async function main() {
  console.log('=== R8 FINAL SUBMISSION ===');
  console.log('Time:', new Date().toISOString());

  // Load ALL training data
  const I = {}, G = {}, R = {};
  const TR = [];
  for (let r = 1; r <= 7; r++) {
    if (r === 3) continue;
    const rn = `R${r}`;
    const iF = path.join(DD, `inits_${rn}.json`);
    const gF = path.join(DD, `gt_${rn}.json`);
    const rF = path.join(DD, `replays_${rn}.json`);
    if (fs.existsSync(iF)) I[rn] = JSON.parse(fs.readFileSync(iF));
    if (fs.existsSync(gF)) G[rn] = JSON.parse(fs.readFileSync(gF));
    if (fs.existsSync(rF)) R[rn] = JSON.parse(fs.readFileSync(rF));
    if (I[rn] && G[rn]) TR.push(rn);
  }
  console.log('Training rounds:', TR.join(', '));
  console.log('Replays:', TR.filter(r => R[r]).map(r => `${r}=${R[r].length}`).join(', '));

  // Load R8 viewport data
  const vpFile = path.join(DD, 'viewport_c5cdf100.json');
  let vpObs = [];
  if (fs.existsSync(vpFile)) {
    vpObs = JSON.parse(fs.readFileSync(vpFile));
    console.log('Viewport observations:', vpObs.length);
  }

  // Load R8 initial states
  const { data: rd } = await GET('/rounds/' + R8);
  const inits = rd.initial_states.map(is => is.grid);
  console.log('R8 seeds loaded');

  // === Build multi-level GT-weighted cross-round model ===
  const GTW = 20;
  const TEMP = 1.05;
  const ALPHA = 0.05;

  // LOO validation to find best config
  console.log('\n=== LOO VALIDATION ===');
  const configs = [
    { gtW: 10, temp: 1.0, label: 'gtW=10 t=1.0' },
    { gtW: 10, temp: 1.05, label: 'gtW=10 t=1.05' },
    { gtW: 20, temp: 1.0, label: 'gtW=20 t=1.0' },
    { gtW: 20, temp: 1.05, label: 'gtW=20 t=1.05' },
    { gtW: 20, temp: 1.1, label: 'gtW=20 t=1.1' },
    { gtW: 30, temp: 1.05, label: 'gtW=30 t=1.05' },
    { gtW: 50, temp: 1.05, label: 'gtW=50 t=1.05' },
  ];

  let bestConfig = null, bestLOO = -1;
  for (const cfg of configs) {
    const scores = [];
    for (const holdout of TR) {
      const trainFold = TR.filter(r => r !== holdout);
      // Build model for this fold
      const model = {};
      for (const level of ['d0', 'd1', 'd2', 'd3', 'd4']) {
        const gtM = buildGT(G, I, trainFold, level, ALPHA);
        const repRounds = trainFold.filter(r => R[r]);
        const repM = repRounds.length > 0 ? buildRep(R, I, repRounds, level, ALPHA) : {};
        const allKeys = new Set([...Object.keys(gtM), ...Object.keys(repM)]);
        for (const k of allKeys) {
          const gm = gtM[k], rm = repM[k];
          if (gm && rm) {
            const c = new Float64Array(C);
            for (let i = 0; i < C; i++) c[i] = rm.counts[i] + gm.counts[i] * cfg.gtW;
            const t = Array.from(c).reduce((a, b) => a + b, 0) + C * ALPHA;
            if (!model[k]) model[k] = { n: rm.n + gm.n * cfg.gtW, counts: c, a: Array.from(c).map(v => (v + ALPHA) / t) };
          } else if (gm) {
            const c = new Float64Array(C);
            for (let i = 0; i < C; i++) c[i] = gm.counts[i] * cfg.gtW;
            const t = Array.from(c).reduce((a, b) => a + b, 0) + C * ALPHA;
            if (!model[k]) model[k] = { n: gm.n * cfg.gtW, counts: c, a: Array.from(c).map(v => (v + ALPHA) / t) };
          } else if (rm) {
            if (!model[k]) model[k] = { n: rm.n, counts: rm.counts, a: rm.a.slice() };
          }
        }
      }
      // Score on holdout
      for (let si = 0; si < SEEDS; si++) {
        if (!I[holdout] || !I[holdout][si] || !G[holdout] || !G[holdout][si]) continue;
        const p = predict(I[holdout][si], model, cfg.temp);
        scores.push(computeScore(p, G[holdout][si]));
      }
    }
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    console.log(`  ${cfg.label}: LOO=${avg.toFixed(2)}`);
    if (avg > bestLOO) { bestLOO = avg; bestConfig = cfg; }
  }
  console.log(`\nBest: ${bestConfig.label} LOO=${bestLOO.toFixed(2)}`);

  // === Build FINAL model with ALL training data + best config ===
  console.log('\n=== BUILDING FINAL MODEL ===');
  const crossModel = {};
  for (const level of ['d0', 'd1', 'd2', 'd3', 'd4']) {
    const gtM = buildGT(G, I, TR, level, ALPHA);
    const repRounds = TR.filter(r => R[r]);
    const repM = repRounds.length > 0 ? buildRep(R, I, repRounds, level, ALPHA) : {};
    const allKeys = new Set([...Object.keys(gtM), ...Object.keys(repM)]);
    for (const k of allKeys) {
      const gm = gtM[k], rm = repM[k];
      if (gm && rm) {
        const c = new Float64Array(C);
        for (let i = 0; i < C; i++) c[i] = rm.counts[i] + gm.counts[i] * bestConfig.gtW;
        const t = Array.from(c).reduce((a, b) => a + b, 0) + C * ALPHA;
        if (!crossModel[k]) crossModel[k] = { n: rm.n + gm.n * bestConfig.gtW, counts: c, a: Array.from(c).map(v => (v + ALPHA) / t) };
      } else if (gm) {
        const c = new Float64Array(C);
        for (let i = 0; i < C; i++) c[i] = gm.counts[i] * bestConfig.gtW;
        const t = Array.from(c).reduce((a, b) => a + b, 0) + C * ALPHA;
        if (!crossModel[k]) crossModel[k] = { n: gm.n * bestConfig.gtW, counts: c, a: Array.from(c).map(v => (v + ALPHA) / t) };
      } else if (rm) {
        if (!crossModel[k]) crossModel[k] = { n: rm.n, counts: rm.counts, a: rm.a.slice() };
      }
    }
  }
  console.log('Cross-round model:', Object.keys(crossModel).length, 'keys');

  // === Fuse with viewport D0 data ===
  if (vpObs.length > 0) {
    console.log('\n=== FUSING WITH VIEWPORT DATA ===');
    // Build D0 viewport model using each seed's init grid
    // Group obs by seed
    const obsBySeed = {};
    for (const obs of vpObs) {
      const si = obs.si !== undefined ? obs.si : 0;
      if (!obsBySeed[si]) obsBySeed[si] = [];
      obsBySeed[si].push(obs);
    }
    console.log('Obs by seed:', Object.entries(obsBySeed).map(([s, o]) => `s${s}=${o.length}`).join(', '));

    // Build per-seed viewport D0 models and per-cell models
    const vpModels = {};
    const cellModels = {};
    for (let si = 0; si < SEEDS; si++) {
      const seedObs = obsBySeed[si] || [];
      if (seedObs.length === 0) continue;

      // D0 viewport model for this seed
      const vm = {};
      for (const obs of seedObs) {
        for (let dy = 0; dy < obs.grid.length; dy++) {
          for (let dx = 0; dx < obs.grid[0].length; dx++) {
            const gy = obs.vy + dy, gx = obs.vx + dx;
            if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
            const keys = cf(inits[si], gy, gx);
            if (!keys) continue;
            const k = keys.d0;
            const fc = t2c(obs.grid[dy][dx]);
            if (!vm[k]) vm[k] = { n: 0, counts: new Float64Array(C) };
            vm[k].n++;
            vm[k].counts[fc]++;
          }
        }
      }
      for (const k of Object.keys(vm)) {
        const total = vm[k].n + C * 0.1;
        vm[k].a = Array.from(vm[k].counts).map(v => (v + 0.1) / total);
      }
      vpModels[si] = vm;

      // Per-cell model
      const cells = {};
      for (const obs of seedObs) {
        for (let dy = 0; dy < obs.grid.length; dy++) {
          for (let dx = 0; dx < obs.grid[0].length; dx++) {
            const gy = obs.vy + dy, gx = obs.vx + dx;
            if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
            if (inits[si][gy][gx] === 10 || inits[si][gy][gx] === 5) continue;
            const k = `${gy},${gx}`;
            const fc = t2c(obs.grid[dy][dx]);
            if (!cells[k]) cells[k] = { n: 0, counts: new Float64Array(C) };
            cells[k].n++;
            cells[k].counts[fc]++;
          }
        }
      }
      cellModels[si] = cells;
      console.log(`  Seed ${si}: ${Object.keys(vm).length} D0 keys, ${Object.keys(cells).filter(k => cells[k].n >= 3).length} cells (>=3 obs)`);
    }

    // Also build combined D0 viewport (all seeds together — features transfer)
    const combinedVP = {};
    for (const obs of vpObs) {
      const si = obs.si !== undefined ? obs.si : 0;
      for (let dy = 0; dy < obs.grid.length; dy++) {
        for (let dx = 0; dx < obs.grid[0].length; dx++) {
          const gy = obs.vy + dy, gx = obs.vx + dx;
          if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
          const keys = cf(inits[si], gy, gx);
          if (!keys) continue;
          const k = keys.d0;
          const fc = t2c(obs.grid[dy][dx]);
          if (!combinedVP[k]) combinedVP[k] = { n: 0, counts: new Float64Array(C) };
          combinedVP[k].n++;
          combinedVP[k].counts[fc]++;
        }
      }
    }
    for (const k of Object.keys(combinedVP)) {
      const total = combinedVP[k].n + C * 0.1;
      combinedVP[k].a = Array.from(combinedVP[k].counts).map(v => (v + 0.1) / total);
    }
    console.log('Combined D0 viewport:', Object.keys(combinedVP).length, 'keys');

    // Test different crossWeights for viewport fusion
    console.log('\n=== TESTING VIEWPORT FUSION WEIGHTS ===');
    const cwTests = [10, 15, 20, 30, 50];

    // We can't LOO the viewport (it's R8-specific), but we can check consistency
    // Use the combined viewport + cross-round model
    for (const cw of cwTests) {
      // Build fused model
      const fusedModel = {};
      // Start with all cross-round keys
      for (const [k, v] of Object.entries(crossModel)) {
        fusedModel[k] = { n: v.n, a: v.a.slice() };
      }
      // Fuse D0 viewport with cross-round D0 prior
      for (const [d0key, vm] of Object.entries(combinedVP)) {
        const bm = crossModel[d0key];
        if (bm) {
          const priorAlpha = bm.a.map(p => p * cw);
          const posterior = priorAlpha.map((a, c) => a + vm.counts[c]);
          const total = posterior.reduce((a, b) => a + b, 0);
          fusedModel[d0key] = { n: bm.n + vm.n, a: posterior.map(v => v / total) };
        } else {
          // Use D1 key as fallback prior
          const parts = d0key.split('_');
          const t = parts[1], sa = parseInt(parts[2]), co = parts[3], sb2 = parts[4];
          const d1key = `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`;
          const cm = crossModel[d1key];
          if (cm) {
            const priorAlpha = cm.a.map(p => p * cw);
            const posterior = priorAlpha.map((a, c) => a + vm.counts[c]);
            const total = posterior.reduce((a, b) => a + b, 0);
            fusedModel[d0key] = { n: vm.n, a: posterior.map(v => v / total) };
          } else {
            fusedModel[d0key] = { n: vm.n, a: vm.a.slice() };
          }
        }
      }
      console.log(`  cw=${cw}: ${Object.keys(fusedModel).length} keys`);
    }

    // === SUBMIT with cw=20 (balanced prior + viewport) ===
    const BEST_CW = 20;
    console.log(`\n=== SUBMITTING R8 with cw=${BEST_CW} ===`);

    for (let si = 0; si < SEEDS; si++) {
      // Build seed-specific fused model
      const fusedModel = {};
      for (const [k, v] of Object.entries(crossModel)) {
        fusedModel[k] = { n: v.n, a: v.a.slice() };
      }

      // Use combined viewport (all seeds' features transfer)
      for (const [d0key, vm] of Object.entries(combinedVP)) {
        const bm = crossModel[d0key];
        if (bm) {
          const priorAlpha = bm.a.map(p => p * BEST_CW);
          const posterior = priorAlpha.map((a, c) => a + vm.counts[c]);
          const total = posterior.reduce((a, b) => a + b, 0);
          fusedModel[d0key] = { n: bm.n + vm.n, a: posterior.map(v => v / total) };
        } else {
          const parts = d0key.split('_');
          const t = parts[1], sa = parseInt(parts[2]), co = parts[3], sb2 = parts[4];
          const d1key = `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`;
          const cm = crossModel[d1key];
          if (cm) {
            const priorAlpha = cm.a.map(p => p * BEST_CW);
            const posterior = priorAlpha.map((a, c) => a + vm.counts[c]);
            const total = posterior.reduce((a, b) => a + b, 0);
            fusedModel[d0key] = { n: vm.n, a: posterior.map(v => v / total) };
          } else {
            fusedModel[d0key] = { n: vm.n, a: vm.a.slice() };
          }
        }
      }

      // Predict with temperature
      let pred = predict(inits[si], fusedModel, bestConfig.temp);

      // Apply per-cell corrections
      if (cellModels[si]) {
        for (const [key, cell] of Object.entries(cellModels[si])) {
          if (cell.n < 3) continue;
          const [y, x] = key.split(',').map(Number);
          if (inits[si][y][x] === 10 || inits[si][y][x] === 5) continue;
          const prior = pred[y][x];
          const posterior = new Array(C);
          let total = 0;
          for (let c = 0; c < C; c++) {
            posterior[c] = prior[c] * 5 + cell.counts[c];
            total += posterior[c];
          }
          for (let c = 0; c < C; c++) posterior[c] /= total;
          pred[y][x] = posterior;
        }
      }

      // Validate
      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const s = pred[y][x].reduce((a, b) => a + b, 0);
        if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false;
      }
      if (!valid) { console.log(`Seed ${si}: VALIDATION FAILED`); continue; }

      const res = await POST('/submit', { round_id: R8, seed_index: si, prediction: pred });
      console.log(`Seed ${si}: ${res.ok ? 'ACCEPTED' : 'FAILED'} ${JSON.stringify(res.data).slice(0, 80)}`);
      await sleep(500);
    }

    console.log('\n=== R8 SUBMITTED ===');
    console.log(`Model: multi-level GTW=${bestConfig.gtW} + D0 viewport cw=${BEST_CW} + temp=${bestConfig.temp} + per-cell`);
    console.log('Need score > 80.3 to beat #1 (118.63)');
  } else {
    // No viewport — just submit cross-round model
    console.log('\nNo viewport data — submitting cross-round only');
    for (let si = 0; si < SEEDS; si++) {
      const pred = predict(inits[si], crossModel, bestConfig.temp);
      const res = await POST('/submit', { round_id: R8, seed_index: si, prediction: pred });
      console.log(`Seed ${si}: ${res.ok ? 'ACCEPTED' : 'FAILED'} ${JSON.stringify(res.data).slice(0, 80)}`);
      await sleep(500);
    }
  }
}

main().catch(e => console.error('Error:', e.message, e.stack));
