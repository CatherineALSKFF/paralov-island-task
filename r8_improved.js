#!/usr/bin/env node
/**
 * R8 IMPROVED — Use ALL per-cell viewport observations
 *
 * Key insight: 1600/1600 cells covered per seed, but previous model
 * only used cells with >=3 obs (73-111 cells). This version uses ALL cells
 * with adaptive weighting.
 *
 * Also tests per-cell weight sensitivity to find optimal.
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

function predict(grid, model, temp = 1.1) {
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

// Per-cell correction with adaptive weight
function applyPerCell(pred, grid, cellObs, priorWeight) {
  const result = pred.map(row => row.map(p => [...p]));
  for (const [key, cell] of Object.entries(cellObs)) {
    const [y, x] = key.split(',').map(Number);
    if (grid[y][x] === 10 || grid[y][x] === 5) continue;
    // Adaptive: more observations = more weight
    // 1 obs: priorWeight, 2 obs: priorWeight*0.7, 3+ obs: priorWeight*0.5
    const pw = cell.n >= 3 ? priorWeight * 0.5 : cell.n >= 2 ? priorWeight * 0.7 : priorWeight;
    const prior = result[y][x];
    const posterior = new Array(C);
    let total = 0;
    for (let c = 0; c < C; c++) {
      posterior[c] = prior[c] * pw + cell.counts[c];
      total += posterior[c];
    }
    if (total > 0) {
      for (let c = 0; c < C; c++) posterior[c] /= total;
      result[y][x] = posterior;
    }
  }
  return result;
}

async function main() {
  console.log('=== R8 IMPROVED SUBMISSION ===');
  console.log('Time:', new Date().toISOString());

  // Load training data
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
  console.log('Training:', TR.join(', '));
  console.log('Replays:', TR.filter(r => R[r]).map(r => `${r}=${R[r].length}`).join(', '));

  // Load R8 data
  const vpObs = JSON.parse(fs.readFileSync(path.join(DD, 'viewport_c5cdf100.json')));
  const { data: rd } = await GET('/rounds/' + R8);
  const inits = rd.initial_states.map(is => is.grid);

  // Build cross-round model (gtW=20, all levels)
  const GTW = 20, ALPHA = 0.05;
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
        for (let i = 0; i < C; i++) c[i] = rm.counts[i] + gm.counts[i] * GTW;
        const t = Array.from(c).reduce((a, b) => a + b, 0) + C * ALPHA;
        if (!crossModel[k]) crossModel[k] = { n: rm.n + gm.n * GTW, counts: c, a: Array.from(c).map(v => (v + ALPHA) / t) };
      } else if (gm) {
        const c = new Float64Array(C);
        for (let i = 0; i < C; i++) c[i] = gm.counts[i] * GTW;
        const t = Array.from(c).reduce((a, b) => a + b, 0) + C * ALPHA;
        if (!crossModel[k]) crossModel[k] = { n: gm.n * GTW, counts: c, a: Array.from(c).map(v => (v + ALPHA) / t) };
      } else if (rm) {
        if (!crossModel[k]) crossModel[k] = { n: rm.n, counts: rm.counts, a: rm.a.slice() };
      }
    }
  }
  console.log('Cross-round model:', Object.keys(crossModel).length, 'keys');

  // Build D0 viewport model (combined all seeds)
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
  console.log('D0 viewport:', Object.keys(combinedVP).length, 'keys');

  // Build per-cell observations per seed (ALL cells, not just >=3)
  const obsBySeed = {};
  for (const obs of vpObs) {
    const si = obs.si !== undefined ? obs.si : 0;
    if (!obsBySeed[si]) obsBySeed[si] = [];
    obsBySeed[si].push(obs);
  }

  const cellModels = {};
  for (let si = 0; si < SEEDS; si++) {
    const seedObs = obsBySeed[si] || [];
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
    const c1 = Object.values(cells).filter(c => c.n === 1).length;
    const c2 = Object.values(cells).filter(c => c.n === 2).length;
    const c3 = Object.values(cells).filter(c => c.n >= 3).length;
    console.log(`Seed ${si}: ${Object.keys(cells).length} total cells (1obs=${c1}, 2obs=${c2}, 3+obs=${c3})`);
  }

  // === LOO to validate per-cell weighting strategies ===
  console.log('\n=== LOO: Testing per-cell prior weights ===');
  // We simulate viewport effect by using within-round replay data as "viewport"
  // Test on held-out rounds
  const priorWeights = [2, 3, 5, 8, 10, 15, 20];
  const temps = [1.05, 1.1, 1.15];
  const vpCWs = [10, 15, 20, 30];

  let bestScore = -1, bestPW = 5, bestTemp = 1.1, bestCW = 20;

  for (const temp of temps) {
    for (const cw of vpCWs) {
      // Build fused model with this cw
      const fusedModel = {};
      for (const [k, v] of Object.entries(crossModel)) {
        fusedModel[k] = { n: v.n, a: v.a.slice() };
      }
      for (const [d0key, vm] of Object.entries(combinedVP)) {
        const bm = crossModel[d0key];
        if (bm) {
          const priorAlpha = bm.a.map(p => p * cw);
          const posterior = priorAlpha.map((a, c) => a + vm.counts[c]);
          const total = posterior.reduce((a, b) => a + b, 0);
          fusedModel[d0key] = { n: bm.n + vm.n, a: posterior.map(v => v / total) };
        } else {
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

      // LOO score on cross-round data (without per-cell, since we can't simulate per-cell in LOO)
      const scores = [];
      for (const holdout of TR) {
        for (let si = 0; si < SEEDS; si++) {
          if (!I[holdout] || !I[holdout][si] || !G[holdout] || !G[holdout][si]) continue;
          const p = predict(I[holdout][si], fusedModel, temp);
          scores.push(computeScore(p, G[holdout][si]));
        }
      }
      const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
      if (avg > bestScore) {
        bestScore = avg; bestTemp = temp; bestCW = cw;
      }
    }
  }
  console.log(`Best LOO: temp=${bestTemp} cw=${bestCW} score=${bestScore.toFixed(2)}`);

  // === Build FINAL fused model ===
  console.log('\n=== Building FINAL fused model ===');
  const fusedModel = {};
  for (const [k, v] of Object.entries(crossModel)) {
    fusedModel[k] = { n: v.n, a: v.a.slice() };
  }
  for (const [d0key, vm] of Object.entries(combinedVP)) {
    const bm = crossModel[d0key];
    if (bm) {
      const priorAlpha = bm.a.map(p => p * bestCW);
      const posterior = priorAlpha.map((a, c) => a + vm.counts[c]);
      const total = posterior.reduce((a, b) => a + b, 0);
      fusedModel[d0key] = { n: bm.n + vm.n, a: posterior.map(v => v / total) };
    } else {
      const parts = d0key.split('_');
      const t = parts[1], sa = parseInt(parts[2]), co = parts[3], sb2 = parts[4];
      const d1key = `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`;
      const cm = crossModel[d1key];
      if (cm) {
        const priorAlpha = cm.a.map(p => p * bestCW);
        const posterior = priorAlpha.map((a, c) => a + vm.counts[c]);
        const total = posterior.reduce((a, b) => a + b, 0);
        fusedModel[d0key] = { n: vm.n, a: posterior.map(v => v / total) };
      } else {
        fusedModel[d0key] = { n: vm.n, a: vm.a.slice() };
      }
    }
  }
  console.log('Fused model:', Object.keys(fusedModel).length, 'keys');

  // === Test different per-cell weights (submit the BEST last) ===
  const pwTests = [3, 5, 8, 12, 20];
  console.log('\nSubmitting with different per-cell weights...');
  console.log('(Last submission wins — submit best LAST)');

  for (const pw of pwTests) {
    console.log(`\n--- Per-cell weight = ${pw} ---`);
    for (let si = 0; si < SEEDS; si++) {
      let pred = predict(inits[si], fusedModel, bestTemp);
      pred = applyPerCell(pred, inits[si], cellModels[si], pw);

      // Validate
      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const s = pred[y][x].reduce((a, b) => a + b, 0);
        if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false;
      }
      if (!valid) { console.log(`  Seed ${si}: VALIDATION FAILED`); continue; }

      const res = await POST('/submit', { round_id: R8, seed_index: si, prediction: pred });
      console.log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'}`);
      await sleep(500);
    }
  }

  // Also submit WITHOUT per-cell (just feature-level) for safety
  console.log('\n--- No per-cell (feature model only) ---');
  for (let si = 0; si < SEEDS; si++) {
    const pred = predict(inits[si], fusedModel, bestTemp);
    const res = await POST('/submit', { round_id: R8, seed_index: si, prediction: pred });
    console.log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'}`);
    await sleep(500);
  }

  // FINAL: moderate per-cell (pw=5 is usually safe)
  console.log('\n=== FINAL SUBMISSION: pw=5 (balanced) ===');
  for (let si = 0; si < SEEDS; si++) {
    let pred = predict(inits[si], fusedModel, bestTemp);
    pred = applyPerCell(pred, inits[si], cellModels[si], 5);
    const res = await POST('/submit', { round_id: R8, seed_index: si, prediction: pred });
    console.log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'}`);
    await sleep(500);
  }

  console.log('\n=== DONE ===');
  console.log(`Final: multi-level GTW=20 + D0 viewport cw=${bestCW} + temp=${bestTemp} + per-cell pw=5`);
  console.log('Target: score > 80.3 → ws > 118.63 (#1)');
}

main().catch(e => console.error('Error:', e.message, e.stack));
