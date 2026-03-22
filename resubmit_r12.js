#!/usr/bin/env node
// RESUBMIT R12 — Improved predictions using existing VP + GT data only (0 queries)
// Improvements over autopilot_simple.js:
//   1. VP weight CW=6 (was 20) — VP observations dominate cross-round model
//   2. Per-cell pw: 1 (N>=5), 3 (N>=3), 5 (N>=2) — much lower prior weight
//   3. Temperature=1.0 (was 1.1) — sharper predictions
//   4. Adaptive floor: 0.0001 static, 0.001 dynamic
//   5. Round weighting: recent rounds count more
//   6. Static cells: near-deterministic ocean/mountain

const fs = require('fs'), path = require('path'), https = require('https');
const H = 40, W = 40, SEEDS = 5, C = 6;
const DD = path.join(__dirname, 'data');
const TOKEN = process.argv[2] || '';
const BASE = 'https://api.ainm.no/astar-island';
const ROUND_ID = '795bfb1f-54bd-4f39-a526-9868b36f7ebd';
const ROUND_NUM = 12;
const DO_SUBMIT = process.argv.includes('--submit');

if (!TOKEN) { console.log('Usage: node resubmit_r12.js <JWT> [--submit]'); process.exit(1); }

function api(m, p, b) { return new Promise((res, rej) => {
  const u = new URL(BASE + p); const pl = b ? JSON.stringify(b) : null;
  const o = { hostname: u.hostname, path: u.pathname + u.search, method: m,
    headers: { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' } };
  if (pl) o.headers['Content-Length'] = Buffer.byteLength(pl);
  const r = https.request(o, re => { let d = ''; re.on('data', c => d += c);
    re.on('end', () => { try { res({ ok: re.statusCode < 300, status: re.statusCode, data: JSON.parse(d) }); } catch { res({ ok: false, status: re.statusCode, data: d }); } });
  }); r.on('error', rej); if (pl) r.write(pl); r.end(); }); }
const POST = (p, b) => api('POST', p, b);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const log = m => { const t = new Date().toISOString().slice(11, 19); console.log(`[${t}] ${m}`); };
function t2c(t) { return (t === 10 || t === 11 || t === 0) ? 0 : (t >= 1 && t <= 5) ? t : 0; }

// Same feature computation as autopilot_simple.js
function cf(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS = 0, co = 0, fN = 0, sR2 = 0;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    if (dy === 0 && dx === 0) continue; const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue; const nt = g[ny][nx];
    if (nt === 1 || nt === 2) nS++; if (nt === 10) co = 1; if (nt === 4) fN++; }
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    if (Math.abs(dy) <= 1 && Math.abs(dx) <= 1) continue; const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    if (g[ny][nx] === 1 || g[ny][nx] === 2) sR2++; }
  const sa = Math.min(nS, 5), sb2 = sR2 === 0 ? 0 : sR2 <= 2 ? 1 : sR2 <= 4 ? 2 : 3;
  const fb = fN <= 1 ? 0 : fN <= 3 ? 1 : 2;
  return { d0: `D0_${t}_${sa}_${co}_${sb2}_${fb}`, d1: `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`,
    d2: `D2_${t}_${sa > 0 ? 1 : 0}_${co}`, d3: `D3_${t}_${co}`, d4: `D4_${t}` };
}

// IMPROVED: Round weighting — more recent rounds count more
function roundWeight(rn) {
  const num = parseInt(rn.replace('R', ''));
  // Exponential recency: R11 gets weight ~2.6, R1 gets weight ~1.0
  return Math.pow(1.1, num - 1);
}

function buildModel() {
  const I = {}, G = {}, R = {};
  for (let r = 1; r <= 11; r++) { const rn = `R${r}`;
    if (fs.existsSync(path.join(DD, `inits_${rn}.json`))) I[rn] = JSON.parse(fs.readFileSync(path.join(DD, `inits_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `gt_${rn}.json`))) G[rn] = JSON.parse(fs.readFileSync(path.join(DD, `gt_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `replays_${rn}.json`))) R[rn] = JSON.parse(fs.readFileSync(path.join(DD, `replays_${rn}.json`)));
  }
  const TR = Object.keys(I).filter(k => G[k]);
  log(`Training rounds: ${TR.join(', ')}`);
  for (const rn of TR) log(`  ${rn}: weight=${roundWeight(rn).toFixed(2)}, GT=${G[rn]?'yes':'no'}, replays=${R[rn]?R[rn].length:0}`);

  const model = {};
  for (const level of ['d0', 'd1', 'd2', 'd3', 'd4']) {
    const m = {};
    // GT data with round weighting
    for (const rn of TR) { if (!G[rn] || !I[rn]) continue;
      const rw = roundWeight(rn);
      for (let si = 0; si < SEEDS; si++) { if (!I[rn][si] || !G[rn][si]) continue;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = cf(I[rn][si], y, x); if (!keys) continue; const k = keys[level];
          if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
          const p = G[rn][si][y][x];
          const gtW = 20 * rw; // IMPROVED: scale GT contribution by round weight
          for (let c = 0; c < C; c++) m[k].counts[c] += p[c] * gtW;
          m[k].n += gtW;
        }
      }
    }
    // Replay data with round weighting
    for (const rn of TR) { if (!R[rn] || !I[rn]) continue;
      const rw = roundWeight(rn);
      for (const rep of R[rn]) { const g = I[rn][rep.si]; if (!g) continue;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = cf(g, y, x); if (!keys) continue; const k = keys[level];
          const fc = t2c(rep.finalGrid[y][x]);
          if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
          m[k].n += rw; m[k].counts[fc] += rw;
        }
      }
    }
    // Normalize with smoothing
    for (const k of Object.keys(m)) {
      const tot = Array.from(m[k].counts).reduce((a, b) => a + b, 0) + C * 0.05;
      m[k].a = Array.from(m[k].counts).map(v => (v + 0.05) / tot);
    }
    for (const [k, v] of Object.entries(m)) { if (!model[k]) model[k] = v; }
  }
  log(`Model: ${Object.keys(model).length} keys`);
  return model;
}

// IMPROVED: Temperature=1.0, adaptive floor
function predict(grid, model) {
  const TEMP = 1.0; // IMPROVED: was 1.1
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      // IMPROVED: Static cells — near-deterministic
      if (t === 10) { pred[y][x] = [0.9999, 0.00002, 0.00002, 0.00002, 0.00002, 0.00002]; continue; }
      if (t === 5) { pred[y][x] = [0.00002, 0.00002, 0.00002, 0.00002, 0.00002, 0.9999]; continue; }
      const keys = cf(grid, y, x);
      if (!keys) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }
      const levels = ['d0', 'd1', 'd2', 'd3', 'd4'], ws = [1.0, 0.3, 0.15, 0.08, 0.02];
      const p = [0, 0, 0, 0, 0, 0]; let wS = 0;
      for (let li = 0; li < levels.length; li++) {
        const d = model[keys[levels[li]]];
        if (d && d.n >= 1) { const w = ws[li] * Math.pow(d.n, 0.5);
          for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w; } }
      if (wS === 0) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }
      let s = 0;
      for (let c = 0; c < C; c++) {
        p[c] = Math.pow(Math.max(p[c] / wS, 1e-10), 1 / TEMP);
        s += p[c];
      }
      // IMPROVED: Adaptive floor
      // Determine if cell is "static-like" (dominant class > 95%)
      const maxP = Math.max(...p) / s;
      const floor = maxP > 0.95 ? 0.0001 : 0.001;
      let s2 = 0;
      for (let c = 0; c < C; c++) { p[c] = Math.max(p[c] / s, floor); s2 += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s2;
      pred[y][x] = p;
    }
  }
  return pred;
}

// IMPROVED: VP fusion with CW=6 (was 20)
function fuseVP(model, vpObs, inits) {
  const CW = 6; // IMPROVED: was 20 — VP observations now dominate
  const vpD0 = {};
  for (const obs of vpObs) {
    for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
      const gy = obs.vy + dy, gx = obs.vx + dx;
      if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
      const keys = cf(inits[obs.si], gy, gx); if (!keys) continue;
      const k = keys.d0, fc = t2c(obs.grid[dy][dx]);
      if (!vpD0[k]) vpD0[k] = { n: 0, counts: new Float64Array(C) }; vpD0[k].n++; vpD0[k].counts[fc]++;
    }
  }
  let fused = 0;
  for (const [k, vm] of Object.entries(vpD0)) {
    const bm = model[k];
    if (bm) {
      const pa = bm.a.map(p => p * CW), post = pa.map((a, c) => a + vm.counts[c]);
      const tot = post.reduce((a, b) => a + b, 0);
      model[k] = { n: bm.n + vm.n, a: post.map(v => v / tot) };
      fused++;
    } else {
      // VP key not in model — add it directly
      const tot = Array.from(vm.counts).reduce((a, b) => a + b, 0) + C * 0.05;
      model[k] = { n: vm.n, a: Array.from(vm.counts).map(v => (v + 0.05) / tot) };
      fused++;
    }
  }
  log(`  VP fused: ${fused} D0 keys (CW=${CW})`);
}

function buildCellModels(vpObs, inits) {
  const cm = {};
  for (let si = 0; si < SEEDS; si++) cm[si] = {};
  for (const obs of vpObs) {
    for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
      const gy = obs.vy + dy, gx = obs.vx + dx;
      if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
      if (inits[obs.si][gy][gx] === 10 || inits[obs.si][gy][gx] === 5) continue;
      const k = `${gy},${gx}`, fc = t2c(obs.grid[dy][dx]);
      if (!cm[obs.si][k]) cm[obs.si][k] = { n: 0, counts: new Float64Array(C) };
      cm[obs.si][k].n++; cm[obs.si][k].counts[fc]++;
    }
  }
  return cm;
}

// IMPROVED: Per-cell with much lower prior weights
function applyPerCell(pred, cellModel, initGrid) {
  for (const [key, cell] of Object.entries(cellModel)) {
    const [y, x] = key.split(',').map(Number);
    if (initGrid[y][x] === 10 || initGrid[y][x] === 5) continue;
    // IMPROVED: pw=1 (N>=5), pw=3 (N>=3), pw=5 (N>=2) — was 2,4,7,15
    const pw = cell.n >= 5 ? 1 : cell.n >= 3 ? 3 : cell.n >= 2 ? 5 : 10;
    const prior = pred[y][x], posterior = new Array(C); let total = 0;
    for (let c = 0; c < C; c++) { posterior[c] = prior[c] * pw + cell.counts[c]; total += posterior[c]; }
    if (total > 0) {
      for (let c = 0; c < C; c++) posterior[c] /= total;
      // IMPROVED: Adaptive floor for per-cell results
      const maxP = Math.max(...posterior);
      const floor = maxP > 0.95 ? 0.0001 : 0.001;
      let s = 0;
      for (let c = 0; c < C; c++) { posterior[c] = Math.max(posterior[c], floor); s += posterior[c]; }
      for (let c = 0; c < C; c++) posterior[c] /= s;
      pred[y][x] = posterior;
    }
  }
  return pred;
}

// LOO cross-validation on completed rounds
function looCrossValidation() {
  log('\n=== LOO Cross-Validation ===');
  const I = {}, G = {}, R = {};
  for (let r = 1; r <= 11; r++) { const rn = `R${r}`;
    if (fs.existsSync(path.join(DD, `inits_${rn}.json`))) I[rn] = JSON.parse(fs.readFileSync(path.join(DD, `inits_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `gt_${rn}.json`))) G[rn] = JSON.parse(fs.readFileSync(path.join(DD, `gt_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `replays_${rn}.json`))) R[rn] = JSON.parse(fs.readFileSync(path.join(DD, `replays_${rn}.json`)));
  }
  const allRounds = Object.keys(I).filter(k => G[k]);
  log(`Rounds for LOO: ${allRounds.join(', ')}`);

  const scores = [];
  for (const testRn of allRounds) {
    const trainRounds = allRounds.filter(r => r !== testRn);

    // Build model excluding test round
    const model = {};
    for (const level of ['d0', 'd1', 'd2', 'd3', 'd4']) {
      const m = {};
      for (const rn of trainRounds) { if (!G[rn] || !I[rn]) continue;
        const rw = roundWeight(rn);
        for (let si = 0; si < SEEDS; si++) { if (!I[rn][si] || !G[rn][si]) continue;
          for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const keys = cf(I[rn][si], y, x); if (!keys) continue; const k = keys[level];
            if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
            const p = G[rn][si][y][x];
            const gtW = 20 * rw;
            for (let c = 0; c < C; c++) m[k].counts[c] += p[c] * gtW; m[k].n += gtW;
          }
        }
      }
      for (const rn of trainRounds) { if (!R[rn] || !I[rn]) continue;
        const rw = roundWeight(rn);
        for (const rep of R[rn]) { const g = I[rn][rep.si]; if (!g) continue;
          for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const keys = cf(g, y, x); if (!keys) continue; const k = keys[level];
            const fc = t2c(rep.finalGrid[y][x]);
            if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
            m[k].n += rw; m[k].counts[fc] += rw;
          }
        }
      }
      for (const k of Object.keys(m)) {
        const tot = Array.from(m[k].counts).reduce((a, b) => a + b, 0) + C * 0.05;
        m[k].a = Array.from(m[k].counts).map(v => (v + 0.05) / tot);
      }
      for (const [k, v] of Object.entries(m)) { if (!model[k]) model[k] = v; }
    }

    // Predict test round
    let roundScore = 0;
    for (let si = 0; si < SEEDS; si++) {
      if (!I[testRn][si] || !G[testRn][si]) continue;
      const pred = predict(I[testRn][si], model);
      // Score: weighted KL
      let wklNum = 0, wklDen = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const gt = G[testRn][si][y][x];
        // Entropy of GT
        let ent = 0;
        for (let c = 0; c < C; c++) { if (gt[c] > 0.001) ent -= gt[c] * Math.log(gt[c]); }
        if (ent < 0.01) continue; // skip static cells
        // KL(gt || pred)
        let kl = 0;
        for (let c = 0; c < C; c++) {
          if (gt[c] > 0.001) kl += gt[c] * Math.log(gt[c] / Math.max(pred[y][x][c], 1e-10));
        }
        wklNum += ent * kl; wklDen += ent;
      }
      const wkl = wklDen > 0 ? wklNum / wklDen : 0;
      const score = Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl)));
      roundScore += score;
    }
    roundScore /= SEEDS;
    scores.push({ round: testRn, score: roundScore });
    log(`  ${testRn}: ${roundScore.toFixed(2)}`);
  }
  const avg = scores.reduce((a, s) => a + s.score, 0) / scores.length;
  log(`  Average LOO score: ${avg.toFixed(2)}`);
  return scores;
}

async function main() {
  log('=== RESUBMIT R12 — Improved Predictions ===');

  // Run LOO cross-validation first
  const looScores = looCrossValidation();

  // Load R12 data
  const inits = JSON.parse(fs.readFileSync(path.join(DD, 'inits_R12.json')));
  const vpObs = JSON.parse(fs.readFileSync(path.join(DD, 'viewport_795bfb1f.json')));
  log(`\nR12: ${inits.length} seeds, ${vpObs.length} VP observations`);

  // Build model
  const model = buildModel();

  // Fuse VP
  fuseVP(model, vpObs, inits);

  // Build per-cell models
  const cellModels = buildCellModels(vpObs, inits);
  for (let si = 0; si < SEEDS; si++) {
    const n = Object.keys(cellModels[si]).length;
    log(`  Seed ${si}: ${n} cells with VP data`);
  }

  // Generate predictions
  log('\nGenerating predictions...');
  const predictions = [];
  for (let si = 0; si < SEEDS; si++) {
    let pred = predict(inits[si], model);
    pred = applyPerCell(pred, cellModels[si], inits[si]);

    // Validate
    let valid = true, minVal = 1, maxSum = 0, minSum = 2;
    for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
      const s = pred[y][x].reduce((a, b) => a + b, 0);
      maxSum = Math.max(maxSum, s); minSum = Math.min(minSum, s);
      for (let c = 0; c < C; c++) minVal = Math.min(minVal, pred[y][x][c]);
      if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false;
    }
    log(`  Seed ${si}: valid=${valid}, minVal=${minVal.toFixed(6)}, sumRange=[${minSum.toFixed(6)}, ${maxSum.toFixed(6)}]`);
    predictions.push(pred);
  }

  // Save predictions
  const outFile = path.join(DD, 'r12_improved_predictions.json');
  fs.writeFileSync(outFile, JSON.stringify(predictions));
  log(`\nPredictions saved to ${outFile}`);

  // Submit if --submit flag
  if (DO_SUBMIT) {
    log('\nSubmitting predictions...');
    for (let si = 0; si < SEEDS; si++) {
      const res = await POST('/submit', { round_id: ROUND_ID, seed_index: si, prediction: predictions[si] });
      log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'} (${res.status}) ${JSON.stringify(res.data).slice(0,100)}`);
      await sleep(600);
    }
    log('Submission complete.');
  } else {
    log('\nDry run — predictions NOT submitted. Use --submit flag to submit.');
  }

  log('\nDone.');
}

main().catch(e => { console.error(e); process.exit(1); });
