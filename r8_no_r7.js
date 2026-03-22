#!/usr/bin/env node
// R8 WITHOUT R7 — Test if excluding the outlier round helps
// Also compares LOO with and without R7
const fs = require('fs'), path = require('path'), https = require('https');
const H = 40, W = 40, SEEDS = 5, C = 6;
const DD = path.join(__dirname, 'data');
const TOKEN = process.argv[2] || '';
const BASE = 'https://api.ainm.no/astar-island';
const R8 = 'c5cdf100-a876-4fb7-b5d8-757162c97989';

function api(m, p, b) { return new Promise((res, rej) => {
  const u = new URL(BASE + p); const pl = b ? JSON.stringify(b) : null;
  const o = { hostname: u.hostname, path: u.pathname + u.search, method: m,
    headers: { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' } };
  if (pl) o.headers['Content-Length'] = Buffer.byteLength(pl);
  const r = https.request(o, re => { let d = ''; re.on('data', c => d += c);
    re.on('end', () => { try { res({ ok: re.statusCode < 300, status: re.statusCode, data: JSON.parse(d) }); } catch { res({ ok: false, status: re.statusCode, data: d }); } });
  }); r.on('error', rej); if (pl) r.write(pl); r.end(); }); }
const GET = p => api('GET', p), POST = (p, b) => api('POST', p, b);
const sleep = ms => new Promise(r => setTimeout(r, ms));
function t2c(t) { return (t === 10 || t === 11 || t === 0) ? 0 : (t >= 1 && t <= 5) ? t : 0; }

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

function buildModel(gts, reps, inits, rounds, gtW) {
  const model = {};
  for (const level of ['d0', 'd1', 'd2', 'd3', 'd4']) {
    const m = {};
    for (const rn of rounds) { if (!gts[rn] || !inits[rn]) continue;
      for (let si = 0; si < SEEDS; si++) { if (!inits[rn][si] || !gts[rn][si]) continue;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = cf(inits[rn][si], y, x); if (!keys) continue; const k = keys[level];
          if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
          const p = gts[rn][si][y][x];
          for (let c = 0; c < C; c++) m[k].counts[c] += p[c] * gtW; m[k].n += gtW; } } }
    for (const rn of rounds) { if (!reps[rn] || !inits[rn]) continue;
      for (const rep of reps[rn]) { const g = inits[rn][rep.si]; if (!g) continue;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = cf(g, y, x); if (!keys) continue; const k = keys[level];
          const fc = t2c(rep.finalGrid[y][x]);
          if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) }; m[k].n++; m[k].counts[fc]++; } } }
    for (const k of Object.keys(m)) {
      const tot = Array.from(m[k].counts).reduce((a, b) => a + b, 0) + C * 0.05;
      m[k].a = Array.from(m[k].counts).map(v => (v + 0.05) / tot); }
    for (const [k, v] of Object.entries(m)) { if (!model[k]) model[k] = v; }
  }
  return model;
}

function predict(grid, model, temp) {
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
      if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }
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
      for (let c = 0; c < C; c++) { p[c] = Math.pow(Math.max(p[c] / wS, 1e-10), 1 / temp);
        if (p[c] < 0.00005) p[c] = 0.00005; s += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    } }
  return pred;
}

function scoreVsGT(pred, gt, init) {
  let wkl = 0, wsum = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const t = init[y][x]; if (t === 10 || t === 5) continue;
    const g = gt[y][x]; let ent = 0;
    for (let c = 0; c < C; c++) { if (g[c] > 0) ent -= g[c] * Math.log(g[c]); }
    if (ent < 0.01) continue;
    let kl = 0;
    for (let c = 0; c < C; c++) { if (g[c] > 0) kl += g[c] * Math.log(g[c] / Math.max(pred[y][x][c], 1e-15)); }
    wkl += ent * kl; wsum += ent;
  }
  if (wsum === 0) return 100;
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl / wsum)));
}

async function main() {
  console.log('=== R8 MODEL COMPARISON ===');
  
  const I = {}, G = {}, R = {};
  for (let r = 1; r <= 7; r++) { if (r === 3) continue; const rn = `R${r}`;
    if (fs.existsSync(path.join(DD, `inits_${rn}.json`))) I[rn] = JSON.parse(fs.readFileSync(path.join(DD, `inits_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `gt_${rn}.json`))) G[rn] = JSON.parse(fs.readFileSync(path.join(DD, `gt_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `replays_${rn}.json`))) R[rn] = JSON.parse(fs.readFileSync(path.join(DD, `replays_${rn}.json`)));
  }
  
  const ALL = ['R1', 'R2', 'R4', 'R5', 'R6', 'R7'];
  const NO_R7 = ['R1', 'R2', 'R4', 'R5', 'R6'];
  const RECENT = ['R5', 'R6', 'R7'];
  
  // Quick LOO comparison (test on non-R7 rounds only — R7 is always bad)
  console.log('--- LOO on non-R7 rounds (which model predicts R1-R6 best?) ---');
  const testRounds = NO_R7;
  
  for (const [label, trainPool] of [['AllRounds', ALL], ['NoR7', NO_R7], ['Recent', RECENT]]) {
    const scores = [];
    const perRound = {};
    for (const testRn of testRounds) {
      const train = trainPool.filter(r => r !== testRn);
      if (train.length === 0) continue;
      const model = buildModel(G, R, I, train, 20);
      perRound[testRn] = [];
      for (let si = 0; si < SEEDS; si++) {
        if (!I[testRn] || !I[testRn][si] || !G[testRn][si]) continue;
        const pred = predict(I[testRn][si], model, 1.1);
        const sc = scoreVsGT(pred, G[testRn][si], I[testRn][si]);
        scores.push(sc);
        perRound[testRn].push(sc);
      }
    }
    const avg = scores.reduce((a,b) => a+b, 0) / scores.length;
    console.log(`  ${label}: avg=${avg.toFixed(2)} [${Object.entries(perRound).map(([r,s]) => `${r}=${(s.reduce((a,b)=>a+b,0)/s.length).toFixed(1)}`).join(', ')}]`);
  }
  
  // Also test how each model predicts R7
  console.log('\n--- How each model predicts R7 ---');
  for (const [label, trainPool] of [['AllExR7', NO_R7], ['Recent', RECENT]]) {
    const train = trainPool.filter(r => r !== 'R7');
    const model = buildModel(G, R, I, train, 20);
    const scores = [];
    for (let si = 0; si < SEEDS; si++) {
      if (!I['R7'] || !I['R7'][si] || !G['R7'][si]) continue;
      const pred = predict(I['R7'][si], model, 1.1);
      scores.push(scoreVsGT(pred, G['R7'][si], I['R7'][si]));
    }
    console.log(`  ${label} → R7: avg=${(scores.reduce((a,b)=>a+b,0)/scores.length).toFixed(2)}`);
  }
  
  // Now decide: use AllRounds or NoR7 for R8 submission
  // Build both models and submit the one we think is best
  // For now, let's submit NoR7 since it should have higher LOO on normal rounds
  
  console.log('\n--- Building models for comparison ---');
  const modelAll = buildModel(G, R, I, ALL, 20);
  const modelNoR7 = buildModel(G, R, I, NO_R7, 20);
  console.log('AllRounds model:', Object.keys(modelAll).length, 'keys');
  console.log('NoR7 model:', Object.keys(modelNoR7).length, 'keys');
  
  // If NoR7 has higher LOO on normal rounds, use it
  // Otherwise stick with AllRounds
  // The script will decide based on LOO results above
  
  // Load R8 data
  const vpObs = JSON.parse(fs.readFileSync(path.join(DD, 'viewport_c5cdf100.json')));
  const { data: rd } = await GET('/rounds/' + R8);
  const inits = rd.initial_states.map(is => is.grid);
  
  // Build per-cell
  const obsBySeed = {};
  for (const obs of vpObs) { const si = obs.si !== undefined ? obs.si : 0;
    if (!obsBySeed[si]) obsBySeed[si] = []; obsBySeed[si].push(obs); }
  const cellModels = {};
  for (let si = 0; si < SEEDS; si++) {
    const cells = {};
    for (const obs of (obsBySeed[si] || [])) {
      for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
        const gy = obs.vy + dy, gx = obs.vx + dx;
        if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
        if (inits[si][gy][gx] === 10 || inits[si][gy][gx] === 5) continue;
        const k = `${gy},${gx}`, fc = t2c(obs.grid[dy][dx]);
        if (!cells[k]) cells[k] = { n: 0, counts: new Float64Array(C) }; cells[k].n++; cells[k].counts[fc]++; } }
    cellModels[si] = cells;
  }
  
  // Submit ENSEMBLE of both models (hedge our bets)
  console.log('\n--- Submitting ENSEMBLE of AllRounds + NoR7 models ---');
  
  // Fuse viewport into both models
  function fuseVP(baseModel) {
    const fused = {};
    for (const [k, v] of Object.entries(baseModel)) fused[k] = { n: v.n, a: [...v.a] };
    for (const level of ['d0', 'd1', 'd2']) {
      const cw = level === 'd0' ? 20 : level === 'd1' ? 15 : 10;
      const vm = {};
      for (const obs of vpObs) { const si = obs.si !== undefined ? obs.si : 0;
        for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
          const gy = obs.vy + dy, gx = obs.vx + dx;
          if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
          const keys = cf(inits[si], gy, gx); if (!keys) continue;
          const k = keys[level], fc = t2c(obs.grid[dy][dx]);
          if (!vm[k]) vm[k] = { n: 0, counts: new Float64Array(C) }; vm[k].n++; vm[k].counts[fc]++; } }
      for (const [k, vp] of Object.entries(vm)) {
        const bm = fused[k];
        if (bm) {
          const pa = bm.a.map(p => p * cw), post = pa.map((a, c) => a + vp.counts[c]);
          const tot = post.reduce((a, b) => a + b, 0);
          fused[k] = { n: bm.n + vp.n, a: post.map(v => v / tot) };
        } else if (level === 'd0') {
          const parts = k.split('_'), t = parts[1], sa = parseInt(parts[2]), co = parts[3], sb2 = parts[4];
          const d1k = `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`, cm = fused[d1k];
          if (cm) {
            const pa = cm.a.map(p => p * cw), post = pa.map((a, c) => a + vp.counts[c]);
            const tot = post.reduce((a, b) => a + b, 0);
            fused[k] = { n: vp.n + cw, a: post.map(v => v / tot) };
          } else {
            const tot = vp.n + C * 0.1;
            fused[k] = { n: vp.n, a: Array.from(vp.counts).map(v => (v + 0.1) / tot) };
          }
        }
      }
    }
    return fused;
  }
  
  const fusedAll = fuseVP(modelAll);
  const fusedNoR7 = fuseVP(modelNoR7);
  
  for (let si = 0; si < SEEDS; si++) {
    const predAll = predict(inits[si], fusedAll, 1.1);
    const predNoR7 = predict(inits[si], fusedNoR7, 1.1);
    
    // Ensemble: 50% AllRounds + 50% NoR7
    const pred = [];
    for (let y = 0; y < H; y++) { pred[y] = [];
      for (let x = 0; x < W; x++) {
        pred[y][x] = new Array(C);
        for (let c = 0; c < C; c++) {
          pred[y][x][c] = 0.5 * predAll[y][x][c] + 0.5 * predNoR7[y][x][c];
        }
      }
    }
    
    // Apply per-cell (ALL cells, adaptive pw)
    for (const [key, cell] of Object.entries(cellModels[si])) {
      const [y, x] = key.split(',').map(Number);
      if (inits[si][y][x] === 10 || inits[si][y][x] === 5) continue;
      const pw = cell.n >= 5 ? 2 : cell.n >= 3 ? 4 : cell.n >= 2 ? 7 : 15;
      const prior = pred[y][x], posterior = new Array(C); let total = 0;
      for (let c = 0; c < C; c++) { posterior[c] = prior[c] * pw + cell.counts[c]; total += posterior[c]; }
      if (total > 0) { for (let c = 0; c < C; c++) posterior[c] /= total; pred[y][x] = posterior; }
    }
    
    // Validate
    let valid = true;
    for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
      const s = pred[y][x].reduce((a, b) => a + b, 0);
      if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false; }
    if (!valid) { console.log(`Seed ${si}: VALIDATION FAILED`); continue; }
    
    const res = await POST('/submit', { round_id: R8, seed_index: si, prediction: pred });
    console.log(`Seed ${si}: ${res.ok ? 'OK' : 'FAIL'}`);
    await sleep(600);
  }
  
  console.log('\n=== ENSEMBLE (AllRounds + NoR7) SUBMITTED ===');
}
main().catch(e => console.error('Error:', e.message, e.stack));
