#!/usr/bin/env node
// R8 Comprehensive Optimizer — LOO + Viewport + Per-cell optimization
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

function buildModel(gts, reps, inits, rounds, alpha, gtW) {
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
      const tot = Array.from(m[k].counts).reduce((a, b) => a + b, 0) + C * alpha;
      m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / tot); }
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
    const g = gt[y][x];
    let ent = 0;
    for (let c = 0; c < C; c++) { if (g[c] > 0) ent -= g[c] * Math.log(g[c]); }
    if (ent < 0.01) continue;
    let kl = 0;
    for (let c = 0; c < C; c++) {
      if (g[c] > 0) kl += g[c] * Math.log(g[c] / Math.max(pred[y][x][c], 1e-15));
    }
    wkl += ent * kl; wsum += ent;
  }
  if (wsum === 0) return 100;
  const wklAvg = wkl / wsum;
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wklAvg)));
}

async function main() {
  console.log('=== R8 COMPREHENSIVE OPTIMIZER ===');
  console.log('Time:', new Date().toISOString());

  // Load data
  const I = {}, G = {}, R = {}, TR = [];
  for (let r = 1; r <= 7; r++) { if (r === 3) continue; const rn = `R${r}`;
    if (fs.existsSync(path.join(DD, `inits_${rn}.json`))) I[rn] = JSON.parse(fs.readFileSync(path.join(DD, `inits_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `gt_${rn}.json`))) G[rn] = JSON.parse(fs.readFileSync(path.join(DD, `gt_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `replays_${rn}.json`))) R[rn] = JSON.parse(fs.readFileSync(path.join(DD, `replays_${rn}.json`)));
    if (I[rn] && G[rn]) TR.push(rn); }
  console.log('Training:', TR.join(', '));
  console.log('Replays:', TR.filter(r => R[r]).map(r => `${r}=${R[r].length}`).join(', '));

  // === PHASE 1: LOO Grid Search ===
  console.log('\n--- PHASE 1: LOO Grid Search ---');
  const GTWs = [15, 20, 25, 30];
  const TEMPs = [0.95, 1.0, 1.05, 1.1, 1.15, 1.2];
  const results = [];
  
  for (const gtW of GTWs) {
    for (const temp of TEMPs) {
      const scores = [];
      for (const testRn of TR) {
        const trainRounds = TR.filter(r => r !== testRn);
        const model = buildModel(G, R, I, trainRounds, 0.05, gtW);
        for (let si = 0; si < SEEDS; si++) {
          if (!I[testRn][si] || !G[testRn][si]) continue;
          const pred = predict(I[testRn][si], model, temp);
          scores.push(scoreVsGT(pred, G[testRn][si], I[testRn][si]));
        }
      }
      const avg = scores.reduce((a,b) => a+b, 0) / scores.length;
      const min = Math.min(...scores);
      results.push({ gtW, temp, avg, min, n: scores.length });
    }
  }
  
  results.sort((a, b) => b.avg - a.avg);
  console.log('\nTop 10 configs (by avg):');
  for (let i = 0; i < 10; i++) {
    const r = results[i];
    console.log(`  gtW=${r.gtW} temp=${r.temp.toFixed(2)}: avg=${r.avg.toFixed(2)}, min=${r.min.toFixed(2)}`);
  }
  
  const best = results[0];
  console.log(`\nBest: gtW=${best.gtW}, temp=${best.temp}, avg=${best.avg.toFixed(2)}`);
  
  // === PHASE 2: Build best model and load viewport ===
  console.log('\n--- PHASE 2: Build R8 Model ---');
  const model = buildModel(G, R, I, TR, 0.05, best.gtW);
  console.log('Cross-round model:', Object.keys(model).length, 'keys');
  
  const vpObs = JSON.parse(fs.readFileSync(path.join(DD, 'viewport_c5cdf100.json')));
  const { data: rd } = await GET('/rounds/' + R8);
  const inits = rd.initial_states.map(is => is.grid);
  console.log('Viewport observations:', vpObs.length);
  
  // === PHASE 3: Viewport D0 fusion with multiple cw values ===
  console.log('\n--- PHASE 3: Viewport Fusion Optimization ---');
  
  // Build viewport model
  const vpModel = {};
  for (const obs of vpObs) { const si = obs.si !== undefined ? obs.si : 0;
    for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
      const gy = obs.vy + dy, gx = obs.vx + dx;
      if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
      const keys = cf(inits[si], gy, gx); if (!keys) continue;
      const k = keys.d0, fc = t2c(obs.grid[dy][dx]);
      if (!vpModel[k]) vpModel[k] = { n: 0, counts: new Float64Array(C) }; vpModel[k].n++; vpModel[k].counts[fc]++; } }
  console.log('VP D0 keys:', Object.keys(vpModel).length);
  
  // Build per-cell models
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
    const n3 = Object.values(cells).filter(c => c.n >= 3).length;
    const n2 = Object.values(cells).filter(c => c.n >= 2).length;
    const nAll = Object.keys(cells).length;
    console.log(`Seed ${si}: ${nAll} cells total, ${n2} >=2obs, ${n3} >=3obs`);
  }
  
  // Test combinations: cw × pw × threshold
  const CWs = [10, 15, 20, 30, 50];
  const PWs = [3, 5, 8, 10];
  const THRESHs = [1, 2, 3]; // Min obs threshold for per-cell
  
  // We can't LOO the viewport part, but we can test the cross-round part
  // For the submission, we'll use the best LOO config + try all viewport combos
  // Since we can re-submit, we'll try a few and see what scores best
  
  console.log('\n--- PHASE 4: Submit Best Configs ---');
  
  // Function to create fused model with given cw
  function fuseModel(baseModel, vpModel, cw) {
    const fused = {};
    for (const [k, v] of Object.entries(baseModel)) {
      fused[k] = { n: v.n, a: [...v.a] };
    }
    for (const [k, vm] of Object.entries(vpModel)) {
      const bm = fused[k];
      if (bm) {
        const pa = bm.a.map(p => p * cw), post = pa.map((a, c) => a + vm.counts[c]);
        const tot = post.reduce((a, b) => a + b, 0);
        fused[k] = { n: bm.n + vm.n, a: post.map(v => v / tot) };
      } else {
        const parts = k.split('_'), t = parts[1], sa = parseInt(parts[2]), co = parts[3], sb2 = parts[4];
        const d1k = `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`, cm = fused[d1k];
        if (cm) {
          const pa = cm.a.map(p => p * cw), post = pa.map((a, c) => a + vm.counts[c]);
          const tot = post.reduce((a, b) => a + b, 0);
          fused[k] = { n: vm.n + cw, a: post.map(v => v / tot) };
        } else {
          const tot = vm.n + C * 0.1;
          fused[k] = { n: vm.n, a: Array.from(vm.counts).map(v => (v + 0.1) / tot) };
        }
      }
    }
    return fused;
  }
  
  // Submit the BEST config: best LOO params + moderate viewport/per-cell
  // Strategy: submit 3 configs and the last one wins
  // Config A: Conservative (high cw, high pw, >=3 threshold) — safe
  // Config B: Aggressive (lower cw, lower pw, >=1 threshold) — more viewport influence  
  // Config C: Balanced (medium everything)
  
  const configs = [
    { name: 'Aggressive', cw: 15, pw: 3, thresh: 1, temp: best.temp },
    { name: 'Balanced', cw: 20, pw: 5, thresh: 2, temp: best.temp },
    { name: 'Conservative', cw: 30, pw: 8, thresh: 3, temp: best.temp },
    // Also try with slightly different temps
    { name: 'Agg+LowTemp', cw: 15, pw: 3, thresh: 1, temp: Math.max(0.9, best.temp - 0.1) },
    { name: 'Bal+HighTemp', cw: 20, pw: 5, thresh: 2, temp: best.temp + 0.05 },
  ];
  
  // Actually, let's just find the best combo and submit that
  // We can't validate viewport config via LOO, so let's think about what's most robust
  // Best approach: Submit the one that uses viewport data most effectively
  // The viewport has 10 obs per seed of 15x15 = 225 cells, with overlap
  // For 1-obs cells, the observation is informative but noisy
  // For 2+ obs cells, much more reliable
  
  // Let's try submitting with balanced config first, then aggressive
  // The last submission wins
  
  const SUBMIT_CONFIG = { cw: 20, pw: 5, thresh: 1, temp: best.temp };
  // Use thresh=1 (all cells) but with adaptive pw:
  // 1 obs → pw=10 (barely shifts), 2 obs → pw=7, 3+ obs → pw=5
  
  console.log(`\nSubmitting: gtW=${best.gtW} temp=${SUBMIT_CONFIG.temp.toFixed(2)} cw=${SUBMIT_CONFIG.cw} adaptive-pw thresh=${SUBMIT_CONFIG.thresh}`);
  
  const fusedModel = fuseModel(model, vpModel, SUBMIT_CONFIG.cw);
  console.log('Fused model:', Object.keys(fusedModel).length, 'keys');
  
  for (let si = 0; si < SEEDS; si++) {
    let pred = predict(inits[si], fusedModel, SUBMIT_CONFIG.temp);
    
    // Apply per-cell with adaptive weighting
    let applied = 0;
    for (const [key, cell] of Object.entries(cellModels[si])) {
      if (cell.n < SUBMIT_CONFIG.thresh) continue;
      const [y, x] = key.split(',').map(Number);
      if (inits[si][y][x] === 10 || inits[si][y][x] === 5) continue;
      
      // Adaptive pw: fewer obs = higher prior weight (more conservative)
      let pw;
      if (cell.n >= 5) pw = 3;
      else if (cell.n >= 3) pw = 5;
      else if (cell.n >= 2) pw = 8;
      else pw = 12; // 1 obs: very conservative, barely shifts
      
      const prior = pred[y][x], posterior = new Array(C); let total = 0;
      for (let c = 0; c < C; c++) { posterior[c] = prior[c] * pw + cell.counts[c]; total += posterior[c]; }
      if (total > 0) { for (let c = 0; c < C; c++) posterior[c] /= total; pred[y][x] = posterior; }
      applied++;
    }
    
    // Validate
    let valid = true;
    for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
      const s = pred[y][x].reduce((a, b) => a + b, 0);
      if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false; }
    if (!valid) { console.log(`Seed ${si}: VALIDATION FAILED`); continue; }
    
    const res = await POST('/submit', { round_id: R8, seed_index: si, prediction: pred });
    console.log(`Seed ${si}: ${res.ok ? 'ACCEPTED' : 'FAILED'} (${applied} cells corrected) ${JSON.stringify(res.data).slice(0, 80)}`);
    await sleep(600);
  }
  
  console.log('\n=== SUBMITTED ===');
  console.log(`Config: gtW=${best.gtW} temp=${best.temp.toFixed(2)} cw=${SUBMIT_CONFIG.cw} adaptive-pw`);
  console.log('Per-cell: 1obs→pw=12, 2obs→pw=8, 3obs→pw=5, 5+obs→pw=3');
}

main().catch(e => console.error('Error:', e.message, e.stack));
