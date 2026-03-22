#!/usr/bin/env node
// R8 Fast optimizer — Skip slow grid search, use known-good params, focus on viewport
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
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl / wsum)));
}

async function main() {
  console.log('=== R8 FAST OPTIMIZER ===');
  console.log('Time:', new Date().toISOString());

  const I = {}, G = {}, R = {}, TR = [];
  for (let r = 1; r <= 7; r++) { if (r === 3) continue; const rn = `R${r}`;
    if (fs.existsSync(path.join(DD, `inits_${rn}.json`))) I[rn] = JSON.parse(fs.readFileSync(path.join(DD, `inits_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `gt_${rn}.json`))) G[rn] = JSON.parse(fs.readFileSync(path.join(DD, `gt_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `replays_${rn}.json`))) R[rn] = JSON.parse(fs.readFileSync(path.join(DD, `replays_${rn}.json`)));
    if (I[rn] && G[rn]) TR.push(rn); }
  console.log('Training:', TR.join(', '));
  console.log('Replays:', TR.filter(r => R[r]).map(r => `${r}=${R[r].length}`).join(', '));

  // Quick LOO with 3 configs (using known-good range)
  console.log('\n--- Quick LOO (3 configs) ---');
  const configs = [
    { gtW: 20, temp: 1.05 },
    { gtW: 20, temp: 1.1 },
    { gtW: 20, temp: 1.15 },
  ];
  
  for (const cfg of configs) {
    const scores = [];
    for (const testRn of TR) {
      const trainRounds = TR.filter(r => r !== testRn);
      // Build model
      const model = {};
      for (const level of ['d0', 'd1', 'd2', 'd3', 'd4']) {
        const m = {};
        for (const rn of trainRounds) { if (!G[rn] || !I[rn]) continue;
          for (let si = 0; si < SEEDS; si++) { if (!I[rn][si] || !G[rn][si]) continue;
            for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
              const keys = cf(I[rn][si], y, x); if (!keys) continue; const k = keys[level];
              if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
              const p = G[rn][si][y][x];
              for (let c = 0; c < C; c++) m[k].counts[c] += p[c] * cfg.gtW; m[k].n += cfg.gtW; } } }
        for (const rn of trainRounds) { if (!R[rn] || !I[rn]) continue;
          for (const rep of R[rn]) { const g = I[rn][rep.si]; if (!g) continue;
            for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
              const keys = cf(g, y, x); if (!keys) continue; const k = keys[level];
              const fc = t2c(rep.finalGrid[y][x]);
              if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) }; m[k].n++; m[k].counts[fc]++; } } }
        for (const k of Object.keys(m)) {
          const tot = Array.from(m[k].counts).reduce((a, b) => a + b, 0) + C * 0.05;
          m[k].a = Array.from(m[k].counts).map(v => (v + 0.05) / tot); }
        for (const [k, v] of Object.entries(m)) { if (!model[k]) model[k] = v; }
      }
      // Score
      for (let si = 0; si < SEEDS; si++) {
        if (!I[testRn][si] || !G[testRn][si]) continue;
        const pred = predict(I[testRn][si], model, cfg.temp);
        scores.push(scoreVsGT(pred, G[testRn][si], I[testRn][si]));
      }
    }
    const avg = scores.reduce((a,b) => a+b, 0) / scores.length;
    const min = Math.min(...scores);
    cfg.avg = avg; cfg.min = min;
    console.log(`  gtW=${cfg.gtW} temp=${cfg.temp}: avg=${avg.toFixed(2)}, min=${min.toFixed(2)}`);
  }
  
  const best = configs.sort((a,b) => b.avg - a.avg)[0];
  console.log(`Best: gtW=${best.gtW}, temp=${best.temp}, avg=${best.avg.toFixed(2)}`);
  
  // Build full model with best config
  console.log('\n--- Building full model ---');
  const model = {};
  for (const level of ['d0', 'd1', 'd2', 'd3', 'd4']) {
    const m = {};
    for (const rn of TR) { if (!G[rn] || !I[rn]) continue;
      for (let si = 0; si < SEEDS; si++) { if (!I[rn][si] || !G[rn][si]) continue;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = cf(I[rn][si], y, x); if (!keys) continue; const k = keys[level];
          if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
          const p = G[rn][si][y][x];
          for (let c = 0; c < C; c++) m[k].counts[c] += p[c] * best.gtW; m[k].n += best.gtW; } } }
    for (const rn of TR) { if (!R[rn] || !I[rn]) continue;
      for (const rep of R[rn]) { const g = I[rn][rep.si]; if (!g) continue;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = cf(g, y, x); if (!keys) continue; const k = keys[level];
          const fc = t2c(rep.finalGrid[y][x]);
          if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) }; m[k].n++; m[k].counts[fc]++; } } }
    for (const k of Object.keys(m)) {
      const tot = Array.from(m[k].counts).reduce((a, b) => a + b, 0) + C * 0.05;
      m[k].a = Array.from(m[k].counts).map(v => (v + 0.05) / tot); }
    for (const [k, v] of Object.entries(m)) { if (!model[k]) model[k] = v; }
  }
  console.log('Cross-round model:', Object.keys(model).length, 'keys');

  // Load R8 data
  const vpObs = JSON.parse(fs.readFileSync(path.join(DD, 'viewport_c5cdf100.json')));
  const { data: rd } = await GET('/rounds/' + R8);
  const inits = rd.initial_states.map(is => is.grid);
  
  // Build viewport D0 model
  const vpModel = {};
  for (const obs of vpObs) { const si = obs.si !== undefined ? obs.si : 0;
    for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
      const gy = obs.vy + dy, gx = obs.vx + dx;
      if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
      const keys = cf(inits[si], gy, gx); if (!keys) continue;
      const k = keys.d0, fc = t2c(obs.grid[dy][dx]);
      if (!vpModel[k]) vpModel[k] = { n: 0, counts: new Float64Array(C) }; vpModel[k].n++; vpModel[k].counts[fc]++; } }
  
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
  }

  // Fuse viewport into model
  function fuseVP(baseModel, cw) {
    const fused = {};
    for (const [k, v] of Object.entries(baseModel)) fused[k] = { n: v.n, a: [...v.a] };
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

  // Submit multiple configs, last one wins
  // Strategy: submit 3 variants from conservative to aggressive
  const submitConfigs = [
    { name: 'A_safe', cw: 20, adaptivePW: { 1: 15, 2: 10, 3: 5, 5: 3 }, thresh: 1, temp: best.temp },
    { name: 'B_balanced', cw: 20, adaptivePW: { 1: 12, 2: 8, 3: 5, 5: 3 }, thresh: 1, temp: best.temp },
    { name: 'C_aggressive', cw: 15, adaptivePW: { 1: 8, 2: 5, 3: 3, 5: 2 }, thresh: 1, temp: best.temp },
  ];
  
  for (const scfg of submitConfigs) {
    console.log(`\n--- Submitting ${scfg.name}: cw=${scfg.cw} thresh=${scfg.thresh} ---`);
    const fusedModel = fuseVP(model, scfg.cw);
    
    for (let si = 0; si < SEEDS; si++) {
      let pred = predict(inits[si], fusedModel, scfg.temp);
      
      let applied = 0;
      for (const [key, cell] of Object.entries(cellModels[si])) {
        if (cell.n < scfg.thresh) continue;
        const [y, x] = key.split(',').map(Number);
        if (inits[si][y][x] === 10 || inits[si][y][x] === 5) continue;
        
        let pw;
        if (cell.n >= 5) pw = scfg.adaptivePW[5];
        else if (cell.n >= 3) pw = scfg.adaptivePW[3];
        else if (cell.n >= 2) pw = scfg.adaptivePW[2];
        else pw = scfg.adaptivePW[1];
        
        const prior = pred[y][x], posterior = new Array(C); let total = 0;
        for (let c = 0; c < C; c++) { posterior[c] = prior[c] * pw + cell.counts[c]; total += posterior[c]; }
        if (total > 0) { for (let c = 0; c < C; c++) posterior[c] /= total; pred[y][x] = posterior; }
        applied++;
      }
      
      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const s = pred[y][x].reduce((a, b) => a + b, 0);
        if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false; }
      if (!valid) { console.log(`  Seed ${si}: VALIDATION FAILED`); continue; }
      
      const res = await POST('/submit', { round_id: R8, seed_index: si, prediction: pred });
      console.log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'} (${applied} cells) ${JSON.stringify(res.data).slice(0, 60)}`);
      await sleep(600);
    }
  }
  
  console.log('\n=== DONE ===');
  console.log(`Last submitted: C_aggressive (most viewport influence)`);
  console.log(`LOO: ${best.avg.toFixed(2)} (cross-round only)`);
  console.log('Viewport + per-cell should add 2-5 points on top');
}
main().catch(e => console.error('Error:', e.message, e.stack));
