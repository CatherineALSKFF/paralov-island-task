#!/usr/bin/env node
// SMART VP: Model-informed spread + extreme confidence variants
// Instead of uniform spread, distribute "uncertainty" per the model's prediction
const fs = require('fs'), path = require('path'), https = require('https');
const H = 40, W = 40, SEEDS = 5, C = 6;
const DD = path.join(__dirname, 'data');
const TOKEN = process.argv[2] || '';
const ROUND_ID = '2a341ace-0f57-4309-9b89-e59fe0f09179';
const BASE = 'https://api.ainm.no/astar-island';

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

async function main() {
  console.log('=== SMART VP SUBMISSION ===');
  console.log('Time:', new Date().toISOString());

  // Load VP
  const vpFile = path.join(DD, `viewport_${ROUND_ID.slice(0,8)}.json`);
  const vpObs = JSON.parse(fs.readFileSync(vpFile, 'utf8'));
  const vpCells = {};
  for (let si = 0; si < SEEDS; si++) vpCells[si] = {};
  for (const obs of vpObs) {
    const si = obs.si;
    for (let dy = 0; dy < obs.grid.length; dy++)
      for (let dx = 0; dx < obs.grid[0].length; dx++) {
        const gy = obs.vy + dy, gx = obs.vx + dx;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W)
          vpCells[si][gy + ',' + gx] = t2c(obs.grid[dy][dx]);
      }
  }

  // Load round and build model
  const { data: rd } = await GET('/rounds/' + ROUND_ID);
  const inits = rd.initial_states.map(is => is.grid);
  const I = {}, G = {}, R = {};
  for (let r = 1; r <= 20; r++) { const rn = `R${r}`;
    if (fs.existsSync(path.join(DD, `inits_${rn}.json`))) I[rn] = JSON.parse(fs.readFileSync(path.join(DD, `inits_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `gt_${rn}.json`))) G[rn] = JSON.parse(fs.readFileSync(path.join(DD, `gt_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `replays_${rn}.json`))) R[rn] = JSON.parse(fs.readFileSync(path.join(DD, `replays_${rn}.json`)));
  }
  const TR = Object.keys(I).filter(k => G[k]);
  const model = {};
  for (const level of ['d0', 'd1', 'd2', 'd3', 'd4']) {
    const m = {};
    for (const rn of TR) { if (!G[rn] || !I[rn]) continue;
      for (let si = 0; si < SEEDS; si++) { if (!I[rn][si] || !G[rn][si]) continue;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = cf(I[rn][si], y, x); if (!keys) continue; const k = keys[level];
          if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
          const p = G[rn][si][y][x];
          for (let c = 0; c < C; c++) m[k].counts[c] += p[c] * 20; m[k].n += 20; } } }
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

  let submitCount = 0;

  // STRATEGY 1: Model-informed spread
  // VP class gets vpConf%, remaining (1-vpConf%) distributed per model prediction
  const configs = [
    { name: 'smart_vp995', vpConf: 0.995 },
    { name: 'smart_vp99', vpConf: 0.99 },
    { name: 'smart_vp98', vpConf: 0.98 },
    { name: 'smart_vp95', vpConf: 0.95 },
    { name: 'smart_vp90', vpConf: 0.90 },
    { name: 'smart_vp85', vpConf: 0.85 },
  ];

  for (const cfg of configs) {
    console.log(`\n--- ${cfg.name}: vpConf=${cfg.vpConf} + model-informed spread ---`);
    for (let si = 0; si < SEEDS; si++) {
      const modelPred = predict(inits[si], model, 1.15);
      const pred = [];
      for (let y = 0; y < H; y++) { pred[y] = [];
        for (let x = 0; x < W; x++) {
          const t = inits[si][y][x];
          if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
          if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }

          const vpKey = y + ',' + x;
          if (vpCells[si][vpKey] !== undefined) {
            const vpClass = vpCells[si][vpKey];
            const p = new Array(C);
            // Distribute (1-vpConf) proportional to model prediction
            let otherSum = 0;
            for (let c = 0; c < C; c++) if (c !== vpClass) otherSum += modelPred[y][x][c];
            for (let c = 0; c < C; c++) {
              if (c === vpClass) {
                p[c] = cfg.vpConf;
              } else {
                p[c] = otherSum > 0 ? (1 - cfg.vpConf) * modelPred[y][x][c] / otherSum : (1 - cfg.vpConf) / (C - 1);
              }
              p[c] = Math.max(p[c], 0.00005);
            }
            const sum = p.reduce((a, b) => a + b, 0);
            for (let c = 0; c < C; c++) p[c] /= sum;
            pred[y][x] = p;
          } else {
            pred[y][x] = modelPred[y][x];
          }
        } }

      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const s = pred[y][x].reduce((a, b) => a + b, 0);
        if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false; }
      if (!valid) { console.log(`  Seed ${si}: VALIDATION FAILED`); continue; }

      const res = await POST('/submit', { round_id: ROUND_ID, seed_index: si, prediction: pred });
      if (si === 0 || si === 4) console.log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'} ${JSON.stringify(res.data).slice(0, 60)}`);
      submitCount++;
      await sleep(550);
    }
  }

  // STRATEGY 2: Spatial-smoothed VP
  // For cells where VP terrain matches most neighbors' VP terrain → higher confidence
  // For cells at terrain boundaries → lower confidence
  console.log('\n=== SPATIAL-SMOOTHED VP ===');
  const spatialConfigs = [
    { name: 'spatial_high', baseConf: 0.97, borderConf: 0.85, innerConf: 0.995 },
    { name: 'spatial_med', baseConf: 0.92, borderConf: 0.75, innerConf: 0.98 },
  ];

  for (const cfg of spatialConfigs) {
    console.log(`\n--- ${cfg.name} ---`);
    for (let si = 0; si < SEEDS; si++) {
      const modelPred = predict(inits[si], model, 1.15);
      const pred = [];
      let inner = 0, border = 0;
      for (let y = 0; y < H; y++) { pred[y] = [];
        for (let x = 0; x < W; x++) {
          const t = inits[si][y][x];
          if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
          if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }

          const vpKey = y + ',' + x;
          if (vpCells[si][vpKey] !== undefined) {
            const vpClass = vpCells[si][vpKey];
            // Check neighbors in VP
            let sameNeighbors = 0, totalNeighbors = 0;
            for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
              if (dy === 0 && dx === 0) continue;
              const nk = (y+dy) + ',' + (x+dx);
              if (vpCells[si][nk] !== undefined) {
                totalNeighbors++;
                if (vpCells[si][nk] === vpClass) sameNeighbors++;
              }
            }
            // If all neighbors match → interior cell → very high confidence
            // If some neighbors differ → boundary → lower confidence
            let conf;
            if (totalNeighbors > 0 && sameNeighbors === totalNeighbors) {
              conf = cfg.innerConf; inner++;
            } else if (totalNeighbors > 0 && sameNeighbors / totalNeighbors < 0.5) {
              conf = cfg.borderConf; border++;
            } else {
              conf = cfg.baseConf;
            }

            const p = new Array(C);
            let otherSum = 0;
            for (let c = 0; c < C; c++) if (c !== vpClass) otherSum += modelPred[y][x][c];
            for (let c = 0; c < C; c++) {
              if (c === vpClass) p[c] = conf;
              else p[c] = otherSum > 0 ? (1 - conf) * modelPred[y][x][c] / otherSum : (1 - conf) / (C - 1);
              p[c] = Math.max(p[c], 0.00005);
            }
            const sum = p.reduce((a, b) => a + b, 0);
            for (let c = 0; c < C; c++) p[c] /= sum;
            pred[y][x] = p;
          } else {
            pred[y][x] = modelPred[y][x];
          }
        } }

      if (si === 0) console.log(`  inner=${inner}, border=${border}`);

      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const s = pred[y][x].reduce((a, b) => a + b, 0);
        if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false; }
      if (!valid) { console.log(`  Seed ${si}: VALIDATION FAILED`); continue; }

      const res = await POST('/submit', { round_id: ROUND_ID, seed_index: si, prediction: pred });
      if (si === 0 || si === 4) console.log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'}`);
      submitCount++;
      await sleep(550);
    }
  }

  // STRATEGY 3: VP with terrain-type-specific confidence
  // Ocean/mountain: near-certain (always static)
  // Plains: high confidence (usually stable)
  // Settlement: moderate-high (could be port/ruin in rare cases)
  // Forest: moderate-high (could be cut for settlement)
  // Port/Ruin: moderate (less common, possibly transient)
  console.log('\n=== TERRAIN-SPECIFIC VP CONFIDENCE ===');
  const terrainConfigs = [
    { name: 'terrain_high', cls0: 0.998, cls1: 0.97, cls2: 0.92, cls3: 0.90, cls4: 0.97, cls5: 0.998 },
    { name: 'terrain_med', cls0: 0.995, cls1: 0.93, cls2: 0.85, cls3: 0.82, cls4: 0.93, cls5: 0.995 },
    { name: 'terrain_vhigh', cls0: 0.999, cls1: 0.99, cls2: 0.97, cls3: 0.95, cls4: 0.99, cls5: 0.999 },
  ];

  for (const cfg of terrainConfigs) {
    console.log(`\n--- ${cfg.name} ---`);
    const confMap = [cfg.cls0, cfg.cls1, cfg.cls2, cfg.cls3, cfg.cls4, cfg.cls5];
    for (let si = 0; si < SEEDS; si++) {
      const modelPred = predict(inits[si], model, 1.15);
      const pred = [];
      for (let y = 0; y < H; y++) { pred[y] = [];
        for (let x = 0; x < W; x++) {
          const t = inits[si][y][x];
          if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
          if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }

          const vpKey = y + ',' + x;
          if (vpCells[si][vpKey] !== undefined) {
            const vpClass = vpCells[si][vpKey];
            const conf = confMap[vpClass];
            const p = new Array(C);
            let otherSum = 0;
            for (let c = 0; c < C; c++) if (c !== vpClass) otherSum += modelPred[y][x][c];
            for (let c = 0; c < C; c++) {
              if (c === vpClass) p[c] = conf;
              else p[c] = otherSum > 0 ? (1 - conf) * modelPred[y][x][c] / otherSum : (1 - conf) / (C - 1);
              p[c] = Math.max(p[c], 0.00005);
            }
            const sum = p.reduce((a, b) => a + b, 0);
            for (let c = 0; c < C; c++) p[c] /= sum;
            pred[y][x] = p;
          } else {
            pred[y][x] = modelPred[y][x];
          }
        } }

      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const s = pred[y][x].reduce((a, b) => a + b, 0);
        if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false; }
      if (!valid) { console.log(`  Seed ${si}: VALIDATION FAILED`); continue; }

      const res = await POST('/submit', { round_id: ROUND_ID, seed_index: si, prediction: pred });
      if (si === 0 || si === 4) console.log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'}`);
      submitCount++;
      await sleep(550);
    }
  }

  console.log(`\n=== DONE: ${submitCount} total seed submissions ===`);
  console.log('Time:', new Date().toISOString());
}

main().catch(e => console.error('Error:', e.message, e.stack));
