#!/usr/bin/env node
// R8 CLEAN FINAL — Cross-round (gtW=20, temp=1.1) + D0 viewport (cw=20) + per-cell (ALL cells, pw=5)
const fs = require('fs'), path = require('path'), https = require('https');
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
  return { d0: `D0_${t}_${sa}_${co}_${sb2}_${fb}`, d1: `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`,
    d2: `D2_${t}_${sa > 0 ? 1 : 0}_${co}`, d3: `D3_${t}_${co}`, d4: `D4_${t}` };
}

function buildModel(gts, reps, inits, rounds, level, alpha, gtW) {
  const m = {};
  // GT
  for (const rn of rounds) {
    if (!gts[rn] || !inits[rn]) continue;
    for (let si = 0; si < SEEDS; si++) {
      if (!inits[rn][si] || !gts[rn][si]) continue;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cf(inits[rn][si], y, x); if (!keys) continue;
        const k = keys[level];
        if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
        const p = gts[rn][si][y][x];
        for (let c = 0; c < C; c++) m[k].counts[c] += p[c] * gtW;
        m[k].n += gtW;
      }
    }
  }
  // Replays
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
    const tot = Array.from(m[k].counts).reduce((a, b) => a + b, 0) + C * alpha;
    m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / tot);
  }
  return m;
}

function predict(grid, model, temp) {
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
  console.log('=== R8 CLEAN SUBMISSION ===');
  console.log('Time:', new Date().toISOString());

  const I = {}, G = {}, R = {}, TR = [];
  for (let r = 1; r <= 7; r++) {
    if (r === 3) continue;
    const rn = `R${r}`;
    if (fs.existsSync(path.join(DD, `inits_${rn}.json`))) I[rn] = JSON.parse(fs.readFileSync(path.join(DD, `inits_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `gt_${rn}.json`))) G[rn] = JSON.parse(fs.readFileSync(path.join(DD, `gt_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `replays_${rn}.json`))) R[rn] = JSON.parse(fs.readFileSync(path.join(DD, `replays_${rn}.json`)));
    if (I[rn] && G[rn]) TR.push(rn);
  }
  console.log('Training:', TR.join(', '));
  console.log('Replays:', TR.filter(r => R[r]).map(r => `${r}=${R[r].length}`).join(', '));

  const vpObs = JSON.parse(fs.readFileSync(path.join(DD, 'viewport_c5cdf100.json')));
  const { data: rd } = await GET('/rounds/' + R8);
  const inits = rd.initial_states.map(is => is.grid);
  console.log('R8 loaded, viewport obs:', vpObs.length);

  // Build multi-level cross-round model (GTW=20)
  const model = {};
  for (const level of ['d0', 'd1', 'd2', 'd3', 'd4']) {
    const lm = buildModel(G, R, I, TR, level, 0.05, 20);
    for (const [k, v] of Object.entries(lm)) {
      if (!model[k]) model[k] = v;
    }
  }
  console.log('Cross-round:', Object.keys(model).length, 'keys');

  // Fuse D0 viewport with cross-round (cw=20)
  const CW = 20;
  for (const obs of vpObs) {
    const si = obs.si !== undefined ? obs.si : 0;
    for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
      const gy = obs.vy + dy, gx = obs.vx + dx;
      if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
      const keys = cf(inits[si], gy, gx);
      if (!keys) continue;
      const k = keys.d0;
      if (!model[k]) model[k] = { n: 0, counts: new Float64Array(C), a: Array(C).fill(1/C) };
      // This is additive — we just add viewport counts on top
      // The existing D0 key already has cross-round data
    }
  }
  // Better: build separate viewport model and merge
  const vpModel = {};
  for (const obs of vpObs) {
    const si = obs.si !== undefined ? obs.si : 0;
    for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
      const gy = obs.vy + dy, gx = obs.vx + dx;
      if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
      const keys = cf(inits[si], gy, gx);
      if (!keys) continue;
      const k = keys.d0;
      const fc = t2c(obs.grid[dy][dx]);
      if (!vpModel[k]) vpModel[k] = { n: 0, counts: new Float64Array(C) };
      vpModel[k].n++;
      vpModel[k].counts[fc]++;
    }
  }
  // Fuse
  for (const [k, vm] of Object.entries(vpModel)) {
    const bm = model[k];
    if (bm) {
      const priorAlpha = bm.a.map(p => p * CW);
      const posterior = priorAlpha.map((a, c) => a + vm.counts[c]);
      const total = posterior.reduce((a, b) => a + b, 0);
      model[k] = { n: bm.n + vm.n, a: posterior.map(v => v / total) };
    } else {
      // No cross-round D0, try D1 as prior
      const parts = k.split('_');
      const t = parts[1], sa = parseInt(parts[2]), co = parts[3], sb2 = parts[4];
      const d1key = `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`;
      const cm = model[d1key];
      if (cm) {
        const priorAlpha = cm.a.map(p => p * CW);
        const posterior = priorAlpha.map((a, c) => a + vm.counts[c]);
        const total = posterior.reduce((a, b) => a + b, 0);
        model[k] = { n: vm.n + CW, a: posterior.map(v => v / total) };
      } else {
        const total = vm.n + C * 0.1;
        model[k] = { n: vm.n, a: Array.from(vm.counts).map(v => (v + 0.1) / total) };
      }
    }
  }
  console.log('Fused model:', Object.keys(model).length, 'keys');

  // Build per-cell models
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
      for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
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
    cellModels[si] = cells;
  }

  // Submit with temp=1.1, per-cell pw=5 (all cells, not just >=3)
  const TEMP = 1.1;
  const PW = 5;
  console.log(`\nSubmitting: temp=${TEMP}, per-cell pw=${PW} (ALL cells)`);

  for (let si = 0; si < SEEDS; si++) {
    let pred = predict(inits[si], model, TEMP);

    // Apply per-cell for ALL observed cells
    const cells = cellModels[si];
    for (const [key, cell] of Object.entries(cells)) {
      const [y, x] = key.split(',').map(Number);
      if (inits[si][y][x] === 10 || inits[si][y][x] === 5) continue;
      // Adaptive weight: more obs = less prior weight needed
      const pw = cell.n >= 3 ? PW * 0.5 : cell.n >= 2 ? PW * 0.7 : PW;
      const prior = pred[y][x];
      const posterior = new Array(C);
      let total = 0;
      for (let c = 0; c < C; c++) {
        posterior[c] = prior[c] * pw + cell.counts[c];
        total += posterior[c];
      }
      if (total > 0) {
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
    await sleep(600);
  }

  console.log('\n=== R8 SUBMITTED ===');
  console.log(`Model: GTW=20 + VP cw=${CW} + temp=${TEMP} + per-cell pw=${PW} (all cells)`);
  console.log('Per-cell coverage: ~1350-1384 cells per seed (full 15x15 grid × 10 obs)');
  console.log('Need: score > 80.3 for ws > 118.63 (#1)');
}

main().catch(e => console.error('Error:', e.message, e.stack));
