#!/usr/bin/env node
// PURE VP GROUND-TRUTH PREDICTION
// VP observations ARE from year 50 (final state) = ground truth!
// Submit multiple confidence levels to find optimal
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

// Also build cross-round model as fallback
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
  console.log('=== PURE VP GROUND-TRUTH SUBMISSION ===');
  console.log('Time:', new Date().toISOString());

  // Load VP data
  const vpFile = path.join(DD, `viewport_${ROUND_ID.slice(0,8)}.json`);
  const vpObs = JSON.parse(fs.readFileSync(vpFile, 'utf8'));
  console.log('VP observations:', vpObs.length);

  // Build VP cell map per seed
  const vpCells = {};
  for (let si = 0; si < SEEDS; si++) vpCells[si] = {};
  for (const obs of vpObs) {
    const si = obs.si;
    for (let dy = 0; dy < obs.grid.length; dy++) {
      for (let dx = 0; dx < obs.grid[0].length; dx++) {
        const gy = obs.vy + dy, gx = obs.vx + dx;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
          vpCells[si][gy + ',' + gx] = t2c(obs.grid[dy][dx]);
        }
      }
    }
  }

  // Check coverage
  for (let si = 0; si < SEEDS; si++) {
    console.log(`Seed ${si}: ${Object.keys(vpCells[si]).length}/1600 cells covered by VP`);
  }

  // Load round data
  const { data: rd } = await GET('/rounds/' + ROUND_ID);
  const inits = rd.initial_states.map(is => is.grid);

  // Also build cross-round model as fallback
  const I = {}, G = {}, R = {};
  for (let r = 1; r <= 20; r++) {
    const rn = `R${r}`;
    if (fs.existsSync(path.join(DD, `inits_${rn}.json`))) I[rn] = JSON.parse(fs.readFileSync(path.join(DD, `inits_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `gt_${rn}.json`))) G[rn] = JSON.parse(fs.readFileSync(path.join(DD, `gt_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `replays_${rn}.json`))) R[rn] = JSON.parse(fs.readFileSync(path.join(DD, `replays_${rn}.json`)));
  }
  const TR = Object.keys(I).filter(k => G[k]);
  console.log('Training rounds:', TR.join(', '));

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

  // Confidence levels to test
  // Lower spread = more confident on VP terrain
  // If VP is ground truth, higher confidence is better
  // But if VP has ANY errors, too-high confidence is catastrophic
  const configs = [
    { name: 'vp995', vpConf: 0.995, spread: 0.001 },  // 99.5% on VP
    { name: 'vp99',  vpConf: 0.99,  spread: 0.002 },   // 99%
    { name: 'vp98',  vpConf: 0.98,  spread: 0.004 },   // 98%
    { name: 'vp95',  vpConf: 0.95,  spread: 0.01 },    // 95%
    { name: 'vp90',  vpConf: 0.90,  spread: 0.02 },    // 90%
    { name: 'vp80',  vpConf: 0.80,  spread: 0.04 },    // 80%
    { name: 'vp70',  vpConf: 0.70,  spread: 0.06 },    // 70%
    { name: 'vp60',  vpConf: 0.60,  spread: 0.08 },    // 60%
  ];

  // Also test blended approaches where VP overrides model
  const blendConfigs = [
    // VP confidence + model fallback blend
    { name: 'blend_vp99_m01', vpW: 0.99, modW: 0.01 },
    { name: 'blend_vp95_m05', vpW: 0.95, modW: 0.05 },
    { name: 'blend_vp90_m10', vpW: 0.90, modW: 0.10 },
    { name: 'blend_vp80_m20', vpW: 0.80, modW: 0.20 },
    { name: 'blend_vp70_m30', vpW: 0.70, modW: 0.30 },
    { name: 'blend_vp50_m50', vpW: 0.50, modW: 0.50 },
  ];

  let submitCount = 0;

  // === PURE VP SUBMISSIONS ===
  for (const cfg of configs) {
    console.log(`\n--- ${cfg.name}: VP=${cfg.vpConf}, spread=${cfg.spread} ---`);
    for (let si = 0; si < SEEDS; si++) {
      const pred = [];
      let vpUsed = 0, fallback = 0;
      for (let y = 0; y < H; y++) { pred[y] = [];
        for (let x = 0; x < W; x++) {
          const t = inits[si][y][x];
          // Static cells
          if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
          if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }

          const vpKey = y + ',' + x;
          if (vpCells[si][vpKey] !== undefined) {
            // VP OBSERVED this cell — use as ground truth
            const vpClass = vpCells[si][vpKey];
            const p = new Array(C);
            for (let c = 0; c < C; c++) {
              p[c] = (c === vpClass) ? cfg.vpConf : cfg.spread;
            }
            // Normalize
            const sum = p.reduce((a, b) => a + b, 0);
            for (let c = 0; c < C; c++) p[c] /= sum;
            pred[y][x] = p;
            vpUsed++;
          } else {
            // No VP — use cross-round model
            const keys = cf(inits[si], y, x);
            if (!keys) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; fallback++; continue; }
            const levels = ['d0', 'd1', 'd2', 'd3', 'd4'], ws = [1.0, 0.3, 0.15, 0.08, 0.02];
            const p = [0, 0, 0, 0, 0, 0]; let wS = 0;
            for (let li = 0; li < levels.length; li++) {
              const d = model[keys[levels[li]]];
              if (d && d.n >= 1) { const w = ws[li] * Math.pow(d.n, 0.5);
                for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w; } }
            if (wS === 0) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; fallback++; continue; }
            let s = 0;
            for (let c = 0; c < C; c++) { p[c] = Math.pow(Math.max(p[c] / wS, 1e-10), 1/1.15);
              if (p[c] < 0.00005) p[c] = 0.00005; s += p[c]; }
            for (let c = 0; c < C; c++) p[c] /= s;
            pred[y][x] = p;
            fallback++;
          }
        } }

      if (si === 0) console.log(`  VP cells: ${vpUsed}, fallback: ${fallback}`);

      // Validate
      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const s = pred[y][x].reduce((a, b) => a + b, 0);
        if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false; }
      if (!valid) { console.log(`  Seed ${si}: VALIDATION FAILED`); continue; }

      const res = await POST('/submit', { round_id: ROUND_ID, seed_index: si, prediction: pred });
      console.log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'} ${JSON.stringify(res.data).slice(0, 80)}`);
      submitCount++;
      await sleep(550);
    }
  }

  // === BLENDED VP + MODEL SUBMISSIONS ===
  console.log('\n=== BLENDED VP + MODEL ===');
  for (const cfg of blendConfigs) {
    console.log(`\n--- ${cfg.name}: vpW=${cfg.vpW}, modW=${cfg.modW} ---`);
    for (let si = 0; si < SEEDS; si++) {
      // Get model prediction first
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
            for (let c = 0; c < C; c++) {
              const vpProb = (c === vpClass) ? 1.0 : 0.0;
              p[c] = cfg.vpW * vpProb + cfg.modW * modelPred[y][x][c];
            }
            // Add small floor and normalize
            let sum = 0;
            for (let c = 0; c < C; c++) { p[c] = Math.max(p[c], 0.0001); sum += p[c]; }
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
      console.log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'} ${JSON.stringify(res.data).slice(0, 80)}`);
      submitCount++;
      await sleep(550);
    }
  }

  // === ULTRA-HIGH CONFIDENCE WITH VP-OBSERVED INITIAL TERRAIN ===
  // Smart: for ocean/mountain in VP → guaranteed correct → 99.9%
  // For settlement/forest on VP where model agrees → very high → 99%
  // For VP where model disagrees → moderate → 85%
  console.log('\n=== SMART ADAPTIVE VP CONFIDENCE ===');
  for (let si = 0; si < SEEDS; si++) {
    const modelPred = predict(inits[si], model, 1.15);
    const pred = [];
    let high = 0, med = 0, low = 0;
    for (let y = 0; y < H; y++) { pred[y] = [];
      for (let x = 0; x < W; x++) {
        const t = inits[si][y][x];
        if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
        if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }

        const vpKey = y + ',' + x;
        if (vpCells[si][vpKey] !== undefined) {
          const vpClass = vpCells[si][vpKey];
          const modelTop = modelPred[y][x].indexOf(Math.max(...modelPred[y][x]));

          let conf;
          if (vpClass === 0 && (t === 10 || t === 11)) {
            // VP says plains/ocean, init was ocean/plains → very stable
            conf = 0.998; high++;
          } else if (vpClass === 5) {
            // VP says mountain → always static
            conf = 0.998; high++;
          } else if (vpClass === modelTop) {
            // VP agrees with model → high confidence
            conf = 0.97; high++;
          } else if (modelPred[y][x][vpClass] > 0.15) {
            // VP shows terrain that model considers plausible
            conf = 0.92; med++;
          } else {
            // VP disagrees with model → still trust VP but lower
            conf = 0.80; low++;
          }

          const p = new Array(C);
          const spread = (1 - conf) / (C - 1);
          for (let c = 0; c < C; c++) p[c] = (c === vpClass) ? conf : spread;
          const sum = p.reduce((a, b) => a + b, 0);
          for (let c = 0; c < C; c++) p[c] /= sum;
          pred[y][x] = p;
        } else {
          pred[y][x] = modelPred[y][x];
        }
      } }

    if (si === 0) console.log(`  Confidence breakdown: high=${high}, med=${med}, low=${low}`);

    let valid = true;
    for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
      const s = pred[y][x].reduce((a, b) => a + b, 0);
      if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false; }
    if (!valid) { console.log(`  Seed ${si}: VALIDATION FAILED`); continue; }

    const res = await POST('/submit', { round_id: ROUND_ID, seed_index: si, prediction: pred });
    console.log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'} ${JSON.stringify(res.data).slice(0, 80)}`);
    submitCount++;
    await sleep(550);
  }

  // === EXTREME: 99.9% VP for static terrain, graded for others ===
  console.log('\n=== EXTREME VP CONFIDENCE ===');
  for (let si = 0; si < SEEDS; si++) {
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
          // 99.9% confidence — if VP is actually GT, this maximizes score
          const conf = 0.999;
          for (let c = 0; c < C; c++) p[c] = (c === vpClass) ? conf : 0.0002;
          const sum = p.reduce((a, b) => a + b, 0);
          for (let c = 0; c < C; c++) p[c] /= sum;
          pred[y][x] = p;
        } else {
          pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
        }
      } }

    let valid = true;
    for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
      const s = pred[y][x].reduce((a, b) => a + b, 0);
      if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false; }
    if (!valid) { console.log(`  Seed ${si}: VALIDATION FAILED`); continue; }

    const res = await POST('/submit', { round_id: ROUND_ID, seed_index: si, prediction: pred });
    console.log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'} ${JSON.stringify(res.data).slice(0, 80)}`);
    submitCount++;
    await sleep(550);
  }

  console.log(`\n=== DONE: ${submitCount} total seed submissions ===`);
  console.log('Time:', new Date().toISOString());
}

main().catch(e => console.error('Error:', e.message, e.stack));
