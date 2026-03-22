#!/usr/bin/env node
// SIMPLE AUTOPILOT — proven approach only, no experiments
// Cross-round model + moderate VP fusion + per-cell corrections, temp=1.1
// This scored 89.9 on R8. Submit ONCE, don't overwrite.
const fs = require('fs'), path = require('path'), https = require('https');
const H = 40, W = 40, SEEDS = 5, C = 6;
const DD = path.join(__dirname, 'data');
const TOKEN = process.argv[2] || '';
const BASE = 'https://api.ainm.no/astar-island';
if (!TOKEN) { console.log('Usage: node autopilot_simple.js <JWT>'); process.exit(1); }
if (!fs.existsSync(DD)) fs.mkdirSync(DD, { recursive: true });

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
const log = m => { const t = new Date().toISOString().slice(11, 19); console.log(`[${t}] ${m}`); };
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

function buildModel() {
  const I = {}, G = {}, R = {};
  for (let r = 1; r <= 30; r++) { const rn = `R${r}`;
    if (fs.existsSync(path.join(DD, `inits_${rn}.json`))) I[rn] = JSON.parse(fs.readFileSync(path.join(DD, `inits_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `gt_${rn}.json`))) G[rn] = JSON.parse(fs.readFileSync(path.join(DD, `gt_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `replays_${rn}.json`))) R[rn] = JSON.parse(fs.readFileSync(path.join(DD, `replays_${rn}.json`)));
  }
  const TR = Object.keys(I).filter(k => G[k]);
  log(`Training: ${TR.join(', ')}`);
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
  log(`Model: ${Object.keys(model).length} keys`);
  return model;
}

function predict(grid, model) {
  const TEMP = 1.1;
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
      for (let c = 0; c < C; c++) { p[c] = Math.pow(Math.max(p[c] / wS, 1e-10), 1 / TEMP);
        if (p[c] < 0.00005) p[c] = 0.00005; s += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    } }
  return pred;
}

async function collectVP(roundId, maxQueries) {
  log('Collecting viewport...');
  const vpObs = [];
  let used = 0;
  const base = [];
  for (const y of [0, 13, 25]) for (const x of [0, 13, 25]) base.push({ y, x });
  const extra = [{ y: 7, x: 7 }, { y: 7, x: 20 }, { y: 20, x: 7 }, { y: 20, x: 20 }, { y: 12, x: 12 }];
  for (let si = 0; si < SEEDS; si++) {
    for (const pos of [...base, extra[si]]) {
      if (used >= maxQueries) break;
      try {
        const res = await POST('/simulate', { round_id: roundId, seed_index: si, viewport_y: pos.y, viewport_x: pos.x });
        if (res.ok && res.data && res.data.grid) { vpObs.push({ si, vy: pos.y, vx: pos.x, grid: res.data.grid }); used++; }
        await sleep(250);
      } catch {}
    }
  }
  log(`  ${vpObs.length} VP observations collected`);
  return vpObs;
}

function fuseVP(model, vpObs, inits) {
  const CW = 20;
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
  for (const [k, vm] of Object.entries(vpD0)) {
    const bm = model[k];
    if (bm) {
      const pa = bm.a.map(p => p * CW), post = pa.map((a, c) => a + vm.counts[c]);
      const tot = post.reduce((a, b) => a + b, 0);
      model[k] = { n: bm.n + vm.n, a: post.map(v => v / tot) };
    }
  }
  log(`  VP fused: ${Object.keys(vpD0).length} D0 keys`);
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

function applyPerCell(pred, cellModel, initGrid) {
  for (const [key, cell] of Object.entries(cellModel)) {
    const [y, x] = key.split(',').map(Number);
    if (initGrid[y][x] === 10 || initGrid[y][x] === 5) continue;
    const pw = cell.n >= 5 ? 2 : cell.n >= 3 ? 4 : cell.n >= 2 ? 7 : 15;
    const prior = pred[y][x], posterior = new Array(C); let total = 0;
    for (let c = 0; c < C; c++) { posterior[c] = prior[c] * pw + cell.counts[c]; total += posterior[c]; }
    if (total > 0) { for (let c = 0; c < C; c++) posterior[c] /= total; pred[y][x] = posterior; }
  }
  return pred;
}

async function collectReplays(roundId, si) {
  try {
    const res = await POST('/replay', { round_id: roundId, seed_index: si });
    if (!res.ok || !res.data.frames) return null;
    return { si, finalGrid: res.data.frames[res.data.frames.length - 1].grid };
  } catch { return null; }
}

async function collectPostRound(round) {
  const rn = `R${round.round_number}`;
  const initFile = path.join(DD, `inits_${rn}.json`);
  if (!fs.existsSync(initFile)) {
    const { data: rd } = await GET('/rounds/' + round.id);
    fs.writeFileSync(initFile, JSON.stringify(rd.initial_states.map(is => is.grid)));
    log(`  Saved inits ${rn}`);
  }
  const gtFile = path.join(DD, `gt_${rn}.json`);
  if (!fs.existsSync(gtFile)) {
    const gts = [];
    for (let si = 0; si < SEEDS; si++) {
      const res = await GET(`/analysis/${round.id}/${si}`);
      if (res.ok && res.data.ground_truth) gts[si] = res.data.ground_truth;
    }
    if (gts.length === SEEDS && gts.every(g => g)) {
      fs.writeFileSync(gtFile, JSON.stringify(gts));
      log(`  Saved GT ${rn}`);
    }
  }
  const repFile = path.join(DD, `replays_${rn}.json`);
  let existing = fs.existsSync(repFile) ? JSON.parse(fs.readFileSync(repFile)) : [];
  if (existing.length < 500) {
    const needed = Math.min(100, 500 - existing.length);
    log(`  Collecting ${needed} replays for ${rn}...`);
    for (let i = 0; i < needed; i++) {
      const r = await collectReplays(round.id, i % SEEDS);
      if (r) existing.push(r);
      await sleep(50);
    }
    fs.writeFileSync(repFile, JSON.stringify(existing));
    log(`  ${rn}: ${existing.length} replays`);
  }
}

async function handleRound(round) {
  const roundId = round.id, rn = `R${round.round_number}`;
  const weight = Math.pow(1.05, round.round_number);
  log(`\n=== ${rn} (${roundId.slice(0,8)}) weight=${weight.toFixed(4)} ===`);

  const { data: rd } = await GET('/rounds/' + roundId);
  const inits = rd.initial_states.map(is => is.grid);
  fs.writeFileSync(path.join(DD, `inits_${rn}.json`), JSON.stringify(inits));

  // Build model
  const model = buildModel();

  // Collect VP
  const vpFile = path.join(DD, `viewport_${roundId.slice(0,8)}.json`);
  let vpObs = fs.existsSync(vpFile) ? JSON.parse(fs.readFileSync(vpFile)) : [];
  const remaining = 50 - vpObs.length;
  if (remaining > 0) {
    const newObs = await collectVP(roundId, remaining);
    vpObs.push(...newObs);
    fs.writeFileSync(vpFile, JSON.stringify(vpObs));
  }

  // Fuse VP into model
  if (vpObs.length > 0) fuseVP(model, vpObs, inits);

  // Build per-cell models
  const cellModels = vpObs.length > 0 ? buildCellModels(vpObs, inits) : null;

  // Submit
  log('Submitting predictions...');
  for (let si = 0; si < SEEDS; si++) {
    let pred = predict(inits[si], model);
    if (cellModels && cellModels[si]) pred = applyPerCell(pred, cellModels[si], inits[si]);
    // Validate
    let valid = true;
    for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
      const s = pred[y][x].reduce((a, b) => a + b, 0);
      if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false; }
    if (!valid) { log(`  Seed ${si}: INVALID`); continue; }
    const res = await POST('/submit', { round_id: roundId, seed_index: si, prediction: pred });
    log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'}`);
    await sleep(600);
  }
  log('Done.');
}

async function main() {
  log('=== SIMPLE AUTOPILOT ===');
  const handled = new Set();
  while (true) {
    try {
      const { data: rounds } = await GET('/rounds');
      if (!rounds) { await sleep(30000); continue; }
      for (const r of rounds.filter(r => r.status === 'active')) {
        if (!handled.has('a-' + r.id)) { handled.add('a-' + r.id); await handleRound(r); }
      }
      for (const r of rounds.filter(r => r.status === 'completed')) {
        if (!handled.has('c-' + r.id)) { handled.add('c-' + r.id); await collectPostRound(r); }
      }
    } catch (e) { log(`Error: ${e.message}`); }
    await sleep(30000);
  }
}
main().catch(e => { console.error(e); process.exit(1); });
