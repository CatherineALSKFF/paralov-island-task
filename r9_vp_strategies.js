#!/usr/bin/env node
/**
 * R9 VP STRATEGIES — Test different viewport weighting approaches
 * Since we have 100% VP coverage, the VP weighting is critical
 *
 * Strategy 1: No VP at all (pure model)
 * Strategy 2: D0 VP fusion only (no per-cell)
 * Strategy 3: Heavy VP per-cell (trust VP more)
 * Strategy 4: Confidence-weighted per-cell (trust VP when model is uncertain)
 * Strategy 5: VP as replay data (treat each VP obs as a replay)
 * Strategy 6: Multi-level VP fusion (D0 + D1 + D2)
 *
 * Usage: node r9_vp_strategies.js <JWT>
 */
const fs = require('fs'), path = require('path'), https = require('https');
const H = 40, W = 40, SEEDS = 5, C = 6;
const DD = path.join(__dirname, 'data');
const TOKEN = process.argv[2] || '';
const BASE = 'https://api.ainm.no/astar-island';

if (!TOKEN) { console.log('Usage: node r9_vp_strategies.js <JWT>'); process.exit(1); }

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

function buildModel(G, R, I, trainRounds, gtW, alpha) {
  const model = {};
  for (const level of ['d0', 'd1', 'd2', 'd3', 'd4']) {
    const m = {};
    for (const rn of trainRounds) {
      if (!G[rn] || !I[rn]) continue;
      for (let si = 0; si < SEEDS; si++) {
        if (!I[rn][si] || !G[rn][si]) continue;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = cf(I[rn][si], y, x); if (!keys) continue; const k = keys[level];
          if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
          const p = G[rn][si][y][x];
          for (let c = 0; c < C; c++) m[k].counts[c] += p[c] * gtW; m[k].n += gtW;
        }
      }
    }
    for (const rn of trainRounds) {
      if (!R[rn] || !I[rn]) continue;
      for (const rep of R[rn]) { const g = I[rn][rep.si]; if (!g) continue;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = cf(g, y, x); if (!keys) continue; const k = keys[level];
          const fc = t2c(rep.finalGrid[y][x]);
          if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) }; m[k].n++; m[k].counts[fc]++;
        }
      }
    }
    for (const k of Object.keys(m)) {
      const tot = Array.from(m[k].counts).reduce((a, b) => a + b, 0) + C * alpha;
      m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / tot);
    }
    for (const [k, v] of Object.entries(m)) { if (!model[k]) model[k] = v; }
  }
  return model;
}

function predict(grid, model, temp) {
  const levels = ['d0', 'd1', 'd2', 'd3', 'd4'], ws = [1.0, 0.3, 0.15, 0.08, 0.02];
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
      if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }
      const keys = cf(grid, y, x);
      if (!keys) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }
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

function validate(pred) {
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const s = pred[y][x].reduce((a, b) => a + b, 0);
    if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) return false;
  }
  return true;
}

async function submitPred(roundId, si, pred, name) {
  if (!validate(pred)) { console.log(`  ${name} seed ${si}: VALIDATION FAILED`); return; }
  const res = await POST('/submit', { round_id: roundId, seed_index: si, prediction: pred });
  console.log(`  ${name} seed ${si}: ${res.ok ? 'OK' : 'FAIL'}`);
  await sleep(600);
}

async function main() {
  console.log('=== R9 VP STRATEGIES ===');
  console.log('Time:', new Date().toISOString());

  // Load data
  const I = {}, G = {}, R = {};
  const trainRounds = [];
  for (let r = 1; r <= 20; r++) {
    if (r === 3) continue;
    const rn = `R${r}`;
    if (fs.existsSync(path.join(DD, `inits_${rn}.json`))) I[rn] = JSON.parse(fs.readFileSync(path.join(DD, `inits_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `gt_${rn}.json`))) G[rn] = JSON.parse(fs.readFileSync(path.join(DD, `gt_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `replays_${rn}.json`))) R[rn] = JSON.parse(fs.readFileSync(path.join(DD, `replays_${rn}.json`)));
    if (I[rn] && G[rn]) trainRounds.push(rn);
  }
  console.log('Training:', trainRounds.join(', '));

  const { data: rounds } = await GET('/rounds');
  const r9 = rounds.find(r => r.round_number === 9);
  if (!r9 || r9.status !== 'active') { console.log('R9 not active!'); return; }
  const { data: rd } = await GET('/rounds/' + r9.id);
  const inits = rd.initial_states.map(is => is.grid);

  const vpFile = path.join(DD, `viewport_${r9.id.slice(0,8)}.json`);
  const vpObs = JSON.parse(fs.readFileSync(vpFile));
  console.log('VP observations:', vpObs.length);

  // Build base model
  const baseModel = buildModel(G, R, I, trainRounds, 20, 0.05);
  console.log('Base model keys:', Object.keys(baseModel).length);

  // Get VP data per seed
  function getVPCells(si) {
    const cells = {};
    const seedObs = vpObs.filter(o => (o.si !== undefined ? o.si : 0) === si);
    for (const obs of seedObs) {
      for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
        const gy = obs.vy + dy, gx = obs.vx + dx;
        if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
        if (inits[si][gy][gx] === 10 || inits[si][gy][gx] === 5) continue;
        const k = `${gy},${gx}`, fc = t2c(obs.grid[dy][dx]);
        if (!cells[k]) cells[k] = { n: 0, counts: new Float64Array(C) };
        cells[k].n++; cells[k].counts[fc]++;
      }
    }
    return cells;
  }

  function getVPD0(si, cw) {
    const model = {};
    for (const [k, v] of Object.entries(baseModel)) model[k] = { n: v.n, a: [...v.a] };
    const vpD0 = {};
    const seedObs = vpObs.filter(o => (o.si !== undefined ? o.si : 0) === si);
    for (const obs of seedObs) {
      for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
        const gy = obs.vy + dy, gx = obs.vx + dx;
        if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
        const keys = cf(inits[si], gy, gx); if (!keys) continue;
        const k = keys.d0, fc = t2c(obs.grid[dy][dx]);
        if (!vpD0[k]) vpD0[k] = { n: 0, counts: new Float64Array(C) };
        vpD0[k].n++; vpD0[k].counts[fc]++;
      }
    }
    for (const [k, vm] of Object.entries(vpD0)) {
      const bm = model[k];
      if (bm) {
        const pa = bm.a.map(p => p * cw), post = pa.map((a, c) => a + vm.counts[c]);
        const tot = post.reduce((a, b) => a + b, 0);
        model[k] = { n: bm.n + vm.n, a: post.map(v => v / tot) };
      } else {
        const tot = vm.n + C * 0.1;
        model[k] = { n: vm.n, a: Array.from(vm.counts).map(v => (v + 0.1) / tot) };
      }
    }
    return model;
  }

  // Multi-level VP fusion (D0 + D1 + D2)
  function getVPMultiLevel(si, cw) {
    const model = {};
    for (const [k, v] of Object.entries(baseModel)) model[k] = { n: v.n, a: [...v.a] };

    for (const level of ['d0', 'd1', 'd2']) {
      const vpKeys = {};
      const seedObs = vpObs.filter(o => (o.si !== undefined ? o.si : 0) === si);
      for (const obs of seedObs) {
        for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
          const gy = obs.vy + dy, gx = obs.vx + dx;
          if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
          const keys = cf(inits[si], gy, gx); if (!keys) continue;
          const k = keys[level], fc = t2c(obs.grid[dy][dx]);
          if (!vpKeys[k]) vpKeys[k] = { n: 0, counts: new Float64Array(C) };
          vpKeys[k].n++; vpKeys[k].counts[fc]++;
        }
      }
      const levelCw = level === 'd0' ? cw : cw * 0.5; // Less weight for coarser levels
      for (const [k, vm] of Object.entries(vpKeys)) {
        const bm = model[k];
        if (bm) {
          const pa = bm.a.map(p => p * levelCw), post = pa.map((a, c) => a + vm.counts[c]);
          const tot = post.reduce((a, b) => a + b, 0);
          model[k] = { n: bm.n + vm.n, a: post.map(v => v / tot) };
        } else {
          const tot = vm.n + C * 0.1;
          model[k] = { n: vm.n, a: Array.from(vm.counts).map(v => (v + 0.1) / tot) };
        }
      }
    }
    return model;
  }

  const strategies = [];

  // Strategy 1: No VP at all
  for (const temp of [1.1, 1.15, 1.2]) {
    strategies.push({
      name: `noVP_t${temp}`,
      gen: (si) => predict(inits[si], baseModel, temp)
    });
  }

  // Strategy 2: D0 VP fusion only (no per-cell), different cw
  for (const cw of [10, 15, 20, 30, 40]) {
    for (const temp of [1.1, 1.15]) {
      strategies.push({
        name: `D0only_cw${cw}_t${temp}`,
        gen: (si) => {
          const model = getVPD0(si, cw);
          return predict(inits[si], model, temp);
        }
      });
    }
  }

  // Strategy 3: D0 VP + aggressive per-cell
  for (const [pwName, pwFn] of [
    ['ultraAgg', (n) => n >= 3 ? 0.5 : n >= 2 ? 1 : 3],
    ['veryAgg', (n) => n >= 3 ? 1 : n >= 2 ? 2 : 5],
    ['modAgg', (n) => n >= 5 ? 1 : n >= 3 ? 3 : n >= 2 ? 5 : 10],
    ['default', (n) => n >= 5 ? 2 : n >= 3 ? 4 : n >= 2 ? 7 : 15],
    ['conservative', (n) => n >= 5 ? 5 : n >= 3 ? 10 : n >= 2 ? 15 : 25],
  ]) {
    strategies.push({
      name: `D0_pc${pwName}_t1.15`,
      gen: (si) => {
        const model = getVPD0(si, 20);
        const pred = predict(inits[si], model, 1.15);
        const result = pred.map(row => row.map(cell => [...cell]));
        const cells = getVPCells(si);
        for (const [key, cell] of Object.entries(cells)) {
          const [y, x] = key.split(',').map(Number);
          const pw = pwFn(cell.n);
          const prior = result[y][x], posterior = new Array(C); let total = 0;
          for (let c = 0; c < C; c++) { posterior[c] = prior[c] * pw + cell.counts[c]; total += posterior[c]; }
          if (total > 0) { for (let c = 0; c < C; c++) posterior[c] /= total; result[y][x] = posterior; }
        }
        return result;
      }
    });
  }

  // Strategy 4: Confidence-weighted per-cell (trust VP more when model uncertain)
  strategies.push({
    name: 'confWeighted_t1.15',
    gen: (si) => {
      const model = getVPD0(si, 20);
      const pred = predict(inits[si], model, 1.15);
      const result = pred.map(row => row.map(cell => [...cell]));
      const cells = getVPCells(si);
      for (const [key, cell] of Object.entries(cells)) {
        const [y, x] = key.split(',').map(Number);
        // Calculate model entropy
        let ent = 0;
        for (let c = 0; c < C; c++) {
          if (result[y][x][c] > 0) ent -= result[y][x][c] * Math.log(result[y][x][c]);
        }
        const maxEnt = Math.log(C);
        // High entropy = trust VP more (lower pw)
        const entRatio = ent / maxEnt; // 0 = very confident, 1 = maximum uncertainty
        const pw = Math.max(0.5, 2 + 15 * (1 - entRatio)); // pw from 0.5 (uncertain) to 17 (confident)
        const prior = result[y][x], posterior = new Array(C); let total = 0;
        for (let c = 0; c < C; c++) { posterior[c] = prior[c] * pw + cell.counts[c]; total += posterior[c]; }
        if (total > 0) { for (let c = 0; c < C; c++) posterior[c] /= total; result[y][x] = posterior; }
      }
      return result;
    }
  });

  // Strategy 5: Multi-level VP fusion + per-cell
  strategies.push({
    name: 'multiLevel_t1.15',
    gen: (si) => {
      const model = getVPMultiLevel(si, 20);
      const pred = predict(inits[si], model, 1.15);
      const result = pred.map(row => row.map(cell => [...cell]));
      const cells = getVPCells(si);
      for (const [key, cell] of Object.entries(cells)) {
        const [y, x] = key.split(',').map(Number);
        const pw = cell.n >= 5 ? 2 : cell.n >= 3 ? 4 : cell.n >= 2 ? 7 : 15;
        const prior = result[y][x], posterior = new Array(C); let total = 0;
        for (let c = 0; c < C; c++) { posterior[c] = prior[c] * pw + cell.counts[c]; total += posterior[c]; }
        if (total > 0) { for (let c = 0; c < C; c++) posterior[c] /= total; result[y][x] = posterior; }
      }
      return result;
    }
  });

  // Strategy 6: VP as replay-like data with Dirichlet smoothing
  strategies.push({
    name: 'vpReplay_t1.15',
    gen: (si) => {
      const model = getVPD0(si, 20);
      const pred = predict(inits[si], model, 1.15);
      const result = pred.map(row => row.map(cell => [...cell]));
      const cells = getVPCells(si);

      // Treat VP like replays — use adaptive Dirichlet like replay_resubmit
      for (const [key, cell] of Object.entries(cells)) {
        const [y, x] = key.split(',').map(Number);
        const N = cell.n;

        // Count unique classes
        let unique = 0, maxC = 0;
        for (let c = 0; c < C; c++) { if (cell.counts[c] > 0) unique++; if (cell.counts[c] > maxC) maxC = cell.counts[c]; }

        let alpha;
        if (unique <= 1) alpha = 0.5; // Very few samples, trust model more
        else alpha = 1.0;

        // Blend model prior with VP counts
        const prior = result[y][x];
        const posterior = new Array(C);
        let total = 0;
        for (let c = 0; c < C; c++) {
          // Model as pseudo-count of 5, VP as actual counts
          posterior[c] = prior[c] * 5 + cell.counts[c] + alpha / C;
          total += posterior[c];
        }
        if (total > 0) { for (let c = 0; c < C; c++) posterior[c] /= total; result[y][x] = posterior; }
      }
      return result;
    }
  });

  // Strategy 7: Higher temp (more spread) with aggressive VP
  for (const temp of [1.3, 1.5, 2.0]) {
    strategies.push({
      name: `highTemp_t${temp}_aggVP`,
      gen: (si) => {
        const model = getVPD0(si, 20);
        const pred = predict(inits[si], model, temp);
        const result = pred.map(row => row.map(cell => [...cell]));
        const cells = getVPCells(si);
        for (const [key, cell] of Object.entries(cells)) {
          const [y, x] = key.split(',').map(Number);
          const pw = cell.n >= 3 ? 1 : cell.n >= 2 ? 2 : 5;
          const prior = result[y][x], posterior = new Array(C); let total = 0;
          for (let c = 0; c < C; c++) { posterior[c] = prior[c] * pw + cell.counts[c]; total += posterior[c]; }
          if (total > 0) { for (let c = 0; c < C; c++) posterior[c] /= total; result[y][x] = posterior; }
        }
        return result;
      }
    });
  }

  // Strategy 8: Low temp (more peaked) for confident model
  for (const temp of [0.8, 0.9, 0.95]) {
    strategies.push({
      name: `lowTemp_t${temp}_defaultVP`,
      gen: (si) => {
        const model = getVPD0(si, 20);
        const pred = predict(inits[si], model, temp);
        const result = pred.map(row => row.map(cell => [...cell]));
        const cells = getVPCells(si);
        for (const [key, cell] of Object.entries(cells)) {
          const [y, x] = key.split(',').map(Number);
          const pw = cell.n >= 5 ? 2 : cell.n >= 3 ? 4 : cell.n >= 2 ? 7 : 15;
          const prior = result[y][x], posterior = new Array(C); let total = 0;
          for (let c = 0; c < C; c++) { posterior[c] = prior[c] * pw + cell.counts[c]; total += posterior[c]; }
          if (total > 0) { for (let c = 0; c < C; c++) posterior[c] /= total; result[y][x] = posterior; }
        }
        return result;
      }
    });
  }

  console.log(`\nSubmitting ${strategies.length} strategies...`);

  // Submit all strategies
  for (let si_strat = 0; si_strat < strategies.length; si_strat++) {
    const strat = strategies[si_strat];
    console.log(`\n[${si_strat+1}/${strategies.length}] ${strat.name}`);
    for (let si = 0; si < SEEDS; si++) {
      const pred = strat.gen(si);
      await submitPred(r9.id, si, pred, strat.name);
    }
  }

  // Final leaderboard check
  console.log('\n=== FINAL CHECK ===');
  const lb = await GET('/leaderboard');
  if (lb && lb.data) {
    const us = lb.data.find(t => t.team_name && t.team_name.includes('CAL'));
    if (us) {
      const rank = lb.data.indexOf(us) + 1;
      console.log(`Rank: #${rank} ws=${us.weighted_score.toFixed(4)}`);
    }
  }
  console.log('Time:', new Date().toISOString());
}

main().catch(e => console.error('Error:', e.message, e.stack));
