#!/usr/bin/env node
/**
 * R9 ENSEMBLE — Average predictions from multiple model configs
 * Ensemble averaging reduces prediction variance → lower KL → higher score
 * KL divergence is convex: E[KL(GT, avg_pred)] ≤ E[avg(KL(GT, pred_i))]
 *
 * Usage: node r9_ensemble.js <JWT>
 */
const fs = require('fs'), path = require('path'), https = require('https');
const H = 40, W = 40, SEEDS = 5, C = 6;
const DD = path.join(__dirname, 'data');
const TOKEN = process.argv[2] || '';
const BASE = 'https://api.ainm.no/astar-island';

if (!TOKEN) { console.log('Usage: node r9_ensemble.js <JWT>'); process.exit(1); }

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

function fuseVP(model, vpObs, inits, cw) {
  const fused = {};
  for (const [k, v] of Object.entries(model)) {
    fused[k] = { n: v.n, a: [...v.a] };
  }
  if (vpObs.length === 0) return fused;
  const vpD0 = {};
  for (const obs of vpObs) {
    const si = obs.si !== undefined ? obs.si : 0;
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

function predict(grid, model, temp, dWeights) {
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
      if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }
      const keys = cf(grid, y, x);
      if (!keys) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }
      const levels = ['d0', 'd1', 'd2', 'd3', 'd4'];
      const p = [0, 0, 0, 0, 0, 0]; let wS = 0;
      for (let li = 0; li < levels.length; li++) {
        const d = model[keys[levels[li]]];
        if (d && d.n >= 1) { const w = dWeights[li] * Math.pow(d.n, 0.5);
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

function applyPerCell(pred, vpObs, inits, si, pwSchedule) {
  const result = pred.map(row => row.map(cell => [...cell]));
  const cells = {};
  const obsBySeed = vpObs.filter(obs => (obs.si !== undefined ? obs.si : 0) === si);
  for (const obs of obsBySeed) {
    for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
      const gy = obs.vy + dy, gx = obs.vx + dx;
      if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
      if (inits[si][gy][gx] === 10 || inits[si][gy][gx] === 5) continue;
      const k = `${gy},${gx}`, fc = t2c(obs.grid[dy][dx]);
      if (!cells[k]) cells[k] = { n: 0, counts: new Float64Array(C) };
      cells[k].n++; cells[k].counts[fc]++;
    }
  }
  for (const [key, cell] of Object.entries(cells)) {
    const [y, x] = key.split(',').map(Number);
    let pw;
    if (cell.n >= pwSchedule[0][0]) pw = pwSchedule[0][1];
    else if (cell.n >= pwSchedule[1][0]) pw = pwSchedule[1][1];
    else if (cell.n >= pwSchedule[2][0]) pw = pwSchedule[2][1];
    else pw = pwSchedule[3][1];
    const prior = result[y][x], posterior = new Array(C); let total = 0;
    for (let c = 0; c < C; c++) { posterior[c] = prior[c] * pw + cell.counts[c]; total += posterior[c]; }
    if (total > 0) { for (let c = 0; c < C; c++) posterior[c] /= total; result[y][x] = posterior; }
  }
  return result;
}

async function main() {
  console.log('=== R9 ENSEMBLE SUBMISSION ===');
  console.log('Time:', new Date().toISOString());

  // Load all data
  const I = {}, G = {}, R = {};
  const allRounds = [];
  for (let r = 1; r <= 20; r++) {
    const rn = `R${r}`;
    if (fs.existsSync(path.join(DD, `inits_${rn}.json`))) I[rn] = JSON.parse(fs.readFileSync(path.join(DD, `inits_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `gt_${rn}.json`))) G[rn] = JSON.parse(fs.readFileSync(path.join(DD, `gt_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `replays_${rn}.json`))) R[rn] = JSON.parse(fs.readFileSync(path.join(DD, `replays_${rn}.json`)));
    if (I[rn] && G[rn]) allRounds.push(rn);
  }
  const trainRounds = allRounds.filter(r => r !== 'R3');
  console.log('Training:', trainRounds.join(', '));
  console.log('Replays:', allRounds.filter(r => R[r]).map(r => `${r}=${R[r].length}`).join(', '));

  // Get R9
  const { data: rounds } = await GET('/rounds');
  const r9 = rounds.find(r => r.round_number === 9);
  if (!r9 || r9.status !== 'active') { console.log('R9 not active!'); return; }
  console.log('R9:', r9.id, 'closes:', r9.closes_at);

  const { data: rd } = await GET('/rounds/' + r9.id);
  const inits = rd.initial_states.map(is => is.grid);

  // Load VP
  const vpFile = path.join(DD, `viewport_${r9.id.slice(0,8)}.json`);
  let vpObs = [];
  if (fs.existsSync(vpFile)) {
    vpObs = JSON.parse(fs.readFileSync(vpFile));
    console.log('VP observations:', vpObs.length);
  }

  const dW = [1.0, 0.3, 0.15, 0.08, 0.02];
  const pwDefault = [[5, 2], [3, 4], [2, 7], [1, 15]];

  // Define ensemble members — diverse configs
  const members = [
    { gtW: 20, alpha: 0.05, cw: 20, temp: 1.15, name: 'baseline' },
    { gtW: 20, alpha: 0.05, cw: 20, temp: 1.10, name: 't1.10' },
    { gtW: 20, alpha: 0.05, cw: 20, temp: 1.20, name: 't1.20' },
    { gtW: 15, alpha: 0.05, cw: 20, temp: 1.15, name: 'gtW15' },
    { gtW: 30, alpha: 0.05, cw: 20, temp: 1.15, name: 'gtW30' },
    { gtW: 20, alpha: 0.02, cw: 20, temp: 1.15, name: 'a0.02' },
    { gtW: 20, alpha: 0.10, cw: 20, temp: 1.15, name: 'a0.10' },
    { gtW: 20, alpha: 0.05, cw: 15, temp: 1.15, name: 'cw15' },
    { gtW: 20, alpha: 0.05, cw: 30, temp: 1.15, name: 'cw30' },
    { gtW: 10, alpha: 0.05, cw: 20, temp: 1.15, name: 'gtW10' },
    { gtW: 20, alpha: 0.05, cw: 10, temp: 1.15, name: 'cw10' },
    { gtW: 25, alpha: 0.08, cw: 25, temp: 1.12, name: 'alt1' },
  ];

  console.log(`\nBuilding ${members.length} ensemble members...`);

  // Build predictions for each member, for each seed
  const memberPreds = []; // memberPreds[mi][si] = pred
  for (let mi = 0; mi < members.length; mi++) {
    const m = members[mi];
    const baseModel = buildModel(G, R, I, trainRounds, m.gtW, m.alpha);
    const model = fuseVP(baseModel, vpObs, inits, m.cw);

    const preds = [];
    for (let si = 0; si < SEEDS; si++) {
      let pred = predict(inits[si], model, m.temp, dW);
      pred = applyPerCell(pred, vpObs, inits, si, pwDefault);
      preds.push(pred);
    }
    memberPreds.push(preds);
    console.log(`  Member ${mi}: ${m.name} built`);
  }

  // ============================================
  // Submit ensembles of different sizes
  // ============================================
  const ensembleSizes = [
    { n: members.length, name: 'all12' },
    { n: 7, name: 'top7' },  // First 7 (temp + gtW + alpha diversity)
    { n: 5, name: 'top5' },  // First 5 (core diversity)
    { n: 3, name: 'top3' },  // Just temp diversity
    { n: 9, name: 'top9' },  // Temp + gtW + alpha + cw diversity
  ];

  // Also submit single best configs with per-cell
  const singleConfigs = [0, 1, 2, 3, 4, 5, 6]; // indices into members

  console.log('\n=== SUBMITTING ENSEMBLES ===');

  for (const ens of ensembleSizes) {
    const N = ens.n;
    console.log(`\nEnsemble ${ens.name} (${N} members):`);

    for (let si = 0; si < SEEDS; si++) {
      // Average predictions across ensemble members
      const avgPred = [];
      for (let y = 0; y < H; y++) {
        avgPred[y] = [];
        for (let x = 0; x < W; x++) {
          avgPred[y][x] = new Array(C).fill(0);
          for (let mi = 0; mi < N; mi++) {
            for (let c = 0; c < C; c++) {
              avgPred[y][x][c] += memberPreds[mi][si][y][x][c] / N;
            }
          }
          // Ensure normalization
          let sum = 0;
          for (let c = 0; c < C; c++) sum += avgPred[y][x][c];
          for (let c = 0; c < C; c++) avgPred[y][x][c] /= sum;
        }
      }

      // Validate
      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const s = avgPred[y][x].reduce((a, b) => a + b, 0);
        if (Math.abs(s - 1) > 0.02 || avgPred[y][x].some(v => v < 0)) valid = false;
      }
      if (!valid) { console.log(`  Seed ${si}: VALIDATION FAILED`); continue; }

      const res = await POST('/submit', { round_id: r9.id, seed_index: si, prediction: avgPred });
      const score = res.data && res.data.score !== undefined ? res.data.score : '?';
      console.log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'} score=${score}`);
      await sleep(600);
    }
  }

  // Also submit best individual members
  console.log('\n=== SUBMITTING INDIVIDUAL MEMBERS ===');
  for (const mi of singleConfigs) {
    const m = members[mi];
    console.log(`\nMember ${mi}: ${m.name}`);
    for (let si = 0; si < SEEDS; si++) {
      const pred = memberPreds[mi][si];
      const res = await POST('/submit', { round_id: r9.id, seed_index: si, prediction: pred });
      const score = res.data && res.data.score !== undefined ? res.data.score : '?';
      console.log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'} score=${score}`);
      await sleep(600);
    }
  }

  // Also try geometric mean ensemble (alternative to arithmetic mean)
  console.log('\n=== GEOMETRIC MEAN ENSEMBLE (all members) ===');
  for (let si = 0; si < SEEDS; si++) {
    const geoPred = [];
    for (let y = 0; y < H; y++) {
      geoPred[y] = [];
      for (let x = 0; x < W; x++) {
        geoPred[y][x] = new Array(C).fill(0);
        // Geometric mean in log space
        for (let c = 0; c < C; c++) {
          let logSum = 0;
          for (let mi = 0; mi < members.length; mi++) {
            logSum += Math.log(Math.max(memberPreds[mi][si][y][x][c], 1e-15));
          }
          geoPred[y][x][c] = Math.exp(logSum / members.length);
        }
        // Normalize
        let sum = 0;
        for (let c = 0; c < C; c++) sum += geoPred[y][x][c];
        if (sum > 0) for (let c = 0; c < C; c++) geoPred[y][x][c] /= sum;
        else for (let c = 0; c < C; c++) geoPred[y][x][c] = 1 / C;
      }
    }
    const res = await POST('/submit', { round_id: r9.id, seed_index: si, prediction: geoPred });
    const score = res.data && res.data.score !== undefined ? res.data.score : '?';
    console.log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'} score=${score}`);
    await sleep(600);
  }

  // Check leaderboard
  console.log('\n=== FINAL LEADERBOARD CHECK ===');
  const lb = await GET('/leaderboard');
  if (lb && lb.data) {
    const top5 = lb.data.slice(0, 5);
    top5.forEach((t, i) => console.log(`#${i+1} ${t.team_name} ws=${t.weighted_score.toFixed(2)}`));
    const us = lb.data.find(t => t.team_name && t.team_name.includes('CAL'));
    if (us) {
      const rank = lb.data.indexOf(us) + 1;
      console.log(`\nUS: #${rank} ${us.team_name} ws=${us.weighted_score.toFixed(2)}`);
    }
  }

  console.log('\n=== ENSEMBLE SUBMISSION COMPLETE ===');
}

main().catch(e => console.error('Error:', e.message, e.stack));
