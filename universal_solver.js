#!/usr/bin/env node
/**
 * UNIVERSAL SOLVER v2 — Phased, adaptive, no auto-fire.
 *
 * Usage:
 *   node universal_solver.js <TOKEN> --status        # Show rounds + budget
 *   node universal_solver.js <TOKEN> --loo           # LOO cross-validation (offline)
 *   node universal_solver.js <TOKEN> --phase 0       # Analyze + submit baseline (0 queries)
 *   node universal_solver.js <TOKEN> --phase 1       # Test stochastic (2 queries)
 *   node universal_solver.js <TOKEN> --phase 2       # Recon (5-10 queries)
 *   node universal_solver.js <TOKEN> --phase 3       # Main (remaining queries)
 *   node universal_solver.js <TOKEN> --replay-all    # Replay all completed rounds
 */
'use strict';
const fs = require('fs'), path = require('path');
const API = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5, C = 6;
const DD = path.join(__dirname, 'data');
if (!fs.existsSync(DD)) fs.mkdirSync(DD, { recursive: true });

// === CLI ===
const TOKEN = process.argv[2];
if (!TOKEN || TOKEN.startsWith('--')) { console.error('Usage: node universal_solver.js <TOKEN> --phase N'); process.exit(1); }
const args = process.argv.slice(3);
function flag(n) { return args.includes('--' + n); }
function arg(n) { const i = args.indexOf('--' + n); return i >= 0 && i + 1 < args.length ? args[i + 1] : null; }
const PHASE = arg('phase') !== null ? parseInt(arg('phase')) : -1;

// === API ===
const headers = { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' };
async function api(endpoint, method = 'GET', body = null) {
  const opts = { method, headers };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(API + endpoint, opts);
  if (!res.ok) { const t = await res.text(); throw new Error(`${method} ${endpoint}: ${res.status} ${t}`); }
  return res.json();
}
function log(m) { console.log(`[${new Date().toISOString().slice(11, 19)}] ${m}`); }
function t2c(t) { return (t === 10 || t === 11 || t === 0) ? 0 : (t >= 1 && t <= 5) ? t : 0; }

// === FEATURE EXTRACTION (D0-D4) ===
function featureKeys(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS = 0, co = 0, fN = 0, sR2 = 0;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    if (!dy && !dx) continue; const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue; const nt = g[ny][nx];
    if (nt === 1 || nt === 2) nS++; if (nt === 10) co = 1; if (nt === 4) fN++;
  }
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    if (Math.abs(dy) <= 1 && Math.abs(dx) <= 1) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    if (g[ny][nx] === 1 || g[ny][nx] === 2) sR2++;
  }
  const sa = Math.min(nS, 5), sb2 = sR2 === 0 ? 0 : sR2 <= 2 ? 1 : sR2 <= 4 ? 2 : 3;
  const fb = fN <= 1 ? 0 : fN <= 3 ? 1 : 2;
  return {
    d0: `D0_${t}_${sa}_${co}_${sb2}_${fb}`, d1: `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`,
    d2: `D2_${t}_${sa > 0 ? 1 : 0}_${co}`, d3: `D3_${t}_${co}`, d4: `D4_${t}`
  };
}

// === REGIME FINGERPRINT ===
function regimeFingerprint(grid, isGT = false) {
  // Compute terrain fractions on land cells
  let sett = 0, ruin = 0, forest = 0, plains = 0, land = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (isGT) {
      // GT is probability distribution per cell
      const p = grid[y][x];
      // Skip ocean (p[0] > 0.99) and mountain (p[5] > 0.99)
      if (p[0] > 0.99 || p[5] > 0.99) continue;
      land++;
      sett += (p[1] || 0) + (p[2] || 0); // settlement + port
      ruin += p[3] || 0;
      forest += p[4] || 0;
      plains += p[0] || 0;
    } else {
      // Raw terrain grid from VP or initial state
      const t = grid[y][x];
      if (t === 10 || t === 5) continue; // skip ocean, mountain
      land++;
      const c = t2c(t);
      if (c === 1 || c === 2) sett++;
      else if (c === 3) ruin++;
      else if (c === 4) forest++;
      else if (c === 0) plains++;
    }
  }
  if (land === 0) return { sett: 0, ruin: 0, forest: 0, plains: 0 };
  return { sett: sett / land, ruin: ruin / land, forest: forest / land, plains: plains / land };
}

function regimeDistance(a, b) {
  return Math.sqrt((a.sett - b.sett) ** 2 + (a.ruin - b.ruin) ** 2 +
    (a.forest - b.forest) ** 2 + (a.plains - b.plains) ** 2);
}

function regimeWeights(currentFP, roundFPs, sigma = 0.05) {
  const weights = {};
  let minDist = Infinity;
  for (const [rn, fp] of Object.entries(roundFPs)) {
    const d = regimeDistance(currentFP, fp);
    weights[rn] = Math.exp(-d * d / (2 * sigma * sigma));
    if (d < minDist) minDist = d;
  }
  return { weights, minDist };
}

// === MODEL BUILDING ===
function buildModel(excludeRound, roundWeights) {
  const model = {};
  const levels = ['d0', 'd1', 'd2', 'd3', 'd4'];

  for (const level of levels) {
    for (let r = 1; r <= 20; r++) {
      const rn = `R${r}`;
      if (rn === excludeRound) continue;
      const ip = path.join(DD, `inits_${rn}.json`);
      const gp = path.join(DD, `gt_${rn}.json`);
      const rp = path.join(DD, `replays_${rn}.json`);
      if (!fs.existsSync(ip) || !fs.existsSync(gp)) continue;

      const rw = (roundWeights && roundWeights[rn] !== undefined) ? roundWeights[rn] : 1.0;
      if (rw < 0.001) continue; // skip irrelevant rounds

      const inits = JSON.parse(fs.readFileSync(ip));
      const gt = JSON.parse(fs.readFileSync(gp));
      let replays = [];
      if (fs.existsSync(rp)) replays = JSON.parse(fs.readFileSync(rp));

      for (let si = 0; si < SEEDS; si++) {
        if (!inits[si] || !gt[si]) continue;
        const g = Array.isArray(inits[si].grid) ? inits[si].grid : inits[si];
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = featureKeys(g, y, x); if (!keys) continue;
          const k = keys[level];
          if (!model[k]) model[k] = { n: 0, counts: new Array(C).fill(0) };
          for (let c = 0; c < C; c++) model[k].counts[c] += gt[si][y][x][c] * 20 * rw;
          model[k].n += 20 * rw;
        }
      }

      // Replay data
      for (const rep of replays) {
        const g = inits[rep.si] ? (Array.isArray(inits[rep.si].grid) ? inits[rep.si].grid : inits[rep.si]) : null;
        if (!g) continue;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = featureKeys(g, y, x); if (!keys) continue;
          const k = keys[level];
          if (!model[k]) model[k] = { n: 0, counts: new Array(C).fill(0) };
          model[k].n += rw; model[k].counts[t2c(rep.finalGrid[y][x])] += rw;
        }
      }
    }
  }

  // Normalize
  for (const k of Object.keys(model)) {
    const tot = model[k].counts.reduce((a, b) => a + b, 0) + C * 0.05;
    model[k].a = model[k].counts.map(v => (v + 0.05) / tot);
  }
  return model;
}

// === VP FUSION ===
function fuseVP(model, vpObs, inits, CW = 50) {
  // Pool VP observations across all seeds at D0 feature level
  const vpD0 = {};
  for (const obs of vpObs) {
    const g = Array.isArray(inits[obs.si].grid) ? inits[obs.si].grid : inits[obs.si];
    for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
      const gy = obs.vy + dy, gx = obs.vx + dx;
      if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
      const keys = featureKeys(g, gy, gx); if (!keys) continue;
      const k = keys.d0, fc = t2c(obs.grid[dy][dx]);
      if (!vpD0[k]) vpD0[k] = { n: 0, counts: new Array(C).fill(0) };
      vpD0[k].n++; vpD0[k].counts[fc]++;
    }
  }

  let fused = 0;
  for (const [k, vm] of Object.entries(vpD0)) {
    if (model[k]) {
      const pa = model[k].a.map(p => p * CW);
      const post = pa.map((a, c) => a + vm.counts[c]);
      const tot = post.reduce((a, b) => a + b, 0);
      model[k] = { n: model[k].n + vm.n, a: post.map(v => v / tot) };
      fused++;
    }
  }
  return fused;
}

// === PREDICTION ===
function predict(model, grid, TEMP = 0.9, FLOOR = 0.0001) {
  const pred = [];
  for (let y = 0; y < H; y++) {
    pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
      if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }
      const keys = featureKeys(grid, y, x);
      if (!keys) { pred[y][x] = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]; continue; }
      const lvls = ['d0', 'd1', 'd2', 'd3', 'd4'], ws = [1, 0.3, 0.15, 0.08, 0.02];
      const p = [0, 0, 0, 0, 0, 0]; let wS = 0;
      for (let li = 0; li < lvls.length; li++) {
        const d = model[keys[lvls[li]]];
        if (d && d.n >= 1) { const w = ws[li] * Math.pow(d.n, 0.5); for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w; }
      }
      if (wS === 0) { pred[y][x] = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]; continue; }
      let s = 0; for (let c = 0; c < C; c++) { p[c] = Math.pow(Math.max(p[c] / wS, 1e-10), 1 / TEMP); s += p[c]; }
      let s2 = 0; for (let c = 0; c < C; c++) { p[c] = Math.max(p[c] / s, FLOOR); s2 += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s2;
      pred[y][x] = p;
    }
  }
  return pred;
}

// === SCORING ===
function score(pred, gt) {
  let wklN = 0, wklD = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const gc = gt[y][x]; let ent = 0;
    for (let c = 0; c < C; c++) { if (gc[c] > 0.001) ent -= gc[c] * Math.log(gc[c]); }
    if (ent < 0.01) continue;
    let kl = 0;
    for (let c = 0; c < C; c++) { if (gc[c] > 0.001) kl += gc[c] * Math.log(gc[c] / Math.max(pred[y][x][c], 1e-10)); }
    wklN += ent * kl; wklD += ent;
  }
  const wkl = wklD > 0 ? wklN / wklD : 0;
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl)));
}

// === REPLAY COLLECTION ===
async function collectReplays(roundId, target = 500) {
  const cacheFile = path.join(DD, `replays_${roundId.slice(0, 8)}.json`);
  let replays = [];
  if (fs.existsSync(cacheFile)) replays = JSON.parse(fs.readFileSync(cacheFile));

  const perSeed = {};
  for (const r of replays) perSeed[r.si] = (perSeed[r.si] || 0) + 1;

  let needed = 0;
  for (let si = 0; si < SEEDS; si++) needed += Math.max(0, target - (perSeed[si] || 0));
  if (needed === 0) { log(`Already have ${replays.length} replays`); return replays; }

  log(`Collecting ${needed} more replays...`);
  const BATCH = 20;
  for (let si = 0; si < SEEDS; si++) {
    const have = perSeed[si] || 0;
    const want = Math.max(0, target - have);
    for (let i = 0; i < want; i += BATCH) {
      const batch = Math.min(BATCH, want - i);
      const proms = [];
      for (let j = 0; j < batch; j++) {
        proms.push(api('/replay', 'POST', { round_id: roundId, seed_index: si })
          .then(d => { if (d.frames) { const last = d.frames[d.frames.length - 1]; replays.push({ si, finalGrid: last.grid }); } })
          .catch(() => {}));
      }
      await Promise.all(proms);
      process.stdout.write(`\r  Seed ${si}: ${Math.min(have + i + batch, have + want)}/${target}`);
    }
    console.log();
  }

  fs.writeFileSync(cacheFile, JSON.stringify(replays));
  log(`Saved ${replays.length} replays`);
  return replays;
}

// === REPLAY-BASED PREDICTION ===
function buildReplayPreds(inits, replays) {
  const preds = [];
  for (let si = 0; si < SEEDS; si++) {
    const g = Array.isArray(inits[si].grid) ? inits[si].grid : inits[si];
    const counts = Array.from({ length: H }, () => Array.from({ length: W }, () => new Array(C).fill(0)));
    let nSeed = 0;
    for (const r of replays) {
      if (r.si !== si) continue;
      nSeed++;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) counts[y][x][t2c(r.finalGrid[y][x])]++;
    }
    const baseAlpha = Math.max(0.02, 0.15 * Math.sqrt(150 / Math.max(nSeed, 1)));
    const pred = [];
    for (let y = 0; y < H; y++) {
      pred[y] = [];
      for (let x = 0; x < W; x++) {
        if (g[y][x] === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
        if (g[y][x] === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }
        const total = counts[y][x].reduce((a, b) => a + b, 0);
        const nonzero = counts[y][x].filter(v => v > 0).length;
        let alpha = baseAlpha;
        if (nonzero <= 1) alpha = 0.001;
        else if (nonzero <= 2) alpha = 0.004;
        const denom = total + C * alpha;
        const p = counts[y][x].map(v => (v + alpha) / denom);
        const s = p.reduce((a, b) => a + b, 0);
        pred[y][x] = p.map(v => v / s);
      }
    }
    preds.push(pred);
  }
  return preds;
}

// === SMART VP PLACEMENT ===
function planVPPositions(grid, n) {
  // Find non-ocean, non-mountain cells (dynamic cells that matter for scoring)
  const dynamic = Array.from({ length: H }, () => new Uint8Array(W));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (grid[y][x] !== 10 && grid[y][x] !== 5) dynamic[y][x] = 1;
  }

  // Score each valid viewport position by how many dynamic cells it covers
  const positions = [];
  const covered = Array.from({ length: H }, () => new Uint8Array(W));

  for (let i = 0; i < n; i++) {
    let bestScore = -1, bestY = 0, bestX = 0;
    for (let vy = 0; vy <= H - 15; vy++) for (let vx = 0; vx <= W - 15; vx++) {
      let sc = 0;
      for (let dy = 0; dy < 15; dy++) for (let dx = 0; dx < 15; dx++) {
        if (dynamic[vy + dy][vx + dx] && !covered[vy + dy][vx + dx]) sc++;
      }
      if (sc > bestScore) { bestScore = sc; bestY = vy; bestX = vx; }
    }
    positions.push({ y: bestY, x: bestX, dynamicCells: bestScore });
    for (let dy = 0; dy < 15; dy++) for (let dx = 0; dx < 15; dx++) covered[bestY + dy][bestX + dx] = 1;
  }
  return positions;
}

// === SUBMIT ===
async function submitPreds(roundId, preds) {
  for (let si = 0; si < SEEDS; si++) {
    const res = await api('/submit', 'POST', { round_id: roundId, seed_index: si, prediction: preds[si] });
    log(`  Seed ${si}: ${res.status || 'accepted'}`);
  }
}

// === SAVE ROUND DATA ===
async function saveRoundData(rounds) {
  for (let i = 0; i < rounds.length; i++) {
    const r = rounds[i], rn = `R${i + 1}`;
    const ip = path.join(DD, `inits_${rn}.json`);
    const gp = path.join(DD, `gt_${rn}.json`);

    if (!fs.existsSync(ip)) {
      try {
        const detail = await api(`/rounds/${r.id}`);
        if (detail.initial_states) { fs.writeFileSync(ip, JSON.stringify(detail.initial_states)); log(`Saved ${rn} inits`); }
      } catch {}
    }

    if (r.status === 'completed' && !fs.existsSync(gp)) {
      try {
        const gt = [];
        for (let si = 0; si < SEEDS; si++) {
          const d = await api(`/analysis/${r.id}/${si}`);
          gt.push(d.ground_truth);
        }
        fs.writeFileSync(gp, JSON.stringify(gt));
        log(`Saved ${rn} GT`);
      } catch {}
    }
  }
}

// === MAIN ===
async function main() {
  const rounds = await api('/rounds');
  const completed = rounds.filter(r => r.status === 'completed');
  const active = rounds.find(r => r.status === 'active');

  // --status
  if (flag('status')) {
    log(`${rounds.length} rounds: ${completed.length} completed, ${active ? '1 active' : '0 active'}`);
    if (active) {
      const budget = await api(`/budget?round_id=${active.id}`).catch(() => null);
      log(`Active: R${rounds.indexOf(active) + 1} (${active.id.slice(0, 8)}) budget: ${budget ? budget.queries_used + '/' + budget.queries_max : '?'}`);
    }
    try {
      const my = await api('/my-rounds');
      for (const r of my) if (r.score !== undefined) log(`  R${r.round_number || '?'}: score=${r.score}`);
    } catch {}
    return;
  }

  // Save any missing data
  await saveRoundData(rounds);

  // --loo: LOO cross-validation of regime-weighted model
  if (flag('loo')) {
    log('=== LOO Cross-Validation (regime-weighted) ===');
    // Precompute regime fingerprints from GT
    const fps = {};
    for (let r = 1; r <= rounds.length; r++) {
      const gp = path.join(DD, `gt_R${r}.json`);
      if (!fs.existsSync(gp)) continue;
      const gt = JSON.parse(fs.readFileSync(gp));
      // Average fingerprint across 5 seeds
      const seedFPs = gt.map(g => regimeFingerprint(g, true));
      fps[`R${r}`] = {
        sett: seedFPs.reduce((a, f) => a + f.sett, 0) / SEEDS,
        ruin: seedFPs.reduce((a, f) => a + f.ruin, 0) / SEEDS,
        forest: seedFPs.reduce((a, f) => a + f.forest, 0) / SEEDS,
        plains: seedFPs.reduce((a, f) => a + f.plains, 0) / SEEDS,
      };
      log(`  R${r}: sett=${fps[`R${r}`].sett.toFixed(3)} ruin=${fps[`R${r}`].ruin.toFixed(3)} forest=${fps[`R${r}`].forest.toFixed(3)} plains=${fps[`R${r}`].plains.toFixed(3)}`);
    }

    const sigmas = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 999]; // 999 = equal weights
    for (const sigma of sigmas) {
      const allScores = [];
      for (const testRn of Object.keys(fps)) {
        if (!fs.existsSync(path.join(DD, `inits_${testRn}.json`))) continue;
        const inits = JSON.parse(fs.readFileSync(path.join(DD, `inits_${testRn}.json`)));
        const gt = JSON.parse(fs.readFileSync(path.join(DD, `gt_${testRn}.json`)));

        // Compute regime weights (test round's FP vs all other rounds)
        const otherFPs = {}; for (const [k, v] of Object.entries(fps)) { if (k !== testRn) otherFPs[k] = v; }
        const { weights } = regimeWeights(fps[testRn], otherFPs, sigma);

        const model = buildModel(testRn, weights);
        const TEMP = sigma === 999 ? 0.9 : 0.9; // same temp for fair comparison
        const seeds = [];
        for (let si = 0; si < SEEDS; si++) {
          if (!inits[si] || !gt[si]) continue;
          const g = Array.isArray(inits[si].grid) ? inits[si].grid : inits[si];
          const p = predict(model, g, TEMP);
          seeds.push(score(p, gt[si]));
        }
        const avg = seeds.reduce((a, b) => a + b, 0) / seeds.length;
        allScores.push({ rn: testRn, avg });
      }
      const overall = allScores.reduce((a, s) => a + s.avg, 0) / allScores.length;
      const min = Math.min(...allScores.map(s => s.avg));
      const max = Math.max(...allScores.map(s => s.avg));
      log(`sigma=${sigma === 999 ? 'equal' : sigma}: avg=${overall.toFixed(1)} min=${min.toFixed(1)} max=${max.toFixed(1)} | ${allScores.map(s => s.rn + '=' + s.avg.toFixed(0)).join(' ')}`);
    }
    return;
  }

  // --replay-all: replay all completed rounds
  if (flag('replay-all')) {
    for (let i = 0; i < completed.length; i++) {
      const r = completed[i];
      const rn = rounds.indexOf(r) + 1;
      log(`\n=== R${rn} (${r.id.slice(0, 8)}) ===`);
      const ip = path.join(DD, `inits_R${rn}.json`);
      if (!fs.existsSync(ip)) { log('No inits, skip'); continue; }
      const inits = JSON.parse(fs.readFileSync(ip));
      const replays = await collectReplays(r.id, 500);
      const preds = buildReplayPreds(inits, replays);
      log(`Built replay predictions (${replays.length} replays)`);
      await submitPreds(r.id, preds);
    }
    return;
  }

  // === PHASED ACTIVE ROUND STRATEGY ===
  if (PHASE < 0) { log('Specify --phase N or --loo or --status or --replay-all'); return; }
  if (!active) { log('No active round'); return; }

  const roundId = active.id;
  const roundNum = rounds.indexOf(active) + 1;
  const detail = await api(`/rounds/${roundId}`);
  const inits = detail.initial_states;
  log(`Active: R${roundNum} (${roundId.slice(0, 8)})`);

  const budget = await api(`/budget?round_id=${roundId}`).catch(() => ({ queries_used: '?', queries_max: '?' }));
  log(`Budget: ${budget.queries_used}/${budget.queries_max}`);

  // Load saved VP data
  const vpFile = path.join(DD, `viewport_${roundId.slice(0, 8)}.json`);
  let vpObs = fs.existsSync(vpFile) ? JSON.parse(fs.readFileSync(vpFile)) : [];
  log(`Existing VP observations: ${vpObs.length}`);

  // === PHASE 0: Analyze + submit baseline ===
  if (PHASE === 0) {
    log('\n=== PHASE 0: Analysis + Baseline (0 queries) ===');

    // Analyze initial states
    for (let si = 0; si < SEEDS; si++) {
      const g = inits[si].grid;
      let sett = 0, port = 0, forest = 0, ocean = 0, mtn = 0, plains = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const t = g[y][x];
        if (t === 10) ocean++; else if (t === 5) mtn++; else if (t === 4) forest++;
        else if (t === 1) sett++; else if (t === 2) port++; else if (t === 11) plains++;
      }
      log(`  Seed ${si}: ${sett}S ${port}P ${forest}F ${ocean}O ${mtn}M ${plains}pl | dynamic=${H * W - ocean - mtn}`);
    }

    // Build equal-weight model + submit baseline
    const model = buildModel(null, null);
    log(`Model: ${Object.keys(model).length} keys`);

    // VP fusion if we have VP data
    if (vpObs.length > 0) {
      const fused = fuseVP(model, vpObs, inits);
      log(`VP fused: ${fused} D0 keys`);
    }

    const preds = [];
    for (let si = 0; si < SEEDS; si++) {
      const g = Array.isArray(inits[si].grid) ? inits[si].grid : inits[si];
      preds.push(predict(model, g));
    }
    log('Submitting baseline...');
    await submitPreds(roundId, preds);
    log('PHASE 0 DONE. Run --phase 1 next (costs 2 queries).');
    return;
  }

  // === PHASE 1: Test if /simulate is stochastic ===
  if (PHASE === 1) {
    log('\n=== PHASE 1: Stochastic Test (2 queries) ===');

    // Find a position with dynamic cells
    const g = inits[0].grid;
    const pos = planVPPositions(g, 1)[0];
    log(`Testing position (${pos.y}, ${pos.x}) on seed 0, ${pos.dynamicCells} dynamic cells`);

    // Query same position twice
    const r1 = await api('/simulate', 'POST', {
      round_id: roundId, seed_index: 0,
      viewport_y: pos.y, viewport_x: pos.x, viewport_h: 15, viewport_w: 15
    });
    log('Query 1 done');

    const r2 = await api('/simulate', 'POST', {
      round_id: roundId, seed_index: 0,
      viewport_y: pos.y, viewport_x: pos.x, viewport_h: 15, viewport_w: 15
    });
    log('Query 2 done');

    // Compare
    let same = 0, diff = 0;
    for (let dy = 0; dy < 15; dy++) for (let dx = 0; dx < 15; dx++) {
      if (r1.grid[dy][dx] === r2.grid[dy][dx]) same++; else diff++;
    }

    const stochastic = diff > 0;
    log(`Result: ${same} same, ${diff} different → ${stochastic ? 'STOCHASTIC!' : 'DETERMINISTIC'}`);

    // Save both observations
    vpObs.push({ si: 0, vy: pos.y, vx: pos.x, grid: r1.grid });
    vpObs.push({ si: 0, vy: pos.y, vx: pos.x, grid: r2.grid });
    fs.writeFileSync(vpFile, JSON.stringify(vpObs));

    // Save stochastic flag
    fs.writeFileSync(path.join(DD, `stochastic_${roundId.slice(0, 8)}.json`), JSON.stringify({ stochastic, same, diff }));

    if (stochastic) {
      log('\n*** STOCHASTIC! Each query is an independent sample. ***');
      log('Strategy: Query same positions repeatedly to build distributions.');
      log('Run --phase 2 for recon (5 queries), then --phase 3 for remaining.');
    } else {
      log('\n*** DETERMINISTIC. Single observation per cell. ***');
      log('Strategy: Maximize coverage of dynamic cells + regime-weighted model.');
      log('Run --phase 2 for recon (5 queries), then --phase 3 for remaining.');
    }
    return;
  }

  // === PHASE 2: Recon ===
  if (PHASE === 2) {
    log('\n=== PHASE 2: Recon ===');

    const stochFile = path.join(DD, `stochastic_${roundId.slice(0, 8)}.json`);
    const stochastic = fs.existsSync(stochFile) ? JSON.parse(fs.readFileSync(stochFile)).stochastic : false;
    log(`Mode: ${stochastic ? 'STOCHASTIC' : 'DETERMINISTIC'}`);

    const queriesPerSeed = stochastic ? 2 : 1; // stochastic: 2 per seed (same pos), deterministic: 1 per seed
    log(`Using ${queriesPerSeed} queries per seed (${queriesPerSeed * SEEDS} total)`);

    for (let si = 0; si < SEEDS; si++) {
      const g = inits[si].grid;
      const pos = planVPPositions(g, 1)[0];
      for (let q = 0; q < queriesPerSeed; q++) {
        const r = await api('/simulate', 'POST', {
          round_id: roundId, seed_index: si,
          viewport_y: pos.y, viewport_x: pos.x, viewport_h: 15, viewport_w: 15
        });
        vpObs.push({ si, vy: pos.y, vx: pos.x, grid: r.grid });
        log(`  Seed ${si} query ${q + 1}: pos=(${pos.y},${pos.x})`);
      }
    }
    fs.writeFileSync(vpFile, JSON.stringify(vpObs));
    log(`Total VP observations: ${vpObs.length}`);

    // Compute regime fingerprint from VP observations
    // Build a combined grid from VP observations for fingerprinting
    for (let si = 0; si < SEEDS; si++) {
      const seedObs = vpObs.filter(o => o.si === si);
      const combined = Array.from({ length: H }, () => new Array(W).fill(-1));
      for (const obs of seedObs) {
        for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
          const gy = obs.vy + dy, gx = obs.vx + dx;
          if (gy < H && gx < W) combined[gy][gx] = obs.grid[dy][dx];
        }
      }
      // Count terrain types in observed cells
      let sett = 0, ruin = 0, forest = 0, plains = 0, land = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        if (combined[y][x] === -1) continue;
        const t = combined[y][x];
        if (t === 10 || t === 5) continue;
        land++;
        const c = t2c(t);
        if (c === 1 || c === 2) sett++; else if (c === 3) ruin++; else if (c === 4) forest++; else plains++;
      }
      if (land > 0) log(`  Seed ${si} regime: sett=${(sett / land).toFixed(3)} ruin=${(ruin / land).toFixed(3)} forest=${(forest / land).toFixed(3)} plains=${(plains / land).toFixed(3)}`);
    }

    // Rebuild model with regime weighting + VP fusion, resubmit
    // (compute overall VP fingerprint)
    const allCombined = Array.from({ length: H }, () => new Array(W).fill(-1));
    for (const obs of vpObs) {
      for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
        const gy = obs.vy + dy, gx = obs.vx + dx;
        if (gy < H && gx < W) allCombined[gy][gx] = obs.grid[dy][dx];
      }
    }
    const currentFP = regimeFingerprint(allCombined, false);
    log(`Overall regime: sett=${currentFP.sett.toFixed(3)} ruin=${currentFP.ruin.toFixed(3)} forest=${currentFP.forest.toFixed(3)} plains=${currentFP.plains.toFixed(3)}`);

    // Load historical fingerprints
    const histFPs = {};
    for (let r = 1; r <= rounds.length; r++) {
      const gp = path.join(DD, `gt_R${r}.json`);
      if (!fs.existsSync(gp)) continue;
      const gt = JSON.parse(fs.readFileSync(gp));
      const seedFPs = gt.map(g => regimeFingerprint(g, true));
      histFPs[`R${r}`] = {
        sett: seedFPs.reduce((a, f) => a + f.sett, 0) / SEEDS,
        ruin: seedFPs.reduce((a, f) => a + f.ruin, 0) / SEEDS,
        forest: seedFPs.reduce((a, f) => a + f.forest, 0) / SEEDS,
        plains: seedFPs.reduce((a, f) => a + f.plains, 0) / SEEDS,
      };
    }

    const { weights, minDist } = regimeWeights(currentFP, histFPs, 0.05);
    const sorted = Object.entries(weights).sort((a, b) => b[1] - a[1]);
    log(`Regime weights (top 5): ${sorted.slice(0, 5).map(([k, w]) => `${k}=${w.toFixed(3)}`).join(' ')}`);
    log(`Min distance: ${minDist.toFixed(4)}`);

    const model = buildModel(null, weights);
    const fused = fuseVP(model, vpObs, inits);
    const TEMP = 0.9 + 0.3 * (minDist / 0.05);
    log(`Model: ${Object.keys(model).length} keys, VP fused: ${fused}, temp: ${TEMP.toFixed(2)}`);

    const preds = [];
    for (let si = 0; si < SEEDS; si++) {
      const g = Array.isArray(inits[si].grid) ? inits[si].grid : inits[si];
      preds.push(predict(model, g, TEMP));
    }
    log('Submitting regime-weighted predictions...');
    await submitPreds(roundId, preds);

    const budgetNow = await api(`/budget?round_id=${roundId}`).catch(() => null);
    log(`Budget remaining: ${budgetNow ? (budgetNow.queries_max - budgetNow.queries_used) : '?'}`);
    log('PHASE 2 DONE. Run --phase 3 to use remaining queries.');
    return;
  }

  // === PHASE 3: Main — spend remaining queries ===
  if (PHASE === 3) {
    log('\n=== PHASE 3: Main ===');

    const stochFile = path.join(DD, `stochastic_${roundId.slice(0, 8)}.json`);
    const stochastic = fs.existsSync(stochFile) ? JSON.parse(fs.readFileSync(stochFile)).stochastic : false;
    log(`Mode: ${stochastic ? 'STOCHASTIC' : 'DETERMINISTIC'}`);

    const budgetInfo = await api(`/budget?round_id=${roundId}`).catch(() => null);
    const remaining = budgetInfo ? budgetInfo.queries_max - budgetInfo.queries_used : 0;
    log(`Remaining queries: ${remaining}`);
    if (remaining <= 0) { log('No queries left'); }

    if (remaining > 0) {
      const perSeed = Math.floor(remaining / SEEDS);

      if (stochastic) {
        // STOCHASTIC: query same dynamic-heavy positions repeatedly
        for (let si = 0; si < SEEDS; si++) {
          const g = inits[si].grid;
          // Pick 2-3 positions that cover the most dynamic cells
          const nPositions = Math.min(3, Math.ceil(perSeed / 2));
          const positions = planVPPositions(g, nPositions);
          const queriesPerPos = Math.floor(perSeed / nPositions);
          for (const pos of positions) {
            for (let q = 0; q < queriesPerPos; q++) {
              try {
                const r = await api('/simulate', 'POST', {
                  round_id: roundId, seed_index: si,
                  viewport_y: pos.y, viewport_x: pos.x, viewport_h: 15, viewport_w: 15
                });
                vpObs.push({ si, vy: pos.y, vx: pos.x, grid: r.grid });
              } catch (e) { log(`  Query error: ${e.message}`); break; }
            }
            log(`  Seed ${si} pos=(${pos.y},${pos.x}): ${queriesPerPos} queries`);
          }
        }
      } else {
        // DETERMINISTIC: maximize unique cell coverage
        for (let si = 0; si < SEEDS; si++) {
          const g = inits[si].grid;
          // Find already-observed cells
          const covered = Array.from({ length: H }, () => new Uint8Array(W));
          for (const obs of vpObs.filter(o => o.si === si)) {
            for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
              const gy = obs.vy + dy, gx = obs.vx + dx;
              if (gy < H && gx < W) covered[gy][gx] = 1;
            }
          }
          // Plan new positions avoiding already-covered cells
          const positions = planVPPositions(g, perSeed);
          for (const pos of positions) {
            try {
              const r = await api('/simulate', 'POST', {
                round_id: roundId, seed_index: si,
                viewport_y: pos.y, viewport_x: pos.x, viewport_h: 15, viewport_w: 15
              });
              vpObs.push({ si, vy: pos.y, vx: pos.x, grid: r.grid });
              log(`  Seed ${si} pos=(${pos.y},${pos.x})`);
            } catch (e) { log(`  Query error: ${e.message}`); break; }
          }
        }
      }

      fs.writeFileSync(vpFile, JSON.stringify(vpObs));
      log(`Total VP observations: ${vpObs.length}`);
    }

    // Build final predictions
    if (stochastic && vpObs.length >= 20) {
      // STOCHASTIC PATH: build per-cell distributions from VP samples
      log('Building VP-based distributions (stochastic path)...');
      // Aggregate VP observations like replays
      const preds = [];
      for (let si = 0; si < SEEDS; si++) {
        const g = Array.isArray(inits[si].grid) ? inits[si].grid : inits[si];
        const counts = Array.from({ length: H }, () => Array.from({ length: W }, () => new Array(C).fill(0)));
        const nObs = Array.from({ length: H }, () => new Uint32Array(W));
        for (const obs of vpObs.filter(o => o.si === si)) {
          for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
            const gy = obs.vy + dy, gx = obs.vx + dx;
            if (gy < H && gx < W) { counts[gy][gx][t2c(obs.grid[dy][dx])]++; nObs[gy][gx]++; }
          }
        }

        // Also build regime-weighted model as fallback for unobserved cells
        const allCombined = Array.from({ length: H }, () => new Array(W).fill(-1));
        for (const obs of vpObs) for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
          const gy = obs.vy + dy, gx = obs.vx + dx;
          if (gy < H && gx < W) allCombined[gy][gx] = obs.grid[dy][dx];
        }
        const currentFP = regimeFingerprint(allCombined, false);
        const histFPs = {};
        for (let r = 1; r <= rounds.length; r++) {
          const gp = path.join(DD, `gt_R${r}.json`);
          if (!fs.existsSync(gp)) continue;
          const gt = JSON.parse(fs.readFileSync(gp));
          const seedFPs = gt.map(g2 => regimeFingerprint(g2, true));
          histFPs[`R${r}`] = { sett: seedFPs.reduce((a, f) => a + f.sett, 0) / SEEDS, ruin: seedFPs.reduce((a, f) => a + f.ruin, 0) / SEEDS, forest: seedFPs.reduce((a, f) => a + f.forest, 0) / SEEDS, plains: seedFPs.reduce((a, f) => a + f.plains, 0) / SEEDS };
        }
        const { weights, minDist } = regimeWeights(currentFP, histFPs, 0.05);
        const fallbackModel = buildModel(null, weights);
        fuseVP(fallbackModel, vpObs, inits);
        const TEMP = 0.9 + 0.3 * (minDist / 0.05);

        const pred = [];
        for (let y = 0; y < H; y++) {
          pred[y] = [];
          for (let x = 0; x < W; x++) {
            if (g[y][x] === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
            if (g[y][x] === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }

            if (nObs[y][x] >= 3) {
              // Enough VP samples to build distribution
              const total = nObs[y][x];
              const alpha = Math.max(0.02, 0.15 * Math.sqrt(150 / total));
              const nonzero = counts[y][x].filter(v => v > 0).length;
              const cellAlpha = nonzero <= 1 ? 0.001 : nonzero <= 2 ? 0.004 : alpha;
              const denom = total + C * cellAlpha;
              const p = counts[y][x].map(v => (v + cellAlpha) / denom);
              const s = p.reduce((a, b) => a + b, 0);
              pred[y][x] = p.map(v => v / s);
            } else {
              // Fall back to model
              const keys = featureKeys(g, y, x);
              if (!keys) { pred[y][x] = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]; continue; }
              const lvls = ['d0', 'd1', 'd2', 'd3', 'd4'], ws2 = [1, 0.3, 0.15, 0.08, 0.02];
              const p = [0, 0, 0, 0, 0, 0]; let wS = 0;
              for (let li = 0; li < lvls.length; li++) {
                const d = fallbackModel[keys[lvls[li]]];
                if (d && d.n >= 1) { const w = ws2[li] * Math.pow(d.n, 0.5); for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w; }
              }
              if (wS === 0) { pred[y][x] = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]; continue; }
              let s = 0; for (let c = 0; c < C; c++) { p[c] = Math.pow(Math.max(p[c] / wS, 1e-10), 1 / TEMP); s += p[c]; }
              let s2 = 0; for (let c = 0; c < C; c++) { p[c] = Math.max(p[c] / s, 0.0001); s2 += p[c]; }
              for (let c = 0; c < C; c++) p[c] /= s2;
              pred[y][x] = p;
            }
          }
        }
        preds.push(pred);
      }
      log('Submitting VP distribution predictions...');
      await submitPreds(roundId, preds);
    } else {
      // DETERMINISTIC PATH: regime-weighted model + VP fusion
      const allCombined = Array.from({ length: H }, () => new Array(W).fill(-1));
      for (const obs of vpObs) for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
        const gy = obs.vy + dy, gx = obs.vx + dx;
        if (gy < H && gx < W) allCombined[gy][gx] = obs.grid[dy][dx];
      }
      const currentFP = regimeFingerprint(allCombined, false);
      const histFPs = {};
      for (let r = 1; r <= rounds.length; r++) {
        const gp = path.join(DD, `gt_R${r}.json`);
        if (!fs.existsSync(gp)) continue;
        const gt = JSON.parse(fs.readFileSync(gp));
        const seedFPs = gt.map(g2 => regimeFingerprint(g2, true));
        histFPs[`R${r}`] = { sett: seedFPs.reduce((a, f) => a + f.sett, 0) / SEEDS, ruin: seedFPs.reduce((a, f) => a + f.ruin, 0) / SEEDS, forest: seedFPs.reduce((a, f) => a + f.forest, 0) / SEEDS, plains: seedFPs.reduce((a, f) => a + f.plains, 0) / SEEDS };
      }
      const { weights, minDist } = regimeWeights(currentFP, histFPs, 0.05);
      log(`Regime weights: ${Object.entries(weights).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([k, w]) => `${k}=${w.toFixed(3)}`).join(' ')}`);

      const model = buildModel(null, weights);
      const fused = fuseVP(model, vpObs, inits);
      const TEMP = 0.9 + 0.3 * (minDist / 0.05);
      log(`Model: ${Object.keys(model).length} keys, VP fused: ${fused}, temp: ${TEMP.toFixed(2)}`);

      const preds = [];
      for (let si = 0; si < SEEDS; si++) {
        const g = Array.isArray(inits[si].grid) ? inits[si].grid : inits[si];
        preds.push(predict(model, g, TEMP));
      }
      log('Submitting regime-weighted + VP predictions...');
      await submitPreds(roundId, preds);
    }

    log('PHASE 3 DONE.');
    return;
  }
}

main().catch(e => { console.error('FATAL:', e.message); process.exit(1); });
