#!/usr/bin/env node
/**
 * SUBMIT V4 — Feature-Aggregated VP + Cross-Round Model
 *
 * KEY INSIGHT: 50 VP queries give 1-5 obs/cell. Noisy.
 * But if we group cells by FEATURE KEY (terrain+adjacency+distance),
 * we get 30-100+ observations per feature bucket!
 *
 * Hidden params are SAME for all 5 seeds → VP data from any seed
 * informs the feature model for ALL seeds.
 *
 * This gives us the statistical power of 5000+ observations
 * from just 50 queries.
 *
 * Usage: node submit_v4.js --token YOUR_JWT [--dry-run]
 */
'use strict';

const https = require('https');
const fs = require('fs');
const path = require('path');

const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5, CLASSES = 6;
const DATA_DIR = path.join(__dirname, 'data');

const args = process.argv.slice(2);
function arg(name) { const i = args.indexOf('--' + name); return i >= 0 && i + 1 < args.length ? args[i + 1] : null; }
function flag(name) { return args.indexOf('--' + name) >= 0; }

const TOKEN = arg('token') || process.env.ASTAR_TOKEN;
const DRY_RUN = flag('dry-run');
if (!TOKEN) { console.error('Usage: node submit_v4.js --token YOUR_JWT'); process.exit(1); }

function api(method, endpoint, body) {
  return new Promise((resolve, reject) => {
    const url = new URL(BASE + endpoint);
    const opts = { hostname: url.hostname, path: url.pathname + url.search, method,
      headers: { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' } };
    const req = https.request(opts, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => { try { resolve({ status: res.statusCode, data: JSON.parse(data) }); } catch { resolve({ status: res.statusCode, data }); } });
    });
    req.on('error', reject);
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function log(msg) { console.log(`[${new Date().toISOString().slice(11, 19)}] ${msg}`); }
function t2c(t) { return (t === 10 || t === 11 || t === 0) ? 0 : ((t >= 1 && t <= 5) ? t : 0); }

// ═══════════════════════════════════════════════════════════════════════════
// FEATURE EXTRACTION
// ═══════════════════════════════════════════════════════════════════════════
function precomputeSettlements(grid) {
  const setts = [];
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const c = t2c(grid[y][x]);
      if (c === 1 || c === 2) setts.push([y, x]);
    }
  return setts;
}

function extractFeatures(grid, y, x, settlements) {
  const terrain = t2c(grid[y][x]);
  const raw = grid[y][x];
  const isOcean = raw === 10;
  const isMountain = terrain === 5;

  let nSett1 = 0, nForest1 = 0, nOcean1 = 0, nPort1 = 0, nMtn1 = 0;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    if (!dy && !dx) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    const nc = t2c(grid[ny][nx]);
    if (nc === 1) nSett1++;
    if (nc === 2) nPort1++;
    if (nc === 4) nForest1++;
    if (grid[ny][nx] === 10) nOcean1++;
    if (nc === 5) nMtn1++;
  }
  const coastal = nOcean1 > 0 ? 1 : 0;

  let nSett2 = 0;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    if (Math.abs(dy) <= 1 && Math.abs(dx) <= 1) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    const nc = t2c(grid[ny][nx]);
    if (nc === 1 || nc === 2) nSett2++;
  }

  let minDist = 999;
  if (settlements) for (const [sy, sx] of settlements) {
    const d = Math.abs(y - sy) + Math.abs(x - sx);
    if (d < minDist) minDist = d;
  }
  const db = minDist === 0 ? 0 : minDist <= 1 ? 1 : minDist <= 2 ? 2 : minDist <= 3 ? 3 : minDist <= 5 ? 4 : minDist <= 8 ? 5 : 6;
  const fD = nForest1 >= 4 ? 'H' : nForest1 >= 2 ? 'M' : nForest1 >= 1 ? 'L' : '0';

  return {
    terrain, isOcean, isMountain,
    D0: `${terrain}_s${nSett1}_c${coastal}_f${fD}_d${db}_p${nPort1}_r${Math.min(nSett2, 3)}`,
    D1: `${terrain}_s${nSett1}_c${coastal}_f${fD}_d${db}`,
    D2: `${terrain}_s${Math.min(nSett1, 2)}_c${coastal}_d${Math.min(db, 4)}`,
    D3: `${terrain}_s${Math.min(nSett1, 2)}_d${Math.min(db, 3)}`,
    D4: `${terrain}_d${Math.min(db, 2)}`,
    D5: `${terrain}`,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// CROSS-ROUND MODEL
// ═══════════════════════════════════════════════════════════════════════════
function buildCrossRoundModel(trainingData) {
  const levels = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5'];
  const model = {};
  for (const lvl of levels) model[lvl] = {};

  for (const td of trainingData) {
    const settlements = precomputeSettlements(td.grid);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const feat = extractFeatures(td.grid, y, x, settlements);
      const gt = td.gt[y][x];
      let ent = 0;
      for (let c = 0; c < CLASSES; c++) if (gt[c] > 1e-6) ent -= gt[c] * Math.log(gt[c]);

      for (const lvl of levels) {
        const key = feat[lvl];
        if (!model[lvl][key]) model[lvl][key] = { dist: new Float64Array(CLASSES), weight: 0, count: 0 };
        const e = model[lvl][key];
        const w = Math.max(ent, 0.01);
        for (let c = 0; c < CLASSES; c++) e.dist[c] += gt[c] * w;
        e.weight += w;
        e.count++;
      }
    }
  }
  for (const lvl of levels) for (const key in model[lvl]) {
    const e = model[lvl][key];
    if (e.weight > 0) for (let c = 0; c < CLASSES; c++) e.dist[c] /= e.weight;
  }
  return model;
}

// ═══════════════════════════════════════════════════════════════════════════
// FEATURE-AGGREGATED VP MODEL
// Build a model from VP observations grouped by feature key
// This turns 50 queries into thousands of effective observations
// ═══════════════════════════════════════════════════════════════════════════
function buildVPFeatureModel(vpData, initialStates) {
  const levels = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5'];
  const vpModel = {};
  for (const lvl of levels) vpModel[lvl] = {};

  // Aggregate VP observations across ALL seeds (hidden params are same)
  for (let s = 0; s < SEEDS; s++) {
    const grid = initialStates[s].grid;
    const settlements = precomputeSettlements(grid);

    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const nObs = vpData[s].nObs[y][x];
      if (nObs === 0) continue;

      const feat = extractFeatures(grid, y, x, settlements);
      if (feat.isOcean || feat.isMountain) continue;

      for (const lvl of levels) {
        const key = feat[lvl];
        if (!vpModel[lvl][key]) vpModel[lvl][key] = { counts: new Float64Array(CLASSES), total: 0 };
        const e = vpModel[lvl][key];
        for (let c = 0; c < CLASSES; c++) e.counts[c] += vpData[s].counts[y][x][c];
        e.total += nObs;
      }
    }
  }

  // Log stats
  for (const lvl of levels) {
    const keys = Object.keys(vpModel[lvl]);
    const avgObs = keys.length > 0
      ? keys.reduce((s, k) => s + vpModel[lvl][k].total, 0) / keys.length
      : 0;
    log(`  VP ${lvl}: ${keys.length} keys, avg ${avgObs.toFixed(1)} obs/key`);
  }

  return vpModel;
}

// Predict using VP feature model with Dirichlet smoothing
function vpFeaturePredict(vpModel, features, alpha) {
  const levels = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5'];
  // Use most specific level with enough data, fallback to coarser
  for (const lvl of levels) {
    const key = features[lvl];
    const e = vpModel[lvl][key];
    if (e && e.total >= 3) { // need at least 3 observations
      const pred = [];
      const total = e.total + CLASSES * alpha;
      for (let c = 0; c < CLASSES; c++) pred[c] = (e.counts[c] + alpha) / total;
      return { pred, level: lvl, nObs: e.total };
    }
  }
  return null; // no VP data for this feature
}

// Cross-round model prediction
function crossRoundPredict(model, features, regW) {
  const weights = regW || [1.0, 0.2, 0.05, 0.02, 0.01, 0.005];
  const levels = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5'];
  const pred = new Float64Array(CLASSES);
  let totalWeight = 0;

  for (let i = 0; i < levels.length; i++) {
    const key = features[levels[i]];
    const entry = model[levels[i]][key];
    if (entry && entry.count >= 1) {
      const w = weights[i] * Math.sqrt(entry.count);
      for (let c = 0; c < CLASSES; c++) pred[c] += entry.dist[c] * w;
      totalWeight += w;
    }
  }
  if (totalWeight > 0) for (let c = 0; c < CLASSES; c++) pred[c] /= totalWeight;
  else for (let c = 0; c < CLASSES; c++) pred[c] = 1 / CLASSES;
  return Array.from(pred);
}

function sanitize(pred, floor) {
  const FLOOR = floor || 0.0005;
  let sum = 0;
  for (let c = 0; c < CLASSES; c++) { pred[c] = Math.max(pred[c], FLOOR); sum += pred[c]; }
  let maxIdx = 0, maxVal = 0;
  for (let c = 0; c < CLASSES; c++) { pred[c] = parseFloat((pred[c] / sum).toFixed(6)); if (pred[c] > maxVal) { maxVal = pred[c]; maxIdx = c; } }
  const newSum = pred.reduce((a, b) => a + b, 0);
  pred[maxIdx] = parseFloat((pred[maxIdx] + (1.0 - newSum)).toFixed(6));
  return pred;
}

function scoreAgainstGT(prediction, gt) {
  let totalKL = 0, totalEnt = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const g = gt[y][x];
    let ent = 0;
    for (let c = 0; c < CLASSES; c++) if (g[c] > 1e-6) ent -= g[c] * Math.log(g[c]);
    if (ent < 0.01) continue;
    let kl = 0;
    for (let c = 0; c < CLASSES; c++)
      if (g[c] > 1e-6) kl += g[c] * Math.log(g[c] / Math.max(prediction[y][x][c], 1e-15));
    totalKL += Math.max(0, kl) * ent;
    totalEnt += ent;
  }
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * (totalEnt > 0 ? totalKL / totalEnt : 0))));
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════
async function main() {
  log('╔══════════════════════════════════════════════════════╗');
  log('║   SUBMIT V4 — Feature-Aggregated VP + Model Fusion   ║');
  log('╚══════════════════════════════════════════════════════╝');

  const rounds = (await api('GET', '/rounds')).data;
  const active = rounds.find(r => r.status === 'active');
  const completed = rounds.filter(r => r.status === 'completed');
  if (!active) { log('No active round!'); process.exit(1); }
  log(`Active: R${active.round_number} (${active.id.slice(0, 8)}) — ${((new Date(active.closes_at) - Date.now()) / 60000).toFixed(0)} min left`);

  const detail = (await api('GET', `/rounds/${active.id}`)).data;
  const initialStates = detail.initial_states;

  // Load VP data
  const vpPath = path.join(DATA_DIR, `viewport_${active.id.slice(0, 8)}.json`);
  if (!fs.existsSync(vpPath)) { log('No VP data!'); process.exit(1); }
  const vpRaw = JSON.parse(fs.readFileSync(vpPath, 'utf8'));

  // Process VP into per-cell counts
  const vpPerCell = {};
  for (let s = 0; s < SEEDS; s++) {
    vpPerCell[s] = {
      counts: Array.from({ length: H }, () => Array.from({ length: W }, () => new Float64Array(CLASSES))),
      nObs: Array.from({ length: H }, () => new Float64Array(W)),
    };
  }
  for (const obs of vpRaw) {
    const s = obs.si;
    for (let dy = 0; dy < 15; dy++) for (let dx = 0; dx < 15; dx++) {
      const gy = obs.vy + dy, gx = obs.vx + dx;
      if (gy >= H || gx >= W || !obs.grid[dy]) continue;
      vpPerCell[s].counts[gy][gx][t2c(obs.grid[dy][dx])]++;
      vpPerCell[s].nObs[gy][gx]++;
    }
  }
  log(`VP: ${vpRaw.length} observations loaded`);

  // Build VP feature model (aggregated across all seeds)
  log('\n── Building VP Feature Model ──');
  const vpModel = buildVPFeatureModel(vpPerCell, initialStates);

  // Load cross-round training data
  log('\n── Loading cross-round data ──');
  const trainingData = [];
  for (const round of completed) {
    const rDetail = (await api('GET', `/rounds/${round.id}`)).data;
    if (!rDetail.initial_states) continue;
    for (let s = 0; s < SEEDS; s++) {
      const gtPath = path.join(DATA_DIR, `gt_${round.id.slice(0, 8)}_s${s}.json`);
      if (!fs.existsSync(gtPath)) continue;
      const gtData = JSON.parse(fs.readFileSync(gtPath, 'utf8'));
      if (!gtData.gt) continue;
      let avgS = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) avgS += gtData.gt[y][x][1] || 0;
      if (avgS / (H * W) < 0.01) continue;
      trainingData.push({ grid: rDetail.initial_states[s].grid, gt: gtData.gt, roundNum: round.round_number });
    }
  }
  log(`Cross-round: ${trainingData.length} seed-GTs`);
  const crossModel = buildCrossRoundModel(trainingData);

  // LOO to validate and find best blend
  log('\n── LOO Validation ──');
  const roundNums = [...new Set(trainingData.map(t => t.roundNum))];
  const regW = [1.0, 0.2, 0.05, 0.02, 0.01, 0.005];

  // Test: VP-only, model-only, and blends
  const blendConfigs = [
    { name: 'VP-only', vpW: 1.0, modelW: 0.0 },
    { name: 'VP 80/20', vpW: 0.8, modelW: 0.2 },
    { name: 'VP 70/30', vpW: 0.7, modelW: 0.3 },
    { name: 'VP 60/40', vpW: 0.6, modelW: 0.4 },
    { name: 'VP 50/50', vpW: 0.5, modelW: 0.5 },
    { name: 'Model-only', vpW: 0.0, modelW: 1.0 },
  ];
  const alphas = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5];

  let bestBlend = null, bestBlendScore = 0;

  // Since we can't do LOO on VP data directly (it's for R10 only),
  // we do model-only LOO and pick blend based on VP data quality
  for (const testRound of roundNums) {
    const train = trainingData.filter(t => t.roundNum !== testRound);
    const test = trainingData.filter(t => t.roundNum === testRound);
    const cvModel = buildCrossRoundModel(train);
    let total = 0;
    for (const t of test) {
      const setts = precomputeSettlements(t.grid);
      const pred = Array.from({ length: H }, (_, y) =>
        Array.from({ length: W }, (_, x) => {
          const feat = extractFeatures(t.grid, y, x, setts);
          if (feat.isOcean) return [0.998, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004];
          if (feat.isMountain) return [0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.998];
          return sanitize(crossRoundPredict(cvModel, feat, regW));
        })
      );
      total += scoreAgainstGT(pred, t.gt);
    }
    log(`  R${testRound} LOO: ${(total / test.length).toFixed(2)}`);
  }

  // For R10, we'll try multiple blend configs and submit them
  // The VP feature model is trained on R10-specific data, so it should be better
  // We submit multiple variants, with best guess as LAST

  log('\n── Generating predictions with multiple configs ──');

  const submissions = [];

  for (const alpha of [0.1, 0.2, 0.3]) {
    for (const blend of [
      { vpW: 1.0, modelW: 0.0, name: `pureVP_a${alpha}` },
      { vpW: 0.7, modelW: 0.3, name: `blend70_a${alpha}` },
      { vpW: 0.5, modelW: 0.5, name: `blend50_a${alpha}` },
    ]) {
      const preds = {};
      for (let s = 0; s < SEEDS; s++) {
        const grid = initialStates[s].grid;
        const setts = precomputeSettlements(grid);
        preds[s] = Array.from({ length: H }, (_, y) =>
          Array.from({ length: W }, (_, x) => {
            const feat = extractFeatures(grid, y, x, setts);
            if (feat.isOcean) return [0.998, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004];
            if (feat.isMountain) return [0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.998];

            const vpResult = vpFeaturePredict(vpModel, feat, alpha);
            const mPred = crossRoundPredict(crossModel, feat, regW);

            let finalPred;
            if (vpResult && blend.vpW > 0) {
              finalPred = [];
              for (let c = 0; c < CLASSES; c++) {
                finalPred[c] = blend.vpW * vpResult.pred[c] + blend.modelW * mPred[c];
              }
            } else {
              finalPred = mPred;
            }

            // Also blend with per-cell VP if available (direct observations)
            const nObs = vpPerCell[s].nObs[y][x];
            if (nObs > 0) {
              const cellAlpha = 0.05;
              const cellTotal = nObs + CLASSES * cellAlpha;
              const cellPred = [];
              for (let c = 0; c < CLASSES; c++) cellPred[c] = (vpPerCell[s].counts[y][x][c] + cellAlpha) / cellTotal;

              // Blend: per-cell VP gets weight proportional to observations
              const cellW = nObs / (nObs + 3); // 1 obs → 0.25, 3 obs → 0.5, 5 obs → 0.625
              for (let c = 0; c < CLASSES; c++) {
                finalPred[c] = cellW * cellPred[c] + (1 - cellW) * finalPred[c];
              }
            }

            return sanitize(finalPred);
          })
        );
      }

      // Compute self-consistency score (how well does prediction match raw VP)
      let consistency = 0, cCount = 0;
      for (let s = 0; s < SEEDS; s++) {
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const nObs = vpPerCell[s].nObs[y][x];
          if (nObs < 2) continue;
          const p = preds[s][y][x];
          let kl = 0;
          for (let c = 0; c < CLASSES; c++) {
            const vpP = (vpPerCell[s].counts[y][x][c] + 0.01) / (nObs + 0.06);
            if (vpP > 1e-6) kl += vpP * Math.log(vpP / Math.max(p[c], 1e-15));
          }
          consistency += kl;
          cCount++;
        }
      }
      const avgKL = cCount > 0 ? consistency / cCount : 999;

      submissions.push({ name: blend.name, preds, avgKL, alpha });
      log(`  ${blend.name}: consistency KL=${avgKL.toFixed(4)}`);
    }
  }

  // Sort by consistency (lower KL = better)
  submissions.sort((a, b) => a.avgKL - b.avgKL);
  log(`\n  Best consistency: ${submissions[0].name} (KL=${submissions[0].avgKL.toFixed(4)})`);
  log(`  2nd best: ${submissions[1].name} (KL=${submissions[1].avgKL.toFixed(4)})`);
  log(`  Worst: ${submissions[submissions.length - 1].name} (KL=${submissions[submissions.length - 1].avgKL.toFixed(4)})`);

  // Submit top 3, BEST LAST (since server keeps last submission)
  if (!DRY_RUN) {
    log('\n── SUBMITTING (best consistency = LAST) ──');

    // Submit 3rd best first
    for (let vi = Math.min(2, submissions.length - 1); vi >= 0; vi--) {
      const sub = submissions[vi];
      log(`  [${vi === 0 ? 'FINAL' : 'Variant ' + vi}] ${sub.name} (KL=${sub.avgKL.toFixed(4)})`);
      for (let s = 0; s < SEEDS; s++) {
        const resp = await api('POST', '/submit', { round_id: active.id, seed_index: s, prediction: sub.preds[s] });
        log(`    Seed ${s}: ${resp.status}${vi === 0 ? ' — ' + JSON.stringify(resp.data) : ''}`);
        await sleep(550);
      }
    }

    log('\n═══ V4 SUBMITTED ═══');
  } else {
    log('\n── DRY RUN ──');
  }

  // Leaderboard
  const lb = (await api('GET', '/leaderboard')).data;
  log('\n── Top 5 ──');
  for (let i = 0; i < Math.min(5, lb.length); i++)
    log(`  #${i + 1}: ${lb[i].team_name} — ${lb[i].weighted_score.toFixed(1)}`);
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });
