#!/usr/bin/env node
/**
 * SUBMIT V3 — VP Data Fusion + Cross-Round Model
 *
 * Uses ACTUAL viewport observations from R10 (50 queries, saved in data/)
 * fused with cross-round GT model for best possible predictions.
 *
 * Key insight: VP gives us 1-5 real year-50 samples per cell.
 * Combine with model-based prior using Bayesian update.
 *
 * Usage: node submit_v3.js --token YOUR_JWT [--dry-run]
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
if (!TOKEN) { console.error('Usage: node submit_v3.js --token YOUR_JWT'); process.exit(1); }

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
// FEATURE EXTRACTION (same as v2 but streamlined)
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

  let nSett1 = 0, nForest1 = 0, nOcean1 = 0, nPort1 = 0;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    if (!dy && !dx) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    const nc = t2c(grid[ny][nx]);
    if (nc === 1) nSett1++;
    if (nc === 2) nPort1++;
    if (nc === 4) nForest1++;
    if (grid[ny][nx] === 10) nOcean1++;
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

  // Distance to nearest settlement (bucketed)
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
// MODEL
// ═══════════════════════════════════════════════════════════════════════════
function buildModel(trainingData) {
  const levels = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5'];
  const model = {};
  for (const lvl of levels) model[lvl] = {};

  for (const td of trainingData) {
    const settlements = precomputeSettlements(td.grid);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const feat = extractFeatures(td.grid, y, x, settlements);
        const gtCell = td.gt[y][x];
        let ent = 0;
        for (let c = 0; c < CLASSES; c++)
          if (gtCell[c] > 1e-6) ent -= gtCell[c] * Math.log(gtCell[c]);

        for (const lvl of levels) {
          const key = feat[lvl];
          if (!model[lvl][key]) model[lvl][key] = { dist: new Float64Array(CLASSES), weight: 0, count: 0 };
          const entry = model[lvl][key];
          const w = Math.max(ent, 0.01);
          for (let c = 0; c < CLASSES; c++) entry.dist[c] += gtCell[c] * w;
          entry.weight += w;
          entry.count++;
        }
      }
    }
  }

  for (const lvl of levels)
    for (const key in model[lvl]) {
      const entry = model[lvl][key];
      if (entry.weight > 0)
        for (let c = 0; c < CLASSES; c++) entry.dist[c] /= entry.weight;
    }

  return model;
}

function modelPredict(model, features, regW) {
  const weights = regW || [1.0, 0.3, 0.15, 0.08, 0.04, 0.02];
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

// ═══════════════════════════════════════════════════════════════════════════
// VP DATA PROCESSING
// ═══════════════════════════════════════════════════════════════════════════
function processVPData(vpData) {
  const result = {};
  for (let s = 0; s < SEEDS; s++) {
    result[s] = {
      counts: Array.from({ length: H }, () => Array.from({ length: W }, () => new Float64Array(CLASSES))),
      nObs: Array.from({ length: H }, () => new Float64Array(W)),
    };
  }

  for (const obs of vpData) {
    const s = obs.si;
    for (let dy = 0; dy < 15; dy++) {
      for (let dx = 0; dx < 15; dx++) {
        const gy = obs.vy + dy, gx = obs.vx + dx;
        if (gy >= H || gx >= W || !obs.grid[dy]) continue;
        const cls = t2c(obs.grid[dy][dx]);
        result[s].counts[gy][gx][cls]++;
        result[s].nObs[gy][gx]++;
      }
    }
  }

  return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// BAYESIAN FUSION: VP empirical + model prior
// ═══════════════════════════════════════════════════════════════════════════
function fuseVPModel(vpCounts, nObs, modelPred, priorStrength) {
  // Dirichlet-Multinomial conjugate update
  // Prior: model prediction scaled by priorStrength (pseudo-observations)
  // Likelihood: VP counts
  // Posterior ∝ prior × likelihood

  const ps = priorStrength; // effective prior observations
  const pred = [];
  const total = nObs + ps;

  for (let c = 0; c < CLASSES; c++) {
    pred[c] = (vpCounts[c] + ps * modelPred[c]) / total;
  }
  return pred;
}

function sanitize(pred, floor) {
  const FLOOR = floor || 0.0001;
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
  log('╔══════════════════════════════════════════════╗');
  log('║   SUBMIT V3 — VP Fusion + Cross-Round Model  ║');
  log('╚══════════════════════════════════════════════╝');

  // 1. Get active round
  const rounds = (await api('GET', '/rounds')).data;
  const active = rounds.find(r => r.status === 'active');
  const completed = rounds.filter(r => r.status === 'completed');
  if (!active) { log('No active round!'); process.exit(1); }
  log(`Active: R${active.round_number} (${active.id.slice(0, 8)}) — ${((new Date(active.closes_at) - Date.now()) / 60000).toFixed(0)} min left`);

  const detail = (await api('GET', `/rounds/${active.id}`)).data;
  const initialStates = detail.initial_states;

  // 2. Load VP data
  const vpPath = path.join(DATA_DIR, `viewport_${active.id.slice(0, 8)}.json`);
  if (!fs.existsSync(vpPath)) { log('No VP data found! Run queries first.'); process.exit(1); }
  const vpRaw = JSON.parse(fs.readFileSync(vpPath, 'utf8'));
  const vpData = processVPData(vpRaw);
  log(`VP data: ${vpRaw.length} observations loaded`);

  for (let s = 0; s < SEEDS; s++) {
    let covered = 0, maxObs = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (vpData[s].nObs[y][x] > 0) covered++;
      maxObs = Math.max(maxObs, vpData[s].nObs[y][x]);
    }
    log(`  Seed ${s}: ${covered} cells covered, max ${maxObs} obs`);
  }

  // 3. Load training data
  log('\n── Loading cross-round training data ──');
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
      avgS /= (H * W);
      if (avgS < 0.01) continue; // death round
      trainingData.push({ grid: rDetail.initial_states[s].grid, gt: gtData.gt, roundNum: round.round_number });
    }
  }
  log(`Training: ${trainingData.length} seed-GTs`);

  // 4. Build model
  const model = buildModel(trainingData);

  // 5. LOO to find optimal VP fusion parameters
  log('\n── LOO: Optimizing VP fusion parameters ──');
  // We simulate having VP data by using replay-like data from completed rounds
  // But more importantly, we tune priorStrength and regWeights

  const regConfigs = [
    [1.0, 0.2, 0.05, 0.02, 0.01, 0.005],
    [1.0, 0.3, 0.15, 0.08, 0.04, 0.02],
    [1.0, 0.4, 0.2, 0.1, 0.05, 0.02],
  ];
  const priorStrengths = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0];
  const floorValues = [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01];

  // For LOO, test VP fusion with actual VP data from R10 applied to completed round GT
  // We can't do proper LOO with VP data, so we'll do model-only LOO + VP fusion on R10

  // Model-only LOO first to find best regW
  let bestLOO = 0, bestRegW = regConfigs[0];
  const roundNums = [...new Set(trainingData.map(t => t.roundNum))];

  for (const regW of regConfigs) {
    let total = 0, count = 0;
    for (const testRound of roundNums) {
      const train = trainingData.filter(t => t.roundNum !== testRound);
      const test = trainingData.filter(t => t.roundNum === testRound);
      const cvModel = buildModel(train);
      for (const t of test) {
        const settlements = precomputeSettlements(t.grid);
        const pred = Array.from({ length: H }, (_, y) =>
          Array.from({ length: W }, (_, x) => {
            const feat = extractFeatures(t.grid, y, x, settlements);
            if (feat.isOcean) return [0.998, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004];
            if (feat.isMountain) return [0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.998];
            return sanitize(modelPredict(cvModel, feat, regW));
          })
        );
        total += scoreAgainstGT(pred, t.gt);
        count++;
      }
    }
    const avg = total / count;
    log(`  regW=[${regW.slice(0, 3).join(',')}...]: LOO=${avg.toFixed(2)}`);
    if (avg > bestLOO) { bestLOO = avg; bestRegW = regW; }
  }
  log(`  Best model LOO: ${bestLOO.toFixed(2)} with regW=[${bestRegW.slice(0, 3).join(',')}...]`);

  // 6. Build predictions with VP fusion — sweep priorStrength and floor
  log('\n── Building R10 predictions with VP fusion ──');
  log('  Testing priorStrength × floor combinations...');

  let bestOverall = { score: 0, ps: 0, floor: 0 };
  // Since we can't score R10 yet, we'll submit the variant that makes physical sense
  // and also try multiple submissions

  // Build all predictions for each config, score against SELF-CONSISTENCY
  // (lower KL between VP empirical and prediction = better)
  for (const ps of priorStrengths) {
    for (const floor of floorValues) {
      let totalConsistency = 0;

      for (let s = 0; s < SEEDS; s++) {
        const grid = initialStates[s].grid;
        const settlements = precomputeSettlements(grid);

        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const nObs = vpData[s].nObs[y][x];
          if (nObs < 2) continue; // need multiple obs for consistency check

          const feat = extractFeatures(grid, y, x, settlements);
          if (feat.isOcean || feat.isMountain) continue;

          const mPred = modelPredict(model, feat, bestRegW);
          const fused = sanitize(fuseVPModel(vpData[s].counts[y][x], nObs, mPred, ps), floor);

          // Self-consistency: how well does fused predict held-out VP data?
          // Use LOO within VP observations
          let kl = 0;
          for (let c = 0; c < CLASSES; c++) {
            const vpP = (vpData[s].counts[y][x][c] + 0.001) / (nObs + 0.006);
            if (vpP > 1e-6) kl += vpP * Math.log(vpP / Math.max(fused[c], 1e-15));
          }
          totalConsistency += kl;
        }
      }

      if (bestOverall.score === 0 || totalConsistency < bestOverall.score) {
        bestOverall = { score: totalConsistency, ps, floor };
      }
    }
  }
  log(`  Best VP fusion: priorStrength=${bestOverall.ps}, floor=${bestOverall.floor} (consistency=${bestOverall.score.toFixed(4)})`);

  // 7. Generate final predictions
  log('\n── Generating final predictions ──');
  const predictions = {};

  for (let s = 0; s < SEEDS; s++) {
    const grid = initialStates[s].grid;
    const settlements = precomputeSettlements(grid);
    predictions[s] = Array.from({ length: H }, (_, y) =>
      Array.from({ length: W }, (_, x) => {
        const feat = extractFeatures(grid, y, x, settlements);

        // Static cells
        if (feat.isOcean) return [0.998, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004];
        if (feat.isMountain) return [0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.998];

        const mPred = modelPredict(model, feat, bestRegW);
        const nObs = vpData[s].nObs[y][x];

        if (nObs > 0) {
          // VP fusion
          return sanitize(fuseVPModel(vpData[s].counts[y][x], nObs, mPred, bestOverall.ps), bestOverall.floor);
        } else {
          // Model only
          return sanitize(mPred, bestOverall.floor);
        }
      })
    );

    // Stats
    let vpCells = 0, modelCells = 0, staticCells = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const feat = extractFeatures(grid, y, x, settlements);
      if (feat.isOcean || feat.isMountain) staticCells++;
      else if (vpData[s].nObs[y][x] > 0) vpCells++;
      else modelCells++;
    }
    log(`  Seed ${s}: ${staticCells} static, ${vpCells} VP-fused, ${modelCells} model-only`);
  }

  // 8. Also try pure VP (no model) and pure model (no VP) as comparison variants
  log('\n── Building variant predictions ──');

  // Variant A: Pure VP with heavier smoothing
  const predPureVP = {};
  for (let s = 0; s < SEEDS; s++) {
    const grid = initialStates[s].grid;
    const settlements = precomputeSettlements(grid);
    predPureVP[s] = Array.from({ length: H }, (_, y) =>
      Array.from({ length: W }, (_, x) => {
        const feat = extractFeatures(grid, y, x, settlements);
        if (feat.isOcean) return [0.998, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004];
        if (feat.isMountain) return [0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.998];

        const nObs = vpData[s].nObs[y][x];
        if (nObs > 0) {
          // Pure VP with Dirichlet smoothing
          const alpha = 0.1;
          const total = nObs + CLASSES * alpha;
          const pred = [];
          for (let c = 0; c < CLASSES; c++) pred[c] = (vpData[s].counts[y][x][c] + alpha) / total;
          return sanitize(pred, 0.005);
        } else {
          return sanitize(modelPredict(model, feat, bestRegW), 0.005);
        }
      })
    );
  }
  log('  Variant A (pure VP + heavy smoothing): built');

  // Variant B: Model + VP with strong prior
  const predStrongPrior = {};
  for (let s = 0; s < SEEDS; s++) {
    const grid = initialStates[s].grid;
    const settlements = precomputeSettlements(grid);
    predStrongPrior[s] = Array.from({ length: H }, (_, y) =>
      Array.from({ length: W }, (_, x) => {
        const feat = extractFeatures(grid, y, x, settlements);
        if (feat.isOcean) return [0.998, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004];
        if (feat.isMountain) return [0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.998];
        const mPred = modelPredict(model, feat, bestRegW);
        const nObs = vpData[s].nObs[y][x];
        if (nObs > 0) return sanitize(fuseVPModel(vpData[s].counts[y][x], nObs, mPred, 10.0), 0.001);
        return sanitize(mPred, 0.001);
      })
    );
  }
  log('  Variant B (strong model prior ps=10): built');

  // 9. Submit — try all 3 variants, last one submitted wins
  // Submit the one we think is best LAST (since server keeps last)
  if (!DRY_RUN) {
    log('\n── SUBMITTING 3 VARIANTS (best = last) ──');

    // Submit variant A first (pure VP — aggressive)
    log('  [Variant A: Pure VP]');
    for (let s = 0; s < SEEDS; s++) {
      const resp = await api('POST', '/submit', { round_id: active.id, seed_index: s, prediction: predPureVP[s] });
      log(`    Seed ${s}: ${resp.status}`);
      await sleep(550);
    }

    // Wait, then submit variant B (strong prior — conservative)
    log('  [Variant B: Strong Prior]');
    for (let s = 0; s < SEEDS; s++) {
      const resp = await api('POST', '/submit', { round_id: active.id, seed_index: s, prediction: predStrongPrior[s] });
      log(`    Seed ${s}: ${resp.status}`);
      await sleep(550);
    }

    // Submit the BEST (tuned fusion) LAST — this is what counts
    log('  [FINAL: Tuned VP+Model Fusion]');
    for (let s = 0; s < SEEDS; s++) {
      const resp = await api('POST', '/submit', { round_id: active.id, seed_index: s, prediction: predictions[s] });
      log(`    Seed ${s}: ${resp.status} — ${JSON.stringify(resp.data)}`);
      await sleep(550);
    }

    log('\n═══ V3 SUBMITTED (tuned fusion as final) ═══');
  } else {
    log('\n── DRY RUN ──');
  }

  // 10. Status
  const lb = (await api('GET', '/leaderboard')).data;
  log('\n── Top 5 ──');
  for (let i = 0; i < Math.min(5, lb.length); i++)
    log(`  #${i+1}: ${lb[i].team_name} — ${lb[i].weighted_score.toFixed(1)}`);
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });
