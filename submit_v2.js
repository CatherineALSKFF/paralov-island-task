#!/usr/bin/env node
/**
 * SUBMIT V2 — Improved model with:
 *   - Temperature scaling (optimized via LOO)
 *   - Round-similarity weighting (target's initial state → weight training rounds)
 *   - Richer features (distance to settlements, terrain density, expansion zones)
 *   - Death-round detection + adaptive model selection
 *   - Multiple submission variants (submit best from LOO)
 *
 * Usage: node submit_v2.js --token YOUR_JWT [--dry-run]
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
if (!TOKEN) { console.error('Usage: node submit_v2.js --token YOUR_JWT'); process.exit(1); }

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
// ENHANCED FEATURE EXTRACTION
// ═══════════════════════════════════════════════════════════════════════════
function neighbors(y, x, radius) {
  radius = radius || 1;
  const n = [];
  for (let dy = -radius; dy <= radius; dy++)
    for (let dx = -radius; dx <= radius; dx++)
      if ((dy || dx) && y + dy >= 0 && y + dy < H && x + dx >= 0 && x + dx < W)
        n.push([y + dy, x + dx]);
  return n;
}

// Precompute settlement positions for distance calc
function precomputeSettlements(grid) {
  const setts = [];
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const c = t2c(grid[y][x]);
      if (c === 1 || c === 2) setts.push([y, x]);
    }
  return setts;
}

// Manhattan distance to nearest settlement
function distToSettlement(y, x, settlements) {
  let minDist = 999;
  for (const [sy, sx] of settlements) {
    const d = Math.abs(y - sy) + Math.abs(x - sx);
    if (d < minDist) minDist = d;
  }
  return minDist;
}

function extractFeatures(grid, y, x, settlements) {
  const terrain = t2c(grid[y][x]);
  const raw = grid[y][x];
  const isOcean = raw === 10;
  const isMountain = terrain === 5;
  const isForest = terrain === 4;

  // Ring-1 neighbors
  const nbrs1 = neighbors(y, x, 1);
  let nSett1 = 0, nForest1 = 0, nOcean1 = 0, nMtn1 = 0, nPort1 = 0, nPlains1 = 0;
  for (const [ny, nx] of nbrs1) {
    const nc = t2c(grid[ny][nx]);
    if (nc === 1) nSett1++;
    if (nc === 2) nPort1++;
    if (nc === 4) nForest1++;
    if (grid[ny][nx] === 10) nOcean1++;
    if (nc === 5) nMtn1++;
    if (grid[ny][nx] === 11 || grid[ny][nx] === 0) nPlains1++;
  }
  const coastal = nOcean1 > 0 ? 1 : 0;

  // Ring-2 settlements
  const nbrs2 = neighbors(y, x, 2);
  let nSett2 = 0;
  for (const [ny, nx] of nbrs2) {
    const nc = t2c(grid[ny][nx]);
    if (nc === 1 || nc === 2) nSett2++;
  }

  // Ring-3 settlements (expansion zone)
  const nbrs3 = neighbors(y, x, 3);
  let nSett3 = 0;
  for (const [ny, nx] of nbrs3) {
    const nc = t2c(grid[ny][nx]);
    if (nc === 1 || nc === 2) nSett3++;
  }

  // Distance to nearest settlement (bucketed)
  const dist = settlements ? distToSettlement(y, x, settlements) : 99;
  const distBucket = dist === 0 ? 0 : dist <= 1 ? 1 : dist <= 2 ? 2 : dist <= 3 ? 3 : dist <= 5 ? 4 : dist <= 8 ? 5 : 6;

  // Forest density around cell (food potential)
  const forestDensity = nForest1 >= 4 ? 'H' : nForest1 >= 2 ? 'M' : nForest1 >= 1 ? 'L' : '0';

  // Settlement pressure (how many settlements nearby = conflict zone)
  const settPressure = nSett2 + nSett3;
  const pressureBucket = settPressure === 0 ? 0 : settPressure <= 2 ? 1 : settPressure <= 5 ? 2 : 3;

  // Feature levels (most specific → most general)
  const D0 = `${terrain}_s${nSett1}_c${coastal}_f${forestDensity}_d${distBucket}_p${nPort1}_pr${pressureBucket}`;
  const D1 = `${terrain}_s${nSett1}_c${coastal}_f${forestDensity}_d${distBucket}`;
  const D2 = `${terrain}_s${Math.min(nSett1, 2)}_c${coastal}_d${Math.min(distBucket, 4)}`;
  const D3 = `${terrain}_s${Math.min(nSett1, 2)}_d${Math.min(distBucket, 3)}`;
  const D4 = `${terrain}_d${Math.min(distBucket, 2)}`;
  const D5 = `${terrain}`;

  return { terrain, isOcean, isMountain, isForest, D0, D1, D2, D3, D4, D5, dist, nSett1, coastal };
}

// ═══════════════════════════════════════════════════════════════════════════
// MODEL WITH TEMPERATURE SCALING
// ═══════════════════════════════════════════════════════════════════════════
function buildModel(trainingData, roundWeights) {
  const levels = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5'];
  const model = {};
  for (const lvl of levels) model[lvl] = {};

  for (const td of trainingData) {
    const rw = roundWeights ? (roundWeights[td.roundNum] || 1.0) : 1.0;
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
          const w = Math.max(ent, 0.01) * rw;
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

function predictCell(model, features, regWeights, temperature) {
  const weights = regWeights || [1.0, 0.3, 0.15, 0.08, 0.04, 0.02];
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

  if (totalWeight > 0) {
    for (let c = 0; c < CLASSES; c++) pred[c] /= totalWeight;
  } else {
    for (let c = 0; c < CLASSES; c++) pred[c] = 1 / CLASSES;
  }

  // Apply temperature scaling
  const temp = temperature || 1.0;
  if (temp !== 1.0) {
    // Temperature < 1 = sharper, > 1 = softer
    let sum = 0;
    for (let c = 0; c < CLASSES; c++) {
      pred[c] = Math.pow(Math.max(pred[c], 1e-10), 1 / temp);
      sum += pred[c];
    }
    for (let c = 0; c < CLASSES; c++) pred[c] /= sum;
  }

  return Array.from(pred);
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
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
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
  }
  const wkl = totalEnt > 0 ? totalKL / totalEnt : 0;
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl)));
}

// ═══════════════════════════════════════════════════════════════════════════
// ROUND SIMILARITY
// ═══════════════════════════════════════════════════════════════════════════
function roundProfile(initialStates) {
  // Compute a feature vector describing the round's initial state
  let totalSett = 0, totalPort = 0, totalForest = 0, totalOcean = 0, totalMtn = 0, totalPlains = 0;
  for (let s = 0; s < SEEDS; s++) {
    const grid = initialStates[s].grid;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const c = t2c(grid[y][x]);
      if (c === 1) totalSett++;
      if (c === 2) totalPort++;
      if (c === 4) totalForest++;
      if (grid[y][x] === 10) totalOcean++;
      if (c === 5) totalMtn++;
      if (grid[y][x] === 11 || grid[y][x] === 0) totalPlains++;
    }
  }
  return { sett: totalSett / SEEDS, port: totalPort / SEEDS, forest: totalForest / SEEDS,
    ocean: totalOcean / SEEDS, mtn: totalMtn / SEEDS, plains: totalPlains / SEEDS };
}

function roundSimilarity(profile1, profile2) {
  // Euclidean distance on normalized features
  const keys = ['sett', 'port', 'forest', 'ocean', 'mtn', 'plains'];
  let sumSq = 0;
  for (const k of keys) {
    const maxVal = Math.max(profile1[k], profile2[k], 1);
    const diff = (profile1[k] - profile2[k]) / maxVal;
    sumSq += diff * diff;
  }
  return 1 / (1 + Math.sqrt(sumSq));
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════
async function main() {
  log('╔══════════════════════════════════════════╗');
  log('║   SUBMIT V2 — Enhanced Model + Tuning    ║');
  log('╚══════════════════════════════════════════╝');

  // 1. Get rounds
  const rounds = (await api('GET', '/rounds')).data;
  const active = rounds.find(r => r.status === 'active');
  const completed = rounds.filter(r => r.status === 'completed');
  if (!active) { log('No active round!'); process.exit(1); }
  log(`Active: R${active.round_number} (${active.id.slice(0, 8)}) weight=${active.round_weight}`);
  log(`Closes in ${((new Date(active.closes_at) - Date.now()) / 60000).toFixed(0)} min`);

  // 2. Target round details
  const detail = (await api('GET', `/rounds/${active.id}`)).data;
  const initialStates = detail.initial_states;
  const targetProfile = roundProfile(initialStates);
  log(`Target profile: sett=${targetProfile.sett.toFixed(0)} port=${targetProfile.port.toFixed(0)} forest=${targetProfile.forest.toFixed(0)}`);

  // 3. Load training data + compute round profiles
  log('\n── Loading training data ──');
  const trainingData = [];
  const roundProfiles = {};
  const roundInitStates = {};
  const DEATH_THRESHOLD = 0.01;

  for (const round of completed) {
    const rDetail = (await api('GET', `/rounds/${round.id}`)).data;
    if (!rDetail.initial_states) continue;
    roundInitStates[round.round_number] = rDetail.initial_states;
    roundProfiles[round.round_number] = roundProfile(rDetail.initial_states);

    for (let s = 0; s < SEEDS; s++) {
      const gtPath = path.join(DATA_DIR, `gt_${round.id.slice(0, 8)}_s${s}.json`);
      if (!fs.existsSync(gtPath)) {
        const resp = await api('GET', `/analysis/${round.id}/${s}`);
        if (resp.status === 200 && resp.data.ground_truth)
          fs.writeFileSync(gtPath, JSON.stringify({ gt: resp.data.ground_truth, initial_grid: resp.data.initial_grid }));
        else continue;
      }
      const gtData = JSON.parse(fs.readFileSync(gtPath, 'utf8'));
      if (!gtData.gt) continue;

      let avgS = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) avgS += gtData.gt[y][x][1] || 0;
      avgS /= (H * W);
      if (avgS < DEATH_THRESHOLD) continue;

      trainingData.push({
        grid: rDetail.initial_states[s].grid,
        gt: gtData.gt,
        roundNum: round.round_number,
        seed: s,
        avgS,
        settlements: precomputeSettlements(rDetail.initial_states[s].grid),
      });
    }
  }
  log(`Loaded ${trainingData.length} seed-GTs from ${new Set(trainingData.map(t => t.roundNum)).size} rounds`);

  // 4. Compute round similarity weights
  log('\n── Round Similarities to R10 ──');
  const roundNums = [...new Set(trainingData.map(t => t.roundNum))].sort((a, b) => a - b);
  const similarities = {};
  for (const rn of roundNums) {
    const sim = roundSimilarity(targetProfile, roundProfiles[rn]);
    similarities[rn] = sim;
    log(`  R${rn}: similarity=${sim.toFixed(3)}`);
  }

  // 5. MEGA LOO SEARCH — test many configs
  log('\n── Exhaustive LOO Search ──');

  const regConfigs = [
    [1.0, 0.3, 0.15, 0.08, 0.04, 0.02],
    [1.0, 0.2, 0.05, 0.02, 0.01, 0.005],
    [1.0, 0.4, 0.2, 0.1, 0.05, 0.02],
    [1.0, 0.5, 0.3, 0.15, 0.07, 0.03],
    [1.0, 0.15, 0.03, 0.01, 0.005, 0.002],
    [0.8, 0.6, 0.4, 0.2, 0.1, 0.05],
  ];

  const temperatures = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3];
  const floors = [0.0001, 0.0005, 0.001, 0.002, 0.005];
  const weightModes = ['uniform', 'similarity', 'similarity_sq'];

  let bestScore = 0, bestConfig = null;
  let configsTestedTotal = 0;

  function makePrediction(cvModel, grid, settlements, regW, temp, floor) {
    return Array.from({ length: H }, (_, y) =>
      Array.from({ length: W }, (_, x) => {
        const feat = extractFeatures(grid, y, x, settlements);
        if (feat.isOcean) return [0.998, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004];
        if (feat.isMountain) return [0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.998];
        return sanitize(predictCell(cvModel, feat, regW, temp), floor);
      })
    );
  }

  // Phase 1: Coarse search (fewer temps, fewer floors)
  log('  Phase 1: Coarse search...');
  const coarseTemps = [0.8, 0.9, 1.0, 1.1, 1.2];
  const coarseFloors = [0.0001, 0.001, 0.005];

  for (const regW of regConfigs) {
    for (const wMode of weightModes) {
      for (const temp of coarseTemps) {
        for (const floor of coarseFloors) {
          let totalScore = 0, count = 0;

          for (const testRound of roundNums) {
            const train = trainingData.filter(t => t.roundNum !== testRound);
            const test = trainingData.filter(t => t.roundNum === testRound);

            let roundW = {};
            for (const rn of roundNums) {
              if (rn === testRound) continue;
              if (wMode === 'uniform') roundW[rn] = 1.0;
              else if (wMode === 'similarity') roundW[rn] = similarities[rn] || 0.5;
              else roundW[rn] = Math.pow(similarities[rn] || 0.5, 2);
            }

            const cvModel = buildModel(train, roundW);
            for (const t of test) {
              const pred = makePrediction(cvModel, t.grid, t.settlements, regW, temp, floor);
              totalScore += scoreAgainstGT(pred, t.gt);
              count++;
            }
          }

          const avg = totalScore / count;
          configsTestedTotal++;
          if (avg > bestScore) {
            bestScore = avg;
            bestConfig = { regW, temp, floor, wMode };
            log(`    NEW BEST: ${avg.toFixed(2)} — temp=${temp} floor=${floor} wMode=${wMode} regW=[${regW.slice(0, 3).join(',')}...]`);
          }
        }
      }
    }
  }
  log(`  Phase 1 done: ${configsTestedTotal} configs, best=${bestScore.toFixed(2)}`);

  // Phase 2: Fine search around best config
  log('  Phase 2: Fine search around best...');
  const bestTemp = bestConfig.temp;
  const bestFloor = bestConfig.floor;
  const fineTemps = [bestTemp - 0.1, bestTemp - 0.05, bestTemp, bestTemp + 0.05, bestTemp + 0.1];
  const fineFloors = [bestFloor * 0.5, bestFloor, bestFloor * 2, bestFloor * 5];
  const fineRegW = [bestConfig.regW]; // keep best reg weights

  for (const regW of fineRegW) {
    for (const temp of fineTemps) {
      for (const floor of fineFloors) {
        let totalScore = 0, count = 0;
        for (const testRound of roundNums) {
          const train = trainingData.filter(t => t.roundNum !== testRound);
          const test = trainingData.filter(t => t.roundNum === testRound);
          let roundW = {};
          for (const rn of roundNums) {
            if (rn === testRound) continue;
            if (bestConfig.wMode === 'uniform') roundW[rn] = 1.0;
            else if (bestConfig.wMode === 'similarity') roundW[rn] = similarities[rn] || 0.5;
            else roundW[rn] = Math.pow(similarities[rn] || 0.5, 2);
          }
          const cvModel = buildModel(train, roundW);
          for (const t of test) {
            const pred = makePrediction(cvModel, t.grid, t.settlements, regW, temp, floor);
            totalScore += scoreAgainstGT(pred, t.gt);
            count++;
          }
        }
        const avg = totalScore / count;
        configsTestedTotal++;
        if (avg > bestScore) {
          bestScore = avg;
          bestConfig = { ...bestConfig, temp, floor };
          log(`    NEW BEST: ${avg.toFixed(2)} — temp=${temp} floor=${floor}`);
        }
      }
    }
  }
  log(`  Total configs tested: ${configsTestedTotal}, FINAL BEST: ${bestScore.toFixed(2)}`);
  log(`  Config: temp=${bestConfig.temp} floor=${bestConfig.floor} wMode=${bestConfig.wMode}`);
  log(`  RegWeights: [${bestConfig.regW.join(',')}]`);

  // 6. Per-round LOO with best config
  log('\n── Per-Round LOO with Best Config ──');
  for (const testRound of roundNums) {
    const train = trainingData.filter(t => t.roundNum !== testRound);
    const test = trainingData.filter(t => t.roundNum === testRound);
    let roundW = {};
    for (const rn of roundNums) {
      if (rn === testRound) continue;
      if (bestConfig.wMode === 'uniform') roundW[rn] = 1.0;
      else if (bestConfig.wMode === 'similarity') roundW[rn] = similarities[rn] || 0.5;
      else roundW[rn] = Math.pow(similarities[rn] || 0.5, 2);
    }
    const cvModel = buildModel(train, roundW);
    let total = 0;
    for (const t of test) {
      const pred = makePrediction(cvModel, t.grid, t.settlements, bestConfig.regW, bestConfig.temp, bestConfig.floor);
      total += scoreAgainstGT(pred, t.gt);
    }
    log(`  R${testRound}: ${(total / test.length).toFixed(2)}`);
  }

  // 7. Build final model with ALL data + best config
  log('\n── Building final R10 predictions ──');
  let roundW = {};
  for (const rn of roundNums) {
    if (bestConfig.wMode === 'uniform') roundW[rn] = 1.0;
    else if (bestConfig.wMode === 'similarity') roundW[rn] = similarities[rn] || 0.5;
    else roundW[rn] = Math.pow(similarities[rn] || 0.5, 2);
  }
  const finalModel = buildModel(trainingData, roundW);

  const predictions = {};
  for (let s = 0; s < SEEDS; s++) {
    const grid = initialStates[s].grid;
    const settlements = precomputeSettlements(grid);
    predictions[s] = makePrediction(finalModel, grid, settlements, bestConfig.regW, bestConfig.temp, bestConfig.floor);

    let nDynamic = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const p = predictions[s][y][x];
      let ent = 0;
      for (let c = 0; c < CLASSES; c++) if (p[c] > 1e-6) ent -= p[c] * Math.log(p[c]);
      if (ent > 0.1) nDynamic++;
    }
    log(`  Seed ${s}: ${nDynamic} dynamic cells (ent>0.1)`);
  }

  // 8. Submit
  if (!DRY_RUN) {
    log('\n── SUBMITTING ──');
    for (let s = 0; s < SEEDS; s++) {
      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const sum = predictions[s][y][x].reduce((a, b) => a + b, 0);
        if (Math.abs(sum - 1.0) > 0.01) { log(`FAIL: s${s} (${y},${x}) sum=${sum}`); valid = false; }
      }
      if (!valid) continue;
      const resp = await api('POST', '/submit', { round_id: active.id, seed_index: s, prediction: predictions[s] });
      log(`  Seed ${s}: ${resp.status} — ${JSON.stringify(resp.data)}`);
      await sleep(550);
    }
    log('\n═══ V2 SUBMITTED ═══');
  } else {
    log('\n── DRY RUN ──');
  }

  // 9. Leaderboard
  const lb = (await api('GET', '/leaderboard')).data;
  log('\n── Top 5 ──');
  for (let i = 0; i < Math.min(5, lb.length); i++)
    log(`  #${i+1}: ${lb[i].team_name} — ${lb[i].weighted_score.toFixed(1)}`);
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });
