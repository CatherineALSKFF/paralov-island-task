#!/usr/bin/env node
/**
 * SUBMIT NOW — Fast model-based submission for active rounds.
 * Uses cached GT data from completed rounds. No replay collection.
 *
 * Usage: node submit_now.js --token YOUR_JWT [--dry-run]
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
if (!TOKEN) { console.error('Usage: node submit_now.js --token YOUR_JWT'); process.exit(1); }

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
function neighbors(y, x) {
  const n = [];
  for (let dy = -1; dy <= 1; dy++)
    for (let dx = -1; dx <= 1; dx++)
      if ((dy || dx) && y + dy >= 0 && y + dy < H && x + dx >= 0 && x + dx < W)
        n.push([y + dy, x + dx]);
  return n;
}

function ring2(y, x) {
  const n = [];
  for (let dy = -2; dy <= 2; dy++)
    for (let dx = -2; dx <= 2; dx++) {
      if (Math.abs(dy) <= 1 && Math.abs(dx) <= 1) continue;
      const ny = y + dy, nx = x + dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W) n.push([ny, nx]);
    }
  return n;
}

function extractFeatures(grid, y, x) {
  const terrain = t2c(grid[y][x]);
  const raw = grid[y][x];
  const isOcean = raw === 10;
  const isMountain = terrain === 5;

  const nbrs = neighbors(y, x);
  let nSettlement = 0, nForest = 0, nOcean = 0, nPort = 0;
  for (const [ny, nx] of nbrs) {
    const nc = t2c(grid[ny][nx]);
    if (nc === 1) nSettlement++;
    if (nc === 2) nPort++;
    if (nc === 4 || grid[ny][nx] === 4) nForest++;
    if (grid[ny][nx] === 10) nOcean++;
  }
  const coastal = nOcean > 0 ? 1 : 0;

  let sR2 = 0;
  for (const [ny, nx] of ring2(y, x)) {
    const nc = t2c(grid[ny][nx]);
    if (nc === 1 || nc === 2) sR2++;
  }
  const sR2b = Math.min(sR2, 3);

  const D0 = `${terrain}_s${nSettlement}_c${coastal}_f${Math.min(nForest, 4)}_r2${sR2b}_p${nPort}`;
  const D1 = `${terrain}_s${nSettlement}_c${coastal}_f${Math.min(nForest, 3)}`;
  const D2 = `${terrain}_s${nSettlement}_c${coastal}`;
  const D3 = `${terrain}_s${Math.min(nSettlement, 2)}`;
  const D4 = `${terrain}`;

  return { terrain, isOcean, isMountain, D0, D1, D2, D3, D4 };
}

// ═══════════════════════════════════════════════════════════════════════════
// MODEL: Build from cached GT
// ═══════════════════════════════════════════════════════════════════════════
function buildModel(trainingData) {
  const levels = ['D0', 'D1', 'D2', 'D3', 'D4'];
  const model = {};
  for (const lvl of levels) model[lvl] = {};

  for (const { grid, gt } of trainingData) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const feat = extractFeatures(grid, y, x);
        const gtCell = gt[y][x];
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

function predictCell(model, features, regWeights) {
  const weights = regWeights || [1.0, 0.45, 0.2, 0.1, 0.05];
  const levels = ['D0', 'D1', 'D2', 'D3', 'D4'];
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

  if (totalWeight > 0)
    for (let c = 0; c < CLASSES; c++) pred[c] /= totalWeight;
  else
    for (let c = 0; c < CLASSES; c++) pred[c] = 1 / CLASSES;

  return Array.from(pred);
}

function sanitize(pred) {
  const FLOOR = 0.0001;
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
// MAIN
// ═══════════════════════════════════════════════════════════════════════════
async function main() {
  log('╔══════════════════════════════════════╗');
  log('║     SUBMIT NOW — Fast Model Submit   ║');
  log('╚══════════════════════════════════════╝');

  // 1. Get rounds
  const roundsResp = await api('GET', '/rounds');
  const rounds = roundsResp.data;
  const active = rounds.find(r => r.status === 'active');
  const completed = rounds.filter(r => r.status === 'completed');

  if (!active) { log('No active round!'); process.exit(1); }
  log(`Active: R${active.round_number} (${active.id.slice(0, 8)}) weight=${active.round_weight}`);
  const closes = new Date(active.closes_at);
  log(`Closes in ${((closes - Date.now()) / 60000).toFixed(0)} min`);

  // 2. Get round detail
  const detail = (await api('GET', `/rounds/${active.id}`)).data;
  const initialStates = detail.initial_states;

  // 3. Load cached GT from all completed rounds
  log('\n── Loading training data ──');
  const trainingData = [];
  const DEATH_THRESHOLD = 0.01;

  for (const round of completed) {
    const rDetail = (await api('GET', `/rounds/${round.id}`)).data;
    if (!rDetail.initial_states) continue;

    for (let s = 0; s < SEEDS; s++) {
      const gtPath = path.join(DATA_DIR, `gt_${round.id.slice(0, 8)}_s${s}.json`);
      if (!fs.existsSync(gtPath)) {
        // Try to fetch
        const resp = await api('GET', `/analysis/${round.id}/${s}`);
        if (resp.status === 200 && resp.data.ground_truth) {
          fs.writeFileSync(gtPath, JSON.stringify({ gt: resp.data.ground_truth, initial_grid: resp.data.initial_grid }));
        } else continue;
      }

      const gtData = JSON.parse(fs.readFileSync(gtPath, 'utf8'));
      if (!gtData.gt) continue;

      let avgS = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) avgS += gtData.gt[y][x][1] || 0;
      avgS /= (H * W);

      if (avgS < DEATH_THRESHOLD) continue; // skip death rounds

      trainingData.push({ grid: rDetail.initial_states[s].grid, gt: gtData.gt, roundNum: round.round_number });
    }
  }
  log(`Loaded ${trainingData.length} seed-GTs from ${new Set(trainingData.map(t => t.roundNum)).size} rounds`);

  // 4. Build model
  log('\n── Building model ──');
  const model = buildModel(trainingData);
  const levels = ['D0', 'D1', 'D2', 'D3', 'D4'];
  for (const lvl of levels) log(`  ${lvl}: ${Object.keys(model[lvl]).length} keys`);

  // 5. LOO cross-validation
  log('\n── LOO Cross-Validation ──');
  const roundNums = [...new Set(trainingData.map(t => t.roundNum))];
  let bestRegWeights = [1.0, 0.45, 0.2, 0.1, 0.05];
  let bestAvgScore = 0;

  // Test multiple regularization weight configs
  const regConfigs = [
    { name: 'default', w: [1.0, 0.45, 0.2, 0.1, 0.05] },
    { name: 'flat', w: [1.0, 0.8, 0.6, 0.4, 0.2] },
    { name: 'steep', w: [1.0, 0.3, 0.1, 0.03, 0.01] },
    { name: 'D0-heavy', w: [1.0, 0.2, 0.05, 0.02, 0.01] },
    { name: 'balanced', w: [1.0, 0.5, 0.25, 0.12, 0.06] },
    { name: 'coarse', w: [0.5, 0.8, 1.0, 0.6, 0.3] },
    { name: 'sqrt-count-only', w: [1.0, 1.0, 1.0, 1.0, 1.0] },
  ];

  for (const config of regConfigs) {
    let totalScore = 0, count = 0;
    for (const testRound of roundNums) {
      const train = trainingData.filter(t => t.roundNum !== testRound);
      const test = trainingData.filter(t => t.roundNum === testRound);
      const cvModel = buildModel(train);
      for (const t of test) {
        const pred = Array.from({ length: H }, (_, y) =>
          Array.from({ length: W }, (_, x) => {
            const feat = extractFeatures(t.grid, y, x);
            if (feat.isOcean) return [0.998, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004];
            if (feat.isMountain) return [0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.998];
            return sanitize(predictCell(cvModel, feat, config.w));
          })
        );
        totalScore += scoreAgainstGT(pred, t.gt);
        count++;
      }
    }
    const avg = totalScore / count;
    log(`  ${config.name}: ${avg.toFixed(2)} avg LOO`);
    if (avg > bestAvgScore) { bestAvgScore = avg; bestRegWeights = config.w; }
  }
  log(`  BEST: ${bestAvgScore.toFixed(2)} with weights [${bestRegWeights.join(',')}]`);

  // 6. Build R10 predictions with best config
  log('\n── Building R10 predictions ──');
  const predictions = {};
  for (let s = 0; s < SEEDS; s++) {
    const grid = initialStates[s].grid;
    predictions[s] = Array.from({ length: H }, (_, y) =>
      Array.from({ length: W }, (_, x) => {
        const feat = extractFeatures(grid, y, x);
        if (feat.isOcean) return [0.998, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004];
        if (feat.isMountain) return [0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.998];
        return sanitize(predictCell(model, feat, bestRegWeights));
      })
    );

    // Stats
    let nDynamic = 0, avgEnt = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const p = predictions[s][y][x];
      let ent = 0;
      for (let c = 0; c < CLASSES; c++) if (p[c] > 1e-6) ent -= p[c] * Math.log(p[c]);
      if (ent > 0.01) { nDynamic++; avgEnt += ent; }
    }
    log(`  Seed ${s}: ${nDynamic} dynamic cells, avg entropy ${(avgEnt / Math.max(nDynamic, 1)).toFixed(3)}`);
  }

  // 7. Submit
  if (!DRY_RUN) {
    log('\n── SUBMITTING ──');
    for (let s = 0; s < SEEDS; s++) {
      // Validate
      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const sum = predictions[s][y][x].reduce((a, b) => a + b, 0);
        if (Math.abs(sum - 1.0) > 0.01) { log(`VALIDATION FAIL: s${s} (${y},${x}) sum=${sum}`); valid = false; }
      }
      if (!valid) continue;

      const resp = await api('POST', '/submit', {
        round_id: active.id,
        seed_index: s,
        prediction: predictions[s],
      });
      log(`  Seed ${s}: ${resp.status} — ${JSON.stringify(resp.data)}`);
      await sleep(550);
    }
    log('\n═══ SUBMITTED ═══');
  } else {
    log('\n── DRY RUN — not submitting ──');
  }

  // 8. Leaderboard + our scores
  const lb = (await api('GET', '/leaderboard')).data;
  const myRounds = (await api('GET', '/my-rounds')).data;
  log('\n── Our Scores ──');
  for (const r of myRounds) {
    if (r.seeds_submitted > 0) log(`  R${r.round_number}: ${r.round_score !== null ? r.round_score.toFixed(1) : 'pending'} (rank #${r.rank || '?'})`);
  }
  log('\n── Top 5 Leaderboard ──');
  for (let i = 0; i < Math.min(5, lb.length); i++) {
    log(`  #${i+1}: ${lb[i].team_name} — ${lb[i].weighted_score.toFixed(1)}`);
  }
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });
