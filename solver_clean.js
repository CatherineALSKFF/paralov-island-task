#!/usr/bin/env node
/**
 * Astar Island Clean Solver
 *
 * Usage: node solver_clean.js <JWT_TOKEN> [--submit]
 *
 * Steps:
 * 1. Fetches active round info
 * 2. Fetches GT from all completed rounds (builds model)
 * 3. Runs VP queries (9 per seed, saves to file after EACH query)
 * 4. Combines model + VP observations
 * 5. Validates predictions locally (self-scoring against past rounds)
 * 6. Submits only if --submit flag is set
 */

const fs = require('fs');
const path = require('path');

const TOKEN = process.argv[2] || '';
const SUBMIT = process.argv.includes('--submit');
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40;
const FLOOR = 0.001;
const DATA_DIR = path.join(__dirname, 'data');

if (!TOKEN) {
  console.log('Usage: node solver_clean.js <JWT_TOKEN> [--submit]');
  process.exit(1);
}

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

const headers = {
  'Authorization': 'Bearer ' + TOKEN,
  'Content-Type': 'application/json'
};

async function api(method, endpoint, body = null) {
  const opts = { method, headers };
  if (body) opts.body = JSON.stringify(body);
  const resp = await fetch(BASE + endpoint, opts);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API ${method} ${endpoint}: ${resp.status} ${text}`);
  }
  return resp.json();
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ===== Feature computation =====
function getFeatureKey(grid, settPos, y, x) {
  const init = grid[y][x];
  if (init === 10) return 'O';
  if (init === 5) return 'M';
  const tKey = init === 4 ? 'F' : (init === 1 || init === 2) ? 'S' : 'P';

  let nearS = 0;
  for (let dy = -3; dy <= 3; dy++) {
    for (let dx = -3; dx <= 3; dx++) {
      if (dy === 0 && dx === 0) continue;
      const ny = y + dy, nx = x + dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx)) nearS++;
    }
  }

  let coastal = false;
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) coastal = true;
  }

  const nKey = nearS === 0 ? '0' : nearS <= 2 ? '1' : nearS <= 5 ? '2' : '3';
  return tKey + nKey + (coastal ? 'c' : '');
}

// ===== Map terrain code to prediction class =====
function terrainToClass(code) {
  if (code === 10 || code === 11 || code === 0) return 0; // ocean/plains/empty
  if (code === 1) return 1; // settlement
  if (code === 2) return 2; // port
  if (code === 3) return 3; // ruin
  if (code === 4) return 4; // forest
  if (code === 5) return 5; // mountain
  return 0;
}

// ===== Scoring function (exact competition formula) =====
function computeScore(prediction, groundTruth) {
  let weightedKL = 0, totalEntropy = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const p = groundTruth[y][x];
      const q = prediction[y][x];
      let ent = 0;
      for (let c = 0; c < 6; c++) {
        if (p[c] > 0.001) ent -= p[c] * Math.log(p[c]);
      }
      if (ent < 0.01) continue;
      let kl = 0;
      for (let c = 0; c < 6; c++) {
        if (p[c] > 0.001) kl += p[c] * Math.log(p[c] / Math.max(q[c], 1e-10));
      }
      weightedKL += ent * kl;
      totalEntropy += ent;
    }
  }
  if (totalEntropy === 0) return 100;
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * weightedKL / totalEntropy)));
}

// ===== Build GT model from completed rounds =====
async function buildGTModel() {
  const cacheFile = path.join(DATA_DIR, 'gt_model_buckets.json');

  // Check cache
  if (fs.existsSync(cacheFile)) {
    console.log('Loading cached GT model...');
    return JSON.parse(fs.readFileSync(cacheFile, 'utf8'));
  }

  console.log('Building GT model from completed rounds...');
  const rounds = await api('GET', '/rounds');
  const completed = rounds.filter(r => r.status === 'completed');

  const perRoundBuckets = {};

  for (const round of completed) {
    const rn = round.round_number;
    console.log(`  Processing R${rn}...`);
    perRoundBuckets[rn] = {};

    const detail = await api('GET', '/rounds/' + round.id);

    for (let si = 0; si < 5; si++) {
      try {
        const gt = await api('GET', `/analysis/${round.id}/${si}`);
        if (!gt.ground_truth) continue;

        const grid = detail.initial_states[si].grid;
        const settlements = detail.initial_states[si].settlements;
        const settPos = new Set();
        for (const s of settlements) settPos.add(s.y * W + s.x);

        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const key = getFeatureKey(grid, settPos, y, x);
            const gtVec = gt.ground_truth[y][x];
            if (!perRoundBuckets[rn][key]) perRoundBuckets[rn][key] = { count: 0, sum: [0,0,0,0,0,0] };
            perRoundBuckets[rn][key].count++;
            for (let c = 0; c < 6; c++) perRoundBuckets[rn][key].sum[c] += gtVec[c];
          }
        }
      } catch (e) {
        console.log(`    Seed ${si}: ${e.message}`);
      }
      await sleep(100);
    }
  }

  // Save cache
  fs.writeFileSync(cacheFile, JSON.stringify(perRoundBuckets));
  console.log(`  Cached to ${cacheFile}`);
  return perRoundBuckets;
}

function buildModelFromBuckets(perRoundBuckets, roundNumbers = null) {
  const buckets = {};
  const rounds = roundNumbers || Object.keys(perRoundBuckets).map(Number);
  for (const rn of rounds) {
    if (!perRoundBuckets[rn]) continue;
    for (const [key, val] of Object.entries(perRoundBuckets[rn])) {
      if (!buckets[key]) buckets[key] = { count: 0, sum: [0,0,0,0,0,0] };
      buckets[key].count += val.count;
      for (let c = 0; c < 6; c++) buckets[key].sum[c] += val.sum[c];
    }
  }
  const model = {};
  for (const [key, val] of Object.entries(buckets)) {
    model[key] = val.sum.map(v => v / val.count);
  }
  return model;
}

// ===== Growth rate estimation from VP observations =====
function estimateGrowthRate(vpData, initialStates) {
  // Count how many cells near initial settlements became settlements/ports in VP
  let settCount = 0, dynamicCount = 0;

  for (let seed = 0; seed < 5; seed++) {
    const { grid: vpGrid, count: vpCount } = assembleVPGrid(vpData, seed);
    const initGrid = initialStates[seed].grid;
    const settlements = initialStates[seed].settlements;
    const settPos = new Set();
    for (const s of settlements) settPos.add(s.y * W + s.x);

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (vpCount[y][x] === 0) continue;
        const init = initGrid[y][x];
        if (init === 10 || init === 5) continue; // skip ocean/mountain

        // Check if near any settlement (radius 5)
        let nearSett = false;
        for (let dy = -5; dy <= 5 && !nearSett; dy++) {
          for (let dx = -5; dx <= 5 && !nearSett; dx++) {
            const ny = y + dy, nx = x + dx;
            if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx)) nearSett = true;
          }
        }
        if (!nearSett) continue;

        dynamicCount++;
        const vpClass = terrainToClass(vpGrid[y][x]);
        if (vpClass === 1 || vpClass === 2) settCount++; // settlement or port
      }
    }
  }

  return dynamicCount > 0 ? settCount / dynamicCount : 0.15; // default moderate
}

// ===== Compute growth rates for completed rounds =====
async function computeRoundGrowthRates(perRoundBuckets) {
  const cacheFile = path.join(DATA_DIR, 'growth_rates.json');
  if (fs.existsSync(cacheFile)) {
    return JSON.parse(fs.readFileSync(cacheFile, 'utf8'));
  }

  const rates = {};
  for (const rn of Object.keys(perRoundBuckets)) {
    // Estimate growth rate from bucket averages
    let settProb = 0, dynamicCount = 0;
    for (const [key, val] of Object.entries(perRoundBuckets[rn])) {
      if (key === 'O' || key === 'M') continue;
      const avg = val.sum.map(v => v / val.count);
      let ent = 0;
      for (let c = 0; c < 6; c++) if (avg[c] > 0.001) ent -= avg[c] * Math.log(avg[c]);
      if (ent < 0.01) continue;
      settProb += (avg[1] + avg[2]) * val.count; // settlement + port
      dynamicCount += val.count;
    }
    rates[rn] = dynamicCount > 0 ? settProb / dynamicCount : 0;
  }

  fs.writeFileSync(cacheFile, JSON.stringify(rates));
  return rates;
}

// ===== Select K closest rounds by growth rate =====
function selectClosestRounds(growthRates, targetRate, K = 3) {
  return Object.entries(growthRates)
    .map(([rn, rate]) => ({ rn: parseInt(rn), dist: Math.abs(rate - targetRate) }))
    .sort((a, b) => a.dist - b.dist)
    .slice(0, K)
    .map(c => c.rn);
}

// ===== Run VP queries and save persistently =====
async function runVPQueries(roundId, initialStates) {
  const vpFile = path.join(DATA_DIR, `vp_${roundId}.json`);

  // Load existing VP data
  let vpData = {};
  if (fs.existsSync(vpFile)) {
    vpData = JSON.parse(fs.readFileSync(vpFile, 'utf8'));
    console.log(`Loaded ${Object.keys(vpData).length} existing VP entries`);
  }

  // Check budget
  const budget = await api('GET', '/budget');
  const remaining = budget.queries_max - budget.queries_used;
  console.log(`Budget: ${budget.queries_used}/${budget.queries_max} used, ${remaining} remaining`);

  if (remaining === 0) {
    console.log('No queries remaining. Using existing VP data.');
    return vpData;
  }

  // VP tiling: 3x3 grid per seed
  const vpPositions = [];
  for (const y of [0, 13, 25]) {
    for (const x of [0, 13, 25]) {
      vpPositions.push({ x, y, w: 15, h: 15 });
    }
  }

  let queriesUsed = 0;
  for (let seed = 0; seed < 5; seed++) {
    for (const vp of vpPositions) {
      const vpKey = `${seed}_${vp.x}_${vp.y}`;
      if (vpData[vpKey]) continue; // already have this tile
      if (queriesUsed >= remaining) {
        console.log('Budget exhausted during VP queries');
        break;
      }

      try {
        const result = await api('POST', '/simulate', {
          round_id: roundId,
          seed_index: seed,
          viewport_x: vp.x,
          viewport_y: vp.y,
          viewport_w: vp.w,
          viewport_h: vp.h
        });

        vpData[vpKey] = { seed, vp, grid: result.grid };
        queriesUsed++;

        // SAVE AFTER EVERY QUERY - never lose data!
        fs.writeFileSync(vpFile, JSON.stringify(vpData));

        console.log(`  Seed ${seed} VP (${vp.x},${vp.y}): OK [${budget.queries_used + queriesUsed}/${budget.queries_max}]`);
        await sleep(220); // rate limit
      } catch (e) {
        console.log(`  Seed ${seed} VP (${vp.x},${vp.y}): ${e.message}`);
        if (e.message.includes('429') || e.message.includes('exhausted')) break;
      }
    }
  }

  return vpData;
}

// ===== Assemble full 40x40 grid from VP tiles =====
function assembleVPGrid(vpData, seed) {
  const grid = Array.from({ length: H }, () => Array(W).fill(-1));
  const count = Array.from({ length: H }, () => Array(W).fill(0));

  for (const [key, tile] of Object.entries(vpData)) {
    if (tile.seed !== seed) continue;
    const { vp } = tile;
    for (let ty = 0; ty < tile.grid.length; ty++) {
      for (let tx = 0; tx < tile.grid[ty].length; tx++) {
        const gy = vp.y + ty, gx = vp.x + tx;
        if (gy < H && gx < W) {
          grid[gy][gx] = tile.grid[ty][tx];
          count[gy][gx]++;
        }
      }
    }
  }

  return { grid, count };
}

// ===== Generate prediction for one seed =====
function generatePrediction(model, initialGrid, settlements, vpGrid, vpCount) {
  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const N_PRIOR = 15; // pseudocount for model prior — trust adaptive model heavily, VP is for growth rate estimation + small corrections

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initialGrid, settPos, y, x);

      // Get model prior
      let prior = model[key] ? [...model[key]] : null;
      if (!prior) {
        const fb = key.slice(0, -1);
        prior = model[fb] ? [...model[fb]] : [1/6,1/6,1/6,1/6,1/6,1/6];
      }

      // If we have VP observation, do Bayesian update
      if (vpGrid && vpCount[y][x] > 0) {
        const obsClass = terrainToClass(vpGrid[y][x]);
        // Dirichlet-multinomial update: posterior = (N_PRIOR * prior + observation) / (N_PRIOR + count)
        const nObs = vpCount[y][x];
        const q = prior.map((p, c) => N_PRIOR * p);
        q[obsClass] += nObs;
        const total = N_PRIOR + nObs;
        for (let c = 0; c < 6; c++) q[c] /= total;

        // Apply floor and normalize
        for (let c = 0; c < 6; c++) q[c] = Math.max(q[c], FLOOR);
        const sum = q.reduce((a, b) => a + b, 0);
        row.push(q.map(v => v / sum));
      } else {
        // Model-only prediction
        for (let c = 0; c < 6; c++) prior[c] = Math.max(prior[c], FLOOR);
        const sum = prior.reduce((a, b) => a + b, 0);
        row.push(prior.map(v => v / sum));
      }
    }
    pred.push(row);
  }
  return pred;
}

// ===== Validate prediction format =====
function validatePrediction(pred) {
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const vec = pred[y][x];
      if (vec.length !== 6) return `(${y},${x}): wrong length ${vec.length}`;
      const sum = vec.reduce((a, b) => a + b, 0);
      if (Math.abs(sum - 1.0) > 0.01) return `(${y},${x}): sum=${sum.toFixed(4)}`;
      for (let c = 0; c < 6; c++) {
        if (vec[c] < 0) return `(${y},${x}): negative prob`;
        if (vec[c] === 0) return `(${y},${x}): zero prob (will cause infinite KL!)`;
      }
    }
  }
  return null;
}

// ===== LOO Cross-Validation (adaptive vs static) =====
async function runLOOCV(perRoundBuckets, growthRates) {
  console.log('\n=== Leave-One-Out Cross-Validation ===');
  const rounds = Object.keys(perRoundBuckets).map(Number);

  function proxyScore(model, testBuckets) {
    let wKL = 0, tE = 0;
    for (const [key, val] of Object.entries(testBuckets)) {
      if (key === 'O' || key === 'M') continue;
      const p = val.sum.map(v => v / val.count);
      let q = model[key] ? [...model[key]] : null;
      if (!q) continue;
      let ent = 0;
      for (let c = 0; c < 6; c++) if (p[c] > 0.001) ent -= p[c] * Math.log(p[c]);
      if (ent < 0.01) continue;
      for (let c = 0; c < 6; c++) q[c] = Math.max(q[c], FLOOR);
      const sum = q.reduce((a, b) => a + b, 0);
      for (let c = 0; c < 6; c++) q[c] /= sum;
      let kl = 0;
      for (let c = 0; c < 6; c++) if (p[c] > 0.001) kl += p[c] * Math.log(p[c] / Math.max(q[c], 1e-10));
      wKL += ent * kl * val.count;
      tE += ent * val.count;
    }
    return tE > 0 ? Math.max(0, Math.min(100, 100 * Math.exp(-3 * wKL / tE))) : 100;
  }

  // Test both static (all rounds) and adaptive (K=3 closest by growth rate)
  let staticTotal = 0, adaptiveTotal = 0, n = 0;

  for (const testR of rounds) {
    const trainRds = rounds.filter(r => r !== testR);

    // Static: use all training rounds
    const staticModel = buildModelFromBuckets(perRoundBuckets, trainRds);
    const sScore = proxyScore(staticModel, perRoundBuckets[testR]);

    // Adaptive: select 3 closest by growth rate
    const testGrowth = growthRates[testR];
    const closest = trainRds
      .map(r => ({ rn: r, dist: Math.abs(growthRates[r] - testGrowth) }))
      .sort((a, b) => a.dist - b.dist)
      .slice(0, 3)
      .map(c => c.rn);
    const adaptiveModel = buildModelFromBuckets(perRoundBuckets, closest);
    const aScore = proxyScore(adaptiveModel, perRoundBuckets[testR]);

    console.log(`  R${testR}: static=${sScore.toFixed(1)}, adaptive=${aScore.toFixed(1)} (trained on R${closest.join(',')})`);
    staticTotal += sScore;
    adaptiveTotal += aScore;
    n++;
  }

  console.log(`  Static AVG: ${(staticTotal / n).toFixed(1)}`);
  console.log(`  Adaptive AVG: ${(adaptiveTotal / n).toFixed(1)}`);
}

// ===== Main =====
async function main() {
  console.log('=== Astar Island Adaptive Solver ===\n');

  // Step 1: Get active round
  const rounds = await api('GET', '/rounds');
  const active = rounds.find(r => r.status === 'active');
  if (!active) {
    console.log('No active round found.');
    return;
  }
  console.log(`Active: Round ${active.round_number} (${active.id})`);
  console.log(`Closes: ${active.closes_at}`);
  console.log(`Now: ${new Date().toISOString()}`);

  const detail = await api('GET', '/rounds/' + active.id);
  console.log(`Map: ${detail.map_width}x${detail.map_height}, ${detail.seeds_count} seeds`);
  for (let i = 0; i < detail.seeds_count; i++) {
    const s = detail.initial_states[i];
    console.log(`  Seed ${i}: ${s.settlements.length} settlements, ${s.settlements.filter(s => s.has_port).length} ports`);
  }

  // Step 2: Build GT model + growth rates for past rounds
  const perRoundBuckets = await buildGTModel();
  const growthRates = await computeRoundGrowthRates(perRoundBuckets);
  console.log('\nGrowth rates by round:');
  const sortedRates = Object.entries(growthRates).sort((a, b) => a[1] - b[1]);
  for (const [rn, rate] of sortedRates) {
    console.log(`  R${rn}: ${rate.toFixed(4)}`);
  }

  // Step 3: LOO CV
  await runLOOCV(perRoundBuckets, growthRates);

  // Step 4: Run VP queries (saves to disk after each query!)
  console.log('\n=== VP Queries ===');
  const vpData = await runVPQueries(active.id, detail.initial_states);
  const vpTileCount = Object.keys(vpData).length;
  console.log(`Total VP tiles: ${vpTileCount}`);

  // Step 5: ADAPTIVE MODEL SELECTION
  // Estimate growth rate from VP observations
  let estimatedGrowth;
  if (vpTileCount > 0) {
    estimatedGrowth = estimateGrowthRate(vpData, detail.initial_states);
    console.log(`\nEstimated growth rate from VP: ${estimatedGrowth.toFixed(4)}`);

    // Select 3 closest rounds
    const closest = selectClosestRounds(growthRates, estimatedGrowth, 3);
    console.log(`Selected training rounds: R${closest.join(', R')} (K=3 adaptive)`);
    console.log(`  Growth rates: ${closest.map(r => growthRates[r].toFixed(3)).join(', ')}`);

    // Build adaptive model
    var model = buildModelFromBuckets(perRoundBuckets, closest);
    console.log(`Adaptive model: ${Object.keys(model).length} buckets`);
  } else {
    console.log('\nNo VP data available — using all-rounds model (fallback)');
    estimatedGrowth = null;
    var model = buildModelFromBuckets(perRoundBuckets);
    console.log(`Fallback model: ${Object.keys(model).length} buckets from ${Object.keys(perRoundBuckets).length} rounds`);
  }

  // Step 6: Generate predictions
  console.log('\n=== Generating Predictions ===');
  const predictions = [];
  for (let seed = 0; seed < 5; seed++) {
    const { grid: vpGrid, count: vpCounts } = assembleVPGrid(vpData, seed);
    const covered = vpCounts.flat().filter(c => c > 0).length;
    const pred = generatePrediction(
      model,
      detail.initial_states[seed].grid,
      detail.initial_states[seed].settlements,
      vpGrid,
      vpCounts
    );

    const err = validatePrediction(pred);
    if (err) {
      console.log(`  Seed ${seed}: INVALID - ${err}`);
      return;
    }
    console.log(`  Seed ${seed}: valid, ${covered}/1600 cells have VP data`);
    predictions.push(pred);
  }

  // Step 7: Submit or dry-run
  if (SUBMIT) {
    console.log('\n=== Submitting ===');
    for (let seed = 0; seed < 5; seed++) {
      const result = await api('POST', '/submit', {
        round_id: active.id,
        seed_index: seed,
        prediction: predictions[seed]
      });
      console.log(`  Seed ${seed}: ${result.status}`);
      await sleep(600);
    }
    console.log('\nAll seeds submitted!');
  } else {
    console.log('\n=== DRY RUN (use --submit to actually submit) ===');
    console.log('Predictions generated but NOT submitted.');
  }

  // Save predictions
  const predFile = path.join(DATA_DIR, `predictions_${active.id}.json`);
  fs.writeFileSync(predFile, JSON.stringify(predictions));
  console.log(`Predictions saved to ${predFile}`);
}

main().catch(e => {
  console.error('FATAL:', e.message);
  process.exit(1);
});
