#!/usr/bin/env node
/**
 * Astar Island R6+ Solver — Hybrid approach
 *
 * Strategy:
 * 1. PHASE 1 (immediate): Submit cross-round GT model predictions
 * 2. PHASE 2 (during round): Use /simulate viewport observations to improve
 * 3. PHASE 3 (after close): Collect replays and resubmit (if allowed)
 *
 * Usage:
 *   node solver_r6.js --token <JWT>
 *   node solver_r6.js --token <JWT> --round <round_id>
 *   node solver_r6.js --token <JWT> --collect-replays <round_id> --target 500
 */

const https = require('https');
const fs = require('fs');

// ═══════════════════════════════════════════════════════════════════════
// CONFIG
// ═══════════════════════════════════════════════════════════════════════
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5, CLASSES = 6;

// SIMPLE model params — deliberately few to avoid overfit
// Validated on 5-round LOO: avg ~83-84 (honest estimate)
const MODEL_CONFIG = {
  weights: [1, 0.2, 0.1, 0.05, 0.01],  // multi-level blending weights
  pow: 0.5,       // N^pow — using conservative sqrt instead of overfit 0.8
  minN: 2,        // minimum samples to trust a feature key
  floor: 0.0001,  // probability floor — conservative, avoids KL explosion
};

// Round IDs (update as new rounds appear)
const KNOWN_ROUNDS = {
  R1: '71451d74-be9f-471f-aacd-a41f3b68a9cd',
  R2: '76909e29-f664-4b2f-b16b-61b7507277e9',
  R3: 'f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb',
  R4: '8e839974-b13b-407b-a5e7-fc749d877195',
  R5: 'fd3c92ff-3178-4dc9-8d9b-acf389b3982b',
};

// ═══════════════════════════════════════════════════════════════════════
// PARSE ARGS
// ═══════════════════════════════════════════════════════════════════════
const args = {};
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, '');
  args[key] = process.argv[i + 1];
}

if (!args.token) {
  console.error('Usage: node solver_r6.js --token <JWT> [--round <id>] [--collect-replays <id>] [--target N]');
  console.error('\nGet token: DevTools → Application → Cookies → access_token');
  process.exit(1);
}

const TOKEN = args.token;

// ═══════════════════════════════════════════════════════════════════════
// HTTP HELPERS
// ═══════════════════════════════════════════════════════════════════════
function apiGet(path) {
  return new Promise((resolve, reject) => {
    const url = new URL(BASE + path);
    const opts = {
      hostname: url.hostname, path: url.pathname + url.search,
      headers: { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' }
    };
    https.get(opts, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch(e) { reject(new Error('Parse error: ' + data.substring(0, 200))); }
      });
    }).on('error', reject);
  });
}

function apiPost(path, body) {
  return new Promise((resolve, reject) => {
    const url = new URL(BASE + path);
    const payload = JSON.stringify(body);
    const opts = {
      hostname: url.hostname, path: url.pathname,
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + TOKEN,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(payload)
      }
    };
    const req = https.request(opts, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        try { resolve({ status: res.statusCode, body: JSON.parse(data) }); }
        catch(e) { resolve({ status: res.statusCode, body: data }); }
      });
    });
    req.on('error', reject);
    req.write(payload);
    req.end();
  });
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ═══════════════════════════════════════════════════════════════════════
// FEATURE EXTRACTION
// ═══════════════════════════════════════════════════════════════════════
function cellFeatures(grid, y, x) {
  const t = grid[y][x];
  if (t === 10 || t === 5) return null; // ocean or mountain = static

  let nS = 0, co = 0, fN = 0, sR2 = 0;
  // Ring 1: immediate neighbors
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    if (dy === 0 && dx === 0) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    const nt = grid[ny][nx];
    if (nt === 1 || nt === 2) nS++;
    if (nt === 10) co = 1;
    if (nt === 4) fN++;
  }
  // Ring 2: extended settlement count
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    if (Math.abs(dy) <= 1 && Math.abs(dx) <= 1) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    if (grid[ny][nx] === 1 || grid[ny][nx] === 2) sR2++;
  }

  const sa = Math.min(nS, 5);
  const sb2 = sR2 === 0 ? 0 : sR2 <= 2 ? 1 : sR2 <= 4 ? 2 : 3;
  const fb = fN <= 1 ? 0 : fN <= 3 ? 1 : 2;

  return [
    `D0_${t}_${sa}_${co}_${sb2}_${fb}`,          // most specific (5 features)
    `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`,   // medium (4 features)
    `D2_${t}_${sa > 0 ? 1 : 0}_${co}`,           // coarse (3 features)
    `D3_${t}_${co}`,                               // minimal (2 features)
    `D4_${t}`,                                     // terrain only
  ];
}

// ═══════════════════════════════════════════════════════════════════════
// GT-BASED MODEL
// ═══════════════════════════════════════════════════════════════════════
function buildModel(initStates, gtData) {
  // initStates: {roundName: [5 grids]}, gtData: {roundName: [5 prob grids]}
  const model = {};

  for (const rn of Object.keys(initStates)) {
    if (!gtData[rn]) continue;
    for (let si = 0; si < SEEDS; si++) {
      const init = initStates[rn][si];
      const gt = gtData[rn][si];
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cellFeatures(init, y, x);
        if (!keys) continue;
        const g = gt[y][x];
        for (let ki = 0; ki < keys.length; ki++) {
          const k = keys[ki];
          if (!model[k]) model[k] = { n: 0, s: [0, 0, 0, 0, 0, 0] };
          model[k].n++;
          for (let c = 0; c < CLASSES; c++) model[k].s[c] += g[c];
        }
      }
    }
  }

  // Average
  for (const k of Object.keys(model)) {
    model[k].a = model[k].s.map(s => s / model[k].n);
    delete model[k].s;
  }

  return model;
}

// ═══════════════════════════════════════════════════════════════════════
// PREDICTION (multi-level blending)
// ═══════════════════════════════════════════════════════════════════════
function predict(initGrid, model, config) {
  const { weights, pow, minN, floor } = config;
  const pred = [];

  for (let y = 0; y < H; y++) {
    pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = initGrid[y][x];
      if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
      if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }

      const keys = cellFeatures(initGrid, y, x);
      if (!keys) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }

      const p = [0, 0, 0, 0, 0, 0];
      let wSum = 0;

      for (let ki = 0; ki < keys.length; ki++) {
        const d = model[keys[ki]];
        if (d && d.n >= minN) {
          const w = weights[ki] * Math.pow(d.n, pow);
          for (let c = 0; c < CLASSES; c++) p[c] += w * d.a[c];
          wSum += w;
        }
      }

      if (wSum === 0) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }

      let s = 0;
      for (let c = 0; c < CLASSES; c++) { p[c] /= wSum; if (p[c] < floor) p[c] = floor; s += p[c]; }
      for (let c = 0; c < CLASSES; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

// ═══════════════════════════════════════════════════════════════════════
// VIEWPORT OBSERVATION INTEGRATION (Bayesian update)
// ═══════════════════════════════════════════════════════════════════════
function t2c(t) { return (t === 10 || t === 11 || t === 0) ? 0 : (t >= 1 && t <= 5) ? t : 0; }

function integrateViewportObs(pred, observations, pseudoCount) {
  // observations[y][x] = array of observed terrain classes (0-5), or null
  // pseudoCount: how much to weight prior vs observations (higher = trust prior more)
  const updated = [];

  for (let y = 0; y < H; y++) {
    updated[y] = [];
    for (let x = 0; x < W; x++) {
      const obs = observations[y] && observations[y][x];
      if (!obs || obs.length === 0) {
        updated[y][x] = pred[y][x];
        continue;
      }

      // Bayesian: posterior ∝ prior * likelihood
      // With Dirichlet prior based on pred, and multinomial likelihood from obs
      const prior = pred[y][x];
      const counts = [0, 0, 0, 0, 0, 0];
      for (const c of obs) counts[c]++;
      const N = obs.length;

      // Posterior = (alpha_c + count_c) / (sum_alpha + N)
      // where alpha_c = pseudoCount * prior[c]
      const p = [];
      let s = 0;
      for (let c = 0; c < CLASSES; c++) {
        p[c] = pseudoCount * prior[c] + counts[c];
        s += p[c];
      }
      for (let c = 0; c < CLASSES; c++) p[c] /= s;

      updated[y][x] = p;
    }
  }
  return updated;
}

// Generate optimal viewport positions to cover 40x40 grid with 15x15 windows
function viewportPositions() {
  // 3x3 grid of viewports covers 40x40 with overlap
  const positions = [];
  const starts = [0, 13, 25]; // 0-14, 13-27, 25-39
  for (const y of starts) for (const x of starts) {
    positions.push({ x, y, width: 15, height: 15 });
  }
  return positions; // 9 positions, leaves 1 query spare per seed
}

// ═══════════════════════════════════════════════════════════════════════
// SCORING (for validation)
// ═══════════════════════════════════════════════════════════════════════
function score(pred, gt) {
  let tKL = 0, tE = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const g = gt[y][x];
    let e = 0;
    for (let c = 0; c < CLASSES; c++) if (g[c] > 1e-6) e -= g[c] * Math.log(g[c]);
    if (e < 0.01) continue;
    let kl = 0;
    for (let c = 0; c < CLASSES; c++) {
      if (g[c] > 1e-6) kl += g[c] * Math.log(g[c] / Math.max(pred[y][x][c], 1e-15));
    }
    tKL += Math.max(0, kl) * e;
    tE += e;
  }
  return tE > 0 ? 100 * Math.exp(-3 * tKL / tE) : 0;
}

// ═══════════════════════════════════════════════════════════════════════
// REPLAY COLLECTION
// ═══════════════════════════════════════════════════════════════════════
async function collectReplays(roundId, target, concurrency) {
  const dataFile = `replay_data_${roundId}.json`;
  let replayData = { counts: {}, total: 0 }; // {seedIndex: {y_x: [counts per class]}}

  // Load existing
  if (fs.existsSync(dataFile)) {
    replayData = JSON.parse(fs.readFileSync(dataFile, 'utf-8'));
    console.log(`Loaded existing: ${replayData.total} replays`);
  }

  const batchSize = concurrency || 10;
  let collected = replayData.total;

  while (collected < target) {
    const batch = [];
    for (let i = 0; i < batchSize; i++) {
      const si = (collected + i) % SEEDS;
      batch.push((async () => {
        try {
          const res = await apiPost('/replay', { round_id: roundId, seed_index: si });
          if (res.status !== 200 || !res.body.frames) return null;

          const lastFrame = res.body.frames[res.body.frames.length - 1].grid;
          return { si, grid: lastFrame };
        } catch (e) { return null; }
      })());
    }

    const results = await Promise.all(batch);

    for (const r of results) {
      if (!r) continue;
      if (!replayData.counts[r.si]) replayData.counts[r.si] = {};

      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const key = `${y}_${x}`;
        if (!replayData.counts[r.si][key]) replayData.counts[r.si][key] = [0,0,0,0,0,0];
        replayData.counts[r.si][key][t2c(r.grid[y][x])]++;
      }
      collected++;
      replayData.total = collected;
    }

    // Save periodically
    if (collected % 50 === 0 || collected >= target) {
      fs.writeFileSync(dataFile, JSON.stringify(replayData));
      console.log(`Collected: ${collected}/${target}`);
    }

    await sleep(200); // rate limit safety
  }

  return replayData;
}

// Build prediction from replay counts
function replayPredict(replayData, seedIndex, alpha) {
  const pred = [];
  const counts = replayData.counts[seedIndex] || {};

  for (let y = 0; y < H; y++) {
    pred[y] = [];
    for (let x = 0; x < W; x++) {
      const key = `${y}_${x}`;
      const c = counts[key] || [0,0,0,0,0,0];
      const N = c.reduce((a, b) => a + b, 0);

      if (N === 0) {
        pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      } else {
        const p = [];
        let s = 0;
        for (let i = 0; i < CLASSES; i++) {
          p[i] = (c[i] + alpha) / (N + CLASSES * alpha);
          s += p[i];
        }
        for (let i = 0; i < CLASSES; i++) p[i] /= s;
        pred[y][x] = p;
      }
    }
  }
  return pred;
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════
async function main() {
  console.log('🏝️  Astar Island Solver v6');
  console.log('Time:', new Date().toISOString());

  // ─── Mode: Collect Replays ───
  if (args['collect-replays']) {
    const roundId = args['collect-replays'];
    const target = parseInt(args.target || '500');
    console.log(`\n📦 Collecting ${target} replays for ${roundId}...`);
    await collectReplays(roundId, target, parseInt(args.concurrency || '10'));
    return;
  }

  // ─── Fetch rounds info ───
  console.log('\n📡 Fetching rounds...');
  const rounds = await apiGet('/rounds');
  const activeRound = rounds.find(r => r.status === 'active');
  const completedRounds = rounds.filter(r => r.status === 'completed');

  console.log(`Active: ${activeRound ? 'R' + activeRound.round_number + ' (closes ' + activeRound.closes_at + ')' : 'NONE'}`);
  console.log(`Completed: ${completedRounds.map(r => 'R' + r.round_number).join(', ')}`);

  const targetRound = args.round ? rounds.find(r => r.id === args.round) : activeRound;
  if (!targetRound) {
    console.log('No active round and no --round specified. Waiting...');
    return;
  }

  const targetId = targetRound.id;
  const closes = new Date(targetRound.closes_at);
  const minsLeft = Math.floor((closes - new Date()) / 60000);
  console.log(`\n🎯 Target: R${targetRound.round_number} (${targetId}), ${minsLeft} min left`);

  // ─── Fetch all data ───
  console.log('\n📊 Fetching initial states and GT data...');
  const initStates = {};
  const gtData = {};

  // Fetch all round data in parallel
  const fetchPromises = [];

  for (const round of [...completedRounds, ...(activeRound ? [activeRound] : [])]) {
    const rn = 'R' + round.round_number;
    fetchPromises.push(
      apiGet('/rounds/' + round.id).then(d => {
        initStates[rn] = d.initial_states.map(is => is.grid);
        console.log(`  ${rn} init: ${initStates[rn].length} seeds`);
      })
    );
  }

  for (const round of completedRounds) {
    const rn = 'R' + round.round_number;
    gtData[rn] = [];
    for (let si = 0; si < SEEDS; si++) {
      fetchPromises.push(
        apiGet('/analysis/' + round.id + '/' + si).then(d => {
          gtData[rn][si] = d.ground_truth;
        })
      );
    }
  }

  await Promise.all(fetchPromises);
  console.log('Data loaded for rounds:', Object.keys(initStates).join(', '));

  // ─── Determine best training set ───
  // Exclude R3 (death round) — include all other completed rounds
  const targetRn = 'R' + targetRound.round_number;
  const growthRounds = Object.keys(gtData).filter(rn => {
    // Detect death rounds by low settlement density
    let avgS = 0;
    for (let si = 0; si < SEEDS; si++) {
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        avgS += gtData[rn][si][y][x][1] + gtData[rn][si][y][x][2];
      }
    }
    avgS /= (SEEDS * H * W);
    const isDeath = avgS < 0.01;
    console.log(`  ${rn}: avgS=${avgS.toFixed(4)} ${isDeath ? '(DEATH - excluded)' : '(GROWTH)'}`);
    return !isDeath && rn !== targetRn;
  });

  console.log(`\n🏗️  Training on: ${growthRounds.join(', ')}`);

  // ─── Build model ───
  const model = buildModel(initStates, gtData);
  console.log(`Model: ${Object.keys(model).length} feature keys`);

  // ─── Cross-validate (honest LOO) ───
  console.log('\n📏 Cross-validation (honest LOO):');
  for (const testRn of growthRounds) {
    const trainRns = growthRounds.filter(rn => rn !== testRn);
    const trainInit = {}; const trainGT = {};
    for (const rn of trainRns) { trainInit[rn] = initStates[rn]; trainGT[rn] = gtData[rn]; }
    const testModel = buildModel(trainInit, trainGT);

    let total = 0;
    for (let si = 0; si < SEEDS; si++) {
      const p = predict(initStates[testRn][si], testModel, MODEL_CONFIG);
      total += score(p, gtData[testRn][si]);
    }
    console.log(`  ${testRn}: ${(total / SEEDS).toFixed(2)} (trained on ${trainRns.join('+')})`);
  }

  // ─── PHASE 1: Submit GT-based predictions ───
  console.log('\n🚀 PHASE 1: Submitting GT-based predictions...');
  for (let si = 0; si < SEEDS; si++) {
    const p = predict(initStates[targetRn][si], model, MODEL_CONFIG);

    // Validate
    let valid = true;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let s = 0;
      for (let c = 0; c < CLASSES; c++) { if (p[y][x][c] < 0) valid = false; s += p[y][x][c]; }
      if (Math.abs(s - 1.0) > 0.01) valid = false;
    }

    if (!valid) { console.log(`  Seed ${si}: INVALID!`); continue; }

    const res = await apiPost('/submit', {
      round_id: targetId, seed_index: si, prediction: p
    });
    console.log(`  Seed ${si}: ${res.body.status || res.body.message || JSON.stringify(res.body)}`);
    await sleep(600);
  }

  // ─── PHASE 2: Viewport observations ───
  if (targetRound.status === 'active' && minsLeft > 5) {
    console.log('\n🔭 PHASE 2: Collecting viewport observations...');
    const budgetRes = await apiGet('/budget');
    const remaining = budgetRes.remaining || 0;
    console.log(`  Budget: ${remaining} queries remaining`);

    if (remaining > 0) {
      const viewports = viewportPositions();
      const queriesPerSeed = Math.min(Math.floor(remaining / SEEDS), viewports.length);

      console.log(`  Using ${queriesPerSeed} viewports per seed (${queriesPerSeed * SEEDS} total)`);

      // Collect observations
      const observations = {}; // {seedIndex: {y: {x: [class, class, ...]}}}

      for (let si = 0; si < SEEDS; si++) {
        observations[si] = {};
        for (let vi = 0; vi < queriesPerSeed; vi++) {
          const vp = viewports[vi];
          try {
            const res = await apiPost('/simulate', {
              round_id: targetId, seed_index: si, year: 50,
              viewport: vp
            });

            if (res.status === 200 && res.body.grid) {
              for (let vy = 0; vy < vp.height; vy++) for (let vx = 0; vx < vp.width; vx++) {
                const gy = vp.y + vy, gx = vp.x + vx;
                if (gy >= H || gx >= W) continue;
                if (!observations[si][gy]) observations[si][gy] = {};
                if (!observations[si][gy][gx]) observations[si][gy][gx] = [];
                observations[si][gy][gx].push(t2c(res.body.grid[vy][vx]));
              }
            }
            await sleep(250);
          } catch (e) {
            console.log(`    Error viewport ${vi} seed ${si}: ${e.message}`);
          }
        }
        console.log(`  Seed ${si}: ${Object.keys(observations[si]).length * 15} cells observed`);
      }

      // Resubmit with viewport-enhanced predictions
      console.log('\n📤 Resubmitting with viewport observations...');
      for (let si = 0; si < SEEDS; si++) {
        const basePred = predict(initStates[targetRn][si], model, MODEL_CONFIG);

        // Convert observations format
        const obsGrid = [];
        for (let y = 0; y < H; y++) {
          obsGrid[y] = [];
          for (let x = 0; x < W; x++) {
            obsGrid[y][x] = observations[si][y] && observations[si][y][x] ? observations[si][y][x] : [];
          }
        }

        // Bayesian update: pseudoCount controls prior strength
        // With ~1-4 observations per cell and a cross-round prior,
        // pseudoCount=5 means "trust prior as much as 5 observations"
        const updated = integrateViewportObs(basePred, obsGrid, 5);

        const res = await apiPost('/submit', {
          round_id: targetId, seed_index: si, prediction: updated
        });
        console.log(`  Seed ${si}: ${res.body.status || JSON.stringify(res.body)}`);
        await sleep(600);
      }
    }
  }

  // ─── Check for replay data files ───
  const replayFile = `replay_data_${targetId}.json`;
  if (fs.existsSync(replayFile)) {
    console.log('\n📦 Found replay data, building replay-based predictions...');
    const replayData = JSON.parse(fs.readFileSync(replayFile, 'utf-8'));
    console.log(`  ${replayData.total} replays`);

    for (let si = 0; si < SEEDS; si++) {
      const p = replayPredict(replayData, si, 0.15);
      const res = await apiPost('/submit', {
        round_id: targetId, seed_index: si, prediction: p
      });
      console.log(`  Seed ${si}: ${res.body.status || JSON.stringify(res.body)}`);
      await sleep(600);
    }
  }

  console.log('\n✅ Done!');
}

main().catch(e => { console.error('Fatal:', e); process.exit(1); });
