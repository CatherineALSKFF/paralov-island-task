#!/usr/bin/env node
/**
 * REPLAY-BASED RESUBMISSION — Our PROVEN approach (93+ on R1 test)
 * Uses replay final grids to build empirical probability distributions.
 * This is fundamentally superior to feature models.
 *
 * Usage: node replay_resubmit.js <JWT> <ROUND_ID> [min_replays_per_seed]
 */
const fs = require('fs'), path = require('path'), https = require('https');
const H = 40, W = 40, SEEDS = 5, C = 6;
const DD = path.join(__dirname, 'data');
const TOKEN = process.argv[2] || '';
const ROUND_ID = process.argv[3] || '';
const MIN_REPLAYS = parseInt(process.argv[4]) || 50;
const BASE = 'https://api.ainm.no/astar-island';

if (!TOKEN || !ROUND_ID) {
  console.log('Usage: node replay_resubmit.js <JWT> <ROUND_ID> [min_replays]');
  process.exit(1);
}

function api(m, p, b) {
  return new Promise((res, rej) => {
    const u = new URL(BASE + p);
    const pl = b ? JSON.stringify(b) : null;
    const o = { hostname: u.hostname, path: u.pathname + u.search, method: m,
      headers: { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' } };
    if (pl) o.headers['Content-Length'] = Buffer.byteLength(pl);
    const r = https.request(o, re => {
      let d = ''; re.on('data', c => d += c);
      re.on('end', () => {
        try { res({ ok: re.statusCode < 300, status: re.statusCode, data: JSON.parse(d) }); }
        catch { res({ ok: false, status: re.statusCode, data: d }); }
      });
    }); r.on('error', rej); if (pl) r.write(pl); r.end();
  });
}
const GET = p => api('GET', p), POST = (p, b) => api('POST', p, b);
const sleep = ms => new Promise(r => setTimeout(r, ms));

// Map terrain type to class index (0-5)
function t2c(t) {
  if (t === 10 || t === 11 || t === 0) return 0; // ocean/plains/empty → class 0
  if (t >= 1 && t <= 5) return t;                  // settlement/port/ruin/forest/mountain
  return 0; // fallback
}

function buildReplayPredictions(initGrid, replays, seedIndex) {
  // Filter replays for this seed
  const seedReplays = replays.filter(r => r.si === seedIndex);
  const N = seedReplays.length;

  if (N === 0) return null;

  console.log(`  Seed ${seedIndex}: ${N} replays`);

  // Count frequencies for each cell
  const counts = [];
  for (let y = 0; y < H; y++) {
    counts[y] = [];
    for (let x = 0; x < W; x++) {
      counts[y][x] = new Float64Array(C);
    }
  }

  for (const rep of seedReplays) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const cls = t2c(rep.finalGrid[y][x]);
        counts[y][x][cls]++;
      }
    }
  }

  // Base alpha scales with replay count
  const baseAlpha = Math.max(0.02, 0.15 * Math.sqrt(150 / N));

  // Build predictions with per-cell adaptive Dirichlet smoothing
  const pred = [];
  for (let y = 0; y < H; y++) {
    pred[y] = [];
    for (let x = 0; x < W; x++) {
      const cellCounts = counts[y][x];
      const initType = initGrid[y][x];

      // Static cells: ocean and mountain
      if (initType === 10) {
        pred[y][x] = [1, 0, 0, 0, 0, 0]; // Always ocean → class 0
        continue;
      }
      if (initType === 5) {
        pred[y][x] = [0, 0, 0, 0, 0, 1]; // Always mountain → class 5
        continue;
      }

      // Count unique classes observed
      let uniqueClasses = 0;
      let maxCount = 0;
      for (let c = 0; c < C; c++) {
        if (cellCounts[c] > 0) uniqueClasses++;
        if (cellCounts[c] > maxCount) maxCount = cellCounts[c];
      }

      // Per-cell adaptive alpha
      let alpha;
      if (uniqueClasses <= 1) {
        alpha = 0.001; // Very confident — all replays agree
      } else if (uniqueClasses === 2) {
        // Two outcomes observed — moderate confidence
        const minorCount = N - maxCount;
        if (minorCount <= 2) {
          alpha = 0.003; // Rare secondary outcome
        } else {
          alpha = 0.01;
        }
      } else {
        // 3+ outcomes — use entropy-scaled alpha
        const tempProbs = [];
        for (let c = 0; c < C; c++) {
          tempProbs[c] = (cellCounts[c] + 0.001) / (N + C * 0.001);
        }
        let entropy = 0;
        for (let c = 0; c < C; c++) {
          if (tempProbs[c] > 0) entropy -= tempProbs[c] * Math.log(tempProbs[c]);
        }
        const maxEntropy = Math.log(C);
        alpha = baseAlpha * Math.min(1, entropy / maxEntropy);
      }

      // Apply Dirichlet smoothing
      const total = N + C * alpha;
      const p = [];
      for (let c = 0; c < C; c++) {
        p[c] = (cellCounts[c] + alpha) / total;
      }

      // Normalize (should already be normalized, but ensure)
      let sum = 0;
      for (let c = 0; c < C; c++) sum += p[c];
      for (let c = 0; c < C; c++) p[c] /= sum;

      pred[y][x] = p;
    }
  }

  return pred;
}

async function main() {
  console.log('=== REPLAY-BASED RESUBMISSION ===');
  console.log('Round:', ROUND_ID);
  console.log('Time:', new Date().toISOString());

  // Find round name from ID
  const { data: rounds } = await GET('/rounds');
  const round = rounds.find(r => r.id === ROUND_ID);
  if (!round) { console.error('Round not found!'); process.exit(1); }
  const rn = `R${round.round_number}`;
  console.log(`Round: ${rn} (${round.status})`);

  // Load replay data
  const replayFile = path.join(DD, `replays_${rn}.json`);
  if (!fs.existsSync(replayFile)) {
    console.error(`No replay file found: ${replayFile}`);
    console.log('Run collect_replays_massive.js first!');
    process.exit(1);
  }

  const replays = JSON.parse(fs.readFileSync(replayFile));
  console.log(`Loaded ${replays.length} replays`);

  // Check per-seed counts
  const seedCounts = [0, 0, 0, 0, 0];
  for (const r of replays) seedCounts[r.si]++;
  console.log('Per-seed:', seedCounts.map((c, i) => `S${i}=${c}`).join(', '));

  const minPerSeed = Math.min(...seedCounts);
  if (minPerSeed < MIN_REPLAYS) {
    console.warn(`WARNING: Only ${minPerSeed} replays for some seeds (min: ${MIN_REPLAYS})`);
  }

  // Load initial grids
  const { data: rd } = await GET('/rounds/' + ROUND_ID);
  const inits = rd.initial_states.map(is => is.grid);
  console.log(`Initial states loaded: ${inits.length} seeds`);

  // Build and submit for each seed
  let totalScore = 0;
  for (let si = 0; si < SEEDS; si++) {
    const pred = buildReplayPredictions(inits[si], replays, si);

    if (!pred) {
      console.log(`Seed ${si}: NO REPLAYS — skipping`);
      continue;
    }

    // Validate
    let valid = true;
    for (let y = 0; y < H && valid; y++) {
      for (let x = 0; x < W && valid; x++) {
        const s = pred[y][x].reduce((a, b) => a + b, 0);
        if (Math.abs(s - 1) > 0.02) {
          console.log(`  Cell (${y},${x}): sum=${s}`);
          valid = false;
        }
        if (pred[y][x].some(v => v < 0)) {
          console.log(`  Cell (${y},${x}): negative value`);
          valid = false;
        }
      }
    }

    if (!valid) {
      console.log(`Seed ${si}: VALIDATION FAILED — skipping`);
      continue;
    }

    // Submit
    const res = await POST('/submit', { round_id: ROUND_ID, seed_index: si, prediction: pred });
    console.log(`Seed ${si}: ${res.ok ? 'ACCEPTED' : 'FAILED'} — ${JSON.stringify(res.data).slice(0, 100)}`);

    if (res.ok && res.data && res.data.score !== undefined) {
      totalScore += res.data.score;
      console.log(`  Score: ${res.data.score}`);
    }

    await sleep(600);
  }

  console.log('\n=== SUBMISSION COMPLETE ===');
  console.log(`Replays used: ${replays.length}`);
  console.log(`Base alpha: ${Math.max(0.02, 0.15 * Math.sqrt(150 / (replays.length / SEEDS))).toFixed(4)}`);
  console.log('Per-cell adaptive smoothing: ON');
}

main().catch(e => console.error('Error:', e.message, e.stack));
