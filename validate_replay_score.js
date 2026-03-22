#!/usr/bin/env node
/**
 * VALIDATE REPLAY-BASED PREDICTIONS against Ground Truth
 * Proves our replay approach would score 96+ on R8 (and any round with enough replays)
 *
 * Usage: node validate_replay_score.js <JWT> [round_name]
 */
const fs = require('fs'), path = require('path'), https = require('https');
const H = 40, W = 40, SEEDS = 5, C = 6;
const DD = path.join(__dirname, 'data');
const TOKEN = process.argv[2] || '';
const ROUND_NAME = process.argv[3] || 'R8';
const BASE = 'https://api.ainm.no/astar-island';

function api(m, p) {
  return new Promise((res, rej) => {
    const u = new URL(BASE + p);
    const r = https.request({ hostname: u.hostname, path: u.pathname, method: m,
      headers: { 'Authorization': 'Bearer ' + TOKEN } }, re => {
      let d = ''; re.on('data', c => d += c);
      re.on('end', () => { try { res(JSON.parse(d)); } catch { res(null); } });
    }); r.on('error', rej); r.end();
  });
}

function t2c(t) {
  if (t === 10 || t === 11 || t === 0) return 0;
  if (t >= 1 && t <= 5) return t;
  return 0;
}

function buildReplayPred(initGrid, replays, seedIndex) {
  const seedReplays = replays.filter(r => r.si === seedIndex);
  const N = seedReplays.length;
  if (N === 0) return null;

  const counts = [];
  for (let y = 0; y < H; y++) {
    counts[y] = [];
    for (let x = 0; x < W; x++) counts[y][x] = new Float64Array(C);
  }
  for (const rep of seedReplays) {
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++)
        counts[y][x][t2c(rep.finalGrid[y][x])]++;
  }

  const baseAlpha = Math.max(0.02, 0.15 * Math.sqrt(150 / N));
  const pred = [];
  for (let y = 0; y < H; y++) {
    pred[y] = [];
    for (let x = 0; x < W; x++) {
      if (initGrid[y][x] === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
      if (initGrid[y][x] === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }

      const cc = counts[y][x];
      let unique = 0, maxC = 0;
      for (let c = 0; c < C; c++) { if (cc[c] > 0) unique++; if (cc[c] > maxC) maxC = cc[c]; }

      let alpha;
      if (unique <= 1) alpha = 0.001;
      else if (unique === 2) {
        alpha = (N - maxC <= 2) ? 0.003 : 0.01;
      } else {
        const tp = []; let ent = 0;
        for (let c = 0; c < C; c++) { tp[c] = (cc[c] + 0.001) / (N + C * 0.001); }
        for (let c = 0; c < C; c++) if (tp[c] > 0) ent -= tp[c] * Math.log(tp[c]);
        alpha = baseAlpha * Math.min(1, ent / Math.log(C));
      }

      const total = N + C * alpha;
      const p = [];
      for (let c = 0; c < C; c++) p[c] = (cc[c] + alpha) / total;
      let sum = 0;
      for (let c = 0; c < C; c++) sum += p[c];
      for (let c = 0; c < C; c++) p[c] /= sum;
      pred[y][x] = p;
    }
  }
  return { pred, N };
}

function scoreVsGT(pred, gt) {
  // score = 100 * exp(-3 * weighted_kl)
  // weighted_kl = sum(entropy(cell) * KL(GT, pred)) / sum(entropy(cell))
  let totalEntropy = 0, totalWeightedKL = 0;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const gtP = gt[y][x];
      const prP = pred[y][x];

      // Cell entropy
      let entropy = 0;
      for (let c = 0; c < C; c++) {
        if (gtP[c] > 0) entropy -= gtP[c] * Math.log(gtP[c]);
      }
      if (entropy < 0.01) continue; // Skip static cells

      // KL(GT || pred)
      let kl = 0;
      for (let c = 0; c < C; c++) {
        if (gtP[c] > 0) {
          const q = Math.max(prP[c], 1e-10);
          kl += gtP[c] * Math.log(gtP[c] / q);
        }
      }

      totalEntropy += entropy;
      totalWeightedKL += entropy * kl;
    }
  }

  if (totalEntropy === 0) return 100;
  const wkl = totalWeightedKL / totalEntropy;
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl)));
}

async function main() {
  console.log(`=== VALIDATE REPLAY-BASED PREDICTIONS (${ROUND_NAME}) ===`);

  // Load replays
  const replayFile = path.join(DD, `replays_${ROUND_NAME}.json`);
  if (!fs.existsSync(replayFile)) { console.error('No replays!'); process.exit(1); }
  const replays = JSON.parse(fs.readFileSync(replayFile));
  console.log(`Loaded ${replays.length} replays`);

  // Load GT
  const gtFile = path.join(DD, `gt_${ROUND_NAME}.json`);
  let gt = null;
  if (fs.existsSync(gtFile)) {
    gt = JSON.parse(fs.readFileSync(gtFile));
    console.log('GT loaded from disk');
  } else {
    // Need round ID to fetch GT
    const rounds = await api('GET', '/rounds');
    const round = rounds.find(r => `R${r.round_number}` === ROUND_NAME);
    if (!round) { console.error('Round not found!'); process.exit(1); }
    console.log('Fetching GT from API...');
    gt = [];
    for (let si = 0; si < SEEDS; si++) {
      const d = await api('GET', `/analysis/${round.id}/${si}`);
      gt[si] = d.ground_truth;
    }
    fs.writeFileSync(gtFile, JSON.stringify(gt));
    console.log('GT saved to disk');
  }

  // Load inits
  const initFile = path.join(DD, `inits_${ROUND_NAME}.json`);
  let inits;
  if (fs.existsSync(initFile)) {
    inits = JSON.parse(fs.readFileSync(initFile));
  } else {
    const rounds = await api('GET', '/rounds');
    const round = rounds.find(r => `R${r.round_number}` === ROUND_NAME);
    const rd = await api('GET', '/rounds/' + round.id);
    inits = rd.initial_states.map(is => is.grid);
    fs.writeFileSync(initFile, JSON.stringify(inits));
  }

  // Count per-seed
  const seedCounts = [0, 0, 0, 0, 0];
  for (const r of replays) seedCounts[r.si]++;
  console.log('Per-seed:', seedCounts.map((c, i) => `S${i}=${c}`).join(', '));

  // Test with different replay counts
  const testCounts = [50, 100, 200, 300, 500].filter(n => n <= replays.length);

  console.log('\n=== SCORE VS REPLAY COUNT ===');
  for (const maxTotal of testCounts) {
    const maxPerSeed = Math.floor(maxTotal / SEEDS);
    let totalScore = 0;
    const seedScores = [];

    for (let si = 0; si < SEEDS; si++) {
      const seedReps = replays.filter(r => r.si === si).slice(0, maxPerSeed);
      if (seedReps.length === 0) { seedScores.push(0); continue; }

      const { pred } = buildReplayPred(inits[si], [{ si, finalGrid: seedReps[0].finalGrid }, ...seedReps.slice(1).map(r => ({ si, finalGrid: r.finalGrid }))], si);

      // Actually, just pass the limited replays properly
      const limitedReplays = seedReps.map(r => ({ si, finalGrid: r.finalGrid }));
      const result = buildReplayPred(inits[si], limitedReplays, si);
      if (!result) { seedScores.push(0); continue; }

      const score = scoreVsGT(result.pred, gt[si]);
      seedScores.push(score);
      totalScore += score;
    }

    const avg = totalScore / SEEDS;
    console.log(`N=${maxTotal} (${maxPerSeed}/seed): avg=${avg.toFixed(2)} [${seedScores.map(s => s.toFixed(1)).join(', ')}]`);
  }

  // Also test with ALL replays (full count per seed)
  console.log('\n=== FULL REPLAY COUNT ===');
  let totalScore = 0;
  const seedScores = [];
  for (let si = 0; si < SEEDS; si++) {
    const result = buildReplayPred(inits[si], replays, si);
    if (!result) { seedScores.push(0); continue; }
    console.log(`  Seed ${si}: ${result.N} replays`);
    const score = scoreVsGT(result.pred, gt[si]);
    seedScores.push(score);
    totalScore += score;
    console.log(`  Seed ${si}: score=${score.toFixed(4)}`);
  }
  const avg = totalScore / SEEDS;
  console.log(`\nFINAL: avg=${avg.toFixed(2)} [${seedScores.map(s => s.toFixed(2)).join(', ')}]`);

  // Estimate ws
  const roundNum = parseInt(ROUND_NAME.slice(1));
  const weight = Math.pow(1.05, roundNum);
  console.log(`\nWeighted score: ${avg.toFixed(2)} × ${weight.toFixed(4)} = ${(avg * weight).toFixed(2)}`);
  console.log(`\nFor reference: #1 ws=${140.30}, us ws=${132.88}`);
}

main().catch(e => console.error('Error:', e.message, e.stack));
