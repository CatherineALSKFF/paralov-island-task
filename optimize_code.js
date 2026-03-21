#!/usr/bin/env node
/**
 * Code-level Strategy Optimizer
 *
 * Claude writes full predict() functions as JS files, which get tested via LOO.
 * Each strategy file must export: predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config)
 * Returns: 40x40x6 prediction tensor
 *
 * Usage: node optimize_code.js [iterations]
 */

const fs = require('fs')
const path = require('path')
const { execSync } = require('child_process')
const { runLOO, baselinePredict, loadGTModel, loadGrowthRates, computeScore, getFeatureKey, terrainToClass, mergeBuckets, selectClosestRounds } = require('./eval_harness')

const DATA_DIR = path.join(__dirname, 'data')
const STRAT_DIR = path.join(__dirname, 'strategies')
if (!fs.existsSync(STRAT_DIR)) fs.mkdirSync(STRAT_DIR, { recursive: true })

const MAX_ITERATIONS = parseInt(process.argv[2]) || 10

// Save the shared utilities that strategies can import
const SHARED_CODE = `
const H = 40, W = 40;
function terrainToClass(code) {
  if (code === 10 || code === 11 || code === 0) return 0;
  if (code >= 1 && code <= 5) return code;
  return 0;
}
function getFeatureKey(grid, settPos, y, x) {
  const v = grid[y][x];
  if (v === 10) return 'O'; if (v === 5) return 'M';
  const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P';
  let nS = 0;
  for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
    if (!dy && !dx) continue; const ny = y+dy, nx = x+dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny*W+nx)) nS++;
  }
  let coast = false;
  for (const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y+dy, nx = x+dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) coast = true;
  }
  return t + (nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3') + (coast ? 'c' : '');
}
function mergeBuckets(perRoundBuckets, roundNums) {
  const m = {};
  for (const rn of roundNums) {
    const b = perRoundBuckets[String(rn)]; if (!b) continue;
    for (const [k, v] of Object.entries(b)) {
      if (!m[k]) m[k] = { count: 0, sum: [0,0,0,0,0,0] };
      m[k].count += v.count;
      for (let c = 0; c < 6; c++) m[k].sum[c] += v.sum[c];
    }
  }
  const out = {};
  for (const [k, v] of Object.entries(m)) out[k] = v.sum.map(s => s / v.count);
  return out;
}
function selectClosestRounds(growthRates, targetRate, K) {
  return Object.entries(growthRates)
    .map(([rn, rate]) => ({ rn: parseInt(rn), dist: Math.abs(rate - targetRate) }))
    .sort((a, b) => a.dist - b.dist)
    .slice(0, K)
    .map(c => c.rn);
}
module.exports = { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds };
`

fs.writeFileSync(path.join(STRAT_DIR, 'shared.js'), SHARED_CODE)

// Baseline strategy as a file for reference
const BASELINE_CODE = `
const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const K = config.K || 4;
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const candidates = { ...growthRates }; delete candidates[String(testRound)];
  const closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  const adaptiveModel = mergeBuckets(perRoundBuckets, closestRounds);
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);
  const allModel = mergeBuckets(perRoundBuckets, allRounds);
  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);
  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      let prior = adaptiveModel[key] ? [...adaptiveModel[key]] : allModel[key] ? [...allModel[key]] : null;
      if (!prior) {
        const fb = key.slice(0, -1);
        prior = adaptiveModel[fb] ? [...adaptiveModel[fb]] : allModel[fb] ? [...allModel[fb]] : [1/6,1/6,1/6,1/6,1/6,1/6];
      }
      const floored = prior.map(v => Math.max(v, floor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}
module.exports = { predict };
`
fs.writeFileSync(path.join(STRAT_DIR, 'baseline.js'), BASELINE_CODE)

async function callClaude(prompt) {
  const tmpFile = path.join(STRAT_DIR, '_prompt.txt')
  fs.writeFileSync(tmpFile, prompt)
  try {
    const result = execSync(
      `cat "${tmpFile}" | claude --print --dangerously-skip-permissions`,
      { encoding: 'utf8', timeout: 180000, maxBuffer: 2 * 1024 * 1024 }
    )
    return result.trim()
  } catch (e) {
    console.error('Claude call failed:', e.message?.substring(0, 200))
    return null
  }
}

async function main() {
  console.log('Loading data...')
  const perRoundBuckets = loadGTModel()
  const growthRates = loadGrowthRates()

  // Baseline score
  const baseline = runLOO(baselinePredict, perRoundBuckets, growthRates, { K: 4, FLOOR: 0.0001 })
  let bestScore = baseline.avg
  let bestFile = 'baseline.js'
  console.log(`Baseline: avg=${baseline.avg.toFixed(1)}, min=${baseline.min.toFixed(1)}`)
  console.log(`Per-round: ${baseline.results.map(r => `R${r.round}=${r.avg.toFixed(0)}`).join(' ')}`)

  const history = [{ iter: 0, file: 'baseline.js', avg: baseline.avg, min: baseline.min, status: 'baseline' }]

  for (let iter = 1; iter <= MAX_ITERATIONS; iter++) {
    console.log(`\n--- Iteration ${iter}/${MAX_ITERATIONS} ---`)

    const prompt = `You are optimizing a terrain prediction algorithm for a simulation competition.

TASK: Write a better predict() function. The function receives the initial 40x40 grid, settlement positions, per-round GT probability buckets, growth rates, and the test round number. It must return a 40x40x6 probability tensor.

CURRENT BEST (avg LOO score: ${bestScore.toFixed(1)}):
${fs.readFileSync(path.join(STRAT_DIR, bestFile), 'utf8')}

PER-ROUND SCORES:
${baseline.results.map(r => `  R${r.round}: ${r.avg.toFixed(1)} (growth ${((growthRates[r.round]||0)*100).toFixed(0)}%)`).join('\n')}

HISTORY OF ATTEMPTS:
${history.slice(-5).map(h => `  ${h.file}: avg=${h.avg.toFixed(1)} min=${h.min.toFixed(1)} [${h.status}]`).join('\n')}

AVAILABLE UTILITIES (require('./shared')):
- terrainToClass(code), getFeatureKey(grid, settPos, y, x)
- mergeBuckets(perRoundBuckets, roundNums), selectClosestRounds(growthRates, targetRate, K)
- H=40, W=40

RULES:
- Must export { predict }
- predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) -> 40x40 array of 6-element probability vectors
- Each cell's probabilities must sum to ~1.0 and be non-negative
- EXCLUDE testRound from training data (LOO requirement)
- You can access all per-round buckets: perRoundBuckets[roundNum][featureKey] = {count, sum:[6 floats]}
- growthRates[roundNum] = float (0-0.35)

IDEAS TO TRY:
- Weight rounds by growth rate similarity (exponential decay) instead of uniform average
- Richer feature keys (e.g. distance to nearest settlement instead of count in radius)
- Separate models for different terrain types
- Ensemble of multiple K values
- Per-cell adaptive floor based on entropy

Output ONLY the JavaScript code for the strategy file. No markdown, no explanation. Start with require('./shared').`

    const response = await callClaude(prompt)
    if (!response) { console.log('  Claude failed'); continue }

    // Extract code (remove markdown fences if present)
    let code = response.replace(/^```(?:javascript|js)?\n?/gm, '').replace(/```$/gm, '').trim()

    // Save strategy file
    const stratFile = `strategy_${iter}.js`
    const stratPath = path.join(STRAT_DIR, stratFile)
    fs.writeFileSync(stratPath, code)
    console.log(`  Saved: ${stratFile} (${code.length} chars)`)

    // Try to load and eval it
    try {
      // Clear require cache
      delete require.cache[require.resolve(stratPath)]
      const mod = require(stratPath)
      if (typeof mod.predict !== 'function') throw new Error('No predict function exported')

      // Quick sanity check
      const testInits = require(path.join(DATA_DIR, 'inits_R1.json'))
      const testGrid = testInits[0]
      const testSett = []
      for (let y = 0; y < 40; y++) for (let x = 0; x < 40; x++) if (testGrid[y][x] === 1 || testGrid[y][x] === 2) testSett.push({y, x})
      const testPred = mod.predict(testGrid, testSett, perRoundBuckets, growthRates, 1, { K: 4, FLOOR: 0.0001 })
      if (!testPred || testPred.length !== 40 || testPred[0].length !== 40 || testPred[0][0].length !== 6) throw new Error(`Bad shape: ${testPred?.length}x${testPred?.[0]?.length}x${testPred?.[0]?.[0]?.length}`)

      // Run full LOO
      const result = runLOO(mod.predict, perRoundBuckets, growthRates, { K: 4, FLOOR: 0.0001 })
      const better = result.avg > bestScore

      if (better) {
        bestScore = result.avg
        bestFile = stratFile
        console.log(`  Score: avg=${result.avg.toFixed(1)}, min=${result.min.toFixed(1)} *** NEW BEST ***`)
        console.log(`  Per-round: ${result.results.map(r => `R${r.round}=${r.avg.toFixed(0)}`).join(' ')}`)
      } else {
        console.log(`  Score: avg=${result.avg.toFixed(1)}, min=${result.min.toFixed(1)} (no improvement)`)
      }
      history.push({ iter, file: stratFile, avg: result.avg, min: result.min, status: better ? 'improved' : 'rejected' })

    } catch (e) {
      console.log(`  FAILED: ${e.message}`)
      history.push({ iter, file: stratFile, avg: 0, min: 0, status: `error: ${e.message.substring(0, 100)}` })
    }

    // Save progress
    fs.writeFileSync(path.join(STRAT_DIR, 'history.json'), JSON.stringify({ bestFile, bestScore, history }, null, 2))
  }

  console.log('\n========================================')
  console.log('CODE OPTIMIZATION COMPLETE')
  console.log(`Best: ${bestFile} (avg ${bestScore.toFixed(1)})`)
  console.log('========================================')
  console.log(`\nBest strategy code:\n${fs.readFileSync(path.join(STRAT_DIR, bestFile), 'utf8')}`)
}

main().catch(console.error)
