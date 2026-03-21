#!/usr/bin/env node
/**
 * Eval Harness for Astar Island prediction strategies
 *
 * Runs leave-one-out cross-validation against all completed rounds with GT data.
 * Input: a prediction function (as a JS module that exports `predict(initGrid, settlements, model, growthRates)`)
 * Output: per-round scores + average
 *
 * Usage:
 *   node eval_harness.js                          # evaluate baseline strategy
 *   node eval_harness.js ./strategies/mytest.js   # evaluate custom strategy
 */

const fs = require('fs')
const path = require('path')

const DATA_DIR = path.join(__dirname, 'data')
const H = 40, W = 40, FLOOR = 0.001

// ── Load all data ──
function loadGTModel() {
  const file = path.join(DATA_DIR, 'gt_model_buckets.json')
  return JSON.parse(fs.readFileSync(file, 'utf8'))
}

function loadGrowthRates() {
  const file = path.join(DATA_DIR, 'growth_rates.json')
  return JSON.parse(fs.readFileSync(file, 'utf8'))
}

function loadGT(roundPrefix, seed) {
  const file = path.join(DATA_DIR, `gt_${roundPrefix}_s${seed}.json`)
  if (!fs.existsSync(file)) return null
  return JSON.parse(fs.readFileSync(file, 'utf8'))
}

function loadInits(roundNum) {
  const file = path.join(DATA_DIR, `inits_R${roundNum}.json`)
  if (!fs.existsSync(file)) return null
  const raw = JSON.parse(fs.readFileSync(file, 'utf8'))

  // Normalize: could be array of grids, or array of {grid, settlements}
  const result = []
  for (let s = 0; s < 5; s++) {
    const item = raw[s]
    if (!item) { result.push(null); continue }
    if (Array.isArray(item) && Array.isArray(item[0])) {
      // Raw grid - extract settlements from terrain codes
      const grid = item
      const settlements = []
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++)
          if (grid[y][x] === 1 || grid[y][x] === 2) settlements.push({ y, x, has_port: grid[y][x] === 2 })
      result.push({ grid, settlements })
    } else if (item.grid) {
      result.push(item)
    } else { result.push(null) }
  }
  return result
}

// ── Round ID mapping ──
const ROUND_IDS = {
  1: '71451d74', 2: '76909e29', 3: 'f1dac9a9', 4: '8e839974',
  5: 'fd3c92ff', 6: 'ae78003a', 7: '36e581f1', 8: 'c5cdf100',
  9: '2a341ace', 10: '75e625c3', 11: '324fde07', 12: '795bfb1f',
  13: '7b4bda99', 14: 'd0a2c894', 15: 'cc5442dd',
}

// ── Scoring (exact competition formula) ──
function computeScore(prediction, groundTruth) {
  let weightedKL = 0, totalEntropy = 0
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const p = groundTruth[y][x]
      const q = prediction[y][x]
      let ent = 0
      for (let c = 0; c < 6; c++) {
        if (p[c] > 0.001) ent -= p[c] * Math.log(p[c])
      }
      if (ent < 0.01) continue
      let kl = 0
      for (let c = 0; c < 6; c++) {
        if (p[c] > 0.001) kl += p[c] * Math.log(p[c] / Math.max(q[c], 1e-10))
      }
      weightedKL += ent * kl
      totalEntropy += ent
    }
  }
  if (totalEntropy === 0) return 100
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * weightedKL / totalEntropy)))
}

// ── Feature computation ──
function terrainToClass(code) {
  if (code === 10 || code === 11 || code === 0) return 0
  if (code >= 1 && code <= 5) return code
  return 0
}

function getFeatureKey(grid, settPos, y, x) {
  const v = grid[y][x]
  if (v === 10) return 'O'
  if (v === 5) return 'M'
  const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P'
  let nS = 0
  for (let dy = -3; dy <= 3; dy++)
    for (let dx = -3; dx <= 3; dx++) {
      if (!dy && !dx) continue
      const ny = y + dy, nx = x + dx
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx)) nS++
    }
  let coast = false
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y + dy, nx = x + dx
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) coast = true
  }
  return t + (nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3') + (coast ? 'c' : '')
}

// ── Baseline strategy: adaptive K=3 ──
function mergeBuckets(perRoundBuckets, roundNums) {
  const m = {}
  for (const rn of roundNums) {
    const b = perRoundBuckets[String(rn)]
    if (!b) continue
    for (const [k, v] of Object.entries(b)) {
      if (!m[k]) m[k] = { count: 0, sum: [0,0,0,0,0,0] }
      m[k].count += v.count
      for (let c = 0; c < 6; c++) m[k].sum[c] += v.sum[c]
    }
  }
  const out = {}
  for (const [k, v] of Object.entries(m)) out[k] = v.sum.map(s => s / v.count)
  return out
}

function selectClosestRounds(growthRates, targetRate, K = 3) {
  return Object.entries(growthRates)
    .map(([rn, rate]) => ({ rn: parseInt(rn), dist: Math.abs(rate - targetRate) }))
    .sort((a, b) => a.dist - b.dist)
    .slice(0, K)
    .map(c => c.rn)
}

function baselinePredict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config = {}) {
  const K = config.K ?? 3
  const nPrior = config.N_PRIOR ?? 15
  const floor = config.FLOOR ?? 0.001

  // Get growth rate for this round (for LOO, use the actual rate as "oracle" since VP would estimate it)
  const targetGrowth = growthRates[String(testRound)] ?? 0.15

  // Select K closest rounds EXCLUDING the test round
  const candidates = { ...growthRates }
  delete candidates[String(testRound)]
  const closestRounds = selectClosestRounds(candidates, targetGrowth, K)

  // Build model from those rounds
  const adaptiveModel = mergeBuckets(perRoundBuckets, closestRounds)
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound)
  const allModel = mergeBuckets(perRoundBuckets, allRounds)

  const settPos = new Set()
  for (const s of settlements) settPos.add(s.y * W + s.x)

  const pred = []
  for (let y = 0; y < H; y++) {
    const row = []
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x)
      let prior = adaptiveModel[key] ? [...adaptiveModel[key]] : allModel[key] ? [...allModel[key]] : null
      if (!prior) {
        const fb = key.slice(0, -1)
        prior = adaptiveModel[fb] ? [...adaptiveModel[fb]] : allModel[fb] ? [...allModel[fb]] : [1/6,1/6,1/6,1/6,1/6,1/6]
      }
      const floored = prior.map(v => Math.max(v, floor))
      const sum = floored.reduce((a, b) => a + b, 0)
      row.push(floored.map(v => v / sum))
    }
    pred.push(row)
  }
  return pred
}

// ── Run LOO cross-validation ──
function runLOO(predictFn, perRoundBuckets, growthRates, config = {}) {
  const results = []
  const rounds = Object.keys(ROUND_IDS).map(Number)

  for (const testRound of rounds) {
    const prefix = ROUND_IDS[testRound]
    const inits = loadInits(testRound)
    if (!inits) { continue }

    const seedScores = []
    for (let seed = 0; seed < 5; seed++) {
      const gtRaw = loadGT(prefix, seed)
      if (!gtRaw) continue
      const gt = { ground_truth: gtRaw.ground_truth || gtRaw.gt }
      if (!inits[seed]) continue

      const pred = predictFn(
        inits[seed].grid,
        inits[seed].settlements,
        perRoundBuckets,
        growthRates,
        testRound,
        config
      )
      const score = computeScore(pred, gt.ground_truth)
      seedScores.push(score)
    }

    if (seedScores.length > 0) {
      const avg = seedScores.reduce((a, b) => a + b, 0) / seedScores.length
      results.push({ round: testRound, avg, seeds: seedScores, growth: growthRates[String(testRound)] })
    }
  }

  const overallAvg = results.reduce((a, r) => a + r.avg, 0) / results.length
  const overallMin = Math.min(...results.map(r => r.avg))
  return { results, avg: overallAvg, min: overallMin }
}

// ── Main ──
async function main() {
  const strategyFile = process.argv[2]
  const configJson = process.argv[3]

  console.log('Loading data...')
  const perRoundBuckets = loadGTModel()
  const growthRates = loadGrowthRates()

  let predictFn = baselinePredict
  let config = {}

  if (configJson) {
    try { config = JSON.parse(configJson) } catch { config = {} }
  }

  if (strategyFile) {
    try {
      const mod = require(path.resolve(strategyFile))
      predictFn = mod.predict || mod.default || mod
      console.log(`Loaded custom strategy from ${strategyFile}`)
    } catch (e) {
      console.error(`Failed to load strategy: ${e.message}`)
      process.exit(1)
    }
  }

  console.log(`Config: ${JSON.stringify(config)}`)
  console.log('Running LOO cross-validation...\n')

  const { results, avg, min } = runLOO(predictFn, perRoundBuckets, growthRates, config)

  // Print results
  console.log('Round | Score | Growth | Seeds')
  console.log('------|-------|--------|------')
  for (const r of results.sort((a, b) => a.round - b.round)) {
    const growthPct = ((r.growth ?? 0) * 100).toFixed(0).padStart(3)
    const regime = r.growth < 0.03 ? 'death' : r.growth < 0.1 ? 'low' : r.growth < 0.2 ? 'mid' : r.growth < 0.28 ? 'high' : 'boom'
    console.log(`R${String(r.round).padStart(2)}  | ${r.avg.toFixed(1).padStart(5)} | ${growthPct}% ${regime.padEnd(5)} | ${r.seeds.map(s => s.toFixed(0)).join(', ')}`)
  }
  console.log('------|-------|--------|------')
  console.log(`AVG   | ${avg.toFixed(1).padStart(5)} |        |`)
  console.log(`MIN   | ${min.toFixed(1).padStart(5)} |        |`)
  console.log()

  // Output machine-readable result
  const output = { avg, min, results: results.map(r => ({ round: r.round, score: r.avg })), config }
  fs.writeFileSync(path.join(DATA_DIR, 'eval_latest.json'), JSON.stringify(output, null, 2))
  console.log(`Results saved to data/eval_latest.json`)
}

// Export for use as module
module.exports = { runLOO, baselinePredict, computeScore, getFeatureKey, terrainToClass, mergeBuckets, selectClosestRounds, loadGTModel, loadGrowthRates }

if (require.main === module) main().catch(console.error)
