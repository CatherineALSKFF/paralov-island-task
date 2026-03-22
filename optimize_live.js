#!/usr/bin/env node
/**
 * Live VP-aware optimizer for the current round.
 * Tests strategy variations against VP observations as partial ground truth.
 * Submits the best one.
 *
 * Usage: node optimize_live.js <round_id> <token>
 */
const fs = require('fs')
const path = require('path')
const { getFeatureKey, terrainToClass, loadGTModel, loadGrowthRates, weightedMergeBuckets, mergeBuckets, selectClosestRounds, generateFallbacks } = require('./eval_harness')

const DATA_DIR = path.join(__dirname, 'data')
const H = 40, W = 40
const ROUND_ID = process.argv[2]
const TOKEN = process.argv[3]
if (!ROUND_ID || !TOKEN) { console.error('Usage: node optimize_live.js <round_id> <token>'); process.exit(1) }

const PREFIX = ROUND_ID.substring(0, 8)
const API = 'https://api.ainm.no/astar-island'
const AUTH = { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' }

async function fetchJSON(url, opts = {}) {
  const resp = await fetch(url, { headers: AUTH, ...opts })
  return resp.json()
}

// Load all VP data for this round from disk
function loadVPData() {
  const vpFile1 = path.join(DATA_DIR, `viewport_${PREFIX}.json`)
  const vpFile2 = path.join(DATA_DIR, `vp_${PREFIX}.json`)
  const entries = []
  for (const f of [vpFile1, vpFile2]) {
    if (!fs.existsSync(f)) continue
    const raw = JSON.parse(fs.readFileSync(f, 'utf8'))
    for (const [key, val] of Object.entries(raw)) {
      if (!val.grid) continue
      const parts = key.split('_')
      entries.push({
        seed: parseInt(parts[0].replace('s', '')),
        x: val.viewport?.x ?? parseInt(parts[1]),
        y: val.viewport?.y ?? parseInt(parts[2]),
        grid: val.grid
      })
    }
  }
  return entries
}

// Build VP counts per seed
function buildVPCounts(vpEntries, seed) {
  const counts = Array.from({ length: H }, () => Array.from({ length: W }, () => [0,0,0,0,0,0]))
  const total = Array.from({ length: H }, () => Array(W).fill(0))
  for (const vp of vpEntries) {
    if (vp.seed !== seed) continue
    for (let vy = 0; vy < vp.grid.length; vy++)
      for (let vx = 0; vx < vp.grid[vy].length; vx++) {
        const gy = vp.y + vy, gx = vp.x + vx
        if (gy < H && gx < W) {
          counts[gy][gx][terrainToClass(vp.grid[vy][vx])]++
          total[gy][gx]++
        }
      }
  }
  return { counts, total }
}

// Score a prediction against VP observations (partial KL on observed cells only)
function scoreVsVP(pred, vpCounts, vpTotal) {
  let totalKL = 0, totalWeight = 0
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (vpTotal[y][x] === 0) continue
    const nObs = vpTotal[y][x]
    // Empirical distribution from VP
    const emp = vpCounts[y][x].map(c => c / nObs)
    // Only score cells with some uncertainty
    let ent = 0
    for (let c = 0; c < 6; c++) if (emp[c] > 0.01) ent -= emp[c] * Math.log(emp[c])
    if (ent < 0.01) continue
    // KL(emp || pred)
    let kl = 0
    for (let c = 0; c < 6; c++) {
      if (emp[c] > 0.01) kl += emp[c] * Math.log(emp[c] / Math.max(pred[y][x][c], 1e-10))
    }
    totalKL += ent * kl
    totalWeight += ent
  }
  if (totalWeight === 0) return 100
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * totalKL / totalWeight)))
}

// Generate prediction with given config
function predict(initGrid, settlements, buckets, growthRates, config) {
  const { sigma, floor, nPrior, vpCounts, vpTotal, growthOverride } = config
  const settPos = new Set()
  for (const s of settlements) settPos.add(s.y * W + s.x)

  // Estimate growth from VP
  let growth = growthOverride ?? 0.15

  const adaptiveModel = weightedMergeBuckets(buckets, growthRates, growth, sigma)
  const allModel = mergeBuckets(buckets, Object.keys(buckets).map(Number))

  const pred = []
  for (let y = 0; y < H; y++) {
    const row = []
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x)
      let prior = adaptiveModel[key] ? [...adaptiveModel[key]] : allModel[key] ? [...allModel[key]] : null
      if (!prior) {
        const fbs = generateFallbacks(key)
        for (const fb of fbs) {
          prior = adaptiveModel[fb] ? [...adaptiveModel[fb]] : allModel[fb] ? [...allModel[fb]] : null
          if (prior) break
        }
        if (!prior) prior = [1/6,1/6,1/6,1/6,1/6,1/6]
      }
      if (vpCounts && vpTotal && vpTotal[y][x] > 0) {
        const nObs = vpTotal[y][x]
        const q = prior.map((p, c) => nPrior * p + vpCounts[y][x][c])
        const total = nPrior + nObs
        const floored = q.map(v => Math.max(v / total, floor))
        const sum = floored.reduce((a, b) => a + b, 0)
        row.push(floored.map(v => v / sum))
      } else {
        const floored = prior.map(v => Math.max(v, floor))
        const sum = floored.reduce((a, b) => a + b, 0)
        row.push(floored.map(v => v / sum))
      }
    }
    pred.push(row)
  }
  return pred
}

async function submitPrediction(seed, prediction) {
  const resp = await fetch(`${API}/submit`, {
    method: 'POST',
    headers: AUTH,
    body: JSON.stringify({ round_id: ROUND_ID, seed_index: seed, prediction })
  })
  const data = await resp.json()
  return resp.ok ? 'OK' : data.error
}

function writeStatus(data) {
  fs.writeFileSync(path.join(DATA_DIR, 'live_optimizer_status.json'), JSON.stringify({ ...data, updatedAt: new Date().toISOString() }))
}

async function main() {
  console.log(`Live optimizer for round ${ROUND_ID}`)
  console.log('Loading model data...')

  const buckets = loadGTModel()
  const growthRates = loadGrowthRates()

  // Get round info
  const roundInfo = await fetchJSON(`${API}/rounds/${ROUND_ID}`)
  const initStates = roundInfo.initial_states
  if (!initStates) { console.error('No initial states'); process.exit(1) }

  // Save inits
  const initsFile = path.join(DATA_DIR, `inits_R23.json`)
  const inits = initStates.map(s => s?.grid || null)
  fs.writeFileSync(initsFile, JSON.stringify(inits))
  console.log('Saved R23 inits')

  // Build settlements from grids
  const seedData = []
  for (let seed = 0; seed < 5; seed++) {
    const grid = initStates[seed]?.grid
    if (!grid) { seedData.push(null); continue }
    const settlements = []
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
      if (grid[y][x] === 1 || grid[y][x] === 2) settlements.push({ y, x })
    seedData.push({ grid, settlements })
  }

  let bestConfig = { sigma: 0.05, floor: 0.0001, nPrior: 12 }
  let bestScore = 0
  let iteration = 0

  // Config variations to test
  const variations = [
    { sigma: 0.03 }, { sigma: 0.04 }, { sigma: 0.05 }, { sigma: 0.06 }, { sigma: 0.08 }, { sigma: 0.10 },
    { nPrior: 4 }, { nPrior: 6 }, { nPrior: 8 }, { nPrior: 10 }, { nPrior: 12 }, { nPrior: 15 }, { nPrior: 20 },
    { floor: 0.00001 }, { floor: 0.0001 }, { floor: 0.001 }, { floor: 0.005 },
    // Combos
    { sigma: 0.04, nPrior: 8 }, { sigma: 0.04, nPrior: 10 },
    { sigma: 0.05, nPrior: 8 }, { sigma: 0.05, nPrior: 10 },
    { sigma: 0.06, nPrior: 8 }, { sigma: 0.06, nPrior: 10 },
    { sigma: 0.03, nPrior: 6 }, { sigma: 0.08, nPrior: 15 },
  ]

  while (true) {
    // Check if round still active
    const rounds = await fetchJSON(`${API}/rounds`)
    const active = rounds.find(r => r.id === ROUND_ID && r.status === 'active')
    if (!active) { console.log('Round closed'); break }

    const remaining = new Date(active.closes_at).getTime() - Date.now()
    console.log(`\n=== Iteration ${++iteration} (${Math.floor(remaining/60000)}m remaining) ===`)

    // Load VP data
    const vpEntries = loadVPData()
    if (vpEntries.length === 0) {
      console.log('No VP data yet, waiting 10s...')
      await new Promise(r => setTimeout(r, 10000))
      continue
    }
    console.log(`VP data: ${vpEntries.length} viewports`)

    // Estimate growth from VP for each seed
    for (let seed = 0; seed < 5; seed++) {
      const sd = seedData[seed]
      if (!sd) continue
      const { counts, total } = buildVPCounts(vpEntries, seed)

      // Estimate growth: count settlement cells in VP vs initial
      let vpSett = 0, vpTotal2 = 0, initSett = 0
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        if (total[y][x] > 0) {
          vpTotal2++
          if (counts[y][x][1] > 0 || counts[y][x][2] > 0) vpSett++
          if (sd.grid[y][x] === 1 || sd.grid[y][x] === 2) initSett++
        }
      }
      const growthEst = vpTotal2 > 0 ? (vpSett - initSett) / vpTotal2 : 0.15

      // Test all variations
      for (const variation of variations) {
        const config = {
          ...bestConfig,
          ...variation,
          vpCounts: counts,
          vpTotal: total,
          growthOverride: Math.max(0, growthEst),
        }

        const pred = predict(sd.grid, sd.settlements, buckets, growthRates, config)
        const score = scoreVsVP(pred, counts, total)

        if (score > bestScore) {
          bestScore = score
          bestConfig = { sigma: config.sigma, floor: config.floor, nPrior: config.nPrior }
          console.log(`  *** NEW BEST S${seed}: ${score.toFixed(1)} config=${JSON.stringify(bestConfig)}`)
        }
      }
    }

    console.log(`Best config: ${JSON.stringify(bestConfig)} score=${bestScore.toFixed(1)}`)
    writeStatus({ status: 'running', iteration, bestConfig, bestScore, vpCount: vpEntries.length, remaining: Math.floor(remaining/60000) + 'm' })

    // Submit with best config for all seeds
    console.log('Submitting...')
    for (let seed = 0; seed < 5; seed++) {
      const sd = seedData[seed]
      if (!sd) continue
      const { counts, total } = buildVPCounts(vpEntries, seed)

      let vpSett = 0, vpTotal2 = 0, initSett = 0
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        if (total[y][x] > 0) {
          vpTotal2++
          if (counts[y][x][1] > 0 || counts[y][x][2] > 0) vpSett++
          if (sd.grid[y][x] === 1 || sd.grid[y][x] === 2) initSett++
        }
      }

      const pred = predict(sd.grid, sd.settlements, buckets, growthRates, {
        ...bestConfig,
        vpCounts: counts,
        vpTotal: total,
        growthOverride: Math.max(0, (vpSett - initSett) / Math.max(vpTotal2, 1)),
      })

      const result = await submitPrediction(seed, pred)
      console.log(`  S${seed}: ${result}`)
      await new Promise(r => setTimeout(r, 600))
    }

    // Wait before next iteration
    console.log('Waiting 30s...')
    await new Promise(r => setTimeout(r, 30000))
  }
}

main().catch(console.error)
