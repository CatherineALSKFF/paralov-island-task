#!/usr/bin/env node
/**
 * Rebuild GT model buckets from individual GT files using the current getFeatureKey function.
 * This must be run whenever getFeatureKey changes.
 */
const fs = require('fs')
const path = require('path')
const { getFeatureKey } = require('./eval_harness')

const DATA_DIR = path.join(__dirname, 'data')
const H = 40, W = 40

const ROUND_IDS = {
  1: '71451d74', 2: '76909e29', 3: 'f1dac9a9', 4: '8e839974',
  5: 'fd3c92ff', 6: 'ae78003a', 7: '36e581f1', 8: 'c5cdf100',
  9: '2a341ace', 10: '75e625c3', 11: '324fde07', 12: '795bfb1f',
  13: '7b4bda99', 14: 'd0a2c894', 15: 'cc5442dd',
  16: '8f664aed', 17: '3eb0c25d', 18: 'b0f9d1bf', 19: '597e60cf', 20: 'fd82f643', 21: 'b3a0be6b', 22: 'a8be24e1',
}

const perRoundBuckets = {}

for (const [rnStr, prefix] of Object.entries(ROUND_IDS)) {
  const rn = parseInt(rnStr)
  const initFile = path.join(DATA_DIR, `inits_R${rn}.json`)
  if (!fs.existsSync(initFile)) { console.log(`Skip R${rn}: no inits`); continue }
  const rawInits = JSON.parse(fs.readFileSync(initFile, 'utf8'))

  const buckets = {}
  let seedCount = 0

  for (let seed = 0; seed < 5; seed++) {
    const gtFile = path.join(DATA_DIR, `gt_${prefix}_s${seed}.json`)
    if (!fs.existsSync(gtFile)) continue
    const gtRaw = JSON.parse(fs.readFileSync(gtFile, 'utf8'))
    const gt = gtRaw.ground_truth || gtRaw.gt
    if (!gt) continue

    // Get init grid for this seed
    const item = rawInits[seed]
    let grid
    if (Array.isArray(item) && Array.isArray(item[0])) {
      grid = item
    } else if (item?.grid) {
      grid = item.grid
    } else continue

    // Extract settlements from grid
    const settPos = new Set()
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++)
        if (grid[y][x] === 1 || grid[y][x] === 2) settPos.add(y * W + x)

    // Build buckets
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const key = getFeatureKey(grid, settPos, y, x)
        if (!buckets[key]) buckets[key] = { count: 0, sum: [0, 0, 0, 0, 0, 0] }
        buckets[key].count++
        const probs = gt[y][x]
        for (let c = 0; c < 6; c++) buckets[key].sum[c] += probs[c]
      }
    }
    seedCount++
  }

  if (seedCount > 0) {
    perRoundBuckets[rn] = buckets
    const keyCount = Object.keys(buckets).length
    console.log(`R${rn}: ${seedCount} seeds, ${keyCount} feature keys`)
  }
}

// Also compute growth rates from GT
const growthRates = {}
for (const [rnStr, prefix] of Object.entries(ROUND_IDS)) {
  const rn = parseInt(rnStr)
  let totalSettProb = 0, totalCells = 0, seedCount = 0
  for (let seed = 0; seed < 5; seed++) {
    const gtFile = path.join(DATA_DIR, `gt_${prefix}_s${seed}.json`)
    if (!fs.existsSync(gtFile)) continue
    const gtRaw = JSON.parse(fs.readFileSync(gtFile, 'utf8'))
    const gt = gtRaw.ground_truth || gtRaw.gt
    if (!gt) continue
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        totalSettProb += gt[y][x][1] // settlement probability
        totalCells++
      }
    seedCount++
  }
  if (seedCount > 0) growthRates[rn] = totalSettProb / totalCells
}

// Save
const outFile = path.join(DATA_DIR, 'gt_model_buckets.json')
const backupFile = path.join(DATA_DIR, 'gt_model_buckets_old.json')
if (fs.existsSync(outFile)) fs.copyFileSync(outFile, backupFile)
fs.writeFileSync(outFile, JSON.stringify(perRoundBuckets))
console.log(`\nSaved ${Object.keys(perRoundBuckets).length} rounds to ${outFile}`)
console.log(`Total feature keys: ${new Set(Object.values(perRoundBuckets).flatMap(b => Object.keys(b))).size}`)

// Save growth rates
fs.writeFileSync(path.join(DATA_DIR, 'growth_rates.json'), JSON.stringify(growthRates, null, 2))
console.log(`Growth rates saved`)

// Show sample keys
const allKeys = new Set()
for (const b of Object.values(perRoundBuckets)) for (const k of Object.keys(b)) allKeys.add(k)
console.log(`\nSample keys: ${[...allKeys].sort().slice(0, 30).join(', ')}`)
