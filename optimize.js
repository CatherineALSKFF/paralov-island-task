#!/usr/bin/env node
/**
 * Agentic Strategy Optimizer
 *
 * Uses Claude to iteratively propose and test prediction strategies.
 * Each iteration:
 * 1. Claude sees current best score + strategy code
 * 2. Claude proposes a modification
 * 3. We eval it via LOO cross-validation
 * 4. If better, it becomes the new best
 *
 * Usage: node optimize.js [iterations]
 */

const fs = require('fs')
const path = require('path')
const { runLOO, baselinePredict, loadGTModel, loadGrowthRates } = require('./eval_harness')

const DATA_DIR = path.join(__dirname, 'data')
const RESULTS_DIR = path.join(__dirname, 'data', 'optimization')
if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR, { recursive: true })

const MAX_ITERATIONS = parseInt(process.argv[2]) || 20

// Current best
let bestScore = 0
let bestConfig = {}
let bestCode = null
let history = []

async function callClaude(prompt) {
  // Use Claude CLI to generate a response
  const { execSync } = require('child_process')
  const escapedPrompt = prompt.replace(/'/g, "'\\''")
  try {
    const result = execSync(
      `echo '${escapedPrompt}' | claude --print --dangerously-skip-permissions`,
      { encoding: 'utf8', timeout: 120000, maxBuffer: 1024 * 1024 }
    )
    return result.trim()
  } catch (e) {
    console.error('Claude call failed:', e.message)
    return null
  }
}

async function main() {
  console.log('Loading data...')
  const perRoundBuckets = loadGTModel()
  const growthRates = loadGrowthRates()

  // Run baseline first
  console.log('\n=== Baseline (K=3, FLOOR=0.001) ===')
  const baseline = runLOO(baselinePredict, perRoundBuckets, growthRates, { K: 3, FLOOR: 0.001 })
  bestScore = baseline.avg
  bestConfig = { K: 3, FLOOR: 0.001 }
  console.log(`Baseline: avg=${baseline.avg.toFixed(1)}, min=${baseline.min.toFixed(1)}`)
  history.push({ iteration: 0, config: bestConfig, avg: baseline.avg, min: baseline.min, status: 'baseline' })

  // Quick parameter sweep first (fast, no Claude needed)
  console.log('\n=== Quick parameter sweep ===')
  const sweepConfigs = [
    { K: 1 }, { K: 2 }, { K: 3 }, { K: 4 }, { K: 5 }, { K: 6 }, { K: 7 },
    { K: 3, FLOOR: 0.0001 }, { K: 3, FLOOR: 0.005 }, { K: 3, FLOOR: 0.01 },
    { K: 4, FLOOR: 0.0001 }, { K: 4, FLOOR: 0.001 }, { K: 4, FLOOR: 0.005 },
    { K: 5, FLOOR: 0.0001 },
  ]

  for (const config of sweepConfigs) {
    const result = runLOO(baselinePredict, perRoundBuckets, growthRates, config)
    const label = JSON.stringify(config)
    const better = result.avg > bestScore
    if (better) {
      bestScore = result.avg
      bestConfig = config
    }
    console.log(`  ${label}: avg=${result.avg.toFixed(1)}, min=${result.min.toFixed(1)} ${better ? '*** NEW BEST ***' : ''}`)
    history.push({ iteration: 'sweep', config, avg: result.avg, min: result.min, status: better ? 'improved' : 'rejected' })
  }

  console.log(`\nBest after sweep: ${JSON.stringify(bestConfig)} avg=${bestScore.toFixed(1)}`)

  // Now try Claude-proposed strategy modifications
  console.log('\n=== Claude-guided optimization ===')

  const baseStrategyCode = fs.readFileSync(path.join(__dirname, 'eval_harness.js'), 'utf8')
  const featureKeyFn = baseStrategyCode.match(/function getFeatureKey[\s\S]*?^}/m)?.[0] || ''
  const predictFn = baseStrategyCode.match(/function baselinePredict[\s\S]*?^}/m)?.[0] || ''

  for (let iter = 1; iter <= MAX_ITERATIONS; iter++) {
    console.log(`\n--- Iteration ${iter}/${MAX_ITERATIONS} ---`)

    const prompt = `You are optimizing a prediction algorithm for a simulation competition. The algorithm predicts terrain probability distributions on a 40x40 grid.

Current best config: ${JSON.stringify(bestConfig)} (LOO avg score: ${bestScore.toFixed(1)})

Score history:
${history.slice(-10).map(h => `  ${JSON.stringify(h.config)}: avg=${h.avg.toFixed(1)} min=${h.min.toFixed(1)} [${h.status}]`).join('\n')}

The prediction algorithm:
1. Uses feature keys based on terrain type, nearby settlement count (radius 3), and coastal flag
2. Selects K closest historical rounds by growth rate
3. Averages their GT probability distributions per feature bucket
4. Applies a floor and normalizes

The feature key function:
${featureKeyFn}

Per-round scores (baseline K=3):
${baseline.results.map(r => `  R${r.round}: ${r.avg.toFixed(1)} (growth ${((growthRates[r.round]||0)*100).toFixed(0)}%)`).join('\n')}

Propose ONE specific modification to improve the score. Output ONLY a JSON object with config parameters to test. Available parameters:
- K: number of closest rounds (currently ${bestConfig.K || 3})
- FLOOR: minimum probability floor (currently ${bestConfig.FLOOR || 0.001})
- N_PRIOR: Bayesian prior weight when VP data exists (currently 15)

You can also propose new parameters that would require code changes - describe them in a "notes" field.

Respond with ONLY valid JSON, no explanation.`

    const response = await callClaude(prompt)
    if (!response) { console.log('  Claude failed, skipping'); continue }

    // Parse Claude's response
    let proposedConfig
    try {
      // Extract JSON from response (Claude might wrap it in markdown)
      const jsonMatch = response.match(/\{[\s\S]*\}/)
      if (!jsonMatch) throw new Error('No JSON found')
      proposedConfig = JSON.parse(jsonMatch[0])
    } catch (e) {
      console.log(`  Failed to parse Claude response: ${e.message}`)
      console.log(`  Response was: ${response.substring(0, 200)}`)
      continue
    }

    console.log(`  Proposed: ${JSON.stringify(proposedConfig)}`)
    if (proposedConfig.notes) {
      console.log(`  Notes: ${proposedConfig.notes}`)
    }

    // Remove non-config keys
    const testConfig = { ...proposedConfig }
    delete testConfig.notes

    // Run eval
    const result = runLOO(baselinePredict, perRoundBuckets, growthRates, testConfig)
    const better = result.avg > bestScore
    if (better) {
      bestScore = result.avg
      bestConfig = testConfig
      console.log(`  Result: avg=${result.avg.toFixed(1)}, min=${result.min.toFixed(1)} *** NEW BEST ***`)
    } else {
      console.log(`  Result: avg=${result.avg.toFixed(1)}, min=${result.min.toFixed(1)} (no improvement)`)
    }

    history.push({
      iteration: iter,
      config: testConfig,
      avg: result.avg,
      min: result.min,
      status: better ? 'improved' : 'rejected',
      notes: proposedConfig.notes,
    })

    // Save progress
    fs.writeFileSync(
      path.join(RESULTS_DIR, 'optimization_history.json'),
      JSON.stringify({ bestConfig, bestScore, history }, null, 2)
    )
  }

  // Final summary
  console.log('\n========================================')
  console.log('OPTIMIZATION COMPLETE')
  console.log(`Best config: ${JSON.stringify(bestConfig)}`)
  console.log(`Best score: ${bestScore.toFixed(1)}`)
  console.log(`Iterations: ${history.length}`)
  console.log('========================================')

  // Save final results
  fs.writeFileSync(
    path.join(RESULTS_DIR, 'best_config.json'),
    JSON.stringify({ config: bestConfig, score: bestScore, history }, null, 2)
  )
}

main().catch(console.error)
