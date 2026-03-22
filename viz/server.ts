import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serve } from '@hono/node-server'
import { readFileSync, readdirSync, existsSync } from 'fs'
import { join } from 'path'
import { spawn, type ChildProcess } from 'child_process'

const DATA_DIR = join(import.meta.dirname, '..', 'data')
const app = new Hono()

app.use('*', cors())

const ROUND_IDS: Record<number, string> = {
  1: '71451d74-be9f-471f-aacd-a41f3b68a9cd',
  2: '76909e29-f664-4b2f-b16b-61b7507277e9',
  3: 'f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb',
  4: '8e839974-b13b-407b-a5e7-fc749d877195',
  5: 'fd3c92ff-3178-4dc9-8d9b-acf389b3982b',
  6: 'ae78003a',
  7: '36e581f1',
  8: 'c5cdf100',
  9: '2a341ace-0f57-4309-9b89-e59fe0f09179',
  10: '75e625c3',
  11: '324fde07',
  12: '795bfb1f',
  13: '7b4bda99',
  14: 'd0a2c894',
  15: 'cc5442dd-bc5d-418b-911b-7eb960cb0390',
  16: '8f664aed-8839-4c85-bed0-77a2cac7c6f5',
  17: '3eb0c25d-28fa-48ca-b8e1-fc249e3918e9',
  18: 'b0f9d1bf-4b71-4e6e-816c-19c718d29056',
  19: '597e60cf-d1a1-4627-ac4d-2a61da68b6df',
  20: 'fd82f643-15e2-40e7-9866-8d8f5157081c',
  21: 'b3a0be6b-b48b-419d-916a-b7a77fa58c4d',
  22: 'a8be24e1-bd48-49bb-aa46-c5593da79f6f',
  23: '93c39605-628f-4706-abd9-08582f8b61d7',
}

// Reverse lookup: id prefix -> round number
const PREFIX_TO_ROUND: Record<string, number> = {}
for (const [num, id] of Object.entries(ROUND_IDS)) {
  PREFIX_TO_ROUND[id.substring(0, 8)] = parseInt(num)
}

// === LOCAL DATA ENDPOINTS (free, no API calls) ===

app.get('/api/gt-model', (c) => {
  const file = join(DATA_DIR, 'gt_model_buckets.json')
  if (!existsSync(file)) return c.json({ error: 'No GT model cached. Run solver_clean.js first.' }, 404)
  return c.json(JSON.parse(readFileSync(file, 'utf8')))
})

app.get('/api/growth-rates', (c) => {
  const file = join(DATA_DIR, 'growth_rates.json')
  if (!existsSync(file)) return c.json({})
  return c.json(JSON.parse(readFileSync(file, 'utf8')))
})

app.get('/api/gt/:roundPrefix/:seed', (c) => {
  const { roundPrefix, seed } = c.req.param()
  const file = join(DATA_DIR, `gt_${roundPrefix}_s${seed}.json`)
  if (!existsSync(file)) return c.json({ error: 'Not found' }, 404)
  return c.json(JSON.parse(readFileSync(file, 'utf8')))
})

app.get('/api/viewport/:roundPrefix', (c) => {
  const { roundPrefix } = c.req.param()
  const file1 = join(DATA_DIR, `viewport_${roundPrefix}.json`)
  const file2 = join(DATA_DIR, `vp_${roundPrefix}.json`)
  const merged: Record<string, unknown> = {}

  // Old format: array of {si, vy, vx, grid} -> normalize to "s{si}_{vx}_{vy}" keys
  if (existsSync(file1)) {
    const raw = JSON.parse(readFileSync(file1, 'utf8'))
    if (Array.isArray(raw)) {
      for (const entry of raw) {
        const key = `s${entry.si}_${entry.vx ?? entry.viewport_x ?? 0}_${entry.vy ?? entry.viewport_y ?? 0}`
        merged[key] = { grid: entry.grid, viewport: { x: entry.vx ?? 0, y: entry.vy ?? 0 }, settlements: entry.settlements }
      }
    } else {
      Object.assign(merged, raw)
    }
  }

  // New format: object with "s{seed}_{x}_{y}" keys (from controlled endpoint)
  if (existsSync(file2)) Object.assign(merged, JSON.parse(readFileSync(file2, 'utf8')))

  if (Object.keys(merged).length === 0) return c.json({ error: 'Not found' }, 404)
  return c.json(merged)
})

// Cached initial states (from solver runs)
app.get('/api/inits/:roundNum', (c) => {
  const { roundNum } = c.req.param()
  const file = join(DATA_DIR, `inits_R${roundNum}.json`)
  if (!existsSync(file)) return c.json({ error: 'Not found' }, 404)
  const raw = JSON.parse(readFileSync(file, 'utf8'))
  // Normalize: ensure each seed has {grid, settlements}
  const normalized: Record<number, { grid: number[][]; settlements: { y: number; x: number }[] }> = {}
  for (let s = 0; s < 5; s++) {
    const item = raw[s]
    if (!item) continue
    if (Array.isArray(item) && Array.isArray(item[0])) {
      // Raw grid - extract settlements
      const grid = item
      const settlements: { y: number; x: number }[] = []
      for (let y = 0; y < 40; y++)
        for (let x = 0; x < 40; x++)
          if (grid[y][x] === 1 || grid[y][x] === 2) settlements.push({ y, x })
      normalized[s] = { grid, settlements }
    } else if (item.grid) {
      normalized[s] = { grid: item.grid, settlements: item.settlements || [] }
    }
  }
  return c.json(normalized)
})

// Cached predictions
app.get('/api/predictions/:roundId', (c) => {
  const { roundId } = c.req.param()
  // Try full ID first, then prefix match
  let file = join(DATA_DIR, `predictions_${roundId}.json`)
  if (!existsSync(file)) {
    const files = readdirSync(DATA_DIR).filter(f => f.startsWith(`predictions_${roundId}`))
    if (files.length > 0) file = join(DATA_DIR, files[0])
    else return c.json({ error: 'Not found' }, 404)
  }
  return c.json(JSON.parse(readFileSync(file, 'utf8')))
})

// Full data inventory
app.get('/api/data-summary', (c) => {
  const files = readdirSync(DATA_DIR)
  const gtRounds = new Set<string>()
  for (const f of files) {
    const m = f.match(/^gt_([a-f0-9]+)_s\d+\.json$/)
    if (m) gtRounds.add(m[1])
  }
  return c.json({
    gtFiles: files.filter(f => /^gt_[a-f0-9]+_s\d+\.json$/.test(f)).length,
    gtRounds: [...gtRounds].map(p => ({ prefix: p, roundNum: PREFIX_TO_ROUND[p] })),
    viewportFiles: files.filter(f => f.startsWith('viewport_')).map(f => {
      const prefix = f.replace('viewport_', '').replace('.json', '')
      return { file: f, prefix, roundNum: PREFIX_TO_ROUND[prefix] }
    }),
    replayFiles: files.filter(f => f.startsWith('replays_')),
    initFiles: files.filter(f => f.startsWith('inits_')).map(f => {
      const rn = f.match(/R(\d+)/)?.[1]
      return { file: f, roundNum: rn ? parseInt(rn) : null }
    }),
    predictionFiles: files.filter(f => f.startsWith('predictions_')),
    hasModel: existsSync(join(DATA_DIR, 'gt_model_buckets.json')),
    hasGrowthRates: existsSync(join(DATA_DIR, 'growth_rates.json')),
    roundIds: ROUND_IDS,
  })
})

// Our submitted predictions (from API, read-only)
app.get('/api/my-predictions/:roundId', async (c) => {
  const { roundId } = c.req.param()
  const token = c.req.header('Authorization')
  if (!token) return c.json({ error: 'No token' }, 401)
  try {
    const resp = await fetch(`https://api.ainm.no/astar-island/my-predictions/${roundId}`, {
      headers: { 'Content-Type': 'application/json', Authorization: token },
    })
    return c.json(await resp.json(), resp.status as 200)
  } catch (e) { return c.json({ error: String(e) }, 500) }
})

// Historical score timeline for a round (computes what score the adaptive model would get with progressively more VP data)
app.get('/api/timeline/:roundPrefix', (c) => {
  const { roundPrefix } = c.req.param()

  // Load VP data
  const vpFile1 = join(DATA_DIR, `viewport_${roundPrefix}.json`)
  const vpFile2 = join(DATA_DIR, `vp_${roundPrefix}.json`)
  const vpEntries: { key: string; seed: number; x: number; y: number; grid: number[][] }[] = []

  if (existsSync(vpFile1)) {
    const raw = JSON.parse(readFileSync(vpFile1, 'utf8'))
    if (Array.isArray(raw)) {
      for (const entry of raw) {
        const key = `s${entry.si}_${entry.vx ?? 0}_${entry.vy ?? 0}`
        vpEntries.push({ key, seed: entry.si, x: entry.vx ?? 0, y: entry.vy ?? 0, grid: entry.grid })
      }
    } else {
      for (const [key, val] of Object.entries(raw)) {
        const parts = key.split('_')
        const v = val as { grid: number[][]; viewport?: { x: number; y: number } }
        if (v.grid) vpEntries.push({ key, seed: parseInt(parts[0].replace('s', '')), x: v.viewport?.x ?? parseInt(parts[1]), y: v.viewport?.y ?? parseInt(parts[2]), grid: v.grid })
      }
    }
  }
  if (existsSync(vpFile2)) {
    const raw = JSON.parse(readFileSync(vpFile2, 'utf8'))
    for (const [key, val] of Object.entries(raw)) {
      if (vpEntries.some(e => e.key === key)) continue
      const parts = key.split('_')
      const v = val as { grid: number[][]; viewport?: { x: number; y: number } }
      if (v.grid) vpEntries.push({ key, seed: parseInt(parts[0].replace('s', '')), x: v.viewport?.x ?? parseInt(parts[1]), y: v.viewport?.y ?? parseInt(parts[2]), grid: v.grid })
    }
  }

  // Load GT model
  const modelFile = join(DATA_DIR, 'gt_model_buckets.json')
  const growthFile = join(DATA_DIR, 'growth_rates.json')
  if (!existsSync(modelFile) || !existsSync(growthFile)) {
    return c.json({ error: 'Model or growth rates not cached' }, 404)
  }

  return c.json({
    vpCount: vpEntries.length,
    vpKeys: vpEntries.map(e => e.key),
    seedCoverage: [0, 1, 2, 3, 4].map(s => ({
      seed: s,
      queried: vpEntries.filter(e => e.seed === s).length,
      total: 9,
    })),
  })
})

// === CONTROLLED DANGEROUS ENDPOINTS (require explicit approval header) ===

app.post('/api/controlled/simulate', async (c) => {
  const approval = c.req.header('X-Human-Approved')
  if (approval !== 'yes-i-clicked-the-button') {
    return c.json({ error: 'Missing human approval header' }, 403)
  }
  const token = c.req.header('Authorization')
  if (!token) return c.json({ error: 'No token' }, 401)
  const body = await c.req.json()
  console.log(`[APPROVED QUERY] seed=${body.seed_index} vp=(${body.viewport_x},${body.viewport_y}) ${body.viewport_w}x${body.viewport_h}`)

  try {
    const resp = await fetch('https://api.ainm.no/astar-island/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: token },
      body: JSON.stringify(body),
    })
    const data = await resp.json()

    // ALWAYS save to disk
    if (resp.ok && data.grid) {
      const vpFile = join(DATA_DIR, `vp_${body.round_id.substring(0, 8)}.json`)
      let existing: Record<string, unknown[]> = {}
      if (existsSync(vpFile)) existing = JSON.parse(readFileSync(vpFile, 'utf8'))
      const key = `s${body.seed_index}_${body.viewport_x}_${body.viewport_y}`
      existing[key] = data
      const { writeFileSync } = await import('fs')
      writeFileSync(vpFile, JSON.stringify(existing))
      console.log(`  Saved VP data to ${vpFile} (key=${key})`)
    }

    return c.json(data, resp.status as 200)
  } catch (e) { return c.json({ error: String(e) }, 500) }
})

app.post('/api/controlled/submit', async (c) => {
  const approval = c.req.header('X-Human-Approved')
  if (approval !== 'yes-i-clicked-the-button') {
    return c.json({ error: 'Missing human approval header' }, 403)
  }
  const token = c.req.header('Authorization')
  if (!token) return c.json({ error: 'No token' }, 401)
  const body = await c.req.json()
  console.log(`[APPROVED SUBMIT] round=${body.round_id} seed=${body.seed_index}`)

  try {
    const resp = await fetch('https://api.ainm.no/astar-island/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: token },
      body: JSON.stringify(body),
    })
    const data = await resp.json()

    // Save prediction to disk
    if (resp.ok) {
      const predFile = join(DATA_DIR, `submitted_${body.round_id.substring(0, 8)}_s${body.seed_index}.json`)
      const { writeFileSync } = await import('fs')
      writeFileSync(predFile, JSON.stringify({ ...body, response: data, timestamp: new Date().toISOString() }))
      console.log(`  Saved submission to ${predFile}`)
    }

    return c.json(data, resp.status as 200)
  } catch (e) { return c.json({ error: String(e) }, 500) }
})

// === SAFE READ-ONLY API PROXY (no budget cost) ===

app.all('/api/ainm/:path{.+}', async (c) => {
  const apiPath = '/' + c.req.param('path')

  // BLOCK direct simulate and submit (must go through /controlled/ endpoints)
  if (apiPath.includes('/simulate') || apiPath.includes('/submit')) {
    console.log(`BLOCKED direct dangerous endpoint: ${apiPath} -- use /api/controlled/ instead`)
    return c.json({
      error: 'Use /api/controlled/simulate or /api/controlled/submit with human approval.',
    }, 403)
  }

  // Only allow GET
  if (c.req.method !== 'GET') {
    console.log(`BLOCKED non-GET to: ${apiPath}`)
    return c.json({ error: 'Only GET requests allowed through viz proxy.' }, 403)
  }

  const token = c.req.header('Authorization')
  const url = `https://api.ainm.no/astar-island${apiPath}`
  try {
    const resp = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: token } : {}),
      },
    })
    const data = await resp.json()
    return c.json(data, resp.status as 200)
  } catch (e) {
    return c.json({ error: String(e) }, 500)
  }
})

// Fetch and save GT data for a round (safe, read-only API call)
app.post('/api/fetch-gt', async (c) => {
  const token = c.req.header('Authorization')
  if (!token) return c.json({ error: 'No token' }, 401)
  const { roundId, seed } = await c.req.json()
  const prefix = roundId.substring(0, 8)
  const file = join(DATA_DIR, `gt_${prefix}_s${seed}.json`)
  if (existsSync(file)) return c.json({ status: 'exists' })

  const resp = await fetch(`https://api.ainm.no/astar-island/analysis/${roundId}/${seed}`, {
    headers: { 'Content-Type': 'application/json', Authorization: token },
  })
  if (!resp.ok) return c.json({ error: `API returned ${resp.status}` }, resp.status as 200)
  const data = await resp.json()
  const { writeFileSync } = await import('fs')
  writeFileSync(file, JSON.stringify(data))
  return c.json({ status: 'saved', file: `gt_${prefix}_s${seed}.json` })
})

// === OPTIMIZER ===

const PROJECT_ROOT = join(import.meta.dirname, '..')
const STRAT_DIR = join(PROJECT_ROOT, 'strategies')

let optimizerProcess: ChildProcess | null = null
let optimizerLog: string[] = []
let optimizerStatus: 'idle' | 'running' | 'done' | 'error' = 'idle'

app.post('/api/optimizer/start', (c) => {
  // Kill existing process if any
  if (optimizerProcess) {
    try { optimizerProcess.kill('SIGTERM') } catch {}
    optimizerProcess = null
  }

  const iterations = 10
  optimizerLog = []
  optimizerStatus = 'running'

  optimizerProcess = spawn('node', [join(PROJECT_ROOT, 'optimize_code.js'), String(iterations)], {
    cwd: PROJECT_ROOT,
    env: { ...process.env, PATH: process.env.PATH },
  })

  optimizerProcess.stdout?.on('data', (data: Buffer) => {
    const lines = data.toString().split('\n').filter(Boolean)
    optimizerLog.push(...lines)
    // Keep last 200 lines
    if (optimizerLog.length > 200) optimizerLog = optimizerLog.slice(-200)
    for (const line of lines) console.log(`[optimizer] ${line}`)
  })

  optimizerProcess.stderr?.on('data', (data: Buffer) => {
    optimizerLog.push(`[stderr] ${data.toString().trim()}`)
  })

  optimizerProcess.on('close', (code) => {
    optimizerStatus = code === 0 ? 'done' : 'error'
    optimizerProcess = null
    console.log(`[optimizer] exited with code ${code}`)
  })

  return c.json({ status: 'started', iterations })
})

app.post('/api/optimizer/stop', (c) => {
  if (!optimizerProcess) return c.json({ error: 'Not running' }, 400)
  optimizerProcess.kill('SIGTERM')
  optimizerProcess = null
  optimizerStatus = 'idle'
  return c.json({ status: 'stopped' })
})

// Live optimizer status (from optimize_live.js)
app.get('/api/live-optimizer/status', (c) => {
  const statusFile = join(DATA_DIR, 'live_optimizer_status.json')
  if (!existsSync(statusFile)) return c.json({ status: 'not running' })
  try {
    const data = JSON.parse(readFileSync(statusFile, 'utf8'))
    return c.json(data)
  } catch { return c.json({ status: 'error reading status' }) }
})

app.get('/api/optimizer/status', (c) => {
  // Load best result if available
  let best = null
  const histFile = join(STRAT_DIR, 'history.json')
  if (existsSync(histFile)) {
    try { best = JSON.parse(readFileSync(histFile, 'utf8')) } catch {}
  }

  return c.json({
    status: optimizerStatus,
    logLines: optimizerLog.length,
    log: optimizerLog.slice(-50),
    best: best ? { file: best.bestFile, score: best.bestScore, iterations: best.history?.length ?? 0 } : null,
  })
})

app.get('/api/optimizer/history', (c) => {
  const histFile = join(STRAT_DIR, 'history.json')
  if (!existsSync(histFile)) return c.json({ error: 'No history' }, 404)
  return c.json(JSON.parse(readFileSync(histFile, 'utf8')))
})

app.get('/api/optimizer/best-strategy', (c) => {
  const histFile = join(STRAT_DIR, 'history.json')
  if (!existsSync(histFile)) return c.json({ error: 'No history' }, 404)
  const hist = JSON.parse(readFileSync(histFile, 'utf8'))
  const bestPath = join(STRAT_DIR, hist.bestFile)
  if (!existsSync(bestPath)) return c.json({ error: 'Best file not found' }, 404)
  return c.json({ file: hist.bestFile, score: hist.bestScore, code: readFileSync(bestPath, 'utf8') })
})

// Eval harness endpoint (run a quick eval with current best or baseline)
app.get('/api/eval/baseline', async (c) => {
  try {
    const { execSync } = await import('child_process')
    const result = execSync('node eval_harness.js', { cwd: PROJECT_ROOT, encoding: 'utf8', timeout: 30000 })
    const evalFile = join(DATA_DIR, 'eval_latest.json')
    if (existsSync(evalFile)) return c.json(JSON.parse(readFileSync(evalFile, 'utf8')))
    return c.json({ output: result })
  } catch (e) { return c.json({ error: String(e) }, 500) }
})

serve({ fetch: app.fetch, port: 3456 }, (info) => {
  console.log(`Backend running on http://localhost:${info.port}`)
  console.log(`Simulate and Submit endpoints are BLOCKED`)
  console.log(`Serving data from ${DATA_DIR}`)
})
