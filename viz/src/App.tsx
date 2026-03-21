import { useState, useEffect, useRef, useMemo } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { RefreshCw, Clock, TrendingUp, Trophy, Shield, Crosshair, Zap, Send, AlertTriangle, CheckCircle2, BarChart3 } from 'lucide-react'

// ── Types ──
interface MyRound {
  id: string; round_number: number; status: string; round_score: number | null
  seed_scores: number[] | null; seeds_submitted: number; rank: number | null
  total_teams: number | null; queries_used: number; queries_max: number
}
type GrowthRates = Record<string, number>
type PerRoundBuckets = Record<string, Record<string, { count: number; sum: number[] }>>

// ── Colors (ainm.no exact) ──
const COLORS = ['#c8b88a', '#d4760a', '#0e7490', '#6b7280', '#2d5a27', '#78716c']
const BG_OCEAN = '#1e3a5f'
const CLASS_NAMES = ['Plains', 'Settlement', 'Port', 'Ruin', 'Forest', 'Mountain']
const CELL = 12, MAP_PX = 40 * CELL

const VP_GRID = [
  { x: 0, y: 0 }, { x: 13, y: 0 }, { x: 25, y: 0 },
  { x: 0, y: 13 }, { x: 13, y: 13 }, { x: 25, y: 13 },
  { x: 0, y: 25 }, { x: 13, y: 25 }, { x: 25, y: 25 },
]

// ── Model building (mirrors solver_clean.js) ──
const N_PRIOR = 15 // Bayesian prior weight (calibrated via replay analysis)
const FLOOR = 0.0001 // optimized from 0.001
const K_NEAREST = 4 // optimized from 3

function mergeBuckets(buckets: PerRoundBuckets, roundNums: number[]): Record<string, number[]> {
  const m: Record<string, { count: number; sum: number[] }> = {}
  for (const rn of roundNums) {
    const b = buckets[String(rn)]; if (!b) continue
    for (const [k, v] of Object.entries(b)) {
      if (!m[k]) m[k] = { count: 0, sum: [0, 0, 0, 0, 0, 0] }
      m[k].count += v.count; for (let c = 0; c < 6; c++) m[k].sum[c] += v.sum[c]
    }
  }
  const out: Record<string, number[]> = {}
  for (const [k, v] of Object.entries(m)) out[k] = v.sum.map(s => s / v.count)
  return out
}

// Estimate growth rate from VP observations (solver_clean.js logic)
function estimateGrowthFromVP(
  vpViewports: { x: number; y: number; grid: number[][] }[],
  initGrid: number[][],
  settlements: { y: number; x: number }[]
): number {
  const settPos = new Set<number>()
  if (!settlements) return 0.15
  for (const s of settlements) settPos.add(s.y * 40 + s.x)

  let settCount = 0, dynamicCount = 0
  // Assemble VP into full grid
  const vpFull: (number | null)[][] = Array.from({ length: 40 }, () => Array(40).fill(null))
  for (const vp of vpViewports) {
    for (let vy = 0; vy < vp.grid.length; vy++)
      for (let vx = 0; vx < vp.grid[vy].length; vx++) {
        const gy = vp.y + vy, gx = vp.x + vx
        if (gy < 40 && gx < 40) vpFull[gy][gx] = vp.grid[vy][vx]
      }
  }

  for (let y = 0; y < 40; y++) for (let x = 0; x < 40; x++) {
    if (vpFull[y][x] == null) continue
    const init = initGrid[y][x]
    if (init === 10 || init === 5) continue
    // Check if near a settlement (radius 5)
    let nearSett = false
    for (let dy = -5; dy <= 5 && !nearSett; dy++)
      for (let dx = -5; dx <= 5 && !nearSett; dx++) {
        const ny = y + dy, nx = x + dx
        if (ny >= 0 && ny < 40 && nx >= 0 && nx < 40 && settPos.has(ny * 40 + nx)) nearSett = true
      }
    if (!nearSett) continue
    dynamicCount++
    const cls = terrainToClass(vpFull[y][x]!)
    if (cls === 1 || cls === 2) settCount++
  }
  return dynamicCount > 0 ? settCount / dynamicCount : 0.15
}

// Select K closest rounds by growth rate (solver_clean.js logic)
function selectClosestRounds(growthRates: GrowthRates, targetRate: number, K = 3): number[] {
  return Object.entries(growthRates)
    .map(([rn, rate]) => ({ rn: parseInt(rn), dist: Math.abs(rate - targetRate) }))
    .sort((a, b) => a.dist - b.dist)
    .slice(0, K)
    .map(c => c.rn)
}

// Compute optimal query order across all seeds
// Priority: maximize unobserved dynamic cells (near settlements) and spread across seeds
function computeQueryPlan(
  vpDone: Set<string>,
  allInitData: Map<number, { grid: number[][]; settlements: { y: number; x: number }[] }>,
  budget: number
): { seed: number; x: number; y: number; score: number; reason: string }[] {
  const candidates: { seed: number; x: number; y: number; score: number; reason: string }[] = []

  for (let seed = 0; seed < 5; seed++) {
    const init = allInitData.get(seed)
    if (!init?.settlements || !init?.grid) continue
    const settPos = new Set<number>()
    for (const s of init.settlements) settPos.add(s.y * 40 + s.x)

    // Count how many viewports this seed already has
    const seedDone = VP_GRID.filter(({ x, y }) => vpDone.has(`s${seed}_${x}_${y}`)).length

    for (const vp of VP_GRID) {
      if (vpDone.has(`s${seed}_${vp.x}_${vp.y}`)) continue

      // Count dynamic cells in this viewport (near settlements, not ocean/mountain)
      let dynamicCells = 0
      let settlementCells = 0
      for (let vy = 0; vy < 15; vy++) for (let vx = 0; vx < 15; vx++) {
        const gy = vp.y + vy, gx = vp.x + vx
        if (gy >= 40 || gx >= 40) continue
        const terrain = init.grid[gy][gx]
        if (terrain === 10 || terrain === 5) continue // skip ocean/mountain
        // Near a settlement?
        let near = false
        for (let dy = -5; dy <= 5 && !near; dy++)
          for (let dx = -5; dx <= 5 && !near; dx++) {
            const ny = gy + dy, nx = gx + dx
            if (ny >= 0 && ny < 40 && nx >= 0 && nx < 40 && settPos.has(ny * 40 + nx)) near = true
          }
        if (near) dynamicCells++
        if (terrain === 1 || terrain === 2) settlementCells++
      }

      // Score: prioritize seeds with 0 observations (cross-seed spread),
      // then viewports with more dynamic cells
      const spreadBonus = seedDone === 0 ? 1000 : seedDone === 1 ? 100 : 0
      const score = spreadBonus + dynamicCells * 10 + settlementCells * 5

      const reason = seedDone === 0
        ? `First query for S${seed} (${dynamicCells} dynamic cells)`
        : `S${seed}: ${dynamicCells} dynamic, ${settlementCells} settlement cells`

      candidates.push({ seed, x: vp.x, y: vp.y, score, reason })
    }
  }

  return candidates.sort((a, b) => b.score - a.score).slice(0, budget)
}

// ── Helpers ──
function terrainToClass(code: number) { return (code === 10 || code === 11 || code === 0) ? 0 : (code >= 1 && code <= 5) ? code : 0 }

function getFeatureKey(grid: number[][], sp: Set<number>, y: number, x: number): string {
  const v = grid[y][x]
  if (v === 10) return 'O'; if (v === 5) return 'M'
  const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P'
  let nS = 0
  for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) { if (!dy && !dx) continue; const ny = y+dy, nx = x+dx; if (ny >= 0 && ny < 40 && nx >= 0 && nx < 40 && sp.has(ny*40+nx)) nS++ }
  let coast = false
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) { const ny = y+dy, nx = x+dx; if (ny >= 0 && ny < 40 && nx >= 0 && nx < 40 && grid[ny][nx] === 10) coast = true }
  return t + (nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3') + (coast ? 'c' : '')
}

function buildPrediction(model: Record<string, number[]>, grid: number[][], sett: { y: number; x: number }[]) {
  const sp = new Set<number>(); for (const s of sett) sp.add(s.y*40+s.x)
  const pred = Array.from({ length: 40 }, () => Array.from({ length: 40 }, () => [0,0,0,0,0,0]))
  for (let y = 0; y < 40; y++) for (let x = 0; x < 40; x++) { const m = model[getFeatureKey(grid, sp, y, x)]; if (m) pred[y][x] = [...m]; else { const c = terrainToClass(grid[y][x]); pred[y][x][c] = 1 } }
  return pred
}

// ── API helpers ──
const authHeaders = (token: string) => ({ Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' })
const fetcher = (url: string, token?: string) => fetch(url, token ? { headers: authHeaders(token) } : {}).then(r => r.ok ? r.json() : null)

// ── Components ──
function GrowthBar({ rate, roundNum }: { rate: number; roundNum: number }) {
  const pct = Math.min(rate / 0.35 * 100, 100)
  const bg = rate < 0.03 ? 'bg-red-500' : rate < 0.1 ? 'bg-orange-500' : rate < 0.2 ? 'bg-yellow-500' : rate < 0.28 ? 'bg-green-500' : 'bg-blue-500'
  const label = rate < 0.03 ? 'death' : rate < 0.1 ? 'low' : rate < 0.2 ? 'mid' : rate < 0.28 ? 'high' : 'boom'
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-7 text-muted-foreground font-mono text-right">R{roundNum}</span>
      <div className="flex-1 h-3 bg-muted/50 rounded-sm overflow-hidden"><div className={`h-full ${bg}`} style={{ width: `${pct}%` }} /></div>
      <span className="w-11 text-right font-mono text-[11px]">{(rate * 100).toFixed(1)}%</span>
      <span className="w-9 text-muted-foreground text-[10px]">{label}</span>
    </div>
  )
}

function MapCanvas({ data, mode, size = MAP_PX, initGrid }: { data: number[][][] | number[][] | null; mode: 'initial' | 'prob'; size?: number; initGrid?: number[][] }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current; if (!c || !data) return
    const ctx = c.getContext('2d')!; c.width = MAP_PX; c.height = MAP_PX
    ctx.fillStyle = '#0a0a0a'; ctx.fillRect(0, 0, MAP_PX, MAP_PX)
    for (let y = 0; y < 40; y++) for (let x = 0; x < 40; x++) {
      const px = x * CELL, py = y * CELL
      if (mode === 'initial') { ctx.globalAlpha = 1; ctx.fillStyle = (data as number[][])[y][x] === 10 ? BG_OCEAN : COLORS[terrainToClass((data as number[][])[y][x])] }
      else { const p = (data as number[][][])[y][x]; if (!p) continue; const mx = Math.max(...p), idx = p.indexOf(mx); ctx.fillStyle = (idx === 0 && initGrid?.[y][x] === 10) ? BG_OCEAN : COLORS[idx]; ctx.globalAlpha = 0.3 + 0.7 * mx }
      ctx.fillRect(px, py, CELL, CELL)
    }
    ctx.globalAlpha = 1
  }, [data, mode, initGrid])
  if (!data) return <Skeleton className="rounded" style={{ width: size, height: size }} />
  return <canvas ref={ref} className="rounded border border-zinc-800/30" style={{ width: size, height: size, imageRendering: 'pixelated' }} />
}


// ══════════════════════════════════════════
//   MAIN APP
// ══════════════════════════════════════════

export default function App() {
  const qc = useQueryClient()
  const [selectedRound, setSelectedRound] = useState('17')
  const [selectedSeed, setSelectedSeed] = useState(0)
  const [token, setToken] = useState(() => localStorage.getItem('jwt') || '')
  const [activeTab, setActiveTab] = useState('play')
  const [countdown, setCountdown] = useState('')
  const [log, setLog] = useState<string[]>([])
  const [vpDone, setVpDone] = useState<Set<string>>(new Set())

  // Step-by-step execution state
  const [stepLog, setStepLog] = useState<{ label: string; status: 'done' | 'error'; detail?: string }[]>([])
  const [stepBusy, setStepBusy] = useState(false)

  // ── Queries ──
  const { data: summary } = useQuery({ queryKey: ['data-summary'], queryFn: () => fetcher('/api/data-summary') })
  const { data: growthRates = {} as GrowthRates } = useQuery({ queryKey: ['growth-rates'], queryFn: () => fetcher('/api/growth-rates') })
  const { data: rawBuckets } = useQuery<PerRoundBuckets | null>({ queryKey: ['gt-model'], queryFn: () => fetcher('/api/gt-model').then(d => d?.error ? null : d) })

  const { data: liveRounds = [] } = useQuery({
    queryKey: ['live-rounds', token], queryFn: () => fetcher('/api/ainm/rounds', token), enabled: !!token, refetchInterval: 60_000,
  })
  const { data: myRounds = [] as MyRound[] } = useQuery({
    queryKey: ['my-rounds', token], queryFn: () => fetcher('/api/ainm/my-rounds', token), enabled: !!token, refetchInterval: 30_000,
  })
  // Load existing VP data from disk
  const activeRoundId = liveRounds.find((r: { status: string }) => r.status === 'active')?.id
  useQuery({
    queryKey: ['vp-done', activeRoundId],
    queryFn: async () => {
      if (!activeRoundId) return []
      const prefix = activeRoundId.substring(0, 8)
      const resp = await fetch(`/api/viewport/${prefix}`)
      if (!resp.ok) return []
      const data = await resp.json()
      setVpDone(new Set(Object.keys(data)))
      return Object.keys(data)
    },
    enabled: !!activeRoundId,
  })

  const { data: leaderboard = [] } = useQuery({
    queryKey: ['leaderboard', token], queryFn: () => fetcher('/api/ainm/leaderboard', token).then(d => d?.slice(0, 30) ?? []), enabled: !!token,
  })

  const roundId = summary?.roundIds?.[parseInt(selectedRound)]
  const { data: initData } = useQuery({
    queryKey: ['init-data', selectedRound, selectedSeed, token],
    queryFn: async () => {
      const rn = parseInt(selectedRound)
      const local = await fetcher(`/api/inits/${rn}`)
      if (local?.[selectedSeed]) return { grid: local[selectedSeed].grid, settlements: local[selectedSeed].settlements }
      if (!token || !roundId) return null
      const rd = await fetcher(`/api/ainm/rounds/${roundId}`, token)
      if (rd?.initial_states?.[selectedSeed]) return { grid: rd.initial_states[selectedSeed].grid, settlements: rd.initial_states[selectedSeed].settlements }
      return null
    },
    enabled: !!summary,
  })

  // All seeds init data (for query planning)
  const { data: allSeedsInit } = useQuery({
    queryKey: ['all-seeds-init', selectedRound, token],
    queryFn: async () => {
      const rn = parseInt(selectedRound)
      const rid = summary?.roundIds?.[rn]
      if (!rid) return new Map<number, { grid: number[][]; settlements: { y: number; x: number }[] }>()
      const m = new Map<number, { grid: number[][]; settlements: { y: number; x: number }[] }>()
      // Try local cache first
      const local = await fetcher(`/api/inits/${rn}`)
      if (local) {
        for (let s = 0; s < 5; s++) if (local[s]) m.set(s, { grid: local[s].grid, settlements: local[s].settlements })
        if (m.size === 5) return m
      }
      // Fallback to API
      if (token) {
        const rd = await fetcher(`/api/ainm/rounds/${rid}`, token)
        if (rd?.initial_states) {
          for (let s = 0; s < 5; s++) if (rd.initial_states[s]) m.set(s, { grid: rd.initial_states[s].grid, settlements: rd.initial_states[s].settlements })
        }
      }
      return m
    },
    enabled: !!summary,
  })

  // VP observation data for current round - ALL seeds (used for predictions + timeline)
  const { data: allVpData } = useQuery({
    queryKey: ['vp-observations', activeRoundId],
    queryFn: async () => {
      if (!activeRoundId) return null
      const prefix = activeRoundId.substring(0, 8)
      const resp = await fetch(`/api/viewport/${prefix}`)
      if (!resp.ok) return null
      const allData = await resp.json()
      if (allData.error) return null
      // Parse into per-seed viewport arrays
      const perSeed: Record<number, { x: number; y: number; grid: number[][] }[]> = {}
      for (const [key, val] of Object.entries(allData)) {
        const parts = key.split('_')
        const seed = parseInt(parts[0].replace('s', ''))
        const v = val as { grid?: number[][]; viewport?: { x: number; y: number } }
        if (!v.grid) continue
        if (!perSeed[seed]) perSeed[seed] = []
        perSeed[seed].push({ x: v.viewport?.x ?? parseInt(parts[1]), y: v.viewport?.y ?? parseInt(parts[2]), grid: v.grid })
      }
      return perSeed
    },
    enabled: !!activeRoundId,
  })

  // VP observations for the currently selected seed (convenience)
  const vpObservations = allVpData?.[selectedSeed] ?? null

  const { data: gtData } = useQuery({
    queryKey: ['gt-data', roundId, selectedSeed],
    queryFn: () => roundId ? fetcher(`/api/gt/${roundId.substring(0, 8)}/${selectedSeed}`).then(d => d?.ground_truth ?? null) : null,
    enabled: !!roundId,
  })


  // ── Adaptive prediction (mirrors solver_clean.js exactly) ──
  const { prediction, adaptiveInfo } = useMemo(() => {
    if (!rawBuckets || !initData || !initData.settlements) return { prediction: null, adaptiveInfo: null }

    // Step 1: Estimate growth rate from VP (or default 0.15 if no VP)
    let estimatedGrowth = 0.15
    let hasVP = false
    if (vpObservations && vpObservations.length > 0) {
      estimatedGrowth = estimateGrowthFromVP(vpObservations, initData.grid, initData.settlements)
      hasVP = true
    }

    // Step 2: Select K=3 closest rounds by growth rate
    const closestRounds = selectClosestRounds(growthRates, estimatedGrowth, K_NEAREST)

    // Step 3: Build adaptive model + all-rounds fallback
    const adaptiveModel = mergeBuckets(rawBuckets, closestRounds)
    const allModel = mergeBuckets(rawBuckets, Object.keys(rawBuckets).map(Number))

    // Step 4: Assemble VP grid
    let vpGrid: (number | null)[][] | null = null
    if (vpObservations && vpObservations.length > 0) {
      vpGrid = Array.from({ length: 40 }, () => Array(40).fill(null))
      for (const vp of vpObservations) {
        for (let vy = 0; vy < vp.grid.length; vy++)
          for (let vx = 0; vx < vp.grid[vy].length; vx++) {
            const gy = vp.y + vy, gx = vp.x + vx
            if (gy < 40 && gx < 40) vpGrid![gy][gx] = vp.grid[vy][vx]
          }
      }
    }

    // Step 5: Generate prediction with Bayesian VP update
    const sp = new Set<number>()
    for (const s of initData.settlements) sp.add(s.y * 40 + s.x)
    const pred = Array.from({ length: 40 }, () => Array.from({ length: 40 }, () => [0, 0, 0, 0, 0, 0]))

    for (let y = 0; y < 40; y++) for (let x = 0; x < 40; x++) {
      const key = getFeatureKey(initData.grid, sp, y, x)
      let prior = adaptiveModel[key] ? [...adaptiveModel[key]] : allModel[key] ? [...allModel[key]] : null
      if (!prior) {
        const fb = key.slice(0, -1)
        prior = adaptiveModel[fb] ? [...adaptiveModel[fb]] : allModel[fb] ? [...allModel[fb]] : [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
      }

      if (vpGrid && vpGrid[y][x] != null) {
        const obsClass = terrainToClass(vpGrid[y][x]!)
        const q = prior.map((p, c) => N_PRIOR * p + (c === obsClass ? 1 : 0))
        const total = N_PRIOR + 1
        const floored = q.map(v => Math.max(v / total, FLOOR))
        const sum = floored.reduce((a, b) => a + b, 0)
        pred[y][x] = floored.map(v => v / sum)
      } else {
        const floored = prior.map(v => Math.max(v, FLOOR))
        const sum = floored.reduce((a, b) => a + b, 0)
        pred[y][x] = floored.map(v => v / sum)
      }
    }

    const growthLabel = estimatedGrowth < 0.03 ? 'death' : estimatedGrowth < 0.1 ? 'low' : estimatedGrowth < 0.2 ? 'mid' : estimatedGrowth < 0.28 ? 'high' : 'boom'

    return {
      prediction: pred,
      adaptiveInfo: {
        estimatedGrowth,
        growthLabel,
        closestRounds,
        hasVP,
        vpCells: vpGrid ? vpGrid.flat().filter(v => v != null).length : 0,
      },
    }
  }, [rawBuckets, initData, growthRates, vpObservations])

  // ── Derived ──
  const myScores: Record<number, number> = {}; for (const r of myRounds) if (r.round_score != null) myScores[r.round_number] = r.round_score
  const activeRound = liveRounds.find((r: { status: string }) => r.status === 'active')
  const selectedMyRound = myRounds.find((r: MyRound) => r.round_number === parseInt(selectedRound))
  const sortedGrowth = Object.entries(growthRates).map(([k, v]) => ({ round: +k, rate: v })).sort((a, b) => a.rate - b.rate)
  const nextQuery = VP_GRID.find(({ x, y }) => !vpDone.has(`s${selectedSeed}_${x}_${y}`))
  // busy is defined after autopilot state below
  const saveToken = (t: string) => { setToken(t); localStorage.setItem('jwt', t) }

  // Timeline: how predictions evolved as VP queries were added
  const timeline = useMemo(() => {
    if (!rawBuckets || !initData || !initData.settlements || !vpObservations || vpObservations.length === 0) return null
    const points: { queryNum: number; vpCells: number; growthEst: number; growthLabel: string; trainingRounds: number[]; seedsObserved: number }[] = []

    // Step 0: model-only (no VP)
    const defaultGrowth = 0.15
    const defaultRounds = selectClosestRounds(growthRates, defaultGrowth, 3)
    points.push({ queryNum: 0, vpCells: 0, growthEst: defaultGrowth, growthLabel: 'mid', trainingRounds: defaultRounds, seedsObserved: 0 })

    // Progressive VP additions
    const accum: { x: number; y: number; grid: number[][] }[] = []
    const seenSeeds = new Set<number>()

    // Group VP by order they were likely queried (we don't have timestamps, use key order)
    const sorted = [...vpObservations]

    for (let i = 0; i < sorted.length; i++) {
      accum.push(sorted[i])
      // We don't know which seed each VP belongs to from the observation list alone,
      // but we can count cells
      let totalCells = 0
      for (const vp of accum) {
        for (let vy = 0; vy < vp.grid.length; vy++)
          for (let vx = 0; vx < vp.grid[vy].length; vx++) {
            const gy = vp.y + vy, gx = vp.x + vx
            if (gy < 40 && gx < 40) totalCells++
          }
      }

      const growth = estimateGrowthFromVP(accum, initData.grid, initData.settlements)
      const rounds = selectClosestRounds(growthRates, growth, K_NEAREST)
      const label = growth < 0.03 ? 'death' : growth < 0.1 ? 'low' : growth < 0.2 ? 'mid' : growth < 0.28 ? 'high' : 'boom'

      points.push({
        queryNum: i + 1,
        vpCells: totalCells,
        growthEst: growth,
        growthLabel: label,
        trainingRounds: rounds,
        seedsObserved: seenSeeds.size || 1,
      })
    }

    return points
  }, [rawBuckets, initData, vpObservations, growthRates])

  // Auto-select active round
  useEffect(() => {
    const active = liveRounds.find((r: { status: string }) => r.status === 'active')
    if (active) setSelectedRound(String(active.round_number))
  }, [liveRounds])

  // Countdown
  useEffect(() => {
    if (!activeRound?.closes_at) { setCountdown(''); return }
    const tick = () => { const d = new Date(activeRound.closes_at).getTime() - Date.now(); if (d <= 0) { setCountdown('ENDED'); return }; setCountdown(`${Math.floor(d/3600000)}h ${Math.floor((d%3600000)/60000)}m ${Math.floor((d%60000)/1000)}s`) }
    tick(); const iv = setInterval(tick, 1000); return () => clearInterval(iv)
  }, [activeRound])


  // Autopilot state
  const [autopilotRunning, setAutopilotRunning] = useState(false)
  const [autopilotPhase, setAutopilotPhase] = useState('')
  const autopilotRef = useRef(false) // for cancellation
  const busy = stepBusy || autopilotRunning

  // Submit all 5 seeds with per-seed predictions
  const submitAllSeeds = async (roundId: string, label: string) => {
    let ok = 0
    for (let seed = 0; seed < 5; seed++) {
      const seedPred = buildPredForSeed(seed)
      if (!seedPred) { setLog(l => [...l, `Skip S${seed}: no init data`]); continue }
      try {
        const resp = await fetch('/api/controlled/submit', {
          method: 'POST',
          headers: { ...authHeaders(token), 'X-Human-Approved': 'yes-i-clicked-the-button' },
          body: JSON.stringify({ round_id: roundId, seed_index: seed, prediction: seedPred }),
        })
        const data = await resp.json()
        if (resp.ok) { ok++; setLog(l => [...l, `Submitted S${seed} (${label})`]) }
        else setLog(l => [...l, `Submit FAIL S${seed}: ${data.error}`])
        await new Promise(r => setTimeout(r, 550))
      } catch (e) { setLog(l => [...l, `Submit ERROR: ${e}`]) }
    }
    setStepLog(l => [...l, { label: `Submit (${label})`, status: ok === 5 ? 'done' : 'error', detail: `${ok}/5` }])
    qc.invalidateQueries({ queryKey: ['my-rounds'] })
    return ok
  }

  // Execute one query (returns success)
  const execQuery = async (roundId: string, seed: number, x: number, y: number, reason: string) => {
    try {
      const resp = await fetch('/api/controlled/simulate', {
        method: 'POST',
        headers: { ...authHeaders(token), 'X-Human-Approved': 'yes-i-clicked-the-button' },
        body: JSON.stringify({ round_id: roundId, seed_index: seed, viewport_x: x, viewport_y: y, viewport_w: 15, viewport_h: 15 }),
      })
      const data = await resp.json()
      if (resp.ok) {
        setVpDone(prev => new Set(prev).add(`s${seed}_${x}_${y}`))
        setStepLog(l => [...l, { label: `Query S${seed} (${x},${y})`, status: 'done', detail: `${reason} [${data.queries_used}/${data.queries_max}]` }])
        setLog(l => [...l, `Query OK: S${seed} (${x},${y}) -- ${data.queries_used}/${data.queries_max}`])
        return true
      } else {
        setStepLog(l => [...l, { label: `Query S${seed} (${x},${y})`, status: 'error', detail: data.error }])
        return false
      }
    } catch (e) {
      setStepLog(l => [...l, { label: `Query S${seed} (${x},${y})`, status: 'error', detail: String(e) }])
      return false
    }
  }

  // Full autopilot loop
  const runAutopilot = async () => {
    if (!activeRound?.id || !allSeedsInit) return
    const roundId = activeRound.id
    setAutopilotRunning(true)
    autopilotRef.current = true
    setStepLog([])

    // Phase 0: Immediate model-only submission (safety net)
    setAutopilotPhase('Submitting model-only baseline...')
    await submitAllSeeds(roundId, 'model-only baseline')

    // Phase 1: First pass - 1 query per seed for growth estimation
    setAutopilotPhase('Phase 1: Growth estimation (1 query per seed)...')
    const currentDone = new Set(vpDone)
    for (let seed = 0; seed < 5; seed++) {
      if (!autopilotRef.current) break
      const plan = computeQueryPlan(currentDone, allSeedsInit, 1)
      if (plan.length === 0) break
      const q = plan[0]
      const ok = await execQuery(roundId, q.seed, q.x, q.y, q.reason)
      if (ok) currentDone.add(`s${q.seed}_${q.x}_${q.y}`)
      await new Promise(r => setTimeout(r, 220))
    }

    if (!autopilotRef.current) { setAutopilotRunning(false); return }

    // Resubmit with growth-adapted model
    setAutopilotPhase('Resubmitting with growth-adapted predictions...')
    qc.invalidateQueries({ queryKey: ['vp-observations'] })
    await new Promise(r => setTimeout(r, 500))
    await submitAllSeeds(roundId, 'growth-adapted')

    // Phase 2: Fill coverage - remaining viewports
    setAutopilotPhase('Phase 2: Filling viewport coverage...')
    let queriesUsed = 0
    while (autopilotRef.current) {
      const plan = computeQueryPlan(currentDone, allSeedsInit, 1)
      if (plan.length === 0) break
      const q = plan[0]
      const ok = await execQuery(roundId, q.seed, q.x, q.y, q.reason)
      if (!ok) break
      currentDone.add(`s${q.seed}_${q.x}_${q.y}`)
      queriesUsed++
      await new Promise(r => setTimeout(r, 220))

      // Resubmit every 10 queries
      if (queriesUsed % 10 === 0) {
        setAutopilotPhase(`Resubmitting after ${queriesUsed} queries...`)
        qc.invalidateQueries({ queryKey: ['vp-observations'] })
        await new Promise(r => setTimeout(r, 500))
        await submitAllSeeds(roundId, `after ${queriesUsed} queries`)
      }
    }

    // Phase 3: Submit with all VP data
    if (autopilotRef.current) {
      setAutopilotPhase('Submitting with all VP data...')
      qc.invalidateQueries({ queryKey: ['vp-observations'] })
      qc.invalidateQueries({ queryKey: ['vp-done'] })
      qc.invalidateQueries({ queryKey: ['my-rounds'] })
      await new Promise(r => setTimeout(r, 1000))
      await submitAllSeeds(roundId, 'VP-informed')
    }

    // Phase 4: Optimization loop - Claude iterates on the algorithm until round closes
    if (autopilotRef.current) {
      setAutopilotPhase('Starting optimization loop...')

      // Start the server-side optimizer
      await fetch('/api/optimizer/start', { method: 'POST' })
      setStepLog(l => [...l, { label: 'Optimizer started', status: 'done', detail: 'Claude iterating on predict()' }])

      let lastBestScore = 0
      let iterCount = 0

      while (autopilotRef.current) {
        // Check if round is still active
        const roundsResp = await fetch('/api/ainm/rounds', { headers: authHeaders(token) })
        const rounds = await roundsResp.json()
        const still = rounds?.find?.((r: { id: string; status: string }) => r.id === roundId && r.status === 'active')
        if (!still) {
          setAutopilotPhase('Round closed')
          setStepLog(l => [...l, { label: 'Round closed', status: 'done', detail: 'Stopping optimizer' }])
          break
        }

        // Check optimizer progress
        const statusResp = await fetch('/api/optimizer/status')
        const status = await statusResp.json()
        iterCount = status.best?.iterations ?? 0

        if (status.best && status.best.score > lastBestScore) {
          lastBestScore = status.best.score
          setAutopilotPhase(`Improvement found! Score: ${status.best.score.toFixed(1)} (iter ${iterCount}). Resubmitting...`)
          setStepLog(l => [...l, { label: `Optimizer improved: ${status.best.score.toFixed(1)}`, status: 'done', detail: `${status.best.file} (iter ${iterCount})` }])

          // TODO: load the new strategy and resubmit
          // For now, resubmit with current frontend model (optimizer changes are server-side)
          await submitAllSeeds(roundId, `optimized iter ${iterCount}`)
        } else {
          setAutopilotPhase(`Optimizing... iter ${iterCount}, best: ${status.best?.score?.toFixed(1) ?? '-'} (${status.status})`)
        }

        // If optimizer finished, stop
        if (status.status === 'done' || status.status === 'error') {
          setStepLog(l => [...l, { label: `Optimizer ${status.status}`, status: status.status === 'done' ? 'done' : 'error', detail: `${iterCount} iterations` }])
          break
        }

        await new Promise(r => setTimeout(r, 10000)) // check every 10s
      }

      // Stop optimizer if we're cancelling
      if (!autopilotRef.current) {
        await fetch('/api/optimizer/stop', { method: 'POST' })
      }

      // Final submission
      if (autopilotRef.current) {
        setAutopilotPhase('Final submission...')
        await submitAllSeeds(roundId, 'FINAL')
      }
    }

    setAutopilotPhase('Complete')
    setAutopilotRunning(false)
  }

  const stopAutopilot = () => { autopilotRef.current = false }

  // Build prediction for a specific seed
  const buildPredForSeed = (seed: number) => {
    if (!rawBuckets || !allSeedsInit) return null
    const seedInit = allSeedsInit.get(seed)
    if (!seedInit?.grid || !seedInit?.settlements) return null

    // Get VP observations for this specific seed
    const seedVP = allVpData?.[seed] ?? null

    let growth = 0.15
    if (seedVP && seedVP.length > 0) {
      growth = estimateGrowthFromVP(seedVP, seedInit.grid, seedInit.settlements)
    } else if (adaptiveInfo) {
      growth = adaptiveInfo.estimatedGrowth // use the current estimate
    }

    const closestRounds = selectClosestRounds(growthRates, growth, K_NEAREST)
    const adaptiveModel = mergeBuckets(rawBuckets, closestRounds)
    const allModel = mergeBuckets(rawBuckets, Object.keys(rawBuckets).map(Number))

    // Assemble VP grid for this seed
    let vpGrid: (number | null)[][] | null = null
    if (seedVP && seedVP.length > 0) {
      vpGrid = Array.from({ length: 40 }, () => Array(40).fill(null))
      for (const vp of seedVP) {
        for (let vy = 0; vy < vp.grid.length; vy++)
          for (let vx = 0; vx < vp.grid[vy].length; vx++) {
            const gy = vp.y + vy, gx = vp.x + vx
            if (gy < 40 && gx < 40) vpGrid![gy][gx] = vp.grid[vy][vx]
          }
      }
    }

    const sp = new Set<number>()
    for (const s of seedInit.settlements) sp.add(s.y * 40 + s.x)
    const pred = Array.from({ length: 40 }, () => Array.from({ length: 40 }, () => [0, 0, 0, 0, 0, 0]))

    for (let y = 0; y < 40; y++) for (let x = 0; x < 40; x++) {
      const key = getFeatureKey(seedInit.grid, sp, y, x)
      let prior = adaptiveModel[key] ? [...adaptiveModel[key]] : allModel[key] ? [...allModel[key]] : null
      if (!prior) {
        const fb = key.slice(0, -1)
        prior = adaptiveModel[fb] ? [...adaptiveModel[fb]] : allModel[fb] ? [...allModel[fb]] : [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
      }
      if (vpGrid && vpGrid[y][x] != null) {
        const obsClass = terrainToClass(vpGrid[y][x]!)
        const q = prior.map((p, c) => N_PRIOR * p + (c === obsClass ? 1 : 0))
        const total = N_PRIOR + 1
        const floored = q.map(v => Math.max(v / total, FLOOR))
        const sum = floored.reduce((a, b) => a + b, 0)
        pred[y][x] = floored.map(v => v / sum)
      } else {
        const floored = prior.map(v => Math.max(v, FLOOR))
        const sum = floored.reduce((a, b) => a + b, 0)
        pred[y][x] = floored.map(v => v / sum)
      }
    }
    return pred
  }


  return (
    <div className="dark min-h-screen bg-[#09090b] text-zinc-100">
      {/* Top bar */}
      <div className="border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-4 h-11 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Crosshair className="w-4 h-4 text-zinc-500" />
            <span className="text-sm font-semibold tracking-tight">Astar Island</span>
            <span className="text-[10px] font-mono text-zinc-500 border border-zinc-800 rounded px-1.5 py-0.5">CAL-culated risks</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1 text-[10px] font-mono text-amber-500/80 bg-amber-500/5 border border-amber-500/20 rounded px-2 py-0.5"><Shield className="w-3 h-3" />read-only proxy</div>
            {countdown && <div className="text-[11px] font-mono text-zinc-400 bg-zinc-800/50 rounded px-2 py-0.5"><Clock className="w-3 h-3 inline mr-1" />{countdown}</div>}
            {activeRound && <div className="text-[11px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded px-2 py-0.5">R{activeRound.round_number} active</div>}
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => qc.invalidateQueries()}><RefreshCw className="w-3.5 h-3.5" /></Button>
          </div>
        </div>
      </div>

      <div className="max-w-[1600px] mx-auto p-4 space-y-4">
        {!token && (
          <div className="flex gap-2 max-w-xl">
            <input type="password" placeholder="JWT token..." className="flex-1 bg-zinc-900 border border-zinc-800 rounded-md px-3 py-1.5 text-sm font-mono text-zinc-300 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600"
              onKeyDown={e => { if (e.key === 'Enter') saveToken((e.target as HTMLInputElement).value) }} />
            <Button size="sm" onClick={e => { const i = (e.target as HTMLElement).parentElement?.querySelector('input'); if (i) saveToken(i.value) }}>Connect</Button>
          </div>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="bg-zinc-900/50 border border-zinc-800/50">
            <TabsTrigger value="play" className="gap-1 text-xs"><Crosshair className="w-3 h-3" />Play</TabsTrigger>
            <TabsTrigger value="history" className="gap-1 text-xs"><BarChart3 className="w-3 h-3" />History</TabsTrigger>
            <TabsTrigger value="growth" className="gap-1 text-xs"><TrendingUp className="w-3 h-3" />Growth</TabsTrigger>
            <TabsTrigger value="leaderboard" className="gap-1 text-xs"><Trophy className="w-3 h-3" />Leaderboard</TabsTrigger>
            <TabsTrigger value="optimize" className="gap-1 text-xs"><Zap className="w-3 h-3" />Optimize</TabsTrigger>
          </TabsList>

          {/* ══════ PLAY ══════ */}
          <TabsContent value="play" className="space-y-5">
            {/* Step 1 */}
            <div className="space-y-1">
              <div className="flex items-center gap-3 flex-wrap">
                <Select value={selectedRound} onValueChange={setSelectedRound}>
                  <SelectTrigger className="w-32 h-8 text-xs bg-zinc-900 border-zinc-800"><SelectValue /></SelectTrigger>
                  <SelectContent>{Array.from({ length: 20 }, (_, i) => <SelectItem key={i+1} value={String(i+1)}>Round {i+1}</SelectItem>)}</SelectContent>
                </Select>
                <div className="flex gap-0.5">
                  {[0,1,2,3,4].map(s => <Button key={s} variant={selectedSeed === s ? 'default' : 'ghost'} size="sm" className="font-mono w-8 h-8 text-xs px-0" onClick={() => setSelectedSeed(s)}>S{s}</Button>)}
                </div>
                <Separator orientation="vertical" className="h-5" />
                {selectedMyRound?.round_score != null && <span className="text-xs font-mono text-green-500">{selectedMyRound.round_score.toFixed(1)} pts</span>}
                {selectedMyRound?.rank != null && <span className="text-xs font-mono text-zinc-500">#{selectedMyRound.rank}</span>}
                {selectedMyRound && <span className="text-xs font-mono text-zinc-600">{selectedMyRound.queries_used}/{selectedMyRound.queries_max} queries | {selectedMyRound.seeds_submitted}/5 submitted</span>}
              </div>
              <p className="text-[10px] text-zinc-600">Pick the active round and one of 5 seeds. Each seed is a different random map layout with the same hidden parameters.</p>
            </div>

            {/* Hero: Initial + Prediction + Controls */}
            <div className="grid grid-cols-[auto_auto_1fr] gap-4 items-start">
              <div className="space-y-1.5">
                <div className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider">Initial State</div>
                <MapCanvas data={initData?.grid ?? null} mode="initial" size={440} />
                {initData?.settlements && <div className="text-[10px] font-mono text-zinc-600">{initData.settlements.length} settlements</div>}
              </div>
              <div className="space-y-1.5">
                <div className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider flex items-center gap-2">
                  Our Prediction
                  {adaptiveInfo && (
                    <Badge variant="outline" className="text-[9px] font-mono text-green-500 border-green-500/30">
                      adaptive K=3 ({adaptiveInfo.growthLabel} {(adaptiveInfo.estimatedGrowth * 100).toFixed(0)}%)
                    </Badge>
                  )}
                </div>
                <MapCanvas data={prediction} mode="prob" initGrid={initData?.grid} size={440} />
                {adaptiveInfo && (
                  <div className="text-[10px] font-mono text-zinc-600 space-y-0.5">
                    <div>Training on R{adaptiveInfo.closestRounds.join(', R')} {adaptiveInfo.hasVP ? `| VP: ${adaptiveInfo.vpCells}/1600 cells observed` : '| No VP yet (using default growth estimate)'}</div>
                  </div>
                )}
              </div>

              {/* Right panel */}
              <div className="space-y-3">
                {/* VP query status */}
                {activeRound && (
                  <div className="space-y-2">
                    <div className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider flex items-center gap-1"><Zap className="w-3 h-3 text-amber-500" />VP Coverage (S{selectedSeed})</div>
                    <div className="grid grid-cols-3 gap-1">
                      {VP_GRID.map(({ x, y }, i) => {
                        const done = vpDone.has(`s${selectedSeed}_${x}_${y}`)
                        return <div key={i} className={`font-mono text-[10px] h-6 flex items-center justify-center rounded border ${done ? 'border-green-500/40 text-green-500 bg-green-500/5' : 'border-zinc-800/50 text-zinc-600'}`}>
                          {done ? <CheckCircle2 className="w-3 h-3" /> : `${x},${y}`}
                        </div>
                      })}
                    </div>
                    <div className="text-[9px] text-zinc-600">{VP_GRID.filter(({ x, y }) => vpDone.has(`s${selectedSeed}_${x}_${y}`)).length}/9 viewports observed</div>
                  </div>
                )}

                {/* Seed scores */}
                {selectedMyRound?.seed_scores && (
                  <div className="space-y-1">
                    <div className="text-[10px] font-mono text-zinc-500 uppercase">Seed Scores</div>
                    <div className="grid grid-cols-5 gap-1">
                      {selectedMyRound.seed_scores.map((sc, i) => (
                        <div key={i} className={`text-center py-1 rounded border text-[10px] font-mono ${selectedSeed === i ? 'border-zinc-500 bg-zinc-800/50' : 'border-zinc-800/30'}`}>
                          <div className="text-[8px] text-zinc-600">S{i}</div>
                          <div className={sc >= 80 ? 'text-green-500' : sc >= 60 ? 'text-yellow-500' : 'text-red-500'}>{sc.toFixed(0)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Autopilot */}
                {activeRound && !autopilotRunning && (
                  <div className="space-y-1.5">
                    <div className="text-[10px] font-mono text-zinc-500 uppercase">Autopilot</div>
                    <p className="text-[9px] text-zinc-600 leading-tight">Runs the full loop: submit baseline, query all viewports (smart order), resubmit after each batch, final submission.</p>
                    <Button size="sm" className="w-full bg-emerald-700 hover:bg-emerald-600 text-xs" onClick={runAutopilot}>
                      <Zap className="w-3 h-3 mr-1" />Start Autopilot
                    </Button>
                  </div>
                )}
                {autopilotRunning && (
                  <div className="space-y-1.5">
                    <div className="text-[10px] font-mono text-emerald-400 uppercase flex items-center gap-1">
                      <RefreshCw className="w-3 h-3 animate-spin" /> Autopilot Running
                    </div>
                    <div className="text-[10px] text-zinc-400 font-mono">{autopilotPhase}</div>
                    <Button size="sm" variant="outline" className="w-full text-xs border-red-500/30 text-red-400" onClick={stopAutopilot}>
                      Stop Autopilot
                    </Button>
                  </div>
                )}

                {/* Activity log */}
                {stepLog.length > 0 && (
                  <div className="space-y-1">
                    <div className="text-[10px] font-mono text-zinc-500 uppercase">Activity ({stepLog.length})</div>
                    <ScrollArea className="h-40">
                      <div className="space-y-0.5">
                        {stepLog.map((s, i) => (
                          <div key={i} className="flex items-center gap-1.5 text-[10px]">
                            {s.status === 'done' ? <CheckCircle2 className="w-3 h-3 text-green-500 shrink-0" /> : <AlertTriangle className="w-3 h-3 text-red-500 shrink-0" />}
                            <span className="font-mono text-zinc-400 truncate">{s.label}</span>
                            {s.detail && <span className="text-zinc-600 truncate">{s.detail}</span>}
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </div>
                )}
              </div>
            </div>

            {/* Legend */}
            <div className="flex gap-3">
              <div className="flex items-center gap-1 text-[10px] text-zinc-500"><div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: BG_OCEAN }} />Ocean</div>
              {CLASS_NAMES.map((n, i) => <div key={i} className="flex items-center gap-1 text-[10px] text-zinc-500"><div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS[i] }} />{n}</div>)}
            </div>

            {/* Adaptive model details */}
            {adaptiveInfo && (
              <>
                <Separator className="bg-zinc-800/30" />
                <div className="space-y-2">
                  <div className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider">Adaptive Model Details</div>
                  <p className="text-[10px] text-zinc-600">The model estimates this round's growth rate from VP observations, then selects the 3 most similar past rounds to build predictions. More VP queries = more accurate growth estimation = better round selection.</p>
                  <div className="flex gap-4 flex-wrap text-xs">
                    <div className="bg-zinc-900/50 border border-zinc-800/30 rounded px-3 py-2">
                      <div className="text-[9px] text-zinc-500 uppercase">Growth Estimate</div>
                      <div className="font-mono text-zinc-200">{(adaptiveInfo.estimatedGrowth * 100).toFixed(1)}% <span className="text-zinc-500">({adaptiveInfo.growthLabel})</span></div>
                    </div>
                    <div className="bg-zinc-900/50 border border-zinc-800/30 rounded px-3 py-2">
                      <div className="text-[9px] text-zinc-500 uppercase">Training Rounds</div>
                      <div className="font-mono text-zinc-200">R{adaptiveInfo.closestRounds.join(', R')}</div>
                      <div className="text-[9px] text-zinc-500">{adaptiveInfo.closestRounds.map(r => `${((growthRates[String(r)] ?? 0) * 100).toFixed(0)}%`).join(', ')} growth</div>
                    </div>
                    <div className="bg-zinc-900/50 border border-zinc-800/30 rounded px-3 py-2">
                      <div className="text-[9px] text-zinc-500 uppercase">VP Coverage</div>
                      <div className="font-mono text-zinc-200">{adaptiveInfo.vpCells}/1600 cells</div>
                      <div className="text-[9px] text-zinc-500">{adaptiveInfo.hasVP ? 'Using VP data' : 'Default estimate (no VP)'}</div>
                    </div>
                    <div className="bg-zinc-900/50 border border-zinc-800/30 rounded px-3 py-2">
                      <div className="text-[9px] text-zinc-500 uppercase">Bayesian Prior</div>
                      <div className="font-mono text-zinc-200">N={N_PRIOR}</div>
                      <div className="text-[9px] text-zinc-500">Floor={FLOOR}</div>
                    </div>
                  </div>
                </div>
              </>
            )}

            {/* Timeline: how predictions evolved */}
            {timeline && timeline.length > 1 && (
              <>
                <Separator className="bg-zinc-800/30" />
                <div className="space-y-2">
                  <div className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider">Prediction Evolution</div>
                  <p className="text-[10px] text-zinc-600">How the growth estimate and training round selection changed as VP queries were added for S{selectedSeed}.</p>
                  <div className="flex gap-1 items-end h-24">
                    {timeline.map((pt, i) => {
                      const h = Math.max(4, (pt.growthEst / 0.35) * 80)
                      const color = pt.growthLabel === 'death' ? '#ef4444' : pt.growthLabel === 'low' ? '#f97316' : pt.growthLabel === 'mid' ? '#eab308' : pt.growthLabel === 'high' ? '#22c55e' : '#3b82f6'
                      return (
                        <div key={i} className="flex-1 flex flex-col items-center gap-0.5 group relative">
                          <div className="text-[8px] font-mono text-zinc-600 opacity-0 group-hover:opacity-100 transition-opacity absolute -top-10 bg-zinc-900 border border-zinc-800 rounded px-1.5 py-0.5 whitespace-nowrap z-10">
                            {pt.growthLabel} {(pt.growthEst * 100).toFixed(1)}% | R{pt.trainingRounds.join(',')}
                          </div>
                          <div className="w-full rounded-sm" style={{ height: h, backgroundColor: color, opacity: 0.8 }} />
                          <div className="text-[7px] font-mono text-zinc-600">{pt.queryNum === 0 ? 'base' : `Q${pt.queryNum}`}</div>
                        </div>
                      )
                    })}
                  </div>
                  {/* Show training round changes */}
                  <div className="flex gap-2 flex-wrap">
                    {timeline.filter((pt, i) => i === 0 || pt.trainingRounds.join(',') !== timeline[i - 1].trainingRounds.join(',')).map((pt, i) => (
                      <div key={i} className="text-[9px] font-mono bg-zinc-900/50 border border-zinc-800/30 rounded px-2 py-1">
                        <span className="text-zinc-500">{pt.queryNum === 0 ? 'Before queries' : `After Q${pt.queryNum}`}:</span>
                        <span className="text-zinc-300 ml-1">R{pt.trainingRounds.join(', R')}</span>
                        <span className="text-zinc-600 ml-1">({pt.growthLabel} {(pt.growthEst * 100).toFixed(0)}%)</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            {/* Log */}
            {log.length > 0 && (
              <div className="bg-zinc-950 border border-zinc-800/30 rounded p-2 max-h-24 overflow-y-auto font-mono text-[10px] space-y-0.5">
                {log.map((msg, i) => <div key={i} className={msg.includes('FAIL') || msg.includes('ERROR') ? 'text-red-400' : msg.includes('OK') || msg.includes('Submit') ? 'text-green-400' : 'text-zinc-500'}>{msg}</div>)}
              </div>
            )}
          </TabsContent>

          {/* ══════ HISTORY ══════ */}
          <TabsContent value="history" className="space-y-4">
            {/* Score timeline chart */}
            <Card className="bg-zinc-900/50 border-zinc-800/50">
              <CardHeader className="pb-2"><CardTitle className="text-sm">Score Timeline</CardTitle></CardHeader>
              <CardContent>
                <div className="flex items-end gap-1 h-28">
                  {myRounds.sort((a: MyRound, b: MyRound) => a.round_number - b.round_number).map((r: MyRound) => {
                    const sc = r.round_score
                    const h = sc != null ? Math.max(4, (sc / 100) * 100) : 4
                    const color = sc == null ? '#27272a' : sc >= 80 ? '#16a34a' : sc >= 60 ? '#ca8a04' : sc >= 40 ? '#ea580c' : '#dc2626'
                    return (
                      <div key={r.round_number} className="flex-1 flex flex-col items-center gap-0.5 cursor-pointer" onClick={() => { setSelectedRound(String(r.round_number)); setActiveTab('play') }}>
                        {sc != null && <div className="text-[8px] font-mono text-zinc-400">{sc.toFixed(0)}</div>}
                        <div className="w-full rounded-sm" style={{ height: h, backgroundColor: color, opacity: 0.85 }} />
                        <div className="text-[8px] font-mono text-zinc-600">R{r.round_number}</div>
                      </div>
                    )
                  })}
                </div>
                <div className="flex justify-between mt-2 text-[9px] text-zinc-500">
                  <span>Best: <span className="text-green-500">{Object.values(myScores).length ? Math.max(...Object.values(myScores)).toFixed(1) : '-'}</span></span>
                  <span>Avg: {Object.values(myScores).length ? (Object.values(myScores).reduce((a, b) => a + b, 0) / Object.values(myScores).length).toFixed(1) : '-'}</span>
                  <span>Rounds played: {Object.keys(myScores).length}</span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-zinc-900/50 border-zinc-800/50">
              <CardHeader className="pb-2"><CardTitle className="text-sm">All Submissions</CardTitle></CardHeader>
              <CardContent><ScrollArea className="h-[500px]">
                <Table><TableHeader><TableRow className="border-zinc-800/50">
                  <TableHead className="text-xs w-14">Round</TableHead><TableHead className="text-xs">Status</TableHead>
                  <TableHead className="text-right text-xs">Score</TableHead><TableHead className="text-right text-xs">Rank</TableHead>
                  <TableHead className="text-right text-xs">Seeds</TableHead><TableHead className="text-right text-xs">Queries</TableHead>
                  <TableHead className="text-right text-xs">Growth</TableHead><TableHead className="text-xs">Seed Scores</TableHead>
                </TableRow></TableHeader>
                <TableBody>{myRounds.sort((a: MyRound, b: MyRound) => a.round_number - b.round_number).map((r: MyRound) => (
                  <TableRow key={r.round_number} className="border-zinc-800/30 cursor-pointer hover:bg-zinc-800/30" onClick={() => { setSelectedRound(String(r.round_number)); setActiveTab('play') }}>
                    <TableCell className="font-mono text-xs">R{r.round_number}</TableCell>
                    <TableCell><Badge variant="secondary" className="text-[9px] h-5">{r.status}</Badge></TableCell>
                    <TableCell className={`text-right font-mono text-xs ${!r.round_score ? '' : r.round_score >= 80 ? 'text-green-500' : r.round_score >= 60 ? 'text-yellow-500' : 'text-red-500'}`}>{r.round_score?.toFixed(1) ?? '-'}</TableCell>
                    <TableCell className="text-right font-mono text-xs text-zinc-500">{r.rank ? `#${r.rank}` : '-'}</TableCell>
                    <TableCell className="text-right font-mono text-xs">{r.seeds_submitted}/5</TableCell>
                    <TableCell className="text-right font-mono text-xs text-zinc-500">{r.queries_used}/{r.queries_max}</TableCell>
                    <TableCell className="text-right font-mono text-xs">{growthRates[String(r.round_number)] != null ? `${(growthRates[String(r.round_number)] * 100).toFixed(0)}%` : '-'}</TableCell>
                    <TableCell className="font-mono text-[10px] text-zinc-500">{r.seed_scores ? r.seed_scores.map(s => s.toFixed(0)).join(' / ') : '-'}</TableCell>
                  </TableRow>
                ))}</TableBody></Table>
              </ScrollArea></CardContent>
            </Card>
          </TabsContent>

          {/* ══════ GROWTH ══════ */}
          <TabsContent value="growth">
            <Card className="bg-zinc-900/50 border-zinc-800/50 max-w-2xl">
              <CardHeader className="pb-2"><CardTitle className="text-sm">Growth Rates</CardTitle></CardHeader>
              <CardContent className="space-y-1">{sortedGrowth.map(({ round, rate }) => <GrowthBar key={round} roundNum={round} rate={rate} />)}</CardContent>
            </Card>
          </TabsContent>

          {/* ══════ LEADERBOARD ══════ */}
          <TabsContent value="leaderboard">
            <Card className="bg-zinc-900/50 border-zinc-800/50">
              <CardHeader className="pb-2"><CardTitle className="text-sm">Top 30</CardTitle></CardHeader>
              <CardContent><ScrollArea className="h-[500px]">
                <Table><TableHeader><TableRow className="border-zinc-800/50">
                  <TableHead className="w-10 text-xs">#</TableHead><TableHead className="text-xs">Team</TableHead>
                  <TableHead className="text-right text-xs">Weighted</TableHead><TableHead className="text-right text-xs">Hot Streak</TableHead>
                  <TableHead className="text-right text-xs">Rounds</TableHead>
                </TableRow></TableHeader>
                <TableBody>{leaderboard.map((t: { team_name: string; weighted_score: number; hot_streak_score: number; rounds_participated: number }, i: number) => (
                  <TableRow key={i} className={`border-zinc-800/30 ${t.team_name === 'CAL-culated risks' ? 'bg-blue-500/5' : ''}`}>
                    <TableCell className="font-mono text-xs">{i+1}</TableCell>
                    <TableCell className={`text-xs ${t.team_name === 'CAL-culated risks' ? 'font-bold text-blue-400' : ''}`}>{t.team_name}</TableCell>
                    <TableCell className="text-right font-mono text-xs">{t.weighted_score?.toFixed(1)}</TableCell>
                    <TableCell className="text-right font-mono text-xs text-zinc-500">{t.hot_streak_score?.toFixed(1)}</TableCell>
                    <TableCell className="text-right font-mono text-xs text-zinc-500">{t.rounds_participated}</TableCell>
                  </TableRow>
                ))}</TableBody></Table>
              </ScrollArea></CardContent>
            </Card>
          </TabsContent>

          {/* ══════ OPTIMIZE ══════ */}
          <TabsContent value="optimize" className="space-y-4">
            <OptimizerPanel />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

// ── Optimizer Panel (separate component to keep App clean) ──
function OptimizerPanel() {
  const { data: status, refetch } = useQuery({
    queryKey: ['optimizer-status'],
    queryFn: () => fetch('/api/optimizer/status').then(r => r.json()),
    refetchInterval: 2000,
  })

  const startOptimizer = async () => {
    await fetch('/api/optimizer/start', { method: 'POST' })
    refetch()
  }
  const stopOptimizer = async () => {
    await fetch('/api/optimizer/stop', { method: 'POST' })
    refetch()
  }

  const isRunning = status?.status === 'running'

  return (
    <div className="space-y-4">
      <Card className="bg-zinc-900/50 border-zinc-800/50">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center justify-between">
            <span>Strategy Optimizer</span>
            {status?.best && <Badge variant="outline" className="font-mono text-green-500 border-green-500/30">Best: {status.best.score?.toFixed(1)} avg ({status.best.iterations} iterations)</Badge>}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-[10px] text-zinc-500">Uses Claude to iteratively write and test new prediction strategies against LOO cross-validation on all completed rounds. Each iteration Claude sees the current best code and per-round scores, then proposes a new predict() function.</p>

          <div className="flex gap-2">
            {!isRunning ? (
              <Button size="sm" className="bg-emerald-700 hover:bg-emerald-600 text-xs" onClick={startOptimizer}>
                <Zap className="w-3 h-3 mr-1" />Start Optimizer (10 iterations)
              </Button>
            ) : (
              <Button size="sm" variant="outline" className="text-xs border-red-500/30 text-red-400" onClick={stopOptimizer}>
                Stop Optimizer
              </Button>
            )}
            <Badge variant="secondary" className={`text-[10px] ${isRunning ? 'text-emerald-400' : ''}`}>
              {status?.status ?? 'loading...'}
            </Badge>
          </div>

          {/* Live log */}
          {status?.log && status.log.length > 0 && (
            <ScrollArea className="h-64">
              <pre className="text-[10px] font-mono text-zinc-400 whitespace-pre-wrap">
                {status.log.join('\n')}
              </pre>
            </ScrollArea>
          )}
        </CardContent>
      </Card>

      {/* Best strategy code */}
      <BestStrategyCard />
    </div>
  )
}

function BestStrategyCard() {
  const { data } = useQuery({
    queryKey: ['best-strategy'],
    queryFn: () => fetch('/api/optimizer/best-strategy').then(r => r.ok ? r.json() : null),
    refetchInterval: 5000,
  })

  if (!data) return null

  return (
    <Card className="bg-zinc-900/50 border-zinc-800/50">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center justify-between">
          <span>Best Strategy: {data.file}</span>
          <Badge className="font-mono">{data.score?.toFixed(1)} avg</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-80">
          <pre className="text-[10px] font-mono text-zinc-300 whitespace-pre-wrap">{data.code}</pre>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
