#!/usr/bin/env node
/**
 * Feature-Based Viewport Model (BREAKTHROUGH)
 *
 * KEY INSIGHT: Instead of Bayesian updating PER-CELL (1-2 obs per cell),
 * build a FEATURE-BASED model from viewport observations.
 * This aggregates across all cells with the same feature key,
 * giving ~30-50 observations per key from just 50 viewport queries.
 *
 * The model applies to ALL 5 seeds because feature→outcome is determined
 * by hidden parameters, which are the SAME across seeds.
 *
 * Strategy for active rounds:
 * 1. Focus ALL 50 queries on seed 0, year 50 (cover full map 5x)
 * 2. For each feature key, count terrain outcomes
 * 3. Build current-round feature model
 * 4. Fuse with cross-round prior
 * 5. Predict all 5 seeds
 *
 * Testing: simulate on completed rounds using replays as "viewport data"
 */
const https = require('https');
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5, C = 6;
const TOKEN = process.argv[2] || '';
if (!TOKEN) { console.log('Usage: node feature_viewport_model.js <JWT>'); process.exit(1); }

function api(method, path, body) {
  return new Promise((resolve, reject) => {
    const url = new URL(BASE + path);
    const payload = body ? JSON.stringify(body) : null;
    const opts = { hostname: url.hostname, path: url.pathname + url.search, method,
      headers: { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' } };
    if (payload) opts.headers['Content-Length'] = Buffer.byteLength(payload);
    const req = https.request(opts, res => {
      let data = ''; res.on('data', c => data += c);
      res.on('end', () => { try { resolve({ ok: res.statusCode < 300, status: res.statusCode, data: JSON.parse(data) }); } catch { resolve({ ok: false, status: res.statusCode, data }); } });
    }); req.on('error', reject); if (payload) req.write(payload); req.end();
  });
}
const GET = p => api('GET', p), POST = (p, b) => api('POST', p, b);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const log = m => { const t = new Date().toISOString().slice(11, 19); console.log(`[${t}] ${m}`); };
function t2c(t) { return (t === 10 || t === 11 || t === 0) ? 0 : (t >= 1 && t <= 5) ? t : 0; }

function cf(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS=0,co=0,fN=0,sR2=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];if(nt===1||nt===2)nS++;if(nt===10)co=1;if(nt===4)fN++;}
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;if(g[ny][nx]===1||g[ny][nx]===2)sR2++;}
  const sa=Math.min(nS,5),sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3,fb=fN<=1?0:fN<=3?1:2;
  return[`D0_${t}_${sa}_${co}_${sb2}_${fb}`,`D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,`D2_${t}_${sa>0?1:0}_${co}`,`D3_${t}_${co}`,`D4_${t}`];
}

// Build feature model from "observations" (replay grids used as viewport data)
function buildViewportModel(initGrids, observedGrids, alpha=0.1) {
  const m = {};
  for (const obs of observedGrids) {
    const initGrid = initGrids[obs.si];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const keys = cf(initGrid, y, x); if (!keys) continue;
      const fc = t2c(obs.finalGrid[y][x]);
      for (let ki = 0; ki < keys.length; ki++) {
        const k = keys[ki];
        if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
        m[k].n++;
        m[k].counts[fc]++;
      }
    }
  }
  for (const k of Object.keys(m)) {
    const total = m[k].n + C * alpha;
    m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / total);
  }
  return m;
}

// Build cross-round trajectory model
function buildCrossRoundModel(replays, inits, roundNames, alpha=0.05) {
  const m = {};
  for (const rn of roundNames) {
    if (!replays[rn]) continue;
    for (const rep of replays[rn]) {
      const initGrid = inits[rn][rep.si];
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cf(initGrid, y, x); if (!keys) continue;
        const fc = t2c(rep.finalGrid[y][x]);
        for (let ki = 0; ki < keys.length; ki++) {
          const k = keys[ki];
          if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
          m[k].n++;
          m[k].counts[fc]++;
        }
      }
    }
  }
  for (const k of Object.keys(m)) {
    const total = m[k].n + C * alpha;
    m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / total);
  }
  return m;
}

// Fuse two feature models: cross-round prior + viewport observations
function fuseModels(crossModel, viewportModel, crossWeight) {
  const m = {};
  const allKeys = new Set([...Object.keys(crossModel), ...Object.keys(viewportModel)]);
  for (const k of allKeys) {
    const cm = crossModel[k];
    const vm = viewportModel[k];
    if (cm && vm) {
      // Bayesian fusion: Dirichlet posterior = cross_alpha + viewport_counts
      const priorAlpha = cm.a.map(p => p * crossWeight);
      const posterior = priorAlpha.map((a, c) => a + vm.counts[c]);
      let total = posterior.reduce((a,b)=>a+b, 0);
      m[k] = { n: cm.n + vm.n, a: posterior.map(v => v / total) };
    } else if (vm) {
      m[k] = { n: vm.n, a: vm.a.slice() };
    } else {
      m[k] = { n: cm.n, a: cm.a.slice() };
    }
  }
  return m;
}

const CFG = { ws: [1, 0.3, 0.15, 0.08, 0.02], pow: 0.5, minN: 2, fl: 0.00005 };

function predict(grid, model, cfg) {
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1,0,0,0,0,0]; continue; }
      if (t === 5) { pred[y][x] = [0,0,0,0,0,1]; continue; }
      const keys = cf(grid, y, x);
      if (!keys) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      const p = [0,0,0,0,0,0]; let wS = 0;
      for (let ki = 0; ki < keys.length; ki++) {
        const d = model[keys[ki]];
        if (d && d.n >= cfg.minN) { const w = cfg.ws[ki] * Math.pow(d.n, cfg.pow);
          for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w; }
      }
      if (wS === 0) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      let s = 0; for (let c = 0; c < C; c++) { p[c] /= wS; if (p[c] < cfg.fl) p[c] = cfg.fl; s += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

function score(pred, gt) {
  let tKL = 0, tE = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const g = gt[y][x]; let e = 0;
    for (let c = 0; c < C; c++) if (g[c] > 1e-6) e -= g[c] * Math.log(g[c]);
    if (e < 0.01) continue; let kl = 0;
    for (let c = 0; c < C; c++) if (g[c] > 1e-6) kl += g[c] * Math.log(g[c] / Math.max(pred[y][x][c], 1e-15));
    tKL += Math.max(0, kl) * e; tE += e;
  }
  return tE > 0 ? 100 * Math.exp(-3 * tKL / tE) : 0;
}

async function main() {
  log('═══ Feature-Based Viewport Model Test ═══');
  log('Simulating viewport strategy on completed rounds');

  const RDS = {
    R1:'71451d74-be9f-471f-aacd-a41f3b68a9cd', R2:'76909e29-f664-4b2f-b16b-61b7507277e9',
    R4:'8e839974-b13b-407b-a5e7-fc749d877195', R5:'fd3c92ff-3178-4dc9-8d9b-acf389b3982b',
  };

  log('Loading data...');
  const inits = {}, gts = {};
  await Promise.all(Object.entries(RDS).map(async ([rn, id]) => {
    const { data } = await GET('/rounds/' + id);
    inits[rn] = data.initial_states.map(is => is.grid);
  }));
  await Promise.all(Object.keys(RDS).map(async rn => {
    gts[rn] = [];
    await Promise.all(Array.from({length: SEEDS}, (_, si) =>
      GET('/analysis/' + RDS[rn] + '/' + si).then(r => { gts[rn][si] = r.data.ground_truth; })
    ));
  }));
  log('Data loaded');

  const growthRounds = ['R1','R2','R4','R5'];

  // Collect replays for all rounds
  log('\nCollecting replays (30 per round = viewport simulation data)...');
  const replays = {};
  const REPLAYS = 30, CONC = 8;
  for (const rn of growthRounds) {
    replays[rn] = [];
    let collected = 0, errors = 0;
    while (collected < REPLAYS) {
      const batch = [];
      for (let i = 0; i < CONC && (collected + batch.length) < REPLAYS; i++) {
        const si = (collected + batch.length) % SEEDS;
        batch.push((async () => {
          try {
            const res = await POST('/replay', { round_id: RDS[rn], seed_index: si });
            if (!res.ok || !res.data.frames) { errors++; return null; }
            return { si, finalGrid: res.data.frames[res.data.frames.length - 1].grid };
          } catch { errors++; return null; }
        })());
      }
      const results = await Promise.all(batch);
      for (const r of results) { if (r) { replays[rn].push(r); collected++; } }
      await sleep(200);
    }
    log(`  ${rn}: ${collected} replays`);
  }

  // ═══ TEST: Simulate feature-viewport approach ═══
  log('\n═══ Feature-Viewport Model (LOO) ═══');
  log('For each test round:');
  log('  1. Build cross-round prior from OTHER rounds');
  log('  2. Use test round replays as "viewport observations" from seed 0');
  log('  3. Build feature model from observations');
  log('  4. Fuse with prior');
  log('  5. Predict ALL seeds, score against GT');

  // Baseline: cross-round only
  {
    let total = 0;
    for (const testRn of growthRounds) {
      const trainRns = growthRounds.filter(r => r !== testRn);
      const model = buildCrossRoundModel(replays, inits, trainRns);
      for (let si = 0; si < SEEDS; si++)
        total += score(predict(inits[testRn][si], model, CFG), gts[testRn][si]);
    }
    log(`\nBaseline (cross-round only): LOO=${(total/20).toFixed(2)}`);
  }

  // Feature-viewport model: use N replays from seed 0 as "viewport observations"
  // Then predict ALL 5 seeds
  for (const nViewportObs of [1, 2, 3, 5, 8, 10, 15, 20]) {
    log(`\n--- ${nViewportObs} viewport observations (seed 0 only) ---`);

    for (const crossWeight of [5, 10, 15, 20, 30, 50]) {
      let total = 0;
      for (const testRn of growthRounds) {
        const trainRns = growthRounds.filter(r => r !== testRn);
        const crossModel = buildCrossRoundModel(replays, inits, trainRns);

        // Simulate viewport observations from seed 0
        const seed0Replays = replays[testRn].filter(r => r.si === 0);
        const used = seed0Replays.slice(0, nViewportObs);

        // Build feature model from viewport observations
        // Key: we use seed 0's initial grid as the feature source
        const viewportModel = buildViewportModel(inits[testRn], used, 0.1);

        // Count observations per feature key
        let totalObs = 0, totalKeys = 0;
        for (const k of Object.keys(viewportModel)) { totalObs += viewportModel[k].n; totalKeys++; }

        // Fuse models
        const fusedModel = fuseModels(crossModel, viewportModel, crossWeight);

        // Predict ALL 5 seeds
        let seedTotal = 0;
        for (let si = 0; si < SEEDS; si++)
          seedTotal += score(predict(inits[testRn][si], fusedModel, CFG), gts[testRn][si]);
        total += seedTotal / SEEDS;
      }
      const avg = total / 4;
      if (nViewportObs <= 3 || crossWeight === 20 || crossWeight === 30)
        log(`  crossWeight=${crossWeight}: LOO=${avg.toFixed(2)}`);
    }
  }

  // ═══ DETAILED ANALYSIS: Per-round breakdown for best config ═══
  log('\n═══ Detailed breakdown (best config) ═══');
  const bestNObs = 5;
  const bestCW = 20;
  {
    let total = 0;
    const perRound = {};
    for (const testRn of growthRounds) {
      const trainRns = growthRounds.filter(r => r !== testRn);
      const crossModel = buildCrossRoundModel(replays, inits, trainRns);
      const seed0Replays = replays[testRn].filter(r => r.si === 0);
      const used = seed0Replays.slice(0, bestNObs);
      const viewportModel = buildViewportModel(inits[testRn], used, 0.1);
      const fusedModel = fuseModels(crossModel, viewportModel, bestCW);

      let seedTotal = 0;
      for (let si = 0; si < SEEDS; si++) {
        const s = score(predict(inits[testRn][si], fusedModel, CFG), gts[testRn][si]);
        seedTotal += s;
      }
      perRound[testRn] = seedTotal / SEEDS;
      total += perRound[testRn];
    }
    log(`nObs=${bestNObs}, cw=${bestCW}: LOO=${(total/4).toFixed(2)} [${Object.values(perRound).map(v=>v.toFixed(1)).join(', ')}]`);
  }

  // How many feature keys have observations?
  log('\n═══ Feature key analysis ═══');
  for (const testRn of growthRounds) {
    const seed0Replays = replays[testRn].filter(r => r.si === 0).slice(0, 5);
    const vm = buildViewportModel(inits[testRn], seed0Replays, 0.1);
    let d0=0, d1=0, d2=0, d3=0, d4=0;
    for (const k of Object.keys(vm)) {
      if (k.startsWith('D0_')) d0++;
      else if (k.startsWith('D1_')) d1++;
      else if (k.startsWith('D2_')) d2++;
      else if (k.startsWith('D3_')) d3++;
      else if (k.startsWith('D4_')) d4++;
    }
    log(`  ${testRn}: D0=${d0} D1=${d1} D2=${d2} D3=${d3} D4=${d4} keys`);
    // Average observations per D0 key
    let totalD0Obs = 0, nD0 = 0;
    for (const k of Object.keys(vm)) {
      if (k.startsWith('D0_')) { totalD0Obs += vm[k].n; nD0++; }
    }
    log(`    Avg D0 obs/key: ${nD0 > 0 ? (totalD0Obs/nD0).toFixed(1) : 0}`);
  }
}

main().catch(e => { console.error('Fatal:', e.message, e.stack); process.exit(1); });
