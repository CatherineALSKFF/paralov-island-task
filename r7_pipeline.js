#!/usr/bin/env node
/**
 * R7 Pipeline: Smart viewport queries + trajectory model
 *
 * Strategy:
 * 1. Load cross-round model (trajectory + GT)
 * 2. Identify HIGH-ENTROPY cells that will dominate the score
 * 3. Plan viewport queries to cover those cells
 * 4. Use viewport observations at year 50 as direct evidence
 * 5. Bayesian update: combine model prior with observations
 * 6. Submit and iteratively improve
 *
 * Viewport optimization:
 * - Each query gives 15×15=225 cells
 * - 50 queries = 11,250 cell observations
 * - With 40×40=1600 cells, we can cover everything ~7x
 * - Focus on dynamic cells (ignore ocean/mountain)
 *
 * Key insight: Each viewport observation is an INDEPENDENT simulation!
 * Observing the same cell 10x gives 10 samples from the round's distribution.
 * This is equivalent to 10 replays for that cell, which directly gives us
 * same-round data that breaks the cross-round ceiling.
 */
const https = require('https');
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5, C = 6;
const TOKEN = process.argv[2] || '';
if (!TOKEN) { console.log('Usage: node r7_pipeline.js <JWT> [--round-id ID]'); process.exit(1); }
const ROUND_ARG = process.argv.findIndex(a => a === '--round-id');
const ROUND_ID = ROUND_ARG >= 0 ? process.argv[ROUND_ARG + 1] : null;

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

function buildTrajectoryModel(replayResults, initGrids, roundNames) {
  const m = {};
  for (const rn of roundNames) {
    if (!replayResults[rn]) continue;
    for (const rep of replayResults[rn]) {
      const initGrid = initGrids[rn][rep.si];
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cf(initGrid, y, x); if (!keys) continue;
        const finalClass = t2c(rep.finalGrid[y][x]);
        for (let ki = 0; ki < keys.length; ki++) {
          const k = keys[ki];
          if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
          m[k].n++;
          m[k].counts[finalClass]++;
        }
      }
    }
  }
  const alpha = 0.05;
  for (const k of Object.keys(m)) {
    const total = m[k].n + C * alpha;
    m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / total);
  }
  return m;
}

function buildGTModel(inits, gts, roundNames) {
  const m = {};
  for (const rn of roundNames) {
    for (let si = 0; si < SEEDS; si++) {
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cf(inits[rn][si], y, x); if (!keys) continue;
        const g = gts[rn][si][y][x];
        for (let ki = 0; ki < keys.length; ki++) {
          const k = keys[ki];
          if (!m[k]) m[k] = { n: 0, s: new Float64Array(C) };
          m[k].n++;
          for (let c = 0; c < C; c++) m[k].s[c] += g[c];
        }
      }
    }
  }
  for (const k of Object.keys(m)) { m[k].a = Array.from(m[k].s).map(v => v / m[k].n); delete m[k].s; }
  return m;
}

const CFG = { ws: [1, 0.3, 0.15, 0.08, 0.02], pow: 0.5, minN: 2, fl: 0.00005 };

function predict(grid, model, cfg) {
  const { ws, pow, minN, fl } = cfg;
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
        if (d && d.n >= minN) { const w = ws[ki] * Math.pow(d.n, pow);
          for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w; }
      }
      if (wS === 0) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      let s = 0; for (let c = 0; c < C; c++) { p[c] /= wS; if (p[c] < fl) p[c] = fl; s += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

// ═══ BAYESIAN UPDATE WITH VIEWPORT OBSERVATIONS ═══
// prior[y][x] = [p0..p5], observed counts per cell → posterior
function bayesUpdate(prior, viewportCounts, pseudoCount) {
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const key = `${y}_${x}`;
      if (!viewportCounts[key]) {
        pred[y][x] = prior[y][x].slice();
        continue;
      }
      const counts = viewportCounts[key];
      const totalObs = counts.reduce((a,b)=>a+b, 0);
      if (totalObs === 0) { pred[y][x] = prior[y][x].slice(); continue; }

      // Bayesian: posterior ∝ prior * likelihood
      // prior comes from cross-round model (Dirichlet with prior_alpha)
      // likelihood = multinomial(counts)
      // posterior_alpha[c] = prior_alpha[c] + counts[c]
      const priorAlpha = prior[y][x].map(p => p * pseudoCount);
      const posterior = priorAlpha.map((a, c) => a + counts[c]);
      let s = 0;
      for (let c = 0; c < C; c++) { if (posterior[c] < 0.00001) posterior[c] = 0.00001; s += posterior[c]; }
      for (let c = 0; c < C; c++) posterior[c] /= s;
      pred[y][x] = posterior;
    }
  }
  return pred;
}

// ═══ VIEWPORT PLANNING ═══
// Plan which (y, x, year) to query to maximize information gain
function planViewports(priorPred, initGrid, maxQueries) {
  // Calculate expected entropy for each cell
  const cellEntropy = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (initGrid[y][x] === 10 || initGrid[y][x] === 5) continue; // skip static
    let e = 0;
    for (let c = 0; c < C; c++) {
      const p = priorPred[y][x][c];
      if (p > 1e-6) e -= p * Math.log(p);
    }
    if (e > 0.01) cellEntropy.push({ y, x, e });
  }
  cellEntropy.sort((a, b) => b.e - a.e);

  // Find viewport positions that cover the most high-entropy cells
  // Viewport is 15×15 centered at (cy, cx), so it covers [cy-7..cy+7, cx-7..cx+7]
  const covered = new Set();
  const viewports = [];
  const queriesPerSeed = Math.floor(maxQueries / SEEDS);

  // Greedy: pick viewport center that covers the most uncovered high-entropy cells
  while (viewports.length < queriesPerSeed) {
    let bestCY = 7, bestCX = 7, bestScore = -1;
    // Try all valid viewport centers
    for (let cy = 7; cy < H - 7; cy += 3) for (let cx = 7; cx < W - 7; cx += 3) {
      let vpScore = 0;
      for (let dy = -7; dy <= 7; dy++) for (let dx = -7; dx <= 7; dx++) {
        const ny = cy + dy, nx = cx + dx;
        if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
        const key = `${ny}_${nx}`;
        if (covered.has(key)) continue;
        // Find entropy for this cell
        const ce = cellEntropy.find(c => c.y === ny && c.x === nx);
        if (ce) vpScore += ce.e;
      }
      if (vpScore > bestScore) { bestScore = vpScore; bestCY = cy; bestCX = cx; }
    }
    if (bestScore <= 0) break;
    viewports.push({ cy: bestCY, cx: bestCX });
    // Mark cells as covered
    for (let dy = -7; dy <= 7; dy++) for (let dx = -7; dx <= 7; dx++) {
      const ny = bestCY + dy, nx = bestCX + dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W) covered.add(`${ny}_${nx}`);
    }
  }

  log(`  Planned ${viewports.length} viewports covering ${covered.size} cells`);
  return viewports;
}

// ═══ EXECUTE VIEWPORT QUERIES ═══
async function executeViewports(roundId, seedIndex, viewports, year) {
  const counts = {}; // "y_x" → [c0, c1, c2, c3, c4, c5]
  let queriesUsed = 0;

  for (const vp of viewports) {
    const res = await POST('/simulate', {
      round_id: roundId,
      seed_index: seedIndex,
      year: year,
      viewport: { y: vp.cy - 7, x: vp.cx - 7, height: 15, width: 15 }
    });
    queriesUsed++;

    if (res.ok && res.data.viewport) {
      const grid = res.data.viewport;
      for (let dy = 0; dy < grid.length; dy++) for (let dx = 0; dx < grid[0].length; dx++) {
        const ay = vp.cy - 7 + dy;
        const ax = vp.cx - 7 + dx;
        if (ay < 0 || ay >= H || ax < 0 || ax >= W) continue;
        const key = `${ay}_${ax}`;
        if (!counts[key]) counts[key] = [0,0,0,0,0,0];
        counts[key][t2c(grid[dy][dx])]++;
      }
    } else {
      log(`  Query failed: ${JSON.stringify(res.data).slice(0, 100)}`);
    }
    await sleep(250); // 5/sec rate limit
  }

  return { counts, queriesUsed };
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

function ensemblePreds(preds, weights) {
  const pred = [];
  const totalW = weights.reduce((a,b)=>a+b, 0);
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const p = [0,0,0,0,0,0];
      for (let i = 0; i < preds.length; i++)
        for (let c = 0; c < C; c++) p[c] += weights[i] * preds[i][y][x][c];
      let s = 0;
      for (let c = 0; c < C; c++) { p[c] /= totalW; s += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

// ═══ MAIN ═══
async function main() {
  log('═══ R7 Pipeline: Smart Viewport + Trajectory Model ═══');

  // Find active round
  let roundId = ROUND_ID;
  if (!roundId) {
    const { data: rounds } = await GET('/rounds');
    const active = rounds.find(r => r.status === 'active');
    if (active) {
      roundId = active.id;
      log(`Active round: R${active.round_number} (${roundId})`);
      log(`Closes: ${active.closes_at}`);
    } else {
      log('No active round! Waiting...');
      return;
    }
  }

  // Load all completed round data
  const RDS = {
    R1:'71451d74-be9f-471f-aacd-a41f3b68a9cd', R2:'76909e29-f664-4b2f-b16b-61b7507277e9',
    R4:'8e839974-b13b-407b-a5e7-fc749d877195', R5:'fd3c92ff-3178-4dc9-8d9b-acf389b3982b',
  };

  log('Loading data...');
  const inits = {}, gts = {};

  // Load current round
  const { data: roundData } = await GET('/rounds/' + roundId);
  inits['TARGET'] = roundData.initial_states.map(is => is.grid);

  // Load completed rounds
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

  const growthRounds = Object.keys(RDS);

  // ═══ Phase 1: Collect replays for completed rounds ═══
  log('\nPhase 1: Collecting replays (50 per round)...');
  const replayResults = {};
  const REPLAYS = 50, CONC = 8;
  for (const rn of growthRounds) {
    replayResults[rn] = [];
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
      for (const r of results) { if (r) { replayResults[rn].push(r); collected++; } }
      await sleep(200);
    }
    log(`  ${rn}: ${collected} replays`);
  }

  // Also check if the latest completed round (R6?) has GT now
  const { data: allRounds } = await GET('/rounds');
  for (const r of allRounds) {
    if (r.status === 'completed' && r.id !== roundId && !RDS[`R${r.round_number}`]) {
      const rn = `R${r.round_number}`;
      RDS[rn] = r.id;
      log(`  Found new completed round: ${rn}`);
      // Load GT and init
      const { data: rd } = await GET('/rounds/' + r.id);
      inits[rn] = rd.initial_states.map(is => is.grid);
      gts[rn] = [];
      await Promise.all(Array.from({length: SEEDS}, (_, si) =>
        GET('/analysis/' + r.id + '/' + si).then(res => { gts[rn][si] = res.data.ground_truth; })
      ));
      // Collect replays
      replayResults[rn] = [];
      let collected = 0;
      while (collected < REPLAYS) {
        const batch = [];
        for (let i = 0; i < CONC && (collected + batch.length) < REPLAYS; i++) {
          const si = (collected + batch.length) % SEEDS;
          batch.push((async () => {
            try {
              const res = await POST('/replay', { round_id: r.id, seed_index: si });
              if (!res.ok || !res.data.frames) return null;
              return { si, finalGrid: res.data.frames[res.data.frames.length - 1].grid };
            } catch { return null; }
          })());
        }
        const results = await Promise.all(batch);
        for (const r2 of results) { if (r2) { replayResults[rn].push(r2); collected++; } }
        await sleep(200);
      }
      log(`  ${rn}: ${collected} replays`);
      if (!growthRounds.includes(rn)) growthRounds.push(rn);
    }
  }

  // ═══ Phase 2: Build cross-round model ═══
  log('\nPhase 2: Building trajectory model...');
  const trajModel = buildTrajectoryModel(replayResults, inits, growthRounds);
  const gtModel = buildGTModel(inits, gts, growthRounds);

  // ═══ Phase 3: Submit baseline predictions FIRST ═══
  log('\nPhase 3: Submitting baseline predictions...');
  for (let si = 0; si < SEEDS; si++) {
    const p = predict(inits['TARGET'][si], trajModel, CFG);
    const res = await POST('/submit', { round_id: roundId, seed_index: si, prediction: p });
    log(`  Seed ${si}: ${res.data.status || JSON.stringify(res.data)}`);
    await sleep(600);
  }
  log('Baseline submitted!');

  // ═══ Phase 4: Plan and execute viewport queries ═══
  log('\nPhase 4: Smart viewport observations...');
  const viewportData = {}; // si → { counts }

  for (let si = 0; si < SEEDS; si++) {
    const prior = predict(inits['TARGET'][si], trajModel, CFG);
    const viewports = planViewports(prior, inits['TARGET'][si], 10); // 10 queries per seed = 50 total

    log(`  Seed ${si}: querying ${viewports.length} viewports...`);
    const { counts, queriesUsed } = await executeViewports(roundId, si, viewports, 50);
    viewportData[si] = counts;
    log(`  Seed ${si}: ${queriesUsed} queries used, ${Object.keys(counts).length} cells observed`);
  }

  // ═══ Phase 5: Bayesian update and resubmit ═══
  log('\nPhase 5: Bayesian update + resubmit...');
  for (const pseudoCount of [3, 5, 8, 10, 15]) {
    log(`  Testing pseudoCount=${pseudoCount}:`);
    for (let si = 0; si < SEEDS; si++) {
      const prior = predict(inits['TARGET'][si], trajModel, CFG);
      const posterior = bayesUpdate(prior, viewportData[si], pseudoCount);
      // Can't score (no GT), just validate
      let valid = true;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const s = posterior[y][x].reduce((a,b)=>a+b, 0);
        if (Math.abs(s - 1.0) > 0.02) valid = false;
      }
      log(`    Seed ${si}: ${valid ? 'valid' : 'INVALID'}`);
    }
  }

  // Submit with best pseudoCount (conservative: 5)
  const PSEUDO = 5;
  log(`\nSubmitting with pseudoCount=${PSEUDO}...`);
  for (let si = 0; si < SEEDS; si++) {
    const prior = predict(inits['TARGET'][si], trajModel, CFG);
    const p = bayesUpdate(prior, viewportData[si], PSEUDO);
    const res = await POST('/submit', { round_id: roundId, seed_index: si, prediction: p });
    log(`  Seed ${si}: ${res.data.status || JSON.stringify(res.data)}`);
    await sleep(600);
  }

  log('\n✅ R7 submissions complete!');
  log('Viewport observations added to trajectory model via Bayesian update');
}

main().catch(e => { console.error('Fatal:', e.message, e.stack); process.exit(1); });
