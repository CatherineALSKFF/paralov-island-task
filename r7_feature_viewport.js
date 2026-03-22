#!/usr/bin/env node
/**
 * R7 Feature-Based Viewport Pipeline
 *
 * VALIDATED: Feature-based viewport → LOO=87.7 (vs 84.1 baseline)
 *
 * Strategy:
 * 1. Build cross-round trajectory model from ALL completed rounds
 * 2. Submit baseline predictions immediately (cross-round model)
 * 3. Execute ALL 50 viewport queries on SEED 0, YEAR 50
 *    - 5 full passes of the 40×40 map (9 queries each = 45)
 *    - 5 extra queries for additional coverage
 * 4. Build feature model from viewport observations
 * 5. Fuse with cross-round prior (crossWeight=30, validated optimal)
 * 6. Re-submit improved predictions for ALL 5 seeds
 *
 * Key insight: Feature keys encode local terrain config. The round's hidden
 * parameters determine feature→outcome mapping, which is the SAME across seeds.
 * So observing seed 0 improves predictions for ALL seeds.
 */
const https = require('https');
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5, C = 6;
const TOKEN = process.argv[2] || '';
if (!TOKEN) { console.log('Usage: node r7_feature_viewport.js <JWT> [--round-id ID] [--no-wait]'); process.exit(1); }
const ROUND_ARG = process.argv.findIndex(a => a === '--round-id');
const ROUND_ID_OVERRIDE = ROUND_ARG >= 0 ? process.argv[ROUND_ARG + 1] : null;
const NO_WAIT = process.argv.includes('--no-wait');

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

// Feature function: returns [D0, D1, D2, D3, D4] keys or null for static cells
function cf(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS=0,co=0,fN=0,sR2=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];if(nt===1||nt===2)nS++;if(nt===10)co=1;if(nt===4)fN++;}
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;if(g[ny][nx]===1||g[ny][nx]===2)sR2++;}
  const sa=Math.min(nS,5),sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3,fb=fN<=1?0:fN<=3?1:2;
  return[`D0_${t}_${sa}_${co}_${sb2}_${fb}`,`D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,`D2_${t}_${sa>0?1:0}_${co}`,`D3_${t}_${co}`,`D4_${t}`];
}

// ═══ MODEL BUILDING ═══

// Build cross-round trajectory model from replays
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

// Build viewport feature model from 15×15 observations
// Each observation: { vy, vx, grid: 15×15 terrain at year 50 }
function buildViewportFeatureModel(initGrid, observations, alpha=0.1) {
  const m = {};
  for (const obs of observations) {
    for (let dy = 0; dy < obs.grid.length; dy++) {
      for (let dx = 0; dx < obs.grid[0].length; dx++) {
        const gy = obs.vy + dy;
        const gx = obs.vx + dx;
        if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
        const keys = cf(initGrid, gy, gx);
        if (!keys) continue;
        const fc = t2c(obs.grid[dy][dx]);
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

// Fuse cross-round prior with viewport feature model
function fuseModels(crossModel, viewportModel, crossWeight) {
  const m = {};
  const allKeys = new Set([...Object.keys(crossModel), ...Object.keys(viewportModel)]);
  for (const k of allKeys) {
    const cm = crossModel[k];
    const vm = viewportModel[k];
    if (cm && vm) {
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

// ═══ PREDICTION ═══
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
        if (d && d.n >= cfg.minN) {
          const w = cfg.ws[ki] * Math.pow(d.n, cfg.pow);
          for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w;
        }
      }
      if (wS === 0) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      let s = 0; for (let c = 0; c < C; c++) { p[c] /= wS; if (p[c] < cfg.fl) p[c] = cfg.fl; s += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

// ═══ VIEWPORT QUERY PLANNING ═══
// Generate all 50 viewport positions for seed 0
// 5 full passes of 9 positions + 5 offset positions
function planAllViewports() {
  // 3×3 grid covering 40×40 with 15×15 viewports (proven format from solver_r6.js)
  // Starts: [0, 13, 25] → covers 0-14, 13-27, 25-39 (full coverage with small overlaps)
  const starts = [0, 13, 25];
  const basePositions = [];
  for (const vy of starts) {
    for (const vx of starts) {
      basePositions.push({ vy, vx });
    }
  }

  const viewports = [];
  // 5 full passes = 45 queries
  for (let pass = 0; pass < 5; pass++) {
    for (const pos of basePositions) {
      viewports.push({ ...pos });
    }
  }
  // 5 extra: offset grid for additional feature diversity
  const extraPositions = [
    { vy: 7, vx: 7 }, { vy: 7, vx: 20 },
    { vy: 20, vx: 7 }, { vy: 20, vx: 20 },
    { vy: 12, vx: 12 }
  ];
  for (const pos of extraPositions) {
    viewports.push(pos);
  }

  return viewports; // Exactly 50
}

// ═══ EXECUTE VIEWPORT QUERIES ═══
async function executeViewportQueries(roundId, seedIndex, viewports) {
  const observations = [];
  let queryCount = 0;
  let failures = 0;

  for (const vp of viewports) {
    let success = false;
    for (let retry = 0; retry < 3 && !success; retry++) {
      try {
        const res = await POST('/simulate', {
          round_id: roundId,
          seed_index: seedIndex,
          year: 50,
          viewport: { x: vp.vx, y: vp.vy, width: 15, height: 15 }
        });
        queryCount++;

        if (res.ok && res.data.grid) {
          observations.push({
            vy: vp.vy,
            vx: vp.vx,
            grid: res.data.grid
          });
          success = true;
        } else if (res.status === 429) {
          // Rate limited - wait and retry
          await sleep(1000);
        } else {
          log(`  Viewport (${vp.vy},${vp.vx}) failed: ${JSON.stringify(res.data).slice(0, 100)}`);
          failures++;
          success = true; // Don't retry non-rate-limit errors
        }
      } catch (e) {
        await sleep(500);
      }
    }
    await sleep(220); // Rate limit: 5/sec

    if (queryCount % 10 === 0) {
      log(`  Progress: ${queryCount}/${viewports.length} queries, ${observations.length} observations`);
    }
  }

  return { observations, queryCount, failures };
}

// ═══ REPLAY COLLECTION ═══
async function collectReplays(roundId, count, concurrency = 8) {
  const results = [];
  let collected = 0, errors = 0;
  while (collected < count) {
    const batch = [];
    for (let i = 0; i < concurrency && (collected + batch.length) < count; i++) {
      const si = (collected + batch.length) % SEEDS;
      batch.push((async () => {
        try {
          const res = await POST('/replay', { round_id: roundId, seed_index: si });
          if (!res.ok || !res.data.frames) { errors++; return null; }
          return { si, finalGrid: res.data.frames[res.data.frames.length - 1].grid };
        } catch { errors++; return null; }
      })());
    }
    const batchResults = await Promise.all(batch);
    for (const r of batchResults) { if (r) { results.push(r); collected++; } }
    await sleep(200);
  }
  return { results, errors };
}

// ═══ MAIN PIPELINE ═══
async function main() {
  log('╔══════════════════════════════════════════════════╗');
  log('║  R7 Feature-Based Viewport Pipeline             ║');
  log('║  Expected: LOO ~87.7 (validated, +3.6 over base)║');
  log('╚══════════════════════════════════════════════════╝');

  // ═══ Step 0: Find target round ═══
  // Already-submitted rounds (skip these)
  const SUBMITTED = new Set([
    'ae78003a-4efe-425a-881a-d16a39bca0ad', // R6
  ]);
  let roundId = ROUND_ID_OVERRIDE;
  let roundNumber = '?';

  if (!roundId) {
    log('\nLooking for target round...');
    while (true) {
      try {
        const { data: rounds } = await GET('/rounds');
        const active = rounds.find(r => r.status === 'active' && !SUBMITTED.has(r.id));
        if (active) {
          roundId = active.id;
          roundNumber = active.round_number;
          log(`\n🎯 Target round: R${roundNumber} (${roundId})`);
          log(`   Closes: ${active.closes_at}`);
          log(`   Weight: ${active.round_weight}`);
          break;
        }
        // Check if there's an active round we already submitted to
        const currentActive = rounds.find(r => r.status === 'active');
        if (currentActive && SUBMITTED.has(currentActive.id)) {
          const closes = new Date(currentActive.closes_at);
          const minsLeft = Math.round((closes - Date.now()) / 60000);
          log(`  R${currentActive.round_number} is active but already submitted (${minsLeft} min left). Waiting for next round...`);
        } else {
          log('  No active round yet, checking again in 30s...');
        }
      } catch (e) {
        log(`  Network error: ${e.message}. Retrying in 30s...`);
      }
      if (NO_WAIT) { log('No target round and --no-wait specified. Exiting.'); return; }
      await sleep(30000);
    }
  }

  // ═══ Step 1: Load all data ═══
  log('\n═══ Step 1: Loading round data ═══');

  // Known completed growth rounds
  const RDS = {
    R1:'71451d74-be9f-471f-aacd-a41f3b68a9cd',
    R2:'76909e29-f664-4b2f-b16b-61b7507277e9',
    R4:'8e839974-b13b-407b-a5e7-fc749d877195',
    R5:'fd3c92ff-3178-4dc9-8d9b-acf389b3982b',
    R6:'ae78003a-4efe-425a-881a-d16a39bca0ad',
  };

  // Check for any additional newly completed rounds
  const { data: allRounds } = await GET('/rounds');
  for (const r of allRounds) {
    if (r.status === 'completed' && r.id !== roundId && !Object.values(RDS).includes(r.id)) {
      // Skip death round R3
      if (r.round_number === 3) continue;
      const rn = `R${r.round_number}`;
      RDS[rn] = r.id;
      log(`  Found new completed round: ${rn} (${r.id})`);
    }
  }
  // Remove any rounds that aren't completed yet (like R6 if still active)
  for (const [rn, id] of Object.entries(RDS)) {
    const roundInfo = allRounds.find(r => r.id === id);
    if (!roundInfo || roundInfo.status !== 'completed') {
      log(`  ${rn} not yet completed, will try GT later`);
    }
  }

  const inits = {}, gts = {};

  // Load target round initial states
  const { data: targetRound } = await GET('/rounds/' + roundId);
  inits['TARGET'] = targetRound.initial_states.map(is => is.grid);
  log(`  Target round loaded: ${SEEDS} seeds`);

  // Load completed rounds
  await Promise.all(Object.entries(RDS).map(async ([rn, id]) => {
    const { data } = await GET('/rounds/' + id);
    inits[rn] = data.initial_states.map(is => is.grid);
  }));

  // Load ground truths (gracefully handle rounds without GT)
  await Promise.all(Object.entries(RDS).map(async ([rn, id]) => {
    gts[rn] = [];
    try {
      const results = await Promise.all(Array.from({length: SEEDS}, (_, si) =>
        GET('/analysis/' + id + '/' + si).then(r => {
          if (r.ok && r.data && r.data.ground_truth) { gts[rn][si] = r.data.ground_truth; return true; }
          return false;
        })
      ));
      const gotAll = results.every(r => r);
      if (!gotAll) log(`  Warning: Incomplete GT for ${rn} (${results.filter(r => r).length}/${SEEDS})`);
    } catch (e) {
      log(`  Warning: Could not load GT for ${rn}: ${e.message}`);
    }
  }));

  // Separate rounds into:
  // - roundsWithInits: all rounds with initial states (for replay collection + trajectory model)
  // - roundsWithGT: rounds with GT (for optional validation)
  const roundsWithInits = Object.keys(RDS).filter(rn => inits[rn] && inits[rn].length === SEEDS);
  const roundsWithGT = Object.keys(RDS).filter(rn =>
    inits[rn] && inits[rn].length === SEEDS && gts[rn] && gts[rn].length === SEEDS && gts[rn].every(g => g)
  );
  log(`  Rounds with initial states: ${roundsWithInits.length}: ${roundsWithInits.join(', ')}`);
  log(`  Rounds with GT: ${roundsWithGT.length}: ${roundsWithGT.join(', ')}`);

  if (roundsWithInits.length === 0) {
    log('  ERROR: No rounds with initial states! Cannot build model.');
    return;
  }

  // ═══ Step 2: Collect replays (for ALL rounds with inits, not just GT) ═══
  log('\n═══ Step 2: Collecting replays (50 per round) ═══');
  const replays = {};
  const REPLAYS_PER_ROUND = 50;

  for (const rn of roundsWithInits) {
    log(`  Collecting ${rn}...`);
    const { results, errors } = await collectReplays(RDS[rn], REPLAYS_PER_ROUND);
    replays[rn] = results;
    log(`  ${rn}: ${results.length} replays (${errors} errors)`);
  }

  // ═══ Step 3: Build cross-round model + submit baseline ═══
  // Use ALL rounds with replays (trajectory model doesn't need GT)
  log('\n═══ Step 3: Building cross-round model ═══');
  const growthRounds = roundsWithInits; // Use all available rounds
  const crossModel = buildCrossRoundModel(replays, inits, growthRounds);
  log(`  Model has ${Object.keys(crossModel).filter(k => k.startsWith('D0_')).length} D0 keys`);

  log('\n═══ Step 4: Submitting BASELINE predictions ═══');
  for (let si = 0; si < SEEDS; si++) {
    const p = predict(inits['TARGET'][si], crossModel, CFG);
    const res = await POST('/submit', { round_id: roundId, seed_index: si, prediction: p });
    log(`  Seed ${si}: ${res.ok ? 'ACCEPTED' : 'FAILED'} - ${JSON.stringify(res.data).slice(0, 80)}`);
    await sleep(600);
  }
  log('  ✅ Baseline submitted (cross-round model, expected ~84)');

  // ═══ Step 5: Execute viewport queries ═══
  log('\n═══ Step 5: Executing 50 viewport queries on SEED 0 ═══');
  const viewports = planAllViewports();
  log(`  Planned ${viewports.length} viewport queries`);
  log(`  Coverage: 5 full passes + 5 offset positions`);

  const { observations, queryCount, failures } =
    await executeViewportQueries(roundId, 0, viewports);

  log(`  ✅ ${observations.length} observations from ${queryCount} queries (${failures} failures)`);

  // ═══ Step 6: Build viewport feature model ═══
  log('\n═══ Step 6: Building viewport feature model ═══');
  const viewportModel = buildViewportFeatureModel(inits['TARGET'][0], observations, 0.1);

  // Analyze feature coverage
  let d0Keys = 0, totalD0Obs = 0;
  for (const k of Object.keys(viewportModel)) {
    if (k.startsWith('D0_')) { d0Keys++; totalD0Obs += viewportModel[k].n; }
  }
  log(`  D0 feature keys with observations: ${d0Keys}`);
  log(`  Average observations per D0 key: ${d0Keys > 0 ? (totalD0Obs/d0Keys).toFixed(1) : 0}`);
  log(`  Total feature observations: ${totalD0Obs}`);

  // ═══ Step 7: Fuse models and submit improved predictions ═══
  // Test multiple crossWeights for robustness
  const CROSS_WEIGHT = 30; // Validated optimal

  log(`\n═══ Step 7: Fusing models (crossWeight=${CROSS_WEIGHT}) ═══`);
  const fusedModel = fuseModels(crossModel, viewportModel, CROSS_WEIGHT);
  log(`  Fused model: ${Object.keys(fusedModel).filter(k => k.startsWith('D0_')).length} D0 keys`);

  // Validate predictions before submitting
  let validationOk = true;
  for (let si = 0; si < SEEDS; si++) {
    const p = predict(inits['TARGET'][si], fusedModel, CFG);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const s = p[y][x].reduce((a,b) => a+b, 0);
      if (Math.abs(s - 1.0) > 0.02) { validationOk = false; break; }
      for (let c = 0; c < C; c++) {
        if (p[y][x][c] < 0) { validationOk = false; break; }
      }
    }
  }

  if (!validationOk) {
    log('  ⚠️ Validation FAILED! Predictions don\'t sum to 1.0. NOT submitting.');
    log('  Keeping baseline submission.');
    return;
  }

  log(`\n═══ Step 8: Submitting IMPROVED predictions (crossWeight=${CROSS_WEIGHT}) ═══`);
  for (let si = 0; si < SEEDS; si++) {
    const p = predict(inits['TARGET'][si], fusedModel, CFG);
    const res = await POST('/submit', { round_id: roundId, seed_index: si, prediction: p });
    log(`  Seed ${si}: ${res.ok ? 'ACCEPTED' : 'FAILED'} - ${JSON.stringify(res.data).slice(0, 80)}`);
    await sleep(600);
  }
  log('  ✅ Feature-viewport predictions submitted!');

  // ═══ Step 9: Try alternative crossWeights for robustness ═══
  log('\n═══ Step 9: Testing alternative crossWeights ═══');
  for (const cw of [20, 50]) {
    const altFused = fuseModels(crossModel, viewportModel, cw);
    // Check if it predicts differently enough to matter
    let maxDiff = 0;
    for (let si = 0; si < 1; si++) { // Just check seed 0
      const p1 = predict(inits['TARGET'][si], fusedModel, CFG);
      const p2 = predict(inits['TARGET'][si], altFused, CFG);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
        for (let c = 0; c < C; c++)
          maxDiff = Math.max(maxDiff, Math.abs(p1[y][x][c] - p2[y][x][c]));
    }
    log(`  cw=${cw}: max prediction diff from cw=${CROSS_WEIGHT}: ${maxDiff.toFixed(4)}`);
  }

  log('\n╔══════════════════════════════════════════════════╗');
  log('║  ✅ PIPELINE COMPLETE                            ║');
  log(`║  Round: R${roundNumber} (${roundId.slice(0,8)}...)        ║`);
  log(`║  Viewport observations: ${observations.length}                    ║`);
  log(`║  Cross-weight: ${CROSS_WEIGHT}                              ║`);
  log(`║  Expected score: ~87-88 (validated LOO=87.7)     ║`);
  log('╚══════════════════════════════════════════════════╝');
}

main().catch(e => { console.error('Fatal:', e.message, e.stack); process.exit(1); });
