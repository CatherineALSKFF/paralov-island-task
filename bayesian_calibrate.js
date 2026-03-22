#!/usr/bin/env node
/**
 * Bayesian Viewport Calibration
 * Simulates the viewport-based approach on completed rounds (where we have GT)
 * to find optimal pseudoCount and viewport strategy.
 *
 * Method: Use replays as "viewport observations" for a test round.
 * The replay final grid at year 50 is equivalent to a viewport at year 50.
 * We sample K "observations" per cell and do Bayesian update, then score vs GT.
 */
const https = require('https');
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5, C = 6;
const TOKEN = process.argv[2] || '';
if (!TOKEN) { console.log('Usage: node bayesian_calibrate.js <JWT>'); process.exit(1); }

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

// Bayesian update: prior + observation counts → posterior
function bayesUpdate(prior, obsCounts, pseudoCount) {
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const key = `${y}_${x}`;
      if (!obsCounts[key]) { pred[y][x] = prior[y][x].slice(); continue; }
      const counts = obsCounts[key];
      const totalObs = counts.reduce((a,b)=>a+b, 0);
      if (totalObs === 0) { pred[y][x] = prior[y][x].slice(); continue; }

      // Dirichlet posterior: alpha_post = alpha_prior + counts
      const priorAlpha = prior[y][x].map(p => p * pseudoCount);
      const posterior = priorAlpha.map((a, c) => a + counts[c]);
      let s = posterior.reduce((a,b)=>a+b, 0);
      for (let c = 0; c < C; c++) {
        posterior[c] /= s;
        if (posterior[c] < 0.00001) posterior[c] = 0.00001;
      }
      s = posterior.reduce((a,b)=>a+b, 0);
      for (let c = 0; c < C; c++) posterior[c] /= s;
      pred[y][x] = posterior;
    }
  }
  return pred;
}

// Adaptive pseudoCount: lower for high-entropy cells (trust observations more)
function bayesUpdateAdaptive(prior, obsCounts, basePseudo) {
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const key = `${y}_${x}`;
      if (!obsCounts[key]) { pred[y][x] = prior[y][x].slice(); continue; }
      const counts = obsCounts[key];
      const totalObs = counts.reduce((a,b)=>a+b, 0);
      if (totalObs === 0) { pred[y][x] = prior[y][x].slice(); continue; }

      // Compute prior entropy
      let priorEnt = 0;
      for (let c = 0; c < C; c++) {
        const p = prior[y][x][c];
        if (p > 1e-6) priorEnt -= p * Math.log(p);
      }
      // High entropy → lower pseudoCount (trust observations)
      // Low entropy → higher pseudoCount (trust prior)
      const maxEnt = Math.log(C);
      const entRatio = priorEnt / maxEnt; // 0 = confident, 1 = uniform
      const pseudoCount = basePseudo * (1 - 0.7 * entRatio); // High entropy → 30% of basePseudo

      const priorAlpha = prior[y][x].map(p => p * pseudoCount);
      const posterior = priorAlpha.map((a, c) => a + counts[c]);
      let s = posterior.reduce((a,b)=>a+b, 0);
      for (let c = 0; c < C; c++) {
        posterior[c] /= s;
        if (posterior[c] < 0.00001) posterior[c] = 0.00001;
      }
      s = posterior.reduce((a,b)=>a+b, 0);
      for (let c = 0; c < C; c++) posterior[c] /= s;
      pred[y][x] = posterior;
    }
  }
  return pred;
}

async function main() {
  log('═══ Bayesian Viewport Calibration ═══');
  log('Simulating viewport observations using replays on completed rounds');

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

  // Collect replays for all rounds (these serve as "viewport observations")
  log('\nCollecting replays...');
  const replayResults = {};
  const REPLAYS = 30, CONC = 8;
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

  // ═══ SIMULATION: viewport observations using replays ═══
  log('\n═══ Simulating viewport observations (LOO) ═══');
  log('For each test round: build cross-round model, then use same-round replays as "observations"');

  // For different numbers of "observations" per cell
  for (const nObs of [1, 2, 3, 5, 8, 10, 15, 20]) {
    log(`\n--- ${nObs} observations per cell ---`);

    // Test different pseudoCounts
    for (const pseudo of [1, 2, 3, 5, 8, 10, 15, 20, 30]) {
      let total = 0, count = 0;
      for (const testRn of growthRounds) {
        const trainRns = growthRounds.filter(r => r !== testRn);
        const model = buildTrajectoryModel(replayResults, inits, trainRns);

        let seedTotal = 0;
        for (let si = 0; si < SEEDS; si++) {
          // Build cross-round prior
          const prior = predict(inits[testRn][si], model, CFG);

          // Simulate "viewport observations" using same-round replays
          const testReplays = replayResults[testRn].filter(r => r.si === si);
          const obsCounts = {};
          const used = Math.min(nObs, testReplays.length);
          for (let r = 0; r < used; r++) {
            for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
              const key = `${y}_${x}`;
              if (!obsCounts[key]) obsCounts[key] = [0,0,0,0,0,0];
              obsCounts[key][t2c(testReplays[r].finalGrid[y][x])]++;
            }
          }

          // Bayesian update
          const posterior = bayesUpdate(prior, obsCounts, pseudo);
          seedTotal += score(posterior, gts[testRn][si]);
        }
        total += seedTotal / SEEDS;
        count++;
      }
      log(`  pseudo=${pseudo}: LOO=${(total/count).toFixed(2)}`);
    }
  }

  // Test adaptive pseudoCount
  log('\n═══ Adaptive pseudoCount ═══');
  for (const nObs of [1, 2, 5, 10]) {
    for (const basePseudo of [3, 5, 8, 10, 15]) {
      let total = 0, count = 0;
      for (const testRn of growthRounds) {
        const trainRns = growthRounds.filter(r => r !== testRn);
        const model = buildTrajectoryModel(replayResults, inits, trainRns);

        let seedTotal = 0;
        for (let si = 0; si < SEEDS; si++) {
          const prior = predict(inits[testRn][si], model, CFG);
          const testReplays = replayResults[testRn].filter(r => r.si === si);
          const obsCounts = {};
          const used = Math.min(nObs, testReplays.length);
          for (let r = 0; r < used; r++) {
            for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
              const key = `${y}_${x}`;
              if (!obsCounts[key]) obsCounts[key] = [0,0,0,0,0,0];
              obsCounts[key][t2c(testReplays[r].finalGrid[y][x])]++;
            }
          }
          const posterior = bayesUpdateAdaptive(prior, obsCounts, basePseudo);
          seedTotal += score(posterior, gts[testRn][si]);
        }
        total += seedTotal / SEEDS;
        count++;
      }
      log(`  nObs=${nObs} basePseudo=${basePseudo}: LOO=${(total/count).toFixed(2)}`);
    }
  }

  // Also test: how many viewport observations do we get with 10 queries per seed?
  log('\n═══ Viewport budget analysis ═══');
  log('10 queries/seed × 15×15 viewport = 2250 cell-observations');
  log('40×40 map = 1600 cells, ~800 dynamic');
  log('Coverage: ~2250/800 = 2.8 obs per dynamic cell');
  log('With non-overlapping viewports: ~7 viewports cover full map, 3 overlap for high-entropy');

  // Summary: what score can we expect with viewport observations?
  log('\n═══ Expected R7 scores with viewport budget ═══');
  // With ~2-3 observations per cell and optimal pseudoCount
  // This is what we'd get with 10 viewport queries per seed, covering the whole map
}

main().catch(e => { console.error('Fatal:', e.message, e.stack); process.exit(1); });
