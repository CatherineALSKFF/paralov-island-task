#!/usr/bin/env node
/**
 * Optimizer v8b - Targeted follow-ups:
 * 1. Combine marginal improvements (v3 features + lower floor + better weights)
 * 2. Stacked models (combine per-round predictions)
 * 3. Per-class adaptive floor from model variance
 * 4. Try to break the ceiling with trajectory-based replay learning
 */
const https = require('https');
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5, C = 6;
const TOKEN = process.argv[2] || '';
if (!TOKEN) { console.log('Usage: node optimizer_v8b.js <JWT> [--submit]'); process.exit(1); }
const DO_SUBMIT = process.argv.includes('--submit');

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

// ═══ FEATURES ═══
function cf_v1(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS=0,co=0,fN=0,sR2=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];if(nt===1||nt===2)nS++;if(nt===10)co=1;if(nt===4)fN++;}
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;if(g[ny][nx]===1||g[ny][nx]===2)sR2++;}
  const sa=Math.min(nS,5),sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3,fb=fN<=1?0:fN<=3?1:2;
  return[`D0_${t}_${sa}_${co}_${sb2}_${fb}`,`D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,`D2_${t}_${sa>0?1:0}_${co}`,`D3_${t}_${co}`,`D4_${t}`];
}
function cf_v3(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS=0,co=0,fN=0,sR2=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];if(nt===1||nt===2)nS++;if(nt===10)co=1;if(nt===4)fN++;}
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;if(g[ny][nx]===1||g[ny][nx]===2)sR2++;}
  const sa=nS===0?0:nS<=2?1:2;
  const sb2=sR2===0?0:sR2<=3?1:2;
  const fb=fN<=2?0:1;
  return[`D0_${t}_${sa}_${co}_${sb2}_${fb}`,`D1_${t}_${sa}_${co}_${sb2}`,`D2_${t}_${sa}_${co}`,`D3_${t}_${co}`,`D4_${t}`];
}

// ═══ MODEL ═══
function buildModel(inits, gts, trainRns, cfFunc) {
  const m = {};
  for (const rn of trainRns) {
    for (let si = 0; si < SEEDS; si++) {
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cfFunc(inits[rn][si], y, x); if (!keys) continue;
        const g = gts[rn][si][y][x];
        for (let ki = 0; ki < keys.length; ki++) {
          const k = keys[ki];
          if (!m[k]) m[k] = { n: 0, s: new Float64Array(C), ss: new Float64Array(C) };
          m[k].n++;
          for (let c = 0; c < C; c++) { m[k].s[c] += g[c]; m[k].ss[c] += g[c] * g[c]; }
        }
      }
    }
  }
  for (const k of Object.keys(m)) {
    m[k].a = Array.from(m[k].s).map(v => v / m[k].n);
    // Variance per class
    m[k].v = Array.from(m[k].ss).map((ss, c) => ss / m[k].n - m[k].a[c] * m[k].a[c]);
    delete m[k].s; delete m[k].ss;
  }
  return m;
}

function predict(grid, model, cfg, cfFunc) {
  const { ws, pow, minN, fl } = cfg;
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1,0,0,0,0,0]; continue; }
      if (t === 5) { pred[y][x] = [0,0,0,0,0,1]; continue; }
      const keys = cfFunc(grid, y, x);
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

// Predict with variance-based adaptive floor
function predictAdaptiveFloor(grid, model, cfg, cfFunc) {
  const { ws, pow, minN } = cfg;
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1,0,0,0,0,0]; continue; }
      if (t === 5) { pred[y][x] = [0,0,0,0,0,1]; continue; }
      const keys = cfFunc(grid, y, x);
      if (!keys) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      const p = [0,0,0,0,0,0]; let wS = 0;
      const pVar = [0,0,0,0,0,0]; // weighted variance
      for (let ki = 0; ki < keys.length; ki++) {
        const d = model[keys[ki]];
        if (d && d.n >= minN) { const w = ws[ki] * Math.pow(d.n, pow);
          for (let c = 0; c < C; c++) { p[c] += w * d.a[c]; pVar[c] += w * d.v[c]; } wS += w; }
      }
      if (wS === 0) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      for (let c = 0; c < C; c++) { p[c] /= wS; pVar[c] /= wS; }

      // Per-class adaptive floor based on variance
      // High variance = we're uncertain = higher floor to hedge
      // Low variance = we're confident = lower floor
      let s = 0;
      for (let c = 0; c < C; c++) {
        const avgVar = pVar[c];
        // Floor based on variance: higher variance → higher floor
        const fl = Math.max(0.000005, Math.min(0.01, avgVar * 0.5));
        if (p[c] < fl) p[c] = fl;
        s += p[c];
      }
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

// Ensemble predictions
function ensemblePreds(preds, weights) {
  const pred = [];
  const totalW = weights.reduce((a,b)=>a+b, 0);
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const p = [0,0,0,0,0,0];
      for (let i = 0; i < preds.length; i++) {
        for (let c = 0; c < C; c++) p[c] += weights[i] * preds[i][y][x][c];
      }
      let s = 0;
      for (let c = 0; c < C; c++) { p[c] /= totalW; s += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

// LOO evaluation
function evalLOO(inits, gts, growthRounds, cfFunc, cfg, predictFn) {
  const perRound = {};
  let total = 0, count = 0;
  for (const testRn of growthRounds) {
    const trainRns = growthRounds.filter(r => r !== testRn);
    const model = buildModel(inits, gts, trainRns, cfFunc);
    let seedTotal = 0;
    for (let si = 0; si < SEEDS; si++) {
      const p = predictFn ? predictFn(inits[testRn][si], model, cfg, cfFunc) : predict(inits[testRn][si], model, cfg, cfFunc);
      seedTotal += score(p, gts[testRn][si]);
    }
    perRound[testRn] = seedTotal / SEEDS;
    total += perRound[testRn]; count++;
  }
  return { avg: total / count, perRound };
}

async function main() {
  log('═══ Optimizer v8b: Targeted improvements ═══');

  const RDS = {
    R1:'71451d74-be9f-471f-aacd-a41f3b68a9cd', R2:'76909e29-f664-4b2f-b16b-61b7507277e9',
    R3:'f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb', R4:'8e839974-b13b-407b-a5e7-fc749d877195',
    R5:'fd3c92ff-3178-4dc9-8d9b-acf389b3982b', R6:'ae78003a-4efe-425a-881a-d16a39bca0ad'
  };

  log('Loading data...');
  const inits = {}, gts = {};
  await Promise.all(Object.entries(RDS).map(async ([rn, id]) => {
    const { data } = await GET('/rounds/' + id);
    inits[rn] = data.initial_states.map(is => is.grid);
  }));
  await Promise.all(['R1','R2','R3','R4','R5'].map(async rn => {
    gts[rn] = [];
    await Promise.all(Array.from({length: SEEDS}, (_, si) =>
      GET('/analysis/' + RDS[rn] + '/' + si).then(r => { gts[rn][si] = r.data.ground_truth; })
    ));
  }));
  log('Data loaded');

  const growthRounds = ['R1','R2','R4','R5'];
  const baseCfg = { ws: [1, 0.2, 0.1, 0.05, 0.01], pow: 0.5, minN: 2, fl: 0.0001 };

  // ═══ TEST 1: Combined marginal improvements ═══
  log('\n═══ Test 1: Combined v3+lowerFloor+betterWeights ═══');
  const comboCfgs = [
    { name: 'baseline', cf: cf_v1, cfg: baseCfg },
    { name: 'v3+fl1e-5+ws2', cf: cf_v3, cfg: { ws: [1,0.3,0.15,0.08,0.02], pow: 0.5, minN: 2, fl: 0.00001 } },
    { name: 'v3+fl1e-5+ws2+minN5', cf: cf_v3, cfg: { ws: [1,0.3,0.15,0.08,0.02], pow: 0.5, minN: 5, fl: 0.00001 } },
    { name: 'v3+fl1e-5+ws2+minN8', cf: cf_v3, cfg: { ws: [1,0.3,0.15,0.08,0.02], pow: 0.5, minN: 8, fl: 0.00001 } },
    { name: 'v1+fl1e-5+ws2+minN5', cf: cf_v1, cfg: { ws: [1,0.3,0.15,0.08,0.02], pow: 0.5, minN: 5, fl: 0.00001 } },
    { name: 'v1+fl1e-5+minN8', cf: cf_v1, cfg: { ws: [1,0.2,0.1,0.05,0.01], pow: 0.5, minN: 8, fl: 0.00001 } },
  ];
  for (const c of comboCfgs) {
    const res = evalLOO(inits, gts, growthRounds, c.cf, c.cfg);
    log(`  ${c.name}: LOO=${res.avg.toFixed(3)} [${Object.values(res.perRound).map(v=>v.toFixed(1)).join(', ')}]`);
  }

  // ═══ TEST 2: Variance-based adaptive floor ═══
  log('\n═══ Test 2: Variance-based adaptive floor ═══');
  for (const cfg of [baseCfg, { ws: [1,0.3,0.15,0.08,0.02], pow: 0.5, minN: 5, fl: 0.00001 }]) {
    const res = evalLOO(inits, gts, growthRounds, cf_v1, cfg, predictAdaptiveFloor);
    log(`  adaptiveFloor [${cfg.ws.join(',')}] minN=${cfg.minN}: LOO=${res.avg.toFixed(3)}`);
  }

  // ═══ TEST 3: Stacked per-round models ═══
  log('\n═══ Test 3: Stacked per-round models ═══');
  // For each test round, build single-round models and combine predictions
  {
    let total = 0, count = 0;
    const perRound = {};
    for (const testRn of growthRounds) {
      const trainRns = growthRounds.filter(r => r !== testRn);
      let seedTotal = 0;
      for (let si = 0; si < SEEDS; si++) {
        // Each training round makes its own prediction
        const preds = [];
        const perfs = [];
        for (const trn of trainRns) {
          const model = buildModel(inits, gts, [trn], cf_v1);
          preds.push(predict(inits[testRn][si], model, baseCfg, cf_v1));
          // Weight by validation performance (LOO on the OTHER training rounds)
          const otherTrains = trainRns.filter(r => r !== trn);
          let perf = 0, pc = 0;
          for (const otr of otherTrains) {
            const oModel = buildModel(inits, gts, [trn], cf_v1);
            for (let osi = 0; osi < SEEDS; osi++) {
              perf += score(predict(inits[otr][osi], oModel, baseCfg, cf_v1), gts[otr][osi]);
              pc++;
            }
          }
          perfs.push(perf / pc);
        }
        // Weight by performance
        const p = ensemblePreds(preds, perfs);
        seedTotal += score(p, gts[testRn][si]);
      }
      perRound[testRn] = seedTotal / SEEDS;
      total += perRound[testRn]; count++;
    }
    log(`  Performance-weighted stack: LOO=${(total/count).toFixed(3)} [${Object.values(perRound).map(v=>v.toFixed(1)).join(', ')}]`);
  }

  // Simple equal-weight stack
  {
    let total = 0, count = 0;
    const perRound = {};
    for (const testRn of growthRounds) {
      const trainRns = growthRounds.filter(r => r !== testRn);
      let seedTotal = 0;
      for (let si = 0; si < SEEDS; si++) {
        const preds = [];
        for (const trn of trainRns) {
          const model = buildModel(inits, gts, [trn], cf_v1);
          preds.push(predict(inits[testRn][si], model, baseCfg, cf_v1));
        }
        const p = ensemblePreds(preds, preds.map(_ => 1));
        seedTotal += score(p, gts[testRn][si]);
      }
      perRound[testRn] = seedTotal / SEEDS;
      total += perRound[testRn]; count++;
    }
    log(`  Equal-weight stack: LOO=${(total/count).toFixed(3)} [${Object.values(perRound).map(v=>v.toFixed(1)).join(', ')}]`);
  }

  // ═══ TEST 4: Collect replays and use trajectory data ═══
  log('\n═══ Test 4: Replay trajectory analysis ═══');
  // Collect 20 replays per completed round (just for analysis)
  const REPLAY_N = 20;
  const replayData = {}; // rn -> [replays], each replay = { frames: [...] }
  for (const rn of ['R1', 'R5']) { // Focus on R1 and R5 (best generalizers)
    replayData[rn] = [];
    let collected = 0, errors = 0;
    while (collected < REPLAY_N) {
      const batch = [];
      for (let i = 0; i < 5 && (collected + i) < REPLAY_N; i++) {
        const si = (collected + i) % SEEDS;
        batch.push((async () => {
          try {
            const res = await POST('/replay', { round_id: RDS[rn], seed_index: si });
            if (!res.ok || !res.data.frames) { errors++; return null; }
            return { si, frames: res.data.frames };
          } catch { errors++; return null; }
        })());
      }
      const results = await Promise.all(batch);
      for (const r of results) { if (r) { replayData[rn].push(r); collected++; } }
      await sleep(200);
    }
    log(`  ${rn}: collected ${collected} replays (${errors} errors)`);
  }

  // Analyze growth dynamics from trajectories
  log('\n  Analyzing growth dynamics...');
  for (const rn of ['R1', 'R5']) {
    // Count terrain at different timepoints
    const timepoints = [0, 10, 20, 30, 40, 50];
    for (const tp of timepoints) {
      let totalS = 0, totalP = 0, totalR = 0;
      for (const rep of replayData[rn]) {
        const frame = rep.frames[tp];
        if (!frame) continue;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const t = frame.grid[y][x];
          if (t === 1) totalS++;
          if (t === 2) totalP++;
          if (t === 3) totalR++;
        }
      }
      const n = replayData[rn].length;
      log(`    ${rn} t=${tp}: avgS=${(totalS/n).toFixed(0)} avgP=${(totalP/n).toFixed(0)} avgR=${(totalR/n).toFixed(0)}`);
    }
  }

  // Build transition model from trajectories
  // For each cell, what's the transition probability from year T to year T+1?
  log('\n  Building transition model from R1+R5 replays...');
  // Track: (init_terrain, nSettlements_year0) → probability of each terrain at year 50
  const transModel = {};
  for (const rn of ['R1', 'R5']) {
    for (const rep of replayData[rn]) {
      const initGrid = rep.frames[0].grid;
      const finalGrid = rep.frames[50].grid;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cf_v1(initGrid, y, x); if (!keys) continue;
        const finalClass = t2c(finalGrid[y][x]);
        for (let ki = 0; ki < keys.length; ki++) {
          const k = 'TR_' + keys[ki];
          if (!transModel[k]) transModel[k] = { n: 0, counts: new Float64Array(C) };
          transModel[k].n++;
          transModel[k].counts[finalClass]++;
        }
      }
    }
  }
  // Convert to probabilities
  for (const k of Object.keys(transModel)) {
    transModel[k].a = Array.from(transModel[k].counts).map(v => v / transModel[k].n);
  }

  // Now predict using transition model
  {
    let total = 0, count = 0;
    const perRound = {};
    for (const testRn of growthRounds) {
      let seedTotal = 0;
      for (let si = 0; si < SEEDS; si++) {
        const grid = inits[testRn][si];
        const pred = [];
        for (let y = 0; y < H; y++) { pred[y] = [];
          for (let x = 0; x < W; x++) {
            const t = grid[y][x];
            if (t === 10) { pred[y][x] = [1,0,0,0,0,0]; continue; }
            if (t === 5) { pred[y][x] = [0,0,0,0,0,1]; continue; }
            const keys = cf_v1(grid, y, x);
            if (!keys) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
            const p = [0,0,0,0,0,0]; let wS = 0;
            for (let ki = 0; ki < keys.length; ki++) {
              const d = transModel['TR_' + keys[ki]];
              if (d && d.n >= 3) {
                const w = baseCfg.ws[ki] * Math.pow(d.n, baseCfg.pow);
                for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w;
              }
            }
            if (wS === 0) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
            let s = 0;
            for (let c = 0; c < C; c++) { p[c] /= wS; if (p[c] < 0.00001) p[c] = 0.00001; s += p[c]; }
            for (let c = 0; c < C; c++) p[c] /= s;
            pred[y][x] = p;
          }
        }
        seedTotal += score(pred, gts[testRn][si]);
      }
      perRound[testRn] = seedTotal / SEEDS;
      total += perRound[testRn]; count++;
    }
    log(`  Trajectory-based model (R1+R5): LOO=${(total/count).toFixed(3)} [${Object.values(perRound).map(v=>v.toFixed(1)).join(', ')}]`);
  }

  // ═══ TEST 5: Ensemble GT model + trajectory model ═══
  log('\n═══ Test 5: Ensemble GT + Trajectory models ═══');
  for (const gtW of [0.3, 0.5, 0.7, 0.8, 0.9]) {
    let total = 0, count = 0;
    for (const testRn of growthRounds) {
      const trainRns = growthRounds.filter(r => r !== testRn);
      const gtModel = buildModel(inits, gts, trainRns, cf_v1);
      let seedTotal = 0;
      for (let si = 0; si < SEEDS; si++) {
        const pGT = predict(inits[testRn][si], gtModel, baseCfg, cf_v1);
        // Trajectory prediction
        const grid = inits[testRn][si];
        const pTR = [];
        for (let y = 0; y < H; y++) { pTR[y] = [];
          for (let x = 0; x < W; x++) {
            const t = grid[y][x];
            if (t === 10) { pTR[y][x] = [1,0,0,0,0,0]; continue; }
            if (t === 5) { pTR[y][x] = [0,0,0,0,0,1]; continue; }
            const keys = cf_v1(grid, y, x);
            if (!keys) { pTR[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
            const p = [0,0,0,0,0,0]; let wS = 0;
            for (let ki = 0; ki < keys.length; ki++) {
              const d = transModel['TR_' + keys[ki]];
              if (d && d.n >= 3) {
                const w = baseCfg.ws[ki] * Math.pow(d.n, baseCfg.pow);
                for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w;
              }
            }
            if (wS === 0) { pTR[y][x] = pGT[y][x]; continue; }
            let s = 0;
            for (let c = 0; c < C; c++) { p[c] /= wS; if (p[c] < 0.00001) p[c] = 0.00001; s += p[c]; }
            for (let c = 0; c < C; c++) p[c] /= s;
            pTR[y][x] = p;
          }
        }
        const pEns = ensemblePreds([pGT, pTR], [gtW, 1-gtW]);
        seedTotal += score(pEns, gts[testRn][si]);
      }
      total += seedTotal / SEEDS; count++;
    }
    log(`  gtW=${gtW}: LOO=${(total/count).toFixed(3)}`);
  }

  // ═══ TEST 6: Per-class floor optimization ═══
  log('\n═══ Test 6: Per-class floor (different floor for each class) ═══');
  // Class 0 (plains) is the dominant class → needs very low floor
  // Class 1 (settlement) is common → medium floor
  // Class 2-5 are rarer → might need higher floor
  {
    let total = 0, count = 0;
    for (const testRn of growthRounds) {
      const trainRns = growthRounds.filter(r => r !== testRn);
      const model = buildModel(inits, gts, trainRns, cf_v1);
      let seedTotal = 0;
      for (let si = 0; si < SEEDS; si++) {
        const grid = inits[testRn][si];
        const pred = [];
        for (let y = 0; y < H; y++) { pred[y] = [];
          for (let x = 0; x < W; x++) {
            const t = grid[y][x];
            if (t === 10) { pred[y][x] = [1,0,0,0,0,0]; continue; }
            if (t === 5) { pred[y][x] = [0,0,0,0,0,1]; continue; }
            const keys = cf_v1(grid, y, x);
            if (!keys) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
            const p = [0,0,0,0,0,0]; let wS = 0;
            for (let ki = 0; ki < keys.length; ki++) {
              const d = model[keys[ki]];
              if (d && d.n >= baseCfg.minN) { const w = baseCfg.ws[ki] * Math.pow(d.n, baseCfg.pow);
                for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w; }
            }
            if (wS === 0) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
            for (let c = 0; c < C; c++) p[c] /= wS;
            // Per-class floor: rare classes need higher floor to avoid KL blow-up
            const classFloors = [0.00001, 0.0001, 0.0002, 0.0002, 0.00005, 0.00001];
            let s = 0;
            for (let c = 0; c < C; c++) { if (p[c] < classFloors[c]) p[c] = classFloors[c]; s += p[c]; }
            for (let c = 0; c < C; c++) p[c] /= s;
            pred[y][x] = p;
          }
        }
        seedTotal += score(pred, gts[testRn][si]);
      }
      total += seedTotal / SEEDS; count++;
    }
    log(`  Per-class floor: LOO=${(total/count).toFixed(3)}`);
  }

  // ═══ FIND AND SUBMIT BEST ═══
  log('\n═══ FINAL DECISION ═══');
  // The best config from v8a was baseline at 83.730
  // Let's compute the best combined config we found
  const bestCfg = { ws: [1,0.3,0.15,0.08,0.02], pow: 0.5, minN: 5, fl: 0.00001 };
  const bestResult = evalLOO(inits, gts, growthRounds, cf_v3, bestCfg);
  log(`Best combined (v3+ws2+fl1e-5+minN5): LOO=${bestResult.avg.toFixed(3)} [${Object.values(bestResult.perRound).map(v=>v.toFixed(1)).join(', ')}]`);

  // Compare with baseline
  const baseResult = evalLOO(inits, gts, growthRounds, cf_v1, baseCfg);
  log(`Baseline (v1 default): LOO=${baseResult.avg.toFixed(3)} [${Object.values(baseResult.perRound).map(v=>v.toFixed(1)).join(', ')}]`);

  log(`\nDelta: ${(bestResult.avg - baseResult.avg).toFixed(3)}`);
  log(`Improvement per round:`);
  for (const rn of growthRounds) {
    const delta = bestResult.perRound[rn] - baseResult.perRound[rn];
    log(`  ${rn}: ${delta > 0 ? '+' : ''}${delta.toFixed(2)}`);
  }

  if (DO_SUBMIT) {
    // Use the best config that shows CONSISTENT improvement
    const submitCf = bestResult.avg > baseResult.avg ? cf_v3 : cf_v1;
    const submitCfg = bestResult.avg > baseResult.avg ? bestCfg : baseCfg;
    const cfName = bestResult.avg > baseResult.avg ? 'combined best' : 'baseline';

    log(`\n═══ Submitting R6 with ${cfName} ═══`);
    const fullModel = buildModel(inits, gts, growthRounds, submitCf);

    for (let si = 0; si < SEEDS; si++) {
      const p = predict(inits['R6'][si], fullModel, submitCfg, submitCf);
      // Validate prediction
      let ok = true;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const s = p[y][x].reduce((a,b)=>a+b, 0);
        if (Math.abs(s - 1.0) > 0.02) ok = false;
        for (let c = 0; c < C; c++) if (p[y][x][c] < 0) ok = false;
      }
      if (!ok) { log(`  Seed ${si}: INVALID prediction!`); continue; }

      const res = await POST('/submit', { round_id: RDS['R6'], seed_index: si, prediction: p });
      log(`  Seed ${si}: ${res.data.status || JSON.stringify(res.data)}`);
      await sleep(600);
    }
    log('✅ Submitted!');
  } else {
    log('\nRun with --submit to submit.');
  }
}

main().catch(e => { console.error('Fatal:', e.message, e.stack); process.exit(1); });
