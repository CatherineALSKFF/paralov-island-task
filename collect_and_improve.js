#!/usr/bin/env node
/**
 * Collect replays for R1-R5, build enriched model, test, resubmit R6
 */
const https = require('https');
const fs = require('fs');
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5, C = 6;
const TOKEN = process.argv[2];
if (!TOKEN) { console.log('Usage: node collect_and_improve.js <JWT>'); process.exit(1); }

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

// Features
function cf(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS=0,co=0,fN=0,sR2=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];if(nt===1||nt===2)nS++;if(nt===10)co=1;if(nt===4)fN++;}
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;if(g[ny][nx]===1||g[ny][nx]===2)sR2++;}
  const sa=Math.min(nS,5),sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3,fb=fN<=1?0:fN<=3?1:2;
  return[`D0_${t}_${sa}_${co}_${sb2}_${fb}`,`D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,`D2_${t}_${sa>0?1:0}_${co}`,`D3_${t}_${co}`,`D4_${t}`];
}

// Build model from GT distributions
function buildGT(inits, gts, rns) {
  const m = {};
  for (const rn of rns) for (let si = 0; si < SEEDS; si++) {
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const keys = cf(inits[rn][si], y, x); if (!keys) continue;
      const g = gts[rn][si][y][x];
      for (let ki = 0; ki < keys.length; ki++) {
        const k = keys[ki];
        if (!m[k]) m[k] = { n: 0, s: new Float64Array(C) };
        m[k].n++; for (let c = 0; c < C; c++) m[k].s[c] += g[c];
      }
    }
  }
  for (const k of Object.keys(m)) { m[k].a = Array.from(m[k].s).map(v => v / m[k].n); delete m[k].s; }
  return m;
}

// Build model from replay counts (per-cell class counts)
function buildReplay(inits, replayCounts, rns) {
  const m = {};
  for (const rn of rns) {
    if (!replayCounts[rn]) continue;
    for (let si = 0; si < SEEDS; si++) {
      const counts = replayCounts[rn][si];
      if (!counts) continue;
      const N = counts._N || 1;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cf(inits[rn][si], y, x); if (!keys) continue;
        const key = `${y}_${x}`;
        const cc = counts[key] || [0,0,0,0,0,0];
        const total = cc.reduce((a,b)=>a+b, 0);
        if (total === 0) continue;
        const prob = cc.map(v => v / total);
        for (let ki = 0; ki < keys.length; ki++) {
          const k = keys[ki];
          if (!m[k]) m[k] = { n: 0, s: new Float64Array(C) };
          m[k].n++; for (let c = 0; c < C; c++) m[k].s[c] += prob[c];
        }
      }
    }
  }
  for (const k of Object.keys(m)) { m[k].a = Array.from(m[k].s).map(v => v / m[k].n); delete m[k].s; }
  return m;
}

// Predict
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

// Ensemble two predictions
function ensemble(p1, p2, w1, w2) {
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const p = []; let s = 0;
      for (let c = 0; c < C; c++) { p[c] = w1 * p1[y][x][c] + w2 * p2[y][x][c]; s += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

// Score
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

const CFG = { ws: [1, 0.2, 0.1, 0.05, 0.01], pow: 0.5, minN: 2, fl: 0.0001 };

async function main() {
  log('═══ Astar Island: Collect, Build, Improve, Submit ═══');

  // Fetch all data
  log('Fetching init states + GT...');
  const RDS = {
    R1:'71451d74-be9f-471f-aacd-a41f3b68a9cd', R2:'76909e29-f664-4b2f-b16b-61b7507277e9',
    R3:'f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb', R4:'8e839974-b13b-407b-a5e7-fc749d877195',
    R5:'fd3c92ff-3178-4dc9-8d9b-acf389b3982b', R6:'ae78003a-4efe-425a-881a-d16a39bca0ad'
  };
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
  log('All data loaded');

  // ═══ PHASE A: Collect replays for R1-R5 (focus on R5, most similar to R6) ═══
  log('Collecting replays (R5 priority, then R1/R2/R4)...');
  const replayCounts = {};
  const ROUNDS_TO_COLLECT = ['R5', 'R1', 'R2', 'R4']; // R5 first (most similar to R6)
  const TARGET_PER_ROUND = 50; // 50 replays per round = 200 total
  const CONCURRENCY = 10;

  for (const rn of ROUNDS_TO_COLLECT) {
    replayCounts[rn] = {};
    for (let si = 0; si < SEEDS; si++) replayCounts[rn][si] = { _N: 0 };
    
    let collected = 0, errors = 0;
    while (collected < TARGET_PER_ROUND) {
      const batch = [];
      for (let i = 0; i < CONCURRENCY; i++) {
        const si = (collected + i) % SEEDS;
        batch.push((async () => {
          try {
            const res = await POST('/replay', { round_id: RDS[rn], seed_index: si });
            if (!res.ok || !res.data.frames) { errors++; return null; }
            return { si, grid: res.data.frames[res.data.frames.length - 1].grid };
          } catch { errors++; return null; }
        })());
      }
      const results = await Promise.all(batch);
      for (const r of results) {
        if (!r) continue;
        const counts = replayCounts[rn][r.si];
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const key = `${y}_${x}`;
          if (!counts[key]) counts[key] = [0,0,0,0,0,0];
          counts[key][t2c(r.grid[y][x])]++;
        }
        counts._N++;
        collected++;
      }
      await sleep(150);
    }
    log(`  ${rn}: ${collected} replays (${errors} errors)`);
  }

  // ═══ PHASE B: Build models and test ═══
  log('\n═══ Testing models ═══');
  const growthRounds = ['R1','R2','R4','R5'];

  // Model 1: GT-only (baseline)
  // Model 2: Replay-only
  // Model 3: GT + Replay ensemble

  // Test on each growth round (LOO)
  for (const approach of ['GT-only', 'Replay-only', 'Ensemble 70/30', 'Ensemble 50/50']) {
    let totalScore = 0, count = 0;
    for (const testRn of growthRounds) {
      const trainRns = growthRounds.filter(r => r !== testRn);
      const gtModel = buildGT(inits, gts, trainRns);
      const repModel = buildReplay(inits, replayCounts, trainRns);
      
      let seedTotal = 0;
      for (let si = 0; si < SEEDS; si++) {
        let p;
        if (approach === 'GT-only') {
          p = predict(inits[testRn][si], gtModel, CFG);
        } else if (approach === 'Replay-only') {
          p = predict(inits[testRn][si], repModel, CFG);
        } else if (approach === 'Ensemble 70/30') {
          const p1 = predict(inits[testRn][si], gtModel, CFG);
          const p2 = predict(inits[testRn][si], repModel, CFG);
          p = ensemble(p1, p2, 0.7, 0.3);
        } else {
          const p1 = predict(inits[testRn][si], gtModel, CFG);
          const p2 = predict(inits[testRn][si], repModel, CFG);
          p = ensemble(p1, p2, 0.5, 0.5);
        }
        seedTotal += score(p, gts[testRn][si]);
      }
      totalScore += seedTotal / SEEDS;
      count++;
    }
    log(`  ${approach}: LOO avg = ${(totalScore / count).toFixed(2)}`);
  }

  // ═══ PHASE C: Find best config and submit R6 ═══
  log('\n═══ Finding best approach for R6 ═══');
  
  // Build models on ALL growth rounds
  const gtModelFull = buildGT(inits, gts, growthRounds);
  const repModelFull = buildReplay(inits, replayCounts, growthRounds);
  
  // Test different ensemble weights (validate on each round)
  let bestW = 1.0, bestAvg = 0;
  for (let w = 0; w <= 1.0; w += 0.1) {
    let total = 0;
    for (const testRn of growthRounds) {
      const trainRns = growthRounds.filter(r => r !== testRn);
      const gm = buildGT(inits, gts, trainRns);
      const rm = buildReplay(inits, replayCounts, trainRns);
      for (let si = 0; si < SEEDS; si++) {
        const p1 = predict(inits[testRn][si], gm, CFG);
        const p2 = predict(inits[testRn][si], rm, CFG);
        const p = ensemble(p1, p2, w, 1 - w);
        total += score(p, gts[testRn][si]);
      }
    }
    const avg = total / (growthRounds.length * SEEDS);
    log(`  GT weight=${w.toFixed(1)}: LOO=${avg.toFixed(2)}`);
    if (avg > bestAvg) { bestAvg = avg; bestW = w; }
  }
  log(`Best ensemble: GT weight=${bestW.toFixed(1)}, LOO=${bestAvg.toFixed(2)}`);

  // Submit best model for R6
  log('\n═══ Submitting R6 ═══');
  for (let si = 0; si < SEEDS; si++) {
    const p1 = predict(inits['R6'][si], gtModelFull, CFG);
    const p2 = predict(inits['R6'][si], repModelFull, CFG);
    const p = ensemble(p1, p2, bestW, 1 - bestW);
    
    const res = await POST('/submit', { round_id: RDS['R6'], seed_index: si, prediction: p });
    log(`  Seed ${si}: ${res.data.status || JSON.stringify(res.data)}`);
    await sleep(600);
  }
  
  log('\n✅ Done! Check leaderboard for R6 score.');
}

main().catch(e => { console.error('Fatal:', e.message); process.exit(1); });
