#!/usr/bin/env node
/**
 * AGGRESSIVE R7 OPTIMIZATION
 * Tests temperature scaling, round-similarity weighting, ensembles
 * via LOO validation, then submits the absolute best for ALL seeds.
 */
const https = require('https');
const fs = require('fs');
const path = require('path');
const BASE = 'https://api.ainm.no/astar-island';
const H=40,W=40,SEEDS=5,C=6;
const TOKEN = process.argv[2] || '';
const DATA_DIR = path.join(__dirname, 'data');

function api(method, pth, body) {
  return new Promise((resolve, reject) => {
    const url = new URL(BASE + pth);
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

function t2c(t) { return (t===10||t===11||t===0)?0:(t>=1&&t<=5)?t:0; }
function cf(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS=0,co=0,fN=0,sR2=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
    if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];
    if(nt===1||nt===2)nS++;if(nt===10)co=1;if(nt===4)fN++;
  }
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){
    if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;if(g[ny][nx]===1||g[ny][nx]===2)sR2++;
  }
  const sa=Math.min(nS,5),sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3,fb=fN<=1?0:fN<=3?1:2;
  return {
    d0: `D0_${t}_${sa}_${co}_${sb2}_${fb}`,
    d1: `D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,
    d2: `D2_${t}_${sa>0?1:0}_${co}`,
    d3: `D3_${t}_${co}`,
    d4: `D4_${t}`
  };
}

function buildGTModel(gtsMap, initsMap, rounds, level, alpha) {
  const m = {};
  for (const rn of rounds) {
    if (!gtsMap[rn] || !initsMap[rn]) continue;
    for (let si = 0; si < SEEDS; si++) {
      if (!initsMap[rn][si] || !gtsMap[rn][si]) continue;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cf(initsMap[rn][si], y, x); if (!keys) continue;
        const k = keys[level];
        if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
        const p = gtsMap[rn][si][y][x];
        for (let c = 0; c < C; c++) m[k].counts[c] += p[c];
        m[k].n++;
      }
    }
  }
  for (const k of Object.keys(m)) {
    const total = Array.from(m[k].counts).reduce((a,b)=>a+b,0) + C * alpha;
    m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / total);
  }
  return m;
}

function buildReplayModel(replaysMap, initsMap, rounds, level, alpha) {
  const m = {};
  for (const rn of rounds) {
    if (!replaysMap[rn] || !initsMap[rn]) continue;
    for (const rep of replaysMap[rn]) {
      const g = initsMap[rn][rep.si]; if (!g) continue;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cf(g, y, x); if (!keys) continue;
        const k = keys[level];
        const fc = t2c(rep.finalGrid[y][x]);
        if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
        m[k].n++; m[k].counts[fc]++;
      }
    }
  }
  for (const k of Object.keys(m)) {
    const total = m[k].n + C * alpha;
    m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / total);
  }
  return m;
}

function mergeModels(gtM, repM, gtWeight, alpha) {
  const model = {};
  const allKeys = new Set([...Object.keys(gtM), ...Object.keys(repM)]);
  for (const k of allKeys) {
    const gm = gtM[k], rm = repM[k];
    if (gm && rm) {
      const counts = new Float64Array(C);
      for (let c = 0; c < C; c++) counts[c] = rm.counts[c] + gm.counts[c] * gtWeight;
      const total = Array.from(counts).reduce((a,b)=>a+b,0) + C * alpha;
      model[k] = { n: rm.n + gm.n * gtWeight, counts, a: Array.from(counts).map(v => (v + alpha) / total) };
    } else if (gm) {
      const counts = new Float64Array(C);
      for (let c = 0; c < C; c++) counts[c] = gm.counts[c] * gtWeight;
      const total = Array.from(counts).reduce((a,b)=>a+b,0) + C * alpha;
      model[k] = { n: gm.n * gtWeight, counts, a: Array.from(counts).map(v => (v + alpha) / total) };
    } else {
      model[k] = { n: rm.n, counts: rm.counts, a: rm.a.slice() };
    }
  }
  return model;
}

function predict(grid, model, fl, temp) {
  fl = fl || 0.00005;
  temp = temp || 1.0;
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1,0,0,0,0,0]; continue; }
      if (t === 5) { pred[y][x] = [0,0,0,0,0,1]; continue; }
      const keys = cf(grid, y, x);
      if (!keys) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      const levels = ['d0','d1','d2','d3','d4'];
      const ws = [1.0,0.3,0.15,0.08,0.02];
      const p = [0,0,0,0,0,0]; let wS = 0;
      for (let li = 0; li < levels.length; li++) {
        const d = model[keys[levels[li]]];
        if (d && d.n >= 1) {
          const w = ws[li] * Math.pow(d.n, 0.5);
          for (let c = 0; c < C; c++) p[c] += w * d.a[c];
          wS += w;
        }
      }
      if (wS === 0) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      // Apply temperature
      let s = 0;
      for (let c = 0; c < C; c++) {
        p[c] = Math.pow(Math.max(p[c] / wS, 1e-10), 1/temp);
        if (p[c] < fl) p[c] = fl;
        s += p[c];
      }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

function computeScore(pred, gt) {
  let tE = 0, tWK = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const p = gt[y][x], q = pred[y][x];
    let e = 0; for (let c = 0; c < C; c++) if (p[c] > 0.001) e -= p[c] * Math.log(p[c]);
    if (e < 0.01) continue;
    let kl = 0; for (let c = 0; c < C; c++) if (p[c] > 0.001) kl += p[c] * Math.log(p[c] / Math.max(q[c], 1e-10));
    tE += e; tWK += e * kl;
  }
  if (tE === 0) return 100;
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * tWK / tE)));
}

// Round similarity: compare initial state features
function roundFeatures(inits) {
  let totalS=0, totalP=0, totalF=0, totalO=0, totalR=0, totalCo=0;
  for (let si = 0; si < inits.length; si++) {
    const g = inits[si]; if (!g) continue;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const t = g[y][x];
      if (t === 1) totalS++;
      if (t === 2) totalP++;
      if (t === 4) totalF++;
      if (t === 10) totalO++;
      if (t === 3) totalR++;
      if (t === 10) {
        for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
          if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;
          if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]!==10)totalCo++;
        }
      }
    }
  }
  const n = inits.length * H * W;
  return [totalS/n, totalP/n, totalF/n, totalO/n, totalR/n, totalCo/n];
}

function cosineSim(a, b) {
  let dot=0, na=0, nb=0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10);
}

async function main() {
  console.log('=== AGGRESSIVE R7 OPTIMIZATION ===');
  console.log('Time:', new Date().toISOString());

  // Load data
  const initsMap = {}, gtsMap = {}, replaysMap = {};
  const trainRounds = [];
  const SKIP = [3];
  for (let r = 1; r <= 6; r++) {
    if (SKIP.includes(r)) continue;
    const rn = `R${r}`;
    const initF = path.join(DATA_DIR, `inits_${rn}.json`);
    const gtF = path.join(DATA_DIR, `gt_${rn}.json`);
    const repF = path.join(DATA_DIR, `replays_${rn}.json`);
    if (fs.existsSync(initF)) initsMap[rn] = JSON.parse(fs.readFileSync(initF));
    if (fs.existsSync(gtF)) gtsMap[rn] = JSON.parse(fs.readFileSync(gtF));
    if (fs.existsSync(repF)) replaysMap[rn] = JSON.parse(fs.readFileSync(repF));
    if (initsMap[rn] && gtsMap[rn]) trainRounds.push(rn);
  }
  console.log('Rounds:', trainRounds.join(', '));
  console.log('Replays:', trainRounds.filter(r => replaysMap[r]).map(r => `${r}=${replaysMap[r].length}`).join(', '));

  // Load R7 inits
  const { data: r7Data } = await GET('/rounds/36e581f1-73f8-453f-ab98-cbe3052b701b');
  const r7Inits = r7Data.initial_states.map(is => is.grid);

  // === ROUND SIMILARITY ===
  console.log('\n=== Round Similarity ===');
  const r7Features = roundFeatures(r7Inits);
  console.log('R7 features:', r7Features.map(v => v.toFixed(4)).join(', '));
  
  const similarities = {};
  for (const rn of trainRounds) {
    const feat = roundFeatures(initsMap[rn]);
    const sim = cosineSim(r7Features, feat);
    similarities[rn] = sim;
    console.log(`${rn}: sim=${sim.toFixed(4)} feat=[${feat.map(v=>v.toFixed(4)).join(', ')}]`);
  }

  // === LOO VALIDATION ===
  console.log('\n=== LOO Validation ===');
  
  // Test configs: [gtWeight, alpha, temperature, level, useSimilarity]
  const configs = [];
  for (const gtW of [1, 3, 5, 10]) {
    for (const alpha of [0.03, 0.05, 0.1]) {
      for (const temp of [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]) {
        for (const level of ['d1']) {
          configs.push({ gtW, alpha, temp, level, sim: false });
        }
      }
    }
  }
  // Also test with similarity weighting
  for (const gtW of [3, 5]) {
    for (const temp of [0.8, 0.9, 1.0, 1.1]) {
      configs.push({ gtW, alpha: 0.05, temp, level: 'd1', sim: true });
    }
  }

  let bestScore = 0;
  let bestConfig = null;
  let bestDetails = null;

  for (const cfg of configs) {
    const scores = [];
    for (const testRound of trainRounds) {
      const otherRounds = trainRounds.filter(r => r !== testRound);
      
      // Build model from other rounds
      const model = {};
      for (const level of ['d0','d1','d2','d3','d4']) {
        let gtM, repM;
        if (cfg.sim) {
          // Similarity-weighted GT model
          gtM = {};
          for (const rn of otherRounds) {
            if (!gtsMap[rn]) continue;
            const w = similarities[rn] || 1;
            const partial = buildGTModel(gtsMap, initsMap, [rn], level, 0);
            for (const [k,v] of Object.entries(partial)) {
              if (!gtM[k]) gtM[k] = { n: 0, counts: new Float64Array(C) };
              gtM[k].n += v.n * w;
              for (let c = 0; c < C; c++) gtM[k].counts[c] += v.counts[c] * w;
            }
          }
          // Normalize
          for (const k of Object.keys(gtM)) {
            const total = Array.from(gtM[k].counts).reduce((a,b)=>a+b,0) + C * cfg.alpha;
            gtM[k].a = Array.from(gtM[k].counts).map(v => (v + cfg.alpha) / total);
          }
        } else {
          gtM = buildGTModel(gtsMap, initsMap, otherRounds, level, cfg.alpha);
        }
        
        const repRounds = otherRounds.filter(r => replaysMap[r]);
        repM = repRounds.length > 0 ? buildReplayModel(replaysMap, initsMap, repRounds, level, cfg.alpha) : {};
        
        const merged = mergeModels(gtM, repM, cfg.gtW, cfg.alpha);
        for (const [k,v] of Object.entries(merged)) {
          if (!model[k]) model[k] = v;
        }
      }

      // Score on test round
      for (let si = 0; si < SEEDS; si++) {
        if (!initsMap[testRound][si] || !gtsMap[testRound][si]) continue;
        const pred = predict(initsMap[testRound][si], model, 0.00005, cfg.temp);
        scores.push(computeScore(pred, gtsMap[testRound][si]));
      }
    }

    const avg = scores.reduce((a,b)=>a+b,0)/scores.length;
    if (avg > bestScore) {
      bestScore = avg;
      bestConfig = cfg;
      bestDetails = scores;
    }
  }

  console.log(`\nBEST CONFIG: gtW=${bestConfig.gtW} alpha=${bestConfig.alpha} temp=${bestConfig.temp} sim=${bestConfig.sim}`);
  console.log(`BEST LOO SCORE: ${bestScore.toFixed(3)}`);
  console.log(`Per-round: ${bestDetails.map(s=>s.toFixed(1)).join(', ')}`);

  // Also test: What if we use the D0 level with temperature?
  console.log('\n=== Testing D0 with temperature ===');
  let bestD0 = 0, bestD0Cfg = null;
  for (const temp of [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]) {
    const scores = [];
    for (const testRound of trainRounds) {
      const otherRounds = trainRounds.filter(r => r !== testRound);
      const model = {};
      for (const level of ['d0','d1','d2','d3','d4']) {
        const gtM = buildGTModel(gtsMap, initsMap, otherRounds, level, 0.05);
        const repRounds = otherRounds.filter(r => replaysMap[r]);
        const repM = repRounds.length > 0 ? buildReplayModel(replaysMap, initsMap, repRounds, level, 0.05) : {};
        const merged = mergeModels(gtM, repM, 3, 0.05);
        for (const [k,v] of Object.entries(merged)) if (!model[k]) model[k] = v;
      }
      for (let si = 0; si < SEEDS; si++) {
        if (!initsMap[testRound][si] || !gtsMap[testRound][si]) continue;
        const pred = predict(initsMap[testRound][si], model, 0.00005, temp);
        scores.push(computeScore(pred, gtsMap[testRound][si]));
      }
    }
    const avg = scores.reduce((a,b)=>a+b,0)/scores.length;
    if (avg > bestD0) { bestD0 = avg; bestD0Cfg = { temp }; }
    console.log(`  temp=${temp}: ${avg.toFixed(3)}`);
  }

  // === BUILD FINAL MODEL AND SUBMIT ALL SEEDS ===
  console.log('\n=== Building FINAL model with best config ===');
  const finalCfg = bestConfig;
  const finalModel = {};
  
  for (const level of ['d0','d1','d2','d3','d4']) {
    let gtM;
    if (finalCfg.sim) {
      gtM = {};
      for (const rn of trainRounds) {
        if (!gtsMap[rn]) continue;
        const w = similarities[rn] || 1;
        const partial = buildGTModel(gtsMap, initsMap, [rn], level, 0);
        for (const [k,v] of Object.entries(partial)) {
          if (!gtM[k]) gtM[k] = { n: 0, counts: new Float64Array(C) };
          gtM[k].n += v.n * w;
          for (let c = 0; c < C; c++) gtM[k].counts[c] += v.counts[c] * w;
        }
      }
      for (const k of Object.keys(gtM)) {
        const total = Array.from(gtM[k].counts).reduce((a,b)=>a+b,0) + C * finalCfg.alpha;
        gtM[k].a = Array.from(gtM[k].counts).map(v => (v + finalCfg.alpha) / total);
      }
    } else {
      gtM = buildGTModel(gtsMap, initsMap, trainRounds, level, finalCfg.alpha);
    }

    const repRounds = trainRounds.filter(r => replaysMap[r]);
    const repM = repRounds.length > 0 ? buildReplayModel(replaysMap, initsMap, repRounds, level, finalCfg.alpha) : {};
    const merged = mergeModels(gtM, repM, finalCfg.gtW, finalCfg.alpha);
    for (const [k,v] of Object.entries(merged)) {
      if (!finalModel[k]) finalModel[k] = v;
    }
  }

  console.log(`Final model: ${Object.keys(finalModel).length} keys`);

  // Submit ONLY seed 4 (seeds 0-3 have viewport data, don't overwrite)
  const R7_ID = '36e581f1-73f8-453f-ab98-cbe3052b701b';
  console.log('\n=== Submitting R7 seed 4 with optimized model ===');
  
  const p4 = predict(r7Inits[4], finalModel, 0.00005, finalCfg.temp);
  let valid = true;
  for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
    const s = p4[y][x].reduce((a,b)=>a+b,0);
    if (Math.abs(s-1) > 0.02 || p4[y][x].some(v => v < 0)) valid = false;
  }
  
  if (valid) {
    const res = await POST('/submit', { round_id: R7_ID, seed_index: 4, prediction: p4 });
    console.log(`Seed 4: ${res.ok ? 'ACCEPTED' : 'FAILED'} ${JSON.stringify(res.data).slice(0,100)}`);
  }

  // If LOO score > 84, also submit seeds 0-3 (overwrite viewport predictions)
  if (bestScore > 84) {
    console.log(`\nLOO score ${bestScore.toFixed(2)} > 84, submitting ALL seeds!`);
    for (let si = 0; si < 4; si++) {
      const p = predict(r7Inits[si], finalModel, 0.00005, finalCfg.temp);
      const res = await POST('/submit', { round_id: R7_ID, seed_index: si, prediction: p });
      console.log(`Seed ${si}: ${res.ok ? 'ACCEPTED' : 'FAILED'} ${JSON.stringify(res.data).slice(0,100)}`);
      await sleep(600);
    }
  } else {
    console.log(`\nLOO score ${bestScore.toFixed(2)} <= 84, keeping viewport-enhanced predictions for seeds 0-3`);
  }

  console.log('\n=== DONE ===');
  console.log(`Best LOO: ${bestScore.toFixed(3)}`);
  console.log(`R7 weight: 1.4071`);
  console.log(`Expected ws from cross-round only: ${(bestScore * 1.4071).toFixed(1)}`);
  console.log(`Estimated ws with viewport seeds 0-3 (~87): ${((87*4 + bestScore) / 5 * 1.4071).toFixed(1)}`);
}

main().catch(e => console.error('Error:', e.message, e.stack));
