#!/usr/bin/env node
/**
 * Comprehensive R6 Optimizer v8
 * Tests multiple strategies with HONEST 4-round LOO
 * Guards against over-optimization by requiring consistent improvement across ALL folds
 */
const https = require('https');
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5, C = 6;
const TOKEN = process.argv[2] || '';
if (!TOKEN) { console.log('Usage: node optimizer_v8.js <JWT> [--submit]'); process.exit(1); }
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

// ═══ FEATURE ENGINEERING ═══

// Original 5-level features
function cf_v1(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS=0,co=0,fN=0,sR2=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];if(nt===1||nt===2)nS++;if(nt===10)co=1;if(nt===4)fN++;}
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;if(g[ny][nx]===1||g[ny][nx]===2)sR2++;}
  const sa=Math.min(nS,5),sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3,fb=fN<=1?0:fN<=3?1:2;
  return[`D0_${t}_${sa}_${co}_${sb2}_${fb}`,`D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,`D2_${t}_${sa>0?1:0}_${co}`,`D3_${t}_${co}`,`D4_${t}`];
}

// Enhanced features v2: add distance to mountain, wider settlement radius, ruin proximity
function cf_v2(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS=0,co=0,fN=0,sR2=0,mNear=0,ruinN=0,portN=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];if(nt===1||nt===2)nS++;if(nt===10)co=1;if(nt===4)fN++;if(nt===5)mNear=1;if(nt===3)ruinN++;}
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];if(nt===1||nt===2)sR2++;if(nt===2)portN++;}
  // R3 settlements
  let sR3=0;
  for(let dy=-3;dy<=3;dy++)for(let dx=-3;dx<=3;dx++){if(Math.abs(dy)<=2&&Math.abs(dx)<=2)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;if(g[ny][nx]===1||g[ny][nx]===2)sR3++;}
  const sa=Math.min(nS,5),sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3,fb=fN<=1?0:fN<=3?1:2;
  const sb3=sR3===0?0:sR3<=3?1:2;
  const rn=ruinN>0?1:0;
  return[
    `D0_${t}_${sa}_${co}_${sb2}_${fb}`,
    `D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,
    `D2_${t}_${sa>0?1:0}_${co}_${mNear}`,
    `D3_${t}_${co}`,
    `D4_${t}`
  ];
}

// Features v3: simplified, fewer buckets to reduce overfitting
function cf_v3(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS=0,co=0,fN=0,sR2=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];if(nt===1||nt===2)nS++;if(nt===10)co=1;if(nt===4)fN++;}
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;if(g[ny][nx]===1||g[ny][nx]===2)sR2++;}
  const sa=nS===0?0:nS<=2?1:2; // simplified: 0, 1-2, 3+
  const sb2=sR2===0?0:sR2<=3?1:2;
  const fb=fN<=2?0:1;
  return[
    `D0_${t}_${sa}_${co}_${sb2}_${fb}`,
    `D1_${t}_${sa}_${co}_${sb2}`,
    `D2_${t}_${sa}_${co}`,
    `D3_${t}_${co}`,
    `D4_${t}`
  ];
}

// ═══ ROUND SIMILARITY ═══
function roundStats(inits, rn) {
  let totalS=0, totalF=0, totalM=0, totalP=0, totalR=0, totalO=0;
  for (let si = 0; si < SEEDS; si++) {
    const g = inits[rn][si];
    for (let y=0;y<H;y++) for (let x=0;x<W;x++) {
      const t = g[y][x];
      if (t===1) totalS++;
      else if (t===2) totalP++;
      else if (t===3) totalR++;
      else if (t===4) totalF++;
      else if (t===5) totalM++;
      else if (t===10) totalO++;
    }
  }
  const n = SEEDS;
  return { S: totalS/n, P: totalP/n, R: totalR/n, F: totalF/n, M: totalM/n, O: totalO/n };
}

function roundSimilarity(stats1, stats2) {
  // Euclidean distance in normalized feature space
  const keys = ['S','P','R','F','M','O'];
  let sumSq = 0;
  for (const k of keys) {
    const maxV = Math.max(stats1[k], stats2[k], 1);
    const diff = (stats1[k] - stats2[k]) / maxV;
    sumSq += diff * diff;
  }
  return 1.0 / (1.0 + Math.sqrt(sumSq));
}

// ═══ MODEL BUILDING ═══
function buildModel(inits, gts, trainRns, cfFunc, roundWeights) {
  const m = {};
  for (const rn of trainRns) {
    const w = roundWeights ? (roundWeights[rn] || 1.0) : 1.0;
    for (let si = 0; si < SEEDS; si++) {
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cfFunc(inits[rn][si], y, x); if (!keys) continue;
        const g = gts[rn][si][y][x];
        for (let ki = 0; ki < keys.length; ki++) {
          const k = keys[ki];
          if (!m[k]) m[k] = { n: 0, wn: 0, s: new Float64Array(C) };
          m[k].n++; m[k].wn += w;
          for (let c = 0; c < C; c++) m[k].s[c] += w * g[c];
        }
      }
    }
  }
  for (const k of Object.keys(m)) {
    m[k].a = Array.from(m[k].s).map(v => v / m[k].wn);
    delete m[k].s;
  }
  return m;
}

// ═══ PREDICTION ═══
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
        if (d && d.n >= minN) {
          const w = ws[ki] * Math.pow(d.n, pow);
          for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w;
        }
      }
      if (wS === 0) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      let s = 0; for (let c = 0; c < C; c++) { p[c] /= wS; if (p[c] < fl) p[c] = fl; s += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

// ═══ PREDICTION WITH CONFIDENCE CALIBRATION ═══
// Shrink uncertain predictions toward uniform to reduce catastrophic KL
function predictCalibrated(grid, model, cfg, cfFunc, shrinkFactor) {
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
      let totalN = 0;
      for (let ki = 0; ki < keys.length; ki++) {
        const d = model[keys[ki]];
        if (d && d.n >= minN) {
          const w = ws[ki] * Math.pow(d.n, pow);
          for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w;
          totalN += d.n;
        }
      }
      if (wS === 0) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      for (let c = 0; c < C; c++) p[c] /= wS;

      // Compute prediction entropy
      let predEnt = 0;
      for (let c = 0; c < C; c++) if (p[c] > 1e-10) predEnt -= p[c] * Math.log(p[c]);

      // High entropy = uncertain. Shrink toward uniform proportionally
      // shrinkFactor controls how much to shrink high-entropy predictions
      const shrink = shrinkFactor * (predEnt / Math.log(C));
      const uniform = 1/C;
      let s = 0;
      for (let c = 0; c < C; c++) {
        p[c] = (1 - shrink) * p[c] + shrink * uniform;
        if (p[c] < fl) p[c] = fl;
        s += p[c];
      }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

// ═══ SCORING ═══
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

// ═══ LOO EVALUATION ═══
// Returns { avg, perRound: {R1: x, R2: x, ...}, std }
function evalLOO(inits, gts, growthRounds, cfFunc, cfg, roundWeightsFn, shrinkFactor) {
  const perRound = {};
  let total = 0, count = 0;

  for (const testRn of growthRounds) {
    const trainRns = growthRounds.filter(r => r !== testRn);
    const rw = roundWeightsFn ? roundWeightsFn(testRn, trainRns) : null;
    const model = buildModel(inits, gts, trainRns, cfFunc, rw);

    let seedTotal = 0;
    for (let si = 0; si < SEEDS; si++) {
      let p;
      if (shrinkFactor && shrinkFactor > 0) {
        p = predictCalibrated(inits[testRn][si], model, cfg, cfFunc, shrinkFactor);
      } else {
        p = predict(inits[testRn][si], model, cfg, cfFunc);
      }
      seedTotal += score(p, gts[testRn][si]);
    }
    const avgSeed = seedTotal / SEEDS;
    perRound[testRn] = avgSeed;
    total += avgSeed;
    count++;
  }

  const avg = total / count;
  let variance = 0;
  for (const rn of growthRounds) variance += (perRound[rn] - avg) ** 2;
  const std = Math.sqrt(variance / count);

  return { avg, perRound, std };
}

// ═══ MAIN ═══
async function main() {
  log('═══ Astar Island Optimizer v8 ═══');
  log('Over-optimization guard: require improvement on MAJORITY of folds');

  const RDS = {
    R1:'71451d74-be9f-471f-aacd-a41f3b68a9cd', R2:'76909e29-f664-4b2f-b16b-61b7507277e9',
    R3:'f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb', R4:'8e839974-b13b-407b-a5e7-fc749d877195',
    R5:'fd3c92ff-3178-4dc9-8d9b-acf389b3982b', R6:'ae78003a-4efe-425a-881a-d16a39bca0ad'
  };

  // Load all data
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

  // Round stats
  const allStats = {};
  for (const rn of ['R1','R2','R3','R4','R5','R6']) {
    allStats[rn] = roundStats(inits, rn);
    log(`  ${rn}: S=${allStats[rn].S.toFixed(0)} P=${allStats[rn].P.toFixed(0)} R=${allStats[rn].R.toFixed(0)} F=${allStats[rn].F.toFixed(0)} M=${allStats[rn].M.toFixed(0)}`);
  }

  const growthRounds = ['R1','R2','R4','R5'];

  // ═══ STRATEGY 1: Baseline (current best) ═══
  log('\n═══ Strategy 1: Baseline (v1 features, uniform weights) ═══');
  const baseCfg = { ws: [1, 0.2, 0.1, 0.05, 0.01], pow: 0.5, minN: 2, fl: 0.0001 };
  const baseResult = evalLOO(inits, gts, growthRounds, cf_v1, baseCfg);
  log(`  LOO avg=${baseResult.avg.toFixed(3)} std=${baseResult.std.toFixed(2)}`);
  for (const [rn, s] of Object.entries(baseResult.perRound)) log(`    ${rn}: ${s.toFixed(2)}`);

  // ═══ STRATEGY 2: Round-similarity weighted training ═══
  log('\n═══ Strategy 2: Similarity-weighted training ═══');
  const simWeightFn = (testRn, trainRns) => {
    const w = {};
    for (const trn of trainRns) {
      const sim = roundSimilarity(allStats[testRn], allStats[trn]);
      w[trn] = sim;
    }
    return w;
  };
  // For R6
  const r6simWeightFn = (testRn, trainRns) => {
    const w = {};
    for (const trn of trainRns) {
      const sim = roundSimilarity(allStats['R6'], allStats[trn]);
      w[trn] = sim;
    }
    return w;
  };

  const sim2Result = evalLOO(inits, gts, growthRounds, cf_v1, baseCfg, simWeightFn);
  log(`  LOO avg=${sim2Result.avg.toFixed(3)} std=${sim2Result.std.toFixed(2)}`);
  for (const [rn, s] of Object.entries(sim2Result.perRound)) log(`    ${rn}: ${s.toFixed(2)}`);

  // ═══ STRATEGY 3: Stronger similarity weighting (squared) ═══
  log('\n═══ Strategy 3: Squared similarity weighting ═══');
  const simSqFn = (testRn, trainRns) => {
    const w = {};
    for (const trn of trainRns) {
      const sim = roundSimilarity(allStats[testRn], allStats[trn]);
      w[trn] = sim * sim;
    }
    return w;
  };
  const sim3Result = evalLOO(inits, gts, growthRounds, cf_v1, baseCfg, simSqFn);
  log(`  LOO avg=${sim3Result.avg.toFixed(3)} std=${sim3Result.std.toFixed(2)}`);
  for (const [rn, s] of Object.entries(sim3Result.perRound)) log(`    ${rn}: ${s.toFixed(2)}`);

  // ═══ STRATEGY 4: Confidence calibration ═══
  log('\n═══ Strategy 4: Confidence calibration (shrink uncertain predictions) ═══');
  for (const shrink of [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]) {
    const res = evalLOO(inits, gts, growthRounds, cf_v1, baseCfg, null, shrink);
    log(`  shrink=${shrink}: LOO avg=${res.avg.toFixed(3)} std=${res.std.toFixed(2)}`);
  }

  // ═══ STRATEGY 5: Different feature sets ═══
  log('\n═══ Strategy 5: Feature set comparison ═══');
  const v2Result = evalLOO(inits, gts, growthRounds, cf_v2, baseCfg);
  log(`  v2 features: LOO avg=${v2Result.avg.toFixed(3)} std=${v2Result.std.toFixed(2)}`);
  const v3Result = evalLOO(inits, gts, growthRounds, cf_v3, baseCfg);
  log(`  v3 features (simplified): LOO avg=${v3Result.avg.toFixed(3)} std=${v3Result.std.toFixed(2)}`);

  // ═══ STRATEGY 6: Floor optimization ═══
  log('\n═══ Strategy 6: Floor optimization ═══');
  for (const fl of [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]) {
    const cfg = { ...baseCfg, fl };
    const res = evalLOO(inits, gts, growthRounds, cf_v1, cfg);
    log(`  fl=${fl}: LOO avg=${res.avg.toFixed(3)}`);
  }

  // ═══ STRATEGY 7: Weight optimization ═══
  log('\n═══ Strategy 7: Level weight optimization ═══');
  const weightSets = [
    [1, 0.2, 0.1, 0.05, 0.01],  // current
    [1, 0.3, 0.15, 0.08, 0.02],
    [1, 0.1, 0.05, 0.02, 0.005],
    [1, 0.4, 0.2, 0.1, 0.03],
    [1, 0.15, 0.08, 0.04, 0.01],
    [1, 0.5, 0.25, 0.12, 0.05],
    [1, 0.2, 0.1, 0.05, 0.0],   // no D4
    [1, 0.2, 0.1, 0.0, 0.0],    // no D3/D4
    [1, 0.3, 0.0, 0.0, 0.0],    // D0 + D1 only
    [1, 0.0, 0.0, 0.0, 0.0],    // D0 only
  ];
  for (const ws of weightSets) {
    const cfg = { ...baseCfg, ws };
    const res = evalLOO(inits, gts, growthRounds, cf_v1, cfg);
    log(`  ws=[${ws.join(',')}]: LOO avg=${res.avg.toFixed(3)}`);
  }

  // ═══ STRATEGY 8: Power parameter ═══
  log('\n═══ Strategy 8: Power parameter ═══');
  for (const pow of [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) {
    const cfg = { ...baseCfg, pow };
    const res = evalLOO(inits, gts, growthRounds, cf_v1, cfg);
    log(`  pow=${pow}: LOO avg=${res.avg.toFixed(3)}`);
  }

  // ═══ STRATEGY 9: minN parameter ═══
  log('\n═══ Strategy 9: minN parameter ═══');
  for (const minN of [1, 2, 3, 5, 8, 10, 15, 20]) {
    const cfg = { ...baseCfg, minN };
    const res = evalLOO(inits, gts, growthRounds, cf_v1, cfg);
    log(`  minN=${minN}: LOO avg=${res.avg.toFixed(3)}`);
  }

  // ═══ STRATEGY 10: Exclude worst-matching round from training ═══
  log('\n═══ Strategy 10: Training set selection ═══');
  const trainSets = [
    { name: 'R1,R2,R4,R5', rns: ['R1','R2','R4','R5'] },
    { name: 'R1,R2,R5', rns: ['R1','R2','R5'] },
    { name: 'R1,R4,R5', rns: ['R1','R4','R5'] },
    { name: 'R2,R4,R5', rns: ['R2','R4','R5'] },
    { name: 'R1,R2,R4', rns: ['R1','R2','R4'] },
    { name: 'R5 only', rns: ['R5'] },
    { name: 'R1 only', rns: ['R1'] },
    { name: 'R1,R5', rns: ['R1','R5'] },
    { name: 'R4,R5', rns: ['R4','R5'] },
    { name: 'ALL incl R3', rns: ['R1','R2','R3','R4','R5'] },
  ];
  for (const ts of trainSets) {
    // Can only evaluate on rounds NOT in training
    const testRns = growthRounds.filter(r => !ts.rns.includes(r));
    if (testRns.length === 0) {
      // LOO within the training set
      if (ts.rns.length >= 2) {
        const res = evalLOO(inits, gts, ts.rns, cf_v1, baseCfg);
        log(`  Train=${ts.name}: LOO avg=${res.avg.toFixed(3)} (self-LOO, ${ts.rns.length} rounds)`);
      }
      continue;
    }
    // Test on held-out rounds
    const model = buildModel(inits, gts, ts.rns, cf_v1, null);
    let total = 0, cnt = 0;
    for (const tr of testRns) {
      for (let si = 0; si < SEEDS; si++) {
        total += score(predict(inits[tr][si], model, baseCfg, cf_v1), gts[tr][si]);
        cnt++;
      }
    }
    log(`  Train=${ts.name}: heldout avg=${(total/cnt).toFixed(3)} on ${testRns.join(',')}`);
  }

  // ═══ STRATEGY 11: Combined best strategies ═══
  log('\n═══ Strategy 11: Combined approaches ═══');
  // Combine similarity weighting + confidence calibration + best cfg
  for (const shrink of [0.0, 0.05, 0.1, 0.15]) {
    for (const pow of [0.3, 0.5]) {
      const cfg = { ws: [1, 0.2, 0.1, 0.05, 0.01], pow, minN: 2, fl: 0.0001 };
      const res = evalLOO(inits, gts, growthRounds, cf_v1, cfg, simWeightFn, shrink);
      log(`  sim+shrink=${shrink}+pow=${pow}: LOO avg=${res.avg.toFixed(3)} std=${res.std.toFixed(2)}`);
    }
  }

  // ═══ STRATEGY 12: KL-optimal floor per feature type ═══
  // Instead of global floor, use adaptive floor based on prediction confidence
  log('\n═══ Strategy 12: Adaptive floor by cell confidence ═══');
  {
    const cfg12 = { ...baseCfg };
    // Predict, then apply adaptive floor
    let total = 0, count = 0;
    for (const testRn of growthRounds) {
      const trainRns = growthRounds.filter(r => r !== testRn);
      const model = buildModel(inits, gts, trainRns, cf_v1, null);

      let seedTotal = 0;
      for (let si = 0; si < SEEDS; si++) {
        const pred = [];
        const grid = inits[testRn][si];
        for (let y = 0; y < H; y++) { pred[y] = [];
          for (let x = 0; x < W; x++) {
            const t = grid[y][x];
            if (t === 10) { pred[y][x] = [1,0,0,0,0,0]; continue; }
            if (t === 5) { pred[y][x] = [0,0,0,0,0,1]; continue; }
            const keys = cf_v1(grid, y, x);
            if (!keys) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
            const p = [0,0,0,0,0,0]; let wS = 0;
            let maxN = 0;
            for (let ki = 0; ki < keys.length; ki++) {
              const d = model[keys[ki]];
              if (d && d.n >= cfg12.minN) {
                const w = cfg12.ws[ki] * Math.pow(d.n, cfg12.pow);
                for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w;
                if (d.n > maxN) maxN = d.n;
              }
            }
            if (wS === 0) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
            for (let c = 0; c < C; c++) p[c] /= wS;

            // Count non-zero classes
            let nzClasses = 0;
            for (let c = 0; c < C; c++) if (p[c] > 0.01) nzClasses++;

            // Adaptive floor: more classes = higher floor
            let fl;
            if (nzClasses <= 1) fl = 0.00001;
            else if (nzClasses <= 2) fl = 0.0001;
            else if (nzClasses <= 3) fl = 0.0005;
            else fl = 0.001;

            let s = 0;
            for (let c = 0; c < C; c++) { if (p[c] < fl) p[c] = fl; s += p[c]; }
            for (let c = 0; c < C; c++) p[c] /= s;
            pred[y][x] = p;
          }
        }
        seedTotal += score(pred, gts[testRn][si]);
      }
      total += seedTotal / SEEDS;
      count++;
    }
    log(`  Adaptive floor: LOO avg=${(total/count).toFixed(3)}`);
  }

  // ═══ FIND OVERALL BEST & DECIDE ═══
  log('\n═══ SUMMARY ═══');
  log(`Baseline: ${baseResult.avg.toFixed(3)} (std=${baseResult.std.toFixed(2)})`);
  log(`Decision: only submit if we beat baseline by >= 0.5 consistently`);
  log(`Anti-overfit: std should not increase significantly`);

  // Determine best approach based on all results
  // (The logging above gives us all the data; the submission decision should be manual)

  if (DO_SUBMIT) {
    log('\n═══ Submitting R6 with baseline config (safest) ═══');
    const fullModel = buildModel(inits, gts, growthRounds, cf_v1, null);
    for (let si = 0; si < SEEDS; si++) {
      const p = predict(inits['R6'][si], fullModel, baseCfg, cf_v1);
      const res = await POST('/submit', { round_id: RDS['R6'], seed_index: si, prediction: p });
      log(`  Seed ${si}: ${res.data.status || JSON.stringify(res.data)}`);
      await sleep(600);
    }
    log('Submitted!');
  } else {
    log('\nRun with --submit to submit. Review results first!');
  }
}

main().catch(e => { console.error('Fatal:', e.message); process.exit(1); });
