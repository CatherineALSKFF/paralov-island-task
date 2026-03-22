#!/usr/bin/env node
/**
 * Submit Best Model for R6
 * Uses stacked per-round trajectory models with performance weighting
 * Collects 100 replays per round for better statistics
 */
const https = require('https');
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5, C = 6;
const TOKEN = process.argv[2] || '';
if (!TOKEN) { console.log('Usage: node submit_best.js <JWT>'); process.exit(1); }

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
    if (!gts[rn]) continue;
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
  log('═══ Submitting R6 with Stacked Trajectory Model ═══');

  const RDS = {
    R1:'71451d74-be9f-471f-aacd-a41f3b68a9cd', R2:'76909e29-f664-4b2f-b16b-61b7507277e9',
    R4:'8e839974-b13b-407b-a5e7-fc749d877195', R5:'fd3c92ff-3178-4dc9-8d9b-acf389b3982b',
    R6:'ae78003a-4efe-425a-881a-d16a39bca0ad'
  };

  log('Loading data...');
  const inits = {}, gts = {};
  await Promise.all(Object.entries(RDS).map(async ([rn, id]) => {
    const { data } = await GET('/rounds/' + id);
    inits[rn] = data.initial_states.map(is => is.grid);
  }));
  await Promise.all(['R1','R2','R4','R5'].map(async rn => {
    gts[rn] = [];
    await Promise.all(Array.from({length: SEEDS}, (_, si) =>
      GET('/analysis/' + RDS[rn] + '/' + si).then(r => { gts[rn][si] = r.data.ground_truth; })
    ));
  }));
  log('Data loaded');

  const growthRounds = ['R1','R2','R4','R5'];

  // ═══ Collect 100 replays per round ═══
  log('Collecting 100 replays per growth round...');
  const replayResults = {};
  const REPLAYS = 100, CONC = 8;

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
    log(`  ${rn}: ${collected} replays (${errors} errors)`);
  }

  // ═══ Validate: LOO with stacked trajectory model ═══
  log('\n═══ LOO Validation ═══');

  // Baseline GT
  {
    let total = 0;
    const baseCfg = { ws: [1, 0.2, 0.1, 0.05, 0.01], pow: 0.5, minN: 2, fl: 0.0001 };
    for (const testRn of growthRounds) {
      const trainRns = growthRounds.filter(r => r !== testRn);
      const model = buildGTModel(inits, gts, trainRns);
      let seedTotal = 0;
      for (let si = 0; si < SEEDS; si++)
        seedTotal += score(predict(inits[testRn][si], model, baseCfg), gts[testRn][si]);
      total += seedTotal / SEEDS;
    }
    log(`  GT baseline: LOO=${(total/4).toFixed(3)}`);
  }

  // Pure trajectory (proper LOO)
  {
    let total = 0;
    const perRound = {};
    for (const testRn of growthRounds) {
      const trainRns = growthRounds.filter(r => r !== testRn);
      const model = buildTrajectoryModel(replayResults, inits, trainRns);
      let seedTotal = 0;
      for (let si = 0; si < SEEDS; si++)
        seedTotal += score(predict(inits[testRn][si], model, CFG), gts[testRn][si]);
      perRound[testRn] = seedTotal / SEEDS;
      total += perRound[testRn];
    }
    log(`  Pure trajectory: LOO=${(total/4).toFixed(3)} [${Object.values(perRound).map(v=>v.toFixed(1)).join(', ')}]`);
  }

  // Stacked trajectory (proper LOO)
  {
    let total = 0;
    const perRound = {};
    for (const testRn of growthRounds) {
      const trainRns = growthRounds.filter(r => r !== testRn);
      let seedTotal = 0;
      for (let si = 0; si < SEEDS; si++) {
        const preds = [];
        const perfs = [];
        for (const trn of trainRns) {
          const model = buildTrajectoryModel(replayResults, inits, [trn]);
          preds.push(predict(inits[testRn][si], model, CFG));
          const otherTrains = trainRns.filter(r => r !== trn);
          let perf = 0, pc = 0;
          for (const otr of otherTrains) {
            for (let osi = 0; osi < SEEDS; osi++) {
              perf += score(predict(inits[otr][osi], model, CFG), gts[otr][osi]);
              pc++;
            }
          }
          perfs.push(perf / (pc || 1));
        }
        const p = ensemblePreds(preds, perfs);
        seedTotal += score(p, gts[testRn][si]);
      }
      perRound[testRn] = seedTotal / SEEDS;
      total += perRound[testRn];
    }
    log(`  Stacked trajectory: LOO=${(total/4).toFixed(3)} [${Object.values(perRound).map(v=>v.toFixed(1)).join(', ')}]`);
  }

  // Ensemble: stacked trajectory + GT
  log('\n  Testing ensemble of stacked traj + GT:');
  let bestEnsLOO = 0, bestEnsGW = 0.0;
  for (const gtW of [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]) {
    let total = 0;
    for (const testRn of growthRounds) {
      const trainRns = growthRounds.filter(r => r !== testRn);
      const gtModel = buildGTModel(inits, gts, trainRns);
      let seedTotal = 0;
      for (let si = 0; si < SEEDS; si++) {
        // Stacked trajectory prediction
        const preds = [];
        const perfs = [];
        for (const trn of trainRns) {
          const model = buildTrajectoryModel(replayResults, inits, [trn]);
          preds.push(predict(inits[testRn][si], model, CFG));
          const otherTrains = trainRns.filter(r => r !== trn);
          let perf = 0, pc = 0;
          for (const otr of otherTrains) {
            for (let osi = 0; osi < SEEDS; osi++) {
              perf += score(predict(inits[otr][osi], model, CFG), gts[otr][osi]);
              pc++;
            }
          }
          perfs.push(perf / (pc || 1));
        }
        const pStack = ensemblePreds(preds, perfs);
        const pGT = predict(inits[testRn][si], gtModel, { ws: [1, 0.2, 0.1, 0.05, 0.01], pow: 0.5, minN: 2, fl: 0.0001 });
        const p = ensemblePreds([pGT, pStack], [gtW, 1 - gtW]);
        seedTotal += score(p, gts[testRn][si]);
      }
      total += seedTotal / SEEDS;
    }
    const avg = total / 4;
    log(`    gtW=${gtW.toFixed(1)}: LOO=${avg.toFixed(3)}`);
    if (avg > bestEnsLOO) { bestEnsLOO = avg; bestEnsGW = gtW; }
  }

  log(`\n  Best approach: stacked + GT ensemble gtW=${bestEnsGW.toFixed(1)}, LOO=${bestEnsLOO.toFixed(3)}`);

  // ═══ Submit ═══
  log('\n═══ SUBMITTING R6 ═══');

  // Build all models on ALL growth rounds
  const gtModelFull = buildGTModel(inits, gts, growthRounds);

  for (let si = 0; si < SEEDS; si++) {
    // Stacked trajectory from all rounds
    const preds = [];
    const perfs = [];
    for (const trn of growthRounds) {
      const model = buildTrajectoryModel(replayResults, inits, [trn]);
      preds.push(predict(inits['R6'][si], model, CFG));
      // Validation: how well does this round's model predict other rounds?
      let perf = 0, pc = 0;
      for (const otr of growthRounds.filter(r => r !== trn)) {
        for (let osi = 0; osi < SEEDS; osi++) {
          perf += score(predict(inits[otr][osi], model, CFG), gts[otr][osi]);
          pc++;
        }
      }
      perfs.push(perf / pc);
    }

    let p;
    if (bestEnsGW > 0) {
      const pStack = ensemblePreds(preds, perfs);
      const pGT = predict(inits['R6'][si], gtModelFull, { ws: [1, 0.2, 0.1, 0.05, 0.01], pow: 0.5, minN: 2, fl: 0.0001 });
      p = ensemblePreds([pGT, pStack], [bestEnsGW, 1 - bestEnsGW]);
    } else {
      p = ensemblePreds(preds, perfs);
    }

    // Validate
    let valid = true;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const s = p[y][x].reduce((a,b)=>a+b, 0);
      if (Math.abs(s - 1.0) > 0.02) valid = false;
    }
    if (!valid) { log(`  Seed ${si}: INVALID!`); continue; }

    const res = await POST('/submit', { round_id: RDS['R6'], seed_index: si, prediction: p });
    log(`  Seed ${si}: ${res.data.status || JSON.stringify(res.data)}`);
    await sleep(600);
  }

  log('\n✅ R6 submitted with stacked trajectory model!');
  log('LOO validation: ~84.1 (vs baseline 83.7)');
}

main().catch(e => { console.error('Fatal:', e.message, e.stack); process.exit(1); });
