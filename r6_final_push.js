#!/usr/bin/env node
/**
 * R6 Final Push - Quick targeted tests:
 * 1. R5-heavy trajectory model (R5 most similar to R6)
 * 2. R1+R5 only trajectory model (best generalizers)
 * 3. Bayesian fusion of GT and trajectory at feature level
 * 4. Higher replay count for R5
 */
const https = require('https');
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, SEEDS = 5, C = 6;
const TOKEN = process.argv[2] || '';
if (!TOKEN) { console.log('Usage: node r6_final_push.js <JWT> [--submit]'); process.exit(1); }
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

function cf(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS=0,co=0,fN=0,sR2=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];if(nt===1||nt===2)nS++;if(nt===10)co=1;if(nt===4)fN++;}
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;if(g[ny][nx]===1||g[ny][nx]===2)sR2++;}
  const sa=Math.min(nS,5),sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3,fb=fN<=1?0:fN<=3?1:2;
  return[`D0_${t}_${sa}_${co}_${sb2}_${fb}`,`D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,`D2_${t}_${sa>0?1:0}_${co}`,`D3_${t}_${co}`,`D4_${t}`];
}

function buildTrajectoryModel(replays, inits, roundNames, alpha=0.05) {
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

// Build fused model: combine GT probability vectors + trajectory counts
function buildFusedModel(inits, gts, replays, roundNames, gtWeight=5.0) {
  const m = {};
  // First: accumulate trajectory counts
  for (const rn of roundNames) {
    if (replays[rn]) {
      for (const rep of replays[rn]) {
        const initGrid = inits[rn][rep.si];
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = cf(initGrid, y, x); if (!keys) continue;
          const fc = t2c(rep.finalGrid[y][x]);
          for (let ki = 0; ki < keys.length; ki++) {
            const k = keys[ki];
            if (!m[k]) m[k] = { n_traj: 0, counts: new Float64Array(C), n_gt: 0, gt_sum: new Float64Array(C) };
            m[k].n_traj++;
            m[k].counts[fc]++;
          }
        }
      }
    }
    // Then: accumulate GT probability vectors
    if (gts[rn]) {
      for (let si = 0; si < SEEDS; si++) {
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = cf(inits[rn][si], y, x); if (!keys) continue;
          const g = gts[rn][si][y][x];
          for (let ki = 0; ki < keys.length; ki++) {
            const k = keys[ki];
            if (!m[k]) m[k] = { n_traj: 0, counts: new Float64Array(C), n_gt: 0, gt_sum: new Float64Array(C) };
            m[k].n_gt++;
            for (let c = 0; c < C; c++) m[k].gt_sum[c] += g[c];
          }
        }
      }
    }
  }

  // Fuse: Dirichlet posterior = gtWeight * GT_avg + trajectory_counts + alpha
  const alpha = 0.05;
  for (const k of Object.keys(m)) {
    const gtAvg = m[k].n_gt > 0 ? Array.from(m[k].gt_sum).map(v => v / m[k].n_gt) : Array(C).fill(1/C);
    const posterior = [];
    let total = 0;
    for (let c = 0; c < C; c++) {
      posterior[c] = gtWeight * gtAvg[c] + m[k].counts[c] + alpha;
      total += posterior[c];
    }
    m[k].a = posterior.map(v => v / total);
    m[k].n = m[k].n_traj + m[k].n_gt; // for weighting in predict
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
  log('═══ R6 Final Push ═══');

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

  // Collect replays: 50 each for R1/R2/R4, 100 for R5 (most similar to R6)
  log('\nCollecting replays (R5 extra)...');
  const replays = {};
  const CONC = 8;
  for (const [rn, count] of [['R5', 100], ['R1', 50], ['R2', 50], ['R4', 50]]) {
    replays[rn] = [];
    let collected = 0, errors = 0;
    while (collected < count) {
      const batch = [];
      for (let i = 0; i < CONC && (collected + batch.length) < count; i++) {
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

  const growthRounds = ['R1','R2','R4','R5'];

  // ═══ TEST: Various trajectory model configs ═══
  log('\n═══ Testing models ═══');

  const tests = [
    { name: 'Traj all rounds', rounds: growthRounds },
    { name: 'Traj R1+R5 only', rounds: ['R1','R5'] },
    { name: 'Traj R5 only', rounds: ['R5'] },
    { name: 'Traj R1 only', rounds: ['R1'] },
    { name: 'Traj R1+R2+R5', rounds: ['R1','R2','R5'] },
    { name: 'Traj R1+R4+R5', rounds: ['R1','R4','R5'] },
  ];

  let bestLOO = 0, bestTest = null;

  for (const t of tests) {
    // Proper LOO: only test on rounds NOT in training
    const testRounds = growthRounds.filter(r => !t.rounds.includes(r));
    if (testRounds.length >= 2) {
      // Test on held-out rounds
      let total = 0, count = 0;
      for (const tr of testRounds) {
        const model = buildTrajectoryModel(replays, inits, t.rounds);
        for (let si = 0; si < SEEDS; si++)
          total += score(predict(inits[tr][si], model, CFG), gts[tr][si]);
        count += SEEDS;
      }
      log(`  ${t.name}: heldout avg=${(total/count).toFixed(2)} on ${testRounds.join(',')}`);
    }
    // LOO within training (only if >= 2 rounds)
    if (t.rounds.length >= 2) {
      let total = 0, count = 0;
      for (const tr of t.rounds) {
        const trainRns = t.rounds.filter(r => r !== tr);
        const model = buildTrajectoryModel(replays, inits, trainRns);
        for (let si = 0; si < SEEDS; si++)
          total += score(predict(inits[tr][si], model, CFG), gts[tr][si]);
        count += SEEDS;
      }
      const avg = total / count;
      log(`  ${t.name}: self-LOO=${avg.toFixed(3)}`);
      if (t.rounds.length >= 3 && avg > bestLOO) { bestLOO = avg; bestTest = t; }
    }
  }

  // Full 4-round LOO for reference
  {
    let total = 0;
    const perRound = {};
    for (const tr of growthRounds) {
      const model = buildTrajectoryModel(replays, inits, growthRounds.filter(r => r !== tr));
      let st = 0;
      for (let si = 0; si < SEEDS; si++)
        st += score(predict(inits[tr][si], model, CFG), gts[tr][si]);
      perRound[tr] = st / SEEDS;
      total += perRound[tr];
    }
    log(`  Full 4-round LOO: ${(total/4).toFixed(3)} [${Object.values(perRound).map(v=>v.toFixed(1)).join(', ')}]`);
  }

  // ═══ TEST: Fused GT + Trajectory model ═══
  log('\n═══ Testing fused models ═══');
  for (const gtW of [0, 1, 2, 5, 10, 20, 50]) {
    let total = 0;
    const perRound = {};
    for (const tr of growthRounds) {
      const model = buildFusedModel(inits, gts, replays, growthRounds.filter(r => r !== tr), gtW);
      let st = 0;
      for (let si = 0; si < SEEDS; si++)
        st += score(predict(inits[tr][si], model, CFG), gts[tr][si]);
      perRound[tr] = st / SEEDS;
      total += perRound[tr];
    }
    log(`  Fused gtW=${gtW}: LOO=${(total/4).toFixed(3)} [${Object.values(perRound).map(v=>v.toFixed(1)).join(', ')}]`);
  }

  // ═══ TEST: Dirichlet alpha for trajectory ═══
  log('\n═══ Dirichlet alpha ═══');
  for (const alpha of [0.01, 0.02, 0.05, 0.1, 0.2]) {
    let total = 0;
    for (const tr of growthRounds) {
      const model = buildTrajectoryModel(replays, inits, growthRounds.filter(r => r !== tr), alpha);
      for (let si = 0; si < SEEDS; si++)
        total += score(predict(inits[tr][si], model, CFG), gts[tr][si]);
    }
    log(`  alpha=${alpha}: LOO=${(total/20).toFixed(3)}`);
  }

  // ═══ SUBMIT best if improvement ═══
  if (DO_SUBMIT) {
    log('\n═══ Submitting R6 ═══');
    // Use full trajectory model on all growth rounds
    const model = buildTrajectoryModel(replays, inits, growthRounds);
    for (let si = 0; si < SEEDS; si++) {
      const p = predict(inits['R6'][si], model, CFG);
      const res = await POST('/submit', { round_id: RDS['R6'], seed_index: si, prediction: p });
      log(`  Seed ${si}: ${res.data.status || JSON.stringify(res.data)}`);
      await sleep(600);
    }
    log('✅ Submitted!');
  }
}

main().catch(e => { console.error('Fatal:', e.message, e.stack); process.exit(1); });
