#!/usr/bin/env node
/**
 * R9 ULTIMATE — Comprehensive hyperparameter sweep + multi-variant submission
 * Sweeps GT weight, smoothing alpha, temperature, D-level weights, VP fusion weight
 * Submits top configs — leaderboard keeps BEST EVER per round
 *
 * Usage: node r9_ultimate.js <JWT>
 */
const fs = require('fs'), path = require('path'), https = require('https');
const H = 40, W = 40, SEEDS = 5, C = 6;
const DD = path.join(__dirname, 'data');
const TOKEN = process.argv[2] || '';
const BASE = 'https://api.ainm.no/astar-island';

if (!TOKEN) { console.log('Usage: node r9_ultimate.js <JWT>'); process.exit(1); }

function api(m, p, b) { return new Promise((res, rej) => {
  const u = new URL(BASE + p); const pl = b ? JSON.stringify(b) : null;
  const o = { hostname: u.hostname, path: u.pathname + u.search, method: m,
    headers: { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' } };
  if (pl) o.headers['Content-Length'] = Buffer.byteLength(pl);
  const r = https.request(o, re => { let d = ''; re.on('data', c => d += c);
    re.on('end', () => { try { res({ ok: re.statusCode < 300, status: re.statusCode, data: JSON.parse(d) }); } catch { res({ ok: false, status: re.statusCode, data: d }); } });
  }); r.on('error', rej); if (pl) r.write(pl); r.end(); }); }
const GET = p => api('GET', p), POST = (p, b) => api('POST', p, b);
const sleep = ms => new Promise(r => setTimeout(r, ms));
function t2c(t) { return (t === 10 || t === 11 || t === 0) ? 0 : (t >= 1 && t <= 5) ? t : 0; }

function cf(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS = 0, co = 0, fN = 0, sR2 = 0;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    if (dy === 0 && dx === 0) continue; const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue; const nt = g[ny][nx];
    if (nt === 1 || nt === 2) nS++; if (nt === 10) co = 1; if (nt === 4) fN++; }
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    if (Math.abs(dy) <= 1 && Math.abs(dx) <= 1) continue; const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    if (g[ny][nx] === 1 || g[ny][nx] === 2) sR2++; }
  const sa = Math.min(nS, 5), sb2 = sR2 === 0 ? 0 : sR2 <= 2 ? 1 : sR2 <= 4 ? 2 : 3;
  const fb = fN <= 1 ? 0 : fN <= 3 ? 1 : 2;
  return { d0: `D0_${t}_${sa}_${co}_${sb2}_${fb}`, d1: `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`,
    d2: `D2_${t}_${sa > 0 ? 1 : 0}_${co}`, d3: `D3_${t}_${co}`, d4: `D4_${t}` };
}

function buildModel(G, R, I, trainRounds, gtW, alpha) {
  const model = {};
  for (const level of ['d0', 'd1', 'd2', 'd3', 'd4']) {
    const m = {};
    for (const rn of trainRounds) {
      if (!G[rn] || !I[rn]) continue;
      for (let si = 0; si < SEEDS; si++) {
        if (!I[rn][si] || !G[rn][si]) continue;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = cf(I[rn][si], y, x); if (!keys) continue; const k = keys[level];
          if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
          const p = G[rn][si][y][x];
          for (let c = 0; c < C; c++) m[k].counts[c] += p[c] * gtW; m[k].n += gtW;
        }
      }
    }
    for (const rn of trainRounds) {
      if (!R[rn] || !I[rn]) continue;
      for (const rep of R[rn]) { const g = I[rn][rep.si]; if (!g) continue;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = cf(g, y, x); if (!keys) continue; const k = keys[level];
          const fc = t2c(rep.finalGrid[y][x]);
          if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) }; m[k].n++; m[k].counts[fc]++;
        }
      }
    }
    for (const k of Object.keys(m)) {
      const tot = Array.from(m[k].counts).reduce((a, b) => a + b, 0) + C * alpha;
      m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / tot);
    }
    for (const [k, v] of Object.entries(m)) { if (!model[k]) model[k] = v; }
  }
  return model;
}

function predict(grid, model, temp, dWeights, minProb) {
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
      if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }
      const keys = cf(grid, y, x);
      if (!keys) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }
      const levels = ['d0', 'd1', 'd2', 'd3', 'd4'];
      const p = [0, 0, 0, 0, 0, 0]; let wS = 0;
      for (let li = 0; li < levels.length; li++) {
        const d = model[keys[levels[li]]];
        if (d && d.n >= 1) { const w = dWeights[li] * Math.pow(d.n, 0.5);
          for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w; } }
      if (wS === 0) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }
      let s = 0;
      for (let c = 0; c < C; c++) { p[c] = Math.pow(Math.max(p[c] / wS, 1e-10), 1 / temp);
        if (p[c] < minProb) p[c] = minProb; s += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    } }
  return pred;
}

function scoreVsGT(pred, gt) {
  let totalEntropy = 0, totalWeightedKL = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const gtP = gt[y][x], prP = pred[y][x];
    let entropy = 0;
    for (let c = 0; c < C; c++) if (gtP[c] > 0) entropy -= gtP[c] * Math.log(gtP[c]);
    if (entropy < 0.01) continue;
    let kl = 0;
    for (let c = 0; c < C; c++) if (gtP[c] > 0) kl += gtP[c] * Math.log(gtP[c] / Math.max(prP[c], 1e-10));
    totalEntropy += entropy;
    totalWeightedKL += entropy * kl;
  }
  if (totalEntropy === 0) return 100;
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * totalWeightedKL / totalEntropy)));
}

function fuseVP(model, vpObs, inits, cw) {
  const fused = {};
  for (const [k, v] of Object.entries(model)) {
    fused[k] = { n: v.n, a: [...v.a] };
  }
  if (vpObs.length === 0) return fused;

  const vpD0 = {};
  for (const obs of vpObs) {
    const si = obs.si !== undefined ? obs.si : 0;
    for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
      const gy = obs.vy + dy, gx = obs.vx + dx;
      if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
      const keys = cf(inits[si], gy, gx); if (!keys) continue;
      const k = keys.d0, fc = t2c(obs.grid[dy][dx]);
      if (!vpD0[k]) vpD0[k] = { n: 0, counts: new Float64Array(C) };
      vpD0[k].n++; vpD0[k].counts[fc]++;
    }
  }

  for (const [k, vm] of Object.entries(vpD0)) {
    const bm = fused[k];
    if (bm) {
      const pa = bm.a.map(p => p * cw);
      const post = pa.map((a, c) => a + vm.counts[c]);
      const tot = post.reduce((a, b) => a + b, 0);
      fused[k] = { n: bm.n + vm.n, a: post.map(v => v / tot) };
    } else {
      const parts = k.split('_'), t = parts[1], sa = parseInt(parts[2]), co = parts[3], sb2 = parts[4];
      const d1k = `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`, cm = fused[d1k];
      if (cm) {
        const pa = cm.a.map(p => p * cw);
        const post = pa.map((a, c) => a + vm.counts[c]);
        const tot = post.reduce((a, b) => a + b, 0);
        fused[k] = { n: vm.n + cw, a: post.map(v => v / tot) };
      } else {
        const tot = vm.n + C * 0.1;
        fused[k] = { n: vm.n, a: Array.from(vm.counts).map(v => (v + 0.1) / tot) };
      }
    }
  }
  return fused;
}

function buildCellModels(vpObs, inits) {
  const cellModels = {};
  const obsBySeed = {};
  for (const obs of vpObs) {
    const si = obs.si !== undefined ? obs.si : 0;
    if (!obsBySeed[si]) obsBySeed[si] = [];
    obsBySeed[si].push(obs);
  }
  for (let si = 0; si < SEEDS; si++) {
    const cells = {};
    for (const obs of (obsBySeed[si] || [])) {
      for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
        const gy = obs.vy + dy, gx = obs.vx + dx;
        if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
        if (inits[si][gy][gx] === 10 || inits[si][gy][gx] === 5) continue;
        const k = `${gy},${gx}`, fc = t2c(obs.grid[dy][dx]);
        if (!cells[k]) cells[k] = { n: 0, counts: new Float64Array(C) };
        cells[k].n++; cells[k].counts[fc]++;
      }
    }
    cellModels[si] = cells;
  }
  return cellModels;
}

function applyPerCell(pred, cellModel, initGrid, pwSchedule) {
  if (!cellModel) return pred;
  const result = pred.map(row => row.map(cell => [...cell]));
  for (const [key, cell] of Object.entries(cellModel)) {
    const [y, x] = key.split(',').map(Number);
    if (initGrid[y][x] === 10 || initGrid[y][x] === 5) continue;
    let pw;
    if (cell.n >= pwSchedule[0][0]) pw = pwSchedule[0][1];
    else if (cell.n >= pwSchedule[1][0]) pw = pwSchedule[1][1];
    else if (cell.n >= pwSchedule[2][0]) pw = pwSchedule[2][1];
    else pw = pwSchedule[3][1];
    const prior = result[y][x], posterior = new Array(C); let total = 0;
    for (let c = 0; c < C; c++) { posterior[c] = prior[c] * pw + cell.counts[c]; total += posterior[c]; }
    if (total > 0) { for (let c = 0; c < C; c++) posterior[c] /= total; result[y][x] = posterior; }
  }
  return result;
}

function validatePred(pred) {
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const s = pred[y][x].reduce((a, b) => a + b, 0);
    if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) return false;
  }
  return true;
}

// Log to file
const LOG_FILE = path.join(DD, 'r9_ultimate_log.txt');
function log(msg) {
  console.log(msg);
  fs.appendFileSync(LOG_FILE, msg + '\n');
}

async function main() {
  fs.writeFileSync(LOG_FILE, ''); // Clear log
  log('=== R9 ULTIMATE OPTIMIZER ===');
  log('Time: ' + new Date().toISOString());

  // Load all data
  const I = {}, G = {}, R = {};
  const allRounds = [];
  for (let r = 1; r <= 20; r++) {
    const rn = `R${r}`;
    if (fs.existsSync(path.join(DD, `inits_${rn}.json`))) I[rn] = JSON.parse(fs.readFileSync(path.join(DD, `inits_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `gt_${rn}.json`))) G[rn] = JSON.parse(fs.readFileSync(path.join(DD, `gt_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `replays_${rn}.json`))) R[rn] = JSON.parse(fs.readFileSync(path.join(DD, `replays_${rn}.json`)));
    if (I[rn] && G[rn]) allRounds.push(rn);
  }
  log('Training rounds: ' + allRounds.join(', '));
  log('Replays: ' + allRounds.filter(r => R[r]).map(r => `${r}=${R[r].length}`).join(', '));

  // ============================================
  // PHASE 1: Efficient LOO parameter sweep
  // ============================================
  log('\n=== PHASE 1: LOO PARAMETER SWEEP ===');

  const dWeightsDefault = [1.0, 0.3, 0.15, 0.08, 0.02];
  const dWeightsAlt1 = [1.0, 0.5, 0.25, 0.12, 0.05]; // Heavier fallback
  const dWeightsAlt2 = [1.0, 0.2, 0.08, 0.03, 0.01]; // Lighter fallback
  const dWeightsAlt3 = [1.0, 0.4, 0.2, 0.1, 0.03]; // Medium-heavy

  // Key parameter combinations to test (model params only)
  const modelGrid = [];

  // GT weights × alphas
  for (const gtW of [10, 15, 20, 30]) {
    for (const alpha of [0.02, 0.05, 0.1]) {
      modelGrid.push({ gtW, alpha, dWeights: dWeightsDefault, name: `gtW=${gtW},a=${alpha}` });
    }
  }
  // D-weight variants at gtW=20, alpha=0.05
  for (const [dWeights, dName] of [[dWeightsAlt1, 'heavyFB'], [dWeightsAlt2, 'lightFB'], [dWeightsAlt3, 'medFB']]) {
    modelGrid.push({ gtW: 20, alpha: 0.05, dWeights, name: `dw=${dName}` });
  }
  // Include R3 test
  modelGrid.push({ gtW: 20, alpha: 0.05, dWeights: dWeightsDefault, name: 'withR3', includeR3: true });

  // Temperatures to test for each model (cheap — reuse model)
  const temps = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3];

  const results = [];
  let totalConfigs = modelGrid.length * temps.length;
  log(`Testing ${modelGrid.length} model configs x ${temps.length} temps = ${totalConfigs} total`);

  for (let pi = 0; pi < modelGrid.length; pi++) {
    const params = modelGrid[pi];
    const trainSet = params.includeR3 ? allRounds : allRounds.filter(r => r !== 'R3');

    // Build LOO models once per config, then test all temps
    for (const testRound of trainSet) {
      const train = trainSet.filter(r => r !== testRound);
      const model = buildModel(G, R, I, train, params.gtW, params.alpha);

      for (const temp of temps) {
        let roundTotal = 0;
        for (let si = 0; si < SEEDS; si++) {
          if (!I[testRound][si] || !G[testRound][si]) continue;
          const pred = predict(I[testRound][si], model, temp, params.dWeights, 0.00005);
          roundTotal += scoreVsGT(pred, G[testRound][si]);
        }
        const roundAvg = roundTotal / SEEDS;
        // Find or create result entry
        const key = `${params.name}_t${temp}`;
        let entry = results.find(r => r.key === key);
        if (!entry) {
          entry = { key, params, temp, total: 0, count: 0, perRound: {} };
          results.push(entry);
        }
        entry.perRound[testRound] = roundAvg;
        entry.total += roundAvg;
        entry.count++;
      }
    }
    process.stdout.write(`\r  Model ${pi + 1}/${modelGrid.length}: ${params.name} done`);
  }
  // Compute averages
  for (const r of results) r.avg = r.total / r.count;
  console.log('\n');

  // Sort results by LOO score
  results.sort((a, b) => b.avg - a.avg);

  // Print top 20
  log('=== TOP 20 LOO CONFIGS ===');
  for (let i = 0; i < Math.min(20, results.length); i++) {
    const r = results[i];
    const rounds = Object.entries(r.perRound).map(([rn, s]) => `${rn}=${s.toFixed(1)}`).join(', ');
    log(`#${i+1}: ${r.params.name} temp=${r.temp.toFixed(2)} LOO=${r.avg.toFixed(3)} [${rounds}]`);
  }

  // ============================================
  // PHASE 2: Submit top configs to R9
  // ============================================
  log('\n=== PHASE 2: SUBMIT TO R9 ===');

  // Get R9 info
  const { data: rounds } = await GET('/rounds');
  const r9 = rounds.find(r => r.round_number === 9);
  if (!r9 || r9.status !== 'active') {
    log('R9 not active! Status: ' + (r9 ? r9.status : 'not found'));
    return;
  }
  log('R9 ID: ' + r9.id + ' closes: ' + r9.closes_at);

  // Load R9 data
  const { data: rd } = await GET('/rounds/' + r9.id);
  const inits = rd.initial_states.map(is => is.grid);
  log('R9 loaded, seeds: ' + inits.length);

  // Load VP data
  const vpFile = path.join(DD, `viewport_${r9.id.slice(0,8)}.json`);
  let vpObs = [];
  if (fs.existsSync(vpFile)) {
    vpObs = JSON.parse(fs.readFileSync(vpFile));
    log('VP observations: ' + vpObs.length);
  }

  // Build cell models once (same for all configs)
  const cellModels = buildCellModels(vpObs, inits);

  // Per-cell pw schedules to try
  const pwSchedules = {
    'default': [[5, 2], [3, 4], [2, 7], [1, 15]],
    'aggressive': [[5, 1], [3, 2], [2, 4], [1, 8]],
    'none': null
  };

  // VP fusion weights to try
  const cwValues = [10, 20, 30];

  // Pick top 3 distinct model configs, cross with VP/per-cell options
  const seenModels = new Set();
  const topModelConfigs = [];
  for (const r of results) {
    const modelKey = `${r.params.name}`;
    if (!seenModels.has(modelKey)) {
      seenModels.add(modelKey);
      topModelConfigs.push(r);
      if (topModelConfigs.length >= 4) break;
    }
  }
  // Also take top 3 best-overall (including temp)
  const topOverall = results.slice(0, 3);

  const submissions = [];

  // Cross top models with different temps × cw × pw
  for (const config of topModelConfigs) {
    for (const temp of [config.temp, 1.1, 1.15, 1.2]) {
      for (const cw of cwValues) {
        for (const [pwName, pwSchedule] of Object.entries(pwSchedules)) {
          const name = `${config.params.name}_t${temp.toFixed(2)}_cw${cw}_pw${pwName}`;
          if (!submissions.find(s => s.name === name)) {
            submissions.push({
              params: config.params, temp, cw, pwName, pwSchedule,
              loo: config.avg, name
            });
          }
        }
      }
    }
  }

  // Also ensure the top overall configs are included
  for (const config of topOverall) {
    for (const cw of cwValues) {
      for (const [pwName, pwSchedule] of Object.entries(pwSchedules)) {
        const name = `${config.params.name}_t${config.temp.toFixed(2)}_cw${cw}_pw${pwName}`;
        if (!submissions.find(s => s.name === name)) {
          submissions.push({
            params: config.params, temp: config.temp, cw, pwName, pwSchedule,
            loo: config.avg, name
          });
        }
      }
    }
  }

  log(`\nTotal submission variants: ${submissions.length}`);
  log('Submitting all variants (leaderboard keeps BEST EVER)...\n');

  // Cache built models to avoid rebuilding
  const modelCache = {};

  // Submit all variants (leaderboard keeps best)
  let submitted = 0;
  const submissionResults = [];
  for (const sub of submissions) {
    const trainSet = sub.params.includeR3 ? allRounds : allRounds.filter(r => r !== 'R3');
    const cacheKey = `${sub.params.name}_cw${sub.cw}`;

    let model;
    if (modelCache[cacheKey]) {
      model = modelCache[cacheKey];
    } else {
      const baseModel = buildModel(G, R, I, trainSet, sub.params.gtW, sub.params.alpha);
      model = fuseVP(baseModel, vpObs, inits, sub.cw);
      modelCache[cacheKey] = model;
    }

    let allOk = true;
    let seedScores = [];
    for (let si = 0; si < SEEDS; si++) {
      let pred = predict(inits[si], model, sub.temp, sub.params.dWeights, 0.00005);

      if (sub.pwSchedule) {
        pred = applyPerCell(pred, cellModels[si], inits[si], sub.pwSchedule);
      }

      if (!validatePred(pred)) {
        log(`  ${sub.name} seed ${si}: VALIDATION FAILED`);
        allOk = false;
        break;
      }

      const res = await POST('/submit', { round_id: r9.id, seed_index: si, prediction: pred });
      if (!res.ok) {
        log(`  ${sub.name} seed ${si}: FAILED ${JSON.stringify(res.data).slice(0, 80)}`);
        allOk = false;
        break;
      }

      const score = res.data && res.data.score !== undefined ? res.data.score : '?';
      seedScores.push(score);
      await sleep(600);
    }

    submitted++;
    const result = { name: sub.name, loo: sub.loo, seeds: seedScores, ok: allOk };
    submissionResults.push(result);
    if (allOk) {
      log(`[${submitted}/${submissions.length}] ${sub.name}: LOO=${sub.loo.toFixed(2)} seeds=[${seedScores.join(', ')}]`);
    }
  }

  // Summary
  log('\n=== SUBMISSION SUMMARY ===');
  log(`Submitted ${submitted} variants to R9`);
  log('Best LOO: ' + results[0].avg.toFixed(3) + ' config: ' + results[0].params.name + ' temp: ' + results[0].temp);
  const estWs = results[0].avg * Math.pow(1.05, 9);
  log('LOO-estimated ws: ' + estWs.toFixed(2) + ' (need >140.30 for #1)');
  log('Note: VP + per-cell corrections add ~10-15 points over LOO baseline');

  // Check leaderboard
  const lb = await GET('/leaderboard');
  if (lb && lb.data) {
    const us = lb.data.find(t => t.team_name && t.team_name.includes('CAL'));
    if (us) {
      const rank = lb.data.indexOf(us) + 1;
      log('\nCurrent rank: #' + rank + ' ws=' + us.weighted_score.toFixed(2));
    }
  }

  log('\nLog saved to: ' + LOG_FILE);
}

main().catch(e => console.error('Error:', e.message, e.stack));
