#!/usr/bin/env node
/**
 * R9 NAIVE BAYES — Independent feature model
 * Instead of D0 joint features, treat each feature independently
 * P(class | features) ∝ P(class) × ∏ P(feature_i | class)
 * This avoids data sparsity in high-dimensional feature space
 *
 * Also tries: round-weighted model, different n-powers, calibrated probabilities
 *
 * Usage: node r9_naivebayes.js <JWT>
 */
const fs = require('fs'), path = require('path'), https = require('https');
const H = 40, W = 40, SEEDS = 5, C = 6;
const DD = path.join(__dirname, 'data');
const TOKEN = process.argv[2] || '';
const BASE = 'https://api.ainm.no/astar-island';

if (!TOKEN) { console.log('Usage: node r9_naivebayes.js <JWT>'); process.exit(1); }

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

function getFeatures(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS = 0, nP = 0, nR = 0, nF = 0, nM = 0, co = 0;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    if (dy === 0 && dx === 0) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    const nt = g[ny][nx];
    if (nt === 1) nS++;
    if (nt === 2) nP++;
    if (nt === 3) nR++;
    if (nt === 4) nF++;
    if (nt === 5) nM++;
    if (nt === 10) co = 1;
  }
  let sR2 = 0;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    if (Math.abs(dy) <= 1 && Math.abs(dx) <= 1) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    if (g[ny][nx] === 1 || g[ny][nx] === 2) sR2++;
  }

  return {
    t: t,                                    // terrain type
    sa: Math.min(nS + nP, 5),               // settlement+port count r1
    co: co,                                   // coastal
    sb2: sR2 === 0 ? 0 : sR2 <= 2 ? 1 : sR2 <= 4 ? 2 : 3,  // settlements r2
    fb: nF <= 1 ? 0 : nF <= 3 ? 1 : 2,     // forest count
    rb: nR === 0 ? 0 : 1,                    // ruin nearby
    mb: nM === 0 ? 0 : nM <= 2 ? 1 : 2,    // mountain nearby
    pb: nP === 0 ? 0 : 1,                    // port nearby
  };
}

// Original cf for fallback
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

async function main() {
  console.log('=== R9 NAIVE BAYES + ALTERNATIVE MODELS ===');
  console.log('Time:', new Date().toISOString());

  const I = {}, G = {}, R = {};
  const allRounds = [];
  for (let r = 1; r <= 20; r++) {
    const rn = `R${r}`;
    if (fs.existsSync(path.join(DD, `inits_${rn}.json`))) I[rn] = JSON.parse(fs.readFileSync(path.join(DD, `inits_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `gt_${rn}.json`))) G[rn] = JSON.parse(fs.readFileSync(path.join(DD, `gt_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `replays_${rn}.json`))) R[rn] = JSON.parse(fs.readFileSync(path.join(DD, `replays_${rn}.json`)));
    if (I[rn] && G[rn]) allRounds.push(rn);
  }
  const trainRounds = allRounds.filter(r => r !== 'R3');
  console.log('Training:', trainRounds.join(', '));

  // ========================================
  // Model 1: Naive Bayes
  // ========================================
  console.log('\n=== NAIVE BAYES LOO ===');

  // Build NB feature distributions
  function buildNB(trainSet) {
    const featureNames = ['t', 'sa', 'co', 'sb2', 'fb', 'rb', 'mb', 'pb'];
    const model = {};

    for (const fname of featureNames) {
      model[fname] = {}; // model[feature_name][feature_value] = { counts: Float64Array(C), n: number }
    }

    // Also build a prior
    model._prior = { counts: new Float64Array(C), n: 0 };

    for (const rn of trainSet) {
      if (!G[rn] || !I[rn]) continue;
      // GT data
      for (let si = 0; si < SEEDS; si++) {
        if (!I[rn][si] || !G[rn][si]) continue;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const feats = getFeatures(I[rn][si], y, x);
          if (!feats) continue;
          const gt = G[rn][si][y][x];
          // Update prior
          for (let c = 0; c < C; c++) model._prior.counts[c] += gt[c] * 20;
          model._prior.n += 20;
          // Update each feature
          for (const fname of featureNames) {
            const fval = feats[fname];
            if (!model[fname][fval]) model[fname][fval] = { counts: new Float64Array(C), n: 0 };
            for (let c = 0; c < C; c++) model[fname][fval].counts[c] += gt[c] * 20;
            model[fname][fval].n += 20;
          }
        }
      }
      // Replay data
      if (R[rn] && I[rn]) {
        for (const rep of R[rn]) {
          const g = I[rn][rep.si]; if (!g) continue;
          for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const feats = getFeatures(g, y, x);
            if (!feats) continue;
            const fc = t2c(rep.finalGrid[y][x]);
            model._prior.counts[fc]++;
            model._prior.n++;
            for (const fname of featureNames) {
              const fval = feats[fname];
              if (!model[fname][fval]) model[fname][fval] = { counts: new Float64Array(C), n: 0 };
              model[fname][fval].counts[fc]++;
              model[fname][fval].n++;
            }
          }
        }
      }
    }

    // Compute probabilities
    const alpha = 0.1;
    for (const fname of featureNames) {
      for (const fval of Object.keys(model[fname])) {
        const m = model[fname][fval];
        const tot = Array.from(m.counts).reduce((a, b) => a + b, 0) + C * alpha;
        m.prob = Array.from(m.counts).map(v => (v + alpha) / tot);
      }
    }
    const ptot = Array.from(model._prior.counts).reduce((a, b) => a + b, 0) + C * alpha;
    model._prior.prob = Array.from(model._prior.counts).map(v => (v + alpha) / ptot);

    return model;
  }

  function predictNB(grid, model, temp) {
    const featureNames = ['t', 'sa', 'co', 'sb2', 'fb', 'rb', 'mb', 'pb'];
    const pred = [];
    for (let y = 0; y < H; y++) { pred[y] = [];
      for (let x = 0; x < W; x++) {
        const t = grid[y][x];
        if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
        if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }
        const feats = getFeatures(grid, y, x);
        if (!feats) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }

        // NB: P(c | features) ∝ P(c) × ∏ P(feature_i | c) / P(feature_i)
        // Since we condition on features, we just need: P(c | feature_i) for each feature
        // Combine: log P(c | features) ∝ log P(c) + Σ log(P(c | feature_i) / P(c))
        // = log P(c) + Σ (log P(c | feature_i) - log P(c))
        // = (1 - N_features) log P(c) + Σ log P(c | feature_i)

        const p = new Array(C);
        for (let c = 0; c < C; c++) {
          let logP = Math.log(Math.max(model._prior.prob[c], 1e-15)) * (1 - featureNames.length);
          for (const fname of featureNames) {
            const fval = feats[fname];
            const m = model[fname][fval];
            if (m) {
              logP += Math.log(Math.max(m.prob[c], 1e-15));
            } else {
              logP += Math.log(Math.max(model._prior.prob[c], 1e-15));
            }
          }
          p[c] = Math.exp(logP / temp);
        }

        // Normalize
        let s = 0;
        for (let c = 0; c < C; c++) { if (p[c] < 0.00005) p[c] = 0.00005; s += p[c]; }
        for (let c = 0; c < C; c++) p[c] /= s;
        pred[y][x] = p;
      }
    }
    return pred;
  }

  // LOO for NB
  for (const temp of [1.0, 1.15, 1.3, 1.5]) {
    let total = 0, count = 0;
    for (const testRound of trainRounds) {
      const train = trainRounds.filter(r => r !== testRound);
      const model = buildNB(train);
      let roundTotal = 0;
      for (let si = 0; si < SEEDS; si++) {
        if (!I[testRound][si] || !G[testRound][si]) continue;
        roundTotal += scoreVsGT(predictNB(I[testRound][si], model, temp), G[testRound][si]);
      }
      total += roundTotal / SEEDS;
      count++;
    }
    console.log(`NB temp=${temp}: LOO=${(total/count).toFixed(3)}`);
  }

  // ========================================
  // Model 2: Round-weighted D0 model (weight recent rounds more)
  // ========================================
  console.log('\n=== ROUND-WEIGHTED MODEL LOO ===');

  function buildWeightedModel(G, R, I, trainSet, roundWeights) {
    const model = {};
    for (const level of ['d0', 'd1', 'd2', 'd3', 'd4']) {
      const m = {};
      for (const rn of trainSet) {
        if (!G[rn] || !I[rn]) continue;
        const rw = roundWeights[rn] || 1;
        const gtW = 20 * rw;
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
      for (const rn of trainSet) {
        if (!R[rn] || !I[rn]) continue;
        const rw = roundWeights[rn] || 1;
        for (const rep of R[rn]) { const g = I[rn][rep.si]; if (!g) continue;
          for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const keys = cf(g, y, x); if (!keys) continue; const k = keys[level];
            const fc = t2c(rep.finalGrid[y][x]);
            if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) }; m[k].n += rw; m[k].counts[fc] += rw;
          }
        }
      }
      for (const k of Object.keys(m)) {
        const tot = Array.from(m[k].counts).reduce((a, b) => a + b, 0) + C * 0.05;
        m[k].a = Array.from(m[k].counts).map(v => (v + 0.05) / tot);
      }
      for (const [k, v] of Object.entries(m)) { if (!model[k]) model[k] = v; }
    }
    return model;
  }

  function predictD(grid, model, temp) {
    const levels = ['d0', 'd1', 'd2', 'd3', 'd4'], ws = [1.0, 0.3, 0.15, 0.08, 0.02];
    const pred = [];
    for (let y = 0; y < H; y++) { pred[y] = [];
      for (let x = 0; x < W; x++) {
        const t = grid[y][x];
        if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
        if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }
        const keys = cf(grid, y, x);
        if (!keys) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }
        const p = [0, 0, 0, 0, 0, 0]; let wS = 0;
        for (let li = 0; li < levels.length; li++) {
          const d = model[keys[levels[li]]];
          if (d && d.n >= 1) { const w = ws[li] * Math.pow(d.n, 0.5);
            for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w; } }
        if (wS === 0) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }
        let s = 0;
        for (let c = 0; c < C; c++) { p[c] = Math.pow(Math.max(p[c] / wS, 1e-10), 1 / temp);
          if (p[c] < 0.00005) p[c] = 0.00005; s += p[c]; }
        for (let c = 0; c < C; c++) p[c] /= s;
        pred[y][x] = p;
      } }
    return pred;
  }

  // Test different round weighting schemes
  const weightSchemes = [
    { name: 'uniform', weights: {} },
    { name: 'recency', weights: { R1: 0.5, R2: 0.6, R4: 0.7, R5: 0.8, R6: 0.9, R7: 1.0, R8: 1.2 } },
    { name: 'heavy_recent', weights: { R1: 0.3, R2: 0.3, R4: 0.5, R5: 0.7, R6: 1.0, R7: 1.5, R8: 2.0 } },
  ];

  for (const scheme of weightSchemes) {
    let total = 0, count = 0;
    for (const testRound of trainRounds) {
      const train = trainRounds.filter(r => r !== testRound);
      const model = buildWeightedModel(G, R, I, train, scheme.weights);
      let roundTotal = 0;
      for (let si = 0; si < SEEDS; si++) {
        if (!I[testRound][si] || !G[testRound][si]) continue;
        roundTotal += scoreVsGT(predictD(I[testRound][si], model, 1.15), G[testRound][si]);
      }
      total += roundTotal / SEEDS;
      count++;
    }
    console.log(`${scheme.name}: LOO=${(total/count).toFixed(3)}`);
  }

  // ========================================
  // Model 3: NB + D0 ensemble (blend Naive Bayes with D0 model)
  // ========================================
  console.log('\n=== NB + D0 ENSEMBLE LOO ===');
  for (const nbWeight of [0.2, 0.3, 0.5]) {
    let total = 0, count = 0;
    for (const testRound of trainRounds) {
      const train = trainRounds.filter(r => r !== testRound);
      const nbModel = buildNB(train);
      const d0Model = buildWeightedModel(G, R, I, train, {});
      let roundTotal = 0;
      for (let si = 0; si < SEEDS; si++) {
        if (!I[testRound][si] || !G[testRound][si]) continue;
        const nbPred = predictNB(I[testRound][si], nbModel, 1.15);
        const d0Pred = predictD(I[testRound][si], d0Model, 1.15);
        // Blend
        const blended = [];
        for (let y = 0; y < H; y++) { blended[y] = [];
          for (let x = 0; x < W; x++) {
            blended[y][x] = new Array(C);
            let s = 0;
            for (let c = 0; c < C; c++) {
              blended[y][x][c] = nbWeight * nbPred[y][x][c] + (1 - nbWeight) * d0Pred[y][x][c];
              s += blended[y][x][c];
            }
            for (let c = 0; c < C; c++) blended[y][x][c] /= s;
          }
        }
        roundTotal += scoreVsGT(blended, G[testRound][si]);
      }
      total += roundTotal / SEEDS;
      count++;
    }
    console.log(`NB(${nbWeight})+D0(${1-nbWeight}): LOO=${(total/count).toFixed(3)}`);
  }

  // ========================================
  // Submit best models to R9
  // ========================================
  console.log('\n=== SUBMITTING TO R9 ===');

  const { data: rounds } = await GET('/rounds');
  const r9 = rounds.find(r => r.round_number === 9);
  if (!r9 || r9.status !== 'active') { console.log('R9 not active!'); return; }
  const { data: rd } = await GET('/rounds/' + r9.id);
  const inits = rd.initial_states.map(is => is.grid);

  const vpFile = path.join(DD, `viewport_${r9.id.slice(0,8)}.json`);
  const vpObs = fs.existsSync(vpFile) ? JSON.parse(fs.readFileSync(vpFile)) : [];

  // Submit NB predictions with VP per-cell
  console.log('\nSubmitting NB model:');
  const nbModel = buildNB(trainRounds);
  for (const temp of [1.0, 1.15, 1.5]) {
    console.log(`  NB temp=${temp}:`);
    for (let si = 0; si < SEEDS; si++) {
      let pred = predictNB(inits[si], nbModel, temp);
      // Apply VP per-cell
      const seedObs = vpObs.filter(o => (o.si !== undefined ? o.si : 0) === si);
      const cells = {};
      for (const obs of seedObs) {
        for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
          const gy = obs.vy + dy, gx = obs.vx + dx;
          if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
          if (inits[si][gy][gx] === 10 || inits[si][gy][gx] === 5) continue;
          const k = `${gy},${gx}`, fc = t2c(obs.grid[dy][dx]);
          if (!cells[k]) cells[k] = { n: 0, counts: new Float64Array(C) };
          cells[k].n++; cells[k].counts[fc]++;
        }
      }
      for (const [key, cell] of Object.entries(cells)) {
        const [y, x] = key.split(',').map(Number);
        const pw = cell.n >= 5 ? 2 : cell.n >= 3 ? 4 : cell.n >= 2 ? 7 : 15;
        const prior = pred[y][x], posterior = new Array(C); let total = 0;
        for (let c = 0; c < C; c++) { posterior[c] = prior[c] * pw + cell.counts[c]; total += posterior[c]; }
        if (total > 0) { for (let c = 0; c < C; c++) posterior[c] /= total; pred[y][x] = posterior; }
      }
      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const s = pred[y][x].reduce((a, b) => a + b, 0);
        if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false;
      }
      if (!valid) { console.log(`    Seed ${si}: INVALID`); continue; }
      const res = await POST('/submit', { round_id: r9.id, seed_index: si, prediction: pred });
      console.log(`    Seed ${si}: ${res.ok ? 'OK' : 'FAIL'}`);
      await sleep(600);
    }
  }

  // Submit NB+D0 blend
  console.log('\nSubmitting NB+D0 blend:');
  const d0Model = buildWeightedModel(G, R, I, trainRounds, {});
  for (const nbW of [0.2, 0.3, 0.5]) {
    console.log(`  blend NB=${nbW}:`);
    for (let si = 0; si < SEEDS; si++) {
      const nbPred = predictNB(inits[si], nbModel, 1.15);
      const d0Pred = predictD(inits[si], d0Model, 1.15);
      const pred = [];
      for (let y = 0; y < H; y++) { pred[y] = [];
        for (let x = 0; x < W; x++) {
          pred[y][x] = new Array(C);
          let s = 0;
          for (let c = 0; c < C; c++) {
            pred[y][x][c] = nbW * nbPred[y][x][c] + (1 - nbW) * d0Pred[y][x][c];
            s += pred[y][x][c];
          }
          for (let c = 0; c < C; c++) pred[y][x][c] /= s;
        }
      }
      // VP per-cell
      const seedObs = vpObs.filter(o => (o.si !== undefined ? o.si : 0) === si);
      const cells = {};
      for (const obs of seedObs) {
        for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
          const gy = obs.vy + dy, gx = obs.vx + dx;
          if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
          if (inits[si][gy][gx] === 10 || inits[si][gy][gx] === 5) continue;
          const k = `${gy},${gx}`, fc = t2c(obs.grid[dy][dx]);
          if (!cells[k]) cells[k] = { n: 0, counts: new Float64Array(C) };
          cells[k].n++; cells[k].counts[fc]++;
        }
      }
      for (const [key, cell] of Object.entries(cells)) {
        const [y, x] = key.split(',').map(Number);
        const pw = cell.n >= 5 ? 2 : cell.n >= 3 ? 4 : cell.n >= 2 ? 7 : 15;
        const prior = pred[y][x], posterior = new Array(C); let total = 0;
        for (let c = 0; c < C; c++) { posterior[c] = prior[c] * pw + cell.counts[c]; total += posterior[c]; }
        if (total > 0) { for (let c = 0; c < C; c++) posterior[c] /= total; pred[y][x] = posterior; }
      }
      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const s = pred[y][x].reduce((a, b) => a + b, 0);
        if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false;
      }
      if (!valid) { console.log(`    Seed ${si}: INVALID`); continue; }
      const res = await POST('/submit', { round_id: r9.id, seed_index: si, prediction: pred });
      console.log(`    Seed ${si}: ${res.ok ? 'OK' : 'FAIL'}`);
      await sleep(600);
    }
  }

  // Submit round-weighted model
  console.log('\nSubmitting round-weighted:');
  for (const scheme of weightSchemes.slice(1)) {
    console.log(`  ${scheme.name}:`);
    const model = buildWeightedModel(G, R, I, trainRounds, scheme.weights);
    for (let si = 0; si < SEEDS; si++) {
      let pred = predictD(inits[si], model, 1.15);
      // VP per-cell
      const seedObs = vpObs.filter(o => (o.si !== undefined ? o.si : 0) === si);
      const cells = {};
      for (const obs of seedObs) {
        for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
          const gy = obs.vy + dy, gx = obs.vx + dx;
          if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
          if (inits[si][gy][gx] === 10 || inits[si][gy][gx] === 5) continue;
          const k = `${gy},${gx}`, fc = t2c(obs.grid[dy][dx]);
          if (!cells[k]) cells[k] = { n: 0, counts: new Float64Array(C) };
          cells[k].n++; cells[k].counts[fc]++;
        }
      }
      for (const [key, cell] of Object.entries(cells)) {
        const [y, x] = key.split(',').map(Number);
        const pw = cell.n >= 5 ? 2 : cell.n >= 3 ? 4 : cell.n >= 2 ? 7 : 15;
        const prior = pred[y][x], posterior = new Array(C); let total = 0;
        for (let c = 0; c < C; c++) { posterior[c] = prior[c] * pw + cell.counts[c]; total += posterior[c]; }
        if (total > 0) { for (let c = 0; c < C; c++) posterior[c] /= total; pred[y][x] = posterior; }
      }
      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const s = pred[y][x].reduce((a, b) => a + b, 0);
        if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false;
      }
      if (!valid) { console.log(`    Seed ${si}: INVALID`); continue; }
      const res = await POST('/submit', { round_id: r9.id, seed_index: si, prediction: pred });
      console.log(`    Seed ${si}: ${res.ok ? 'OK' : 'FAIL'}`);
      await sleep(600);
    }
  }

  console.log('\n=== DONE ===');
  console.log('Time:', new Date().toISOString());
}

main().catch(e => console.error('Error:', e.message, e.stack));
