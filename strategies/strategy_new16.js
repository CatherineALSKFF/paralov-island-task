const { H, W, getFeatureKey } = require('./shared');
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'data');

let _distBuckets = null;
function loadDistBuckets() {
  if (_distBuckets) return _distBuckets;
  const f = path.join(DATA_DIR, 'enriched_buckets_v8.json');
  if (fs.existsSync(f)) _distBuckets = JSON.parse(fs.readFileSync(f, 'utf8'));
  else _distBuckets = {};
  return _distBuckets;
}

function getDistKey(grid, settPos, y, x) {
  const v = grid[y][x];
  if (v === 10) return 'O'; if (v === 5) return 'M';
  const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P';
  let nO1 = 0, nS1 = 0;
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
      if (grid[ny][nx] === 10) nO1++;
      if (settPos.has(ny * W + nx)) nS1++;
    }
  }
  const coastal = nO1 > 0;
  if (t === 'S') {
    let nS = 0;
    for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
      if (!dy && !dx) continue;
      const ny = y + dy, nx = x + dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx)) nS++;
    }
    return 'S' + (nS <= 1 ? '0' : nS <= 3 ? '1' : '2') + (coastal ? 'c' : '');
  }
  let minSD = 40;
  for (let dy = -10; dy <= 10; dy++) for (let dx = -10; dx <= 10; dx++) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx))
      minSD = Math.min(minSD, Math.max(Math.abs(dy), Math.abs(dx)));
  }
  const dKey = minSD <= 1 ? '1' : minSD <= 2 ? '2' : minSD <= 3 ? '3' : minSD <= 5 ? '4' : minSD <= 7 ? '5' : '6';
  const suffix = (minSD <= 1 && nS1 > 0) ? 't' : '';
  return t + dKey + (coastal ? 'c' : '') + suffix;
}

function coarsenDist(key) {
  if (key === 'O' || key === 'M' || key.length <= 1) return [];
  const levels = []; let k = key;
  if (k.endsWith('t')) { k = k.slice(0, -1); levels.push(k); }
  if (k.endsWith('c')) { k = k.slice(0, -1); levels.push(k); }
  return levels;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const SIGMA = config.SIGMA || 0.05;
  const LAMBDA = config.LAMBDA || 0.001;
  const DIST_W = config.DIST_W || 0.65;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const distBuckets = loadDistBuckets();
  const hasDistBuckets = Object.keys(distBuckets).length > 0;
  const distRounds = Object.keys(distBuckets).map(Number)
    .filter(n => n !== testRound && growthRates[String(n)] !== undefined);
  const stdRounds = Object.keys(perRoundBuckets).map(Number)
    .filter(n => n !== testRound);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const roundWeights = {};
  for (const rn of [...new Set([...distRounds, ...stdRounds])]) {
    const g = growthRates[String(rn)];
    if (g === undefined) { roundWeights[rn] = 0; continue; }
    const d = g - targetGrowth;
    roundWeights[rn] = Math.exp(-(d * d) / (2 * SIGMA * SIGMA));
  }

  // Build prefix-aggregated models for standard buckets
  // For a basic key like "P0c", aggregate all matching enriched keys (P0c, P0cn, P0cm, P0cf)
  const allDataKeys = new Set();
  for (const rn of stdRounds) {
    const b = perRoundBuckets[String(rn)];
    if (b) for (const k of Object.keys(b)) allDataKeys.add(k);
  }
  const dataKeyList = [...allDataKeys];

  const prefixCache = {};
  function getMatchingKeys(simpleKey) {
    if (prefixCache[simpleKey] !== undefined) return prefixCache[simpleKey];
    const matches = [];
    for (const dk of dataKeyList) {
      if (!dk.startsWith(simpleKey)) continue;
      const suffix = dk.slice(simpleKey.length);
      // For non-coastal keys, don't match coastal sub-keys
      if (!simpleKey.endsWith('c') && suffix.includes('c')) continue;
      matches.push(dk);
    }
    prefixCache[simpleKey] = matches.length > 0 ? matches : null;
    return prefixCache[simpleKey];
  }

  // Per-round aggregated data for a given simple key (prefix-matched)
  function getPerRoundData(simpleKey) {
    const matchKeys = getMatchingKeys(simpleKey);
    if (!matchKeys) return null;
    const perRound = {};
    for (const rn of stdRounds) {
      const b = perRoundBuckets[String(rn)]; if (!b) continue;
      let count = 0;
      const sum = [0, 0, 0, 0, 0, 0];
      for (const dk of matchKeys) {
        if (!b[dk]) continue;
        count += b[dk].count;
        for (let c = 0; c < 6; c++) sum[c] += b[dk].sum[c];
      }
      if (count > 0) perRound[rn] = sum.map(s => s / count);
    }
    return Object.keys(perRound).length > 0 ? perRound : null;
  }

  // Build Gaussian + quadratic regression from per-round data
  function buildPredFromPerRound(perRound) {
    const roundsWithData = Object.keys(perRound).map(Number);
    if (roundsWithData.length === 0) return null;

    // Gaussian weighted average
    const gaussMean = [0, 0, 0, 0, 0, 0];
    let wSum = 0;
    for (const rn of roundsWithData) {
      const w = roundWeights[rn] || 0; if (!w) continue;
      wSum += w;
      for (let c = 0; c < 6; c++) gaussMean[c] += w * perRound[rn][c];
    }
    if (wSum > 0) for (let c = 0; c < 6; c++) gaussMean[c] /= wSum;

    // Quadratic regression
    const regPred = [];
    for (let cls = 0; cls < 6; cls++) {
      const pts = [];
      for (const rn of roundsWithData) {
        const w = roundWeights[rn] || 0; if (!w) continue;
        const g = growthRates[String(rn)];
        if (g === undefined) continue;
        pts.push({ g, p: perRound[rn][cls], w });
      }
      if (pts.length >= 6) {
        let s0=0, s1=0, s2=0, s3=0, s4=0, sy=0, sy1=0, sy2=0;
        for (const {g, p, w} of pts) {
          const g2 = g*g;
          s0 += w; s1 += w*g; s2 += w*g2; s3 += w*g*g2; s4 += w*g2*g2;
          sy += w*p; sy1 += w*g*p; sy2 += w*g2*p;
        }
        const A = [[s0+LAMBDA, s1, s2], [s1, s2+LAMBDA, s3], [s2, s3, s4+LAMBDA]];
        const b = [sy, sy1, sy2];
        const det = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
                  - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
                  + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
        if (Math.abs(det) > 1e-12) {
          const a = (b[0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1]) - A[0][1]*(b[1]*A[2][2]-A[1][2]*b[2]) + A[0][2]*(b[1]*A[2][1]-A[1][1]*b[2])) / det;
          const bC = (A[0][0]*(b[1]*A[2][2]-A[1][2]*b[2]) - b[0]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) + A[0][2]*(A[1][0]*b[2]-b[1]*A[2][0])) / det;
          const c = (A[0][0]*(A[1][1]*b[2]-b[1]*A[2][1]) - A[0][1]*(A[1][0]*b[2]-b[1]*A[2][0]) + b[0]*(A[1][0]*A[2][1]-A[1][1]*A[2][0])) / det;
          const g = targetGrowth;
          regPred.push(Math.max(0, a + bC * g + c * g * g));
        } else {
          regPred.push(Math.max(0, sy / s0));
        }
      } else if (pts.length >= 3) {
        let sW=0, sWX=0, sWY=0, sWXX=0, sWXY=0;
        for (const {g, p, w} of pts) {
          sW += w; sWX += w*g; sWY += w*p; sWXX += w*g*g; sWXY += w*g*p;
        }
        const denom = sW*sWXX - sWX*sWX + LAMBDA*sW;
        const slope = denom > 1e-10 ? (sW*sWXY - sWX*sWY) / denom : 0;
        const intercept = (sWY - slope*sWX) / sW;
        regPred.push(Math.max(0, intercept + slope * targetGrowth));
      } else {
        regPred.push(gaussMean[cls]);
      }
    }

    // Normalize regression prediction
    const regSum = regPred.reduce((a, b) => a + b, 0);
    const normReg = regSum > 0 ? regPred.map(v => v / regSum) : gaussMean;

    // Compute residual-based confidence
    let sse = 0, sseW = 0;
    for (const rn of roundsWithData) {
      const w = roundWeights[rn] || 0; if (!w) continue;
      const g = growthRates[String(rn)]; if (g === undefined) continue;
      for (let c = 0; c < 6; c++) {
        const diff = perRound[rn][c] - normReg[c]; // approximate
        sse += w * diff * diff;
      }
      sseW += w;
    }
    const rmse = sseW > 0 ? Math.sqrt(sse / sseW) : 0.1;
    const regWeight = Math.max(0.2, Math.min(0.8, 0.7 - rmse * 3));

    // Blend gauss and regression
    return gaussMean.map((v, c) => (1 - regWeight) * v + regWeight * normReg[c]);
  }

  // Distance-based models (same as before)
  function buildGauss(bkts, rounds) {
    const model = {};
    for (const rn of rounds) {
      const w = roundWeights[rn]; if (!w) continue;
      const b = bkts[String(rn)]; if (!b) continue;
      for (const [key, val] of Object.entries(b)) {
        if (!model[key]) model[key] = { wsum: new Array(6).fill(0), wtotal: 0 };
        const avg = val.sum.map(v => v / val.count);
        for (let c = 0; c < 6; c++) model[key].wsum[c] += w * avg[c];
        model[key].wtotal += w;
      }
    }
    const out = {};
    for (const [k, v] of Object.entries(model)) out[k] = v.wsum.map(s => s / v.wtotal);
    return out;
  }

  function buildReg(bkts, rounds) {
    const allKeys = new Set();
    for (const rn of rounds) {
      const b = bkts[String(rn)];
      if (b) for (const k of Object.keys(b)) allKeys.add(k);
    }
    const model = {};
    for (const key of allKeys) {
      const classes = [];
      for (let cls = 0; cls < 6; cls++) {
        const pts = [];
        for (const rn of rounds) {
          const w = roundWeights[rn]; if (!w) continue;
          const bucket = bkts[String(rn)]?.[key];
          if (!bucket || bucket.count === 0) continue;
          pts.push({ g: growthRates[String(rn)], p: bucket.sum[cls] / bucket.count, w });
        }
        if (pts.length >= 6) {
          let s0=0, s1=0, s2=0, s3=0, s4=0, sy=0, sy1=0, sy2=0;
          for (const {g, p, w} of pts) {
            const g2 = g*g;
            s0 += w; s1 += w*g; s2 += w*g2; s3 += w*g*g2; s4 += w*g2*g2;
            sy += w*p; sy1 += w*g*p; sy2 += w*g2*p;
          }
          const A = [[s0+LAMBDA, s1, s2], [s1, s2+LAMBDA, s3], [s2, s3, s4+LAMBDA]];
          const b = [sy, sy1, sy2];
          const det = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
                    - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
                    + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
          if (Math.abs(det) > 1e-12) {
            const a = (b[0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1]) - A[0][1]*(b[1]*A[2][2]-A[1][2]*b[2]) + A[0][2]*(b[1]*A[2][1]-A[1][1]*b[2])) / det;
            const bC = (A[0][0]*(b[1]*A[2][2]-A[1][2]*b[2]) - b[0]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) + A[0][2]*(A[1][0]*b[2]-b[1]*A[2][0])) / det;
            const c = (A[0][0]*(A[1][1]*b[2]-b[1]*A[2][1]) - A[0][1]*(A[1][0]*b[2]-b[1]*A[2][0]) + b[0]*(A[1][0]*A[2][1]-A[1][1]*A[2][0])) / det;
            let sse = 0;
            for (const {g, p, w: wt} of pts) {
              const pred = a + bC*g + c*g*g;
              sse += wt * (p-pred)*(p-pred);
            }
            classes.push({ a, b: bC, c, rmse: Math.sqrt(sse/s0) });
          } else {
            classes.push({ a: sy/s0, b: 0, c: 0, rmse: 0.1 });
          }
        } else if (pts.length >= 3) {
          let sW=0, sWX=0, sWY=0, sWXX=0, sWXY=0;
          for (const {g, p, w} of pts) {
            sW += w; sWX += w*g; sWY += w*p; sWXX += w*g*g; sWXY += w*g*p;
          }
          const denom = sW*sWXX - sWX*sWX + LAMBDA*sW;
          const slope = denom > 1e-10 ? (sW*sWXY - sWX*sWY) / denom : 0;
          const intercept = (sWY - slope*sWX) / sW;
          let sse = 0;
          for (const {g, p, w} of pts) sse += w*(p-intercept-slope*g)*(p-intercept-slope*g);
          classes.push({ a: intercept, b: slope, c: 0, rmse: Math.sqrt(sse/sW) });
        } else if (pts.length > 0) {
          let sW=0, sWY=0;
          for (const {p, w} of pts) { sW += w; sWY += w*p; }
          classes.push({ a: sWY/sW, b: 0, c: 0, rmse: 0.15 });
        } else {
          classes.push({ a: 1/6, b: 0, c: 0, rmse: 0.2 });
        }
      }
      model[key] = classes;
    }
    return model;
  }

  function regPredFn(model, key) {
    if (!model[key]) return null;
    const g = targetGrowth;
    const pred = model[key].map(m => Math.max(0, m.a + m.b * g + m.c * g * g));
    const sum = pred.reduce((a, b) => a + b, 0);
    return sum > 0 ? pred.map(v => v / sum) : null;
  }

  function ensD(gm, rm, key) {
    const g = gm[key] ? [...gm[key]] : null;
    const r = regPredFn(rm, key);
    if (g && r && rm[key]) {
      const avgRmse = rm[key].reduce((a, m) => a + m.rmse, 0) / 6;
      const regW = Math.max(0.2, Math.min(0.8, 0.7 - avgRmse * 3));
      return g.map((v, c) => (1 - regW) * v + regW * r[c]);
    }
    return g || r;
  }

  const dG = hasDistBuckets ? buildGauss(distBuckets, distRounds) : {};
  const dR = hasDistBuckets ? buildReg(distBuckets, distRounds) : {};

  // Precompute prefix-aggregated standard model for common keys
  const stdKeyCache = {};
  function getStdPred(simpleKey) {
    if (stdKeyCache[simpleKey] !== undefined) return stdKeyCache[simpleKey];
    const perRound = getPerRoundData(simpleKey);
    if (!perRound) { stdKeyCache[simpleKey] = null; return null; }
    const pred = buildPredFromPerRound(perRound);
    stdKeyCache[simpleKey] = pred;
    return pred;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const dk = getDistKey(initGrid, settPos, y, x);
      const bk = getFeatureKey(initGrid, settPos, y, x);

      // Distance-based prediction
      let dp = ensD(dG, dR, dk);
      if (!dp) for (const ck of coarsenDist(dk)) { dp = ensD(dG, dR, ck); if (dp) break; }

      // Standard prediction with prefix aggregation
      let sp = getStdPred(bk);
      if (!sp && bk.length > 2 && bk.endsWith('c')) sp = getStdPred(bk.slice(0, -1));
      if (!sp && bk.length > 1) sp = getStdPred(bk[0]);

      let prior;
      if (dp && sp) {
        prior = dp.map((v, c) => DIST_W * v + (1 - DIST_W) * sp[c]);
      } else {
        prior = dp || sp || [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6];
      }

      let entropy = 0;
      for (let c = 0; c < 6; c++) {
        if (prior[c] > 0.001) entropy -= prior[c] * Math.log(prior[c]);
      }
      const entRatio = Math.min(entropy / Math.log(6), 1);
      const cellFloor = floor * (0.05 + 0.95 * entRatio);

      const floored = prior.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
