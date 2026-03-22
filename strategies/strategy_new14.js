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

function getEnrichedKey(grid, settPos, y, x) {
  const v = grid[y][x];
  if (v === 10) return 'O'; if (v === 5) return 'M';
  const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P';
  let nS = 0;
  for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
    if (!dy && !dx) continue; const ny = y+dy, nx = x+dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny*W+nx)) nS++;
  }
  let coastal = false;
  for (const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y+dy, nx = x+dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) coastal = true;
  }
  const sKey = nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3';
  let suffix = '';
  if (t !== 'S') {
    let minSD = 40;
    for (let dy = -7; dy <= 7; dy++) for (let dx = -7; dx <= 7; dx++) {
      const ny = y+dy, nx = x+dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny*W+nx))
        minSD = Math.min(minSD, Math.max(Math.abs(dy), Math.abs(dx)));
    }
    if (nS === 0) suffix = minSD <= 5 ? 'n' : minSD <= 7 ? 'm' : 'f';
    else if (nS <= 2) suffix = nS === 1 ? 'a' : 'b';
  }
  return t + sKey + (coastal ? 'c' : '') + suffix;
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

function coarsenEnriched(key) {
  if (key === 'O' || key === 'M' || key.length <= 1) return [];
  const levels = []; let k = key;
  const last = k[k.length - 1];
  if ('nmfab'.includes(last)) { k = k.slice(0, -1); levels.push(k); }
  if (k.endsWith('c')) { k = k.slice(0, -1); levels.push(k); }
  return levels;
}

// Solve 2x2 linear system: A * [a, b]^T = [c1, c2]
function solve2x2(a11, a12, a21, a22, c1, c2) {
  const det = a11 * a22 - a12 * a21;
  if (Math.abs(det) < 1e-12) return null;
  return [(a22 * c1 - a12 * c2) / det, (a11 * c2 - a21 * c1) / det];
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

  // Quadratic ridge regression: p = a + b*g + c*g^2
  function buildQuadReg(bkts, rounds) {
    const allKeys = new Set();
    for (const rn of rounds) {
      const b = bkts[String(rn)];
      if (b) for (const k of Object.keys(b)) allKeys.add(k);
    }
    const model = {};
    for (const key of allKeys) {
      const classes = [];
      for (let cls = 0; cls < 6; cls++) {
        // Collect weighted data points
        const pts = [];
        for (const rn of rounds) {
          const w = roundWeights[rn]; if (!w) continue;
          const bucket = bkts[String(rn)]?.[key];
          if (!bucket || bucket.count === 0) continue;
          const g = growthRates[String(rn)];
          const p = bucket.sum[cls] / bucket.count;
          pts.push({ g, p, w });
        }

        if (pts.length >= 6) {
          // Fit quadratic: p = a + b*g + c*g^2 with ridge
          let s0=0, s1=0, s2=0, s3=0, s4=0, sy=0, sy1=0, sy2=0;
          for (const {g, p, w} of pts) {
            const g2 = g*g;
            s0 += w; s1 += w*g; s2 += w*g2; s3 += w*g*g2; s4 += w*g2*g2;
            sy += w*p; sy1 += w*g*p; sy2 += w*g2*p;
          }
          // Normal equations with ridge: (X'WX + lambda*I) * beta = X'Wy
          const A = [
            [s0 + LAMBDA, s1, s2],
            [s1, s2 + LAMBDA, s3],
            [s2, s3, s4 + LAMBDA]
          ];
          const b = [sy, sy1, sy2];
          // Solve 3x3 using Cramer's rule
          const det = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
                    - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
                    + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
          if (Math.abs(det) > 1e-12) {
            const a = (b[0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1]) - A[0][1]*(b[1]*A[2][2]-A[1][2]*b[2]) + A[0][2]*(b[1]*A[2][1]-A[1][1]*b[2])) / det;
            const bCoef = (A[0][0]*(b[1]*A[2][2]-A[1][2]*b[2]) - b[0]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) + A[0][2]*(A[1][0]*b[2]-b[1]*A[2][0])) / det;
            const c = (A[0][0]*(A[1][1]*b[2]-b[1]*A[2][1]) - A[0][1]*(A[1][0]*b[2]-b[1]*A[2][0]) + b[0]*(A[1][0]*A[2][1]-A[1][1]*A[2][0])) / det;
            classes.push({ a, b: bCoef, c, quad: true });
          } else {
            classes.push({ a: sy / s0, b: 0, c: 0, quad: false });
          }
        } else if (pts.length >= 3) {
          // Linear regression
          let sW=0, sWX=0, sWY=0, sWXX=0, sWXY=0;
          for (const {g, p, w} of pts) {
            sW += w; sWX += w*g; sWY += w*p; sWXX += w*g*g; sWXY += w*g*p;
          }
          const denom = sW*sWXX - sWX*sWX + LAMBDA*sW;
          const slope = denom > 1e-10 ? (sW*sWXY - sWX*sWY) / denom : 0;
          const intercept = (sWY - slope*sWX) / sW;
          classes.push({ a: intercept, b: slope, c: 0, quad: false });
        } else if (pts.length > 0) {
          let sW = 0, sWY = 0;
          for (const {p, w} of pts) { sW += w; sWY += w*p; }
          classes.push({ a: sWY / sW, b: 0, c: 0, quad: false });
        } else {
          classes.push({ a: 1/6, b: 0, c: 0, quad: false });
        }
      }
      model[key] = classes;
    }
    return model;
  }

  function regPred(model, key) {
    if (!model[key]) return null;
    const g = targetGrowth;
    const pred = model[key].map(m => Math.max(0, m.a + m.b * g + m.c * g * g));
    const sum = pred.reduce((a, b) => a + b, 0);
    return sum > 0 ? pred.map(v => v / sum) : null;
  }

  function ens(gm, rm, key) {
    const g = gm[key] ? [...gm[key]] : null;
    const r = regPred(rm, key);
    if (g && r) return g.map((v, c) => 0.5 * v + 0.5 * r[c]);
    return g || r;
  }

  const dG = hasDistBuckets ? buildGauss(distBuckets, distRounds) : {};
  const dR = hasDistBuckets ? buildQuadReg(distBuckets, distRounds) : {};
  const sG = buildGauss(perRoundBuckets, stdRounds);
  const sR = buildQuadReg(perRoundBuckets, stdRounds);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const dk = getDistKey(initGrid, settPos, y, x);
      const ek = getEnrichedKey(initGrid, settPos, y, x);

      let dp = ens(dG, dR, dk);
      if (!dp) for (const ck of coarsenDist(dk)) { dp = ens(dG, dR, ck); if (dp) break; }

      let ep = ens(sG, sR, ek);
      if (!ep) for (const ck of coarsenEnriched(ek)) { ep = ens(sG, sR, ck); if (ep) break; }

      let prior;
      if (dp && ep) {
        prior = dp.map((v, c) => DIST_W * v + (1 - DIST_W) * ep[c]);
      } else {
        prior = dp || ep || [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6];
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
