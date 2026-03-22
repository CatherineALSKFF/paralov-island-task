const { H, W, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const bw = config.BW || 0.042;
  const temp = config.TEMP || 1.10;
  const interpW = config.INTERP || 0.5;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  // Build Gaussian-weighted per-round-normalized model
  const weights = {};
  for (const [rn, rate] of Object.entries(growthRates)) {
    if (parseInt(rn) === testRound) continue;
    if (!perRoundBuckets[rn]) continue;
    weights[rn] = Math.max(0.001, Math.exp(-0.5 * ((rate - targetGrowth) / bw) ** 2));
  }

  const model = {};
  for (const [rn, w] of Object.entries(weights)) {
    const b = perRoundBuckets[rn];
    if (!b) continue;
    for (const [key, v] of Object.entries(b)) {
      if (!model[key]) model[key] = { d: [0,0,0,0,0,0], tw: 0 };
      const rd = v.sum.map(s => s / v.count);
      for (let c = 0; c < 6; c++) model[key].d[c] += w * rd[c];
      model[key].tw += w;
    }
  }
  const M = {};
  for (const [k, v] of Object.entries(model)) {
    M[k] = v.d.map(s => s / v.tw);
  }

  const settPos = new Set();
  const settList = [];
  for (const s of settlements) { settPos.add(s.y * W + s.x); settList.push(s); }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const v = initGrid[y][x];
      if (v === 10) { row.push(M['O'] ? [...M['O']] : [1,0,0,0,0,0]); continue; }
      if (v === 5) { row.push(M['M'] ? [...M['M']] : [0,0,0,0,0,1]); continue; }

      const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P';
      let nS = 0;
      for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
        if (!dy && !dx) continue;
        const ny = y + dy, nx = x + dx;
        if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx)) nS++;
      }
      let coast = false;
      for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
        const ny = y + dy, nx = x + dx;
        if (ny >= 0 && ny < H && nx >= 0 && nx < W && initGrid[ny][nx] === 10) coast = true;
      }
      const std = t + (nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3') + (coast ? 'c' : '');

      // Compute Chebyshev distance to nearest settlement
      let md = 999;
      for (const s of settList) md = Math.min(md, Math.max(Math.abs(s.y - y), Math.abs(s.x - x)));

      // For nS=1-2 cells: min distance to nearest within r=3
      let nearD = 999;
      if (nS >= 1 && nS <= 2) {
        for (const s of settList) {
          const d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
          if (d <= 3) nearD = Math.min(nearD, d);
        }
      }

      // Generate rich keys for interpolation
      let prior;
      if (nS === 0 && t !== 'S' && settList.length > 0) {
        // Distance bins: n(≤4), m(5-8), f(>8). Interpolate at boundaries.
        const keyN = std + 'n', keyM = std + 'm', keyF = std + 'f';
        const pN = M[keyN], pM = M[keyM], pF = M[keyF];

        if (md <= 3) {
          prior = pN || M[std];
        } else if (md === 4) {
          // Boundary: interpolate n and m
          if (pN && pM) {
            prior = new Array(6);
            const alpha = interpW;
            for (let c = 0; c < 6; c++) prior[c] = (1-alpha) * pN[c] + alpha * pM[c];
          } else { prior = pN || pM || M[std]; }
        } else if (md <= 7) {
          prior = pM || M[std];
        } else if (md === 8) {
          // Boundary: interpolate m and f
          if (pM && pF) {
            prior = new Array(6);
            const alpha = interpW;
            for (let c = 0; c < 6; c++) prior[c] = (1-alpha) * pM[c] + alpha * pF[c];
          } else { prior = pM || pF || M[std]; }
        } else {
          prior = pF || M[std];
        }
      } else if (nS >= 1 && nS <= 2 && t !== 'S') {
        // Distance bins: a(≤1), b(2), none(3+)
        const keyA = std + 'a', keyB = std + 'b';
        const pA = M[keyA], pB = M[keyB], pStd = M[std];

        if (nearD <= 1) {
          prior = pA || pStd;
        } else if (nearD === 2) {
          // Interpolate a and b? Actually a=≤1, b=2, so just use b
          prior = pB || pStd;
        } else {
          // nearD >= 3, use standard key
          prior = pStd;
        }
      } else {
        prior = M[std];
      }

      // Fallback chain
      if (!prior) {
        const fb = std.slice(0, -1);
        prior = M[fb];
      }
      if (!prior) { row.push([1/6,1/6,1/6,1/6,1/6,1/6]); continue; }
      prior = [...prior];

      // Temperature scaling
      if (temp > 1.01) {
        let s = 0;
        for (let c = 0; c < 6; c++) {
          prior[c] = Math.pow(Math.max(prior[c], 1e-10), 1 / temp);
          s += prior[c];
        }
        for (let c = 0; c < 6; c++) prior[c] /= s;
      }

      // Adaptive floor
      let entropy = 0;
      for (let c = 0; c < 6; c++) {
        if (prior[c] > 1e-10) entropy -= prior[c] * Math.log(prior[c]);
      }
      const eRatio = entropy / Math.log(6);
      const cellFloor = floor * (0.02 + 0.98 * eRatio * eRatio);

      const floored = prior.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
