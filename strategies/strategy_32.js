const { H, W, selectClosestRounds } = require('./shared');

function computeKeys(grid, sp, sl, y, x) {
  const v = grid[y][x];
  if (v === 10) return { std: 'O', rich: 'O' };
  if (v === 5) return { std: 'M', rich: 'M' };
  const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P';
  let nS = 0;
  for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
    if (!dy && !dx) continue;
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && sp.has(ny * W + nx)) nS++;
  }
  let coast = false;
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) coast = true;
  }
  const std = t + (nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3') + (coast ? 'c' : '');
  let rich = std;
  if (nS === 0 && t !== 'S' && sl.length > 0) {
    let md = 999;
    for (const s of sl) md = Math.min(md, Math.max(Math.abs(s.y - y), Math.abs(s.x - x)));
    rich = std + (md <= 4 ? 'n' : md <= 8 ? 'm' : 'f');
  } else if (nS >= 1 && nS <= 2 && t !== 'S') {
    let md = 999;
    for (const s of sl) {
      const d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
      if (d <= 3) md = Math.min(md, d);
    }
    rich = std + (md <= 1 ? 'a' : md <= 2 ? 'b' : '');
  }
  return { std, rich };
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.00005;
  const bw = config.BW || 0.042;
  const temp = config.TEMP || 1.10;
  const regBlend = config.REG_BLEND || 1.0;
  const regSmooth = config.REG_SMOOTH || 0.01;
  const regBW = config.REG_BW || 0.15;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  // Build Gaussian weights for fallback model
  const weights = {};
  for (const [rn, rate] of Object.entries(growthRates)) {
    if (parseInt(rn) === testRound) continue;
    if (!perRoundBuckets[rn]) continue;
    weights[rn] = Math.max(0.001, Math.exp(-0.5 * ((rate - targetGrowth) / bw) ** 2));
  }

  const allRoundNums = Object.keys(weights).map(Number);

  // Gaussian-weighted model (fallback for keys without enough regression data)
  const gaussModel = {};
  for (const [rn, w] of Object.entries(weights)) {
    const b = perRoundBuckets[rn];
    if (!b) continue;
    for (const [key, v] of Object.entries(b)) {
      if (!gaussModel[key]) gaussModel[key] = { d: [0,0,0,0,0,0], tw: 0 };
      const rd = v.sum.map(s => s / v.count);
      for (let c = 0; c < 6; c++) gaussModel[key].d[c] += w * rd[c];
      gaussModel[key].tw += w;
    }
  }
  const modelDist = {};
  for (const [k, v] of Object.entries(gaussModel)) {
    modelDist[k] = v.d.map(s => s / v.tw);
  }

  // Weighted linear regression: prob_c = a + b * growth, weighted by growth similarity
  const regModel = {};
  for (const key of Object.keys(modelDist)) {
    const xs = [], ys = [], ws = [];
    for (const rn of allRoundNums) {
      const b = perRoundBuckets[String(rn)];
      if (!b || !b[key]) continue;
      xs.push(growthRates[String(rn)]);
      ys.push(b[key].sum.map(s => s / b[key].count));
      ws.push(Math.exp(-0.5 * ((growthRates[String(rn)] - targetGrowth) / regBW) ** 2));
    }
    if (xs.length < 4) continue;

    const n = xs.length;
    const wSum = ws.reduce((a, b) => a + b, 0);
    const yMean = new Array(6).fill(0);
    for (let i = 0; i < n; i++) for (let c = 0; c < 6; c++) yMean[c] += ys[i][c] * ws[i];
    for (let c = 0; c < 6; c++) yMean[c] /= wSum;

    const ySimpleMean = new Array(6).fill(0);
    for (const p of ys) for (let c = 0; c < 6; c++) ySimpleMean[c] += p[c];
    for (let c = 0; c < 6; c++) ySimpleMean[c] /= n;

    const xMean = xs.reduce((a, x, i) => a + x * ws[i], 0) / wSum;
    let xxVarW = 0;
    for (let i = 0; i < n; i++) xxVarW += ws[i] * (xs[i] - xMean) ** 2;
    if (xxVarW < 1e-10) { regModel[key] = yMean; continue; }

    const regPred = new Array(6);
    for (let c = 0; c < 6; c++) {
      let xyCovarW = 0;
      for (let i = 0; i < n; i++) xyCovarW += ws[i] * (xs[i] - xMean) * (ys[i][c] - yMean[c]);
      regPred[c] = Math.max(0, yMean[c] + (xyCovarW / xxVarW) * (targetGrowth - xMean));
    }
    const regSum = regPred.reduce((a, b) => a + b, 0);
    if (regSum > 0) for (let c = 0; c < 6; c++) regPred[c] /= regSum;

    const blended = new Array(6);
    for (let c = 0; c < 6; c++) blended[c] = (1 - regSmooth) * regPred[c] + regSmooth * ySimpleMean[c];
    regModel[key] = blended;
  }

  const settPos = new Set();
  const settList = [];
  for (const s of settlements) { settPos.add(s.y * W + s.x); settList.push(s); }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const { std, rich } = computeKeys(initGrid, settPos, settList, y, x);

      // Lookup Gaussian model (fallback)
      let pGauss = null;
      if (rich !== std) pGauss = modelDist[rich];
      if (!pGauss) pGauss = modelDist[std];
      if (!pGauss && std.length > 1) pGauss = modelDist[std.slice(0, -1)];
      if (!pGauss) { row.push([1/6,1/6,1/6,1/6,1/6,1/6]); continue; }

      // Lookup regression model
      let pReg = null;
      if (rich !== std) pReg = regModel[rich];
      if (!pReg) pReg = regModel[std];
      if (!pReg && std.length > 1) pReg = regModel[std.slice(0, -1)];

      // Blend: primarily regression, Gaussian as fallback
      let prior;
      if (pReg) {
        prior = new Array(6);
        for (let c = 0; c < 6; c++) prior[c] = regBlend * pReg[c] + (1 - regBlend) * pGauss[c];
      } else {
        prior = [...pGauss];
      }

      // Temperature scaling (softens overconfident predictions)
      if (temp > 1.01) {
        let s = 0;
        for (let c = 0; c < 6; c++) {
          prior[c] = Math.pow(Math.max(prior[c], 1e-10), 1 / temp);
          s += prior[c];
        }
        for (let c = 0; c < 6; c++) prior[c] /= s;
      }

      // Adaptive floor (higher for uncertain cells)
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
