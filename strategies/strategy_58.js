const { H, W, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const K = config.K || 4;
  const floor = config.FLOOR || 0.0001;
  const regBlend = config.RB || 0.8;
  const regSigma = config.RS || 0.10;
  const shrinkLambda = config.SL || 2;
  const ridge = config.RIDGE != null ? config.RIDGE : 0.002;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // K-nearest baseline
  const candidates = { ...growthRates }; delete candidates[String(testRound)];
  const closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  const adaptiveModel = mergeBuckets(perRoundBuckets, closestRounds);
  const allModel = mergeBuckets(perRoundBuckets, allRounds);

  // Collect per-key data
  const keyData = {};
  const coarseKeyData = {};
  for (const r of allRounds) {
    const g = growthRates[String(r)];
    if (g === undefined) continue;
    const b = perRoundBuckets[String(r)];
    if (!b) continue;
    for (const [key, val] of Object.entries(b)) {
      if (!keyData[key]) keyData[key] = [];
      keyData[key].push({ g, dist: val.sum.map(v => v / val.count) });
    }
    const roundCoarse = {};
    for (const [key, val] of Object.entries(b)) {
      const ck = key.endsWith('c') ? key.slice(0, -1) : key;
      if (!roundCoarse[ck]) roundCoarse[ck] = { count: 0, sum: [0,0,0,0,0,0] };
      roundCoarse[ck].count += val.count;
      for (let c = 0; c < 6; c++) roundCoarse[ck].sum[c] += val.sum[c];
    }
    for (const [ck, val] of Object.entries(roundCoarse)) {
      if (!coarseKeyData[ck]) coarseKeyData[ck] = [];
      coarseKeyData[ck].push({ g, dist: val.sum.map(v => v / val.count) });
    }
  }

  function fitRegression(data) {
    if (!data || data.length < 3) return null;
    const ws = data.map(d => {
      const diff = d.g - targetGrowth;
      return Math.exp(-diff * diff / (2 * regSigma * regSigma));
    });
    const wSum = ws.reduce((a, b) => a + b, 0);
    if (wSum < 0.01) return null;

    let gMean = 0;
    for (let i = 0; i < data.length; i++) gMean += ws[i] * data[i].g;
    gMean /= wSum;

    let gVar = 0;
    for (let i = 0; i < data.length; i++) {
      const dg = data[i].g - gMean;
      gVar += ws[i] * dg * dg;
    }
    gVar /= wSum;

    const result = new Array(6);
    for (let c = 0; c < 6; c++) {
      let pMean = 0;
      for (let i = 0; i < data.length; i++) pMean += ws[i] * data[i].dist[c];
      pMean /= wSum;

      if (gVar < 1e-10) { result[c] = Math.max(0, pMean); continue; }

      let cov = 0;
      for (let i = 0; i < data.length; i++) {
        cov += ws[i] * (data[i].g - gMean) * (data[i].dist[c] - pMean);
      }
      cov /= wSum;

      // Ridge regression: dampen slope
      const slope = cov / (gVar + ridge);
      result[c] = Math.max(0, pMean + slope * (targetGrowth - gMean));
    }
    const rSum = result.reduce((a, b) => a + b, 0);
    if (rSum > 0) for (let c = 0; c < 6; c++) result[c] /= rSum;
    return { dist: result, nRounds: data.length };
  }

  const regCache = {};
  const coarseRegCache = {};
  for (const key of Object.keys(keyData)) regCache[key] = fitRegression(keyData[key]);
  for (const key of Object.keys(coarseKeyData)) coarseRegCache[key] = fitRegression(coarseKeyData[key]);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const ck = key.endsWith('c') ? key.slice(0, -1) : key;

      let basePrior = adaptiveModel[key] || adaptiveModel[ck] || allModel[key] || allModel[ck] || null;

      let regPrior = null;
      const fineReg = regCache[key];
      const coarseReg = coarseRegCache[ck];
      if (fineReg && coarseReg && key !== ck) {
        const alpha = fineReg.nRounds / (fineReg.nRounds + shrinkLambda);
        regPrior = fineReg.dist.map((v, c) => alpha * v + (1 - alpha) * coarseReg.dist[c]);
      } else if (fineReg) {
        regPrior = fineReg.dist;
      } else if (coarseReg) {
        regPrior = coarseReg.dist;
      }

      let prior;
      if (basePrior && regPrior) {
        prior = basePrior.map((v, c) => (1 - regBlend) * v + regBlend * regPrior[c]);
      } else if (regPrior) {
        prior = [...regPrior];
      } else if (basePrior) {
        prior = [...basePrior];
      } else {
        prior = [1/6,1/6,1/6,1/6,1/6,1/6];
      }

      const floored = prior.map(v => Math.max(v, floor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}
module.exports = { predict };
