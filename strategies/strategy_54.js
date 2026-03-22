const { H, W, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const K = config.K || 4;
  const floor = config.FLOOR || 0.0001;
  const regBlend = config.RB || 0.8;
  const regSigma = config.RS || 0.10;
  const shrinkLambda = config.SL || 2;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Compute test round's settlement count
  const testNSett = settlements.length;

  // Compute per-training-round settlement count from buckets
  const roundNSett = {};
  for (const r of allRounds) {
    const b = perRoundBuckets[String(r)];
    if (!b) continue;
    let nSett = 0;
    for (const [key, val] of Object.entries(b)) {
      if (key.startsWith('S')) nSett += val.count;
    }
    // Bucket counts aggregate 5 seeds, so divide by 5
    roundNSett[r] = nSett / 5;
  }

  // K-nearest baseline
  const candidates = { ...growthRates }; delete candidates[String(testRound)];
  const closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  const adaptiveModel = mergeBuckets(perRoundBuckets, closestRounds);
  const allModel = mergeBuckets(perRoundBuckets, allRounds);

  // Collect per-key data with 2 features: (growth, nSett) -> distribution
  const keyData = {};
  const coarseKeyData = {};
  for (const r of allRounds) {
    const g = growthRates[String(r)];
    if (g === undefined) continue;
    const ns = roundNSett[r] || 30;
    const b = perRoundBuckets[String(r)];
    if (!b) continue;
    for (const [key, val] of Object.entries(b)) {
      if (!keyData[key]) keyData[key] = [];
      keyData[key].push({ g, ns, dist: val.sum.map(v => v / val.count) });
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
      coarseKeyData[ck].push({ g, ns, dist: val.sum.map(v => v / val.count) });
    }
  }

  // Multivariate local linear regression with 2 features
  function fitRegression(data) {
    if (!data || data.length < 4) return null;

    const ws = data.map(d => {
      const dg = d.g - targetGrowth;
      return Math.exp(-dg * dg / (2 * regSigma * regSigma));
    });
    const wSum = ws.reduce((a, b) => a + b, 0);
    if (wSum < 0.01) return null;

    // Weighted means
    let gMean = 0, nsMean = 0;
    for (let i = 0; i < data.length; i++) {
      gMean += ws[i] * data[i].g;
      nsMean += ws[i] * data[i].ns;
    }
    gMean /= wSum; nsMean /= wSum;

    // Weighted covariance matrix for (g, ns)
    let sgg = 0, snn = 0, sgn = 0;
    for (let i = 0; i < data.length; i++) {
      const dg = data[i].g - gMean;
      const dn = data[i].ns - nsMean;
      sgg += ws[i] * dg * dg;
      snn += ws[i] * dn * dn;
      sgn += ws[i] * dg * dn;
    }
    sgg /= wSum; snn /= wSum; sgn /= wSum;

    const det = sgg * snn - sgn * sgn;
    const useNS = det > 1e-10 && snn > 1e-6;

    const result = new Array(6);
    for (let c = 0; c < 6; c++) {
      let pMean = 0;
      for (let i = 0; i < data.length; i++) pMean += ws[i] * data[i].dist[c];
      pMean /= wSum;

      if (sgg < 1e-10) { result[c] = Math.max(0, pMean); continue; }

      // Covariance of features with target
      let sgp = 0, snp = 0;
      for (let i = 0; i < data.length; i++) {
        sgp += ws[i] * (data[i].g - gMean) * (data[i].dist[c] - pMean);
        snp += ws[i] * (data[i].ns - nsMean) * (data[i].dist[c] - pMean);
      }
      sgp /= wSum; snp /= wSum;

      let pred;
      if (useNS) {
        // 2D regression: solve [sgg sgn; sgn snn] * [b1; b2] = [sgp; snp]
        const b1 = (snn * sgp - sgn * snp) / det;
        const b2 = (sgg * snp - sgn * sgp) / det;
        pred = pMean + b1 * (targetGrowth - gMean) + b2 * (testNSett - nsMean);
      } else {
        // 1D regression (growth only)
        pred = pMean + (sgp / sgg) * (targetGrowth - gMean);
      }
      result[c] = Math.max(0, pred);
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
