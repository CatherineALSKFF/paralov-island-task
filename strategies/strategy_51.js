const { H, W, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const K = config.K || 4;
  const floor = config.FLOOR || 0.0001;
  const regBlend = config.REG_BLEND || 0.4;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Standard K-nearest model (baseline)
  const candidates = { ...growthRates }; delete candidates[String(testRound)];
  const closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  const adaptiveModel = mergeBuckets(perRoundBuckets, closestRounds);
  const allModel = mergeBuckets(perRoundBuckets, allRounds);

  // Build per-key regression data: growth_rate -> probability distribution
  const keyData = {};
  for (const r of allRounds) {
    const g = growthRates[String(r)];
    if (g === undefined) continue;
    const b = perRoundBuckets[String(r)];
    if (!b) continue;
    for (const [key, val] of Object.entries(b)) {
      if (!keyData[key]) keyData[key] = [];
      keyData[key].push({ g, dist: val.sum.map(v => v / val.count) });
    }
  }

  // Fit weighted local linear regression for each key
  // p_c(g) = a_c + b_c * (g - gMean), weighted by proximity to targetGrowth
  const regModel = {};
  const regSigma = 0.08;
  for (const [key, data] of Object.entries(keyData)) {
    if (data.length < 4) continue;

    // Growth-proximity weights
    const ws = data.map(d => { const diff = d.g - targetGrowth; return Math.exp(-diff * diff / (2 * regSigma * regSigma)); });
    const wSum = ws.reduce((a, b) => a + b, 0);

    // Weighted mean of growth
    let gMean = 0;
    for (let i = 0; i < data.length; i++) gMean += ws[i] * data[i].g;
    gMean /= wSum;

    // Weighted variance of growth
    let gVar = 0;
    for (let i = 0; i < data.length; i++) { const dg = data[i].g - gMean; gVar += ws[i] * dg * dg; }
    gVar /= wSum;

    if (gVar < 1e-10) continue;

    const result = new Array(6);
    for (let c = 0; c < 6; c++) {
      // Weighted mean of p_c
      let pMean = 0;
      for (let i = 0; i < data.length; i++) pMean += ws[i] * data[i].dist[c];
      pMean /= wSum;

      // Weighted covariance of (g, p_c)
      let cov = 0;
      for (let i = 0; i < data.length; i++) {
        cov += ws[i] * (data[i].g - gMean) * (data[i].dist[c] - pMean);
      }
      cov /= wSum;

      // Slope and prediction at targetGrowth
      const slope = cov / gVar;
      result[c] = Math.max(0, pMean + slope * (targetGrowth - gMean));
    }

    // Normalize
    const rSum = result.reduce((a, b) => a + b, 0);
    if (rSum > 0) {
      regModel[key] = result.map(v => v / rSum);
    }
  }

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const fb = key.slice(0, -1);

      // Get baseline prediction
      let basePrior = adaptiveModel[key] || adaptiveModel[fb] || allModel[key] || allModel[fb] || null;

      // Get regression prediction
      let regPrior = regModel[key] || regModel[fb] || null;

      let prior;
      if (basePrior && regPrior) {
        // Blend baseline with regression
        prior = basePrior.map((v, c) => (1 - regBlend) * v + regBlend * regPrior[c]);
      } else if (basePrior) {
        prior = [...basePrior];
      } else if (regPrior) {
        prior = [...regPrior];
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
