const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 1e-6;
  const sigmaNarrow = config.SIGMA_N || 0.045;
  const sigmaWide = config.SIGMA_W || 0.16;
  const alpha = config.ALPHA || 1;
  const regWeight = config.REG || 0.20;
  const staticHedge = config.HEDGE || 0.10;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const trainRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Gaussian weight by growth rate similarity
  function gaussWeight(r, sigma) {
    const gDiff = (growthRates[String(r)] || 0.15) - targetGrowth;
    return Math.exp(-gDiff * gDiff / (2 * sigma * sigma));
  }

  // Build a cell-count-weighted model from given round weight function
  function buildModel(weightFn) {
    const weights = {};
    let wTotal = 0;
    for (const r of trainRounds) {
      weights[r] = weightFn(r);
      wTotal += weights[r];
    }
    if (wTotal > 0) for (const r of trainRounds) weights[r] /= wTotal;

    const mSum = {}, mCount = {};
    for (const r of trainRounds) {
      const buckets = perRoundBuckets[r];
      if (!buckets) continue;
      const w = weights[r];
      if (w < 1e-15) continue;
      for (const key in buckets) {
        const b = buckets[key];
        if (!mSum[key]) { mSum[key] = new Float64Array(6); mCount[key] = 0; }
        for (let c = 0; c < 6; c++) mSum[key][c] += w * b.sum[c];
        mCount[key] += w * b.count;
      }
    }
    const probs = {}, effN = {};
    for (const key in mSum) {
      const cnt = mCount[key];
      if (cnt > 0) {
        probs[key] = new Array(6);
        for (let c = 0; c < 6; c++) probs[key][c] = mSum[key][c] / cnt;
        effN[key] = cnt;
      }
    }
    return { probs, effN };
  }

  // Narrow model: tight growth-rate bandwidth
  const narrow = buildModel(r => gaussWeight(r, sigmaNarrow));

  // Wide model: broad bandwidth (Dirichlet prior)
  const wide = buildModel(r => gaussWeight(r, sigmaWide));

  // Dirichlet-smoothed prediction:
  // q[c] = (effN * narrow[c] + alpha * wide[c]) / (effN + alpha)
  function getSmoothedProb(key) {
    const nP = narrow.probs[key];
    const wP = wide.probs[key];
    const effN = narrow.effN[key] || 0;

    if (!nP && !wP) return null;
    if (!nP) return wP;
    if (!wP) return nP;

    const result = new Array(6);
    const denom = effN + alpha;
    for (let c = 0; c < 6; c++) {
      result[c] = (effN * nP[c] + alpha * wP[c]) / denom;
    }
    return result;
  }

  // Generate predictions
  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      // Hedge for static cell types (mountain/ocean): in case GT has dynamic
      // behavior not in training data. Free for truly static cells (entropy=0).
      if (key === 'O' || key === 'M') {
        const staticClass = key === 'O' ? 0 : 5;
        const result = new Array(6).fill(staticHedge);
        result[staticClass] = 1 - 5 * staticHedge;
        row.push(result);
        continue;
      }

      const fb = key.slice(0, -1);
      const fine = getSmoothedProb(key);
      const coarse = getSmoothedProb(fb);

      let prior;
      if (fine && coarse) {
        const fineN = narrow.effN[key] || 0;
        const adaptReg = regWeight * Math.max(0.3, 1 - Math.min(fineN / 200, 1) * 0.7);
        prior = new Array(6);
        for (let c = 0; c < 6; c++) prior[c] = (1 - adaptReg) * fine[c] + adaptReg * coarse[c];
      } else {
        prior = fine || coarse || [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      // Adaptive floor: minimal for confident, higher for uncertain
      const ent = -prior.reduce((s, p) => s + (p > 1e-10 ? p * Math.log(p) : 0), 0);
      const aFloor = floor * (0.1 + 0.9 * ent / Math.log(6));

      const floored = prior.map(v => Math.max(v, aFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
