const { H, W, terrainToClass, getFeatureKey } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.00001;
  const regWeight = config.REG || 0.15;
  const lambda = config.LAMBDA || 40;
  const tempBase = config.TEMP || 1.15;
  const tempScale = config.TEMP_SCALE || 0.3;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const trainRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Exponential decay round weights by growth-rate similarity
  // Applied to raw cell counts (Bayesian count-weighting)
  const roundWeights = {};
  for (const r of trainRounds) {
    const g = growthRates[String(r)] || 0.15;
    roundWeights[r] = Math.exp(-lambda * Math.abs(g - targetGrowth));
  }

  // Build weighted-count models (fine + coarse keys)
  const fine = {}, coarse = {};
  // Track per-round distributions for variance computation
  const perRoundProbs = {};

  for (const r of trainRounds) {
    const buckets = perRoundBuckets[r];
    if (!buckets) continue;
    const w = roundWeights[r];
    perRoundProbs[r] = {};

    for (const key in buckets) {
      const b = buckets[key];

      // Per-round probability for variance
      const p = new Array(6);
      for (let c = 0; c < 6; c++) p[c] = b.sum[c] / b.count;
      perRoundProbs[r][key] = p;

      // Fine key weighted aggregate
      if (!fine[key]) fine[key] = { wC: 0, wS: new Float64Array(6) };
      fine[key].wC += b.count * w;
      for (let c = 0; c < 6; c++) fine[key].wS[c] += b.sum[c] * w;

      // Coarse key (drop coastal suffix)
      const ck = key.length > 1 ? key.slice(0, -1) : key;
      if (ck !== key) {
        if (!coarse[ck]) coarse[ck] = { wC: 0, wS: new Float64Array(6) };
        coarse[ck].wC += b.count * w;
        for (let c = 0; c < 6; c++) coarse[ck].wS[c] += b.sum[c] * w;
      }
    }
  }

  // Normalize to probabilities
  const fP = {}, cP = {};
  for (const k in fine) {
    if (fine[k].wC > 0) {
      fP[k] = new Array(6);
      for (let c = 0; c < 6; c++) fP[k][c] = fine[k].wS[c] / fine[k].wC;
    }
  }
  for (const k in coarse) {
    if (coarse[k].wC > 0) {
      cP[k] = new Array(6);
      for (let c = 0; c < 6; c++) cP[k][c] = coarse[k].wS[c] / coarse[k].wC;
    }
  }

  // Compute cross-round variance per key (weighted)
  const keyVariance = {};
  for (const key in fP) {
    let v = 0, wSum = 0;
    for (const r of trainRounds) {
      if (!perRoundProbs[r] || !perRoundProbs[r][key]) continue;
      const w = roundWeights[r];
      wSum += w;
      for (let c = 0; c < 6; c++) {
        const d = perRoundProbs[r][key][c] - fP[key][c];
        v += w * d * d;
      }
    }
    keyVariance[key] = wSum > 0 ? v / wSum : 0;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const ck = key.length > 1 ? key.slice(0, -1) : null;

      const fineD = fP[key];
      const coarseD = ck ? (cP[ck] || fP[ck]) : null;

      let prior;
      if (fineD && coarseD) {
        prior = new Array(6);
        for (let c = 0; c < 6; c++)
          prior[c] = (1 - regWeight) * fineD[c] + regWeight * coarseD[c];
      } else {
        prior = fineD || coarseD || [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      // Variance-aware temperature scaling
      // Softens overconfident predictions, especially for high-variance keys
      const v = keyVariance[key] || 0;
      const temp = tempBase + tempScale * Math.sqrt(v);

      if (temp > 1.001) {
        let sum = 0;
        for (let c = 0; c < 6; c++) {
          prior[c] = Math.pow(Math.max(prior[c], 1e-10), 1 / temp);
          sum += prior[c];
        }
        for (let c = 0; c < 6; c++) prior[c] /= sum;
      }

      // Floor and normalize
      let sum = 0;
      const result = new Array(6);
      for (let c = 0; c < 6; c++) {
        result[c] = Math.max(prior[c], floor);
        sum += result[c];
      }
      for (let c = 0; c < 6; c++) result[c] /= sum;
      row.push(result);
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
