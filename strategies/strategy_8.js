const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const sigma = config.SIGMA || 0.06;
  const regWeight = config.REG_WEIGHT || 0.35;
  const minCount = config.MIN_COUNT || 8;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Gaussian-weighted round merging by growth rate similarity
  const roundWeights = {};
  let wTotal = 0;
  for (const r of allRounds) {
    const g = growthRates[String(r)] || 0.15;
    const d = Math.abs(g - targetGrowth);
    const w = Math.exp(-d * d / (2 * sigma * sigma));
    roundWeights[r] = w;
    wTotal += w;
  }
  for (const r of allRounds) roundWeights[r] /= wTotal;

  // Build weighted model + track raw counts for confidence
  const wModel = {};
  const rawCount = {};
  for (const r of allRounds) {
    const buckets = perRoundBuckets[r];
    if (!buckets) continue;
    const w = roundWeights[r];
    for (const key in buckets) {
      const b = buckets[key];
      if (!wModel[key]) { wModel[key] = [0,0,0,0,0,0]; rawCount[key] = 0; }
      rawCount[key] += b.count;
      for (let c = 0; c < 6; c++) wModel[key][c] += b.sum[c] * w;
    }
  }

  // Normalize weighted model to probabilities
  const wProbs = {};
  for (const key in wModel) {
    const s = wModel[key].reduce((a, b) => a + b, 0);
    if (s > 0) wProbs[key] = wModel[key].map(v => v / s);
  }

  // Uniform all-rounds model for fallback/regularization
  const uModel = {};
  for (const r of allRounds) {
    const buckets = perRoundBuckets[r];
    if (!buckets) continue;
    for (const key in buckets) {
      const b = buckets[key];
      if (!uModel[key]) uModel[key] = [0,0,0,0,0,0];
      for (let c = 0; c < 6; c++) uModel[key][c] += b.sum[c];
    }
  }
  const uProbs = {};
  for (const key in uModel) {
    const s = uModel[key].reduce((a, b) => a + b, 0);
    if (s > 0) uProbs[key] = uModel[key].map(v => v / s);
  }

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const fb = key.slice(0, -1);

      // Hierarchy: weighted specific → weighted coarse → uniform specific → uniform coarse
      const ws = wProbs[key], wc = wProbs[fb];
      const us = uProbs[key], uc = uProbs[fb];
      const rc = rawCount[key] || 0;

      let prior;
      if (ws) {
        // Blend specific with coarse based on sample count for regularization
        const confidence = Math.min(rc / minCount, 1);
        const effectiveReg = regWeight * (1 - 0.5 * confidence);
        const fallback = wc || uc || us;
        if (fallback) {
          prior = ws.map((v, i) => (1 - effectiveReg) * v + effectiveReg * fallback[i]);
        } else {
          prior = [...ws];
        }
      } else if (wc) {
        prior = [...wc];
      } else if (us) {
        const fallback = uc;
        if (fallback) {
          prior = us.map((v, i) => (1 - regWeight) * v + regWeight * fallback[i]);
        } else {
          prior = [...us];
        }
      } else if (uc) {
        prior = [...uc];
      } else {
        prior = [1/6,1/6,1/6,1/6,1/6,1/6];
      }

      // Per-cell adaptive floor: low-entropy cells get much smaller floor
      const ent = prior.reduce((e, p) => p > 0 ? e - p * Math.log(p) : e, 0);
      const normEnt = ent / Math.log(6);
      const cellFloor = floor * (0.05 + 0.95 * normEnt);

      const floored = prior.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };