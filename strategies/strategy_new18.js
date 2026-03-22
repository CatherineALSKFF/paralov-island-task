const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const sigma = config.sigma || 0.05;
  const floor = config.FLOOR || 0.0001;
  const varReg = config.varReg || 0.5;  // variance-based regularization strength

  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Precompute nearest settlement distance (Chebyshev)
  const nearestDist = Array.from({ length: H }, () => Array(W).fill(99));
  for (const s of settlements)
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
        if (d < nearestDist[y][x]) nearestDist[y][x] = d;
      }

  // Enhanced feature key matching bucket structure
  function getEnhancedKey(y, x) {
    const coarseKey = getFeatureKey(initGrid, settPos, y, x);
    if (coarseKey === 'O' || coarseKey === 'M') return coarseKey;
    if (coarseKey[0] === 'S') return coarseKey;
    const nKey = coarseKey[1];
    const minDist = nearestDist[y][x];
    if (nKey === '0') return coarseKey + (minDist === 4 ? 'n' : minDist <= 8 ? 'm' : 'f');
    if (nKey === '1') {
      if (minDist === 1) return coarseKey + 'a';
      if (minDist === 2) return coarseKey + 'b';
    }
    return coarseKey;
  }

  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Gaussian weights by growth rate similarity
  const roundWeights = {};
  let tw = 0;
  for (const rn of allRounds) {
    const dist = Math.abs((growthRates[String(rn)] || 0.15) - targetGrowth);
    roundWeights[rn] = Math.exp(-dist * dist / (2 * sigma * sigma));
    tw += roundWeights[rn];
  }
  for (const rn of allRounds) roundWeights[rn] /= tw;

  // Per-round averages
  const perRoundAvg = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    perRoundAvg[rn] = {};
    for (const [key, val] of Object.entries(b)) {
      perRoundAvg[rn][key] = val.sum.map(v => v / val.count);
    }
  }

  // Compute weighted mean AND weighted variance for each key
  const meanModel = {};
  const varModel = {};
  for (const key of new Set(allRounds.flatMap(rn => Object.keys(perRoundBuckets[String(rn)] || {})))) {
    const mean = [0,0,0,0,0,0];
    let wSum = 0;
    for (const rn of allRounds) {
      if (!perRoundAvg[rn] || !perRoundAvg[rn][key]) continue;
      const w = roundWeights[rn];
      for (let c = 0; c < 6; c++) mean[c] += w * perRoundAvg[rn][key][c];
      wSum += w;
    }
    if (wSum === 0) continue;
    for (let c = 0; c < 6; c++) mean[c] /= wSum;
    meanModel[key] = mean;

    // Weighted variance
    const variance = [0,0,0,0,0,0];
    for (const rn of allRounds) {
      if (!perRoundAvg[rn] || !perRoundAvg[rn][key]) continue;
      const w = roundWeights[rn];
      for (let c = 0; c < 6; c++) {
        const diff = perRoundAvg[rn][key][c] - mean[c];
        variance[c] += w * diff * diff;
      }
    }
    for (let c = 0; c < 6; c++) variance[c] /= wSum;
    varModel[key] = variance;
  }

  // Global terrain-level model (for shrinkage target)
  const terrainModel = {};
  for (const [key, mean] of Object.entries(meanModel)) {
    const tk = key[0];
    if (!terrainModel[tk]) terrainModel[tk] = { sum: [0,0,0,0,0,0], n: 0 };
    for (let c = 0; c < 6; c++) terrainModel[tk].sum[c] += mean[c];
    terrainModel[tk].n++;
  }
  const terrainDist = {};
  for (const [tk, v] of Object.entries(terrainModel)) {
    terrainDist[tk] = v.sum.map(s => s / v.n);
  }

  function lookupKey(fineKey, coarseKey) {
    // Try fine key first, then coarse
    let mean = meanModel[fineKey] || meanModel[coarseKey];
    let variance = varModel[fineKey] || varModel[coarseKey];

    if (!mean) {
      // Progressive fallback
      let fb = coarseKey;
      while (fb.length > 1) {
        fb = fb.slice(0, -1);
        if (meanModel[fb]) { mean = meanModel[fb]; variance = varModel[fb]; break; }
      }
    }
    if (!mean) {
      const tk = coarseKey[0];
      return terrainDist[tk] || [1/6,1/6,1/6,1/6,1/6,1/6];
    }

    // Variance-based regularization:
    // High variance → shrink toward terrain-level prior
    if (variance) {
      const totalVar = variance.reduce((a, b) => a + b, 0);
      const tk = coarseKey[0];
      const prior = terrainDist[tk] || mean;
      // Shrinkage strength proportional to variance
      const lambda = Math.min(0.5, varReg * Math.sqrt(totalVar));
      return mean.map((v, c) => (1 - lambda) * v + lambda * prior[c]);
    }

    return mean;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const enhKey = getEnhancedKey(y, x);
      const coarseKey = getFeatureKey(initGrid, settPos, y, x);
      let prior = lookupKey(enhKey, coarseKey);

      // Adaptive floor
      let entropy = 0;
      for (let c = 0; c < 6; c++) if (prior[c] > 0.001) entropy -= prior[c] * Math.log(prior[c]);
      const cellFloor = entropy > 0.5 ? floor * 2 : floor;

      const floored = prior.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
