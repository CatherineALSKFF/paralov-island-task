const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const sigma = config.sigma || 0.06;
  const floor = config.FLOOR || 0.0001;
  const shrinkage = config.shrinkage || 0.3;
  const minCount = config.minCount || 30;

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

  // Per-round average distributions for each key
  const perRoundAvg = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    perRoundAvg[rn] = {};
    for (const [key, val] of Object.entries(b)) {
      perRoundAvg[rn][key] = val.sum.map(v => v / val.count);
    }
  }

  // Gaussian weights by growth rate similarity
  const roundWeights = {};
  let tw = 0;
  for (const rn of allRounds) {
    const dist = Math.abs((growthRates[String(rn)] || 0.15) - targetGrowth);
    roundWeights[rn] = Math.exp(-dist * dist / (2 * sigma * sigma));
    tw += roundWeights[rn];
  }
  for (const rn of allRounds) roundWeights[rn] /= tw;

  // Build coarse (terrain-type only) model for shrinkage target
  const coarseModel = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    for (const [key, val] of Object.entries(b)) {
      const terrainKey = key[0]; // Just first char: P, F, S, O, M
      if (!coarseModel[terrainKey]) coarseModel[terrainKey] = { count: 0, sum: [0,0,0,0,0,0] };
      const avg = val.sum.map(v => v / val.count);
      const w = roundWeights[rn];
      coarseModel[terrainKey].count += w * val.count;
      for (let c = 0; c < 6; c++) coarseModel[terrainKey].sum[c] += w * avg[c] * val.count;
    }
  }
  const coarseDist = {};
  for (const [k, v] of Object.entries(coarseModel)) {
    coarseDist[k] = v.sum.map(s => s / v.count);
  }

  // Geometric mean ensemble: for each key, compute weighted geometric mean across rounds
  // This is optimal for minimizing expected KL divergence
  function getGeometricMeanDist(key) {
    let hasAny = false;
    const logSum = [0,0,0,0,0,0];
    let wSum = 0;
    let totalDataCount = 0;

    for (const rn of allRounds) {
      if (!perRoundAvg[rn] || !perRoundAvg[rn][key]) continue;
      const dist = perRoundAvg[rn][key];
      const w = roundWeights[rn];
      const b = perRoundBuckets[String(rn)][key];
      totalDataCount += b ? b.count : 0;
      for (let c = 0; c < 6; c++) {
        logSum[c] += w * Math.log(Math.max(dist[c], 1e-8));
      }
      wSum += w;
      hasAny = true;
    }
    if (!hasAny) return null;

    // Geometric mean
    const geo = logSum.map(v => Math.exp(v / wSum));
    const s = geo.reduce((a, b) => a + b, 0);
    return { dist: geo.map(v => v / s), count: totalDataCount };
  }

  // Weighted arithmetic mean (standard approach)
  function getArithmeticMeanDist(key) {
    const result = [0,0,0,0,0,0];
    let wSum = 0;
    let totalCount = 0;
    for (const rn of allRounds) {
      if (!perRoundAvg[rn] || !perRoundAvg[rn][key]) continue;
      const dist = perRoundAvg[rn][key];
      const w = roundWeights[rn];
      const b = perRoundBuckets[String(rn)][key];
      totalCount += b ? b.count : 0;
      for (let c = 0; c < 6; c++) result[c] += w * dist[c];
      wSum += w;
    }
    if (wSum === 0) return null;
    return { dist: result.map(v => v / wSum), count: totalCount };
  }

  // Lookup with hierarchical fallback + shrinkage
  function lookupDist(fineKey, coarseKey) {
    // Try fine key first
    let fineGeo = getGeometricMeanDist(fineKey);
    let fineArith = getArithmeticMeanDist(fineKey);

    // Try coarse key
    let coarseGeo = getGeometricMeanDist(coarseKey);
    let coarseArith = getArithmeticMeanDist(coarseKey);

    // Blend geometric and arithmetic means (geometric is better for KL,
    // but arithmetic is more robust with sparse data)
    function blend(geo, arith) {
      if (!geo || !arith) return geo || arith;
      const geoWeight = 0.6;
      return {
        dist: geo.dist.map((v, c) => geoWeight * v + (1 - geoWeight) * arith.dist[c]),
        count: geo.count
      };
    }

    let fineDist = blend(fineGeo, fineArith);
    let coarseDist2 = blend(coarseGeo, coarseArith);

    if (fineDist && coarseDist2 && fineKey !== coarseKey) {
      // Shrink fine toward coarse proportional to data sparsity
      const lambda = Math.min(shrinkage, shrinkage * minCount / Math.max(fineDist.count, 1));
      return fineDist.dist.map((v, c) => (1 - lambda) * v + lambda * coarseDist2.dist[c]);
    }

    if (fineDist) return fineDist.dist;
    if (coarseDist2) return coarseDist2.dist;

    // Progressive fallback
    let fb = coarseKey;
    while (fb.length > 1) {
      fb = fb.slice(0, -1);
      const d = getArithmeticMeanDist(fb);
      if (d) return d.dist;
    }

    // Terrain type prior
    const terrainKey = coarseKey[0];
    if (coarseDist[terrainKey]) return [...coarseDist[terrainKey]];

    return [1/6,1/6,1/6,1/6,1/6,1/6];
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const enhKey = getEnhancedKey(y, x);
      const coarseKey = getFeatureKey(initGrid, settPos, y, x);
      let prior = lookupDist(enhKey, coarseKey);

      // Adaptive floor: lower for near-static cells, higher for dynamic cells
      let entropy = 0;
      for (let c = 0; c < 6; c++) if (prior[c] > 0.001) entropy -= prior[c] * Math.log(prior[c]);
      const cellFloor = entropy > 0.8 ? floor * 2 : entropy > 0.3 ? floor : floor * 0.1;

      const floored = prior.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
