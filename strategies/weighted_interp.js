const { H, W, terrainToClass, getFeatureKey, mergeBuckets } = require('./shared');

/**
 * Weighted interpolation strategy:
 * Instead of picking K nearest rounds (hard cutoff), weight ALL rounds
 * by exponential decay of growth rate distance. Closer growth = more weight.
 * This avoids the sharp transitions when K-nearest changes.
 *
 * Also adds: per-cell entropy-adaptive floor (confident cells get lower floor).
 */
function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const sigma = config.sigma || 0.05; // controls how sharply to weight by growth similarity
  const baseFloor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  // Build weighted model: each round contributes proportional to exp(-dist^2 / (2*sigma^2))
  const roundNums = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);
  const weights = {};
  let totalWeight = 0;
  for (const rn of roundNums) {
    const dist = Math.abs((growthRates[String(rn)] || 0.15) - targetGrowth);
    const w = Math.exp(-dist * dist / (2 * sigma * sigma));
    weights[rn] = w;
    totalWeight += w;
  }

  // Merge buckets with weights
  const model = {};
  for (const rn of roundNums) {
    const w = weights[rn] / totalWeight;
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    for (const [key, val] of Object.entries(b)) {
      if (!model[key]) model[key] = { count: 0, sum: [0, 0, 0, 0, 0, 0] };
      const avg = val.sum.map(v => v / val.count);
      model[key].count += w * val.count;
      for (let c = 0; c < 6; c++) model[key].sum[c] += w * avg[c] * val.count;
    }
  }

  // Normalize model
  const normalizedModel = {};
  for (const [key, val] of Object.entries(model)) {
    normalizedModel[key] = val.sum.map(v => v / val.count);
  }

  // Also build all-rounds fallback (unweighted)
  const allModel = {};
  for (const rn of roundNums) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    for (const [key, val] of Object.entries(b)) {
      if (!allModel[key]) allModel[key] = { count: 0, sum: [0, 0, 0, 0, 0, 0] };
      allModel[key].count += val.count;
      for (let c = 0; c < 6; c++) allModel[key].sum[c] += val.sum[c];
    }
  }
  const allNorm = {};
  for (const [key, val] of Object.entries(allModel)) {
    allNorm[key] = val.sum.map(v => v / val.count);
  }

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      let prior = normalizedModel[key] || allNorm[key] || null;
      if (!prior) {
        const fb = key.slice(0, -1);
        prior = normalizedModel[fb] || allNorm[fb] || [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      // Entropy-adaptive floor: confident predictions get lower floor
      const maxP = Math.max(...prior);
      const floor = maxP > 0.9 ? baseFloor * 0.1 : maxP > 0.7 ? baseFloor * 0.5 : baseFloor;

      const floored = prior.map(v => Math.max(v, floor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
