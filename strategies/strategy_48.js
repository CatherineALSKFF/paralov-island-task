const { H, W, getFeatureKey, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const K = config.K || 5;
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const candidates = { ...growthRates }; delete candidates[String(testRound)];
  const closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Build per-round models
  const roundModels = {};
  for (const r of allRounds) {
    roundModels[r] = {};
    const b = perRoundBuckets[String(r)];
    if (!b) continue;
    for (const [key, val] of Object.entries(b)) {
      roundModels[r][key] = val.sum.map(v => v / val.count);
    }
  }

  // Median-based merge for K closest rounds
  function medianMerge(rounds, key) {
    const vectors = [];
    for (const r of rounds) {
      if (roundModels[r] && roundModels[r][key]) vectors.push(roundModels[r][key]);
    }
    if (vectors.length === 0) return null;
    if (vectors.length === 1) return [...vectors[0]];
    // Per-class median
    const result = new Array(6);
    for (let c = 0; c < 6; c++) {
      const vals = vectors.map(v => v[c]).sort((a, b) => a - b);
      const mid = Math.floor(vals.length / 2);
      result[c] = vals.length % 2 === 1 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
    }
    return result;
  }

  // Mean-based merge for fallback (all rounds)
  function meanMerge(rounds, key) {
    let count = 0;
    const sum = [0,0,0,0,0,0];
    for (const r of rounds) {
      if (roundModels[r] && roundModels[r][key]) {
        const v = roundModels[r][key];
        for (let c = 0; c < 6; c++) sum[c] += v[c];
        count++;
      }
    }
    if (count === 0) return null;
    return sum.map(v => v / count);
  }

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const fb = key.slice(0, -1);

      // Try median of K closest, then mean of all, then fallback key
      let prior = medianMerge(closestRounds, key);
      if (!prior) prior = meanMerge(allRounds, key);
      if (!prior) prior = medianMerge(closestRounds, fb);
      if (!prior) prior = meanMerge(allRounds, fb);
      if (!prior) prior = [1/6,1/6,1/6,1/6,1/6,1/6];

      const floored = prior.map(v => Math.max(v, floor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}
module.exports = { predict };
