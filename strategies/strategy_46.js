const { H, W, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const K = config.K || 4;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const candidates = { ...growthRates }; delete candidates[String(testRound)];
  const closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  const adaptiveModel = mergeBuckets(perRoundBuckets, closestRounds);
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);
  const allModel = mergeBuckets(perRoundBuckets, allRounds);

  // Build per-round models for variance computation
  const roundModels = {};
  for (const r of closestRounds) {
    roundModels[r] = {};
    const b = perRoundBuckets[String(r)];
    if (!b) continue;
    for (const [key, val] of Object.entries(b)) {
      roundModels[r][key] = val.sum.map(v => v / val.count);
    }
  }

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      let prior = adaptiveModel[key] ? [...adaptiveModel[key]] : allModel[key] ? [...allModel[key]] : null;
      if (!prior) {
        const fb = key.slice(0, -1);
        prior = adaptiveModel[fb] ? [...adaptiveModel[fb]] : allModel[fb] ? [...allModel[fb]] : [1/6,1/6,1/6,1/6,1/6,1/6];
      }

      // Compute inter-round variance for this key
      let totalVar = 0;
      let nRounds = 0;
      for (const r of closestRounds) {
        const rp = roundModels[r][key] || roundModels[r][key.slice(0, -1)];
        if (!rp) continue;
        for (let c = 0; c < 6; c++) {
          const d = rp[c] - prior[c];
          totalVar += d * d;
        }
        nRounds++;
      }
      if (nRounds > 0) totalVar /= nRounds;

      // Adaptive floor: base 0.0001, increases with disagreement
      const cellFloor = 0.0001 + Math.min(0.005, Math.sqrt(totalVar) * 0.04);

      const floored = prior.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
