const { H, W, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const K = config.K || 4;
  const floor = config.FLOOR || 0.0001;
  const blend = config.BLEND || 0.25;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const candidates = { ...growthRates }; delete candidates[String(testRound)];
  const closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  const adaptiveModel = mergeBuckets(perRoundBuckets, closestRounds);
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);
  const allModel = mergeBuckets(perRoundBuckets, allRounds);
  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);
  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const fb = key.slice(0, -1);
      const adaptP = adaptiveModel[key] || adaptiveModel[fb] || null;
      const allP = allModel[key] || allModel[fb] || null;

      let prior;
      if (adaptP && allP) {
        // Blend K-nearest with all-rounds for regularization
        prior = adaptP.map((v, c) => (1 - blend) * v + blend * allP[c]);
      } else if (adaptP) {
        prior = [...adaptP];
      } else if (allP) {
        prior = [...allP];
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
