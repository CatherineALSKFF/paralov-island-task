const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const Ks = config.KS || [2, 3, 4, 5];
  const ensW = config.ENS_W || [0.2, 0.35, 0.3, 0.15]; // ensemble weights

  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const candidates = { ...growthRates };
  delete candidates[String(testRound)];

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Build models for each K value
  const models = [];
  for (const K of Ks) {
    const closestRounds = selectClosestRounds(candidates, targetGrowth, K);
    models.push(mergeBuckets(perRoundBuckets, closestRounds));
  }

  // All-rounds fallback
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);
  const allModel = mergeBuckets(perRoundBuckets, allRounds);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const ensemble = [0, 0, 0, 0, 0, 0];

      for (let i = 0; i < Ks.length; i++) {
        let dist = models[i][key];
        if (!dist) {
          const fb = key.slice(0, -1);
          dist = models[i][fb] || allModel[key] || allModel[fb] || null;
        }
        if (!dist) dist = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
        for (let c = 0; c < 6; c++) ensemble[c] += ensW[i] * dist[c];
      }

      const floored = ensemble.map(v => Math.max(v, floor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
