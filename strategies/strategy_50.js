const { H, W, getFeatureKey } = require('./shared');

// Growth-weighted merge preserving cell counts (unlike strategy_1 which averaged per-round averages)
function growthWeightedMerge(perRoundBuckets, trainRounds, growthRates, targetGrowth, sigma) {
  const model = {};
  for (const r of trainRounds) {
    const g = growthRates[String(r)] || 0.15;
    const d = targetGrowth - g;
    const w = Math.exp(-d * d / (2 * sigma * sigma));
    const b = perRoundBuckets[String(r)];
    if (!b) continue;
    for (const [key, val] of Object.entries(b)) {
      if (!model[key]) model[key] = { count: 0, sum: [0,0,0,0,0,0] };
      model[key].count += w * val.count;
      for (let c = 0; c < 6; c++) model[key].sum[c] += w * val.sum[c];
    }
  }
  const out = {};
  for (const [k, v] of Object.entries(model)) out[k] = v.sum.map(s => s / v.count);
  return out;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const sigma = config.SIGMA || 0.05;
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const adaptiveModel = growthWeightedMerge(perRoundBuckets, allRounds, growthRates, targetGrowth, sigma);
  // Wide sigma fallback
  const fallbackModel = growthWeightedMerge(perRoundBuckets, allRounds, growthRates, targetGrowth, 0.3);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      let prior = adaptiveModel[key] ? [...adaptiveModel[key]] : null;
      if (!prior) {
        const fb = key.slice(0, -1);
        prior = adaptiveModel[fb] ? [...adaptiveModel[fb]] : fallbackModel[key] ? [...fallbackModel[key]] : fallbackModel[fb] ? [...fallbackModel[fb]] : [1/6,1/6,1/6,1/6,1/6,1/6];
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
