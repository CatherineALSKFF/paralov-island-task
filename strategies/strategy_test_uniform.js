const { H, W, getFeatureKey, mergeBuckets } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);
  const model = mergeBuckets(perRoundBuckets, allRounds);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      let prior = model[key] ? [...model[key]] : null;
      if (!prior) {
        const fb = key.slice(0, -1);
        prior = model[fb] ? [...model[fb]] : [1/6,1/6,1/6,1/6,1/6,1/6];
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
