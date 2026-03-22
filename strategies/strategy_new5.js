const { H, W, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function getEnhancedKey(grid, settPos, y, x) {
  const base = getFeatureKey(grid, settPos, y, x);
  if (base === 'O' || base === 'M') return base;
  if (base[0] === 'S') return base;
  if (base[1] !== '1') return base;

  let minSDist = 999;
  for (let dy = -3; dy <= 3; dy++) {
    for (let dx = -3; dx <= 3; dx++) {
      if (!dy && !dx) continue;
      const ny = y + dy, nx = x + dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx)) {
        const d = Math.max(Math.abs(dy), Math.abs(dx));
        if (d < minSDist) minSDist = d;
      }
    }
  }
  if (minSDist === 1) return base + 'a';
  if (minSDist === 2) return base + 'b';
  return base;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const K = config.K || 4;
  const floor = config.FLOOR || 0.0001;
  const fineWeight = 0.6; // blend fine sub-key with parent for stability
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
      const key = getEnhancedKey(initGrid, settPos, y, x);
      const baseKey = getFeatureKey(initGrid, settPos, y, x);

      let prior;
      if (key !== baseKey) {
        // Enhanced sub-key: blend with parent for regularization
        const fine = adaptiveModel[key] || allModel[key];
        const coarse = adaptiveModel[baseKey] || allModel[baseKey];
        if (fine && coarse) {
          prior = fine.map((v, c) => fineWeight * v + (1 - fineWeight) * coarse[c]);
        } else if (fine) {
          prior = [...fine];
        } else if (coarse) {
          prior = [...coarse];
        }
      } else {
        prior = adaptiveModel[key] ? [...adaptiveModel[key]] : allModel[key] ? [...allModel[key]] : null;
      }

      if (!prior) {
        const fb = baseKey.slice(0, -1);
        prior = adaptiveModel[fb] ? [...adaptiveModel[fb]] : allModel[fb] ? [...allModel[fb]] : [1/6,1/6,1/6,1/6,1/6,1/6];
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
