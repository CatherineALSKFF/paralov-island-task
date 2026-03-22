const { H, W, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function getEnhancedKey(grid, settPos, y, x) {
  const base = getFeatureKey(grid, settPos, y, x);
  if (base === 'O' || base === 'M' || base[0] === 'S' || base[1] !== '1') return base;

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

// Count-weighted Gaussian merge (preserves MLE property of mergeBuckets)
function gaussianMerge(perRoundBuckets, growthRates, targetGrowth, sigma, excludeRound) {
  const rounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== excludeRound);
  const weights = {};
  for (const r of rounds) {
    const diff = (growthRates[String(r)] || 0.15) - targetGrowth;
    weights[r] = Math.exp(-diff * diff / (2 * sigma * sigma));
  }

  const model = {};
  for (const r of rounds) {
    const b = perRoundBuckets[String(r)];
    if (!b) continue;
    const w = weights[r];
    for (const [key, val] of Object.entries(b)) {
      if (!model[key]) model[key] = { count: 0, sum: [0,0,0,0,0,0] };
      model[key].count += w * val.count;
      for (let c = 0; c < 6; c++) model[key].sum[c] += w * val.sum[c];
    }
  }

  const out = {};
  for (const [k, v] of Object.entries(model)) {
    if (v.count > 0) out[k] = v.sum.map(s => s / v.count);
  }
  return out;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const sigma = 0.045;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const adaptiveModel = gaussianMerge(perRoundBuckets, growthRates, targetGrowth, sigma, testRound);
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);
  const allModel = mergeBuckets(perRoundBuckets, allRounds);
  const regBlend = 0.08;

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getEnhancedKey(initGrid, settPos, y, x);

      let prior = adaptiveModel[key] ? [...adaptiveModel[key]] : allModel[key] ? [...allModel[key]] : null;
      if (!prior) {
        for (let len = key.length - 1; len >= 1 && !prior; len--) {
          const fb = key.slice(0, len);
          if (adaptiveModel[fb]) prior = [...adaptiveModel[fb]];
          else if (allModel[fb]) prior = [...allModel[fb]];
        }
        if (!prior) prior = [1/6,1/6,1/6,1/6,1/6,1/6];
      }

      // Mild blend toward all-rounds model
      const gk = allModel[key] || allModel[key.slice(0, -1)];
      if (gk) { for (let c = 0; c < 6; c++) prior[c] = (1 - regBlend) * prior[c] + regBlend * gk[c]; }
      const floored = prior.map(v => Math.max(v, floor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
