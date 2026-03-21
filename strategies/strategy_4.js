const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const regWeight = config.regWeight || 0.4;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const candidates = { ...growthRates };
  delete candidates[String(testRound)];
  const allRoundNums = Object.keys(candidates).map(Number);

  const lambdas = [8, 18, 40];
  const models = lambdas.map(lambda => {
    const weights = {};
    for (const r of allRoundNums) {
      const diff = Math.abs(growthRates[String(r)] - targetGrowth);
      weights[r] = Math.exp(-lambda * diff);
    }
    return buildWeightedModel(perRoundBuckets, allRoundNums, weights);
  });

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      const dist = [0, 0, 0, 0, 0, 0];
      const w = 1 / models.length;
      for (const model of models) {
        const d = lookupHierarchical(model, key, regWeight);
        for (let c = 0; c < 6; c++) dist[c] += d[c] * w;
      }

      let entropy = 0;
      for (let c = 0; c < 6; c++) {
        if (dist[c] > 1e-10) entropy -= dist[c] * Math.log(dist[c]);
      }
      const eRatio = entropy / Math.log(6);
      const cellFloor = floor * (0.05 + 2.0 * eRatio);

      const floored = dist.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

function buildWeightedModel(perRoundBuckets, roundNums, weights) {
  const model = {};
  for (const r of roundNums) {
    const buckets = perRoundBuckets[r];
    if (!buckets) continue;
    const w = weights[r];
    for (const fullKey in buckets) {
      const b = buckets[fullKey];
      if (b.count === 0) continue;
      const avg = b.sum.map(v => v / b.count);
      for (let len = 1; len <= fullKey.length; len++) {
        const k = fullKey.slice(0, len);
        if (!model[k]) model[k] = { totalW: 0, dist: [0, 0, 0, 0, 0, 0] };
        model[k].totalW += w;
        for (let c = 0; c < 6; c++) model[k].dist[c] += w * avg[c];
      }
    }
  }
  for (const k in model) {
    const m = model[k];
    if (m.totalW > 0) {
      for (let c = 0; c < 6; c++) m.dist[c] /= m.totalW;
    }
  }
  return model;
}

function lookupHierarchical(model, key, regWeight) {
  const levels = [];
  for (let len = key.length; len >= 1; len--) {
    const k = key.slice(0, len);
    if (model[k]) levels.push(model[k].dist);
  }
  if (levels.length === 0) return [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
  let result = [...levels[0]];
  for (let i = 1; i < levels.length; i++) {
    for (let c = 0; c < 6; c++) {
      result[c] = (1 - regWeight) * result[c] + regWeight * levels[i][c];
    }
  }
  return result;
}

module.exports = { predict };