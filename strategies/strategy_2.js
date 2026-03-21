Looking at the scores, R13 (4.5) and R14 (5.8) are catastrophic — likely due to hard K-selection picking wrong rounds and feature key misses falling back to uniform `[1/6,...]`. Key improvements: exponential decay weighting, multi-bandwidth ensemble, hierarchical fallback blending, and regularization toward the global model.

const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  function computeWeights(sigma) {
    const w = {};
    let sum = 0;
    for (const r of allRounds) {
      const d = growthRates[String(r)] - targetGrowth;
      w[r] = Math.exp(-d * d / (2 * sigma * sigma));
      sum += w[r];
    }
    for (const r of allRounds) w[r] /= sum;
    return w;
  }

  function buildModel(weights) {
    const model = {};
    for (const r of allRounds) {
      const buckets = perRoundBuckets[r];
      if (!buckets) continue;
      const w = weights[r];
      for (const key in buckets) {
        if (!model[key]) model[key] = [0, 0, 0, 0, 0, 0];
        const b = buckets[key];
        for (let c = 0; c < 6; c++) model[key][c] += w * b.sum[c] / b.count;
      }
    }
    for (const key in model) {
      const s = model[key].reduce((a, b) => a + b, 0);
      if (s > 0) for (let c = 0; c < 6; c++) model[key][c] /= s;
    }
    return model;
  }

  const sigmas = [0.03, 0.06, 0.12, 0.24];
  const ensModels = sigmas.map(s => buildModel(computeWeights(s)));

  const uniW = {};
  for (const r of allRounds) uniW[r] = 1 / allRounds.length;
  const uniModel = buildModel(uniW);

  const regW = 0.25;

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      // Hierarchical fallback: try full key, then progressively shorter
      const keys = [key];
      for (let k = key.length - 1; k >= 1; k--) keys.push(key.slice(0, k));

      // Ensemble average with hierarchical lookup per model
      const avg = [0, 0, 0, 0, 0, 0];
      let hitCount = 0;
      for (const model of ensModels) {
        for (const k of keys) {
          if (model[k]) {
            for (let c = 0; c < 6; c++) avg[c] += model[k][c];
            hitCount++;
            break;
          }
        }
      }

      // Uniform model with same hierarchical fallback
      let uni = null;
      for (const k of keys) {
        if (uniModel[k]) { uni = uniModel[k]; break; }
      }

      let dist;
      if (hitCount > 0) {
        for (let c = 0; c < 6; c++) avg[c] /= hitCount;
        if (uni) {
          dist = avg.map((v, c) => (1 - regW) * v + regW * uni[c]);
        } else {
          dist = avg;
        }
      } else if (uni) {
        dist = [...uni];
      } else {
        dist = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6];
      }

      // Adaptive floor: higher for uncertain predictions
      let ent = 0;
      for (let c = 0; c < 6; c++) {
        if (dist[c] > 1e-10) ent -= dist[c] * Math.log(dist[c]);
      }
      const eRatio = ent / Math.log(6);
      const cellFloor = floor + 0.004 * eRatio;

      const floored = dist.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };