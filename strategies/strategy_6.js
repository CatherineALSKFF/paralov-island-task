const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const candidates = { ...growthRates };
  delete candidates[String(testRound)];

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const allRoundNums = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Compute growth-rate similarity weights (exponential decay)
  const lambda = 12.0;
  const roundWeights = {};
  for (const r of allRoundNums) {
    const g = growthRates[String(r)];
    if (g === undefined) continue;
    const dist = Math.abs(g - targetGrowth);
    roundWeights[r] = Math.exp(-lambda * dist);
  }

  // Build weighted model: merge buckets with growth-similarity weights
  function buildWeightedModel(rounds, weights) {
    const model = {};
    for (const r of rounds) {
      const w = weights[r] || 0;
      if (w < 0.001) continue;
      const buckets = perRoundBuckets[r];
      if (!buckets) continue;
      for (const key in buckets) {
        const b = buckets[key];
        if (!model[key]) {
          model[key] = { wsum: new Float64Array(6), wtotal: 0 };
        }
        for (let c = 0; c < 6; c++) {
          model[key].wsum[c] += w * (b.sum[c] / b.count);
        }
        model[key].wtotal += w;
      }
    }
    // Normalize
    const result = {};
    for (const key in model) {
      const m = model[key];
      result[key] = new Float64Array(6);
      for (let c = 0; c < 6; c++) {
        result[key][c] = m.wsum[c] / m.wtotal;
      }
    }
    return result;
  }

  const weightedModel = buildWeightedModel(allRoundNums, roundWeights);

  // Uniform model as fallback
  const uniformWeights = {};
  for (const r of allRoundNums) uniformWeights[r] = 1.0;
  const uniformModel = buildWeightedModel(allRoundNums, uniformWeights);

  // Also build a top-K model for ensemble
  const Ks = [3, 5, 7];
  const kModels = [];
  for (const K of Ks) {
    const closest = selectClosestRounds(candidates, targetGrowth, K);
    const kWeights = {};
    for (const r of closest) {
      const g = growthRates[String(r)];
      const dist = Math.abs(g - targetGrowth);
      kWeights[r] = Math.exp(-lambda * dist);
    }
    kModels.push(buildWeightedModel(closest, kWeights));
  }

  // Feature key hierarchy: full key, then progressively shorter
  function getKeyHierarchy(grid, sPos, y, x) {
    const full = getFeatureKey(grid, sPos, y, x);
    const keys = [full];
    // Try removing last character(s) for coarser features
    for (let i = full.length - 1; i >= 1; i--) {
      keys.push(full.slice(0, i));
    }
    return keys;
  }

  function lookupModel(model, keys) {
    for (const k of keys) {
      if (model[k]) return model[k];
    }
    return null;
  }

  // Regularization weight toward coarser keys
  const regWeight = 0.35;

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const keys = getKeyHierarchy(initGrid, settPos, y, x);

      // Collect predictions from multiple models and blend
      const preds = [];
      const predWeights = [];

      // 1. Weighted model (highest priority)
      const wPred = lookupModel(weightedModel, keys);
      if (wPred) { preds.push(wPred); predWeights.push(3.0); }

      // 2. K-nearest models
      for (const km of kModels) {
        const kPred = lookupModel(km, keys);
        if (kPred) { preds.push(kPred); predWeights.push(1.0); }
      }

      // 3. Uniform fallback
      const uPred = lookupModel(uniformModel, keys);
      if (uPred) { preds.push(uPred); predWeights.push(0.5); }

      // Coarse regularization: lookup with just first char of key
      const coarseKey = keys[keys.length - 1];
      const coarsePred = weightedModel[coarseKey] || uniformModel[coarseKey];

      let blended = new Float64Array(6);
      if (preds.length > 0) {
        let totalW = 0;
        for (let i = 0; i < preds.length; i++) totalW += predWeights[i];
        for (let c = 0; c < 6; c++) {
          for (let i = 0; i < preds.length; i++) {
            blended[c] += predWeights[i] * preds[i][c] / totalW;
          }
        }
      } else {
        blended = new Float64Array([1/6,1/6,1/6,1/6,1/6,1/6]);
      }

      // Regularize toward coarse
      if (coarsePred && preds.length > 0) {
        for (let c = 0; c < 6; c++) {
          blended[c] = (1 - regWeight) * blended[c] + regWeight * coarsePred[c];
        }
      }

      // Adaptive floor: lower for confident cells, higher for uncertain
      let maxP = 0;
      for (let c = 0; c < 6; c++) if (blended[c] > maxP) maxP = blended[c];
      const entropy = -blended.reduce((s, p) => s + (p > 0.001 ? p * Math.log(p) : 0), 0);
      const maxEntropy = Math.log(6);
      const entropyRatio = entropy / maxEntropy;
      // Confident cells (low entropy) get tiny floor, uncertain cells get larger floor
      const adaptiveFloor = floor * (0.1 + 0.9 * entropyRatio * entropyRatio);

      const floored = new Float64Array(6);
      let sum = 0;
      for (let c = 0; c < 6; c++) {
        floored[c] = Math.max(blended[c], adaptiveFloor);
        sum += floored[c];
      }
      const final = new Array(6);
      for (let c = 0; c < 6; c++) final[c] = floored[c] / sum;

      row.push(final);
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };