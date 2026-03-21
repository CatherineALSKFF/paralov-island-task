const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const allRoundNums = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Build growth-weighted models at multiple bandwidths for ensemble
  const lambdas = [8, 15, 30];
  const weightSets = lambdas.map(lambda => {
    const w = {};
    let ws = 0;
    for (const r of allRoundNums) {
      const diff = Math.abs((growthRates[String(r)] || 0.15) - targetGrowth);
      const v = Math.exp(-lambda * diff);
      w[r] = v;
      ws += v;
    }
    for (const r of allRoundNums) w[r] /= ws;
    return w;
  });

  // Build per-key weighted probability models for each bandwidth
  function buildWeightedModel(weights) {
    const model = {};
    for (const r of allRoundNums) {
      const buckets = perRoundBuckets[r];
      if (!buckets) continue;
      for (const key in buckets) {
        const b = buckets[key];
        if (!b || !b.sum) continue;
        const n = b.count || 1;
        if (!model[key]) model[key] = { wsum: new Float64Array(6), wtotal: 0, count: 0 };
        const m = model[key];
        for (let c = 0; c < 6; c++) m.wsum[c] += weights[r] * (b.sum[c] / n);
        m.wtotal += weights[r];
        m.count += n;
      }
    }
    const probs = {};
    for (const key in model) {
      const m = model[key];
      if (m.wtotal > 0) {
        probs[key] = new Array(6);
        for (let c = 0; c < 6; c++) probs[key][c] = m.wsum[c] / m.wtotal;
        probs[key].count = m.count;
      }
    }
    return probs;
  }

  const models = weightSets.map(w => buildWeightedModel(w));

  // Uniform all-rounds model as final fallback
  const uniformW = {};
  for (const r of allRoundNums) uniformW[r] = 1 / allRoundNums.length;
  const uniformModel = buildWeightedModel(uniformW);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const fineKey = getFeatureKey(initGrid, settPos, y, x);
      const coarseKey = fineKey.slice(0, -1);

      // Ensemble: average predictions across bandwidths
      const ensemble = new Float64Array(6);
      let ensembleN = 0;

      for (const model of models) {
        let p = null;
        let pCount = 0;

        // Try fine key
        if (model[fineKey]) {
          p = model[fineKey];
          pCount = p.count || 1;
        }

        // Blend fine + coarse based on sample count (regularization)
        if (p && model[coarseKey] && pCount < 50) {
          const blendW = Math.min(pCount / 50, 1);
          const coarse = model[coarseKey];
          const blended = new Array(6);
          for (let c = 0; c < 6; c++) blended[c] = blendW * p[c] + (1 - blendW) * coarse[c];
          p = blended;
        }

        // Fall back to coarse
        if (!p && model[coarseKey]) p = model[coarseKey];

        // Fall back to uniform model
        if (!p) p = uniformModel[fineKey] || uniformModel[coarseKey] || null;

        if (p) {
          for (let c = 0; c < 6; c++) ensemble[c] += p[c];
          ensembleN++;
        }
      }

      let prior;
      if (ensembleN > 0) {
        prior = new Array(6);
        for (let c = 0; c < 6; c++) prior[c] = ensemble[c] / ensembleN;
      } else {
        prior = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      // Adaptive floor: low for confident (static) cells, higher for uncertain
      let entropy = 0;
      for (let c = 0; c < 6; c++) {
        if (prior[c] > 1e-6) entropy -= prior[c] * Math.log(prior[c]);
      }
      const adaptiveFloor = floor * (0.05 + 0.95 * Math.pow(entropy / Math.log(6), 1.5));

      const floored = prior.map(v => Math.max(v, adaptiveFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };