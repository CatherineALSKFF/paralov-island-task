const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const regWeight = config.REG_WEIGHT || 0.38;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Soft Gaussian-weighted merge over ALL rounds (no hard K cutoff)
  function buildWeightedModel(bw) {
    const weights = {};
    for (const r of allRounds) {
      const diff = Math.abs((growthRates[String(r)] || 0.15) - targetGrowth);
      weights[r] = Math.exp(-diff * diff / (2 * bw * bw));
    }
    const allKeys = new Set();
    for (const r of allRounds) {
      const rb = perRoundBuckets[r];
      if (rb) for (const k of Object.keys(rb)) allKeys.add(k);
    }
    const model = {};
    for (const key of allKeys) {
      const prob = [0, 0, 0, 0, 0, 0];
      let wSum = 0;
      for (const r of allRounds) {
        const rb = perRoundBuckets[r];
        if (!rb || !rb[key]) continue;
        const v = rb[key];
        const w = weights[r];
        wSum += w;
        for (let c = 0; c < 6; c++) prob[c] += w * v.sum[c] / v.count;
      }
      if (wSum > 0) {
        for (let c = 0; c < 6; c++) prob[c] /= wSum;
        model[key] = prob;
      }
    }
    return model;
  }

  // Ensemble: narrow (growth-focused) + medium + broad bandwidth
  const narrowModel = buildWeightedModel(0.05);
  const mediumModel = buildWeightedModel(0.10);
  const broadModel  = buildWeightedModel(0.20);
  const uniformModel = mergeBuckets(perRoundBuckets, allRounds);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const coarseKey = key.slice(0, -1);

      // Ensemble fine prediction: average across bandwidths where available
      let fine = null;
      let fineN = 0;
      const models = [narrowModel, mediumModel, broadModel];
      for (const m of models) {
        if (m[key]) {
          if (!fine) fine = [0, 0, 0, 0, 0, 0];
          for (let c = 0; c < 6; c++) fine[c] += m[key][c];
          fineN++;
        }
      }
      if (fine) for (let c = 0; c < 6; c++) fine[c] /= fineN;

      // Fall back to uniform model for fine if ensemble missed
      if (!fine && uniformModel[key]) fine = [...uniformModel[key]];

      // Coarse prediction for regularization
      let coarse = null;
      let coarseN = 0;
      for (const m of models) {
        if (m[coarseKey]) {
          if (!coarse) coarse = [0, 0, 0, 0, 0, 0];
          for (let c = 0; c < 6; c++) coarse[c] += m[coarseKey][c];
          coarseN++;
        }
      }
      if (coarse) for (let c = 0; c < 6; c++) coarse[c] /= coarseN;
      if (!coarse && uniformModel[coarseKey]) coarse = [...uniformModel[coarseKey]];

      let prior;
      if (fine && coarse) {
        // Regularize fine toward coarse to reduce overfitting
        prior = new Array(6);
        for (let c = 0; c < 6; c++) prior[c] = (1 - regWeight) * fine[c] + regWeight * coarse[c];
      } else if (fine) {
        prior = [...fine];
      } else if (coarse) {
        prior = [...coarse];
      } else {
        prior = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      // Adaptive floor: higher floor for high-entropy (uncertain) cells
      let entropy = 0;
      for (let c = 0; c < 6; c++) {
        if (prior[c] > 1e-6) entropy -= prior[c] * Math.log(prior[c]);
      }
      const cellFloor = entropy > 1.2 ? floor * 4 : entropy > 0.6 ? floor * 2 : floor;

      const floored = new Array(6);
      let sum = 0;
      for (let c = 0; c < 6; c++) {
        floored[c] = prior[c] > cellFloor ? prior[c] : cellFloor;
        sum += floored[c];
      }
      const out = new Array(6);
      for (let c = 0; c < 6; c++) out[c] = floored[c] / sum;
      row.push(out);
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };