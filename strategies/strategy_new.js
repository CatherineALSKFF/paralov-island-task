const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // --- Build soft-weighted model with Gaussian kernel ---
  function buildWeightedModel(bw) {
    const rw = {};
    let ws = 0;
    for (const r of allRounds) {
      const g = growthRates[String(r)] || 0.15;
      const d = Math.abs(g - targetGrowth);
      const w = Math.exp(-d * d / (2 * bw * bw));
      rw[r] = w;
      ws += w;
    }
    if (ws > 0) for (const r of allRounds) rw[r] /= ws;

    const model = {};
    for (const r of allRounds) {
      const buckets = perRoundBuckets[r];
      if (!buckets) continue;
      const w = rw[r];
      for (const key in buckets) {
        const b = buckets[key];
        if (!b || !b.sum) continue;
        const cnt = b.count || 1;
        if (!model[key]) model[key] = new Float64Array(6);
        for (let c = 0; c < 6; c++) model[key][c] += w * (b.sum[c] / cnt);
      }
    }
    for (const key in model) {
      let s = 0;
      for (let c = 0; c < 6; c++) s += model[key][c];
      if (s > 0) for (let c = 0; c < 6; c++) model[key][c] /= s;
    }
    return model;
  }

  // Ensemble: tight (precise for easy rounds), medium, wide (robust for hard rounds)
  const models = [
    { model: buildWeightedModel(0.04), w: 0.30 },
    { model: buildWeightedModel(0.08), w: 0.40 },
    { model: buildWeightedModel(0.20), w: 0.30 },
  ];

  // Uniform model for regularization
  const uniformModel = buildWeightedModel(100);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const regWeight = 0.20;
  const logSix = Math.log(6);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const fb = key.length > 1 ? key.slice(0, -1) : null;
      const fb2 = key.length > 2 ? key.slice(0, 1) : null;

      // Ensemble prediction across bandwidths
      const ens = new Float64Array(6);
      let ensW = 0;

      for (const { model, w } of models) {
        const dist = model[key] || (fb && model[fb]) || (fb2 && model[fb2]);
        if (dist) {
          for (let c = 0; c < 6; c++) ens[c] += w * dist[c];
          ensW += w;
        }
      }

      // Uniform model with fallback chain
      const uDist = uniformModel[key] || (fb && uniformModel[fb]) || (fb2 && uniformModel[fb2]);

      let prior;
      if (ensW > 0 && uDist) {
        prior = new Array(6);
        for (let c = 0; c < 6; c++) {
          prior[c] = (1 - regWeight) * (ens[c] / ensW) + regWeight * uDist[c];
        }
      } else if (ensW > 0) {
        prior = new Array(6);
        for (let c = 0; c < 6; c++) prior[c] = ens[c] / ensW;
      } else if (uDist) {
        prior = Array.from(uDist);
      } else {
        prior = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      // Adaptive floor: confident cells get tiny floor, uncertain cells get full floor
      let entropy = 0;
      for (let c = 0; c < 6; c++) {
        if (prior[c] > 1e-10) entropy -= prior[c] * Math.log(prior[c]);
      }
      const aFloor = floor * Math.max(0.02, entropy / logSix);

      const floored = prior.map(v => Math.max(v, aFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
