const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRoundNums = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Build growth-weighted model at given bandwidth (sigma>50 = uniform)
  function buildModel(sigma) {
    const model = {};
    for (const rn of allRoundNums) {
      const g = growthRates[String(rn)] || 0.15;
      const d = g - targetGrowth;
      const w = sigma > 50 ? 1 : Math.exp(-(d * d) / (2 * sigma * sigma));
      if (w < 0.001) continue;
      const buckets = perRoundBuckets[rn];
      if (!buckets) continue;
      for (const key in buckets) {
        const b = buckets[key];
        if (!model[key]) model[key] = { dist: new Float64Array(6), wSum: 0 };
        const tot = b.sum.reduce((a, v) => a + v, 0);
        if (tot <= 0) continue;
        for (let c = 0; c < 6; c++) model[key].dist[c] += (b.sum[c] / tot) * w;
        model[key].wSum += w;
      }
    }
    return model;
  }

  function resolve(entry) {
    if (!entry || entry.wSum <= 0) return null;
    const d = new Float64Array(6);
    const s = entry.dist.reduce((a, b) => a + b, 0);
    if (s <= 0) return null;
    for (let c = 0; c < 6; c++) d[c] = entry.dist[c] / s;
    return { dist: d, confidence: entry.wSum };
  }

  // Multi-scale ensemble: narrow captures growth-specific patterns,
  // wider scales provide robustness when few similar rounds exist
  const narrow = buildModel(0.04);
  const medium = buildModel(0.08);
  const wide   = buildModel(0.16);
  const global = buildModel(999);

  const models       = [narrow, medium, wide, global];
  const modelPriority = [1.0,   0.8,   0.5,  0.3];

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const fbKey = key.slice(0, -1);

      const combined = new Float64Array(6);
      let totalW = 0;

      for (let i = 0; i < models.length; i++) {
        const entry = models[i][key] || models[i][fbKey];
        const r = resolve(entry);
        if (!r) continue;
        const w = modelPriority[i] * Math.min(r.confidence, 5);
        for (let c = 0; c < 6; c++) combined[c] += r.dist[c] * w;
        totalW += w;
      }

      if (totalW > 0) {
        for (let c = 0; c < 6; c++) combined[c] /= totalW;
      } else {
        for (let c = 0; c < 6; c++) combined[c] = 1 / 6;
      }

      // Adaptive floor: uncertain cells (high entropy) get larger floor
      let ent = 0;
      for (let c = 0; c < 6; c++) {
        if (combined[c] > 1e-10) ent -= combined[c] * Math.log(combined[c]);
      }
      const af = floor + floor * 3 * (ent / Math.log(6));

      const out = new Array(6);
      let s = 0;
      for (let c = 0; c < 6; c++) { out[c] = Math.max(combined[c], af); s += out[c]; }
      for (let c = 0; c < 6; c++) out[c] /= s;
      row.push(out);
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };