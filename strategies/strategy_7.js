const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const regWeight = 0.40;

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const allRoundNums = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Gaussian-kernel weighted model builder
  function buildWeightedModel(bw) {
    const rw = {};
    let tw = 0;
    for (const r of allRoundNums) {
      const d = growthRates[String(r)] - targetGrowth;
      const w = Math.exp(-(d * d) / (2 * bw * bw));
      rw[r] = w;
      tw += w;
    }
    for (const r of allRoundNums) rw[r] /= tw;

    const allKeys = new Set();
    for (const r of allRoundNums) {
      const b = perRoundBuckets[r];
      if (b) for (const k in b) allKeys.add(k);
    }

    const model = {};
    for (const key of allKeys) {
      const dist = [0, 0, 0, 0, 0, 0];
      let ws = 0;
      for (const r of allRoundNums) {
        const b = perRoundBuckets[r]?.[key];
        if (!b || b.count === 0) continue;
        const tot = b.sum.reduce((a, v) => a + v, 0);
        if (tot === 0) continue;
        const w = rw[r];
        for (let c = 0; c < 6; c++) dist[c] += w * (b.sum[c] / tot);
        ws += w;
      }
      if (ws > 0) model[key] = dist.map(v => v / ws);
    }
    return model;
  }

  // Ensemble of 3 bandwidths: tight, medium, wide
  const models = [0.03, 0.07, 0.15].map(bw => buildWeightedModel(bw));
  // Uniform fallback (very wide bandwidth)
  const uniformModel = buildWeightedModel(100);

  // Blend fine + coarse keys within a single model
  function blended(model, key) {
    const fine = model[key];
    const ck = key.slice(0, -1);
    const coarse = model[ck];
    if (fine && coarse) return fine.map((v, i) => (1 - regWeight) * v + regWeight * coarse[i]);
    if (fine) return fine;
    const ck2 = ck.slice(0, -1);
    const coarser = model[ck2];
    if (coarse && coarser) return coarse.map((v, i) => (1 - regWeight) * v + regWeight * coarser[i]);
    if (coarse) return coarse;
    if (coarser) return coarser;
    return null;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      // Average across bandwidth ensemble
      const acc = [0, 0, 0, 0, 0, 0];
      let cnt = 0;
      for (const m of models) {
        const d = blended(m, key);
        if (d) { for (let c = 0; c < 6; c++) acc[c] += d[c]; cnt++; }
      }

      let prior;
      if (cnt > 0) {
        prior = acc.map(v => v / cnt);
      } else {
        const fb = blended(uniformModel, key);
        prior = fb || [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      // Adaptive floor: low for peaked distributions, higher for uncertain
      const ent = -prior.reduce((s, p) => s + (p > 1e-12 ? p * Math.log(p) : 0), 0);
      const maxEnt = Math.log(6);
      const r = ent / maxEnt;
      const af = floor * (0.3 + 5 * r * r);

      const floored = prior.map(v => Math.max(v, af));
      const s = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / s));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };