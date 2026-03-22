const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Precompute nearest settlement distance (Chebyshev)
  const nearestDist = new Uint8Array(H * W).fill(99);
  for (const s of settlements) {
    const ylo = Math.max(0, s.y - 12), yhi = Math.min(H - 1, s.y + 12);
    const xlo = Math.max(0, s.x - 12), xhi = Math.min(W - 1, s.x + 12);
    for (let y = ylo; y <= yhi; y++) {
      for (let x = xlo; x <= xhi; x++) {
        const d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
        const idx = y * W + x;
        if (d < nearestDist[idx]) nearestDist[idx] = d;
      }
    }
  }

  // Enriched key matching bucket fine keys
  function getEnrichedKey(y, x) {
    const base = getFeatureKey(initGrid, settPos, y, x);
    if (base === 'O' || base === 'M' || base[0] === 'S') return base;
    const nKey = base[1];
    const minDist = nearestDist[y * W + x];
    if (nKey === '0') return base + (minDist <= 5 ? 'n' : minDist <= 7 ? 'm' : 'f');
    if (nKey === '1') {
      if (minDist === 1) return base + 'a';
      if (minDist === 2) return base + 'b';
    }
    return base;
  }

  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Compute per-round per-key distributions
  const perRoundDist = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    perRoundDist[rn] = {};
    for (const [key, val] of Object.entries(b)) {
      perRoundDist[rn][key] = val.sum.map(v => v / val.count);
    }
  }

  // Build Gaussian-weighted model at given sigma
  function gaussianModel(sigma) {
    const model = {};
    let tw = 0;
    const ws = {};
    for (const rn of allRounds) {
      const diff = (growthRates[String(rn)] || 0.15) - targetGrowth;
      ws[rn] = Math.exp(-diff * diff / (2 * sigma * sigma));
      tw += ws[rn];
    }
    for (const rn of allRounds) ws[rn] /= tw;

    for (const rn of allRounds) {
      if (!perRoundDist[rn]) continue;
      for (const [key, avg] of Object.entries(perRoundDist[rn])) {
        if (!model[key]) model[key] = [0, 0, 0, 0, 0, 0];
        for (let c = 0; c < 6; c++) model[key][c] += ws[rn] * avg[c];
      }
    }
    return model;
  }

  // Build models at 3 different bandwidths
  const tightModel = gaussianModel(0.04);
  const midModel = gaussianModel(0.08);
  const wideModel = gaussianModel(0.20);

  // Compute per-key cross-round variance to measure uncertainty
  const keyVariance = {};
  function getKeyVar(key) {
    if (keyVariance[key] !== undefined) return keyVariance[key];
    const mean = midModel[key];
    if (!mean) { keyVariance[key] = 1; return 1; }

    let varSum = 0, tw = 0;
    for (const rn of allRounds) {
      if (!perRoundDist[rn] || !perRoundDist[rn][key]) continue;
      const diff = (growthRates[String(rn)] || 0.15) - targetGrowth;
      const w = Math.exp(-diff * diff / (2 * 0.08 * 0.08));
      const rd = perRoundDist[rn][key];
      let v = 0;
      for (let c = 0; c < 6; c++) { const d = rd[c] - mean[c]; v += d * d; }
      varSum += w * v;
      tw += w;
    }
    keyVariance[key] = tw > 0 ? varSum / tw : 1;
    return keyVariance[key];
  }

  // Lookup with hierarchical fallback
  function lookup(model, fineKey, coarseKey) {
    if (model[fineKey]) return model[fineKey];
    if (model[coarseKey]) return model[coarseKey];
    for (let len = coarseKey.length - 1; len >= 1; len--) {
      const fb = coarseKey.slice(0, len);
      if (model[fb]) return model[fb];
    }
    return null;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const richKey = getEnrichedKey(y, x);
      const coarseKey = getFeatureKey(initGrid, settPos, y, x);

      // Get predictions from multiple bandwidths
      const tPred = lookup(tightModel, richKey, coarseKey);
      const mPred = lookup(midModel, richKey, coarseKey);
      const wPred = lookup(wideModel, richKey, coarseKey);

      // Adaptive blend: high cross-round variance → favor wider model
      const kv = getKeyVar(richKey !== coarseKey ? richKey : coarseKey);
      const uncertainty = Math.min(kv / 0.02, 1.0);

      const tW = 0.50 * (1 - 0.6 * uncertainty);
      const mW = 0.30;
      const wW = 0.20 + 0.30 * uncertainty;

      let prior = [0, 0, 0, 0, 0, 0];
      let totalW = 0;
      if (tPred) { for (let c = 0; c < 6; c++) prior[c] += tW * tPred[c]; totalW += tW; }
      if (mPred) { for (let c = 0; c < 6; c++) prior[c] += mW * mPred[c]; totalW += mW; }
      if (wPred) { for (let c = 0; c < 6; c++) prior[c] += wW * wPred[c]; totalW += wW; }

      if (totalW > 0) {
        for (let c = 0; c < 6; c++) prior[c] /= totalW;
      } else {
        prior = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      // Hierarchical shrinkage: fine → coarse
      if (richKey !== coarseKey && midModel[coarseKey]) {
        const coarseDist = midModel[coarseKey];
        const shrink = 0.12;
        for (let c = 0; c < 6; c++) prior[c] = (1 - shrink) * prior[c] + shrink * coarseDist[c];
      }

      // Entropy-adaptive floor
      let entropy = 0;
      for (let c = 0; c < 6; c++) {
        if (prior[c] > 0.001) entropy -= prior[c] * Math.log(prior[c]);
      }
      const cellFloor = entropy < 0.15 ? floor * 0.05 :
                         entropy > 1.0 ? floor * 4 : floor;

      const floored = prior.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
