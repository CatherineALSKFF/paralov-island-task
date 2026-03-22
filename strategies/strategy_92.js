const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const sigma = config.sigma || 0.04;
  const loessBW = config.loessBW || 0.08;
  const loessReg = config.loessReg || 0.15;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Per-round per-key average distributions
  const roundAvgs = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    roundAvgs[rn] = {};
    for (const [key, val] of Object.entries(b)) {
      roundAvgs[rn][key] = val.sum.map(v => v / val.count);
    }
  }

  // Gaussian-weighted mean model (fallback for keys with <4 rounds in LOESS)
  let totalWeight = 0;
  const weights = {};
  for (const rn of allRounds) {
    const dist = Math.abs((growthRates[String(rn)] || 0.15) - targetGrowth);
    const w = Math.exp(-dist * dist / (2 * sigma * sigma));
    weights[rn] = w;
    totalWeight += w;
  }
  const gaussModel = {};
  for (const rn of allRounds) {
    const w = weights[rn] / totalWeight;
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    for (const [key, val] of Object.entries(b)) {
      if (!gaussModel[key]) gaussModel[key] = { count: 0, sum: [0, 0, 0, 0, 0, 0] };
      const avg = val.sum.map(v => v / val.count);
      gaussModel[key].count += w * val.count;
      for (let c = 0; c < 6; c++) gaussModel[key].sum[c] += w * avg[c] * val.count;
    }
  }
  const gaussProbs = {};
  for (const [k, v] of Object.entries(gaussModel)) gaussProbs[k] = v.sum.map(s => s / v.count);

  // All-rounds uniform model (deepest fallback)
  const allModel = mergeBuckets(perRoundBuckets, allRounds);

  // LOESS model: weighted local linear regression of probability vs growth rate
  const loessModel = {};
  const allKeys = new Set();
  for (const rn of allRounds) if (roundAvgs[rn]) for (const k in roundAvgs[rn]) allKeys.add(k);

  for (const key of allKeys) {
    const pts = [];
    for (const rn of allRounds) {
      if (!roundAvgs[rn] || !roundAvgs[rn][key]) continue;
      const g = growthRates[String(rn)] || 0.15;
      const d = g - targetGrowth;
      const w = Math.exp(-d * d / (2 * loessBW * loessBW));
      pts.push({ g, p: roundAvgs[rn][key], w });
    }
    if (pts.length < 4) continue;

    const result = [0, 0, 0, 0, 0, 0];
    for (let c = 0; c < 6; c++) {
      let sw = 0, sg = 0, sp = 0, sgg = 0, sgp = 0;
      for (const pt of pts) {
        sw += pt.w;
        sg += pt.w * pt.g;
        sp += pt.w * pt.p[c];
        sgg += pt.w * pt.g * pt.g;
        sgp += pt.w * pt.g * pt.p[c];
      }
      const det = sw * sgg - sg * sg;
      if (Math.abs(det) < 1e-12) {
        result[c] = sp / sw;
      } else {
        const slope = (sw * sgp - sg * sp) / det;
        const intercept = (sp - slope * sg) / sw;
        const regPred = intercept + slope * targetGrowth;
        const meanPred = sp / sw;
        // Blend regression with weighted mean for stability
        result[c] = Math.max(0, (1 - loessReg) * regPred + loessReg * meanPred);
      }
    }
    const s = result.reduce((a, b) => a + b, 0);
    if (s > 0) loessModel[key] = result.map(v => v / s);
  }

  // Settlement positions and distance computation
  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const nearestDist = new Uint8Array(H * W).fill(99);
  for (const s of settlements) {
    for (let dy = -15; dy <= 15; dy++) {
      for (let dx = -15; dx <= 15; dx++) {
        const ny = s.y + dy, nx = s.x + dx;
        if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
        const d = Math.max(Math.abs(dy), Math.abs(dx));
        if (d < nearestDist[ny * W + nx]) nearestDist[ny * W + nx] = d;
      }
    }
  }

  // Enriched feature key with distance-to-nearest-settlement
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

  // Predict
  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const enrichedKey = getEnrichedKey(y, x);
      const baseKey = getFeatureKey(initGrid, settPos, y, x);

      // Fallback chain for Gaussian model
      let gaussPrior = gaussProbs[enrichedKey] || gaussProbs[baseKey] || allModel[enrichedKey] || allModel[baseKey];
      if (!gaussPrior) {
        const fb = baseKey.slice(0, -1);
        gaussPrior = gaussProbs[fb] || allModel[fb] || [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      // Use LOESS as primary model, fall back to Gaussian mean
      const loessPrior = loessModel[enrichedKey] || loessModel[baseKey];
      const prior = loessPrior ? [...loessPrior] : [...gaussPrior];

      const floored = prior.map(v => Math.max(v, floor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
