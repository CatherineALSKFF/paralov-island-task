const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Nearest settlement distance (Chebyshev)
  const nearestDist = new Uint8Array(H * W).fill(99);
  for (const s of settlements) {
    for (let y = Math.max(0, s.y - 12); y <= Math.min(H - 1, s.y + 12); y++) {
      for (let x = Math.max(0, s.x - 12); x <= Math.min(W - 1, s.x + 12); x++) {
        const d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
        const idx = y * W + x;
        if (d < nearestDist[idx]) nearestDist[idx] = d;
      }
    }
  }

  // Enriched key matching bucket builder thresholds
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

  // Build augmented per-round data with 'd' residual keys
  const augBuckets = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    augBuckets[rn] = {};
    for (const [key, val] of Object.entries(b)) {
      augBuckets[rn][key] = { avg: val.sum.map(v => v / val.count), count: val.count };
    }
    const baseSuffixes = [
      ['P1', 'P1a', 'P1b'], ['P1c', 'P1ca', 'P1cb'],
      ['F1', 'F1a', 'F1b'], ['F1c', 'F1ca', 'F1cb'],
    ];
    for (const [base, sub1, sub2] of baseSuffixes) {
      if (!b[base]) continue;
      const baseN = b[base].count;
      const sub1N = b[sub1] ? b[sub1].count : 0;
      const sub2N = b[sub2] ? b[sub2].count : 0;
      const residN = baseN - sub1N - sub2N;
      if (residN < 3) continue;
      const dDist = [0, 0, 0, 0, 0, 0];
      for (let c = 0; c < 6; c++) {
        const total = b[base].sum[c];
        const s1 = b[sub1] ? b[sub1].sum[c] : 0;
        const s2 = b[sub2] ? b[sub2].sum[c] : 0;
        dDist[c] = Math.max(0, (total - s1 - s2) / residN);
      }
      augBuckets[rn][base + 'd'] = { avg: dDist, count: residN };
    }
  }

  // Extended key with 'd' for dist-3
  function getExtKey(y, x) {
    const enriched = getEnrichedKey(y, x);
    const base = getFeatureKey(initGrid, settPos, y, x);
    if (enriched !== base) return enriched;
    if (base.length >= 2 && base[0] !== 'S' && base !== 'O' && base !== 'M') {
      const nKey = base[1];
      const minDist = nearestDist[y * W + x];
      if (nKey === '1' && minDist === 3) return base + 'd';
    }
    return enriched;
  }

  // LOESS: weighted linear regression of distribution vs growth rate
  const loessCache = {};
  function fitLoess(key, bw) {
    const ck = key + ':' + bw;
    if (loessCache[ck] !== undefined) return loessCache[ck];
    const points = [];
    for (const rn of allRounds) {
      if (!augBuckets[rn] || !augBuckets[rn][key]) continue;
      const g = growthRates[String(rn)] || 0.15;
      const diff = g - targetGrowth;
      const w = Math.exp(-diff * diff / (2 * bw * bw));
      points.push({ g, dist: augBuckets[rn][key].avg, w, n: augBuckets[rn][key].count });
    }
    if (points.length < 3) { loessCache[ck] = null; return null; }
    const result = [0, 0, 0, 0, 0, 0];
    for (let c = 0; c < 6; c++) {
      let sw = 0, swg = 0, swp = 0, swgg = 0, swgp = 0;
      for (const pt of points) {
        const wt = pt.w * Math.sqrt(pt.n);
        sw += wt; swg += wt * pt.g; swp += wt * pt.dist[c];
        swgg += wt * pt.g * pt.g; swgp += wt * pt.g * pt.dist[c];
      }
      const denom = sw * swgg - swg * swg;
      if (Math.abs(denom) < 1e-12) result[c] = swp / sw;
      else {
        const b = (sw * swgp - swg * swp) / denom;
        const a = (swp - b * swg) / sw;
        result[c] = Math.max(0, a + b * targetGrowth);
      }
    }
    const s = result.reduce((a, b) => a + b, 0);
    loessCache[ck] = s > 0 ? result.map(v => v / s) : null;
    return loessCache[ck];
  }

  // Gaussian weighted mean
  function buildGaussModel(sigma) {
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
      if (!augBuckets[rn]) continue;
      for (const [key, data] of Object.entries(augBuckets[rn])) {
        if (!model[key]) model[key] = [0, 0, 0, 0, 0, 0];
        for (let c = 0; c < 6; c++) model[key][c] += ws[rn] * data.avg[c];
      }
    }
    return model;
  }

  const gaussModel = buildGaussModel(0.05);
  const wideModel = buildGaussModel(0.20);

  // Lookup with fallback
  function lookupModel(model, extKey, baseKey) {
    if (model[extKey]) return model[extKey];
    if (model[baseKey]) return model[baseKey];
    for (let len = baseKey.length - 1; len >= 1; len--) {
      if (model[baseKey.slice(0, len)]) return model[baseKey.slice(0, len)];
    }
    return null;
  }

  function lookupLoess(extKey, baseKey, bw) {
    let r = fitLoess(extKey, bw);
    if (r) return r;
    r = fitLoess(baseKey, bw);
    if (r) return r;
    for (let len = baseKey.length - 1; len >= 1; len--) {
      r = fitLoess(baseKey.slice(0, len), bw);
      if (r) return r;
    }
    return null;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const extKey = getExtKey(y, x);
      const baseKey = getFeatureKey(initGrid, settPos, y, x);

      const gPred = lookupModel(gaussModel, extKey, baseKey);
      const wPred = lookupModel(wideModel, extKey, baseKey);
      const lPred = lookupLoess(extKey, baseKey, 0.12);

      let prior = [0, 0, 0, 0, 0, 0];
      let tw = 0;
      if (lPred) { for (let c = 0; c < 6; c++) prior[c] += 0.55 * lPred[c]; tw += 0.55; }
      if (gPred) { for (let c = 0; c < 6; c++) prior[c] += 0.30 * gPred[c]; tw += 0.30; }
      if (wPred) { for (let c = 0; c < 6; c++) prior[c] += 0.15 * wPred[c]; tw += 0.15; }

      if (tw > 0) for (let c = 0; c < 6; c++) prior[c] /= tw;
      else prior = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];

      // Shrink fine key toward coarse
      if (extKey !== baseKey) {
        const coarse = gaussModel[baseKey];
        if (coarse) {
          for (let c = 0; c < 6; c++) prior[c] = 0.90 * prior[c] + 0.10 * coarse[c];
        }
      }

      // Entropy-adaptive floor
      let entropy = 0;
      for (let c = 0; c < 6; c++) if (prior[c] > 0.001) entropy -= prior[c] * Math.log(prior[c]);
      const cellFloor = entropy < 0.2 ? floor * 0.1 : entropy > 1.0 ? floor * 3 : floor;

      const floored = prior.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
