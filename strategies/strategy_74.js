const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const sigma = config.sigma || 0.05;
  const floor = config.FLOOR || 0.0001;
  const linWeight = config.linWeight || 0.85;
  const loessSigma = config.loessSigma || 0.15;

  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

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

  // FIXED enriched key matching bucket builder thresholds
  function getEnhancedKey(y, x) {
    const coarseKey = getFeatureKey(initGrid, settPos, y, x);
    if (coarseKey === 'O' || coarseKey === 'M') return coarseKey;
    if (coarseKey[0] === 'S') return coarseKey;
    const nKey = coarseKey[1];
    const minDist = nearestDist[y * W + x];
    if (nKey === '0') return coarseKey + (minDist <= 5 ? 'n' : minDist <= 7 ? 'm' : 'f');
    if (nKey === '1') {
      if (minDist === 1) return coarseKey + 'a';
      if (minDist === 2) return coarseKey + 'b';
      if (minDist === 3) return coarseKey + 'd';
    }
    return coarseKey;
  }

  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Augment buckets with 'd' residual keys
  const augBuckets = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    augBuckets[rn] = {};
    for (const [key, val] of Object.entries(b)) {
      augBuckets[rn][key] = val.sum.map(v => v / val.count);
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
      if (residN < 5) continue;
      const dKey = base + 'd';
      const dDist = [0, 0, 0, 0, 0, 0];
      for (let c = 0; c < 6; c++) {
        const total = b[base].sum[c];
        const s1 = b[sub1] ? b[sub1].sum[c] : 0;
        const s2 = b[sub2] ? b[sub2].sum[c] : 0;
        dDist[c] = Math.max(0, (total - s1 - s2) / residN);
      }
      augBuckets[rn][dKey] = dDist;
    }
  }

  // LOESS-weighted linear regression per key
  const linCache = {};
  function fitLoess(key) {
    if (linCache[key] !== undefined) return linCache[key];
    const points = [];
    for (const rn of allRounds) {
      if (augBuckets[rn] && augBuckets[rn][key]) {
        const g = growthRates[String(rn)] || 0.15;
        const w = Math.exp(-(g - targetGrowth) * (g - targetGrowth) / (2 * loessSigma * loessSigma));
        points.push({ g, dist: augBuckets[rn][key], w });
      }
    }
    if (points.length < 3) { linCache[key] = null; return null; }

    const result = [0, 0, 0, 0, 0, 0];
    for (let c = 0; c < 6; c++) {
      var sumW = 0, sumWG = 0, sumWP = 0, sumWGG = 0, sumWGP = 0;
      for (const pt of points) {
        sumW += pt.w; sumWG += pt.w * pt.g; sumWP += pt.w * pt.dist[c];
        sumWGG += pt.w * pt.g * pt.g; sumWGP += pt.w * pt.g * pt.dist[c];
      }
      const denom = sumW * sumWGG - sumWG * sumWG;
      if (Math.abs(denom) < 1e-12) result[c] = sumWP / sumW;
      else {
        const bCoef = (sumW * sumWGP - sumWG * sumWP) / denom;
        const a = (sumWP - bCoef * sumWG) / sumW;
        result[c] = Math.max(0, a + bCoef * targetGrowth);
      }
    }
    const s = result.reduce((a, b) => a + b, 0);
    linCache[key] = s > 0 ? result.map(v => v / s) : null;
    return linCache[key];
  }

  // Gaussian weighted mean model
  const roundWeights = {};
  var tw = 0;
  for (const rn of allRounds) {
    const dist = Math.abs((growthRates[String(rn)] || 0.15) - targetGrowth);
    roundWeights[rn] = Math.exp(-dist * dist / (2 * sigma * sigma));
    tw += roundWeights[rn];
  }
  for (const rn of allRounds) roundWeights[rn] /= tw;

  const meanModel = {};
  for (const rn of allRounds) {
    if (!augBuckets[rn]) continue;
    for (const [key, avg] of Object.entries(augBuckets[rn])) {
      if (!meanModel[key]) meanModel[key] = [0, 0, 0, 0, 0, 0];
      for (let c = 0; c < 6; c++) meanModel[key][c] += roundWeights[rn] * avg[c];
    }
  }

  function lookupKey(fineKey, baseKey) {
    var wmDist = meanModel[fineKey] ? meanModel[fineKey].slice() :
                 meanModel[baseKey] ? meanModel[baseKey].slice() : null;
    var lnDist = fitLoess(fineKey) || fitLoess(baseKey);

    if (wmDist && lnDist) return wmDist.map(function(v, c) { return (1 - linWeight) * v + linWeight * lnDist[c]; });
    if (lnDist) return lnDist;
    if (wmDist) return wmDist;
    var fb = baseKey;
    while (fb.length > 1) { fb = fb.slice(0, -1); if (meanModel[fb]) return meanModel[fb].slice(); }
    return [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
  }

  var nPrior = config.N_PRIOR || 4;
  var vpCounts = config._vpCounts || null;
  var vpTotal = config._vpTotal || null;

  var pred = [];
  for (var y = 0; y < H; y++) {
    var row = [];
    for (var x = 0; x < W; x++) {
      var enhKey = getEnhancedKey(y, x);
      var coarseKey = getFeatureKey(initGrid, settPos, y, x);
      var prior = lookupKey(enhKey, coarseKey);

      // Bayesian update with VP observations
      if (vpCounts && vpTotal && vpTotal[y][x] > 0) {
        var nObs = vpTotal[y][x];
        var updated = prior.map(function(p, c) { return nPrior * p + vpCounts[y][x][c]; });
        var total = nPrior + nObs;
        var floored = updated.map(function(v) { return Math.max(v / total, floor); });
        var sum = floored.reduce(function(a, b) { return a + b; }, 0);
        row.push(floored.map(function(v) { return v / sum; }));
      } else {
        var entropy = 0;
        for (var c = 0; c < 6; c++) if (prior[c] > 0.001) entropy -= prior[c] * Math.log(prior[c]);
        var cellFloor = entropy > 0.5 ? floor : floor * 0.1;
        var floored = prior.map(function(v) { return Math.max(v, cellFloor); });
        var sum = floored.reduce(function(a, b) { return a + b; }, 0);
        row.push(floored.map(function(v) { return v / sum; }));
      }
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
