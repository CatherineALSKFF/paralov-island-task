const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const sigma = config.sigma || 0.05;
  const floor = config.FLOOR || 0.0001;
  const linWeight = config.linWeight || 0.85;
  const loessSigma = config.loessSigma || 0.15;

  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const nearestDist = Array.from({ length: H }, () => Array(W).fill(99));
  for (const s of settlements) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
        if (d < nearestDist[y][x]) nearestDist[y][x] = d;
      }
    }
  }

  // Enhanced feature key with dist-3 'd' suffix for nS=1 cells
  function getEnhancedKey(y, x) {
    const coarseKey = getFeatureKey(initGrid, settPos, y, x);
    if (coarseKey === 'O' || coarseKey === 'M') return coarseKey;
    if (coarseKey[0] === 'S') return coarseKey;
    const nKey = coarseKey[1];
    const minDist = nearestDist[y][x];
    if (nKey === '0') return coarseKey + (minDist === 4 ? 'n' : minDist <= 8 ? 'm' : 'f');
    if (nKey === '1') {
      if (minDist === 1) return coarseKey + 'a';
      if (minDist === 2) return coarseKey + 'b';
      if (minDist === 3) return coarseKey + 'd'; // new: dist-3 sub-key
    }
    return coarseKey;
  }

  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Augment buckets: compute virtual 'd' (dist-3) keys by subtracting 'a' and 'b' from base
  const augBuckets = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    augBuckets[rn] = {};
    for (const [key, val] of Object.entries(b)) {
      augBuckets[rn][key] = val.sum.map(v => v / val.count); // store as average
    }
    // Compute residual 'd' keys for P1, P1c, F1, F1c, S1, S1c, P2, F2
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
      if (residN < 5) continue; // too few cells, skip
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
      let sumW = 0, sumWG = 0, sumWP = 0, sumWGG = 0, sumWGP = 0;
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

  // Gaussian weighted mean model (from augmented buckets)
  const roundWeights = {};
  let tw = 0;
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
    let wmDist = meanModel[fineKey] ? [...meanModel[fineKey]] :
                 meanModel[baseKey] ? [...meanModel[baseKey]] : null;
    let lnDist = fitLoess(fineKey) || fitLoess(baseKey);

    if (wmDist && lnDist) return wmDist.map((v, c) => (1 - linWeight) * v + linWeight * lnDist[c]);
    if (lnDist) return lnDist;
    if (wmDist) return wmDist;
    let fb = baseKey;
    while (fb.length > 1) { fb = fb.slice(0, -1); if (meanModel[fb]) return [...meanModel[fb]]; }
    return [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const enhKey = getEnhancedKey(y, x);
      const coarseKey = getFeatureKey(initGrid, settPos, y, x);
      let prior = lookupKey(enhKey, coarseKey);

      let entropy = 0;
      for (let c = 0; c < 6; c++) if (prior[c] > 0.001) entropy -= prior[c] * Math.log(prior[c]);
      const cellFloor = entropy > 0.5 ? floor : floor * 0.1;
      const floored = prior.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
