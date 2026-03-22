const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const sigma = config.sigma || 0.05;
  const floor = config.FLOOR || 0.0001;
  const linWeight = config.linWeight || 0.9;
  const loessSigma = config.loessSigma || 0.15;
  const hierBlend = config.hierBlend || 0.2; // blend fine lin toward coarse lin

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
    }
    return coarseKey;
  }

  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Per-round averages for all keys
  const perRoundAvg = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    perRoundAvg[rn] = {};
    for (const [key, val] of Object.entries(b)) {
      perRoundAvg[rn][key] = val.sum.map(v => v / val.count);
    }
  }

  // Gaussian weighted mean model
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
    if (!perRoundAvg[rn]) continue;
    for (const [key, avg] of Object.entries(perRoundAvg[rn])) {
      if (!meanModel[key]) meanModel[key] = [0, 0, 0, 0, 0, 0];
      for (let c = 0; c < 6; c++) meanModel[key][c] += roundWeights[rn] * avg[c];
    }
  }

  // LOESS linear model per key
  function fitLoess(key) {
    const points = [];
    for (const rn of allRounds) {
      if (perRoundAvg[rn] && perRoundAvg[rn][key]) {
        const g = growthRates[String(rn)] || 0.15;
        const w = Math.exp(-(g - targetGrowth) * (g - targetGrowth) / (2 * loessSigma * loessSigma));
        points.push({ g, dist: perRoundAvg[rn][key], w });
      }
    }
    if (points.length < 3) return null;

    const result = [0, 0, 0, 0, 0, 0];
    for (let c = 0; c < 6; c++) {
      let sumW = 0, sumWG = 0, sumWP = 0, sumWGG = 0, sumWGP = 0;
      for (const pt of points) {
        sumW += pt.w; sumWG += pt.w * pt.g; sumWP += pt.w * pt.dist[c];
        sumWGG += pt.w * pt.g * pt.g; sumWGP += pt.w * pt.g * pt.dist[c];
      }
      const denom = sumW * sumWGG - sumWG * sumWG;
      if (Math.abs(denom) < 1e-12) {
        result[c] = sumWP / sumW;
      } else {
        const b = (sumW * sumWGP - sumWG * sumWP) / denom;
        const a = (sumWP - b * sumWG) / sumW;
        result[c] = Math.max(0, a + b * targetGrowth);
      }
    }
    const s = result.reduce((a, b) => a + b, 0);
    return s > 0 ? result.map(v => v / s) : null;
  }

  // Cache linear models
  const linCache = {};
  function getLinModel(key) {
    if (linCache[key] === undefined) linCache[key] = fitLoess(key);
    return linCache[key];
  }

  function lookupKey(fineKey, baseKey) {
    let wmDist = meanModel[fineKey] ? [...meanModel[fineKey]] :
                 meanModel[baseKey] ? [...meanModel[baseKey]] : null;

    // Get linear models for fine and coarse keys
    let fineLin = getLinModel(fineKey);
    let coarseLin = fineKey !== baseKey ? getLinModel(baseKey) : null;

    // Blend fine linear toward coarse linear for regularization
    let lnDist = null;
    if (fineLin && coarseLin) {
      lnDist = fineLin.map((v, c) => (1 - hierBlend) * v + hierBlend * coarseLin[c]);
    } else if (fineLin) {
      lnDist = fineLin;
    } else if (coarseLin) {
      lnDist = coarseLin;
    }

    if (wmDist && lnDist) {
      return wmDist.map((v, c) => (1 - linWeight) * v + linWeight * lnDist[c]);
    }
    if (lnDist) return lnDist;
    if (wmDist) return wmDist;

    let fb = baseKey;
    while (fb.length > 1) {
      fb = fb.slice(0, -1);
      if (meanModel[fb]) return [...meanModel[fb]];
    }
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
