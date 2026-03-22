const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;

  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const nearestDist = Array.from({ length: H }, () => Array(W).fill(99));
  for (const s of settlements)
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
        if (d < nearestDist[y][x]) nearestDist[y][x] = d;
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

  const perRoundAvg = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    perRoundAvg[rn] = {};
    for (const [key, val] of Object.entries(b)) {
      perRoundAvg[rn][key] = val.sum.map(v => v / val.count);
    }
  }

  // Build a single model with given parameters
  function buildModel(sigma, loessSig, linW) {
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
      const b = perRoundBuckets[String(rn)];
      if (!b) continue;
      for (const [key, val] of Object.entries(b)) {
        if (!meanModel[key]) meanModel[key] = [0, 0, 0, 0, 0, 0];
        const avg = val.sum.map(v => v / val.count);
        for (let c = 0; c < 6; c++) meanModel[key][c] += roundWeights[rn] * avg[c];
      }
    }

    const linModel = {};
    if (linW > 0) {
      for (const key of Object.keys(meanModel)) {
        const points = [];
        for (const rn of allRounds) {
          if (!perRoundAvg[rn] || !perRoundAvg[rn][key]) continue;
          const g = growthRates[String(rn)] || 0.15;
          const diff = g - targetGrowth;
          const w = Math.exp(-diff * diff / (2 * loessSig * loessSig));
          points.push({ g, dist: perRoundAvg[rn][key], w });
        }
        if (points.length < 3) continue;
        const result = [0, 0, 0, 0, 0, 0];
        for (let c = 0; c < 6; c++) {
          let sW = 0, sWG = 0, sWP = 0, sWGG = 0, sWGP = 0;
          for (const pt of points) {
            sW += pt.w; sWG += pt.w * pt.g; sWP += pt.w * pt.dist[c];
            sWGG += pt.w * pt.g * pt.g; sWGP += pt.w * pt.g * pt.dist[c];
          }
          const denom = sW * sWGG - sWG * sWG;
          if (Math.abs(denom) < 1e-12) result[c] = sWP / sW;
          else {
            const b = (sW * sWGP - sWG * sWP) / denom;
            const a = (sWP - b * sWG) / sW;
            result[c] = Math.max(0, a + b * targetGrowth);
          }
        }
        const s = result.reduce((a, b) => a + b, 0);
        if (s > 0) linModel[key] = result.map(v => v / s);
      }
    }

    return { meanModel, linModel, linW };
  }

  // Ensemble: multiple model configurations
  const models = [
    buildModel(0.04, 0.12, 0.9),
    buildModel(0.06, 0.15, 0.85),
    buildModel(0.08, 0.20, 0.75),
    buildModel(0.12, 0.30, 0.5),
    buildModel(0.25, 0.50, 0.0),  // nearly uniform, no LOESS
  ];
  const modelWeights = [0.25, 0.30, 0.20, 0.15, 0.10];

  function lookupSingle(model, fineKey, baseKey) {
    const { meanModel, linModel, linW } = model;
    let nm = meanModel[fineKey] || meanModel[baseKey];
    let lm = linModel[fineKey] || linModel[baseKey];
    if (nm && lm) return nm.map((v, c) => (1 - linW) * v + linW * lm[c]);
    if (nm) return nm;
    if (lm) return lm;
    let fb = baseKey;
    while (fb.length > 1) { fb = fb.slice(0, -1); if (meanModel[fb]) return [...meanModel[fb]]; }
    return [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6];
  }

  function ensembleLookup(fineKey, baseKey) {
    const result = [0, 0, 0, 0, 0, 0];
    for (let m = 0; m < models.length; m++) {
      const dist = lookupSingle(models[m], fineKey, baseKey);
      for (let c = 0; c < 6; c++) result[c] += modelWeights[m] * dist[c];
    }
    return result;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const enhKey = getEnhancedKey(y, x);
      const coarseKey = getFeatureKey(initGrid, settPos, y, x);
      let prior = ensembleLookup(enhKey, coarseKey);

      let entropy = 0;
      for (let c = 0; c < 6; c++) if (prior[c] > 0.001) entropy -= prior[c] * Math.log(prior[c]);
      const cellFloor = entropy > 0.5 ? floor * 2 : floor * 0.1;
      const floored = prior.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
