const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.00001;
  const sigma = config.sigma || 0.037;
  const temp = config.temp || 1.165;
  const shrinkStrength = config.shrink || 2;
  const wGauss = config.wGauss || 0.97;
  const wTopK = config.wTopK || 0.01;
  const wUni = config.wUni || 0.02;
  const K = config.K || 3;

  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // --- Gaussian-weighted model (tight sigma for growth specificity) ---
  const roundWeights = {};
  let wTotal = 0;
  for (const r of allRounds) {
    const g = growthRates[String(r)] || 0.15;
    const d = Math.abs(g - targetGrowth);
    roundWeights[r] = Math.exp(-0.5 * (d / sigma) ** 2);
    wTotal += roundWeights[r];
  }
  for (const r of allRounds) roundWeights[r] /= wTotal;

  function buildWeightedProbs(wts) {
    const buckets = {};
    for (const r of allRounds) {
      const rb = perRoundBuckets[r];
      if (!rb) continue;
      const w = wts[r];
      if (w < 1e-10) continue;
      for (const key in rb) {
        const b = rb[key];
        if (!buckets[key]) buckets[key] = { count: 0, sum: new Float64Array(6) };
        buckets[key].count += b.count * w;
        for (let c = 0; c < 6; c++) buckets[key].sum[c] += b.sum[c] * w;
      }
    }
    const probs = {};
    for (const key in buckets) {
      const s = buckets[key].sum;
      const tot = s[0] + s[1] + s[2] + s[3] + s[4] + s[5];
      if (tot > 0) probs[key] = { p: Array.from(s, v => v / tot), n: buckets[key].count };
    }
    return probs;
  }

  const gaussModel = buildWeightedProbs(roundWeights);

  // --- Top-K model (tiny weight, acts as minor diversifier) ---
  const candidates = { ...growthRates };
  delete candidates[String(testRound)];
  const closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  const topKWeights = {};
  for (const r of allRounds) topKWeights[r] = 0;
  for (const r of closestRounds) topKWeights[r] = 1 / closestRounds.length;
  const topKModel = buildWeightedProbs(topKWeights);

  // --- Uniform model (tiny regularizer) ---
  const uniWeights = {};
  for (const r of allRounds) uniWeights[r] = 1 / allRounds.length;
  const uniModel = buildWeightedProbs(uniWeights);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // --- Lookup with shrinkage toward coarser keys ---
  function lookupWithShrinkage(model, key) {
    let result = null;
    let nEff = 0;
    if (model[key]) {
      result = [...model[key].p];
      nEff = model[key].n;
    }
    for (let trim = 1; trim < key.length; trim++) {
      const coarse = key.slice(0, -trim);
      if (!model[coarse]) continue;
      if (!result) {
        result = [...model[coarse].p];
        nEff = model[coarse].n;
      } else {
        const alpha = nEff / (nEff + shrinkStrength);
        const cp = model[coarse].p;
        for (let c = 0; c < 6; c++) result[c] = alpha * result[c] + (1 - alpha) * cp[c];
      }
      break;
    }
    return result;
  }

  // --- Predict ---
  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      const pGauss = lookupWithShrinkage(gaussModel, key);
      const pTopK = lookupWithShrinkage(topKModel, key);
      const pUni = lookupWithShrinkage(uniModel, key);

      let prior = new Array(6).fill(0);
      let totalW = 0;
      if (pGauss) { for (let c = 0; c < 6; c++) prior[c] += wGauss * pGauss[c]; totalW += wGauss; }
      if (pTopK) { for (let c = 0; c < 6; c++) prior[c] += wTopK * pTopK[c]; totalW += wTopK; }
      if (pUni) { for (let c = 0; c < 6; c++) prior[c] += wUni * pUni[c]; totalW += wUni; }

      if (totalW > 0) {
        for (let c = 0; c < 6; c++) prior[c] /= totalW;
      } else {
        prior = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      // Temperature scaling: soften predictions to reduce KL on mismatched cells
      let tS = 0;
      for (let c = 0; c < 6; c++) {
        prior[c] = Math.pow(Math.max(prior[c], 1e-15), 1 / temp);
        tS += prior[c];
      }
      for (let c = 0; c < 6; c++) prior[c] /= tS;

      // Adaptive floor: higher for uncertain cells, lower for confident
      const entropy = -prior.reduce((s, p) => s + (p > 1e-12 ? p * Math.log(p) : 0), 0);
      const maxEnt = Math.log(6);
      const ratio = entropy / maxEnt;
      const adaptFloor = floor * (0.5 + 4.5 * ratio * ratio);

      const floored = prior.map(v => Math.max(v, adaptFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
