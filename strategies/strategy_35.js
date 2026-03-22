const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.00005;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const sigma = config.sigma || 0.04;
  const K = config.K || 2;
  const shrinkStrength = config.shrink || 1;
  const wGauss = config.wGauss || 0.70;
  const wTopK = config.wTopK || 0.25;
  const wUni = config.wUni || 0.05;
  const tempCoeff = config.tempCoeff || 1.35;

  // --- Gaussian weights ---
  const roundWeights = {};
  let wTotal = 0;
  for (const r of allRounds) {
    const g = growthRates[String(r)] || 0.15;
    const d = Math.abs(g - targetGrowth);
    const w = Math.exp(-0.5 * (d / sigma) ** 2);
    roundWeights[r] = w;
    wTotal += w;
  }
  for (const r of allRounds) roundWeights[r] /= wTotal;

  // Build per-round normalized distributions for disagreement
  const perRoundNorm = {};
  for (const r of allRounds) {
    const rb = perRoundBuckets[r];
    if (!rb) continue;
    perRoundNorm[r] = {};
    for (const key in rb) {
      const b = rb[key];
      const tot = b.sum.reduce((a, v) => a + v, 0);
      if (tot > 0) perRoundNorm[r][key] = b.sum.map(v => v / tot);
    }
  }

  function buildWeightedProbs(wts) {
    const buckets = {};
    for (const r of allRounds) {
      const rb = perRoundBuckets[r];
      if (!rb) continue;
      const w = wts[r];
      if (w < 1e-12) continue;
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

  // --- Top-K model ---
  const candidates = { ...growthRates };
  delete candidates[String(testRound)];
  const closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  const topKWeights = {};
  for (const r of allRounds) topKWeights[r] = 0;
  for (const r of closestRounds) topKWeights[r] = 1 / closestRounds.length;
  const topKModel = buildWeightedProbs(topKWeights);

  // --- Uniform model ---
  const uniWeights = {};
  for (const r of allRounds) uniWeights[r] = 1 / allRounds.length;
  const uniModel = buildWeightedProbs(uniWeights);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // --- Full hierarchical shrinkage through ALL levels ---
  function lookupWithShrinkage(model, key) {
    const keys = [key];
    for (let i = key.length - 1; i >= 1; i--) keys.push(key.slice(0, i));

    let result = null;
    let nEff = 0;
    for (const k of keys) {
      if (!model[k]) continue;
      if (!result) {
        result = [...model[k].p];
        nEff = model[k].n;
      } else {
        const alpha = nEff / (nEff + shrinkStrength);
        const cp = model[k].p;
        for (let c = 0; c < 6; c++) result[c] = alpha * result[c] + (1 - alpha) * cp[c];
        nEff += model[k].n * 0.3;
      }
    }
    return result;
  }

  // --- Compute disagreement for a key across rounds ---
  function computeDisagreement(key) {
    const predictions = [];
    const weights = [];
    for (const r of allRounds) {
      const rn = perRoundNorm[r];
      if (!rn) continue;
      let p = rn[key];
      if (!p) {
        for (let i = key.length - 1; i >= 1; i--) {
          const coarse = key.slice(0, i);
          if (rn[coarse]) { p = rn[coarse]; break; }
        }
      }
      if (!p) continue;
      predictions.push(p);
      weights.push(roundWeights[r] || 0);
    }
    if (predictions.length < 3) return 0;
    const wSum = weights.reduce((a, b) => a + b, 0);
    if (wSum < 1e-10) return 0;

    let totalDis = 0;
    for (let c = 0; c < 6; c++) {
      let wMean = 0;
      for (let i = 0; i < predictions.length; i++) wMean += weights[i] * predictions[i][c];
      wMean /= wSum;
      let wVar = 0;
      for (let i = 0; i < predictions.length; i++) {
        wVar += weights[i] * (predictions[i][c] - wMean) ** 2;
      }
      totalDis += Math.sqrt(wVar / wSum);
    }
    return totalDis;
  }

  // --- Predict ---
  const disCache = {};
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

      // Disagreement-based temperature scaling
      if (!(key in disCache)) disCache[key] = computeDisagreement(key);
      const dis = disCache[key];
      if (dis > 0.08 && tempCoeff > 0) {
        const temp = 1.0 + tempCoeff * Math.min(dis, 1.2);
        let s = 0;
        for (let c = 0; c < 6; c++) {
          prior[c] = Math.pow(Math.max(prior[c], 1e-12), 1 / temp);
          s += prior[c];
        }
        for (let c = 0; c < 6; c++) prior[c] /= s;
      }

      // Adaptive floor
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
