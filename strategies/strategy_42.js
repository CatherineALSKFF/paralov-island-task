const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // For each key, compute per-round mean distributions
  // Then fit weighted local linear regression: P(class) = a + b * growth
  const keyRoundData = {}; // key -> [{growth, dist:[6], count}]
  for (const rn of allRounds) {
    const rb = perRoundBuckets[String(rn)];
    if (!rb) continue;
    const g = growthRates[String(rn)];
    if (g === undefined) continue;
    for (const [key, val] of Object.entries(rb)) {
      if (!keyRoundData[key]) keyRoundData[key] = [];
      keyRoundData[key].push({
        growth: g,
        dist: val.sum.map(v => v / val.count),
        count: val.count
      });
    }
  }

  // Fit local linear regression for a key at target growth
  const sigma = 0.06;
  const regressionCache = {};

  function getRegression(key) {
    if (regressionCache[key]) return regressionCache[key];
    const data = keyRoundData[key];
    if (!data || data.length === 0) return null;

    // Compute Gaussian weights
    const weights = data.map(d => Math.exp(-0.5 * ((d.growth - targetGrowth) / sigma) ** 2));
    const wSum = weights.reduce((a, b) => a + b, 0);

    if (data.length < 3 || wSum < 0.01) {
      // Not enough data for regression, fall back to weighted average
      const avg = [0,0,0,0,0,0];
      for (let i = 0; i < data.length; i++) {
        const w = weights[i] / wSum;
        for (let c = 0; c < 6; c++) avg[c] += w * data[i].dist[c];
      }
      regressionCache[key] = avg;
      return avg;
    }

    // Weighted linear regression per class: P(c) = a_c + b_c * growth
    const result = [0,0,0,0,0,0];
    // Compute weighted moments
    let sw = 0, swg = 0, swg2 = 0;
    for (let i = 0; i < data.length; i++) {
      const w = weights[i];
      sw += w;
      swg += w * data[i].growth;
      swg2 += w * data[i].growth * data[i].growth;
    }
    const gMean = swg / sw;
    const gVar = swg2 / sw - gMean * gMean;

    for (let c = 0; c < 6; c++) {
      let swp = 0, swgp = 0;
      for (let i = 0; i < data.length; i++) {
        const w = weights[i];
        swp += w * data[i].dist[c];
        swgp += w * data[i].growth * data[i].dist[c];
      }
      const pMean = swp / sw;
      const gpCov = swgp / sw - gMean * pMean;

      if (gVar > 1e-10) {
        const b = gpCov / gVar;
        const a = pMean - b * gMean;
        result[c] = a + b * targetGrowth;
      } else {
        result[c] = pMean;
      }
    }

    // Ensure non-negative (regression can go negative)
    for (let c = 0; c < 6; c++) result[c] = Math.max(result[c], 0);
    // Normalize
    const sum = result.reduce((a, b) => a + b, 0);
    if (sum > 0) for (let c = 0; c < 6; c++) result[c] /= sum;
    else result.fill(1/6);

    // Shrink toward weighted average (regularization)
    const shrinkAlpha = 0.7; // blend: 70% regression, 30% weighted average
    const avg = [0,0,0,0,0,0];
    for (let i = 0; i < data.length; i++) {
      const w = weights[i] / wSum;
      for (let c = 0; c < 6; c++) avg[c] += w * data[i].dist[c];
    }
    for (let c = 0; c < 6; c++) {
      result[c] = shrinkAlpha * result[c] + (1 - shrinkAlpha) * avg[c];
    }

    regressionCache[key] = result;
    return result;
  }

  // Also build a Gaussian-weighted model as fallback (for keys not in regression)
  const gaussModel = {};
  for (const [key, data] of Object.entries(keyRoundData)) {
    const weights = data.map(d => Math.exp(-0.5 * ((d.growth - targetGrowth) / sigma) ** 2));
    const wSum = weights.reduce((a, b) => a + b, 0);
    if (wSum === 0) continue;
    const avg = [0,0,0,0,0,0];
    for (let i = 0; i < data.length; i++) {
      const w = weights[i] / wSum;
      for (let c = 0; c < 6; c++) avg[c] += w * data[i].dist[c];
    }
    gaussModel[key] = avg;
  }

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      // Get regression prediction
      let prior = getRegression(key);

      // Fallback hierarchy
      if (!prior) {
        // Try coarser keys
        for (let trim = 1; trim < key.length; trim++) {
          const coarse = key.slice(0, -trim);
          prior = getRegression(coarse);
          if (prior) break;
        }
      }

      if (!prior) {
        prior = [1/6,1/6,1/6,1/6,1/6,1/6];
      } else {
        // Shrink toward coarser key
        for (let trim = 1; trim < key.length; trim++) {
          const coarse = key.slice(0, -trim);
          const coarsePrior = gaussModel[coarse];
          if (coarsePrior) {
            const nRounds = (keyRoundData[key] || []).length;
            const alpha = nRounds / (nRounds + 5);
            prior = prior.map((v, c) => alpha * v + (1 - alpha) * coarsePrior[c]);
            break;
          }
        }
      }

      // Adaptive floor
      const entropy = -prior.reduce((s, p) => s + (p > 1e-12 ? p * Math.log(p) : 0), 0);
      const maxEnt = Math.log(6);
      const ratio = entropy / maxEnt;
      const adaptFloor = floor * (0.5 + 4.0 * ratio * ratio);

      const floored = prior.map(v => Math.max(v, adaptFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
