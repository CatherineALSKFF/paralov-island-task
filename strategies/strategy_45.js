const { H, W, getFeatureKey } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.00005;
  const sigma = config.sigma || 0.06;
  const blendReg = config.blendReg || 0.6;
  const coarseShrink = config.cs || 2;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // ── Model 1: Gaussian-weighted merge (per-cell weighting, baseline-style) ──
  const m1 = {};
  {
    const weights = {};
    let wTotal = 0;
    for (const rn of allRounds) {
      const g = growthRates[String(rn)];
      if (g === undefined) continue;
      const w = Math.exp(-0.5 * ((g - targetGrowth) / sigma) ** 2);
      weights[rn] = w;
      wTotal += w;
    }
    if (wTotal > 0) for (const rn of allRounds) if (weights[rn]) weights[rn] /= wTotal;

    for (const rn of allRounds) {
      const w = weights[rn] || 0;
      if (w < 1e-10) continue;
      const rb = perRoundBuckets[String(rn)];
      if (!rb) continue;
      for (const [key, val] of Object.entries(rb)) {
        if (!m1[key]) m1[key] = { count: 0, sum: [0,0,0,0,0,0] };
        const avg = val.sum.map(v => v / val.count);
        m1[key].count += w * val.count;
        for (let c = 0; c < 6; c++) m1[key].sum[c] += w * avg[c] * val.count;
      }
    }
    for (const key of Object.keys(m1)) {
      const val = m1[key];
      m1[key] = val.sum.map(v => v / val.count);
    }
  }

  // ── Model 2: Per-round local linear regression ──
  const keyRoundData = {};
  for (const rn of allRounds) {
    const rb = perRoundBuckets[String(rn)];
    if (!rb) continue;
    const g = growthRates[String(rn)];
    if (g === undefined) continue;
    for (const [key, val] of Object.entries(rb)) {
      if (!keyRoundData[key]) keyRoundData[key] = [];
      keyRoundData[key].push({ growth: g, dist: val.sum.map(v => v / val.count) });
    }
  }

  const regCache = {};
  function getReg(key) {
    if (regCache[key] !== undefined) return regCache[key];
    const data = keyRoundData[key];
    if (!data || data.length < 4) { regCache[key] = null; return null; }

    const weights = data.map(d => Math.exp(-0.5 * ((d.growth - targetGrowth) / sigma) ** 2));
    const wSum = weights.reduce((a, b) => a + b, 0);
    if (wSum < 0.01) { regCache[key] = null; return null; }

    const avg = [0,0,0,0,0,0];
    for (let i = 0; i < data.length; i++) {
      const w = weights[i] / wSum;
      for (let c = 0; c < 6; c++) avg[c] += w * data[i].dist[c];
    }

    let swg = 0, swg2 = 0;
    for (let i = 0; i < data.length; i++) {
      swg += weights[i] * data[i].growth;
      swg2 += weights[i] * data[i].growth ** 2;
    }
    const gMean = swg / wSum;
    const gVar = swg2 / wSum - gMean ** 2;
    if (gVar < 1e-10) { regCache[key] = avg; return avg; }

    const result = new Array(6);
    for (let c = 0; c < 6; c++) {
      let swgp = 0;
      for (let i = 0; i < data.length; i++) {
        swgp += weights[i] * data[i].growth * data[i].dist[c];
      }
      const b = (swgp / wSum - gMean * avg[c]) / gVar;
      result[c] = Math.max(avg[c] + b * (targetGrowth - gMean), 0);
    }
    const rSum = result.reduce((a, b) => a + b, 0);
    if (rSum > 0) for (let c = 0; c < 6; c++) result[c] /= rSum;

    regCache[key] = result;
    return result;
  }

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      let p1 = m1[key] || (key.length > 1 ? m1[key.slice(0, -1)] : null);
      let p2 = getReg(key) || (key.length > 1 ? getReg(key.slice(0, -1)) : null);

      let prior;
      if (p1 && p2) {
        prior = new Array(6);
        for (let c = 0; c < 6; c++) {
          prior[c] = (1 - blendReg) * p1[c] + blendReg * p2[c];
        }
      } else {
        prior = p1 || p2 || [1/6,1/6,1/6,1/6,1/6,1/6];
      }

      // Coarse-key shrinkage
      if (key.length > 1) {
        const coarse = key.slice(0, -1);
        const cp1 = m1[coarse];
        const cp2 = getReg(coarse);
        if (cp1 || cp2) {
          const coarsePrior = new Array(6);
          if (cp1 && cp2) {
            for (let c = 0; c < 6; c++) coarsePrior[c] = (1 - blendReg) * cp1[c] + blendReg * cp2[c];
          } else {
            const cp = cp1 || cp2;
            for (let c = 0; c < 6; c++) coarsePrior[c] = cp[c];
          }
          const nR = (keyRoundData[key] || []).length;
          const alpha = nR / (nR + coarseShrink);
          for (let c = 0; c < 6; c++) {
            prior[c] = alpha * prior[c] + (1 - alpha) * coarsePrior[c];
          }
        }
      }

      const floored = prior.map(v => Math.max(v, floor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
