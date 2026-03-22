const { H, W, getFeatureKey } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = 0.0001;
  const sigma = 0.12;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const trainRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Collect all feature keys from training data
  const allKeys = new Set();
  for (const rn of trainRounds) {
    const b = perRoundBuckets[String(rn)];
    if (b) for (const k of Object.keys(b)) allKeys.add(k);
  }

  // LOESS: locally weighted linear regression per key per class
  // Fits P(class|key,growth) = a + b*growth using Gaussian kernel weights
  // This captures the TREND of how class probabilities change with growth rate,
  // rather than just averaging nearby points (kernel approach).
  const model = {};
  for (const key of allKeys) {
    const result = new Float64Array(6);
    for (let c = 0; c < 6; c++) {
      let S = 0, Sg = 0, Sgg = 0, Sp = 0, Sgp = 0;
      for (const rn of trainRounds) {
        const b = perRoundBuckets[String(rn)]?.[key];
        if (!b) continue;
        const gr = growthRates[String(rn)] || 0.15;
        const p = b.sum[c] / b.count;
        const diff = gr - targetGrowth;
        const gw = Math.exp(-(diff * diff) / (2 * sigma * sigma));
        const w = gw * b.count; // weight by cell count for statistical reliability
        S += w; Sg += w * gr; Sgg += w * gr * gr;
        Sp += w * p; Sgp += w * gr * p;
      }
      if (S < 1e-15) { result[c] = 1 / 6; continue; }
      const det = S * Sgg - Sg * Sg;
      if (Math.abs(det) < 1e-15 * S * S) {
        // Degenerate (all points at same growth rate): fall back to weighted mean
        result[c] = Math.max(0, Sp / S);
        continue;
      }
      const a = (Sp * Sgg - Sgp * Sg) / det;
      const bCoeff = (S * Sgp - Sg * Sp) / det;
      result[c] = Math.max(0, a + bCoeff * targetGrowth);
    }
    let sum = 0;
    for (let c = 0; c < 6; c++) sum += result[c];
    if (sum > 0) model[key] = Array.from(result, v => v / sum);
    else model[key] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
  }

  // Uniform fallback for missing keys
  const uniformModel = {};
  for (const key of allKeys) {
    const num = new Float64Array(6);
    let den = 0;
    for (const rn of trainRounds) {
      const b = perRoundBuckets[String(rn)]?.[key];
      if (!b) continue;
      for (let c = 0; c < 6; c++) num[c] += b.sum[c];
      den += b.count;
    }
    if (den > 0) uniformModel[key] = Array.from(num, v => v / den);
  }

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      let prior = model[key] || uniformModel[key];
      if (!prior) {
        const fb = key.slice(0, -1);
        prior = model[fb] || uniformModel[fb] || [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }
      prior = [...prior];

      const floored = prior.map(v => Math.max(v, floor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
