const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const allRoundNums = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // --- Gaussian kernel weights by growth-rate similarity ---
  const sigma = 0.05;
  const gWeights = {};
  for (const r of allRoundNums) {
    const diff = (growthRates[String(r)] || 0.15) - targetGrowth;
    gWeights[r] = Math.exp(-(diff * diff) / (2 * sigma * sigma));
  }

  // --- Feature key hierarchy ---
  // "F2c" -> ["F2c","F2","F"], "S1" -> ["S1","S"], "O" -> ["O"], "M" -> ["M"]
  function getKeyHierarchy(key) {
    if (key === 'O' || key === 'M') return [key];
    const keys = [key];
    if (key.endsWith('c')) keys.push(key.slice(0, -1));
    if (keys[keys.length - 1].length > 1) keys.push(keys[keys.length - 1][0]);
    return keys;
  }

  // --- Build weighted model with hierarchy aggregation ---
  function buildModel(weights) {
    const model = {};
    for (const r of allRoundNums) {
      const buckets = perRoundBuckets[String(r)];
      if (!buckets) continue;
      const w = weights[r];
      for (const [origKey, v] of Object.entries(buckets)) {
        if (!v || !v.sum || v.count === 0) continue;
        const hierarchy = getKeyHierarchy(origKey);
        for (const key of hierarchy) {
          if (!model[key]) model[key] = { wsum: new Float64Array(6), wcount: 0 };
          const m = model[key];
          for (let c = 0; c < 6; c++) m.wsum[c] += w * v.sum[c];
          m.wcount += w * v.count;
        }
      }
    }
    const probs = {};
    for (const [key, m] of Object.entries(model)) {
      if (m.wcount > 0) {
        probs[key] = new Array(6);
        for (let c = 0; c < 6; c++) probs[key][c] = m.wsum[c] / m.wcount;
      }
    }
    return probs;
  }

  // Gaussian-weighted model (favors growth-similar rounds)
  const gModel = buildModel(gWeights);

  // Uniform-weighted model (all rounds equal — safety net for outlier rounds)
  const uWeights = {};
  for (const r of allRoundNums) uWeights[r] = 1.0;
  const uModel = buildModel(uWeights);

  // Hierarchy blend weights: [most specific, middle, broadest]
  const hierW = [0.50, 0.30, 0.20];

  function getHierBlend(model, key) {
    const hierarchy = getKeyHierarchy(key);
    const blended = new Float64Array(6);
    let tw = 0;
    for (let i = 0; i < hierarchy.length; i++) {
      const p = model[hierarchy[i]];
      if (p) {
        const w = hierW[Math.min(i, hierW.length - 1)];
        for (let c = 0; c < 6; c++) blended[c] += w * p[c];
        tw += w;
      }
    }
    if (tw === 0) return null;
    const out = new Array(6);
    for (let c = 0; c < 6; c++) out[c] = blended[c] / tw;
    return out;
  }

  // 65% growth-weighted, 35% uniform (hedges against misleading growth-similar rounds)
  const gBlend = 0.65;

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      const gp = getHierBlend(gModel, key);
      const up = getHierBlend(uModel, key);

      let prior;
      if (gp && up) {
        prior = gp.map((v, i) => gBlend * v + (1 - gBlend) * up[i]);
      } else if (gp) {
        prior = gp;
      } else if (up) {
        prior = up;
      } else {
        prior = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6];
      }

      // Per-cell adaptive floor based on prediction entropy
      const entropy = -prior.reduce((s, p) => s + (p > 1e-9 ? p * Math.log(p) : 0), 0);
      const maxEntropy = Math.log(6);
      const ratio = entropy / maxEntropy; // 0=certain, 1=uniform
      // Confident → tiny floor; uncertain → larger floor to hedge KL risk
      const af = floor + (0.008 - floor) * ratio * ratio;

      const floored = prior.map(v => Math.max(v, af));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
