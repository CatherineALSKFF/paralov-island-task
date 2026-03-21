const { H, W, terrainToClass, getFeatureKey } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);
  const regWeight = 0.40;
  const lambdas = [5, 10, 20];

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Precompute round weights for each lambda
  const weightSets = lambdas.map(lambda => {
    const weights = {};
    for (const r of allRounds) {
      const g = growthRates[String(r)] || 0.15;
      weights[r] = Math.exp(-lambda * Math.abs(g - targetGrowth));
    }
    return weights;
  });

  // Precompute per-key per-round averages and cross-round disagreement
  const allKeys = new Set();
  for (const rn of allRounds) {
    if (perRoundBuckets[rn]) {
      for (const k of Object.keys(perRoundBuckets[rn])) allKeys.add(k);
    }
  }

  // Per-key disagreement: how much do training rounds disagree?
  const keyDisagreement = {};
  for (const key of allKeys) {
    const avgs = [];
    for (const rn of allRounds) {
      const b = perRoundBuckets[rn]?.[key];
      if (b && b.count > 0) {
        avgs.push(b.sum.map(v => v / b.count));
      }
    }
    if (avgs.length < 2) {
      keyDisagreement[key] = 1.0;
    } else {
      // Average pairwise L1 distance (0 = all agree, up to 2 = max disagreement)
      let totalL1 = 0, pairs = 0;
      for (let i = 0; i < avgs.length; i++) {
        for (let j = i + 1; j < avgs.length; j++) {
          let l1 = 0;
          for (let c = 0; c < 6; c++) l1 += Math.abs(avgs[i][c] - avgs[j][c]);
          totalL1 += l1;
          pairs++;
        }
      }
      keyDisagreement[key] = totalL1 / pairs;
    }
  }

  // Weighted distribution for a single feature key
  function getWeightedDist(featureKey, weights) {
    const ws = new Array(6).fill(0);
    let tw = 0;
    for (const r of allRounds) {
      const buckets = perRoundBuckets[r];
      if (!buckets || !buckets[featureKey]) continue;
      const b = buckets[featureKey];
      const w = weights[r];
      const cnt = b.count || 1;
      for (let c = 0; c < 6; c++) ws[c] += w * (b.sum[c] / cnt);
      tw += w;
    }
    if (tw === 0) return null;
    return ws.map(v => v / tw);
  }

  // Hierarchical blend: fine → coarse with regularization
  function getBlendedDist(key, weights) {
    const levels = [];
    let k = key;
    while (k.length >= 1) {
      const d = getWeightedDist(k, weights);
      if (d) levels.push(d);
      if (k.length <= 1) break;
      k = k.slice(0, -1);
    }
    if (levels.length === 0) return [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
    if (levels.length === 1) return levels[0];

    // Blend from coarsest to finest
    let result = [...levels[levels.length - 1]];
    for (let i = levels.length - 2; i >= 0; i--) {
      const a = 1 - regWeight;
      for (let c = 0; c < 6; c++) {
        result[c] = a * levels[i][c] + regWeight * result[c];
      }
    }
    return result;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      // Ensemble over lambda bandwidths
      const ensemble = new Array(6).fill(0);
      for (const weights of weightSets) {
        const d = getBlendedDist(key, weights);
        for (let c = 0; c < 6; c++) ensemble[c] += d[c];
      }
      const nL = weightSets.length;
      const prior = ensemble.map(v => v / nL);

      // Adaptive floor based on confidence AND cross-round disagreement
      // Static cells (ocean/mountain) → near-deterministic, tiny floor
      // Dynamic cells with low disagreement → moderate floor
      // Dynamic cells with high disagreement → high floor (prevent catastrophic KL)
      let maxP = 0;
      for (let c = 0; c < 6; c++) if (prior[c] > maxP) maxP = prior[c];

      const dis = keyDisagreement[key] !== undefined ? keyDisagreement[key] : 1.0;

      let af;
      if (maxP > 0.97) {
        // Static cell (ocean, mountain) — near-certain
        af = 0.00005;
      } else {
        // Dynamic cell — floor scales with disagreement
        // dis in [0, 2] → floor in [0.002, 0.02]
        af = 0.002 + 0.009 * Math.min(dis, 2);
      }

      const floored = prior.map(v => Math.max(v, af));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };