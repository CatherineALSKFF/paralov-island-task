const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Compute per-round weights using Gaussian kernel on growth similarity
  const bandwidth = config.BW || 0.06;
  const roundWeights = {};
  let wSum = 0;
  for (const r of allRounds) {
    const g = growthRates[String(r)] || 0.15;
    const diff = Math.abs(g - targetGrowth);
    const w = Math.exp(-0.5 * (diff / bandwidth) ** 2);
    roundWeights[r] = w;
    wSum += w;
  }
  // Normalize
  for (const r of allRounds) roundWeights[r] /= wSum;

  // Build weighted model: for each feature key, weighted average of per-round distributions
  const weightedModel = {};
  const uniformModel = {};
  for (const r of allRounds) {
    const buckets = perRoundBuckets[r];
    if (!buckets) continue;
    for (const key in buckets) {
      const b = buckets[key];
      if (!b || !b.sum) continue;
      const cnt = b.count || 1;
      // Weighted model
      if (!weightedModel[key]) weightedModel[key] = { weight: 0, sum: [0,0,0,0,0,0] };
      const wm = weightedModel[key];
      const rw = roundWeights[r];
      for (let c = 0; c < 6; c++) wm.sum[c] += rw * (b.sum[c] / cnt);
      wm.weight += rw;
      // Uniform model (all rounds equal)
      if (!uniformModel[key]) uniformModel[key] = { count: 0, sum: [0,0,0,0,0,0] };
      const um = uniformModel[key];
      for (let c = 0; c < 6; c++) um.sum[c] += b.sum[c] / cnt;
      um.count += 1;
    }
  }

  // Convert models to probability distributions
  const toProbs = (entry) => {
    if (!entry) return null;
    const s = entry.weight !== undefined ? entry.weight : entry.count;
    if (s <= 0) return null;
    const p = entry.sum.map(v => v / s);
    const total = p.reduce((a, b) => a + b, 0);
    if (total <= 0) return null;
    return p.map(v => v / total);
  };

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Also build K-nearest models for ensemble
  const kValues = [3, 5, 8];
  const kModels = kValues.map(k => {
    const candidates = { ...growthRates };
    delete candidates[String(testRound)];
    const closest = selectClosestRounds(candidates, targetGrowth, Math.min(k, allRounds.length));
    return mergeBuckets(perRoundBuckets, closest);
  });

  // Regularization weight toward coarser features
  const regWeight = config.REG || 0.35;

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const coarseKey = key.slice(0, -1);
      const terrainOnly = key.slice(0, 1);

      // Collect candidate distributions from multiple sources
      const candidates = [];

      // 1. Weighted growth-similarity model (detailed key)
      let wp = toProbs(weightedModel[key]);
      if (wp) candidates.push({ p: wp, w: 3.0 });

      // 2. K-nearest models (detailed key)
      for (let ki = 0; ki < kValues.length; ki++) {
        const km = kModels[ki][key];
        if (km) {
          const cnt = km.count || 1;
          const p = km.sum.map(v => v / cnt);
          const s = p.reduce((a, b) => a + b, 0);
          if (s > 0) candidates.push({ p: p.map(v => v / s), w: 1.0 });
        }
      }

      // If no detailed key found, try coarse
      if (candidates.length === 0) {
        wp = toProbs(weightedModel[coarseKey]);
        if (wp) candidates.push({ p: wp, w: 2.0 });
        for (let ki = 0; ki < kValues.length; ki++) {
          const km = kModels[ki][coarseKey];
          if (km) {
            const cnt = km.count || 1;
            const p = km.sum.map(v => v / cnt);
            const s = p.reduce((a, b) => a + b, 0);
            if (s > 0) candidates.push({ p: p.map(v => v / s), w: 0.8 });
          }
        }
      }

      // Regularization: blend in coarser prediction
      let coarseP = toProbs(weightedModel[coarseKey]) || toProbs(uniformModel[coarseKey]);
      if (!coarseP) coarseP = toProbs(weightedModel[terrainOnly]) || toProbs(uniformModel[terrainOnly]);

      let final;
      if (candidates.length > 0) {
        // Weighted ensemble of all candidates
        final = [0,0,0,0,0,0];
        let tw = 0;
        for (const c of candidates) { tw += c.w; }
        for (const c of candidates) {
          for (let i = 0; i < 6; i++) final[i] += (c.w / tw) * c.p[i];
        }
        // Regularize toward coarser
        if (coarseP) {
          for (let i = 0; i < 6; i++) {
            final[i] = (1 - regWeight) * final[i] + regWeight * coarseP[i];
          }
        }
      } else if (coarseP) {
        final = [...coarseP];
      } else {
        final = [1/6,1/6,1/6,1/6,1/6,1/6];
      }

      // Per-cell adaptive floor based on distribution entropy
      let entropy = 0;
      for (let i = 0; i < 6; i++) {
        if (final[i] > 1e-10) entropy -= final[i] * Math.log(final[i]);
      }
      const maxEntropy = Math.log(6);
      const entropyRatio = entropy / maxEntropy;
      // Low entropy (confident) → very small floor; high entropy → larger floor
      const adaptiveFloor = floor * (0.1 + 0.9 * entropyRatio);

      const floored = final.map(v => Math.max(v, adaptiveFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };

Key changes from baseline:

1. **Gaussian-weighted growth similarity** — all rounds contribute, weighted by `exp(-0.5*(Δgrowth/bandwidth)²)` instead of hard K-nearest cutoff
2. **Ensemble of K=3,5,8 models** — reduces variance from any single K choice
3. **Hierarchical feature fallback** — detailed → coarse → terrain-only keys
4. **Regularization toward coarser features** (35% blend) — prevents overfitting to sparse detailed keys
5. **Per-cell adaptive floor** — confident cells get tiny floor (0.1×base), uncertain cells get larger floor, reducing KL penalty on static cells