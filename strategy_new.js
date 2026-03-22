const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.005;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRoundNums = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  function toProbs(entry) {
    if (!entry) return null;
    if (Array.isArray(entry)) return entry;
    if (entry.sum && entry.count > 0) return entry.sum.map(v => v / entry.count);
    return null;
  }

  function toCount(entry) {
    if (!entry) return 0;
    return entry.count || 1;
  }

  // Collect per-key, per-round probability vectors
  const keyData = {};
  for (const r of allRoundNums) {
    const g = growthRates[String(r)];
    if (g === undefined) continue;
    const buckets = perRoundBuckets[r];
    if (!buckets) continue;
    for (const key in buckets) {
      const p = toProbs(buckets[key]);
      if (!p) continue;
      if (!keyData[key]) keyData[key] = [];
      keyData[key].push({ p, cnt: toCount(buckets[key]), g });
    }
  }

  // Growth-weighted prediction for a key (Gaussian kernel)
  function growthWeightedPredict(key, bw) {
    const entries = keyData[key];
    if (!entries || entries.length === 0) return null;
    const wsum = [0, 0, 0, 0, 0, 0];
    let wtotal = 0;
    for (const e of entries) {
      const diff = e.g - targetGrowth;
      const w = Math.exp(-(diff * diff) / (2 * bw * bw)) * Math.sqrt(Math.max(1, e.cnt));
      for (let c = 0; c < 6; c++) wsum[c] += w * e.p[c];
      wtotal += w;
    }
    if (wtotal < 1e-10) return null;
    return wsum.map(v => v / wtotal);
  }

  // Equal-weighted (all-rounds) model via mergeBuckets
  const allModel = mergeBuckets(perRoundBuckets, allRoundNums);

  const U = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6];

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const ck = key.slice(0, -1);

      // Growth-weighted prediction (narrow + medium bandwidth ensemble)
      const narrow = growthWeightedPredict(key, 0.04) || growthWeightedPredict(ck, 0.04);
      const medium = growthWeightedPredict(key, 0.10) || growthWeightedPredict(ck, 0.10);

      // All-rounds prediction
      const all = toProbs(allModel[key]) || toProbs(allModel[ck]);

      // Blend available sources: narrow(35%), medium(25%), all-rounds(40%)
      // Heavy all-rounds weight provides robustness when growth rate misleads
      const sources = [];
      if (narrow) sources.push({ p: narrow, w: 0.35 });
      if (medium) sources.push({ p: medium, w: 0.25 });
      if (all) sources.push({ p: all, w: 0.40 });

      let prior;
      if (sources.length > 0) {
        const wt = sources.reduce((a, s) => a + s.w, 0);
        prior = [0, 0, 0, 0, 0, 0];
        for (const s of sources) {
          const nw = s.w / wt;
          for (let c = 0; c < 6; c++) prior[c] += nw * s.p[c];
        }
      } else {
        prior = [...U];
      }

      // Shrinkage toward uniform — limits overconfidence on hard cells
      // Cost on static cells is negligible (low entropy weight in scoring)
      const shrink = 0.06;
      for (let c = 0; c < 6; c++) {
        prior[c] = (1 - shrink) * prior[c] + shrink * U[c];
      }

      // Floor and normalize
      let sum = 0;
      for (let c = 0; c < 6; c++) {
        prior[c] = Math.max(prior[c], floor);
        sum += prior[c];
      }
      for (let c = 0; c < 6; c++) prior[c] /= sum;
      row.push(prior);
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
