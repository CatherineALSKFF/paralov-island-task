const { H, W, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const K = config.K || 4;
  const floor = 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const candidates = { ...growthRates }; delete candidates[String(testRound)];
  const closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  const trainRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // K-nearest model (baseline approach)
  const kModel = mergeBuckets(perRoundBuckets, closestRounds);

  // Growth-weighted all-rounds model (soft alternative)
  const lambda = 12;
  const softAcc = {};
  for (const r of trainRounds) {
    const buckets = perRoundBuckets[r];
    if (!buckets) continue;
    const g = growthRates[String(r)] || 0.15;
    const w = Math.exp(-lambda * Math.abs(g - targetGrowth));
    for (const key in buckets) {
      const b = buckets[key];
      if (!b || b.count === 0) continue;
      if (!softAcc[key]) softAcc[key] = { s: new Float64Array(6), n: 0 };
      for (let c = 0; c < 6; c++) softAcc[key].s[c] += w * b.sum[c];
      softAcc[key].n += w * b.count;
    }
  }
  const softModel = {};
  for (const key in softAcc) {
    const m = softAcc[key];
    if (m.n > 0) softModel[key] = Array.from(m.s, v => v / m.n);
  }

  // LOO cross-validation within K-nearest to detect disagreement
  const roundKLs = [];
  for (const r of closestRounds) {
    const others = closestRounds.filter(x => x !== r);
    const otherModel = mergeBuckets(perRoundBuckets, others);
    const rb = perRoundBuckets[r];
    if (!rb) { roundKLs.push(0); continue; }
    let totalKL = 0, totalW = 0;
    for (const key in rb) {
      const pred = otherModel[key];
      if (!pred) continue;
      const b = rb[key];
      const actual = b.sum.map(s => s / b.count);
      let kl = 0;
      for (let c = 0; c < 6; c++) {
        if (actual[c] > 0.001) kl += actual[c] * Math.log(actual[c] / Math.max(pred[c], 0.0001));
      }
      totalKL += Math.max(kl, 0) * b.count;
      totalW += b.count;
    }
    roundKLs.push(totalW > 0 ? totalKL / totalW : 0);
  }

  // Adaptive beta: blend more toward soft model when K-nearest disagree
  const maxKL = Math.max(...roundKLs);
  const beta = Math.min(0.05 + 1.5 * maxKL, 0.40);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // 3-level key lookup: full -> coarse (drop coastal) -> terrain only
  function lookup3(model, key) {
    if (model[key]) return model[key];
    if (key.length > 1) {
      const ck = key.slice(0, -1);
      if (model[ck]) return model[ck];
      const tk = key[0];
      if (tk !== ck && model[tk]) return model[tk];
    }
    return null;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      const kProb = lookup3(kModel, key);
      const sProb = lookup3(softModel, key);

      let prior;
      if (kProb && sProb) {
        prior = new Array(6);
        for (let c = 0; c < 6; c++) prior[c] = (1 - beta) * kProb[c] + beta * sProb[c];
      } else if (kProb) {
        prior = [...kProb];
      } else if (sProb) {
        prior = [...sProb];
      } else {
        prior = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
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
