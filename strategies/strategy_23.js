const { H, W, getFeatureKey } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Build multi-level weighted model for a given bandwidth
  // Levels: 0=terrain only, 1=terrain+settlementBucket, 2=exact key
  const bandwidths = [0.035, 0.07, 0.14];

  function buildMultiLevelModel(bw) {
    const weights = {};
    let total = 0;
    for (const r of allRounds) {
      const g = growthRates[String(r)] || 0.15;
      const d = g - targetGrowth;
      weights[r] = Math.exp(-d * d / (2 * bw * bw));
      total += weights[r];
    }
    if (total > 0) for (const r of allRounds) weights[r] /= total;

    const levels = [{}, {}, {}];
    for (const r of allRounds) {
      const b = perRoundBuckets[String(r)];
      if (!b) continue;
      const w = weights[r];
      for (const [key, val] of Object.entries(b)) {
        const rd = val.sum.map(s => s / val.count);
        const wc = val.count * w;
        let k0, k1;
        if (key === 'O' || key === 'M') { k0 = key; k1 = key; }
        else { k0 = key[0]; k1 = key.endsWith('c') ? key.slice(0, -1) : key; }
        const ks = [k0, k1, key];
        for (let lvl = 0; lvl < 3; lvl++) {
          const k = ks[lvl];
          if (!levels[lvl][k]) levels[lvl][k] = { d: [0,0,0,0,0,0], tw: 0, n: 0 };
          for (let c = 0; c < 6; c++) levels[lvl][k].d[c] += w * rd[c];
          levels[lvl][k].tw += w;
          levels[lvl][k].n += wc;
        }
      }
    }
    for (let lvl = 0; lvl < 3; lvl++) {
      for (const v of Object.values(levels[lvl])) {
        v.p = v.d.map(s => s / v.tw);
      }
    }
    return levels;
  }

  const models = bandwidths.map(bw => buildMultiLevelModel(bw));
  const pseudo = 4;

  // Bayesian shrinkage: blend fine key toward coarser keys
  function shrinkLookup(levels, key) {
    let k0, k1;
    if (key === 'O' || key === 'M') { k0 = key; k1 = key; }
    else { k0 = key[0]; k1 = key.endsWith('c') ? key.slice(0, -1) : key; }
    const ks = [k0, k1, key];
    let prior = null;
    for (let lvl = 0; lvl < 3; lvl++) {
      const entry = levels[lvl][ks[lvl]];
      if (!entry) continue;
      if (!prior) {
        prior = [...entry.p];
      } else {
        const blend = entry.n / (entry.n + pseudo);
        prior = entry.p.map((v, c) => blend * v + (1 - blend) * prior[c]);
      }
    }
    return prior;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      // Ensemble: geometric mean across bandwidths (optimal for KL scoring)
      const bwPreds = [];
      for (const model of models) {
        const p = shrinkLookup(model, key);
        if (p) bwPreds.push(p);
      }

      let prior;
      if (bwPreds.length > 1) {
        prior = new Array(6);
        const invN = 1 / bwPreds.length;
        for (let c = 0; c < 6; c++) {
          let logSum = 0;
          for (const p of bwPreds) logSum += Math.log(Math.max(p[c], 1e-12));
          prior[c] = Math.exp(logSum * invN);
        }
      } else if (bwPreds.length === 1) {
        prior = bwPreds[0];
      } else {
        prior = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      // Adaptive floor: low for peaked cells, higher for uncertain cells
      let ent = 0;
      for (let c = 0; c < 6; c++) {
        if (prior[c] > 1e-6) ent -= prior[c] * Math.log(prior[c]);
      }
      const adaptFloor = floor * (0.05 + 0.95 * (ent / 1.7918));

      const floored = prior.map(v => Math.max(v, adaptFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
