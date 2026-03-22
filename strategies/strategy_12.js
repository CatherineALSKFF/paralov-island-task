const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const bw = 0.042;
  const asymGrowth = 4.0;
  const deathThresh = 0.04;
  const floor = 0.0001;

  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Asymmetric Laplace kernel: for growth rounds, penalize lower-growth training
  // rounds more than higher-growth ones (settlements tend to be more stable than
  // growth rate alone predicts)
  const asym = targetGrowth > deathThresh ? asymGrowth : 1.0;
  const gw = {};
  for (const r of allRounds) {
    const g = growthRates[String(r)] || 0.15;
    const d = g - targetGrowth;
    const effD = d < 0 ? Math.abs(d) * asym : Math.abs(d);
    gw[r] = Math.exp(-effD / bw);
  }

  // Pre-compute per-round normalized average distributions
  const pra = {};
  for (const r of allRounds) {
    const bk = perRoundBuckets[String(r)];
    if (!bk) continue;
    pra[r] = {};
    for (const [k, v] of Object.entries(bk)) {
      pra[r][k] = v.sum.map(s => s / v.count);
    }
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const fbKey = key.length > 1 ? key.slice(0, -1) : key;
      const fine = [0, 0, 0, 0, 0, 0];
      let fw = 0;
      for (const r of allRounds) {
        if (!pra[r]) continue;
        const f = pra[r][key] || pra[r][fbKey];
        if (f) {
          const w = gw[r];
          for (let c = 0; c < 6; c++) fine[c] += w * f[c];
          fw += w;
        }
      }
      if (fw === 0) { row.push([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]); continue; }
      for (let c = 0; c < 6; c++) fine[c] /= fw;
      let sum = 0;
      const out = new Array(6);
      for (let c = 0; c < 6; c++) { out[c] = Math.max(fine[c], floor); sum += out[c]; }
      for (let c = 0; c < 6; c++) out[c] /= sum;
      row.push(out);
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
