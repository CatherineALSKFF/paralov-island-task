const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.00001;
  const baseLambda = config.BASE_LAMBDA || 15;
  const extremeScale = config.EXTREME_SCALE || 125;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Adaptive lambda: higher when target growth is extreme (far from median)
  const trainGrowths = allRounds.map(r => growthRates[String(r)] || 0.15).sort((a, b) => a - b);
  const median = trainGrowths[Math.floor(trainGrowths.length / 2)];
  const extremeness = Math.abs(targetGrowth - median);
  const lambda = baseLambda + extremeScale * extremeness;

  // Growth-weighted model using all rounds
  const model = {};
  for (const r of allRounds) {
    const b = perRoundBuckets[String(r)];
    if (!b) continue;
    const g = growthRates[String(r)] || 0.15;
    const w = Math.exp(-lambda * Math.abs(g - targetGrowth));
    for (const [k, v] of Object.entries(b)) {
      if (!model[k]) model[k] = { wc: 0, ws: [0, 0, 0, 0, 0, 0] };
      model[k].wc += v.count * w;
      for (let c = 0; c < 6; c++) model[k].ws[c] += v.sum[c] * w;
    }
  }
  const probs = {};
  for (const [k, v] of Object.entries(model)) {
    if (v.wc > 0) probs[k] = v.ws.map(s => s / v.wc);
  }

  // Settlement positions
  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const coarseKey = getFeatureKey(initGrid, settPos, y, x);

      // Compute enriched key using min Chebyshev distance to nearest settlement
      let enrichedKey = coarseKey;
      if (coarseKey !== 'O' && coarseKey !== 'M') {
        let minDist = 99;
        for (const s of settlements) {
          const d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
          if (d < minDist) minDist = d;
        }
        const nKey = coarseKey[1];
        if (nKey === '0') {
          // 0 settlements in R3: sub-keys by distance ring
          enrichedKey = coarseKey + (minDist === 4 ? 'n' : minDist <= 8 ? 'm' : 'f');
        } else if (nKey === '1') {
          // 1-2 settlements in R3: sub-keys by adjacency
          if (minDist === 1) enrichedKey = coarseKey + 'a';
          else if (minDist === 2) enrichedKey = coarseKey + 'b';
        }
      }

      // Lookup: enriched → coarse → coarse-fallback
      let prior = probs[enrichedKey] || probs[coarseKey];
      if (!prior) {
        const fb = coarseKey.slice(0, -1);
        prior = probs[fb];
      }
      if (!prior) {
        prior = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6];
      } else {
        prior = [...prior];
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
