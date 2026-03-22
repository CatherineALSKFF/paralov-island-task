const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const trainRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);
  const nTrain = trainRounds.length;

  // Kernel-weighted round weights at two bandwidths (narrow + wide ensemble)
  const bandwidths = [0.04, 0.10];
  const bwBlend = [0.5, 0.5];
  const allBwWeights = bandwidths.map(bw => {
    const w = {};
    let s = 0;
    for (const r of trainRounds) {
      const g = growthRates[String(r)] || 0.15;
      const d = Math.abs(g - targetGrowth);
      w[r] = Math.exp(-d * d / (2 * bw * bw));
      s += w[r];
    }
    for (const r of trainRounds) w[r] /= s;
    return w;
  });

  // Collect all feature keys
  const allKeys = new Set();
  for (const r of trainRounds) {
    for (const key in perRoundBuckets[r]) allKeys.add(key);
  }

  // Build per-bandwidth weighted models + uniform model
  const wModels = bandwidths.map(() => ({}));
  const uniformModel = {};
  const keyRoundCount = {};

  for (const key of allKeys) {
    const uDist = [0, 0, 0, 0, 0, 0];
    let tc = 0, rc = 0;
    const bwDists = bandwidths.map(() => [0, 0, 0, 0, 0, 0]);
    const bwTotals = bandwidths.map(() => 0);

    for (const r of trainRounds) {
      const b = perRoundBuckets[r][key];
      if (b && b.count > 0) {
        const rd = b.sum.map(v => v / b.count);
        for (let c = 0; c < 6; c++) uDist[c] += b.sum[c];
        tc += b.count;
        rc++;
        for (let bi = 0; bi < bandwidths.length; bi++) {
          const rw = allBwWeights[bi][r];
          for (let c = 0; c < 6; c++) bwDists[bi][c] += rw * rd[c];
          bwTotals[bi] += rw;
        }
      }
    }

    if (tc > 0) {
      for (let c = 0; c < 6; c++) uDist[c] /= tc;
      uniformModel[key] = uDist;
    }
    keyRoundCount[key] = rc;
    for (let bi = 0; bi < bandwidths.length; bi++) {
      if (bwTotals[bi] > 0) {
        for (let c = 0; c < 6; c++) bwDists[bi][c] /= bwTotals[bi];
        wModels[bi][key] = bwDists[bi];
      }
    }
  }

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      // Lookup with fallback chain (full key, then trim 1, 2, 3 chars)
      let wEnsemble = null, uPrior = null, roundCount = 0;
      for (let trim = 0; trim <= 3; trim++) {
        const k = trim === 0 ? key : key.slice(0, -trim);
        if (k.length === 0) break;

        if (!wEnsemble) {
          // Ensemble the bandwidth models for this key
          const eDist = [0, 0, 0, 0, 0, 0];
          let eW = 0;
          for (let bi = 0; bi < bandwidths.length; bi++) {
            if (wModels[bi][k]) {
              for (let c = 0; c < 6; c++) eDist[c] += bwBlend[bi] * wModels[bi][k][c];
              eW += bwBlend[bi];
            }
          }
          if (eW > 0) {
            for (let c = 0; c < 6; c++) eDist[c] /= eW;
            wEnsemble = eDist;
            roundCount = keyRoundCount[k] || 0;
          }
        }
        if (!uPrior && uniformModel[k]) {
          uPrior = uniformModel[k];
        }
        if (wEnsemble && uPrior) break;
      }

      // Blend weighted ensemble with uniform based on data coverage
      let prior;
      if (wEnsemble && uPrior) {
        const alpha = Math.min(roundCount / nTrain, 0.85);
        prior = wEnsemble.map((v, c) => alpha * v + (1 - alpha) * uPrior[c]);
      } else if (wEnsemble) {
        prior = [...wEnsemble];
      } else if (uPrior) {
        prior = [...uPrior];
      } else {
        prior = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6];
      }

      // Adaptive floor: confident predictions get lower floor, uncertain get higher
      const maxP = Math.max(...prior);
      const adaptiveFloor = maxP > 0.95 ? floor * 0.3 :
                            maxP > 0.8  ? floor :
                            maxP < 0.4  ? floor * 3 : floor * 1.5;

      const floored = prior.map(v => Math.max(v, adaptiveFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };