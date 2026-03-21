const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const bandwidths = [0.04, 0.08, 0.20];
  const reg = 0.2;
  const uniformBlend = 0.03;

  const growthDiffs = {};
  for (const r of allRounds) {
    growthDiffs[r] = Math.abs((growthRates[String(r)] || 0.15) - targetGrowth);
  }

  const weightSets = bandwidths.map(bw => {
    const w = {};
    for (const r of allRounds) {
      w[r] = Math.exp(-growthDiffs[r] * growthDiffs[r] / (2 * bw * bw));
    }
    return w;
  });

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const fbKey = key.slice(0, -1);

      const ensemble = new Float64Array(6);
      const nBW = bandwidths.length;

      for (let bi = 0; bi < nBW; bi++) {
        const rw = weightSets[bi];

        const fine = new Float64Array(6);
        let fineW = 0;
        const coarse = new Float64Array(6);
        let coarseW = 0;

        for (const r of allRounds) {
          const buckets = perRoundBuckets[r];
          if (!buckets) continue;
          const w = rw[r];

          const b = buckets[key];
          if (b && b.count > 0) {
            for (let c = 0; c < 6; c++) fine[c] += w * b.sum[c] / b.count;
            fineW += w;
          }

          const bfb = buckets[fbKey];
          if (bfb && bfb.count > 0) {
            for (let c = 0; c < 6; c++) coarse[c] += w * bfb.sum[c] / bfb.count;
            coarseW += w;
          }
        }

        let prior;
        if (fineW > 0 && coarseW > 0) {
          prior = new Float64Array(6);
          for (let c = 0; c < 6; c++)
            prior[c] = (1 - reg) * fine[c] / fineW + reg * coarse[c] / coarseW;
        } else if (fineW > 0) {
          prior = new Float64Array(6);
          for (let c = 0; c < 6; c++) prior[c] = fine[c] / fineW;
        } else if (coarseW > 0) {
          prior = new Float64Array(6);
          for (let c = 0; c < 6; c++) prior[c] = coarse[c] / coarseW;
        } else {
          prior = new Float64Array(6);
          for (let c = 0; c < 6; c++) prior[c] = 1 / 6;
        }

        for (let c = 0; c < 6; c++) ensemble[c] += prior[c];
      }

      for (let c = 0; c < 6; c++) {
        ensemble[c] = (1 - uniformBlend) * ensemble[c] / nBW + uniformBlend / 6;
      }

      let ent = 0;
      for (let c = 0; c < 6; c++) {
        if (ensemble[c] > 1e-10) ent -= ensemble[c] * Math.log(ensemble[c]);
      }
      const entRatio = ent / Math.log(6);
      const adaptiveFloor = 0.001 + 0.009 * entRatio;

      let sum = 0;
      for (let c = 0; c < 6; c++) {
        ensemble[c] = Math.max(ensemble[c], adaptiveFloor);
        sum += ensemble[c];
      }

      const result = new Array(6);
      for (let c = 0; c < 6; c++) result[c] = ensemble[c] / sum;
      row.push(result);
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };