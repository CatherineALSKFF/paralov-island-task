const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRoundNums = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Per-round distribution with shrinkage toward coarser feature key
  const shrinkThreshold = 20;

  function getRoundDist(roundNum, key) {
    const buckets = perRoundBuckets[roundNum];
    if (!buckets) return null;
    const fine = buckets[key];
    const coarseKey = key.slice(0, -1);
    const coarse = buckets[coarseKey];
    if (!fine && !coarse) return null;
    if (fine && coarse) {
      const fineAvg = fine.sum.map(v => v / fine.count);
      const coarseAvg = coarse.sum.map(v => v / coarse.count);
      const shrinkage = 1 / (1 + fine.count / shrinkThreshold);
      return fineAvg.map((v, i) => (1 - shrinkage) * v + shrinkage * coarseAvg[i]);
    }
    if (fine) return fine.sum.map(v => v / fine.count);
    return coarse.sum.map(v => v / coarse.count);
  }

  // Gaussian kernel weights for growth-rate similarity
  function computeWeights(bandwidth) {
    const weights = {};
    for (const r of allRoundNums) {
      const diff = (growthRates[String(r)] || 0.15) - targetGrowth;
      weights[r] = Math.exp(-0.5 * (diff / bandwidth) ** 2);
    }
    return weights;
  }

  // Multi-bandwidth ensemble weighted by effective sample size
  const bandwidths = [0.03, 0.06, 0.12, 0.25];
  const allWeightSets = bandwidths.map(bw => computeWeights(bw));

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      const ensemble = new Float64Array(6);
      let ensembleWeightSum = 0;

      for (const weights of allWeightSets) {
        const dist = new Float64Array(6);
        let wTotal = 0;
        let wSqTotal = 0;
        let contributing = 0;

        for (const r of allRoundNums) {
          const d = getRoundDist(r, key);
          if (!d) continue;
          const w = weights[r];
          for (let c = 0; c < 6; c++) dist[c] += w * d[c];
          wTotal += w;
          wSqTotal += w * w;
          contributing++;
        }

        if (wTotal > 0 && contributing > 0) {
          // Effective sample size as confidence in this bandwidth
          const ess = (wTotal * wTotal) / wSqTotal;
          const bwWeight = Math.sqrt(ess);
          for (let c = 0; c < 6; c++) ensemble[c] += bwWeight * dist[c] / wTotal;
          ensembleWeightSum += bwWeight;
        }
      }

      let prior;
      if (ensembleWeightSum > 0) {
        prior = Array.from(ensemble).map(v => v / ensembleWeightSum);
      } else {
        prior = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      // Adaptive floor: higher for uncertain (high-entropy) cells
      const entropy = prior.reduce((s, p) => p > 0.001 ? s - p * Math.log(p) : s, 0);
      const adaptiveFloor = floor * (1 + 5 * entropy / Math.log(6));

      const floored = prior.map(v => Math.max(v, adaptiveFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };