const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const trainRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const sigmas = [0.03, 0.06, 0.12, 0.30];

  function computeWeights(sigma) {
    const w = {};
    let total = 0;
    for (const rn of trainRounds) {
      const g = growthRates[String(rn)] || 0.15;
      const diff = Math.abs(g - targetGrowth);
      w[rn] = Math.exp(-0.5 * (diff / sigma) ** 2);
      total += w[rn];
    }
    if (total > 0) for (const rn of trainRounds) w[rn] /= total;
    return w;
  }

  function weightedMerge(roundWeights) {
    const dist = {};
    const counts = {};
    for (const rn of trainRounds) {
      const w = roundWeights[rn];
      if (w < 1e-15) continue;
      const buckets = perRoundBuckets[rn];
      if (!buckets) continue;
      for (const [key, bucket] of Object.entries(buckets)) {
        if (!dist[key]) { dist[key] = new Float64Array(6); counts[key] = 0; }
        for (let c = 0; c < 6; c++) dist[key][c] += w * bucket.sum[c];
        counts[key] += w * bucket.count;
      }
    }
    for (const key of Object.keys(dist)) {
      if (counts[key] > 0) {
        for (let c = 0; c < 6; c++) dist[key][c] /= counts[key];
      }
    }
    return { dist, counts };
  }

  const models = sigmas.map(s => weightedMerge(computeWeights(s)));

  const uniformW = {};
  for (const rn of trainRounds) uniformW[rn] = 1 / trainRounds.length;
  const uniformModel = weightedMerge(uniformW);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const baseReg = 0.4;
  const pred = [];
  const logSix = Math.log(6);

  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const ensemble = new Float64Array(6);

      for (const model of models) {
        let fine = model.dist[key];
        let fineCount = model.counts[key] || 0;

        let coarse = null;
        for (let trim = 1; trim < key.length; trim++) {
          const sk = key.slice(0, -trim);
          if (sk.length === 0) break;
          if (model.dist[sk]) { coarse = model.dist[sk]; break; }
        }

        if (!fine) {
          fine = uniformModel.dist[key];
          fineCount = uniformModel.counts[key] || 0;
        }
        if (!fine && coarse) { fine = coarse; coarse = null; }
        if (!fine) {
          for (let trim = 1; trim < key.length; trim++) {
            const sk = key.slice(0, -trim);
            if (sk.length === 0) break;
            if (uniformModel.dist[sk]) { fine = uniformModel.dist[sk]; break; }
          }
        }
        if (!fine) fine = new Float64Array([1/6,1/6,1/6,1/6,1/6,1/6]);

        const reg = coarse ? baseReg / (1 + fineCount / 30) : 0;
        for (let c = 0; c < 6; c++) {
          ensemble[c] += coarse ? (1 - reg) * fine[c] + reg * coarse[c] : fine[c];
        }
      }

      const nM = models.length;
      let entropy = 0;
      for (let c = 0; c < 6; c++) {
        ensemble[c] /= nM;
        if (ensemble[c] > 0) entropy -= ensemble[c] * Math.log(ensemble[c]);
      }

      const cellFloor = floor * (0.1 + 0.9 * entropy / logSix);
      const result = new Array(6);
      let sum = 0;
      for (let c = 0; c < 6; c++) {
        result[c] = Math.max(ensemble[c], cellFloor);
        sum += result[c];
      }
      for (let c = 0; c < 6; c++) result[c] /= sum;
      row.push(result);
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };