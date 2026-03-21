const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const regWeight = config.REG_WEIGHT || 0.45;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const trainRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const lambdas = [4, 8, 16];
  const roundWeightsByLambda = lambdas.map(lambda => {
    const w = {};
    for (const r of trainRounds) {
      const diff = Math.abs((growthRates[String(r)] || 0.15) - targetGrowth);
      w[r] = Math.exp(-lambda * diff);
    }
    return w;
  });

  const roundProbs = {};
  for (const r of trainRounds) {
    roundProbs[r] = {};
    const buckets = perRoundBuckets[r];
    if (!buckets) continue;
    for (const key in buckets) {
      const b = buckets[key];
      const p = new Array(6);
      for (let c = 0; c < 6; c++) p[c] = b.sum[c] / b.count;
      roundProbs[r][key] = p;
    }
  }

  const allKeys = new Set();
  for (const r of trainRounds) {
    for (const key in roundProbs[r]) allKeys.add(key);
  }

  const ensembleModel = {};
  for (const key of allKeys) {
    const ens = new Float64Array(6);
    let ensCt = 0;
    for (let li = 0; li < lambdas.length; li++) {
      const wts = roundWeightsByLambda[li];
      const acc = new Float64Array(6);
      let wT = 0;
      for (const r of trainRounds) {
        if (!roundProbs[r][key]) continue;
        const w = wts[r];
        const rp = roundProbs[r][key];
        for (let c = 0; c < 6; c++) acc[c] += w * rp[c];
        wT += w;
      }
      if (wT > 0) {
        for (let c = 0; c < 6; c++) ens[c] += acc[c] / wT;
        ensCt++;
      }
    }
    if (ensCt > 0) {
      const res = new Array(6);
      for (let c = 0; c < 6; c++) res[c] = ens[c] / ensCt;
      ensembleModel[key] = res;
    }
  }

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const coarseKey = key.slice(0, -1);
      const fineP = ensembleModel[key];
      const coarseP = ensembleModel[coarseKey];

      let prior;
      if (fineP && coarseP) {
        prior = new Array(6);
        for (let c = 0; c < 6; c++) {
          prior[c] = (1 - regWeight) * fineP[c] + regWeight * coarseP[c];
        }
      } else {
        prior = fineP || coarseP || [1/6,1/6,1/6,1/6,1/6,1/6];
      }

      let maxP = 0;
      for (let c = 0; c < 6; c++) if (prior[c] > maxP) maxP = prior[c];
      const af = maxP > 0.95 ? floor * 0.1 : maxP > 0.8 ? floor : floor * 2;

      const floored = prior.map(v => Math.max(v, af));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };