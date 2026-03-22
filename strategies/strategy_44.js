const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const sigma = config.sigma || 0.05;
  const geoBlend = config.geoBlend || 0.4;
  const regBlend = config.regBlend || 0.3;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Compute round weights
  const roundWeights = {};
  let wTotal = 0;
  for (const rn of allRounds) {
    const g = growthRates[String(rn)];
    if (g === undefined) continue;
    const d = Math.abs(g - targetGrowth);
    const w = Math.exp(-0.5 * (d / sigma) ** 2);
    roundWeights[rn] = w;
    wTotal += w;
  }
  for (const rn of allRounds) roundWeights[rn] = (roundWeights[rn] || 0) / (wTotal || 1);

  // Collect per-round per-key averages
  const keyRounds = {};
  for (const rn of allRounds) {
    const rb = perRoundBuckets[String(rn)];
    if (!rb) continue;
    for (const [key, val] of Object.entries(rb)) {
      if (!keyRounds[key]) keyRounds[key] = [];
      keyRounds[key].push({
        rn,
        dist: val.sum.map(v => v / val.count),
        weight: roundWeights[rn] || 0,
      });
    }
  }

  const keyModel = {};
  for (const [key, rounds] of Object.entries(keyRounds)) {
    let sw = 0;
    for (const r of rounds) sw += r.weight;
    if (sw === 0) continue;

    // Arithmetic mean
    const arith = [0,0,0,0,0,0];
    for (const r of rounds) {
      for (let c = 0; c < 6; c++) arith[c] += r.weight * r.dist[c];
    }
    for (let c = 0; c < 6; c++) arith[c] /= sw;

    // Geometric mean (log-space)
    const logAvg = [0,0,0,0,0,0];
    for (const r of rounds) {
      if (r.weight < 1e-8) continue;
      for (let c = 0; c < 6; c++) {
        logAvg[c] += r.weight * Math.log(Math.max(r.dist[c], 1e-8));
      }
    }
    const geo = new Array(6);
    for (let c = 0; c < 6; c++) geo[c] = Math.exp(logAvg[c] / sw);
    const geoSum = geo.reduce((a, b) => a + b, 0);
    if (geoSum > 0) for (let c = 0; c < 6; c++) geo[c] /= geoSum;

    // Local linear regression
    const reg = [...arith];
    if (rounds.length >= 4) {
      let swg = 0, swg2 = 0;
      for (const r of rounds) {
        const g = growthRates[String(r.rn)] || 0.15;
        swg += r.weight * g;
        swg2 += r.weight * g * g;
      }
      const gMean = swg / sw;
      const gVar = swg2 / sw - gMean * gMean;
      if (gVar > 1e-10) {
        for (let c = 0; c < 6; c++) {
          let swgp = 0;
          for (const r of rounds) swgp += r.weight * (growthRates[String(r.rn)] || 0.15) * r.dist[c];
          const b = (swgp / sw - gMean * arith[c]) / gVar;
          reg[c] = Math.max(arith[c] + b * (targetGrowth - gMean), 0);
        }
        const regSum = reg.reduce((a, b) => a + b, 0);
        if (regSum > 0) for (let c = 0; c < 6; c++) reg[c] /= regSum;
      }
    }

    // Combine: (1-geoBlend-regBlend)*arith + geoBlend*geo + regBlend*reg
    const arithW = 1 - geoBlend - regBlend;
    const blended = new Array(6);
    for (let c = 0; c < 6; c++) {
      blended[c] = arithW * arith[c] + geoBlend * geo[c] + regBlend * reg[c];
    }

    keyModel[key] = { dist: blended, nRounds: rounds.length };
  }

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      let prior = null;

      if (keyModel[key]) {
        prior = [...keyModel[key].dist];
        // Shrink toward coarser key
        if (key.length > 1) {
          const nR = keyModel[key].nRounds;
          for (let trim = 1; trim < key.length; trim++) {
            const coarse = key.slice(0, -trim);
            if (keyModel[coarse]) {
              const alpha = nR / (nR + 4);
              for (let c = 0; c < 6; c++) {
                prior[c] = alpha * prior[c] + (1 - alpha) * keyModel[coarse].dist[c];
              }
              break;
            }
          }
        }
      }

      if (!prior) {
        for (let trim = 1; trim < key.length; trim++) {
          const coarse = key.slice(0, -trim);
          if (keyModel[coarse]) {
            prior = [...keyModel[coarse].dist];
            break;
          }
        }
      }

      if (!prior) prior = [1/6,1/6,1/6,1/6,1/6,1/6];

      // Adaptive floor
      const entropy = -prior.reduce((s, p) => s + (p > 1e-12 ? p * Math.log(p) : 0), 0);
      const maxEnt = Math.log(6);
      const ratio = entropy / maxEnt;
      const adaptFloor = floor * (0.5 + 4.0 * ratio * ratio);

      const floored = prior.map(v => Math.max(v, adaptFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
