const { H, W, getFeatureKey } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const sigma = config.SIGMA || 0.11;
  const floor = config.FLOOR || 0.0001;
  const shrinkLambda = config.SL || 0.3;
  const tempCoeff = config.TEMP || 0.75;
  const regBlend = config.RB || 0.98;

  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Gaussian weights per round
  const roundWeights = {};
  let wSum = 0;
  for (const r of allRounds) {
    const g = growthRates[String(r)] || 0.15;
    const d = g - targetGrowth;
    roundWeights[r] = Math.exp(-0.5 * (d / sigma) * (d / sigma));
    wSum += roundWeights[r];
  }
  if (wSum > 0) for (const r of allRounds) roundWeights[r] /= wSum;

  // Collect per-key per-round data
  const keyRoundData = {};
  for (const r of allRounds) {
    const b = perRoundBuckets[String(r)];
    if (!b) continue;
    const w = roundWeights[r];
    const g = growthRates[String(r)] || 0.15;
    for (const [key, val] of Object.entries(b)) {
      if (!keyRoundData[key]) keyRoundData[key] = [];
      keyRoundData[key].push({ dist: val.sum.map(v => v / val.count), weight: w, growth: g });
    }
  }

  // Weighted average + local linear regression
  const keyStats = {};
  for (const [key, data] of Object.entries(keyRoundData)) {
    const avg = [0,0,0,0,0,0];
    let tw = 0;
    for (const d of data) {
      for (let c = 0; c < 6; c++) avg[c] += d.weight * d.dist[c];
      tw += d.weight;
    }
    if (tw > 0) for (let c = 0; c < 6; c++) avg[c] /= tw;

    // Disagreement
    let dis = 0;
    if (data.length >= 2) {
      for (let c = 0; c < 6; c++) {
        let wVar = 0;
        for (const d of data) {
          const diff = d.dist[c] - avg[c];
          wVar += d.weight * diff * diff;
        }
        dis += Math.sqrt(wVar / tw);
      }
    }

    // Local linear regression
    let regPred = null;
    if (data.length >= 3) {
      let gMean = 0;
      for (const d of data) gMean += d.weight * d.growth;
      gMean /= tw;

      let gVar = 0;
      for (const d of data) {
        const dg = d.growth - gMean;
        gVar += d.weight * dg * dg;
      }
      gVar /= tw;

      if (gVar > 1e-10) {
        regPred = new Array(6);
        for (let c = 0; c < 6; c++) {
          let cov = 0;
          for (const d of data) {
            cov += d.weight * (d.growth - gMean) * (d.dist[c] - avg[c]);
          }
          cov /= tw;
          regPred[c] = Math.max(0, avg[c] + (cov / gVar) * (targetGrowth - gMean));
        }
        const rSum = regPred.reduce((a, b) => a + b, 0);
        if (rSum > 0) for (let c = 0; c < 6; c++) regPred[c] /= rSum;
      }
    }

    let final;
    if (regPred) {
      final = avg.map((v, c) => (1 - regBlend) * v + regBlend * regPred[c]);
    } else {
      final = avg;
    }

    keyStats[key] = { avg: final, dis, nRounds: data.length };
  }

  function lookup(key) {
    if (key === 'O') return { probs: [1,0,0,0,0,0], dis: 0 };
    if (key === 'M') return { probs: [0,0,0,0,0,1], dis: 0 };

    let result = null;
    let nEff = 0;
    let dis = 0;

    if (keyStats[key]) {
      result = [...keyStats[key].avg];
      nEff = keyStats[key].nRounds;
      dis = keyStats[key].dis;
    }

    for (let trim = 1; trim < key.length && trim <= 2; trim++) {
      const coarse = key.slice(0, -trim);
      if (!coarse || !keyStats[coarse]) continue;
      if (result) {
        const alpha = nEff / (nEff + shrinkLambda);
        const cp = keyStats[coarse].avg;
        for (let c = 0; c < 6; c++) result[c] = alpha * result[c] + (1 - alpha) * cp[c];
        dis = Math.max(dis, keyStats[coarse].dis * 0.5);
      } else {
        result = [...keyStats[coarse].avg];
        nEff = keyStats[coarse].nRounds;
        dis = keyStats[coarse].dis;
      }
      break;
    }

    return { probs: result || [1/6,1/6,1/6,1/6,1/6,1/6], dis };
  }

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      let { probs, dis } = lookup(key);

      if (dis > 0.1 && tempCoeff > 0) {
        const temp = 1.0 + tempCoeff * Math.min(dis, 1.0);
        let s = 0;
        for (let c = 0; c < 6; c++) {
          probs[c] = Math.pow(Math.max(probs[c], 1e-12), 1 / temp);
          s += probs[c];
        }
        for (let c = 0; c < 6; c++) probs[c] /= s;
      }

      const entropy = -probs.reduce((s, v) => s + (v > 1e-12 ? v * Math.log(v) : 0), 0);
      const ratio = entropy / Math.log(6);
      const adaptFloor = floor * (1 + 8 * ratio * ratio);

      const floored = probs.map(v => Math.max(v, adaptFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
