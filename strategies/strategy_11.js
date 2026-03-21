const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Per-round normalized distributions (each round contributes equally before weighting)
  const perRound = {};
  for (const r of allRounds) {
    const bk = perRoundBuckets[r];
    if (!bk) continue;
    perRound[r] = {};
    for (const key in bk) {
      const b = bk[key];
      if (b.count > 0) {
        const total = b.sum.reduce((a, v) => a + v, 0);
        if (total > 0) perRound[r][key] = b.sum.map(v => v / total);
      }
    }
  }

  // Build growth-weighted fine-key model for given round weights
  function buildFineModel(ws) {
    const model = {};
    for (const r of allRounds) {
      if (!perRound[r]) continue;
      const w = ws[r];
      for (const key in perRound[r]) {
        if (!model[key]) model[key] = { d: new Float64Array(6), tw: 0 };
        const v = perRound[r][key];
        for (let c = 0; c < 6; c++) model[key].d[c] += w * v[c];
        model[key].tw += w;
      }
    }
    for (const key in model) {
      const m = model[key];
      if (m.tw > 0) for (let c = 0; c < 6; c++) m.d[c] /= m.tw;
    }
    return model;
  }

  // Build terrain-level aggregated model (aggregate all keys sharing first char)
  function buildTerrainModel(ws) {
    const model = {};
    for (const r of allRounds) {
      if (!perRound[r]) continue;
      const w = ws[r];
      const sums = {}, counts = {};
      for (const key in perRound[r]) {
        const tc = key[0];
        if (!sums[tc]) { sums[tc] = [0, 0, 0, 0, 0, 0]; counts[tc] = 0; }
        const v = perRound[r][key];
        for (let c = 0; c < 6; c++) sums[tc][c] += v[c];
        counts[tc]++;
      }
      for (const tc in sums) {
        if (!model[tc]) model[tc] = { d: new Float64Array(6), tw: 0 };
        const n = counts[tc];
        for (let c = 0; c < 6; c++) model[tc].d[c] += w * (sums[tc][c] / n);
        model[tc].tw += w;
      }
    }
    for (const tc in model) {
      const m = model[tc];
      if (m.tw > 0) for (let c = 0; c < 6; c++) m.d[c] /= m.tw;
    }
    return model;
  }

  // Bandwidth ensemble: wide (conservative) to narrow (growth-specific)
  const lambdas = [2, 5, 12, 25, 50];

  const roundWS = lambdas.map(lambda => {
    const w = {};
    let s = 0;
    for (const r of allRounds) {
      const g = growthRates[String(r)] || 0.15;
      w[r] = Math.exp(-lambda * Math.abs(g - targetGrowth));
      s += w[r];
    }
    for (const r of allRounds) w[r] /= s;
    return w;
  });

  // Pre-build all models
  const fineModels = roundWS.map(ws => buildFineModel(ws));
  const terrModels = roundWS.map(ws => buildTerrainModel(ws));
  const nBw = lambdas.length;

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const fb = key.length > 1 ? key.slice(0, -1) : null;
      const tc = key[0];

      const ens = [0, 0, 0, 0, 0, 0];
      let ensN = 0;

      for (let bi = 0; bi < nBw; bi++) {
        const fm = fineModels[bi];
        const tm = terrModels[bi];

        const fine = fm[key] || null;
        const mid = fb ? (fm[fb] || null) : null;
        const coarse = tm[tc] || null;

        // Two-stage hierarchical shrinkage
        // Stage 1: regularize mid toward coarse
        let midReg = null;
        if (mid && coarse) {
          const a = Math.min(mid.tw * 2, 0.8);
          midReg = new Array(6);
          for (let c = 0; c < 6; c++) midReg[c] = a * mid.d[c] + (1 - a) * coarse.d[c];
        } else {
          midReg = mid ? mid.d : (coarse ? coarse.d : null);
        }

        // Stage 2: regularize fine toward midReg (or coarse)
        const reg = midReg || (coarse ? coarse.d : null);
        let d;
        if (fine && reg) {
          const a = Math.min(fine.tw * 2, 0.85);
          d = new Array(6);
          for (let c = 0; c < 6; c++) d[c] = a * fine.d[c] + (1 - a) * reg[c];
        } else if (fine) {
          d = fine.d;
        } else if (reg) {
          d = reg;
        } else {
          continue;
        }

        for (let c = 0; c < 6; c++) ens[c] += d[c];
        ensN++;
      }

      let final;
      if (ensN > 0) {
        final = ens.map(v => v / ensN);
      } else {
        final = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6];
      }

      const floored = final.map(v => Math.max(v, floor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
