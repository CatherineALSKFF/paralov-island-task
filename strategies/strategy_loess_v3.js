const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Nearest settlement distance
  const nearestDist = new Uint8Array(H * W).fill(99);
  for (const s of settlements) {
    for (let dy = -15; dy <= 15; dy++) for (let dx = -15; dx <= 15; dx++) {
      const ny = s.y + dy, nx = s.x + dx;
      if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
      const d = Math.max(Math.abs(dy), Math.abs(dx));
      const idx = ny * W + nx;
      if (d < nearestDist[idx]) nearestDist[idx] = d;
    }
  }

  function getEnrichedKey(y, x) {
    const base = getFeatureKey(initGrid, settPos, y, x);
    if (base === 'O' || base === 'M' || base[0] === 'S') return base;
    const nKey = base[1];
    const minDist = nearestDist[y * W + x];
    if (nKey === '0') return base + (minDist <= 5 ? 'n' : minDist <= 7 ? 'm' : 'f');
    if (nKey === '1') {
      if (minDist === 1) return base + 'a';
      if (minDist === 2) return base + 'b';
    }
    return base;
  }

  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const roundAvgs = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    roundAvgs[rn] = {};
    for (const [key, val] of Object.entries(b))
      roundAvgs[rn][key] = val.sum.map(v => v / val.count);
    // Residual 'd' keys (same as 84.1)
    for (const [base, sub1, sub2] of [
      ['P1', 'P1a', 'P1b'], ['P1c', 'P1ca', 'P1cb'],
      ['F1', 'F1a', 'F1b'], ['F1c', 'F1ca', 'F1cb'],
    ]) {
      if (!b[base]) continue;
      const baseN = b[base].count;
      const sub1N = b[sub1] ? b[sub1].count : 0;
      const sub2N = b[sub2] ? b[sub2].count : 0;
      const residN = baseN - sub1N - sub2N;
      if (residN < 2) continue;
      const dDist = [0, 0, 0, 0, 0, 0];
      for (let c = 0; c < 6; c++) {
        const s1 = b[sub1] ? b[sub1].sum[c] : 0;
        const s2 = b[sub2] ? b[sub2].sum[c] : 0;
        dDist[c] = Math.max(0, (b[base].sum[c] - s1 - s2) / residN);
      }
      roundAvgs[rn][base + 'd'] = dDist;
    }
  }

  function weightedMedian(values, weights) {
    const pairs = values.map((v, i) => ({ v, w: weights[i] })).sort((a, b) => a.v - b.v);
    const totalW = pairs.reduce((a, p) => a + p.w, 0);
    let cumW = 0;
    for (const p of pairs) { cumW += p.w; if (cumW >= totalW * 0.5) return p.v; }
    return pairs[pairs.length - 1].v;
  }

  function buildMedianModel(sigma) {
    const model = {};
    const ws = {};
    for (const rn of allRounds) {
      const diff = (growthRates[String(rn)] || 0.15) - targetGrowth;
      ws[rn] = Math.exp(-diff * diff / (2 * sigma * sigma));
    }
    const allKeys = new Set();
    for (const rn of allRounds) if (roundAvgs[rn]) for (const k of Object.keys(roundAvgs[rn])) allKeys.add(k);
    for (const key of allKeys) {
      const rns = allRounds.filter(rn => roundAvgs[rn] && roundAvgs[rn][key]);
      if (rns.length < 2) continue;
      const weights = rns.map(rn => ws[rn]);
      const result = [0, 0, 0, 0, 0, 0];
      for (let c = 0; c < 6; c++)
        result[c] = weightedMedian(rns.map(rn => roundAvgs[rn][key][c]), weights);
      const s = result.reduce((a, b) => a + b, 0);
      if (s > 0) model[key] = result.map(v => v / s);
    }
    return model;
  }

  function buildGaussModel(sigma) {
    const model = {};
    let tw = 0;
    const ws = {};
    for (const rn of allRounds) {
      const diff = (growthRates[String(rn)] || 0.15) - targetGrowth;
      ws[rn] = Math.exp(-diff * diff / (2 * sigma * sigma));
      tw += ws[rn];
    }
    for (const rn of allRounds) ws[rn] /= tw;
    for (const rn of allRounds) {
      if (!roundAvgs[rn]) continue;
      for (const [key, avg] of Object.entries(roundAvgs[rn])) {
        if (!model[key]) model[key] = [0, 0, 0, 0, 0, 0];
        for (let c = 0; c < 6; c++) model[key][c] += ws[rn] * avg[c];
      }
    }
    return model;
  }

  function buildLoessModel(bw) {
    const model = {};
    const allKeys = new Set();
    for (const rn of allRounds) if (roundAvgs[rn]) for (const k of Object.keys(roundAvgs[rn])) allKeys.add(k);
    for (const key of allKeys) {
      const points = [];
      for (const rn of allRounds) {
        if (!roundAvgs[rn] || !roundAvgs[rn][key]) continue;
        const g = growthRates[String(rn)] || 0.15;
        const diff = g - targetGrowth;
        points.push({ g, dist: roundAvgs[rn][key], w: Math.exp(-diff * diff / (2 * bw * bw)) });
      }
      if (points.length < 3) continue;
      const result = [0, 0, 0, 0, 0, 0];
      for (let c = 0; c < 6; c++) {
        let sw = 0, swg = 0, swp = 0, swgg = 0, swgp = 0;
        for (const pt of points) {
          sw += pt.w; swg += pt.w * pt.g; swp += pt.w * pt.dist[c];
          swgg += pt.w * pt.g * pt.g; swgp += pt.w * pt.g * pt.dist[c];
        }
        const denom = sw * swgg - swg * swg;
        if (Math.abs(denom) < 1e-12) result[c] = swp / sw;
        else {
          const b = (sw * swgp - swg * swp) / denom;
          const a = (swp - b * swg) / sw;
          result[c] = Math.max(0, a + b * targetGrowth);
        }
      }
      const s = result.reduce((a, b) => a + b, 0);
      if (s > 0) model[key] = result.map(v => v / s);
    }
    return model;
  }

  const loessModel = buildLoessModel(0.12);
  const narrowModel = buildGaussModel(0.05);
  const wideModel = buildGaussModel(0.20);
  const medianNarrow = buildMedianModel(0.06);
  const medianWide = buildMedianModel(0.15);

  // Also build a second LOESS at wider bandwidth for robustness
  const loessWide = buildLoessModel(0.20);

  function lookup(model, enrichedKey, baseKey) {
    if (model[enrichedKey]) return model[enrichedKey];
    if (model[baseKey]) return model[baseKey];
    for (let len = baseKey.length - 1; len >= 1; len--) {
      if (model[baseKey.slice(0, len)]) return model[baseKey.slice(0, len)];
    }
    return null;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const enrichedKey = getEnrichedKey(y, x);
      const baseKey = getFeatureKey(initGrid, settPos, y, x);

      const loess = lookup(loessModel, enrichedKey, baseKey);
      const lW = lookup(loessWide, enrichedKey, baseKey);
      const narrow = lookup(narrowModel, enrichedKey, baseKey);
      const wide = lookup(wideModel, enrichedKey, baseKey);
      const medN = lookup(medianNarrow, enrichedKey, baseKey);
      const medW = lookup(medianWide, enrichedKey, baseKey);

      // Same proven weights as 84.1, plus small weight for wide LOESS
      let prior = [0, 0, 0, 0, 0, 0];
      let tw = 0;
      if (loess) { for (let c = 0; c < 6; c++) prior[c] += 0.30 * loess[c]; tw += 0.30; }
      if (lW) { for (let c = 0; c < 6; c++) prior[c] += 0.05 * lW[c]; tw += 0.05; }
      if (narrow) { for (let c = 0; c < 6; c++) prior[c] += 0.20 * narrow[c]; tw += 0.20; }
      if (medN) { for (let c = 0; c < 6; c++) prior[c] += 0.20 * medN[c]; tw += 0.20; }
      if (medW) { for (let c = 0; c < 6; c++) prior[c] += 0.15 * medW[c]; tw += 0.15; }
      if (wide) { for (let c = 0; c < 6; c++) prior[c] += 0.10 * wide[c]; tw += 0.10; }

      if (tw > 0) for (let c = 0; c < 6; c++) prior[c] /= tw;
      else prior = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];

      // Shrink enriched → coarse (proven in 84.1)
      if (enrichedKey !== baseKey) {
        const coarse = narrowModel[baseKey];
        if (coarse) for (let c = 0; c < 6; c++) prior[c] = 0.90 * prior[c] + 0.10 * coarse[c];
      }

      let entropy = 0;
      for (let c = 0; c < 6; c++) if (prior[c] > 0.001) entropy -= prior[c] * Math.log(prior[c]);
      const cellFloor = entropy < 0.1 ? floor * 0.05 :
                        entropy < 0.3 ? floor * 0.3 :
                        entropy > 1.2 ? floor * 4 :
                        entropy > 0.8 ? floor * 2 : floor;

      const floored = prior.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
