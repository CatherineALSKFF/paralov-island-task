const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Nearest settlement distance for each cell
  const nearestDist = new Uint8Array(H * W).fill(99);
  for (const s of settlements) {
    for (let dy = -15; dy <= 15; dy++) {
      for (let dx = -15; dx <= 15; dx++) {
        const ny = s.y + dy, nx = s.x + dx;
        if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
        const d = Math.max(Math.abs(dy), Math.abs(dx));
        const idx = ny * W + nx;
        if (d < nearestDist[idx]) nearestDist[idx] = d;
      }
    }
  }

  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Per-round averages
  const roundAvgs = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    roundAvgs[rn] = {};
    for (const [key, val] of Object.entries(b)) {
      roundAvgs[rn][key] = val.sum.map(v => v / val.count);
    }
  }

  function weightedMedian(values, weights) {
    const pairs = values.map((v, i) => ({ v, w: weights[i] })).sort((a, b) => a.v - b.v);
    const totalW = pairs.reduce((a, p) => a + p.w, 0);
    let cumW = 0;
    for (const p of pairs) {
      cumW += p.w;
      if (cumW >= totalW * 0.5) return p.v;
    }
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
      for (let c = 0; c < 6; c++) {
        result[c] = weightedMedian(rns.map(rn => roundAvgs[rn][key][c]), weights);
      }
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

  const loessNarrow = buildLoessModel(0.10);
  const loessWide = buildLoessModel(0.18);
  const narrowModel = buildGaussModel(0.05);
  const wideModel = buildGaussModel(0.20);
  const medianNarrow = buildMedianModel(0.06);
  const medianWide = buildMedianModel(0.15);

  function lookup(model, key) {
    if (model[key]) return model[key];
    for (let len = key.length - 1; len >= 1; len--) {
      if (model[key.slice(0, len)]) return model[key.slice(0, len)];
    }
    return null;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const terrain = key[0];
      const bucket = key.length > 1 ? key[1] : null;
      const minDist = nearestDist[y * W + x];

      const lN = lookup(loessNarrow, key);
      const lW = lookup(loessWide, key);
      const narrow = lookup(narrowModel, key);
      const wide = lookup(wideModel, key);
      const medN = lookup(medianNarrow, key);
      const medW = lookup(medianWide, key);

      // Distance-adaptive weighting: cells far from settlements benefit more
      // from growth-rate-specific (narrow) models
      const farFromSett = (bucket === '0' || (bucket === '1' && minDist > 2));
      const nearSett = (bucket === '2' || bucket === '3' || terrain === 'S');

      let prior = [0, 0, 0, 0, 0, 0];
      let tw = 0;

      if (farFromSett) {
        // Far cells: emphasize LOESS narrow + narrow Gauss (growth-rate specific)
        if (lN) { for (let c = 0; c < 6; c++) prior[c] += 0.35 * lN[c]; tw += 0.35; }
        if (narrow) { for (let c = 0; c < 6; c++) prior[c] += 0.25 * narrow[c]; tw += 0.25; }
        if (medN) { for (let c = 0; c < 6; c++) prior[c] += 0.20 * medN[c]; tw += 0.20; }
        if (medW) { for (let c = 0; c < 6; c++) prior[c] += 0.10 * medW[c]; tw += 0.10; }
        if (wide) { for (let c = 0; c < 6; c++) prior[c] += 0.10 * wide[c]; tw += 0.10; }
      } else if (nearSett) {
        // Near-settlement cells: more robust blending
        if (lW) { for (let c = 0; c < 6; c++) prior[c] += 0.25 * lW[c]; tw += 0.25; }
        if (narrow) { for (let c = 0; c < 6; c++) prior[c] += 0.15 * narrow[c]; tw += 0.15; }
        if (medN) { for (let c = 0; c < 6; c++) prior[c] += 0.20 * medN[c]; tw += 0.20; }
        if (medW) { for (let c = 0; c < 6; c++) prior[c] += 0.20 * medW[c]; tw += 0.20; }
        if (wide) { for (let c = 0; c < 6; c++) prior[c] += 0.20 * wide[c]; tw += 0.20; }
      } else {
        // Default blend (similar to 84.1 proven weights)
        if (lN) { for (let c = 0; c < 6; c++) prior[c] += 0.30 * lN[c]; tw += 0.30; }
        if (narrow) { for (let c = 0; c < 6; c++) prior[c] += 0.20 * narrow[c]; tw += 0.20; }
        if (medN) { for (let c = 0; c < 6; c++) prior[c] += 0.20 * medN[c]; tw += 0.20; }
        if (medW) { for (let c = 0; c < 6; c++) prior[c] += 0.15 * medW[c]; tw += 0.15; }
        if (wide) { for (let c = 0; c < 6; c++) prior[c] += 0.15 * wide[c]; tw += 0.15; }
      }

      if (tw > 0) for (let c = 0; c < 6; c++) prior[c] /= tw;
      else prior = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];

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
