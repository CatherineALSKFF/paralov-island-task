const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const sigma = config.sigma || 0.15;
  const floor = config.FLOOR || 0.0001;
  const regWeight = config.regWeight || 0.05;
  const linBlend = config.linBlend || 0.95;
  const temp = config.temp || 1.1;

  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Compute min Chebyshev distance to nearest settlement
  const nearestDist = Array.from({ length: H }, () => Array(W).fill(99));
  for (const s of settlements) {
    for (let y = Math.max(0, s.y - 40); y < Math.min(H, s.y + 41); y++) {
      for (let x = Math.max(0, s.x - 40); x < Math.min(W, s.x + 41); x++) {
        const d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
        if (d < nearestDist[y][x]) nearestDist[y][x] = d;
      }
    }
  }

  // Enhanced feature key matching bucket format
  function getEnhancedKey(y, x) {
    const ck = getFeatureKey(initGrid, settPos, y, x);
    if (ck === 'O' || ck === 'M' || ck[0] === 'S') return ck;
    const nk = ck[1], d = nearestDist[y][x];
    if (nk === '0') return ck + (d === 4 ? 'n' : d <= 8 ? 'm' : 'f');
    if (nk === '1') { if (d === 1) return ck + 'a'; if (d === 2) return ck + 'b'; }
    return ck;
  }

  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Round weights (Gaussian kernel on growth rate distance)
  const rw = {};
  let tw = 0;
  for (const rn of allRounds) {
    const dist = Math.abs((growthRates[String(rn)] || 0.15) - targetGrowth);
    const w = Math.exp(-dist * dist / (2 * sigma * sigma));
    rw[rn] = w; tw += w;
  }

  // Build weighted mean model
  const meanModel = {};
  for (const rn of allRounds) {
    const w = rw[rn] / tw;
    const rb = perRoundBuckets[String(rn)];
    if (!rb) continue;
    for (const [key, val] of Object.entries(rb)) {
      if (!meanModel[key]) meanModel[key] = [0, 0, 0, 0, 0, 0];
      const avg = val.sum.map(v => v / val.count);
      for (let c = 0; c < 6; c++) meanModel[key][c] += w * avg[c];
    }
  }

  // Build weighted linear regression model: p(c|key) = intercept + slope * growth
  const linModel = {};
  const allKeys = new Set();
  for (const rn of allRounds) {
    const rb = perRoundBuckets[String(rn)];
    if (rb) for (const k of Object.keys(rb)) allKeys.add(k);
  }

  for (const key of allKeys) {
    const points = [];
    for (const rn of allRounds) {
      const rb = perRoundBuckets[String(rn)];
      if (!rb || !rb[key]) continue;
      points.push({
        g: growthRates[String(rn)] || 0.15,
        p: rb[key].sum.map(v => v / rb[key].count),
        w: rw[rn] / tw
      });
    }
    if (points.length < 5) continue;

    let sw = 0, swG = 0, swG2 = 0;
    for (const pt of points) { sw += pt.w; swG += pt.w * pt.g; swG2 += pt.w * pt.g * pt.g; }
    const wMG = swG / sw, wVG = swG2 / sw - wMG * wMG;
    if (wVG < 1e-10) continue;

    const pred = [0, 0, 0, 0, 0, 0];
    for (let c = 0; c < 6; c++) {
      let swP = 0, swGP = 0;
      for (const pt of points) { swP += pt.w * pt.p[c]; swGP += pt.w * pt.g * pt.p[c]; }
      const wMP = swP / sw;
      const slope = (swGP / sw - wMG * wMP) / wVG;
      pred[c] = Math.max(wMP + slope * (targetGrowth - wMG), 0);
    }
    linModel[key] = pred;
  }

  // Lookup helpers
  function lookupMean(fk, bk) {
    let p = null;
    if (meanModel[fk]) {
      p = [...meanModel[fk]];
      if (fk !== bk && meanModel[bk])
        for (let c = 0; c < 6; c++) p[c] = (1 - regWeight) * p[c] + regWeight * meanModel[bk][c];
    } else if (meanModel[bk]) {
      p = [...meanModel[bk]];
    } else {
      let fb = bk;
      while (fb.length > 1) { fb = fb.slice(0, -1); if (meanModel[fb]) { p = [...meanModel[fb]]; break; } }
    }
    return p || [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
  }

  function lookupLin(fk, bk) {
    if (linModel[fk]) return [...linModel[fk]];
    if (linModel[bk]) return [...linModel[bk]];
    let fb = bk;
    while (fb.length > 1) { fb = fb.slice(0, -1); if (linModel[fb]) return [...linModel[fb]]; }
    return null;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const ek = getEnhancedKey(y, x);
      const ck = getFeatureKey(initGrid, settPos, y, x);

      let p = lookupMean(ek, ck);

      // Blend with linear model
      const lin = lookupLin(ek, ck);
      if (lin) for (let c = 0; c < 6; c++) p[c] = (1 - linBlend) * p[c] + linBlend * lin[c];

      // Temperature scaling: temp > 1 softens distributions, reducing overconfidence
      if (temp !== 1.0) {
        const logP = p.map(v => Math.log(Math.max(v, 1e-12)) / temp);
        const maxLP = Math.max(...logP);
        const expP = logP.map(v => Math.exp(v - maxLP));
        const s = expP.reduce((a, b) => a + b, 0);
        p = expP.map(v => v / s);
      }

      // Adaptive floor
      let ent = 0;
      for (let c = 0; c < 6; c++) if (p[c] > 0.001) ent -= p[c] * Math.log(p[c]);
      const cf = ent > 0.5 ? floor : floor * 0.1;

      const fl = p.map(v => Math.max(v, cf));
      const sum = fl.reduce((a, b) => a + b, 0);
      row.push(fl.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
