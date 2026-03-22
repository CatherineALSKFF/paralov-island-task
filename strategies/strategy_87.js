const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const sigma = config.sigma || 0.08;
  const floor = config.FLOOR || 0.0001;
  const linWeight = config.linWeight || 0.88;
  const loessSigma = config.loessSigma || 0.15;
  const safetyBlend = config.safetyBlend || 0.10;

  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const nSett = settlements.length;
  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const nearestDist = new Uint8Array(H * W).fill(99);
  for (const s of settlements) {
    for (let y = Math.max(0, s.y - 12); y <= Math.min(H - 1, s.y + 12); y++)
      for (let x = Math.max(0, s.x - 12); x <= Math.min(W - 1, s.x + 12); x++) {
        const d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
        const idx = y * W + x;
        if (d < nearestDist[idx]) nearestDist[idx] = d;
      }
  }

  function getEnhancedKey(y, x) {
    const ck = getFeatureKey(initGrid, settPos, y, x);
    if (ck === 'O' || ck === 'M' || ck[0] === 'S') return ck;
    const nKey = ck[1], minDist = nearestDist[y * W + x];
    if (nKey === '0') return ck + (minDist <= 5 ? 'n' : minDist <= 7 ? 'm' : 'f');
    if (nKey === '1') return ck + (minDist <= 1 ? 'a' : 'b');
    return ck;
  }

  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const roundNSett = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    let sc = 0;
    for (const key in b) if (key[0] === 'S') sc += b[key].count;
    roundNSett[rn] = sc / 5;
  }

  // Coastal settlement counts
  let testCoastalSett = 0;
  for (const s of settlements) {
    for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
      const ny = s.y + dy, nx = s.x + dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && initGrid[ny][nx] === 10) {
        testCoastalSett++; break;
      }
    }
  }
  const roundCoastalSett = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    let cs = 0;
    for (const key in b) if (key[0] === 'S' && key.endsWith('c')) cs += b[key].count;
    roundCoastalSett[rn] = cs / 5;
  }

  // Augmented buckets
  const augBuckets = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    augBuckets[rn] = {};
    for (const key in b) augBuckets[rn][key] = b[key].sum.map(v => v / b[key].count);
    for (const [base, sub1, sub2] of [['P1','P1a','P1b'],['P1c','P1ca','P1cb'],['F1','F1a','F1b'],['F1c','F1ca','F1cb']]) {
      if (!b[base]) continue;
      const baseN = b[base].count, sub1N = b[sub1]?b[sub1].count:0, sub2N = b[sub2]?b[sub2].count:0;
      const residN = baseN - sub1N - sub2N;
      if (residN < 5) continue;
      const dDist = [];
      for (let c = 0; c < 6; c++) {
        const total = b[base].sum[c], s1v = b[sub1]?b[sub1].sum[c]:0, s2v = b[sub2]?b[sub2].sum[c]:0;
        dDist.push(Math.max(0, (total - s1v - s2v) / residN));
      }
      augBuckets[rn][base + 'd'] = dDist;
    }
  }

  // Robust LOESS with per-class IRLS
  const linCache = {};
  function fitRobustLoess(key) {
    if (linCache[key] !== undefined) return linCache[key];
    const points = [];
    for (const rn of allRounds) {
      if (!augBuckets[rn] || !augBuckets[rn][key]) continue;
      const g = growthRates[String(rn)] || 0.15;
      const gdiff = g - targetGrowth;
      const sdiff = ((roundNSett[rn] || 40) - nSett) / 40;
      const cdiff = ((roundCoastalSett[rn] || 0) - testCoastalSett) / 10;
      const w = Math.exp(-gdiff*gdiff/(2*loessSigma*loessSigma)
                         - sdiff*sdiff/(2*0.3*0.3)
                         - cdiff*cdiff/(2*0.5*0.5));
      points.push({ g, dist: augBuckets[rn][key], w });
    }
    if (points.length < 3) { linCache[key] = null; return null; }

    const HUBER_K = 0.08;
    const result = new Array(6);

    for (let c = 0; c < 6; c++) {
      let sw = 0, swg = 0, swp = 0, swgg = 0, swgp = 0;
      for (const pt of points) {
        sw += pt.w; swg += pt.w*pt.g; swp += pt.w*pt.dist[c];
        swgg += pt.w*pt.g*pt.g; swgp += pt.w*pt.g*pt.dist[c];
      }
      let denom = sw*swgg - swg*swg;
      let slope0, int0;
      if (Math.abs(denom) < 1e-12) { slope0 = 0; int0 = swp/sw; }
      else { slope0 = (sw*swgp - swg*swp)/denom; int0 = (swp - slope0*swg)/sw; }

      // IRLS pass
      sw = swg = swp = swgg = swgp = 0;
      for (const pt of points) {
        const pred = int0 + slope0*pt.g;
        const resid = Math.abs(pt.dist[c] - pred);
        const hW = resid > HUBER_K ? HUBER_K/resid : 1;
        const w = pt.w * hW;
        sw += w; swg += w*pt.g; swp += w*pt.dist[c];
        swgg += w*pt.g*pt.g; swgp += w*pt.g*pt.dist[c];
      }
      denom = sw*swgg - swg*swg;
      if (Math.abs(denom) < 1e-12) result[c] = swp/sw;
      else {
        const slope = (sw*swgp - swg*swp)/denom;
        const intercept = (swp - slope*swg)/sw;
        result[c] = Math.max(0, intercept + slope*targetGrowth);
      }
    }
    const s = result.reduce((a, b) => a + b, 0);
    linCache[key] = s > 0 ? result.map(v => v/s) : null;
    return linCache[key];
  }

  // Gaussian weighted mean model (3D)
  const roundWeights = {};
  let tw = 0;
  for (const rn of allRounds) {
    const gdiff = (growthRates[String(rn)] || 0.15) - targetGrowth;
    const sdiff = ((roundNSett[rn] || 40) - nSett) / 40;
    const cdiff = ((roundCoastalSett[rn] || 0) - testCoastalSett) / 10;
    roundWeights[rn] = Math.exp(-gdiff*gdiff/(2*sigma*sigma)
                                -sdiff*sdiff/(2*0.3*0.3)
                                -cdiff*cdiff/(2*0.5*0.5));
    tw += roundWeights[rn];
  }
  for (const rn of allRounds) roundWeights[rn] /= tw;

  const meanModel = {};
  for (const rn of allRounds) {
    if (!augBuckets[rn]) continue;
    for (const key in augBuckets[rn]) {
      if (!meanModel[key]) meanModel[key] = [0,0,0,0,0,0];
      const avg = augBuckets[rn][key];
      for (let c = 0; c < 6; c++) meanModel[key][c] += roundWeights[rn] * avg[c];
    }
  }

  // All-rounds safety model (uniform weights)
  const safeModel = {};
  for (const rn of allRounds) {
    if (!augBuckets[rn]) continue;
    for (const key in augBuckets[rn]) {
      if (!safeModel[key]) safeModel[key] = { sum: [0,0,0,0,0,0], n: 0 };
      const avg = augBuckets[rn][key];
      for (let c = 0; c < 6; c++) safeModel[key].sum[c] += avg[c];
      safeModel[key].n++;
    }
  }
  const safeAvg = {};
  for (const key in safeModel) {
    safeAvg[key] = safeModel[key].sum.map(v => v / safeModel[key].n);
  }

  // Per-key cross-round disagreement for adaptive safety blend
  const keyDisagreement = {};
  for (const key in safeModel) {
    const avg = safeAvg[key];
    let dis = 0, n = 0;
    for (const rn of allRounds) {
      if (!augBuckets[rn] || !augBuckets[rn][key]) continue;
      const d = augBuckets[rn][key];
      for (let c = 0; c < 6; c++) dis += (d[c] - avg[c]) ** 2;
      n++;
    }
    keyDisagreement[key] = n > 1 ? Math.sqrt(dis / n) : 0;
  }

  function lookupKey(fineKey, baseKey) {
    const lnFine = fitRobustLoess(fineKey);
    const lnBase = fitRobustLoess(baseKey);
    const wmFine = meanModel[fineKey] ? meanModel[fineKey].slice() : null;
    const wmBase = meanModel[baseKey] ? meanModel[baseKey].slice() : null;

    let lnDist;
    if (lnFine && lnBase && fineKey !== baseKey) {
      lnDist = lnFine.map((v, c) => 0.7*v + 0.3*lnBase[c]);
    } else {
      lnDist = lnFine || lnBase;
    }
    const wmDist = wmFine || wmBase;

    let mainPred;
    if (wmDist && lnDist) mainPred = wmDist.map((v, c) => (1-linWeight)*v + linWeight*lnDist[c]);
    else if (lnDist) mainPred = lnDist;
    else if (wmDist) mainPred = wmDist;
    else {
      const terrKey = baseKey[0];
      const lnTerr = fitRobustLoess(terrKey);
      if (lnTerr) return lnTerr;
      if (meanModel[terrKey]) return meanModel[terrKey].slice();
      return [1/6,1/6,1/6,1/6,1/6,1/6];
    }

    // Safety blend: mix with all-rounds average, proportional to disagreement
    const safeDist = safeAvg[baseKey] || safeAvg[baseKey[0]] || null;
    if (safeDist) {
      const dis = keyDisagreement[baseKey] || 0;
      // More disagreement → more safety blend (up to safetyBlend * 3)
      const alpha = Math.min(0.40, safetyBlend + dis * 2.0);
      return mainPred.map((v, c) => (1-alpha)*v + alpha*safeDist[c]);
    }
    return mainPred;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const enhKey = getEnhancedKey(y, x);
      const coarseKey = getFeatureKey(initGrid, settPos, y, x);
      const prior = lookupKey(enhKey, coarseKey);

      let entropy = 0;
      for (let c = 0; c < 6; c++) if (prior[c] > 0.001) entropy -= prior[c] * Math.log(prior[c]);
      const cellFloor = floor * (0.1 + Math.min(entropy, 1.5));

      const floored = prior.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
