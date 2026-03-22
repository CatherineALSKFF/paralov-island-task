const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const sigma = config.sigma || 0.05;
  const floor = config.FLOOR || 0.0001;
  const linWeight = config.linWeight || 0.85;
  const loessSigma = config.loessSigma || 0.15;
  const useQuad = config.useQuad !== undefined ? config.useQuad : true;

  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const nearestDist = Array.from({ length: H }, () => Array(W).fill(99));
  for (const s of settlements) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
        if (d < nearestDist[y][x]) nearestDist[y][x] = d;
      }
    }
  }

  function getEnhancedKey(y, x) {
    const coarseKey = getFeatureKey(initGrid, settPos, y, x);
    if (coarseKey === 'O' || coarseKey === 'M') return coarseKey;
    if (coarseKey[0] === 'S') return coarseKey;
    const nKey = coarseKey[1];
    const minDist = nearestDist[y][x];
    if (nKey === '0') return coarseKey + (minDist === 4 ? 'n' : minDist <= 8 ? 'm' : 'f');
    if (nKey === '1') {
      if (minDist === 1) return coarseKey + 'a';
      if (minDist === 2) return coarseKey + 'b';
    }
    return coarseKey;
  }

  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Per-round averages
  const perRoundAvg = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    perRoundAvg[rn] = {};
    for (const [key, val] of Object.entries(b)) {
      perRoundAvg[rn][key] = val.sum.map(v => v / val.count);
    }
  }

  // Gaussian weighted mean model
  const roundWeights = {};
  let tw = 0;
  for (const rn of allRounds) {
    const dist = Math.abs((growthRates[String(rn)] || 0.15) - targetGrowth);
    roundWeights[rn] = Math.exp(-dist * dist / (2 * sigma * sigma));
    tw += roundWeights[rn];
  }
  for (const rn of allRounds) roundWeights[rn] /= tw;

  const meanModel = {};
  for (const rn of allRounds) {
    if (!perRoundAvg[rn]) continue;
    for (const [key, avg] of Object.entries(perRoundAvg[rn])) {
      if (!meanModel[key]) meanModel[key] = [0, 0, 0, 0, 0, 0];
      for (let c = 0; c < 6; c++) meanModel[key][c] += roundWeights[rn] * avg[c];
    }
  }

  // LOESS-weighted quadratic/linear regression per key
  const linCache = {};
  function fitModel(key) {
    if (linCache[key] !== undefined) return linCache[key];
    const points = [];
    for (const rn of allRounds) {
      if (perRoundAvg[rn] && perRoundAvg[rn][key]) {
        const g = growthRates[String(rn)] || 0.15;
        const w = Math.exp(-(g - targetGrowth) * (g - targetGrowth) / (2 * loessSigma * loessSigma));
        points.push({ g, dist: perRoundAvg[rn][key], w });
      }
    }
    if (points.length < 3) { linCache[key] = null; return null; }

    const result = [0, 0, 0, 0, 0, 0];
    const tg = targetGrowth;

    for (let c = 0; c < 6; c++) {
      if (useQuad && points.length >= 5) {
        // Weighted quadratic: p = a + b*g + c*g^2
        let sw=0, swg=0, swg2=0, swg3=0, swg4=0, swp=0, swgp=0, swg2p=0;
        for (const pt of points) {
          const g = pt.g, w = pt.w, p = pt.dist[c];
          sw += w; swg += w*g; swg2 += w*g*g; swg3 += w*g*g*g; swg4 += w*g*g*g*g;
          swp += w*p; swgp += w*g*p; swg2p += w*g*g*p;
        }
        // Solve 3x3 system using Cramer's rule
        const A = [[sw,swg,swg2],[swg,swg2,swg3],[swg2,swg3,swg4]];
        const B = [swp, swgp, swg2p];
        const det3 = (m) => m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
        const D = det3(A);
        if (Math.abs(D) > 1e-15) {
          const Dx = det3([[B[0],A[0][1],A[0][2]],[B[1],A[1][1],A[1][2]],[B[2],A[2][1],A[2][2]]]);
          const Dy = det3([[A[0][0],B[0],A[0][2]],[A[1][0],B[1],A[1][2]],[A[2][0],B[2],A[2][2]]]);
          const Dz = det3([[A[0][0],A[0][1],B[0]],[A[1][0],A[1][1],B[1]],[A[2][0],A[2][1],B[2]]]);
          result[c] = Math.max(0, Dx/D + (Dy/D)*tg + (Dz/D)*tg*tg);
          continue;
        }
      }
      // Fallback to linear
      let sumW=0, sumWG=0, sumWP=0, sumWGG=0, sumWGP=0;
      for (const pt of points) {
        sumW += pt.w; sumWG += pt.w*pt.g; sumWP += pt.w*pt.dist[c];
        sumWGG += pt.w*pt.g*pt.g; sumWGP += pt.w*pt.g*pt.dist[c];
      }
      const denom = sumW*sumWGG - sumWG*sumWG;
      if (Math.abs(denom) < 1e-12) {
        result[c] = sumWP/sumW;
      } else {
        const b = (sumW*sumWGP - sumWG*sumWP)/denom;
        const a = (sumWP - b*sumWG)/sumW;
        result[c] = Math.max(0, a + b*tg);
      }
    }
    const s = result.reduce((a, b) => a + b, 0);
    linCache[key] = s > 0 ? result.map(v => v / s) : null;
    return linCache[key];
  }

  function lookupKey(fineKey, baseKey) {
    let wmDist = meanModel[fineKey] ? [...meanModel[fineKey]] :
                 meanModel[baseKey] ? [...meanModel[baseKey]] : null;
    let lnDist = fitModel(fineKey) || fitModel(baseKey);

    if (wmDist && lnDist) return wmDist.map((v, c) => (1 - linWeight) * v + linWeight * lnDist[c]);
    if (lnDist) return lnDist;
    if (wmDist) return wmDist;
    let fb = baseKey;
    while (fb.length > 1) { fb = fb.slice(0, -1); if (meanModel[fb]) return [...meanModel[fb]]; }
    return [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const enhKey = getEnhancedKey(y, x);
      const coarseKey = getFeatureKey(initGrid, settPos, y, x);
      let prior = lookupKey(enhKey, coarseKey);

      let entropy = 0;
      for (let c = 0; c < 6; c++) if (prior[c] > 0.001) entropy -= prior[c] * Math.log(prior[c]);
      const cellFloor = entropy > 0.5 ? floor : floor * 0.1;
      const floored = prior.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
