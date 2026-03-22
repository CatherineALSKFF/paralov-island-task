const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 1e-6;
  const sigmaN = config.SN || 0.045;
  const sigmaW = config.SW || 0.16;
  const alpha = config.ALPHA || 1;
  const regWeight = config.REG || 0.20;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const trainRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  function gaussWeight(r, sigma) {
    const gDiff = (growthRates[String(r)] || 0.15) - targetGrowth;
    return Math.exp(-gDiff * gDiff / (2 * sigma * sigma));
  }

  // Build round-level-averaged model (each round = one vote, weighted by Gaussian)
  function buildModel(sigma) {
    const mSum = {}, mCount = {};
    let wTotal = 0;
    const weights = {};
    for (const r of trainRounds) {
      weights[r] = gaussWeight(r, sigma);
      wTotal += weights[r];
    }
    if (wTotal > 0) for (const r of trainRounds) weights[r] /= wTotal;

    for (const r of trainRounds) {
      const bk = perRoundBuckets[r];
      if (!bk) continue;
      const w = weights[r];
      if (w < 1e-15) continue;
      for (const key in bk) {
        const b = bk[key];
        if (!mSum[key]) { mSum[key] = new Float64Array(6); mCount[key] = 0; }
        // Per-round normalized: sum[c]/count gives round's average for this key
        for (let c = 0; c < 6; c++) mSum[key][c] += w * (b.sum[c] / b.count);
        mCount[key] += w;
      }
    }
    const probs = {}, effN = {};
    for (const key in mSum) {
      const cnt = mCount[key];
      if (cnt > 0) {
        probs[key] = new Array(6);
        for (let c = 0; c < 6; c++) probs[key][c] = mSum[key][c] / cnt;
        effN[key] = cnt;
      }
    }
    return { probs, effN };
  }

  const narrow = buildModel(sigmaN);
  const wide = buildModel(sigmaW);

  // Dirichlet-smoothed: (effN * narrow + alpha * wide) / (effN + alpha)
  function getSmoothedProb(key) {
    const nP = narrow.probs[key];
    const wP = wide.probs[key];
    const eN = narrow.effN[key] || 0;
    if (!nP && !wP) return null;
    if (!nP) return wP;
    if (!wP) return nP;
    const result = new Array(6);
    const denom = eN + alpha;
    for (let c = 0; c < 6; c++) result[c] = (eN * nP[c] + alpha * wP[c]) / denom;
    return result;
  }

  // STEP 1: Predict regular cells
  const predGrid = [];
  const isMO = [];
  for (let y = 0; y < H; y++) {
    predGrid.push(new Array(W));
    isMO.push(new Array(W));
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      if (key === 'O' || key === 'M') {
        isMO[y][x] = true;
        predGrid[y][x] = null;
        continue;
      }
      isMO[y][x] = false;

      const fb = key.slice(0, -1);
      const tc = key[0];
      const fine = getSmoothedProb(key);
      const coarse = getSmoothedProb(fb);
      const broad = getSmoothedProb(tc);

      let prior;
      if (fine && coarse) {
        const fineN = narrow.effN[key] || 0;
        const adaptReg = regWeight * Math.max(0.3, 1 - Math.min(fineN / 200, 1) * 0.7);
        prior = new Array(6);
        for (let c = 0; c < 6; c++) prior[c] = (1 - adaptReg) * fine[c] + adaptReg * coarse[c];
      } else if (fine) {
        prior = fine;
      } else if (coarse && broad) {
        prior = new Array(6);
        for (let c = 0; c < 6; c++) prior[c] = 0.8 * coarse[c] + 0.2 * broad[c];
      } else {
        prior = coarse || broad || [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      // Adaptive floor by entropy of prediction
      const ent = -prior.reduce((s, p) => s + (p > 1e-10 ? p * Math.log(p) : 0), 0);
      const aFloor = floor * (0.1 + 0.9 * ent / Math.log(6));

      const floored = prior.map(v => Math.max(v, aFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      predGrid[y][x] = floored.map(v => v / sum);
    }
  }

  // STEP 2: M/O cells — neighbor + surrogate hedging (free for static, crucial for dynamic)
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (!isMO[y][x]) continue;

      // Neighbor average
      const nAvg = [0,0,0,0,0,0];
      let nW = 0;
      for (let dy = -3; dy <= 3; dy++) {
        for (let dx = -3; dx <= 3; dx++) {
          if (dy === 0 && dx === 0) continue;
          const ny = y + dy, nx = x + dx;
          if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
          if (isMO[ny][nx]) continue;
          const p = predGrid[ny][nx];
          if (!p) continue;
          const dist = Math.sqrt(dy * dy + dx * dx);
          const w = 1 / (1 + dist);
          for (let c = 0; c < 6; c++) nAvg[c] += w * p[c];
          nW += w;
        }
      }
      let neighborPred = nW > 0 ? nAvg.map(v => v / nW) : [1/6,1/6,1/6,1/6,1/6,1/6];

      // Terrain surrogates
      let nS = 0;
      for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
        if (!dy && !dx) continue;
        const ny = y + dy, nx = x + dx;
        if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx)) nS++;
      }
      let coast = false;
      for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
        const ny = y + dy, nx = x + dx;
        if (ny >= 0 && ny < H && nx >= 0 && nx < W && initGrid[ny][nx] === 10) coast = true;
      }
      const sBucket = nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3';
      const cSuffix = coast ? 'c' : '';

      const surrogateAvg = [0,0,0,0,0,0];
      let sCount = 0;
      for (const t of ['P', 'F', 'S']) {
        let d = getSmoothedProb(t + sBucket + cSuffix);
        if (!d && cSuffix) d = getSmoothedProb(t + sBucket);
        if (d) { for (let c = 0; c < 6; c++) surrogateAvg[c] += d[c]; sCount++; }
      }
      let surrogatePred = sCount > 0 ? surrogateAvg.map(v => v / sCount) : [1/6,1/6,1/6,1/6,1/6,1/6];

      const blended = new Array(6);
      for (let c = 0; c < 6; c++) {
        blended[c] = 0.40 * neighborPred[c] + 0.35 * surrogatePred[c] + 0.25 / 6;
      }
      const moFloor = 0.02;
      const floored = blended.map(v => Math.max(v, moFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      predGrid[y][x] = floored.map(v => v / sum);
    }
  }

  return predGrid;
}

module.exports = { predict };
