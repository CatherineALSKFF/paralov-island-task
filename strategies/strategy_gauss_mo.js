const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const sigma = config.SIGMA || 0.05;
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Per-round normalized distributions
  const perRound = {};
  for (const r of allRounds) {
    const bk = perRoundBuckets[r];
    if (!bk) continue;
    perRound[r] = {};
    for (const key in bk) {
      const b = bk[key];
      if (b.count > 0) perRound[r][key] = b.sum.map(v => v / b.count);
    }
  }

  // Gaussian weights
  const roundW = {};
  for (const r of allRounds) {
    const g = growthRates[String(r)] || 0.15;
    const d = g - targetGrowth;
    roundW[r] = Math.exp(-0.5 * (d / sigma) * (d / sigma));
  }

  // Per-key Gaussian-weighted prediction
  const keyCache = {};
  function getKeyPred(key) {
    if (keyCache[key] !== undefined) return keyCache[key];
    const avg = [0, 0, 0, 0, 0, 0];
    let wSum = 0;
    for (const r of allRounds) {
      if (!perRound[r] || !perRound[r][key]) continue;
      const w = roundW[r];
      const p = perRound[r][key];
      for (let c = 0; c < 6; c++) avg[c] += w * p[c];
      wSum += w;
    }
    if (wSum === 0) { keyCache[key] = null; return null; }
    for (let c = 0; c < 6; c++) avg[c] /= wSum;
    keyCache[key] = avg;
    return avg;
  }

  // Also build K=4 baseline model for blending
  const candidates = { ...growthRates }; delete candidates[String(testRound)];
  const closestRounds = selectClosestRounds(candidates, targetGrowth, 4);
  const adaptiveModel = mergeBuckets(perRoundBuckets, closestRounds);
  const allModel = mergeBuckets(perRoundBuckets, allRounds);

  // Predict regular cells: blend Gaussian + baseline
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

      // Gaussian prediction
      let gaussPred = getKeyPred(key);
      if (!gaussPred) {
        const fb = key.slice(0, -1);
        gaussPred = getKeyPred(fb);
      }

      // Baseline prediction
      let basePred = adaptiveModel[key] || allModel[key] || null;
      if (!basePred) {
        const fb = key.slice(0, -1);
        basePred = adaptiveModel[fb] || allModel[fb] || null;
      }

      let prior;
      if (gaussPred && basePred) {
        // Blend: 50% Gaussian + 50% baseline
        prior = gaussPred.map((v, i) => 0.5 * v + 0.5 * basePred[i]);
      } else {
        prior = gaussPred || basePred || [1/6,1/6,1/6,1/6,1/6,1/6];
      }

      const floored = prior.map(v => Math.max(v, floor));
      const sum = floored.reduce((a, b) => a + b, 0);
      predGrid[y][x] = floored.map(v => v / sum);
    }
  }

  // M/O hedging
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (!isMO[y][x]) continue;

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
        let d = getKeyPred(t + sBucket + cSuffix);
        if (!d && cSuffix) d = getKeyPred(t + sBucket);
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
