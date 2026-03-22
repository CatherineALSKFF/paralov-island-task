const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const sigma = config.sigma || 0.14;
  const loessWeight = config.loessWeight || 0.95;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  // Build settlement position set and array
  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);
  const settArr = settlements.map(s => [s.y, s.x]);

  // Exclude test round from training
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Compute Gaussian weights by growth-rate similarity
  const roundData = [];
  let totalWeight = 0;
  for (const rn of allRounds) {
    const rate = growthRates[String(rn)];
    if (rate === undefined) continue;
    const dist = Math.abs(rate - targetGrowth);
    const w = Math.exp(-dist * dist / (2 * sigma * sigma));
    roundData.push({ rn, rate, w });
    totalWeight += w;
  }

  // LOESS: weighted local linear regression on growth rate
  function loessPredict(key) {
    let sW = 0, sWX = 0, sWXX = 0, cnt = 0;
    const sWY = [0,0,0,0,0,0], sWXY = [0,0,0,0,0,0];

    for (const rd of roundData) {
      const b = perRoundBuckets[String(rd.rn)];
      if (!b || !b[key]) continue;
      const avg = b[key].sum.map(v => v / b[key].count);
      sW += rd.w;
      sWX += rd.w * rd.rate;
      sWXX += rd.w * rd.rate * rd.rate;
      cnt++;
      for (let c = 0; c < 6; c++) {
        sWY[c] += rd.w * avg[c];
        sWXY[c] += rd.w * rd.rate * avg[c];
      }
    }

    if (cnt < 2 || sW < 1e-10) {
      return sW > 1e-10 ? sWY.map(v => v / sW) : null;
    }

    const xbar = sWX / sW;
    const denom = sWXX - sWX * xbar;

    if (Math.abs(denom) < 1e-15) {
      return sWY.map(v => v / sW);
    }

    const result = new Array(6);
    for (let c = 0; c < 6; c++) {
      const ybar = sWY[c] / sW;
      const slope = Math.max(-3, Math.min(3, (sWXY[c] - sWX * ybar) / denom));
      result[c] = Math.max(0, ybar + slope * (targetGrowth - xbar));
    }

    const sum = result.reduce((a, b) => a + b, 0);
    if (sum < 1e-10) return sWY.map(v => v / sW);
    return result.map(v => v / sum);
  }

  // Kernel regression: Gaussian-weighted average (degree-0 polynomial)
  function kernelPredict(key) {
    let sW = 0;
    const result = [0,0,0,0,0,0];
    for (const rd of roundData) {
      const b = perRoundBuckets[String(rd.rn)];
      if (!b || !b[key]) continue;
      const avg = b[key].sum.map(v => v / b[key].count);
      sW += rd.w;
      for (let c = 0; c < 6; c++) result[c] += rd.w * avg[c];
    }
    return sW > 1e-10 ? result.map(v => v / sW) : null;
  }

  // Prediction cache: LOESS-kernel ensemble
  const cache = {};
  function getPrediction(key) {
    if (cache[key] !== undefined) return cache[key];
    const lp = loessPredict(key);
    const kp = kernelPredict(key);
    if (lp && kp) {
      cache[key] = lp.map((v, c) => v * loessWeight + kp[c] * (1 - loessWeight));
    } else {
      cache[key] = lp || kp;
    }
    return cache[key];
  }

  // Chebyshev distance to nearest settlement for sub-key computation
  const nearestDist = Array.from({ length: H }, () => Array(W).fill(99));
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      for (const [sy, sx] of settArr) {
        const d = Math.max(Math.abs(y - sy), Math.abs(x - sx));
        if (d < nearestDist[y][x]) nearestDist[y][x] = d;
      }
    }
  }

  // Enhanced feature key with distance-based sub-keys
  // nS=0: n (dist=4), m (dist 5-8), f (dist 9+)
  // nS=1-2: a (dist=1, adjacent), b (dist=2)
  function getEnhancedKey(y, x) {
    const coarseKey = getFeatureKey(initGrid, settPos, y, x);
    if (coarseKey === 'O' || coarseKey === 'M') return coarseKey;
    if (coarseKey[0] === 'S') return coarseKey;

    const nKey = coarseKey[1];
    const minDist = nearestDist[y][x];

    if (nKey === '0') {
      return coarseKey + (minDist === 4 ? 'n' : minDist <= 8 ? 'm' : 'f');
    } else if (nKey === '1') {
      if (minDist === 1) return coarseKey + 'a';
      if (minDist === 2) return coarseKey + 'b';
    }
    return coarseKey;
  }

  // Generate predictions
  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const enhKey = getEnhancedKey(y, x);
      const coarseKey = getFeatureKey(initGrid, settPos, y, x);

      // Lookup: enhanced key → coarse key → drop coastal → terrain+'0'
      let prior = getPrediction(enhKey) || getPrediction(coarseKey);
      if (!prior) {
        const fb = coarseKey.endsWith('c') ? coarseKey.slice(0, -1) : null;
        if (fb) prior = getPrediction(fb);
      }
      if (!prior) {
        prior = getPrediction(coarseKey[0] + '0') || [1/6,1/6,1/6,1/6,1/6,1/6];
      }
      prior = [...prior];

      // Adaptive floor: confident predictions get smaller floor to avoid dilution
      let entropy = 0;
      for (let c = 0; c < 6; c++) {
        if (prior[c] > 0.001) entropy -= prior[c] * Math.log(prior[c]);
      }
      const adaptFloor = entropy < 0.1 ? floor * 0.1 : entropy < 0.3 ? floor * 0.3 : floor;

      const floored = prior.map(v => Math.max(v, adaptFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
