const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = 0.00006;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const trainRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // --- Round composition profiles for multi-dim matching ---
  function getRoundProfile(buckets) {
    let nS = 0, nF = 0, nO = 0, nM = 0, total = 0;
    for (const [key, val] of Object.entries(buckets)) {
      const c = val.count; total += c;
      const t = key[0];
      if (t === 'O') nO += c;
      else if (t === 'M') nM += c;
      else if (t === 'S') nS += c;
      else if (t === 'F') nF += c;
    }
    if (total === 0) total = 1;
    return [nS / total, nF / total, nO / total, nM / total];
  }

  // Test round profile from initGrid
  let tS = 0, tF = 0, tO = 0, tM = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const v = initGrid[y][x];
    if (v === 10) tO++;
    else if (v === 5) tM++;
    else if (v === 4) tF++;
    else if (v === 1 || v === 2) tS++;
  }
  const N = H * W;
  const testProfile = [tS / N, tF / N, tO / N, tM / N];

  const trainProfiles = {};
  for (const rn of trainRounds) trainProfiles[rn] = getRoundProfile(perRoundBuckets[rn]);

  // --- Weighted merge with multi-dim similarity ---
  function weightedMerge(sigmaG, sigmaC) {
    const w = {};
    let wt = 0;
    for (const rn of trainRounds) {
      const gd = Math.abs((growthRates[String(rn)] || 0.15) - targetGrowth);
      let rw = Math.exp(-0.5 * (gd / sigmaG) ** 2);
      if (sigmaC > 0) {
        const prof = trainProfiles[rn];
        let cd2 = 0;
        for (let i = 0; i < testProfile.length; i++) cd2 += (prof[i] - testProfile[i]) ** 2;
        rw *= Math.exp(-0.5 * cd2 / (sigmaC * sigmaC));
      }
      w[rn] = rw; wt += rw;
    }
    if (wt > 0) for (const rn of trainRounds) w[rn] /= wt;

    const dist = {}, counts = {};
    for (const rn of trainRounds) {
      if (w[rn] < 1e-15) continue;
      const b = perRoundBuckets[rn]; if (!b) continue;
      for (const [key, val] of Object.entries(b)) {
        if (!dist[key]) { dist[key] = new Float64Array(6); counts[key] = 0; }
        for (let c = 0; c < 6; c++) dist[key][c] += w[rn] * val.sum[c];
        counts[key] += w[rn] * val.count;
      }
    }
    for (const key of Object.keys(dist)) {
      if (counts[key] > 0) for (let c = 0; c < 6; c++) dist[key][c] /= counts[key];
    }
    return { dist, counts };
  }

  // Three models at different bandwidths
  const tight  = weightedMerge(0.03, 0.06);
  const medium = weightedMerge(0.07, 0.10);
  const wide   = weightedMerge(0.25, 0);

  // Feature hierarchy lookup
  function getKeyHierarchy(key) {
    const levels = [key];
    if (key.length > 1 && key.endsWith('c')) levels.push(key.slice(0, -1));
    const t = key[0];
    if (t !== 'O' && t !== 'M') {
      if (key.endsWith('c')) levels.push(t + 'c');
      levels.push(t);
    }
    return levels;
  }

  function lookupWithReg(model, hierarchy, regWeight) {
    let pred = null, dataCount = 0;
    for (const lk of hierarchy) {
      const d = model.dist[lk];
      if (!d) continue;
      const c = model.counts[lk] || 0;
      if (!pred) {
        pred = new Float64Array(d);
        dataCount = c;
      } else {
        const rw = Math.min(regWeight, regWeight / (1 + dataCount / 20));
        for (let cc = 0; cc < 6; cc++) pred[cc] = (1 - rw) * pred[cc] + rw * d[cc];
      }
    }
    return pred;
  }

  const logSix = Math.log(6);
  const pred = [];

  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const hier = getKeyHierarchy(key);

      const tP = lookupWithReg(tight,  hier, 0.35);
      const mP = lookupWithReg(medium, hier, 0.35);
      const wP = lookupWithReg(wide,   hier, 0.30);

      // Adaptive blending: measure disagreement between tight and wide
      let disagree = 0;
      if (tP && wP) {
        for (let c = 0; c < 6; c++) disagree += Math.abs(tP[c] - wP[c]);
        disagree /= 2; // normalize to [0,1]
      }

      // When tight and wide agree: trust tight (specific).
      // When they disagree: shift toward wide (robust).
      // disagree ~0 → tightW=0.50, medW=0.30, wideW=0.20
      // disagree ~0.5 → tightW=0.15, medW=0.25, wideW=0.60
      const alpha = Math.min(1, disagree * 3);
      const tW = 0.50 * (1 - alpha) + 0.10 * alpha;
      const mW = 0.30 * (1 - alpha) + 0.25 * alpha;
      const wW = 0.20 * (1 - alpha) + 0.65 * alpha;

      const ensemble = new Float64Array(6);
      let totalW = 0;
      if (tP) { for (let c = 0; c < 6; c++) ensemble[c] += tW * tP[c]; totalW += tW; }
      if (mP) { for (let c = 0; c < 6; c++) ensemble[c] += mW * mP[c]; totalW += mW; }
      if (wP) { for (let c = 0; c < 6; c++) ensemble[c] += wW * wP[c]; totalW += wW; }

      if (totalW > 0) for (let c = 0; c < 6; c++) ensemble[c] /= totalW;
      else for (let c = 0; c < 6; c++) ensemble[c] = 1 / 6;

      // Per-cell adaptive floor based on prediction entropy
      let entropy = 0;
      for (let c = 0; c < 6; c++) {
        if (ensemble[c] > 1e-10) entropy -= ensemble[c] * Math.log(ensemble[c]);
      }
      const entropyRatio = entropy / logSix;
      const cellFloor = floor * (0.05 + 0.95 * entropyRatio);

      const result = new Array(6);
      let sum = 0;
      for (let c = 0; c < 6; c++) {
        result[c] = Math.max(ensemble[c], cellFloor);
        sum += result[c];
      }
      for (let c = 0; c < 6; c++) result[c] /= sum;
      row.push(result);
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
