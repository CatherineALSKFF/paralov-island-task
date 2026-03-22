const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function getRichKey(grid, settPos, y, x) {
  const v = grid[y][x];
  if (v === 10) return 'O';
  if (v === 5) return 'M';
  const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P';
  let nS = 0, minDist = 99;
  for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
    if (!dy && !dx) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    if (settPos.has(ny * W + nx)) {
      nS++;
      const d = Math.max(Math.abs(dy), Math.abs(dx));
      if (d < minDist) minDist = d;
    }
  }
  let coast = false;
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) coast = true;
  }
  const nSBucket = nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3';
  const c = coast ? 'c' : '';
  if (t === 'S' || nS === 0 || nS > 2) return t + nSBucket + c;
  let distSuffix = '';
  if (minDist === 1) distSuffix = 'a';
  else if (minDist === 2) distSuffix = 'b';
  return t + nSBucket + c + distSuffix;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Dual-sigma: tight for precise matching, wide for safety
  function computeGaussWeights(sigma) {
    const w = {};
    let total = 0;
    for (const r of allRounds) {
      const g = growthRates[String(r)] || 0.15;
      const d = Math.abs(g - targetGrowth);
      w[r] = Math.exp(-0.5 * (d / sigma) ** 2);
      total += w[r];
    }
    for (const r of allRounds) w[r] /= total;
    return w;
  }

  const tightWeights = computeGaussWeights(config.sigma1 || 0.04);
  const wideWeights = computeGaussWeights(config.sigma2 || 0.09);
  const blend = config.sigmaBlend || 0.65; // weight on tight
  const roundWeights = {};
  for (const r of allRounds) roundWeights[r] = blend * tightWeights[r] + (1 - blend) * wideWeights[r];

  function buildWeightedProbs(wts) {
    const buckets = {};
    for (const r of allRounds) {
      const rb = perRoundBuckets[r];
      if (!rb) continue;
      const w = wts[r];
      for (const key in rb) {
        const b = rb[key];
        if (!buckets[key]) buckets[key] = { count: 0, sum: new Float64Array(6) };
        buckets[key].count += b.count * w;
        for (let c = 0; c < 6; c++) buckets[key].sum[c] += b.sum[c] * w;
      }
    }
    const probs = {};
    for (const key in buckets) {
      const s = buckets[key].sum;
      const tot = s[0] + s[1] + s[2] + s[3] + s[4] + s[5];
      if (tot > 0) probs[key] = { p: Array.from(s, v => v / tot), n: buckets[key].count };
    }
    return probs;
  }

  const gaussModel = buildWeightedProbs(roundWeights);

  const K = config.K || 2;
  const candidates = { ...growthRates };
  delete candidates[String(testRound)];
  const closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  const topKWeights = {};
  for (const r of allRounds) topKWeights[r] = 0;
  for (const r of closestRounds) topKWeights[r] = 1 / closestRounds.length;
  const topKModel = buildWeightedProbs(topKWeights);

  const uniWeights = {};
  for (const r of allRounds) uniWeights[r] = 1 / allRounds.length;
  const uniModel = buildWeightedProbs(uniWeights);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  function getKeyChain(key) {
    if (key === 'O' || key === 'M') return [key];
    const chain = [key];
    const t = key[0];
    if (key.length >= 4) { chain.push(key.slice(0, -1)); chain.push(t + key[1]); chain.push(t); }
    else if (key.length === 3) { chain.push(t + key[1]); chain.push(t); }
    else if (key.length === 2) { chain.push(t); }
    return chain;
  }

  function lookupWithShrinkage(model, keyChain) {
    let result = null;
    let nEff = 0;
    const shrinkStrength = config.shrink || 2;
    for (const k of keyChain) {
      if (!model[k]) continue;
      if (!result) { result = [...model[k].p]; nEff = model[k].n; }
      else {
        const alpha = nEff / (nEff + shrinkStrength);
        const cp = model[k].p;
        for (let c = 0; c < 6; c++) result[c] = alpha * result[c] + (1 - alpha) * cp[c];
        nEff += model[k].n * 0.3;
      }
    }
    return result;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getRichKey(initGrid, settPos, y, x);
      const keyChain = getKeyChain(key);

      const pGauss = lookupWithShrinkage(gaussModel, keyChain);
      const pTopK = lookupWithShrinkage(topKModel, keyChain);
      const pUni = lookupWithShrinkage(uniModel, keyChain);

      const wGauss = config.wGauss || 0.7;
      const wTopK = config.wTopK || 0.2;
      const wUni = config.wUni || 0.1;

      let prior = new Array(6).fill(0);
      let totalW = 0;
      if (pGauss) { for (let c = 0; c < 6; c++) prior[c] += wGauss * pGauss[c]; totalW += wGauss; }
      if (pTopK) { for (let c = 0; c < 6; c++) prior[c] += wTopK * pTopK[c]; totalW += wTopK; }
      if (pUni) { for (let c = 0; c < 6; c++) prior[c] += wUni * pUni[c]; totalW += wUni; }

      if (totalW > 0) { for (let c = 0; c < 6; c++) prior[c] /= totalW; }
      else { prior = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; }

      const entropy = -prior.reduce((s, p) => s + (p > 1e-12 ? p * Math.log(p) : 0), 0);
      const maxEnt = Math.log(6);
      const ratio = entropy / maxEnt;
      const adaptFloor = floor * (0.5 + 4.5 * ratio * ratio);

      const floored = prior.map(v => Math.max(v, adaptFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
