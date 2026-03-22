const { H, W, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);
  const candidates = { ...growthRates }; delete candidates[String(testRound)];

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Compute nearest settlement distance for each cell
  const nearestDist = new Uint8Array(H * W).fill(99);
  for (const s of settlements) {
    for (let dy = -12; dy <= 12; dy++) for (let dx = -12; dx <= 12; dx++) {
      const ny = s.y + dy, nx = s.x + dx;
      if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
      const d = Math.max(Math.abs(dy), Math.abs(dx));
      const idx = ny * W + nx;
      if (d < nearestDist[idx]) nearestDist[idx] = d;
    }
  }

  // Enhanced feature key matching bucket data format
  function getEnhancedKey(y, x) {
    const base = getFeatureKey(initGrid, settPos, y, x);
    if (base === 'O' || base === 'M' || base[0] === 'S') return base;
    const density = base[1];
    const md = nearestDist[y * W + x];
    if (density === '0') {
      return base + (md <= 5 ? 'n' : md <= 7 ? 'm' : 'f');
    }
    if (density === '1') {
      if (md === 1) return base + 'a';
      if (md === 2) return base + 'b';
    }
    return base;
  }

  // Build merged bucket model with count info
  function buildModel(rounds) {
    const m = {};
    for (const rn of rounds) {
      const b = perRoundBuckets[String(rn)]; if (!b) continue;
      for (const [k, v] of Object.entries(b)) {
        if (!m[k]) m[k] = { count: 0, sum: [0,0,0,0,0,0] };
        m[k].count += v.count;
        for (let c = 0; c < 6; c++) m[k].sum[c] += v.sum[c];
      }
    }
    const out = {};
    for (const [k, v] of Object.entries(m)) {
      out[k] = v.sum.map(s => s / v.count);
    }
    return out;
  }

  // K=4 adaptive model + all-rounds fallback (same structure as baseline)
  const closestRounds = selectClosestRounds(candidates, targetGrowth, 4);
  const adaptiveModel = buildModel(closestRounds);
  const allModel = buildModel(allRounds);

  // Lookup with fallback chain: enhanced → base → truncated
  function lookup(model, enhKey, baseKey) {
    if (model[enhKey]) return model[enhKey];
    if (enhKey !== baseKey && model[baseKey]) return model[baseKey];
    // Truncate
    for (let len = baseKey.length - 1; len >= 1; len--) {
      const fb = baseKey.slice(0, len);
      if (model[fb]) return model[fb];
    }
    return null;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const baseKey = getFeatureKey(initGrid, settPos, y, x);
      const enhKey = getEnhancedKey(y, x);

      let prior = lookup(adaptiveModel, enhKey, baseKey)
               || lookup(allModel, enhKey, baseKey)
               || [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];

      const floored = prior.map(v => Math.max(v, floor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
