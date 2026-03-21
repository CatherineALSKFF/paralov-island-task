const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const regWeight = config.REG || 0.40;
  const bandwidth = config.BW || 0.07;
  const uniformBlend = config.UB || 0.15;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const allRoundNums = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Gaussian kernel weights by growth-rate similarity
  const roundWeights = {};
  let wTotal = 0;
  for (const rn of allRoundNums) {
    const rate = growthRates[String(rn)] || 0.15;
    const d = Math.abs(rate - targetGrowth);
    const w = Math.exp(-0.5 * (d / bandwidth) ** 2);
    roundWeights[rn] = w;
    wTotal += w;
  }

  // Weighted merge: each round's buckets scaled by Gaussian weight
  function weightedMerge(roundNums, weights) {
    const m = {};
    for (const rn of roundNums) {
      const b = perRoundBuckets[String(rn)];
      if (!b) continue;
      const w = weights[rn] || 0;
      for (const [k, v] of Object.entries(b)) {
        if (!m[k]) m[k] = { wc: 0, ws: [0,0,0,0,0,0] };
        m[k].wc += v.count * w;
        for (let c = 0; c < 6; c++) m[k].ws[c] += v.sum[c] * w;
      }
    }
    const out = {};
    for (const [k, v] of Object.entries(m)) {
      if (v.wc > 0) out[k] = v.ws.map(s => s / v.wc);
    }
    return out;
  }

  const wModel = weightedMerge(allRoundNums, roundWeights);
  const uModel = mergeBuckets(perRoundBuckets, allRoundNums);

  // Blend weighted + uniform for robustness against outlier rounds
  const blended = {};
  const allKeys = new Set([...Object.keys(wModel), ...Object.keys(uModel)]);
  for (const k of allKeys) {
    const wv = wModel[k], uv = uModel[k];
    if (wv && uv) {
      blended[k] = wv.map((v, i) => (1 - uniformBlend) * v + uniformBlend * uv[i]);
    } else {
      blended[k] = wv || uv;
    }
  }

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      // Static cells — hard-code with certainty
      if (key === 'O') { row.push([1,0,0,0,0,0]); continue; }
      if (key === 'M') { row.push([0,0,0,0,0,1]); continue; }

      // Multi-level lookup: full key → strip trailing modifier → terrain-only
      const coarseKey = key.length > 1 ? key.slice(0, -1) : key;
      const terrainKey = key[0];

      const fine = blended[key] || null;
      const mid = blended[coarseKey] || null;
      const broad = blended[terrainKey] || [1/6,1/6,1/6,1/6,1/6,1/6];

      let result;
      if (fine) {
        // Regularise fine toward coarser level
        const coarse = mid || broad;
        result = fine.map((v, i) => (1 - regWeight) * v + regWeight * coarse[i]);
      } else if (mid) {
        result = mid.map((v, i) => (1 - regWeight * 0.5) * v + regWeight * 0.5 * broad[i]);
      } else {
        result = [...broad];
      }

      // Per-cell adaptive floor: confident cells get smaller floor
      const ent = -result.reduce((s, p) => s + (p > 1e-12 ? p * Math.log(p) : 0), 0);
      const ratio = ent / Math.log(6);
      const aFloor = floor * (0.05 + 0.95 * ratio);

      const floored = result.map(v => Math.max(v, aFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
