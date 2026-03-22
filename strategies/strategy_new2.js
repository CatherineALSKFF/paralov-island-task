const { H, W, getFeatureKey } = require('./shared');

function getRichKey(grid, settPos, setts, y, x) {
  const baseKey = getFeatureKey(grid, settPos, y, x);
  if (baseKey === 'O' || baseKey === 'M') return baseKey;
  const t = baseKey[0], nS = baseKey[1];

  if (nS === '0' && (t === 'P' || t === 'F')) {
    let minCheb = 999;
    for (const s of setts) {
      const d = Math.max(Math.abs(y - s.y), Math.abs(x - s.x));
      if (d < minCheb) minCheb = d;
    }
    return baseKey + (minCheb === 4 ? 'n' : minCheb <= 8 ? 'm' : 'f');
  }

  if (nS === '1' && (t === 'P' || t === 'F')) {
    let sR1 = 0, sR2 = 0;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      if (!dy && !dx) continue;
      const ny = y + dy, nx = x + dx;
      if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
      if (!settPos.has(ny * W + nx)) continue;
      const d = Math.max(Math.abs(dy), Math.abs(dx));
      if (d <= 1) sR1++; else sR2++;
    }
    if (sR1 > 0) return baseKey + 'a';
    if (sR2 > 0) return baseKey + 'b';
  }

  return baseKey;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const sigma = config.sigma || 0.045;
  const shrinkN = config.shrinkN || 20;
  const temp = config.temp || 1.09;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Gaussian weights
  const rw = {};
  let wT = 0;
  for (const r of allRounds) {
    const g = growthRates[String(r)] || 0.15;
    const diff = g - targetGrowth;
    rw[r] = Math.exp(-diff * diff / (2 * sigma * sigma));
    wT += rw[r];
  }
  for (const r of allRounds) rw[r] /= wT;

  // Gaussian-weighted model
  const gaussModel = {};
  for (const r of allRounds) {
    const w = rw[r];
    if (w < 1e-8) continue;
    const b = perRoundBuckets[String(r)];
    if (!b) continue;
    for (const [k, v] of Object.entries(b)) {
      if (!gaussModel[k]) gaussModel[k] = { count: 0, sum: [0, 0, 0, 0, 0, 0] };
      gaussModel[k].count += w * v.count;
      for (let c = 0; c < 6; c++) gaussModel[k].sum[c] += w * v.sum[c];
    }
  }

  // All-rounds fallback
  const allModel = {};
  for (const r of allRounds) {
    const b = perRoundBuckets[String(r)];
    if (!b) continue;
    for (const [k, v] of Object.entries(b)) {
      if (!allModel[k]) allModel[k] = { count: 0, sum: [0, 0, 0, 0, 0, 0] };
      allModel[k].count += v.count;
      for (let c = 0; c < 6; c++) allModel[k].sum[c] += v.sum[c];
    }
  }

  function getP(model, key) {
    const b = model[key];
    if (!b || b.count === 0) return null;
    return { probs: b.sum.map(s => s / b.count), count: b.count };
  }

  const settPos = new Set();
  const setts = [];
  for (const s of settlements) { settPos.add(s.y * W + s.x); setts.push(s); }

  const invTemp = 1 / temp;
  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const richKey = getRichKey(initGrid, settPos, setts, y, x);
      const baseKey = getFeatureKey(initGrid, settPos, y, x);

      let prior = null;

      for (const model of [gaussModel, allModel]) {
        if (prior) break;
        const rich = getP(model, richKey);
        const base = getP(model, baseKey);

        if (rich && base && richKey !== baseKey) {
          const alpha = rich.count / (rich.count + shrinkN);
          prior = new Array(6);
          for (let c = 0; c < 6; c++)
            prior[c] = alpha * rich.probs[c] + (1 - alpha) * base.probs[c];
        } else if (rich) {
          prior = [...rich.probs];
        } else if (base) {
          prior = [...base.probs];
        }
      }

      if (!prior) {
        const fb = baseKey.slice(0, -1);
        const fbG = getP(gaussModel, fb);
        const fbA = getP(allModel, fb);
        prior = fbG ? [...fbG.probs] : fbA ? [...fbA.probs] : [1/6,1/6,1/6,1/6,1/6,1/6];
      }

      // Temperature scaling (>1 softens, <1 sharpens)
      let sum = 0;
      for (let c = 0; c < 6; c++) {
        prior[c] = Math.pow(Math.max(prior[c], 1e-15), invTemp);
        sum += prior[c];
      }
      for (let c = 0; c < 6; c++) prior[c] /= sum;

      // Floor + normalize
      const floored = prior.map(v => Math.max(v, floor));
      const fsum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / fsum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
