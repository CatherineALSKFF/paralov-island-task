const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).filter(rn => Number(rn) !== testRound);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const keys = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      row.push(getFeatureKey(initGrid, settPos, y, x));
    }
    keys.push(row);
  }

  const bandwidths = [0.04, 0.09, 0.22];
  const ensembleW = 1 / bandwidths.length;
  const regWeight = 0.35;
  const floor = 0.012;

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) row.push(new Float64Array(6));
    pred.push(row);
  }

  for (const bw of bandwidths) {
    const wts = {};
    let wSum = 0;
    for (const rn of allRounds) {
      const gr = growthRates[String(rn)] || 0.15;
      const d = gr - targetGrowth;
      const w = Math.exp(-d * d / (2 * bw * bw));
      wts[rn] = w;
      wSum += w;
    }
    for (const rn of allRounds) wts[rn] /= wSum;

    const fine = {}, coarse = {};
    for (const rn of allRounds) {
      const w = wts[rn];
      const buckets = perRoundBuckets[rn];
      if (!buckets) continue;
      for (const [key, bucket] of Object.entries(buckets)) {
        const n = bucket.count;
        if (!fine[key]) fine[key] = { s: new Float64Array(6), tw: 0 };
        for (let c = 0; c < 6; c++) fine[key].s[c] += (bucket.sum[c] / n) * w;
        fine[key].tw += w;

        const ck = key.slice(0, -1);
        if (!coarse[ck]) coarse[ck] = { s: new Float64Array(6), tw: 0 };
        for (let c = 0; c < 6; c++) coarse[ck].s[c] += (bucket.sum[c] / n) * w;
        coarse[ck].tw += w;
      }
    }

    const fN = {};
    for (const [k, d] of Object.entries(fine)) {
      fN[k] = new Float64Array(6);
      if (d.tw > 0) for (let c = 0; c < 6; c++) fN[k][c] = d.s[c] / d.tw;
    }
    const cN = {};
    for (const [k, d] of Object.entries(coarse)) {
      cN[k] = new Float64Array(6);
      if (d.tw > 0) for (let c = 0; c < 6; c++) cN[k][c] = d.s[c] / d.tw;
    }

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const key = keys[y][x];
        const ck = key.slice(0, -1);
        const f = fN[key], co = cN[ck];

        if (f && co) {
          for (let c = 0; c < 6; c++)
            pred[y][x][c] += ((1 - regWeight) * f[c] + regWeight * co[c]) * ensembleW;
        } else if (f) {
          for (let c = 0; c < 6; c++) pred[y][x][c] += f[c] * ensembleW;
        } else if (co) {
          for (let c = 0; c < 6; c++) pred[y][x][c] += co[c] * ensembleW;
        } else {
          for (let c = 0; c < 6; c++) pred[y][x][c] += (1 / 6) * ensembleW;
        }
      }
    }
  }

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let sum = 0;
      for (let c = 0; c < 6; c++) {
        if (pred[y][x][c] < floor) pred[y][x][c] = floor;
        sum += pred[y][x][c];
      }
      const out = new Array(6);
      for (let c = 0; c < 6; c++) out[c] = pred[y][x][c] / sum;
      pred[y][x] = out;
    }
  }

  return pred;
}

module.exports = { predict };