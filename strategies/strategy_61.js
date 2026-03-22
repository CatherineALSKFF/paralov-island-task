const { H, W, getFeatureKey, mergeBuckets } = require('./shared');
const fs = require('fs');
const path = require('path');

const ROUND_IDS = {
  1:'71451d74', 2:'76909e29', 3:'f1dac9a9', 4:'8e839974', 5:'fd3c92ff',
  6:'ae78003a', 7:'36e581f1', 8:'c5cdf100', 9:'2a341ace', 10:'75e625c3',
  11:'324fde07', 12:'795bfb1f', 13:'7b4bda99', 14:'d0a2c894', 15:'cc5442dd'
};

const DATA_DIR = path.join(__dirname, '..', 'data');

function getKeyDist(grid, settPos, y, x) {
  const v = grid[y][x];
  if (v === 10) return 'O';
  if (v === 5) return 'M';
  const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P';
  let nS = 0, minDist = 99;
  for (let dy = -3; dy <= 3; dy++)
    for (let dx = -3; dx <= 3; dx++) {
      if (!dy && !dx) continue;
      const ny = y + dy, nx = x + dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx)) {
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
  const nKey = nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3';
  let dKey = '';
  if (nS > 0) dKey = minDist <= 1 ? 'a' : minDist <= 2 ? 'b' : 'd';
  return t + nKey + (coast ? 'c' : '') + dKey;
}

function buildGTModel(trainRounds, growthRates, targetGrowth, sigma) {
  const weights = {};
  let wSum = 0;
  for (const r of trainRounds) {
    const g = growthRates[String(r)] || 0.15;
    const w = Math.exp(-((g - targetGrowth) ** 2) / (2 * sigma * sigma));
    weights[r] = w;
    wSum += w;
  }
  for (const r of trainRounds) weights[r] /= wSum;

  const model = {};
  const counts = {};
  for (const r of trainRounds) {
    const initsFile = path.join(DATA_DIR, 'inits_R' + r + '.json');
    if (!fs.existsSync(initsFile)) continue;
    const inits = JSON.parse(fs.readFileSync(initsFile, 'utf8'));
    const prefix = ROUND_IDS[r];
    if (!prefix) continue;

    for (let si = 0; si < 5; si++) {
      const gtFile = path.join(DATA_DIR, 'gt_' + prefix + '_s' + si + '.json');
      if (!fs.existsSync(gtFile)) continue;
      const gtRaw = JSON.parse(fs.readFileSync(gtFile, 'utf8'));
      const gt = gtRaw.ground_truth || gtRaw.gt;
      if (!gt) continue;
      const item = inits[si];
      const grid = Array.isArray(item[0]) ? item : item.grid;
      if (!grid) continue;

      const settPos = new Set();
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++)
          if (grid[y][x] === 1 || grid[y][x] === 2) settPos.add(y * W + x);

      const w = weights[r];
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++) {
          const key = getKeyDist(grid, settPos, y, x);
          if (!model[key]) { model[key] = [0,0,0,0,0,0]; counts[key] = 0; }
          const gtVec = gt[y][x];
          for (let c = 0; c < 6; c++) model[key][c] += w * gtVec[c];
          counts[key] += w;
        }
    }
  }

  const result = {};
  for (const key of Object.keys(model))
    result[key] = model[key].map(v => v / counts[key]);
  return result;
}

function weightedMergeBuckets(perRoundBuckets, growthRates, targetGrowth, sigma, excludeRound) {
  const roundNums = Object.keys(perRoundBuckets).map(Number).filter(n => n !== excludeRound);
  let totalWeight = 0;
  const weights = {};
  for (const rn of roundNums) {
    const dist = Math.abs((growthRates[String(rn)] || 0.15) - targetGrowth);
    const w = Math.exp(-dist * dist / (2 * sigma * sigma));
    weights[rn] = w;
    totalWeight += w;
  }
  const model = {};
  for (const rn of roundNums) {
    const w = weights[rn] / totalWeight;
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    for (const [key, val] of Object.entries(b)) {
      if (!model[key]) model[key] = { count: 0, sum: [0,0,0,0,0,0] };
      const avg = val.sum.map(v => v / val.count);
      model[key].count += w * val.count;
      for (let c = 0; c < 6; c++) model[key].sum[c] += w * avg[c] * val.count;
    }
  }
  const out = {};
  for (const [k, v] of Object.entries(model)) out[k] = v.sum.map(s => s / v.count);
  return out;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.00005;
  const gtSigma = config.gtSigma || 0.045;
  const bSigma = config.bSigma || 0.04;
  const blend = config.blend || 0.65;

  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const trainRounds = Object.keys(ROUND_IDS).map(Number)
    .filter(n => n !== testRound && growthRates[String(n)] !== undefined);

  const gtModel = buildGTModel(trainRounds, growthRates, targetGrowth, gtSigma);
  const bucketModel = weightedMergeBuckets(perRoundBuckets, growthRates, targetGrowth, bSigma, testRound);
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);
  const bucketFallback = mergeBuckets(perRoundBuckets, allRounds);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const keyDist = getKeyDist(initGrid, settPos, y, x);
      const keyBase = getFeatureKey(initGrid, settPos, y, x);

      let pGT = gtModel[keyDist] || null;
      if (!pGT) {
        for (let t = 1; t <= 3 && !pGT; t++) {
          const fb = keyDist.slice(0, -t);
          if (fb.length > 0 && gtModel[fb]) pGT = gtModel[fb];
        }
      }

      let pBucket = bucketModel[keyBase] || bucketFallback[keyBase] || null;
      if (!pBucket) {
        const fb = keyBase.slice(0, -1);
        pBucket = bucketModel[fb] || bucketFallback[fb] || null;
      }

      let p;
      if (pGT && pBucket) {
        p = pGT.map((v, c) => blend * v + (1 - blend) * pBucket[c]);
      } else {
        p = pGT || pBucket || [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      const floored = p.map(v => Math.max(v, floor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
