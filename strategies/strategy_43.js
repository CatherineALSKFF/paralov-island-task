const { H, W, getFeatureKey } = require('./shared');
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'data');
const ROUND_IDS = {
  1:'71451d74', 2:'76909e29', 3:'f1dac9a9', 4:'8e839974',
  5:'fd3c92ff', 6:'ae78003a', 7:'36e581f1', 8:'c5cdf100',
  9:'2a341ace', 10:'75e625c3', 11:'324fde07', 12:'795bfb1f',
  13:'7b4bda99', 14:'d0a2c894', 15:'cc5442dd',
};

let gtCache = null;
let gtCacheExclude = -1;

function loadAllGT(excludeRound) {
  if (gtCache && gtCacheExclude === excludeRound) return gtCache;
  const data = {};
  for (const [rn, prefix] of Object.entries(ROUND_IDS)) {
    if (parseInt(rn) === excludeRound) continue;
    const initFile = path.join(DATA_DIR, `inits_R${rn}.json`);
    if (!fs.existsSync(initFile)) continue;
    const inits = JSON.parse(fs.readFileSync(initFile, 'utf8'));
    data[rn] = [];
    for (let s = 0; s < 5; s++) {
      const gtFile = path.join(DATA_DIR, `gt_${prefix}_s${s}.json`);
      if (!fs.existsSync(gtFile)) continue;
      const gtRaw = JSON.parse(fs.readFileSync(gtFile, 'utf8'));
      const gt = gtRaw.ground_truth || gtRaw.gt;
      if (!gt) continue;
      const item = inits[s];
      if (!item) continue;
      const grid = Array.isArray(item[0]) ? item : item.grid;
      if (!grid) continue;
      const settlements = [];
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++)
          if (grid[y][x] === 1 || grid[y][x] === 2)
            settlements.push({ y, x });
      data[rn].push({ grid, settlements, gt });
    }
  }
  gtCache = data;
  gtCacheExclude = excludeRound;
  return data;
}

function cellFeatures(grid, settPos, settList, y, x) {
  const v = grid[y][x];
  const terrain = v === 10 ? 0 : v === 5 ? 5 : v === 4 ? 4 : (v === 1 || v === 2) ? 1 : 0;
  let nS1 = 0, nS3 = 0, nS5 = 0, nF1 = 0, nF3 = 0;
  for (let dy = -5; dy <= 5; dy++) {
    for (let dx = -5; dx <= 5; dx++) {
      if (!dy && !dx) continue;
      const ny = y + dy, nx = x + dx;
      if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
      const r = Math.max(Math.abs(dy), Math.abs(dx));
      if (settPos.has(ny * W + nx)) {
        if (r <= 1) nS1++;
        if (r <= 3) nS3++;
        nS5++;
      }
      if (grid[ny][nx] === 4) {
        if (r <= 1) nF1++;
        if (r <= 3) nF3++;
      }
    }
  }
  let coast = 0;
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) coast = 1;
  }
  let minDist = 99;
  for (const s of settList) {
    const d = Math.abs(y - s.y) + Math.abs(x - s.x);
    if (d < minDist) minDist = d;
  }
  return [terrain, nS1, nS3, nS5, nF1, nF3, coast, Math.min(minDist, 20)];
}

function featureDist(f1, f2) {
  if (f1[0] !== f2[0]) return 1e10;
  let d = 0;
  d += ((f1[1] - f2[1]) / 2) ** 2;
  d += ((f1[2] - f2[2]) / 5) ** 2;
  d += ((f1[3] - f2[3]) / 10) ** 2;
  d += ((f1[4] - f2[4]) / 3) ** 2;
  d += ((f1[5] - f2[5]) / 8) ** 2;
  d += (f1[6] - f2[6]) ** 2 * 2;
  d += ((f1[7] - f2[7]) / 5) ** 2;
  return d;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const sigma = config.sigma || 0.03;
  const knnBlend = config.knnBlend || 0.7;
  const knnBandwidth = config.knnBandwidth || 0.35;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const allGT = loadAllGT(testRound);

  const roundNums = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);
  const rWeights = {};
  let rwTotal = 0;
  for (const rn of roundNums) {
    const g = growthRates[String(rn)];
    if (g === undefined) continue;
    const d = Math.abs(g - targetGrowth);
    const w = Math.exp(-d * d / (2 * sigma * sigma));
    rWeights[rn] = w;
    rwTotal += w;
  }
  for (const rn of roundNums) rWeights[rn] = (rWeights[rn] || 0) / (rwTotal || 1);

  // Bucket-level base model (Gaussian-weighted)
  const bucketModel = {};
  for (const rn of roundNums) {
    const w = rWeights[rn];
    if (!w) continue;
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    for (const [key, val] of Object.entries(b)) {
      if (!bucketModel[key]) bucketModel[key] = { count: 0, sum: [0,0,0,0,0,0] };
      const avg = val.sum.map(v => v / val.count);
      bucketModel[key].count += w * val.count;
      for (let c = 0; c < 6; c++) bucketModel[key].sum[c] += w * avg[c] * val.count;
    }
  }
  const baseModel = {};
  for (const [k, v] of Object.entries(bucketModel)) {
    baseModel[k] = v.sum.map(s => s / v.count);
  }

  // Precompute training cell features for dynamic cells
  const trainCells = {};
  for (const [rn, seeds] of Object.entries(allGT)) {
    const rw = rWeights[parseInt(rn)] || 0;
    if (rw < 0.001) continue;
    for (const { grid, settlements: setts, gt } of seeds) {
      const sp = new Set();
      for (const s of setts) sp.add(s.y * W + s.x);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const v = grid[y][x];
          if (v === 10 || v === 5) continue;
          const gtVec = gt[y][x];
          let ent = 0;
          for (let c = 0; c < 6; c++) {
            if (gtVec[c] > 0.001) ent -= gtVec[c] * Math.log(gtVec[c]);
          }
          if (ent < 0.01) continue;
          const feat = cellFeatures(grid, sp, setts, y, x);
          const terrain = feat[0];
          if (!trainCells[terrain]) trainCells[terrain] = [];
          trainCells[terrain].push({ feat, gt: gtVec, rw });
        }
      }
    }
  }

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      let basePred = baseModel[key] ? [...baseModel[key]] : null;
      if (!basePred) {
        const fb = key.slice(0, -1);
        basePred = baseModel[fb] ? [...baseModel[fb]] : [1/6,1/6,1/6,1/6,1/6,1/6];
      }

      const v = initGrid[y][x];
      let knnPred = null;
      if (v !== 10 && v !== 5 && knnBlend > 0) {
        const feat = cellFeatures(initGrid, settPos, settlements, y, x);
        const terrain = feat[0];
        const candidates = trainCells[terrain];
        if (candidates && candidates.length > 0) {
          knnPred = [0,0,0,0,0,0];
          let wSum = 0;
          for (const tc of candidates) {
            const d = featureDist(feat, tc.feat);
            if (d > 20) continue;
            const w = tc.rw * Math.exp(-d / (2 * knnBandwidth * knnBandwidth));
            for (let c = 0; c < 6; c++) knnPred[c] += w * tc.gt[c];
            wSum += w;
          }
          if (wSum > 0) {
            for (let c = 0; c < 6; c++) knnPred[c] /= wSum;
          } else {
            knnPred = null;
          }
        }
      }

      let prior;
      if (knnPred) {
        prior = new Array(6);
        for (let c = 0; c < 6; c++) {
          prior[c] = (1 - knnBlend) * basePred[c] + knnBlend * knnPred[c];
        }
      } else {
        prior = basePred;
      }

      const floored = prior.map(v => Math.max(v, floor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
