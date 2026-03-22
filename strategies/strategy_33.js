const fs = require('fs');
const path = require('path');
const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

const DATA_DIR = path.join(__dirname, '..', 'data');
const ROUND_IDS = {
  1: '71451d74', 2: '76909e29', 3: 'f1dac9a9', 4: '8e839974',
  5: 'fd3c92ff', 6: 'ae78003a', 7: '36e581f1', 8: 'c5cdf100',
  9: '2a341ace', 10: '75e625c3', 11: '324fde07', 12: '795bfb1f',
  13: '7b4bda99', 14: 'd0a2c894', 15: 'cc5442dd',
};

// Cache for cell-level DB across calls
let cachedCellDB = null, cachedTestRound = null;

// Richer cell features for nearest-neighbor matching
function cellFeatures(grid, settPos, y, x) {
  const v = grid[y][x];
  const terrain = v === 4 ? 2 : (v === 1 || v === 2) ? 1 : 0;

  let nS1 = 0, nS2 = 0, nS3 = 0;
  let nF1 = 0, nM1 = 0;
  let minSettDist = 99;

  for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
    if (!dy && !dx) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    const cd = Math.max(Math.abs(dy), Math.abs(dx));
    const isSett = settPos.has(ny * W + nx);

    if (isSett) {
      if (cd <= 1) nS1++;
      if (cd <= 2) nS2++;
      nS3++;
      if (cd < minSettDist) minSettDist = cd;
    }
    if (cd <= 1) {
      if (grid[ny][nx] === 4) nF1++;
      if (grid[ny][nx] === 5) nM1++;
    }
  }

  let coastal = false;
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) coastal = true;
  }

  return { terrain, nS1, nS2, nS3, nF1, nM1, minSettDist, coastal: coastal ? 1 : 0 };
}

function featureDist(a, b) {
  if (a.terrain !== b.terrain) return 1000;
  return (
    3.0 * (a.nS3 - b.nS3) ** 2 +
    2.0 * (a.nS1 - b.nS1) ** 2 +
    1.5 * (a.nF1 - b.nF1) ** 2 +
    1.5 * (a.nM1 - b.nM1) ** 2 +
    4.0 * (a.coastal - b.coastal) ** 2 +
    2.0 * Math.min((a.minSettDist - b.minSettDist) ** 2, 16)
  );
}

function buildCellDB(growthRates, testRound) {
  if (cachedTestRound === testRound && cachedCellDB) return cachedCellDB;

  const db = { 0: [], 1: [], 2: [] };
  const allRounds = Object.keys(ROUND_IDS).map(Number).filter(n => n !== testRound);

  for (const r of allRounds) {
    const prefix = ROUND_IDS[r];
    const growth = growthRates[String(r)] || 0.15;
    const initFile = path.join(DATA_DIR, `inits_R${r}.json`);
    if (!fs.existsSync(initFile)) continue;
    const initsRaw = JSON.parse(fs.readFileSync(initFile, 'utf8'));

    for (let seed = 0; seed < 5; seed++) {
      const gtFile = path.join(DATA_DIR, `gt_${prefix}_s${seed}.json`);
      if (!fs.existsSync(gtFile)) continue;
      const gtData = JSON.parse(fs.readFileSync(gtFile, 'utf8'));
      const gt = gtData.ground_truth || gtData.gt;
      if (!gt) continue;

      const initItem = initsRaw[seed];
      if (!initItem) continue;
      const grid = Array.isArray(initItem) && Array.isArray(initItem[0]) ? initItem : (initItem.grid || null);
      if (!grid) continue;

      const settPos = new Set();
      for (let yy = 0; yy < H; yy++)
        for (let xx = 0; xx < W; xx++)
          if (grid[yy][xx] === 1 || grid[yy][xx] === 2) settPos.add(yy * W + xx);

      for (let yy = 0; yy < H; yy++) {
        for (let xx = 0; xx < W; xx++) {
          if (grid[yy][xx] === 10 || grid[yy][xx] === 5) continue;
          const feat = cellFeatures(grid, settPos, yy, xx);
          db[feat.terrain].push({ feat, gt: gt[yy][xx], growth });
        }
      }
    }
  }

  cachedCellDB = db;
  cachedTestRound = testRound;
  return db;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // ===== MODEL 1: Bucket-based with Gaussian growth weighting =====
  const sigma = config.sigma || 0.06;

  // Heavy-tailed Gaussian: mix 85% Gaussian + 15% uniform
  const gaussWeights = {};
  let gTotal = 0;
  for (const r of allRounds) {
    const g = growthRates[String(r)] || 0.15;
    const d = Math.abs(g - targetGrowth);
    const w = 0.85 * Math.exp(-0.5 * (d / sigma) ** 2) + 0.15 / allRounds.length;
    gaussWeights[r] = w;
    gTotal += w;
  }
  for (const r of allRounds) gaussWeights[r] /= gTotal;

  // Multi-K ensemble weights
  const multiKWeights = {};
  for (const r of allRounds) multiKWeights[r] = 0;
  const candidates = { ...growthRates };
  delete candidates[String(testRound)];
  for (const K of [2, 3, 5, 8]) {
    const closest = selectClosestRounds(candidates, targetGrowth, K);
    for (const cr of closest) multiKWeights[cr] += 1 / (K * 4);
  }
  let mTotal = 0;
  for (const r of allRounds) mTotal += multiKWeights[r];
  if (mTotal > 0) for (const r of allRounds) multiKWeights[r] /= mTotal;

  // Uniform weights
  const uniWeights = {};
  for (const r of allRounds) uniWeights[r] = 1 / allRounds.length;

  function buildModel(wts) {
    const buckets = {};
    for (const r of allRounds) {
      const rb = perRoundBuckets[r];
      if (!rb) continue;
      const w = wts[r];
      if (w < 1e-12) continue;
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

  const gaussModel = buildModel(gaussWeights);
  const multiKModel = buildModel(multiKWeights);
  const uniModel = buildModel(uniWeights);

  // Full hierarchical shrinkage through ALL levels
  function hierarchicalLookup(model, key) {
    const keys = [key];
    for (let i = key.length - 1; i >= 1; i--) keys.push(key.slice(0, i));

    let result = null;
    let nEff = 0;
    const shrinkStrength = config.shrink || 4;

    for (const k of keys) {
      if (!model[k]) continue;
      if (!result) {
        result = [...model[k].p];
        nEff = model[k].n;
      } else {
        const alpha = nEff / (nEff + shrinkStrength);
        const cp = model[k].p;
        for (let c = 0; c < 6; c++) result[c] = alpha * result[c] + (1 - alpha) * cp[c];
        nEff += model[k].n * 0.3;
      }
    }
    return result;
  }

  // ===== MODEL 2: Cell-level nearest-neighbor with richer features =====
  const cellDB = buildCellDB(growthRates, testRound);
  const cellSigma = config.cellSigma || 0.05;
  const fBandwidth = config.fBandwidth || 6.0;

  function cellNNPredict(feat) {
    const cands = cellDB[feat.terrain];
    if (!cands || cands.length === 0) return null;

    // Group by growth rate (round), compute per-round weighted averages
    const roundAvgs = {};
    const roundCounts = {};
    for (let i = 0; i < cands.length; i++) {
      const m = cands[i];
      const fd = featureDist(feat, m.feat);
      const fw = Math.exp(-fd / fBandwidth);
      if (fw < 0.01) continue;

      const g = m.growth;
      if (!roundAvgs[g]) { roundAvgs[g] = new Float64Array(6); roundCounts[g] = 0; }
      for (let c = 0; c < 6; c++) roundAvgs[g][c] += fw * m.gt[c];
      roundCounts[g] += fw;
    }

    const avg = new Float64Array(6);
    let tw = 0;
    for (const g of Object.keys(roundAvgs)) {
      const cnt = roundCounts[g];
      if (cnt < 0.01) continue;
      const gf = parseFloat(g);
      const gDiff = gf - targetGrowth;
      const gw = Math.exp(-0.5 * (gDiff / cellSigma) ** 2);
      if (gw < 0.001) continue;
      for (let c = 0; c < 6; c++) avg[c] += gw * roundAvgs[g][c] / cnt;
      tw += gw;
    }
    if (tw < 0.01) return null;
    const result = new Array(6);
    for (let c = 0; c < 6; c++) result[c] = avg[c] / tw;
    return result;
  }

  // ===== Adaptive ensemble weights =====
  const wGauss = config.wGauss || 0.40;
  const wMultiK = config.wTopK || 0.25;
  const wUni = config.wUni || 0.10;
  const wCell = config.wCell || 0.25;

  // ===== Predict =====
  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const v = initGrid[y][x];
      if (v === 10) { row.push([1, 0, 0, 0, 0, 0]); continue; }
      if (v === 5) { row.push([0, 0, 0, 0, 0, 1]); continue; }

      const key = getFeatureKey(initGrid, settPos, y, x);

      // Bucket models with hierarchical shrinkage
      const pGauss = hierarchicalLookup(gaussModel, key);
      const pMultiK = hierarchicalLookup(multiKModel, key);
      const pUni = hierarchicalLookup(uniModel, key);

      // Cell-level NN model
      const feat = cellFeatures(initGrid, settPos, y, x);
      const pCell = cellNNPredict(feat);

      // Weighted linear combination
      let prior = new Array(6).fill(0);
      let totalW = 0;
      if (pGauss) { for (let c = 0; c < 6; c++) prior[c] += wGauss * pGauss[c]; totalW += wGauss; }
      if (pMultiK) { for (let c = 0; c < 6; c++) prior[c] += wMultiK * pMultiK[c]; totalW += wMultiK; }
      if (pUni) { for (let c = 0; c < 6; c++) prior[c] += wUni * pUni[c]; totalW += wUni; }
      if (pCell) { for (let c = 0; c < 6; c++) prior[c] += wCell * pCell[c]; totalW += wCell; }

      if (totalW > 0) {
        for (let c = 0; c < 6; c++) prior[c] /= totalW;
      } else {
        prior = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      // Adaptive floor
      const entropy = -prior.reduce((s, p) => s + (p > 1e-12 ? p * Math.log(p) : 0), 0);
      const maxEnt = Math.log(6);
      const ratio = entropy / maxEnt;
      const adaptFloor = floor * (0.3 + 5 * ratio * ratio);

      const floored = prior.map(v => Math.max(v, adaptFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
