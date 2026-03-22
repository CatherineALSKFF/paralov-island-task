const fs = require('fs');
const path = require('path');
const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

const DATA_DIR = path.join(__dirname, '..', 'data');
const ROUND_IDS = {
  1:'71451d74',2:'76909e29',3:'f1dac9a9',4:'8e839974',5:'fd3c92ff',
  6:'ae78003a',7:'36e581f1',8:'c5cdf100',9:'2a341ace',10:'75e625c3',
  11:'324fde07',12:'795bfb1f',13:'7b4bda99',14:'d0a2c894',15:'cc5442dd',
};

let cachedCellDB = null, cachedTestRound = null;

function cellFeatures(grid, settPos, y, x) {
  const v = grid[y][x];
  const terrain = v === 4 ? 2 : (v === 1 || v === 2) ? 1 : 0;
  let nS1 = 0, nS3 = 0, nF1 = 0, nM1 = 0, minSettDist = 99;
  for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
    if (!dy && !dx) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    const cd = Math.max(Math.abs(dy), Math.abs(dx));
    if (settPos.has(ny * W + nx)) {
      if (cd <= 1) nS1++;
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
  return { terrain, nS1, nS3, nF1, nM1, minSettDist, coastal: coastal ? 1 : 0 };
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
      const sp = new Set();
      for (let yy = 0; yy < H; yy++) for (let xx = 0; xx < W; xx++)
        if (grid[yy][xx] === 1 || grid[yy][xx] === 2) sp.add(yy * W + xx);
      for (let yy = 0; yy < H; yy++) for (let xx = 0; xx < W; xx++) {
        if (grid[yy][xx] === 10 || grid[yy][xx] === 5) continue;
        const feat = cellFeatures(grid, sp, yy, xx);
        db[feat.terrain].push({ feat, gt: gt[yy][xx], growth });
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

  const sigma = config.sigma || 0.035;
  const K = config.K || 3;
  const shrinkStrength = config.shrink || 2;
  const wGauss = config.wGauss || 0.55;
  const wTopK = config.wTopK || 0.20;
  const wUni = config.wUni || 0.10;
  const wCell = config.wCell || 0.15;
  const tempCoeff = config.tempCoeff || 1.35;
  const cellSigma = config.cellSigma || 0.04;
  const fBandwidth = config.fBandwidth || 6.0;

  // --- Gaussian weights ---
  const roundWeights = {};
  let wTotal = 0;
  for (const r of allRounds) {
    const g = growthRates[String(r)] || 0.15;
    const d = Math.abs(g - targetGrowth);
    const w = Math.exp(-0.5 * (d / sigma) ** 2);
    roundWeights[r] = w;
    wTotal += w;
  }
  for (const r of allRounds) roundWeights[r] /= wTotal;

  // Per-round normalized distributions
  const perRoundNorm = {};
  for (const r of allRounds) {
    const rb = perRoundBuckets[r];
    if (!rb) continue;
    perRoundNorm[r] = {};
    for (const key in rb) {
      const b = rb[key];
      const tot = b.sum.reduce((a, v) => a + v, 0);
      if (tot > 0) perRoundNorm[r][key] = b.sum.map(v => v / tot);
    }
  }

  function buildWeightedProbs(wts) {
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

  const gaussModel = buildWeightedProbs(roundWeights);

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

  // Full hierarchical shrinkage
  function lookupWithShrinkage(model, key) {
    const keys = [key];
    for (let i = key.length - 1; i >= 1; i--) keys.push(key.slice(0, i));
    let result = null, nEff = 0;
    for (const k of keys) {
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

  // Disagreement
  const disCache = {};
  function computeDisagreement(key) {
    if (key in disCache) return disCache[key];
    const preds = [], wts = [];
    for (const r of allRounds) {
      const rn = perRoundNorm[r];
      if (!rn) continue;
      let p = rn[key];
      if (!p) for (let i = key.length - 1; i >= 1; i--) {
        const c = key.slice(0, i);
        if (rn[c]) { p = rn[c]; break; }
      }
      if (!p) continue;
      preds.push(p);
      wts.push(roundWeights[r] || 0);
    }
    if (preds.length < 3) { disCache[key] = 0; return 0; }
    const wSum = wts.reduce((a, b) => a + b, 0);
    if (wSum < 1e-10) { disCache[key] = 0; return 0; }
    let dis = 0;
    for (let c = 0; c < 6; c++) {
      let wMean = 0;
      for (let i = 0; i < preds.length; i++) wMean += wts[i] * preds[i][c];
      wMean /= wSum;
      let wVar = 0;
      for (let i = 0; i < preds.length; i++) wVar += wts[i] * (preds[i][c] - wMean) ** 2;
      dis += Math.sqrt(wVar / wSum);
    }
    disCache[key] = dis;
    return dis;
  }

  // Cell-level NN
  const cellDB = buildCellDB(growthRates, testRound);
  function cellNNPredict(feat) {
    const cands = cellDB[feat.terrain];
    if (!cands || cands.length === 0) return null;
    const roundAvgs = {}, roundCounts = {};
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
      const gw = Math.exp(-0.5 * ((gf - targetGrowth) / cellSigma) ** 2);
      if (gw < 0.001) continue;
      for (let c = 0; c < 6; c++) avg[c] += gw * roundAvgs[g][c] / cnt;
      tw += gw;
    }
    if (tw < 0.01) return null;
    return Array.from(avg, v => v / tw);
  }

  // Predict
  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const v = initGrid[y][x];
      if (v === 10) { row.push([1,0,0,0,0,0]); continue; }
      if (v === 5) { row.push([0,0,0,0,0,1]); continue; }

      const key = getFeatureKey(initGrid, settPos, y, x);
      const pGauss = lookupWithShrinkage(gaussModel, key);
      const pTopK = lookupWithShrinkage(topKModel, key);
      const pUni = lookupWithShrinkage(uniModel, key);
      const feat = cellFeatures(initGrid, settPos, y, x);
      const pCell = cellNNPredict(feat);

      let prior = new Array(6).fill(0);
      let totalW = 0;
      if (pGauss) { for (let c = 0; c < 6; c++) prior[c] += wGauss * pGauss[c]; totalW += wGauss; }
      if (pTopK) { for (let c = 0; c < 6; c++) prior[c] += wTopK * pTopK[c]; totalW += wTopK; }
      if (pUni) { for (let c = 0; c < 6; c++) prior[c] += wUni * pUni[c]; totalW += wUni; }
      if (pCell) { for (let c = 0; c < 6; c++) prior[c] += wCell * pCell[c]; totalW += wCell; }

      if (totalW > 0) for (let c = 0; c < 6; c++) prior[c] /= totalW;
      else prior = [1/6,1/6,1/6,1/6,1/6,1/6];

      // Disagreement-based temperature
      const dis = computeDisagreement(key);
      if (dis > 0.08 && tempCoeff > 0) {
        const temp = 1.0 + tempCoeff * Math.min(dis, 1.2);
        let s = 0;
        for (let c = 0; c < 6; c++) { prior[c] = Math.pow(Math.max(prior[c], 1e-12), 1/temp); s += prior[c]; }
        for (let c = 0; c < 6; c++) prior[c] /= s;
      }

      const entropy = -prior.reduce((s, p) => s + (p > 1e-12 ? p * Math.log(p) : 0), 0);
      const ratio = entropy / Math.log(6);
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
