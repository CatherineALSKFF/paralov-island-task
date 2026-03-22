const fs = require('fs');
const path = require('path');
const { H, W, terrainToClass, getFeatureKey } = require('./shared');

const DATA_DIR = path.join(__dirname, '..', 'data');
const ROUND_IDS = {
  1: '71451d74', 2: '76909e29', 3: 'f1dac9a9', 4: '8e839974',
  5: 'fd3c92ff', 6: 'ae78003a', 7: '36e581f1', 8: 'c5cdf100',
  9: '2a341ace', 10: '75e625c3', 11: '324fde07', 12: '795bfb1f',
  13: '7b4bda99', 14: 'd0a2c894', 15: 'cc5442dd',
};

function toSimpleKey(k) {
  if (k === 'O' || k === 'M') return k;
  const m = k.match(/^([FPS]\d)(c?)/);
  if (m) return m[1] + m[2];
  return k;
}

// ── Cell-level GT database ──
let cachedDB = null, cachedTestRound = null;
function buildCellDB(growthRates, testRound) {
  if (cachedTestRound === testRound && cachedDB) return cachedDB;
  const db = {};
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
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++)
          if (grid[y][x] === 1 || grid[y][x] === 2) sp.add(y * W + x);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const key = getFeatureKey(grid, sp, y, x);
          if (key === 'O' || key === 'M') continue;
          let nS = 0, nF = 0;
          for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
            if (!dy && !dx) continue;
            const ny = y + dy, nx = x + dx;
            if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
            if (sp.has(ny * W + nx)) nS++;
            if (grid[ny][nx] === 4) nF++;
          }
          if (!db[key]) db[key] = [];
          db[key].push({ gt: gt[y][x], nS, nF, growth });
        }
      }
    }
  }
  cachedDB = db;
  cachedTestRound = testRound;
  return db;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.00001;
  const sigma = config.sigma || 0.037;
  const temp = config.temp || 1.15;
  const shrinkStr = config.shrink || 2;
  const cellBlend = config.cellBlend || 0.15;
  const fSigma = config.fSigma || 2.0;

  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRoundNums = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Regime-aware filtering: exclude death rounds from growth predictions and vice versa
  const isDeath = targetGrowth < 0.04;
  const filteredRounds = isDeath
    ? allRoundNums.filter(r => (growthRates[String(r)] || 0.15) < 0.08)
    : allRoundNums.filter(r => (growthRates[String(r)] || 0.15) >= 0.04);
  const useRounds = filteredRounds.length >= 3 ? filteredRounds : allRoundNums;

  // ── Growth weights ──
  const roundWeights = {};
  let wTotal = 0;
  for (const r of useRounds) {
    const g = growthRates[String(r)] || 0.15;
    roundWeights[r] = Math.exp(-0.5 * ((g - targetGrowth) / sigma) ** 2);
    wTotal += roundWeights[r];
  }
  for (const r of useRounds) roundWeights[r] /= wTotal;

  // ── Build bucket model from aggregated sub-keys ──
  function buildWeightedProbs(wts, rounds) {
    const buckets = {};
    for (const r of rounds) {
      const rb = perRoundBuckets[r];
      if (!rb) continue;
      const w = wts[r];
      if (!w || w < 1e-10) continue;
      for (const rawKey in rb) {
        const key = toSimpleKey(rawKey);
        const b = rb[rawKey];
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

  const gaussModel = buildWeightedProbs(roundWeights, useRounds);
  const uniWeights = {};
  for (const r of useRounds) uniWeights[r] = 1 / useRounds.length;
  const uniModel = buildWeightedProbs(uniWeights, useRounds);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // ── Cell-level DB ──
  const db = buildCellDB(growthRates, testRound);

  const gwCache = {};
  function growthWeight(g) {
    if (gwCache[g] !== undefined) return gwCache[g];
    const d = g - targetGrowth;
    return gwCache[g] = Math.exp(-0.5 * (d / 0.045) ** 2);
  }

  // ── Lookup with shrinkage ──
  function lookupShrink(model, key) {
    let result = null, nEff = 0;
    if (model[key]) { result = [...model[key].p]; nEff = model[key].n; }
    for (let trim = 1; trim < key.length; trim++) {
      const coarse = key.slice(0, -trim);
      if (!model[coarse]) continue;
      if (!result) { result = [...model[coarse].p]; nEff = model[coarse].n; }
      else {
        const alpha = nEff / (nEff + shrinkStr);
        for (let c = 0; c < 6; c++)
          result[c] = alpha * result[c] + (1 - alpha) * model[coarse].p[c];
      }
      break;
    }
    return result;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);

      if (key === 'O' || key === 'M') {
        row.push(key === 'O' ? [1, 0, 0, 0, 0, 0] : [0, 0, 0, 0, 0, 1]);
        continue;
      }

      // ── Bucket prediction ──
      const pGauss = lookupShrink(gaussModel, key);
      const pUni = lookupShrink(uniModel, key);
      let bucketPred = new Array(6).fill(0);
      let bW = 0;
      if (pGauss) { for (let c = 0; c < 6; c++) bucketPred[c] += 0.98 * pGauss[c]; bW += 0.98; }
      if (pUni) { for (let c = 0; c < 6; c++) bucketPred[c] += 0.02 * pUni[c]; bW += 0.02; }
      if (bW > 0) for (let c = 0; c < 6; c++) bucketPred[c] /= bW;
      else bucketPred.fill(1 / 6);

      // ── Cell-level prediction with spatial feature matching ──
      let cellPred = null;
      const matches = db[key];
      if (matches && matches.length > 0) {
        let nS = 0, nF = 0;
        for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
          if (!dy && !dx) continue;
          const ny = y + dy, nx = x + dx;
          if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
          if (settPos.has(ny * W + nx)) nS++;
          if (initGrid[ny][nx] === 4) nF++;
        }
        const fs2 = 2 * fSigma * fSigma;
        const avg = new Array(6).fill(0);
        let tw = 0;
        for (const m of matches) {
          const gw = growthWeight(m.growth);
          if (gw < 0.001) continue;
          const nsDiff = m.nS - nS, nfDiff = m.nF - nF;
          const fw = Math.exp(-(nsDiff * nsDiff + nfDiff * nfDiff) / fs2);
          const w = gw * fw;
          if (w < 1e-8) continue;
          for (let c = 0; c < 6; c++) avg[c] += w * m.gt[c];
          tw += w;
        }
        if (tw > 0) cellPred = avg.map(v => v / tw);
      }

      // ── Blend bucket + cell ──
      let prior;
      if (cellPred) {
        prior = new Array(6);
        for (let c = 0; c < 6; c++)
          prior[c] = (1 - cellBlend) * bucketPred[c] + cellBlend * cellPred[c];
      } else {
        prior = [...bucketPred];
      }

      // ── Temperature ──
      let tS = 0;
      for (let c = 0; c < 6; c++) {
        prior[c] = Math.pow(Math.max(prior[c], 1e-15), 1 / temp);
        tS += prior[c];
      }
      for (let c = 0; c < 6; c++) prior[c] /= tS;

      // ── Adaptive floor ──
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
