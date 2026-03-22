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

let cachedDB = null, cachedTestRound = null;

function buildCellDB(growthRates, testRound) {
  if (cachedTestRound === testRound && cachedDB) return cachedDB;

  const db = {}; // featureKey -> [{gt:[6], growth:float}]
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
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++)
          if (grid[y][x] === 1 || grid[y][x] === 2) settPos.add(y * W + x);

      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const key = getFeatureKey(grid, settPos, y, x);
          if (key === 'O' || key === 'M') continue;
          if (!db[key]) db[key] = [];
          db[key].push({ gt: gt[y][x], growth });
        }
      }
    }
  }

  cachedDB = db;
  cachedTestRound = testRound;
  return db;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const sigma = config.SIGMA || 0.045;
  const floor = config.FLOOR || 1e-8;
  const tempCoeff = config.TEMP || 0.80;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const db = buildCellDB(growthRates, testRound);

  // Pre-compute growth weights
  const growthWeightCache = {};
  function growthWeight(g) {
    if (growthWeightCache[g] !== undefined) return growthWeightCache[g];
    const d = g - targetGrowth;
    growthWeightCache[g] = Math.exp(-0.5 * (d / sigma) * (d / sigma));
    return growthWeightCache[g];
  }

  // Per-key: compute weighted average + disagreement for adaptive smoothing
  const keyCache = {};
  function getKeyPred(key) {
    if (keyCache[key] !== undefined) return keyCache[key];
    const matches = db[key];
    if (!matches || matches.length === 0) { keyCache[key] = null; return null; }

    // Group by round (growth rate) to compute per-round averages
    const roundAvgs = {};
    const roundCounts = {};
    for (const m of matches) {
      const g = m.growth;
      if (!roundAvgs[g]) { roundAvgs[g] = [0,0,0,0,0,0]; roundCounts[g] = 0; }
      for (let c = 0; c < 6; c++) roundAvgs[g][c] += m.gt[c];
      roundCounts[g]++;
    }

    // Weighted average of per-round averages
    const avg = [0,0,0,0,0,0];
    let tw = 0;
    const roundPreds = [];
    const roundWeights = [];
    for (const g of Object.keys(roundAvgs)) {
      const cnt = roundCounts[g];
      const gf = parseFloat(g);
      const w = growthWeight(gf);
      const rAvg = roundAvgs[g].map(v => v / cnt);
      for (let c = 0; c < 6; c++) avg[c] += w * rAvg[c];
      tw += w;
      roundPreds.push(rAvg);
      roundWeights.push(w);
    }
    if (tw > 0) for (let c = 0; c < 6; c++) avg[c] /= tw;

    // Weighted disagreement across rounds
    let dis = 0;
    if (roundPreds.length >= 2) {
      for (let c = 0; c < 6; c++) {
        let wVar = 0;
        for (let i = 0; i < roundPreds.length; i++) {
          const diff = roundPreds[i][c] - avg[c];
          wVar += roundWeights[i] * diff * diff;
        }
        dis += Math.sqrt(wVar / tw);
      }
    }

    keyCache[key] = { avg, dis, nRounds: roundPreds.length };
    return keyCache[key];
  }

  const predGrid = [];
  const isMO = [];

  for (let y = 0; y < H; y++) {
    predGrid.push(new Array(W));
    isMO.push(new Array(W));
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      if (key === 'O' || key === 'M') {
        isMO[y][x] = true;
        predGrid[y][x] = null;
        continue;
      }
      isMO[y][x] = false;

      const fineData = getKeyPred(key);
      const fb = key.length > 1 ? key.slice(0, -1) : null;
      const coarseData = fb ? getKeyPred(fb) : null;

      let probs, dis;
      if (fineData) {
        probs = [...fineData.avg];
        dis = fineData.dis;
        // Mild regularization toward coarser key
        if (coarseData) {
          const regW = 0.10;
          for (let c = 0; c < 6; c++) probs[c] = (1 - regW) * probs[c] + regW * coarseData.avg[c];
        }
      } else if (coarseData) {
        probs = [...coarseData.avg];
        dis = coarseData.dis;
      } else {
        predGrid[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6];
        continue;
      }

      // Mild temperature softening based on disagreement
      // Only apply when disagreement is significant
      if (dis > 0.1 && tempCoeff > 0) {
        const temp = 1.0 + tempCoeff * Math.min(dis, 1.0);
        let s = 0;
        for (let c = 0; c < 6; c++) {
          probs[c] = Math.pow(Math.max(probs[c], 1e-12), 1 / temp);
          s += probs[c];
        }
        for (let c = 0; c < 6; c++) probs[c] /= s;
      }

      // Adaptive floor: higher when disagreement is high
      const aFloor = floor + 0.003 * Math.min(dis, 1.0);
      const floored = probs.map(v => Math.max(v, aFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      predGrid[y][x] = floored.map(v => v / sum);
    }
  }

  // M/O hedging
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (!isMO[y][x]) continue;

      const nAvg = [0,0,0,0,0,0];
      let nW = 0;
      for (let dy = -3; dy <= 3; dy++) {
        for (let dx = -3; dx <= 3; dx++) {
          if (dy === 0 && dx === 0) continue;
          const ny = y + dy, nx = x + dx;
          if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
          if (isMO[ny][nx]) continue;
          const p = predGrid[ny][nx];
          if (!p) continue;
          const dist = Math.sqrt(dy * dy + dx * dx);
          const w = 1 / (1 + dist);
          for (let c = 0; c < 6; c++) nAvg[c] += w * p[c];
          nW += w;
        }
      }
      let neighborPred = nW > 0 ? nAvg.map(v => v / nW) : [1/6,1/6,1/6,1/6,1/6,1/6];

      let nS = 0;
      for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
        if (!dy && !dx) continue;
        const ny = y + dy, nx = x + dx;
        if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx)) nS++;
      }
      let coast = false;
      for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
        const ny = y + dy, nx = x + dx;
        if (ny >= 0 && ny < H && nx >= 0 && nx < W && initGrid[ny][nx] === 10) coast = true;
      }
      const sBucket = nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3';
      const cSuffix = coast ? 'c' : '';

      const surrogateAvg = [0,0,0,0,0,0];
      let sCount = 0;
      for (const t of ['P', 'F', 'S']) {
        let d = getKeyPred(t + sBucket + cSuffix);
        if (!d && cSuffix) d = getKeyPred(t + sBucket);
        if (d) { for (let c = 0; c < 6; c++) surrogateAvg[c] += d.avg[c]; sCount++; }
      }
      let surrogatePred = sCount > 0 ? surrogateAvg.map(v => v / sCount) : [1/6,1/6,1/6,1/6,1/6,1/6];

      const blended = new Array(6);
      for (let c = 0; c < 6; c++) {
        blended[c] = 0.40 * neighborPred[c] + 0.35 * surrogatePred[c] + 0.25 / 6;
      }
      const moFloor = 0.02;
      const floored = blended.map(v => Math.max(v, moFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      predGrid[y][x] = floored.map(v => v / sum);
    }
  }

  return predGrid;
}

module.exports = { predict };
