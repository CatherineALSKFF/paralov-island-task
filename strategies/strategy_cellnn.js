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

// Module-level cache
let cachedDB = null;
let cachedTestRound = null;

function buildCellDB(growthRates, testRound) {
  if (cachedTestRound === testRound && cachedDB) return cachedDB;

  const db = {}; // featureKey -> [{gt:[6], nS:int, nF:int, growth:float}]
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

          let nS = 0, nF = 0;
          for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
            if (!dy && !dx) continue;
            const ny = y + dy, nx = x + dx;
            if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
            if (settPos.has(ny * W + nx)) nS++;
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
  const sigma = config.SIGMA || 0.05;
  const fSigma = config.FS || 2.0;
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const db = buildCellDB(growthRates, testRound);

  // Pre-compute Gaussian growth weights per unique growth rate
  const growthWeightCache = {};
  function growthWeight(g) {
    if (growthWeightCache[g] !== undefined) return growthWeightCache[g];
    const d = g - targetGrowth;
    growthWeightCache[g] = Math.exp(-0.5 * (d / sigma) * (d / sigma));
    return growthWeightCache[g];
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

      let nS = 0, nF = 0;
      for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
        if (!dy && !dx) continue;
        const ny = y + dy, nx = x + dx;
        if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
        if (settPos.has(ny * W + nx)) nS++;
        if (initGrid[ny][nx] === 4) nF++;
      }

      let matches = db[key];
      if (!matches || matches.length === 0) {
        const fb = key.slice(0, -1);
        matches = db[fb];
      }

      if (!matches || matches.length === 0) {
        predGrid[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6];
        continue;
      }

      const avg = [0,0,0,0,0,0];
      let tw = 0;
      const fs2 = 2 * fSigma * fSigma;
      for (let i = 0; i < matches.length; i++) {
        const m = matches[i];
        const gw = growthWeight(m.growth);
        const nsDiff = m.nS - nS;
        const nfDiff = m.nF - nF;
        const fw = Math.exp(-(nsDiff * nsDiff + nfDiff * nfDiff) / fs2);
        const w = gw * fw;
        if (w < 1e-10) continue;
        for (let c = 0; c < 6; c++) avg[c] += w * m.gt[c];
        tw += w;
      }
      if (tw > 0) for (let c = 0; c < 6; c++) avg[c] /= tw;
      else { avg.fill(1/6); }

      const floored = avg.map(v => Math.max(v, floor));
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

      // Terrain surrogates
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

      // Use cell-level DB for surrogates too
      const surrogateAvg = [0,0,0,0,0,0];
      let sCount = 0;
      for (const t of ['P', 'F', 'S']) {
        const sKey = t + sBucket + cSuffix;
        const matches = db[sKey] || (cSuffix ? db[t + sBucket] : null);
        if (matches && matches.length > 0) {
          let sw = 0;
          const sAvg = [0,0,0,0,0,0];
          for (const m of matches) {
            const w = growthWeight(m.growth);
            for (let c = 0; c < 6; c++) sAvg[c] += w * m.gt[c];
            sw += w;
          }
          if (sw > 0) {
            for (let c = 0; c < 6; c++) surrogateAvg[c] += sAvg[c] / sw;
            sCount++;
          }
        }
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
