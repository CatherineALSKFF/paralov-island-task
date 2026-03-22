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

// Compute a richer feature vector for a cell
function cellFeatures(grid, settPos, y, x) {
  const v = grid[y][x];
  const terrain = v === 4 ? 2 : (v === 1 || v === 2) ? 1 : 0; // 0=plains, 1=sett, 2=forest

  // Settlement counts at multiple radii
  let nS1 = 0, nS3 = 0, nS5 = 0;
  let nF1 = 0, nF3 = 0;
  let nO1 = 0;
  let minSettDist = 99;

  for (let dy = -5; dy <= 5; dy++) for (let dx = -5; dx <= 5; dx++) {
    if (!dy && !dx) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    const dist = Math.abs(dy) + Math.abs(dx); // Manhattan distance
    const isSett = settPos.has(ny * W + nx);
    const isForest = grid[ny][nx] === 4;
    const isOcean = grid[ny][nx] === 10;

    if (isSett) {
      if (dist <= 5) nS5++;
      if (dist <= 3) nS3++;
      if (dist <= 1) nS1++;
      if (dist < minSettDist) minSettDist = dist;
    }
    if (isForest && dist <= 3) nF3++;
    if (isForest && dist <= 1) nF1++;
    if (isOcean && dist <= 1) nO1++;
  }

  return { terrain, nS1, nS3, nS5, nF1, nF3, nO1, minSettDist };
}

// Feature distance between two cells
function featureDist(a, b) {
  if (a.terrain !== b.terrain) return 100; // different terrain = infinite distance
  return (
    2.0 * (a.nS3 - b.nS3) ** 2 +
    0.5 * (a.nS5 - b.nS5) ** 2 +
    1.0 * (a.nF3 - b.nF3) ** 2 +
    3.0 * (a.nO1 - b.nO1) ** 2 +
    1.5 * Math.min((a.minSettDist - b.minSettDist) ** 2, 25)
  );
}

function buildCellDB(growthRates, testRound) {
  if (cachedTestRound === testRound && cachedDB) return cachedDB;

  // Group by terrain type for efficiency
  const db = { 0: [], 1: [], 2: [] }; // plains, settlement, forest
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
          if (grid[yy][xx] === 10 || grid[yy][xx] === 5) continue; // skip M/O
          const feat = cellFeatures(grid, settPos, yy, xx);
          db[feat.terrain].push({ feat, gt: gt[yy][xx], growth });
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
  const fBandwidth = config.FB || 8.0;
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const db = buildCellDB(growthRates, testRound);

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

      const feat = cellFeatures(initGrid, settPos, y, x);
      const candidates = db[feat.terrain];

      if (!candidates || candidates.length === 0) {
        predGrid[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6];
        continue;
      }

      const avg = [0,0,0,0,0,0];
      let tw = 0;
      for (let i = 0; i < candidates.length; i++) {
        const m = candidates[i];
        const gDiff = m.growth - targetGrowth;
        const gw = Math.exp(-0.5 * (gDiff / sigma) * (gDiff / sigma));
        if (gw < 0.001) continue; // skip very distant growth rates
        const fd = featureDist(feat, m.feat);
        const fw = Math.exp(-fd / fBandwidth);
        const w = gw * fw;
        if (w < 1e-8) continue;
        for (let c = 0; c < 6; c++) avg[c] += w * m.gt[c];
        tw += w;
      }
      if (tw > 0) for (let c = 0; c < 6; c++) avg[c] /= tw;
      else avg.fill(1/6);

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
        const tIdx = t === 'P' ? 0 : t === 'S' ? 1 : 2;
        const tCands = db[tIdx];
        if (!tCands || tCands.length === 0) continue;
        const sAvg = [0,0,0,0,0,0];
        let sw = 0;
        for (const m of tCands) {
          const gDiff = m.growth - targetGrowth;
          const gw = Math.exp(-0.5 * (gDiff / sigma) * (gDiff / sigma));
          if (gw < 0.001) continue;
          const w = gw;
          for (let c = 0; c < 6; c++) sAvg[c] += w * m.gt[c];
          sw += w;
        }
        if (sw > 0) {
          for (let c = 0; c < 6; c++) surrogateAvg[c] += sAvg[c] / sw;
          sCount++;
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
