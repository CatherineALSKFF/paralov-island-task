const { H, W, getFeatureKey } = require('./shared');
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'data');
const ROUND_IDS = {
  1:'71451d74',2:'76909e29',3:'f1dac9a9',4:'8e839974',5:'fd3c92ff',
  6:'ae78003a',7:'36e581f1',8:'c5cdf100',9:'2a341ace',10:'75e625c3',
  11:'324fde07',12:'795bfb1f',13:'7b4bda99',14:'d0a2c894',15:'cc5442dd',
};

// Distance-based feature key
function getDistKey(grid, settPos, y, x) {
  const v = grid[y][x];
  if (v === 10) return 'O';
  if (v === 5) return 'M';
  const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P';
  let nO1 = 0, nS1 = 0;
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
      if (grid[ny][nx] === 10) nO1++;
      if (settPos.has(ny * W + nx)) nS1++;
    }
  }
  const coastal = nO1 > 0;
  if (t === 'S') {
    let nS = 0;
    for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
      if (!dy && !dx) continue;
      const ny = y + dy, nx = x + dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx)) nS++;
    }
    return 'S' + (nS <= 1 ? '0' : nS <= 3 ? '1' : '2') + (coastal ? 'c' : '');
  }
  let minSD = 40;
  for (let dy = -10; dy <= 10; dy++) for (let dx = -10; dx <= 10; dx++) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx))
      minSD = Math.min(minSD, Math.max(Math.abs(dy), Math.abs(dx)));
  }
  const dKey = minSD <= 1 ? '1' : minSD <= 2 ? '2' : minSD <= 3 ? '3' : minSD <= 5 ? '4' : minSD <= 7 ? '5' : '6';
  const suffix = (minSD <= 1 && nS1 > 0) ? 't' : '';
  return t + dKey + (coastal ? 'c' : '') + suffix;
}

// Build/load distance-based buckets
let _distBuckets = null;
function getDistBuckets() {
  if (_distBuckets) return _distBuckets;
  const cacheFile = path.join(DATA_DIR, 'enriched_buckets_v8.json');
  if (fs.existsSync(cacheFile)) {
    _distBuckets = JSON.parse(fs.readFileSync(cacheFile, 'utf8'));
    return _distBuckets;
  }
  _distBuckets = {};
  for (const [rn, prefix] of Object.entries(ROUND_IDS)) {
    _distBuckets[rn] = {};
    const initsFile = path.join(DATA_DIR, 'inits_R' + rn + '.json');
    if (!fs.existsSync(initsFile)) continue;
    const inits = JSON.parse(fs.readFileSync(initsFile, 'utf8'));
    for (let si = 0; si < 5; si++) {
      const gtFile = path.join(DATA_DIR, 'gt_' + prefix + '_s' + si + '.json');
      if (!fs.existsSync(gtFile)) continue;
      const gtRaw = JSON.parse(fs.readFileSync(gtFile, 'utf8'));
      const gt = gtRaw.ground_truth || gtRaw.gt;
      if (!gt) continue;
      const item = inits[si];
      if (!item) continue;
      const grid = Array.isArray(item) && Array.isArray(item[0]) ? item : item.grid;
      if (!grid) continue;
      const settPos = new Set();
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
        if (grid[y][x] === 1 || grid[y][x] === 2) settPos.add(y * W + x);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const key = getDistKey(grid, settPos, y, x);
        if (!_distBuckets[rn][key]) _distBuckets[rn][key] = { count: 0, sum: [0,0,0,0,0,0] };
        _distBuckets[rn][key].count++;
        for (let c = 0; c < 6; c++) _distBuckets[rn][key].sum[c] += gt[y][x][c];
      }
    }
  }
  fs.writeFileSync(cacheFile, JSON.stringify(_distBuckets));
  return _distBuckets;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.00001;
  const LAMBDA = config.LAMBDA || 0.001;
  const SIGMA = config.SIGMA || 0.12;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const distBuckets = getDistBuckets();
  const allRounds = Object.keys(distBuckets).map(Number).filter(n => n !== testRound);

  // Gaussian weights for weighted regression
  const roundWeights = {};
  for (const rn of allRounds) {
    const g = growthRates[String(rn)];
    if (g === undefined) { roundWeights[rn] = 0.1; continue; }
    const d = g - targetGrowth;
    roundWeights[rn] = Math.exp(-(d * d) / (2 * SIGMA * SIGMA));
  }

  // Collect all distance-based keys
  const allKeys = new Set();
  for (const rn of allRounds) {
    const b = distBuckets[String(rn)];
    if (b) for (const k of Object.keys(b)) allKeys.add(k);
  }

  // Weighted ridge regression on distance-based buckets
  const regModel = {};
  for (const key of allKeys) {
    const classes = [];
    for (let c = 0; c < 6; c++) {
      let sW = 0, sWX = 0, sWY = 0, sWXX = 0, sWXY = 0, n = 0;
      for (const rn of allRounds) {
        const g = growthRates[String(rn)];
        if (g === undefined) continue;
        const bucket = distBuckets[String(rn)]?.[key];
        if (!bucket || bucket.count === 0) continue;
        const w = roundWeights[rn];
        const p = bucket.sum[c] / bucket.count;
        sW += w; sWX += w * g; sWY += w * p;
        sWXX += w * g * g; sWXY += w * g * p; n++;
      }
      if (n >= 4 && sW > 0) {
        const denom = sW * sWXX - sWX * sWX + LAMBDA * sW;
        const slope = denom > 1e-10 ? (sW * sWXY - sWX * sWY) / denom : 0;
        const intercept = (sWY - slope * sWX) / sW;
        classes.push({ intercept, slope });
      } else if (sW > 0) {
        classes.push({ intercept: sWY / sW, slope: 0 });
      } else {
        classes.push({ intercept: 1/6, slope: 0 });
      }
    }
    regModel[key] = classes;
  }

  // Also build regression for standard keys as fallback
  const stdAllKeys = new Set();
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (b) for (const k of Object.keys(b)) stdAllKeys.add(k);
  }
  const stdRegModel = {};
  for (const key of stdAllKeys) {
    const classes = [];
    for (let c = 0; c < 6; c++) {
      let sW = 0, sWX = 0, sWY = 0, sWXX = 0, sWXY = 0, n = 0;
      for (const rn of allRounds) {
        const g = growthRates[String(rn)];
        if (g === undefined) continue;
        const bucket = perRoundBuckets[String(rn)]?.[key];
        if (!bucket || bucket.count === 0) continue;
        const w = roundWeights[rn];
        const p = bucket.sum[c] / bucket.count;
        sW += w; sWX += w * g; sWY += w * p;
        sWXX += w * g * g; sWXY += w * g * p; n++;
      }
      if (n >= 4 && sW > 0) {
        const denom = sW * sWXX - sWX * sWX + LAMBDA * sW;
        const slope = denom > 1e-10 ? (sW * sWXY - sWX * sWY) / denom : 0;
        const intercept = (sWY - slope * sWX) / sW;
        classes.push({ intercept, slope });
      } else if (sW > 0) {
        classes.push({ intercept: sWY / sW, slope: 0 });
      } else {
        classes.push({ intercept: 1/6, slope: 0 });
      }
    }
    stdRegModel[key] = classes;
  }

  function regPred(model, key) {
    if (!model[key]) return null;
    const pred = model[key].map(m => Math.max(0, m.intercept + m.slope * targetGrowth));
    const sum = pred.reduce((a, b) => a + b, 0);
    return sum > 0 ? pred.map(v => v / sum) : null;
  }

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const dKey = getDistKey(initGrid, settPos, y, x);

      // Try distance-based regression first
      let prior = regPred(regModel, dKey);
      if (!prior) {
        const fb = dKey.slice(0, -1);
        prior = regPred(regModel, fb);
      }
      if (!prior) {
        // Fallback to standard key regression
        const sKey = getFeatureKey(initGrid, settPos, y, x);
        prior = regPred(stdRegModel, sKey);
        if (!prior) {
          const fb = sKey.slice(0, -1);
          prior = regPred(stdRegModel, fb) || [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
        }
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
