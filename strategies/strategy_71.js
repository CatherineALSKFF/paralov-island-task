const fs = require('fs');
const path = require('path');
const { H, W, terrainToClass, getFeatureKey, mergeBuckets } = require('./shared');

const DATA_DIR = path.join(__dirname, '..', 'data');
const ROUND_IDS = {
  1:'71451d74',2:'76909e29',3:'f1dac9a9',4:'8e839974',
  5:'fd3c92ff',6:'ae78003a',7:'36e581f1',8:'c5cdf100',
  9:'2a341ace',10:'75e625c3',11:'324fde07',12:'795bfb1f',
  13:'7b4bda99',14:'d0a2c894',15:'cc5442dd',
};

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
  const floor = config.FLOOR || 0.0001;
  const sigma = config.sigma || 0.045;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Nearest settlement distance
  const nearestDist = new Float32Array(H * W).fill(99);
  for (const s of settlements) {
    for (let y = Math.max(0, s.y - 12); y <= Math.min(H - 1, s.y + 12); y++) {
      for (let x = Math.max(0, s.x - 12); x <= Math.min(W - 1, s.x + 12); x++) {
        const d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
        const idx = y * W + x;
        if (d < nearestDist[idx]) nearestDist[idx] = d;
      }
    }
  }

  function getEnhancedKey(y, x) {
    const base = getFeatureKey(initGrid, settPos, y, x);
    if (base === 'O' || base === 'M' || base[0] === 'S') return base;
    const bucket = base[1];
    const minDist = nearestDist[y * W + x];
    if (bucket === '0') {
      if (minDist === 4) return base + 'n';
      if (minDist <= 8) return base + 'm';
      return base + 'f';
    }
    if (bucket === '1') {
      if (minDist === 1) return base + 'a';
      if (minDist === 2) return base + 'b';
    }
    return base;
  }

  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // MODEL 1: Gaussian-weighted merge with count-based MLE
  function gaussianMerge(sig) {
    const weights = {};
    let tw = 0;
    for (const rn of allRounds) {
      const diff = (growthRates[String(rn)] || 0.15) - targetGrowth;
      weights[rn] = Math.exp(-diff * diff / (2 * sig * sig));
      tw += weights[rn];
    }
    const model = {};
    for (const rn of allRounds) {
      const b = perRoundBuckets[String(rn)];
      if (!b) continue;
      const w = weights[rn] / tw;
      for (const [key, val] of Object.entries(b)) {
        if (!model[key]) model[key] = { wc: 0, ws: [0,0,0,0,0,0] };
        model[key].wc += w * val.count;
        for (let c = 0; c < 6; c++) model[key].ws[c] += w * val.sum[c];
      }
    }
    const out = {};
    for (const [k, v] of Object.entries(model)) {
      if (v.wc > 0) out[k] = v.ws.map(s => s / v.wc);
    }
    return out;
  }

  const gaussModel = gaussianMerge(sigma);
  const wideModel = gaussianMerge(0.15);

  // MODEL 2: Per-round-average LOESS (captures growth trends)
  const db = buildCellDB(growthRates, testRound);
  const loessCache = {};

  function fitLoess(key) {
    if (loessCache[key] !== undefined) return loessCache[key];
    const matches = db[key];
    if (!matches || matches.length < 20) { loessCache[key] = null; return null; }

    // Group by growth rate (round)
    const byGrowth = {};
    for (const m of matches) {
      const g = m.growth;
      if (!byGrowth[g]) { byGrowth[g] = { sum: [0,0,0,0,0,0], n: 0 }; }
      for (let c = 0; c < 6; c++) byGrowth[g].sum[c] += m.gt[c];
      byGrowth[g].n++;
    }

    const points = Object.entries(byGrowth).map(([g, v]) => ({
      g: parseFloat(g), avg: v.sum.map(s => s / v.n), n: v.n
    }));
    if (points.length < 3) { loessCache[key] = null; return null; }

    const loessBw = 0.12;
    const result = [0,0,0,0,0,0];
    for (let c = 0; c < 6; c++) {
      let sww = 0, swg = 0, swp = 0, swgg = 0, swgp = 0;
      for (const pt of points) {
        const gd = pt.g - targetGrowth;
        const w = Math.exp(-gd * gd / (2 * loessBw * loessBw)) * Math.sqrt(pt.n);
        sww += w; swg += w * pt.g; swp += w * pt.avg[c];
        swgg += w * pt.g * pt.g; swgp += w * pt.g * pt.avg[c];
      }
      const denom = sww * swgg - swg * swg;
      if (Math.abs(denom) < 1e-12) result[c] = swp / sww;
      else {
        const b = (sww * swgp - swg * swp) / denom;
        const a = (swp - b * swg) / sww;
        result[c] = Math.max(0, a + b * targetGrowth);
      }
    }
    const s = result.reduce((a, b) => a + b, 0);
    loessCache[key] = s > 0 ? result.map(v => v / s) : null;
    return loessCache[key];
  }

  // MODEL 3: Per-key inter-round disagreement for adaptive hedging
  const disCache = {};
  function getDisagreement(key) {
    if (disCache[key] !== undefined) return disCache[key];
    const avg = gaussModel[key];
    if (!avg) { disCache[key] = 0; return 0; }
    let dis = 0, tw = 0;
    for (const rn of allRounds) {
      const b = perRoundBuckets[String(rn)];
      if (!b || !b[key] || b[key].count < 3) continue;
      const diff = (growthRates[String(rn)] || 0.15) - targetGrowth;
      const w = Math.exp(-diff * diff / (2 * sigma * sigma));
      const rAvg = b[key].sum.map(v => v / b[key].count);
      let d = 0;
      for (let c = 0; c < 6; c++) { const dd = rAvg[c] - avg[c]; d += dd * dd; }
      dis += w * Math.sqrt(d);
      tw += w;
    }
    disCache[key] = tw > 0 ? dis / tw : 0;
    return disCache[key];
  }

  // Lookup with fallback
  function lookup(model, richKey, baseKey) {
    if (model[richKey]) return model[richKey];
    if (model[baseKey]) return model[baseKey];
    for (let len = baseKey.length - 1; len >= 1; len--) {
      const fb = baseKey.slice(0, len);
      if (model[fb]) return model[fb];
    }
    return null;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const richKey = getEnhancedKey(y, x);
      const baseKey = getFeatureKey(initGrid, settPos, y, x);

      // Get predictions from multiple models
      const gPred = lookup(gaussModel, richKey, baseKey);
      const wPred = lookup(wideModel, richKey, baseKey);
      const lPred = fitLoess(baseKey); // LOESS uses simple key (more data points)

      // Ensemble with data-driven blending
      const dis = getDisagreement(richKey) || getDisagreement(baseKey);

      // High disagreement → more weight on LOESS (captures trends) and wide model (stable)
      // Low disagreement → more weight on tight Gaussian (precise)
      const disNorm = Math.min(dis / 0.3, 1.0); // 0-1 scale
      const gW = 0.50 * (1 - 0.3 * disNorm);
      const lW = lPred ? 0.35 + 0.15 * disNorm : 0;
      const wW = 0.15 + 0.15 * disNorm;

      let prior = [0,0,0,0,0,0];
      let tw = 0;
      if (gPred) { for (let c=0;c<6;c++) prior[c] += gW * gPred[c]; tw += gW; }
      if (lPred) { for (let c=0;c<6;c++) prior[c] += lW * lPred[c]; tw += lW; }
      if (wPred) { for (let c=0;c<6;c++) prior[c] += wW * wPred[c]; tw += wW; }

      if (tw > 0) for (let c=0;c<6;c++) prior[c] /= tw;
      else prior = [1/6,1/6,1/6,1/6,1/6,1/6];

      // Regularize toward coarser key
      if (richKey !== baseKey) {
        const coarsePred = gaussModel[baseKey];
        if (coarsePred) {
          const regW = 0.08;
          for (let c=0;c<6;c++) prior[c] = (1-regW)*prior[c] + regW*coarsePred[c];
        }
      }

      // Adaptive floor: higher for uncertain predictions
      let entropy = 0;
      for (let c = 0; c < 6; c++) if (prior[c] > 0.001) entropy -= prior[c] * Math.log(prior[c]);
      const adaptiveFloor = entropy < 0.2 ? floor * 0.1 :
                            entropy > 1.0 ? floor * 3 : floor;

      const floored = prior.map(v => Math.max(v, adaptiveFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
