const { H, W, getFeatureKey, mergeBuckets } = require('./shared');
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'data');

let _distBuckets = null;
function loadDistBuckets() {
  if (_distBuckets) return _distBuckets;
  const f = path.join(DATA_DIR, 'enriched_buckets_v8.json');
  if (fs.existsSync(f)) _distBuckets = JSON.parse(fs.readFileSync(f, 'utf8'));
  else _distBuckets = {};
  return _distBuckets;
}

// Compute round-level features from initial grids
const _roundFeats = {};
function getRoundFeatures(roundNum) {
  if (_roundFeats[roundNum]) return _roundFeats[roundNum];
  const f = path.join(DATA_DIR, 'inits_R' + roundNum + '.json');
  if (!fs.existsSync(f)) return null;
  const raw = JSON.parse(fs.readFileSync(f, 'utf8'));
  let totS = 0, totF = 0, totO = 0, nSeeds = 0, totCluster = 0;
  for (let s = 0; s < 5; s++) {
    const item = raw[s]; if (!item) continue;
    const grid = Array.isArray(item) && Array.isArray(item[0]) ? item : item.grid;
    if (!grid) continue;
    nSeeds++;
    const setts = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const v = grid[y][x];
      if (v === 1 || v === 2) { totS++; setts.push({ y, x }); }
      else if (v === 4) totF++;
      else if (v === 10) totO++;
    }
    // Avg min distance between settlements (clustering measure)
    let cs = 0;
    for (let i = 0; i < setts.length; i++) {
      let md = 80;
      for (let j = 0; j < setts.length; j++) {
        if (i === j) continue;
        md = Math.min(md, Math.abs(setts[i].y - setts[j].y) + Math.abs(setts[i].x - setts[j].x));
      }
      cs += md;
    }
    totCluster += setts.length > 1 ? cs / setts.length : 10;
  }
  _roundFeats[roundNum] = {
    sett: totS / nSeeds,
    forest: totF / nSeeds,
    ocean: totO / nSeeds,
    cluster: totCluster / nSeeds,
  };
  return _roundFeats[roundNum];
}

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

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const SIGMA_G = config.SIGMA_G || 0.05;
  const SIGMA_S = config.SIGMA_S || 8;    // settlement count bandwidth
  const LAMBDA = config.LAMBDA || 0.0005;
  const REG_BLEND = config.REG_BLEND || 0.05;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const distBuckets = loadDistBuckets();
  const hasDistBuckets = Object.keys(distBuckets).length > 0;

  const distRounds = Object.keys(distBuckets).map(Number)
    .filter(n => n !== testRound && growthRates[String(n)] !== undefined);
  const stdRounds = Object.keys(perRoundBuckets).map(Number)
    .filter(n => n !== testRound);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Multi-feature round similarity: growth rate + settlement count
  const testFeats = getRoundFeatures(testRound);
  const testSett = testFeats ? testFeats.sett : 45;

  const roundWeights = {};
  for (const rn of [...new Set([...distRounds, ...stdRounds])]) {
    const g = growthRates[String(rn)];
    if (g === undefined) { roundWeights[rn] = 0; continue; }
    const dg = g - targetGrowth;
    let w = Math.exp(-(dg * dg) / (2 * SIGMA_G * SIGMA_G));
    // Add settlement count similarity if available
    const rnFeats = getRoundFeatures(rn);
    if (rnFeats && testFeats) {
      const ds = rnFeats.sett - testSett;
      w *= Math.exp(-(ds * ds) / (2 * SIGMA_S * SIGMA_S));
    }
    roundWeights[rn] = w;
  }

  // Gaussian-weighted average model
  function buildGaussModel(buckets, rounds) {
    const model = {};
    for (const rn of rounds) {
      const w = roundWeights[rn]; if (!w) continue;
      const b = buckets[String(rn)]; if (!b) continue;
      for (const [key, val] of Object.entries(b)) {
        if (!model[key]) model[key] = { wsum: new Array(6).fill(0), wtotal: 0 };
        const avg = val.sum.map(v => v / val.count);
        for (let c = 0; c < 6; c++) model[key].wsum[c] += w * avg[c];
        model[key].wtotal += w;
      }
    }
    const out = {};
    for (const [k, v] of Object.entries(model)) {
      out[k] = v.wsum.map(s => s / v.wtotal);
    }
    return out;
  }

  // Ridge regression model
  function buildRegModel(buckets, rounds) {
    const allKeys = new Set();
    for (const rn of rounds) {
      const b = buckets[String(rn)];
      if (b) for (const k of Object.keys(b)) allKeys.add(k);
    }
    const model = {};
    for (const key of allKeys) {
      const classes = [];
      for (let c = 0; c < 6; c++) {
        let sW = 0, sWX = 0, sWY = 0, sWXX = 0, sWXY = 0, n = 0;
        for (const rn of rounds) {
          const w = roundWeights[rn]; if (!w) continue;
          const bucket = buckets[String(rn)]?.[key];
          if (!bucket || bucket.count === 0) continue;
          const g = growthRates[String(rn)];
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
          classes.push({ intercept: 1 / 6, slope: 0 });
        }
      }
      model[key] = classes;
    }
    return model;
  }

  const distGauss = hasDistBuckets ? buildGaussModel(distBuckets, distRounds) : {};
  const stdGauss = buildGaussModel(perRoundBuckets, stdRounds);
  const distReg = hasDistBuckets ? buildRegModel(distBuckets, distRounds) : {};
  const stdReg = buildRegModel(perRoundBuckets, stdRounds);

  function regPred(model, key) {
    if (!model[key]) return null;
    const pred = model[key].map(m => Math.max(0, m.intercept + m.slope * targetGrowth));
    const sum = pred.reduce((a, b) => a + b, 0);
    return sum > 0 ? pred.map(v => v / sum) : null;
  }

  function ensembleLookup(gaussModel, regModel, key) {
    const g = gaussModel[key] ? [...gaussModel[key]] : null;
    const r = regPred(regModel, key);
    if (g && r) return g.map((v, c) => 0.5 * v + 0.5 * r[c]);
    return g || r;
  }

  function coarsenDist(key) {
    if (key === 'O' || key === 'M' || key.length <= 1) return [];
    const levels = [];
    let k = key;
    if (k.endsWith('t')) { k = k.slice(0, -1); levels.push(k); }
    if (k.endsWith('c')) { k = k.slice(0, -1); levels.push(k); }
    return levels;
  }

  function lookupDist(key) {
    let p = ensembleLookup(distGauss, distReg, key);
    if (p) return p;
    for (const ck of coarsenDist(key)) {
      p = ensembleLookup(distGauss, distReg, ck);
      if (p) return p;
    }
    return null;
  }

  function lookupStd(key) {
    let p = ensembleLookup(stdGauss, stdReg, key);
    if (p) return p;
    if (key.length > 1) {
      p = ensembleLookup(stdGauss, stdReg, key.slice(0, -1));
      if (p) return p;
    }
    return null;
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const dKey = getDistKey(initGrid, settPos, y, x);
      const sKey = getFeatureKey(initGrid, settPos, y, x);

      let specific = lookupDist(dKey);
      if (!specific) specific = lookupStd(sKey);

      let coarse = null;
      const cks = coarsenDist(dKey);
      for (const ck of cks) {
        coarse = ensembleLookup(distGauss, distReg, ck);
        if (coarse) break;
      }
      if (!coarse && sKey.length > 1) {
        coarse = ensembleLookup(stdGauss, stdReg, sKey.slice(0, -1));
      }

      let prior;
      if (specific && coarse) {
        prior = specific.map((v, c) => (1 - REG_BLEND) * v + REG_BLEND * coarse[c]);
      } else if (specific) {
        prior = specific;
      } else {
        prior = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6];
      }

      let entropy = 0;
      for (let c = 0; c < 6; c++) {
        if (prior[c] > 0.001) entropy -= prior[c] * Math.log(prior[c]);
      }
      const entRatio = Math.min(entropy / Math.log(6), 1);
      const cellFloor = floor * (0.05 + 0.95 * entRatio);

      const floored = prior.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
