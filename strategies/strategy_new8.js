const { H, W, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function getEnhancedKey(grid, settPos, y, x) {
  const init = grid[y][x];
  if (init === 10) return 'O';
  if (init === 5) return 'M';
  const t = init === 4 ? 'F' : (init === 1 || init === 2) ? 'S' : 'P';

  let nS = 0, minSDist = 999;
  for (let dy = -3; dy <= 3; dy++) {
    for (let dx = -3; dx <= 3; dx++) {
      if (!dy && !dx) continue;
      const ny = y + dy, nx = x + dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx)) {
        nS++;
        const d = Math.max(Math.abs(dy), Math.abs(dx));
        if (d < minSDist) minSDist = d;
      }
    }
  }

  let coastal = false;
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) coastal = true;
  }

  const nKey = nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3';
  const base = t + nKey + (coastal ? 'c' : '');

  if (t !== 'S') {
    if (nS === 0) {
      let nM = 0, nF = 0;
      for (let dy = -3; dy <= 3; dy++) {
        for (let dx = -3; dx <= 3; dx++) {
          if (!dy && !dx) continue;
          const ny = y + dy, nx = x + dx;
          if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
            if (grid[ny][nx] === 5) nM++;
            if (grid[ny][nx] === 4) nF++;
          }
        }
      }
      if (nM > 0) return base + 'm';
      if (t === 'P' && nF === 0) return base + 'f';
      return base + 'n';
    } else if (nS <= 2) {
      if (minSDist === 1) return base + 'a';
      if (minSDist === 2) return base + 'b';
    }
  }
  return base;
}

function gaussianMerge(perRoundBuckets, growthRates, targetGrowth, sigma, excludeRound) {
  const rounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== excludeRound);
  const weights = {};
  for (const r of rounds) {
    const diff = (growthRates[String(r)] || 0.15) - targetGrowth;
    weights[r] = Math.exp(-diff * diff / (2 * sigma * sigma));
  }
  const model = {};
  for (const r of rounds) {
    const b = perRoundBuckets[String(r)];
    if (!b) continue;
    const w = weights[r];
    for (const [key, val] of Object.entries(b)) {
      if (!model[key]) model[key] = { count: 0, sum: [0,0,0,0,0,0] };
      model[key].count += w * val.count;
      for (let c = 0; c < 6; c++) model[key].sum[c] += w * val.sum[c];
    }
  }
  const out = {};
  for (const [k, v] of Object.entries(model)) {
    if (v.count > 0) out[k] = v.sum.map(s => s / v.count);
  }
  return out;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const sigma = 0.045;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const adaptiveModel = gaussianMerge(perRoundBuckets, growthRates, targetGrowth, sigma, testRound);
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);
  const allModel = mergeBuckets(perRoundBuckets, allRounds);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getEnhancedKey(initGrid, settPos, y, x);

      let prior = adaptiveModel[key] ? [...adaptiveModel[key]] : allModel[key] ? [...allModel[key]] : null;
      if (!prior) {
        for (let len = key.length - 1; len >= 1 && !prior; len--) {
          const fb = key.slice(0, len);
          if (adaptiveModel[fb]) prior = [...adaptiveModel[fb]];
          else if (allModel[fb]) prior = [...allModel[fb]];
        }
        if (!prior) prior = [1/6,1/6,1/6,1/6,1/6,1/6];
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
