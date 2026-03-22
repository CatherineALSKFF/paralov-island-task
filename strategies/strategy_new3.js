const { H, W, mergeBuckets, selectClosestRounds } = require('./shared');

// Enhanced feature key matching the richer bucket data keys.
// Verified: a/b match perfectly, m/n/f approximate but meaningful.
// - nS=1-2: 'a' = adj settlement (Chebyshev 1), 'b' = dist 2, no suffix = dist 3
// - nS=0:   'm' = mountain in radius 3, 'n' = near forest, 'f' = isolated plains
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

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const K = config.K || 4;
  const floor = config.FLOOR || 0.0001;
  const regBlend = 0.10;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const candidates = { ...growthRates }; delete candidates[String(testRound)];

  const closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  const adaptiveModel = mergeBuckets(perRoundBuckets, closestRounds);

  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);
  const allModel = mergeBuckets(perRoundBuckets, allRounds);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getEnhancedKey(initGrid, settPos, y, x);

      // Primary lookup with hierarchical fallback
      let prior = null;
      for (let len = key.length; len >= 1 && !prior; len--) {
        const k = key.slice(0, len);
        if (adaptiveModel[k]) prior = [...adaptiveModel[k]];
        else if (allModel[k]) prior = [...allModel[k]];
      }
      if (!prior) prior = [1/6,1/6,1/6,1/6,1/6,1/6];

      // Mild blend toward all-rounds model for stability on bad K=4 selections
      const globalKey = allModel[key] ? key : key.slice(0, -1);
      const globalPrior = allModel[globalKey];
      if (globalPrior) {
        for (let c = 0; c < 6; c++) {
          prior[c] = (1 - regBlend) * prior[c] + regBlend * globalPrior[c];
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
