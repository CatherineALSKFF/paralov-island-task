const { H, W, selectClosestRounds } = require('./shared');

function mergeBkts(perRoundBuckets, roundNums) {
  const m = {};
  for (const rn of roundNums) {
    const b = perRoundBuckets[String(rn)]; if (!b) continue;
    for (const [k, v] of Object.entries(b)) {
      if (!m[k]) m[k] = { count: 0, sum: [0,0,0,0,0,0] };
      m[k].count += v.count;
      for (let c = 0; c < 6; c++) m[k].sum[c] += v.sum[c];
    }
  }
  const out = {};
  for (const [k, v] of Object.entries(m)) out[k] = v.sum.map(s => s / v.count);
  return out;
}

function computeKeys(grid, settPos, settList, y, x) {
  const v = grid[y][x];
  if (v === 10) return { std: 'O', rich: 'O' };
  if (v === 5) return { std: 'M', rich: 'M' };

  const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P';
  let nS = 0;
  for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
    if (!dy && !dx) continue;
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx)) nS++;
  }
  let coast = false;
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) coast = true;
  }
  const std = t + (nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3') + (coast ? 'c' : '');

  let rich = std;
  if (nS === 0 && t !== 'S' && settList.length > 0) {
    let minDist = 999;
    for (const s of settList) minDist = Math.min(minDist, Math.max(Math.abs(s.y - y), Math.abs(s.x - x)));
    rich = std + (minDist <= 6 ? 'n' : minDist <= 10 ? 'm' : 'f');
  } else if (nS >= 1 && nS <= 2 && t !== 'S') {
    let minDist = 999;
    for (const s of settList) {
      const d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
      if (d <= 3) minDist = Math.min(minDist, d);
    }
    rich = std + (minDist <= 1 ? 'a' : minDist <= 2 ? 'b' : '');
  }

  return { std, rich };
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const K = config.K || 3;
  const floor = config.FLOOR || 0.0001;
  const wRich = config.W_RICH !== undefined ? config.W_RICH : 0.5;
  const wStd = config.W_STD !== undefined ? config.W_STD : 0.35;
  const wCoarse = config.W_COARSE !== undefined ? config.W_COARSE : 0.15;

  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const candidates = { ...growthRates };
  delete candidates[String(testRound)];
  const closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const fineModel = mergeBkts(perRoundBuckets, closestRounds);
  const allModel = mergeBkts(perRoundBuckets, allRounds);

  // Build terrain-only model (coarsest level)
  const terrainModel = {};
  for (const rn of closestRounds) {
    const b = perRoundBuckets[String(rn)]; if (!b) continue;
    for (const [k, v] of Object.entries(b)) {
      const tKey = k[0];
      if (!terrainModel[tKey]) terrainModel[tKey] = { count: 0, sum: [0,0,0,0,0,0] };
      terrainModel[tKey].count += v.count;
      for (let c = 0; c < 6; c++) terrainModel[tKey].sum[c] += v.sum[c];
    }
  }
  const terrainDist = {};
  for (const [k, v] of Object.entries(terrainModel)) {
    terrainDist[k] = v.sum.map(s => s / v.count);
  }

  const settPos = new Set();
  const settList = [];
  for (const s of settlements) { settPos.add(s.y * W + s.x); settList.push(s); }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const { std, rich } = computeKeys(initGrid, settPos, settList, y, x);

      // Get predictions at each level
      let richPred = (rich !== std) ? (fineModel[rich] || allModel[rich]) : null;
      let stdPred = fineModel[std] || allModel[std];
      if (!stdPred) {
        const fb = std.slice(0, -1);
        stdPred = fineModel[fb] || allModel[fb];
      }
      const coarsePred = terrainDist[std[0]] || [1/6,1/6,1/6,1/6,1/6,1/6];

      if (!stdPred) stdPred = coarsePred;

      // Weighted blend of available levels
      const dist = new Array(6);
      if (richPred) {
        const totalW = wRich + wStd + wCoarse;
        for (let c = 0; c < 6; c++) {
          dist[c] = Math.max((wRich * richPred[c] + wStd * stdPred[c] + wCoarse * coarsePred[c]) / totalW, floor);
        }
      } else {
        const totalW = wStd + wCoarse;
        for (let c = 0; c < 6; c++) {
          dist[c] = Math.max((wStd * stdPred[c] + wCoarse * coarsePred[c]) / totalW, floor);
        }
      }

      const sum = dist.reduce((a, b) => a + b, 0);
      row.push(dist.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
