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
  const K = config.K || 4;
  const floor = config.FLOOR || 0.0001;
  const richBlend = config.RICH_BLEND !== undefined ? config.RICH_BLEND : 0.5;

  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const candidates = { ...growthRates };
  delete candidates[String(testRound)];
  const closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const fineModel = mergeBkts(perRoundBuckets, closestRounds);
  const allModel = mergeBkts(perRoundBuckets, allRounds);

  const settPos = new Set();
  const settList = [];
  for (const s of settlements) {
    settPos.add(s.y * W + s.x);
    settList.push(s);
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const { std, rich } = computeKeys(initGrid, settPos, settList, y, x);

      // Standard prediction with fallback
      let stdPred = fineModel[std];
      if (!stdPred) {
        const fb = std.slice(0, -1);
        stdPred = fineModel[fb] || allModel[std] || allModel[fb] || null;
      }
      if (!stdPred) stdPred = [1/6,1/6,1/6,1/6,1/6,1/6];

      // Rich prediction (only if different key)
      let richPred = null;
      if (rich !== std) {
        richPred = fineModel[rich] || allModel[rich] || null;
      }

      // Blend
      const dist = new Array(6);
      if (richPred && richBlend > 0) {
        for (let c = 0; c < 6; c++) {
          dist[c] = Math.max((1 - richBlend) * stdPred[c] + richBlend * richPred[c], floor);
        }
      } else {
        for (let c = 0; c < 6; c++) {
          dist[c] = Math.max(stdPred[c], floor);
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
