
const H = 40, W = 40;
function terrainToClass(code) {
  if (code === 10 || code === 11 || code === 0) return 0;
  if (code >= 1 && code <= 5) return code;
  return 0;
}
function getFeatureKey(grid, settPos, y, x) {
  const v = grid[y][x];
  if (v === 10) return 'O'; if (v === 5) return 'M';
  const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P';
  let nS = 0;
  for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
    if (!dy && !dx) continue; const ny = y+dy, nx = x+dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny*W+nx)) nS++;
  }
  let coast = false;
  for (const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y+dy, nx = x+dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) coast = true;
  }
  return t + (nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3') + (coast ? 'c' : '');
}
function mergeBuckets(perRoundBuckets, roundNums) {
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
function selectClosestRounds(growthRates, targetRate, K) {
  return Object.entries(growthRates)
    .map(([rn, rate]) => ({ rn: parseInt(rn), dist: Math.abs(rate - targetRate) }))
    .sort((a, b) => a.dist - b.dist)
    .slice(0, K)
    .map(c => c.rn);
}
module.exports = { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds };
