const { H, W, selectClosestRounds } = require('./shared');

function computeKeys(grid, sp, sl, y, x) {
  const v = grid[y][x];
  if (v === 10) return { std: 'O', rich: 'O', dist: 999 };
  if (v === 5) return { std: 'M', rich: 'M', dist: 999 };
  const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P';
  let nS = 0;
  for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
    if (!dy && !dx) continue;
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && sp.has(ny * W + nx)) nS++;
  }
  let coast = false;
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) coast = true;
  }
  const std = t + (nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3') + (coast ? 'c' : '');
  let rich = std;
  let md = 999;
  if (sl.length > 0) {
    for (const s of sl) md = Math.min(md, Math.max(Math.abs(s.y - y), Math.abs(s.x - x)));
  }
  if (nS === 0 && t !== 'S' && sl.length > 0) {
    rich = std + (md <= 4 ? 'n' : md <= 8 ? 'm' : 'f');
  } else if (nS >= 1 && nS <= 2 && t !== 'S') {
    let nearD = 999;
    for (const s of sl) {
      const d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
      if (d <= 3) nearD = Math.min(nearD, d);
    }
    rich = std + (nearD <= 1 ? 'a' : nearD <= 2 ? 'b' : '');
  }
  return { std, rich, dist: md };
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const bw = config.BW || 0.042;
  const temp = config.TEMP || 1.10;
  const regW = config.REG_W || 0.0;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  // Build Gaussian-weighted per-round-normalized model
  const weights = {};
  for (const [rn, rate] of Object.entries(growthRates)) {
    if (parseInt(rn) === testRound) continue;
    if (!perRoundBuckets[rn]) continue;
    weights[rn] = Math.max(0.001, Math.exp(-0.5 * ((rate - targetGrowth) / bw) ** 2));
  }

  const model = {};
  for (const [rn, w] of Object.entries(weights)) {
    const b = perRoundBuckets[rn];
    if (!b) continue;
    for (const [key, v] of Object.entries(b)) {
      if (!model[key]) model[key] = { d: [0,0,0,0,0,0], tw: 0 };
      const rd = v.sum.map(s => s / v.count);
      for (let c = 0; c < 6; c++) model[key].d[c] += w * rd[c];
      model[key].tw += w;
    }
  }
  const modelDist = {};
  for (const [k, v] of Object.entries(model)) {
    modelDist[k] = v.d.map(s => s / v.tw);
  }

  const settPos = new Set();
  const settList = [];
  for (const s of settlements) { settPos.add(s.y * W + s.x); settList.push(s); }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const { std, rich } = computeKeys(initGrid, settPos, settList, y, x);

      // Lookup with fallback chain
      let fineP = null, coarseP = null;
      if (rich !== std) fineP = modelDist[rich];
      if (!fineP) fineP = modelDist[std];
      const fb = std.length > 1 ? std.slice(0, -1) : null;
      if (fb) coarseP = modelDist[fb];

      let p = fineP || coarseP;
      if (!p) { row.push([1/6,1/6,1/6,1/6,1/6,1/6]); continue; }

      let prior = [...p];

      // Regularization toward coarser key
      if (regW > 0.001 && fineP && coarseP) {
        for (let c = 0; c < 6; c++) {
          prior[c] = (1 - regW) * fineP[c] + regW * coarseP[c];
        }
      }

      // Temperature scaling
      if (temp > 1.01) {
        let s = 0;
        for (let c = 0; c < 6; c++) {
          prior[c] = Math.pow(Math.max(prior[c], 1e-10), 1 / temp);
          s += prior[c];
        }
        for (let c = 0; c < 6; c++) prior[c] /= s;
      }

      // Adaptive floor (quadratic in entropy ratio)
      let entropy = 0;
      for (let c = 0; c < 6; c++) {
        if (prior[c] > 1e-10) entropy -= prior[c] * Math.log(prior[c]);
      }
      const eRatio = entropy / Math.log(6);
      const cellFloor = floor * (0.02 + 0.98 * eRatio * eRatio);

      const floored = prior.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
