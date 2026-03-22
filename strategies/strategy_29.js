const { H, W, selectClosestRounds } = require('./shared');

function computeKeys(grid, sp, sl, y, x) {
  const v = grid[y][x];
  if (v === 10) return { std: 'O', rich: 'O' };
  if (v === 5) return { std: 'M', rich: 'M' };
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
  if (nS === 0 && t !== 'S' && sl.length > 0) {
    let md = 999;
    for (const s of sl) md = Math.min(md, Math.max(Math.abs(s.y - y), Math.abs(s.x - x)));
    rich = std + (md <= 4 ? 'n' : md <= 8 ? 'm' : 'f');
  } else if (nS >= 1 && nS <= 2 && t !== 'S') {
    let md = 999;
    for (const s of sl) {
      const d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
      if (d <= 3) md = Math.min(md, d);
    }
    rich = std + (md <= 1 ? 'a' : md <= 2 ? 'b' : '');
  }
  return { std, rich };
}

function buildGaussModel(perRoundBuckets, growthRates, targetGrowth, testRound, bw, minW) {
  const model = {};
  for (const [rn, rate] of Object.entries(growthRates)) {
    if (parseInt(rn) === testRound) continue;
    const b = perRoundBuckets[rn];
    if (!b) continue;
    const dist = Math.abs(rate - targetGrowth);
    const w = Math.max(minW, Math.exp(-0.5 * (dist / bw) ** 2));
    for (const [key, v] of Object.entries(b)) {
      if (!model[key]) model[key] = { d: [0,0,0,0,0,0], tw: 0 };
      const rd = v.sum.map(s => s / v.count);
      for (let c = 0; c < 6; c++) model[key].d[c] += w * rd[c];
      model[key].tw += w;
    }
  }
  const out = {};
  for (const [k, v] of Object.entries(model)) {
    out[k] = v.d.map(s => s / v.tw);
  }
  return out;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const temp = config.TEMP || 1.10;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  // Ensemble of models with different bandwidths
  const bandwidths = [0.03, 0.042, 0.06];
  const bwWeights = [0.25, 0.50, 0.25];
  const models = bandwidths.map(bw => buildGaussModel(perRoundBuckets, growthRates, targetGrowth, testRound, bw, 0.001));

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

      // Ensemble lookup
      let prior = new Array(6).fill(0);
      let totalW = 0;
      for (let m = 0; m < models.length; m++) {
        let p = null;
        if (rich !== std) p = models[m][rich];
        if (!p) p = models[m][std];
        if (!p) { const fb = std.slice(0, -1); p = models[m][fb]; }
        if (p) {
          for (let c = 0; c < 6; c++) prior[c] += bwWeights[m] * p[c];
          totalW += bwWeights[m];
        }
      }
      if (totalW < 0.01) { row.push([1/6,1/6,1/6,1/6,1/6,1/6]); continue; }
      for (let c = 0; c < 6; c++) prior[c] /= totalW;

      // Temperature scaling
      if (temp > 1.01) {
        let s = 0;
        for (let c = 0; c < 6; c++) {
          prior[c] = Math.pow(Math.max(prior[c], 1e-10), 1 / temp);
          s += prior[c];
        }
        for (let c = 0; c < 6; c++) prior[c] /= s;
      }

      // Adaptive floor
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
