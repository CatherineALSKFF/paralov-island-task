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

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const bw = config.BW || 0.042;
  const minW = config.MIN_W || 0.001;
  const temp = config.TEMP || 1.10;
  const countBlend = config.COUNT_BLEND || 0.0;  // 0=per-round, 1=count-weighted

  const targetGrowth = growthRates[String(testRound)] || 0.15;

  // Build weights
  const weights = {};
  for (const [rn, rate] of Object.entries(growthRates)) {
    if (parseInt(rn) === testRound) continue;
    if (!perRoundBuckets[rn]) continue;
    const dist = Math.abs(rate - targetGrowth);
    weights[rn] = Math.max(minW, Math.exp(-0.5 * (dist / bw) ** 2));
  }

  // Per-round normalized model (each round gets equal vote weighted by growth similarity)
  const prModel = {};
  for (const [rn, w] of Object.entries(weights)) {
    const b = perRoundBuckets[rn];
    if (!b) continue;
    for (const [key, v] of Object.entries(b)) {
      if (!prModel[key]) prModel[key] = { d: [0,0,0,0,0,0], tw: 0 };
      const rd = v.sum.map(s => s / v.count);
      for (let c = 0; c < 6; c++) prModel[key].d[c] += w * rd[c];
      prModel[key].tw += w;
    }
  }

  // Count-weighted model (cells contribute proportionally to count)
  const cwModel = {};
  for (const [rn, w] of Object.entries(weights)) {
    const b = perRoundBuckets[rn];
    if (!b) continue;
    for (const [key, v] of Object.entries(b)) {
      if (!cwModel[key]) cwModel[key] = { s: [0,0,0,0,0,0], n: 0 };
      for (let c = 0; c < 6; c++) cwModel[key].s[c] += w * v.sum[c];
      cwModel[key].n += w * v.count;
    }
  }

  // Merge into final model
  const modelDist = {};
  for (const k in prModel) {
    const pr = prModel[k].d.map(s => s / prModel[k].tw);
    if (countBlend > 0.01 && cwModel[k] && cwModel[k].n > 0) {
      const cw = cwModel[k].s.map(s => s / cwModel[k].n);
      modelDist[k] = pr.map((p, c) => (1 - countBlend) * p + countBlend * cw[c]);
    } else {
      modelDist[k] = pr;
    }
  }

  // Per-key uncertainty (for potential variance-based adjustments)
  const keyVariance = {};
  for (const key in prModel) {
    const mean = modelDist[key];
    let totalVar = 0, tw = 0;
    for (const [rn, w] of Object.entries(weights)) {
      const b = perRoundBuckets[rn];
      if (!b || !b[key]) continue;
      const rd = b[key].sum.map(s => s / b[key].count);
      for (let c = 0; c < 6; c++) {
        const diff = rd[c] - mean[c];
        totalVar += w * diff * diff;
      }
      tw += w;
    }
    keyVariance[key] = tw > 0 ? totalVar / tw : 0;
  }

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

      let p = null;
      if (rich !== std) p = modelDist[rich];
      if (!p) p = modelDist[std];
      if (!p) {
        const fb = std.slice(0, -1);
        p = modelDist[fb];
      }
      if (!p) { row.push([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]); continue; }

      let prior = [...p];

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
