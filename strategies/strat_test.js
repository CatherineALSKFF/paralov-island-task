const { H, W, getFeatureKey } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  var sigma = config.sigma || 0.05;
  var floor = config.FLOOR || 0.0001;
  var targetGrowth = growthRates[String(testRound)] || 0.15;
  var settPos = new Set();
  for (var s of settlements) settPos.add(s.y * W + s.x);

  var nearestDist = new Uint8Array(H * W).fill(99);
  for (var s of settlements) {
    for (var y = Math.max(0, s.y - 12); y <= Math.min(H - 1, s.y + 12); y++)
      for (var x = Math.max(0, s.x - 12); x <= Math.min(W - 1, s.x + 12); x++) {
        var d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
        if (d < nearestDist[y * W + x]) nearestDist[y * W + x] = d;
      }
  }
  function getEK(y, x) {
    var base = getFeatureKey(initGrid, settPos, y, x);
    if (base === 'O' || base === 'M' || base[0] === 'S') return base;
    var b = base[1], md = nearestDist[y * W + x];
    if (b === '0') return base + (md <= 5 ? 'n' : md <= 7 ? 'm' : 'f');
    if (b === '1') { if (md === 1) return base + 'a'; if (md === 2) return base + 'b'; if (md === 3) return base + 'd'; }
    return base;
  }

  var allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);
  var rw = {}, tw = 0;
  for (var rn of allRounds) {
    var d = (growthRates[String(rn)] || 0.15) - targetGrowth;
    rw[rn] = Math.exp(-d * d / (2 * sigma * sigma)); tw += rw[rn];
  }
  for (var rn of allRounds) rw[rn] /= tw;

  // Build augBuckets + meanModel (same as strategy_76 with linWeight=0)
  var augBuckets = {};
  for (var rn of allRounds) {
    var b = perRoundBuckets[String(rn)];
    if (!b) continue;
    augBuckets[rn] = {};
    for (var key in b) augBuckets[rn][key] = b[key].sum.map(v => v / b[key].count);
    for (var [base, s1, s2] of [['P1','P1a','P1b'],['P1c','P1ca','P1cb'],['F1','F1a','F1b'],['F1c','F1ca','F1cb']]) {
      if (!b[base]) continue;
      var bN = b[base].count, s1N = b[s1]?b[s1].count:0, s2N = b[s2]?b[s2].count:0;
      var rN = bN - s1N - s2N;
      if (rN < 5) continue;
      var dd = [];
      for (var c = 0; c < 6; c++) dd.push(Math.max(0, (b[base].sum[c] - (b[s1]?b[s1].sum[c]:0) - (b[s2]?b[s2].sum[c]:0)) / rN));
      augBuckets[rn][base + 'd'] = dd;
    }
  }

  var meanModel = {};
  for (var rn of allRounds) {
    if (!augBuckets[rn]) continue;
    for (var key in augBuckets[rn]) {
      if (!meanModel[key]) meanModel[key] = [0,0,0,0,0,0];
      var avg = augBuckets[rn][key];
      for (var c = 0; c < 6; c++) meanModel[key][c] += rw[rn] * avg[c];
    }
  }

  function lookup(rk, bk) {
    if (meanModel[rk]) return meanModel[rk].slice();
    if (meanModel[bk]) return meanModel[bk].slice();
    var fb = bk;
    while (fb.length > 1) { fb = fb.slice(0,-1); if (meanModel[fb]) return meanModel[fb].slice(); }
    return [1/6,1/6,1/6,1/6,1/6,1/6];
  }

  // Phase 1: raw predictions
  var rawPred = [], isStatic = [];
  for (var y = 0; y < H; y++) {
    rawPred.push(new Array(W)); isStatic.push(new Array(W));
    for (var x = 0; x < W; x++) {
      var bk = getFeatureKey(initGrid, settPos, y, x);
      if (bk === 'O' || bk === 'M') {
        isStatic[y][x] = true;
        rawPred[y][x] = bk === 'O' ? [1,0,0,0,0,0] : [0,0,0,0,0,1];
        continue;
      }
      isStatic[y][x] = false;
      rawPred[y][x] = lookup(getEK(y, x), bk);
    }
  }

  // Phase 2: spatial smoothing + floor
  var spatialW = config.spatialW || 0.03;
  var pred = [];
  for (var y = 0; y < H; y++) {
    var row = [];
    for (var x = 0; x < W; x++) {
      if (isStatic[y][x]) { row.push(rawPred[y][x]); continue; }
      var raw = rawPred[y][x];

      var nAvg = [0,0,0,0,0,0], nw = 0;
      for (var dy = -1; dy <= 1; dy++) for (var dx = -1; dx <= 1; dx++) {
        if (dy === 0 && dx === 0) continue;
        var ny = y+dy, nx = x+dx;
        if (ny < 0 || ny >= H || nx < 0 || nx >= W || isStatic[ny][nx]) continue;
        for (var c = 0; c < 6; c++) nAvg[c] += rawPred[ny][nx][c];
        nw++;
      }

      var smoothed;
      if (nw > 0) {
        smoothed = raw.map((v, c) => (1-spatialW)*v + spatialW*(nAvg[c]/nw));
      } else {
        smoothed = raw;
      }

      var entropy = 0;
      for (var c = 0; c < 6; c++) if (smoothed[c] > 1e-8) entropy -= smoothed[c] * Math.log(smoothed[c]);
      var cellFloor = entropy > 0.5 ? floor : floor * 0.1;
      var floored = smoothed.map(v => Math.max(v, cellFloor));
      var sum = floored.reduce((a,b) => a+b, 0);
      row.push(floored.map(v => v/sum));
    }
    pred.push(row);
  }
  return pred;
}
module.exports = { predict };
