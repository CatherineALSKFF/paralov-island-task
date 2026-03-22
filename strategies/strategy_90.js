const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');
function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  var sigma = config.sigma || 0.08;
  var floor = config.FLOOR || 0.0001;
  var linWeight = config.linWeight || 0.90;
  var targetGrowth = growthRates[String(testRound)] || 0.15;
  var nSett = settlements.length;
  var settPos = new Set();
  for (var s of settlements) settPos.add(s.y * W + s.x);
  var nearestDist = new Uint8Array(H * W).fill(99);
  for (var s of settlements)
    for (var y = Math.max(0, s.y - 12); y <= Math.min(H - 1, s.y + 12); y++)
      for (var x = Math.max(0, s.x - 12); x <= Math.min(W - 1, s.x + 12); x++) {
        var d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
        if (d < nearestDist[y * W + x]) nearestDist[y * W + x] = d;
      }
  function getEK(y, x) {
    var ck = getFeatureKey(initGrid, settPos, y, x);
    if (ck === 'O' || ck === 'M' || ck[0] === 'S') return ck;
    var n = ck[1], md = nearestDist[y * W + x];
    if (n === '0') return ck + (md <= 5 ? 'n' : md <= 7 ? 'm' : 'f');
    if (n === '1') return ck + (md <= 1 ? 'a' : 'b');
    return ck;
  }
  var allR = Object.keys(perRoundBuckets).map(Number).filter(function(n) { return n !== testRound; });
  var rnS = {};
  for (var rn of allR) { var b = perRoundBuckets[String(rn)]; if (!b) continue; var sc = 0; for (var k in b) if (k[0] === 'S') sc += b[k].count; rnS[rn] = sc / 5; }
  // Augmented buckets
  var aug = {};
  for (var rn of allR) {
    var b = perRoundBuckets[String(rn)]; if (!b) continue;
    aug[rn] = {};
    for (var k in b) aug[rn][k] = b[k].sum.map(function(v) { return v / b[k].count; });
    for (var si = 0, bs = [['P1','P1a','P1b'],['P1c','P1ca','P1cb'],['F1','F1a','F1b'],['F1c','F1ca','F1cb']]; si < bs.length; si++) {
      var ba = bs[si][0], s1 = bs[si][1], s2 = bs[si][2];
      if (!b[ba]) continue;
      var bN = b[ba].count, s1N = b[s1]?b[s1].count:0, s2N = b[s2]?b[s2].count:0, rN = bN - s1N - s2N;
      if (rN < 5) continue;
      var dd = [];
      for (var c = 0; c < 6; c++) dd.push(Math.max(0, (b[ba].sum[c] - (b[s1]?b[s1].sum[c]:0) - (b[s2]?b[s2].sum[c]:0)) / rN));
      aug[rn][ba + 'd'] = dd;
    }
  }
  // LOESS ensemble: two bandwidths
  var loessSigmas = [0.10, 0.25];
  var loessWeights = [0.55, 0.45];
  var linCaches = [{}, {}];

  function fitLoess(key, idx) {
    var lc = linCaches[idx];
    if (lc[key] !== undefined) return lc[key];
    var ls = loessSigmas[idx];
    var pts = [];
    for (var rn of allR) {
      if (!aug[rn] || !aug[rn][key]) continue;
      var g = growthRates[String(rn)] || 0.15;
      var gd = g - targetGrowth, sd = ((rnS[rn] || 40) - nSett) / 40;
      pts.push({ g: g, dist: aug[rn][key], w: Math.exp(-gd*gd/(2*ls*ls) - sd*sd/(2*0.3*0.3)) });
    }
    if (pts.length < 3) { lc[key] = null; return null; }
    var r = [0,0,0,0,0,0];
    for (var c = 0; c < 6; c++) {
      var sw=0,swg=0,swp=0,swgg=0,swgp=0;
      for (var pt of pts) { sw+=pt.w; swg+=pt.w*pt.g; swp+=pt.w*pt.dist[c]; swgg+=pt.w*pt.g*pt.g; swgp+=pt.w*pt.g*pt.dist[c]; }
      var dn = sw*swgg - swg*swg;
      if (Math.abs(dn) < 1e-12) r[c] = swp/sw;
      else { var bC = (sw*swgp - swg*swp)/dn; r[c] = Math.max(0, (swp - bC*swg)/sw + bC*targetGrowth); }
    }
    var s = r.reduce(function(a,b){return a+b;}, 0);
    lc[key] = s > 0 ? r.map(function(v){return v/s;}) : null;
    return lc[key];
  }

  function fitLoessEnsemble(key) {
    var results = [];
    for (var i = 0; i < loessSigmas.length; i++) {
      var r = fitLoess(key, i);
      if (r) results.push({ dist: r, w: loessWeights[i] });
    }
    if (results.length === 0) return null;
    var out = [0,0,0,0,0,0]; var tw = 0;
    for (var r of results) { tw += r.w; for (var c = 0; c < 6; c++) out[c] += r.w * r.dist[c]; }
    return out.map(function(v) { return v / tw; });
  }

  // Gaussian weighted mean model (2D)
  var rw = {}, tw = 0;
  for (var rn of allR) {
    var gd = (growthRates[String(rn)]||0.15) - targetGrowth, sd = ((rnS[rn]||40)-nSett)/40;
    rw[rn] = Math.exp(-gd*gd/(2*sigma*sigma) - sd*sd/(2*0.3*0.3));
    tw += rw[rn];
  }
  for (var rn of allR) rw[rn] /= tw;
  var mm = {};
  for (var rn of allR) {
    if (!aug[rn]) continue;
    for (var k in aug[rn]) { if (!mm[k]) mm[k] = [0,0,0,0,0,0]; var av = aug[rn][k]; for (var c = 0; c < 6; c++) mm[k][c] += rw[rn] * av[c]; }
  }

  function lookupKey(fk, bk) {
    var lnFine = fitLoessEnsemble(fk), lnBase = fitLoessEnsemble(bk);
    var wmFine = mm[fk] ? mm[fk].slice() : null, wmBase = mm[bk] ? mm[bk].slice() : null;
    var lnDist;
    if (lnFine && lnBase && fk !== bk) lnDist = lnFine.map(function(v,c){return 0.65*v + 0.35*lnBase[c];});
    else lnDist = lnFine || lnBase;
    var wmDist = wmFine || wmBase;
    if (wmDist && lnDist) return wmDist.map(function(v,c){return (1-linWeight)*v + linWeight*lnDist[c];});
    if (lnDist) return lnDist;
    if (wmDist) return wmDist;
    var tk = bk[0];
    var lt = fitLoessEnsemble(tk);
    if (lt) return lt;
    if (mm[tk]) return mm[tk].slice();
    var fb = bk;
    while (fb.length > 1) { fb = fb.slice(0,-1); if (mm[fb]) return mm[fb].slice(); }
    return [1/6,1/6,1/6,1/6,1/6,1/6];
  }

  // Phase 1: raw predictions
  var rawPred = [], entMap = [];
  for (var y = 0; y < H; y++) {
    rawPred.push([]); entMap.push([]);
    for (var x = 0; x < W; x++) {
      var p = lookupKey(getEK(y,x), getFeatureKey(initGrid, settPos, y, x));
      rawPred[y][x] = p;
      var ent = 0;
      for (var c = 0; c < 6; c++) if (p[c] > 0.001) ent -= p[c] * Math.log(p[c]);
      entMap[y][x] = ent;
    }
  }
  // Phase 2: entropy-adaptive spatial smoothing + floor
  var pred = [];
  for (var y = 0; y < H; y++) {
    var row = [];
    for (var x = 0; x < W; x++) {
      var raw = rawPred[y][x];
      var ck = getFeatureKey(initGrid, settPos, y, x);
      if (ck === 'O' || ck === 'M') { row.push(raw); continue; }
      var ent = entMap[y][x];
      var spatW = ent < 0.3 ? 0.06 : ent < 0.7 ? 0.04 : ent < 1.0 ? 0.02 : 0.01;
      var nAvg = [0,0,0,0,0,0], nw = 0;
      for (var dy = -1; dy <= 1; dy++) for (var dx = -1; dx <= 1; dx++) {
        if (dy === 0 && dx === 0) continue;
        var ny = y+dy, nx = x+dx;
        if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
        var nk = getFeatureKey(initGrid, settPos, ny, nx);
        if (nk === 'O' || nk === 'M') continue;
        var dd = Math.sqrt(dy*dy+dx*dx), ww = 1/(1+dd);
        for (var c = 0; c < 6; c++) nAvg[c] += ww * rawPred[ny][nx][c];
        nw += ww;
      }
      var smoothed;
      if (nw > 0) smoothed = raw.map(function(v,c){return (1-spatW)*v + spatW*(nAvg[c]/nw);});
      else smoothed = raw;
      var cellFloor = ent > 0.5 ? floor : floor * 0.1;
      var floored = smoothed.map(function(v){return Math.max(v, cellFloor);});
      var sum = floored.reduce(function(a,b){return a+b;}, 0);
      row.push(floored.map(function(v){return v/sum;}));
    }
    pred.push(row);
  }
  return pred;
}
module.exports = { predict };
