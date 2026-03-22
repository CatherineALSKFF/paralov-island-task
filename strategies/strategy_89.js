const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');
function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  var sigma = config.sigma || 0.08;
  var floor = config.FLOOR || 0.0001;
  var linWeight = config.linWeight || 0.90;
  var loessSigma = config.loessSigma || 0.15;
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
  function getEnhancedKey(y, x) {
    var ck = getFeatureKey(initGrid, settPos, y, x);
    if (ck === 'O' || ck === 'M' || ck[0] === 'S') return ck;
    var nKey = ck[1], minDist = nearestDist[y * W + x];
    if (nKey === '0') return ck + (minDist <= 5 ? 'n' : minDist <= 7 ? 'm' : 'f');
    if (nKey === '1') return ck + (minDist <= 1 ? 'a' : 'b');
    return ck;
  }
  var allRounds = Object.keys(perRoundBuckets).map(Number).filter(function(n) { return n !== testRound; });
  var roundNSett = {};
  for (var rn of allRounds) {
    var b = perRoundBuckets[String(rn)]; if (!b) continue;
    var sc = 0; for (var key in b) if (key[0] === 'S') sc += b[key].count;
    roundNSett[rn] = sc / 5;
  }
  var augBuckets = {};
  for (var rn of allRounds) {
    var b = perRoundBuckets[String(rn)]; if (!b) continue;
    augBuckets[rn] = {};
    for (var key in b) augBuckets[rn][key] = b[key].sum.map(function(v) { return v / b[key].count; });
    for (var si = 0, bases = [['P1','P1a','P1b'],['P1c','P1ca','P1cb'],['F1','F1a','F1b'],['F1c','F1ca','F1cb']]; si < bases.length; si++) {
      var base = bases[si][0], sub1 = bases[si][1], sub2 = bases[si][2];
      if (!b[base]) continue;
      var baseN = b[base].count, sub1N = b[sub1]?b[sub1].count:0, sub2N = b[sub2]?b[sub2].count:0;
      var residN = baseN - sub1N - sub2N;
      if (residN < 5) continue;
      var dDist = [];
      for (var c = 0; c < 6; c++) dDist.push(Math.max(0, (b[base].sum[c] - (b[sub1]?b[sub1].sum[c]:0) - (b[sub2]?b[sub2].sum[c]:0)) / residN));
      augBuckets[rn][base + 'd'] = dDist;
    }
  }
  var linCache = {};
  function fitLoess(key) {
    if (linCache[key] !== undefined) return linCache[key];
    var points = [];
    for (var rn of allRounds) {
      if (!augBuckets[rn] || !augBuckets[rn][key]) continue;
      var g = growthRates[String(rn)] || 0.15;
      var gdiff = g - targetGrowth, sdiff = ((roundNSett[rn]||40) - nSett) / 40;
      points.push({ g: g, dist: augBuckets[rn][key], w: Math.exp(-gdiff*gdiff/(2*loessSigma*loessSigma) - sdiff*sdiff/(2*0.3*0.3)) });
    }
    if (points.length < 3) { linCache[key] = null; return null; }
    var result = [0,0,0,0,0,0];
    for (var c = 0; c < 6; c++) {
      var sw=0,swg=0,swp=0,swgg=0,swgp=0;
      for (var pt of points) { sw+=pt.w; swg+=pt.w*pt.g; swp+=pt.w*pt.dist[c]; swgg+=pt.w*pt.g*pt.g; swgp+=pt.w*pt.g*pt.dist[c]; }
      var dn = sw*swgg - swg*swg;
      if (Math.abs(dn) < 1e-12) result[c] = swp/sw;
      else { var bC = (sw*swgp - swg*swp)/dn; result[c] = Math.max(0, (swp - bC*swg)/sw + bC*targetGrowth); }
    }
    var s = result.reduce(function(a,b){return a+b;}, 0);
    linCache[key] = s > 0 ? result.map(function(v){return v/s;}) : null;
    return linCache[key];
  }
  var roundWeights = {}, tw = 0;
  for (var rn of allRounds) {
    var gdiff = (growthRates[String(rn)]||0.15) - targetGrowth, sdiff = ((roundNSett[rn]||40)-nSett)/40;
    roundWeights[rn] = Math.exp(-gdiff*gdiff/(2*sigma*sigma) - sdiff*sdiff/(2*0.3*0.3));
    tw += roundWeights[rn];
  }
  for (var rn of allRounds) roundWeights[rn] /= tw;
  var meanModel = {};
  for (var rn of allRounds) {
    if (!augBuckets[rn]) continue;
    for (var key in augBuckets[rn]) {
      if (!meanModel[key]) meanModel[key] = [0,0,0,0,0,0];
      var avg = augBuckets[rn][key];
      for (var c = 0; c < 6; c++) meanModel[key][c] += roundWeights[rn] * avg[c];
    }
  }
  function lookupKey(fineKey, baseKey) {
    var lnFine = fitLoess(fineKey), lnBase = fitLoess(baseKey);
    var wmFine = meanModel[fineKey]?meanModel[fineKey].slice():null, wmBase = meanModel[baseKey]?meanModel[baseKey].slice():null;
    var lnDist;
    if (lnFine && lnBase && fineKey !== baseKey) lnDist = lnFine.map(function(v,c){return 0.7*v+0.3*lnBase[c];});
    else lnDist = lnFine || lnBase;
    var wmDist = wmFine || wmBase;
    if (wmDist && lnDist) return wmDist.map(function(v,c){return (1-linWeight)*v+linWeight*lnDist[c];});
    if (lnDist) return lnDist;
    if (wmDist) return wmDist;
    var terrKey = baseKey[0];
    var lnTerr = fitLoess(terrKey);
    if (lnTerr) return lnTerr;
    if (meanModel[terrKey]) return meanModel[terrKey].slice();
    var fb = baseKey;
    while (fb.length > 1) { fb = fb.slice(0,-1); if (meanModel[fb]) return meanModel[fb].slice(); }
    return [1/6,1/6,1/6,1/6,1/6,1/6];
  }
  // Phase 1: raw predictions + entropy map
  var rawPred = [], entMap = [];
  for (var y = 0; y < H; y++) {
    rawPred.push([]); entMap.push([]);
    for (var x = 0; x < W; x++) {
      var enhKey = getEnhancedKey(y, x);
      var coarseKey = getFeatureKey(initGrid, settPos, y, x);
      var prior = lookupKey(enhKey, coarseKey);
      rawPred[y][x] = prior;
      var ent = 0;
      for (var c = 0; c < 6; c++) if (prior[c] > 0.001) ent -= prior[c] * Math.log(prior[c]);
      entMap[y][x] = ent;
    }
  }
  // Phase 2: entropy-adaptive spatial smoothing
  var pred = [];
  for (var y = 0; y < H; y++) {
    var row = [];
    for (var x = 0; x < W; x++) {
      var raw = rawPred[y][x];
      var ck = getFeatureKey(initGrid, settPos, y, x);
      if (ck === 'O' || ck === 'M') { row.push(raw); continue; }
      var ent = entMap[y][x];
      // Less smoothing for high-entropy cells (where predictions are uncertain)
      var spatW = ent < 0.3 ? 0.06 : ent < 0.7 ? 0.04 : ent < 1.0 ? 0.02 : 0.01;
      var nAvg = [0,0,0,0,0,0], nw = 0;
      for (var dy = -1; dy <= 1; dy++) for (var dx = -1; dx <= 1; dx++) {
        if (dy === 0 && dx === 0) continue;
        var ny = y+dy, nx = x+dx;
        if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
        var nk = getFeatureKey(initGrid, settPos, ny, nx);
        if (nk === 'O' || nk === 'M') continue;
        var dd = Math.sqrt(dy*dy + dx*dx), ww = 1/(1+dd);
        for (var c = 0; c < 6; c++) nAvg[c] += ww * rawPred[ny][nx][c];
        nw += ww;
      }
      var smoothed;
      if (nw > 0) smoothed = raw.map(function(v,c){return (1-spatW)*v + spatW*(nAvg[c]/nw);});
      else smoothed = raw;
      // Entropy-adaptive floor
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
