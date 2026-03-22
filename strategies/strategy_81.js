const { H, W, getFeatureKey } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  var sigma = config.sigma != null ? config.sigma : 0.08;
  var floor = config.FLOOR != null ? config.FLOOR : 0.0001;
  var linWeight = config.linWeight != null ? config.linWeight : 0.90;
  var loessSigma = config.loessSigma != null ? config.loessSigma : 0.15;
  var spatialW = config.spatialW != null ? config.spatialW : 0.045;
  var shrinkW = config.shrinkW != null ? config.shrinkW : 0;

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

  function getEnhancedKey(y, x) {
    var coarseKey = getFeatureKey(initGrid, settPos, y, x);
    if (coarseKey === 'O' || coarseKey === 'M' || coarseKey[0] === 'S') return coarseKey;
    var nKey = coarseKey[1];
    var minDist = nearestDist[y * W + x];
    if (nKey === '0') return coarseKey + (minDist <= 5 ? 'n' : minDist <= 7 ? 'm' : 'f');
    if (nKey === '1') return coarseKey + (minDist <= 1 ? 'a' : 'b');
    return coarseKey;
  }

  var allRounds = Object.keys(perRoundBuckets).map(Number).filter(function(n) { return n !== testRound; });

  var augBuckets = {};
  for (var rn of allRounds) {
    var b = perRoundBuckets[String(rn)];
    if (!b) continue;
    augBuckets[rn] = {};
    for (var key in b) augBuckets[rn][key] = b[key].sum.map(function(v) { return v / b[key].count; });
    var baseSuffixes = [['P1','P1a','P1b'],['P1c','P1ca','P1cb'],['F1','F1a','F1b'],['F1c','F1ca','F1cb']];
    for (var si = 0; si < baseSuffixes.length; si++) {
      var base = baseSuffixes[si][0], sub1 = baseSuffixes[si][1], sub2 = baseSuffixes[si][2];
      if (!b[base]) continue;
      var baseN = b[base].count, sub1N = b[sub1] ? b[sub1].count : 0, sub2N = b[sub2] ? b[sub2].count : 0;
      var residN = baseN - sub1N - sub2N;
      if (residN < 5) continue;
      var dDist = [];
      for (var c = 0; c < 6; c++) {
        var total = b[base].sum[c], s1v = b[sub1] ? b[sub1].sum[c] : 0, s2v = b[sub2] ? b[sub2].sum[c] : 0;
        dDist.push(Math.max(0, (total - s1v - s2v) / residN));
      }
      augBuckets[rn][base + 'd'] = dDist;
    }
  }

  // LOESS
  var loessCache = {};
  function fitLoess(key) {
    if (loessCache[key] !== undefined) return loessCache[key];
    var points = [];
    for (var rn of allRounds) {
      if (augBuckets[rn] && augBuckets[rn][key]) {
        var g = growthRates[String(rn)] || 0.15;
        var gd = g - targetGrowth;
        var w = Math.exp(-gd * gd / (2 * loessSigma * loessSigma));
        points.push({ g: g, dist: augBuckets[rn][key], w: w });
      }
    }
    if (points.length < 3) { loessCache[key] = null; return null; }
    var result = [0, 0, 0, 0, 0, 0];
    for (var c = 0; c < 6; c++) {
      var sw = 0, swg = 0, swp = 0, swgg = 0, swgp = 0;
      for (var pt of points) {
        sw += pt.w; swg += pt.w * pt.g; swp += pt.w * pt.dist[c];
        swgg += pt.w * pt.g * pt.g; swgp += pt.w * pt.g * pt.dist[c];
      }
      var denom = sw * swgg - swg * swg;
      if (Math.abs(denom) < 1e-12) result[c] = swp / sw;
      else {
        var bCoef = (sw * swgp - swg * swp) / denom;
        var a = (swp - bCoef * swg) / sw;
        result[c] = Math.max(0, a + bCoef * targetGrowth);
      }
    }
    var s = result.reduce(function(a, b) { return a + b; }, 0);
    loessCache[key] = s > 0 ? result.map(function(v) { return v / s; }) : null;
    return loessCache[key];
  }

  // Gaussian-weighted mean model
  var roundWeights = {};
  var tw = 0;
  for (var rn of allRounds) {
    var gd = (growthRates[String(rn)] || 0.15) - targetGrowth;
    roundWeights[rn] = Math.exp(-gd * gd / (2 * sigma * sigma));
    tw += roundWeights[rn];
  }
  for (var rn of allRounds) roundWeights[rn] /= tw;

  var meanModel = {};
  for (var rn of allRounds) {
    if (!augBuckets[rn]) continue;
    for (var key in augBuckets[rn]) {
      if (!meanModel[key]) meanModel[key] = [0, 0, 0, 0, 0, 0];
      var avg = augBuckets[rn][key];
      for (var c = 0; c < 6; c++) meanModel[key][c] += roundWeights[rn] * avg[c];
    }
  }

  // Lookup: LOESS + Gaussian blend, with mild shrinkage to base key
  function lookupKey(fineKey, baseKey) {
    var lnDist = fitLoess(fineKey) || fitLoess(baseKey);
    var wmDist = meanModel[fineKey] ? meanModel[fineKey].slice() :
                 meanModel[baseKey] ? meanModel[baseKey].slice() : null;

    var pred;
    if (wmDist && lnDist) {
      pred = wmDist.map(function(v, c) { return (1 - linWeight) * v + linWeight * lnDist[c]; });
    } else if (lnDist) {
      pred = lnDist;
    } else if (wmDist) {
      pred = wmDist;
    } else {
      var fb = baseKey;
      while (fb.length > 1) { fb = fb.slice(0, -1); if (meanModel[fb]) return meanModel[fb].slice(); }
      return [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
    }

    // Mild shrinkage: blend fine prediction toward coarser base key prediction
    if (fineKey !== baseKey && shrinkW > 0) {
      var baseLn = fitLoess(baseKey);
      var baseWm = meanModel[baseKey] ? meanModel[baseKey].slice() : null;
      var basePred = null;
      if (baseWm && baseLn) {
        basePred = baseWm.map(function(v, c) { return (1 - linWeight) * v + linWeight * baseLn[c]; });
      } else if (baseLn) {
        basePred = baseLn;
      } else if (baseWm) {
        basePred = baseWm;
      }
      if (basePred) {
        pred = pred.map(function(v, c) { return (1 - shrinkW) * v + shrinkW * basePred[c]; });
      }
    }

    return pred;
  }

  // Phase 1: raw predictions
  var rawPred = [];
  var isStatic = [];
  for (var y = 0; y < H; y++) {
    rawPred.push(new Array(W));
    isStatic.push(new Array(W));
    for (var x = 0; x < W; x++) {
      var coarseKey = getFeatureKey(initGrid, settPos, y, x);
      if (coarseKey === 'O' || coarseKey === 'M') {
        isStatic[y][x] = true;
        rawPred[y][x] = coarseKey === 'O' ? [1,0,0,0,0,0] : [0,0,0,0,0,1];
        continue;
      }
      isStatic[y][x] = false;
      rawPred[y][x] = lookupKey(getEnhancedKey(y, x), coarseKey);
    }
  }

  // Phase 2: spatial smoothing + floor
  var pred = [];
  for (var y = 0; y < H; y++) {
    var row = [];
    for (var x = 0; x < W; x++) {
      if (isStatic[y][x]) { row.push(rawPred[y][x]); continue; }
      var raw = rawPred[y][x];

      var smoothed;
      if (spatialW > 0) {
        var nAvg = [0,0,0,0,0,0], nw = 0;
        for (var dy = -1; dy <= 1; dy++) for (var dx = -1; dx <= 1; dx++) {
          if (dy === 0 && dx === 0) continue;
          var ny = y + dy, nx = x + dx;
          if (ny < 0 || ny >= H || nx < 0 || nx >= W || isStatic[ny][nx]) continue;
          for (var c = 0; c < 6; c++) nAvg[c] += rawPred[ny][nx][c];
          nw++;
        }
        if (nw > 0) {
          smoothed = raw.map(function(v, c) { return (1 - spatialW) * v + spatialW * (nAvg[c] / nw); });
        } else {
          smoothed = raw;
        }
      } else {
        smoothed = raw;
      }

      var entropy = 0;
      for (var c = 0; c < 6; c++) if (smoothed[c] > 0.001) entropy -= smoothed[c] * Math.log(smoothed[c]);
      var cellFloor = entropy > 0.5 ? floor : floor * 0.1;

      var floored = smoothed.map(function(v) { return Math.max(v, cellFloor); });
      var sum = floored.reduce(function(a, b) { return a + b; }, 0);
      row.push(floored.map(function(v) { return v / sum; }));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
