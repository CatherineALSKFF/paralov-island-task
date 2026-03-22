const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  var sigma = config.sigma || 0.05;
  var floor = config.FLOOR || 0.0001;
  var loessBW = config.loessBW || 0.15;
  var slopeReg = config.slopeReg || 0.3;  // regularize LOESS slope toward 0
  var wideWeight = config.wideWeight || 0.08; // safety net weight for wide model

  var targetGrowth = growthRates[String(testRound)] || 0.15;
  var settPos = new Set();
  for (var i = 0; i < settlements.length; i++) settPos.add(settlements[i].y * W + settlements[i].x);

  var nearestDist = new Uint8Array(H * W).fill(99);
  for (var i = 0; i < settlements.length; i++) {
    var s = settlements[i];
    for (var y = Math.max(0, s.y - 12); y <= Math.min(H - 1, s.y + 12); y++) {
      for (var x = Math.max(0, s.x - 12); x <= Math.min(W - 1, s.x + 12); x++) {
        var d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
        var idx = y * W + x;
        if (d < nearestDist[idx]) nearestDist[idx] = d;
      }
    }
  }

  function getEnhancedKey(y, x) {
    var coarseKey = getFeatureKey(initGrid, settPos, y, x);
    if (coarseKey === 'O' || coarseKey === 'M' || coarseKey[0] === 'S') return coarseKey;
    var nKey = coarseKey[1];
    var minDist = nearestDist[y * W + x];
    if (nKey === '0') return coarseKey + (minDist <= 5 ? 'n' : minDist <= 7 ? 'm' : 'f');
    if (nKey === '1') {
      if (minDist === 1) return coarseKey + 'a';
      if (minDist === 2) return coarseKey + 'b';
      if (minDist === 3) return coarseKey + 'd';
    }
    return coarseKey;
  }

  var allRounds = Object.keys(perRoundBuckets).map(Number).filter(function(n) { return n !== testRound; });

  // Augment with 'd' keys
  var augBuckets = {};
  for (var ri = 0; ri < allRounds.length; ri++) {
    var rn = allRounds[ri];
    var b = perRoundBuckets[String(rn)];
    if (!b) continue;
    augBuckets[rn] = {};
    for (var key in b) {
      augBuckets[rn][key] = b[key].sum.map(function(v) { return v / b[key].count; });
    }
    var baseSuffixes = [
      ['P1', 'P1a', 'P1b'], ['P1c', 'P1ca', 'P1cb'],
      ['F1', 'F1a', 'F1b'], ['F1c', 'F1ca', 'F1cb'],
    ];
    for (var si = 0; si < baseSuffixes.length; si++) {
      var base = baseSuffixes[si][0], sub1 = baseSuffixes[si][1], sub2 = baseSuffixes[si][2];
      if (!b[base]) continue;
      var baseN = b[base].count;
      var sub1N = b[sub1] ? b[sub1].count : 0;
      var sub2N = b[sub2] ? b[sub2].count : 0;
      var residN = baseN - sub1N - sub2N;
      if (residN < 5) continue;
      var dDist = [0, 0, 0, 0, 0, 0];
      for (var c = 0; c < 6; c++) {
        var total = b[base].sum[c];
        var s1v = b[sub1] ? b[sub1].sum[c] : 0;
        var s2v = b[sub2] ? b[sub2].sum[c] : 0;
        dDist[c] = Math.max(0, (total - s1v - s2v) / residN);
      }
      augBuckets[rn][base + 'd'] = dDist;
    }
  }

  // Regularized LOESS: weighted linear regression with slope shrinkage
  var linCache = {};
  function fitLoess(key) {
    if (linCache[key] !== undefined) return linCache[key];
    var points = [];
    for (var ri = 0; ri < allRounds.length; ri++) {
      var rn = allRounds[ri];
      if (!augBuckets[rn] || !augBuckets[rn][key]) continue;
      var g = growthRates[String(rn)] || 0.15;
      var diff = g - targetGrowth;
      var w = Math.exp(-diff * diff / (2 * loessBW * loessBW));
      points.push({ g: g, dist: augBuckets[rn][key], w: w });
    }
    if (points.length < 3) { linCache[key] = null; return null; }

    var result = [0, 0, 0, 0, 0, 0];
    for (var c = 0; c < 6; c++) {
      var sw = 0, swg = 0, swp = 0, swgg = 0, swgp = 0;
      for (var pi = 0; pi < points.length; pi++) {
        var pt = points[pi];
        sw += pt.w; swg += pt.w * pt.g; swp += pt.w * pt.dist[c];
        swgg += pt.w * pt.g * pt.g; swgp += pt.w * pt.g * pt.dist[c];
      }
      var denom = sw * swgg - swg * swg;
      var intercept, slope;
      if (Math.abs(denom) < 1e-12) {
        intercept = swp / sw;
        slope = 0;
      } else {
        slope = (sw * swgp - swg * swp) / denom;
        intercept = (swp - slope * swg) / sw;
      }
      // Regularize slope toward 0
      slope = slope * (1 - slopeReg);
      result[c] = Math.max(0, intercept + slope * targetGrowth);
    }
    var s = result.reduce(function(a, b) { return a + b; }, 0);
    linCache[key] = s > 0 ? result.map(function(v) { return v / s; }) : null;
    return linCache[key];
  }

  // Gaussian weighted mean models
  function buildGaussModel(sig) {
    var model = {};
    var ws = {}, tw = 0;
    for (var ri = 0; ri < allRounds.length; ri++) {
      var rn = allRounds[ri];
      var diff = (growthRates[String(rn)] || 0.15) - targetGrowth;
      ws[rn] = Math.exp(-diff * diff / (2 * sig * sig));
      tw += ws[rn];
    }
    for (var ri = 0; ri < allRounds.length; ri++) ws[allRounds[ri]] /= tw;
    for (var ri = 0; ri < allRounds.length; ri++) {
      var rn = allRounds[ri];
      if (!augBuckets[rn]) continue;
      for (var key in augBuckets[rn]) {
        if (!model[key]) model[key] = [0, 0, 0, 0, 0, 0];
        var avg = augBuckets[rn][key];
        for (var c = 0; c < 6; c++) model[key][c] += ws[rn] * avg[c];
      }
    }
    return model;
  }

  var gaussModel = buildGaussModel(sigma);
  var wideModel = buildGaussModel(0.25);

  function lookupKey(fineKey, baseKey) {
    // LOESS prediction
    var lnDist = fitLoess(fineKey) || fitLoess(baseKey);
    // Gaussian prediction
    var wmDist = gaussModel[fineKey] ? gaussModel[fineKey].slice() :
                 gaussModel[baseKey] ? gaussModel[baseKey].slice() : null;
    // Wide model (safety net)
    var wideDist = wideModel[fineKey] ? wideModel[fineKey].slice() :
                   wideModel[baseKey] ? wideModel[baseKey].slice() : null;

    // Blend
    var result = null;
    if (lnDist && wmDist) {
      result = wmDist.map(function(v, c) { return 0.15 * v + 0.85 * lnDist[c]; });
    } else if (lnDist) {
      result = lnDist;
    } else if (wmDist) {
      result = wmDist;
    }

    // Add wide model safety net
    if (result && wideDist) {
      result = result.map(function(v, c) { return (1 - wideWeight) * v + wideWeight * wideDist[c]; });
    } else if (!result && wideDist) {
      result = wideDist;
    }

    if (!result) {
      var fb = baseKey;
      while (fb.length > 1) {
        fb = fb.slice(0, -1);
        if (gaussModel[fb]) return gaussModel[fb].slice();
      }
      return [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
    }
    return result;
  }

  var pred = [];
  for (var y = 0; y < H; y++) {
    var row = [];
    for (var x = 0; x < W; x++) {
      var enhKey = getEnhancedKey(y, x);
      var coarseKey = getFeatureKey(initGrid, settPos, y, x);
      var prior = lookupKey(enhKey, coarseKey);

      // Shrink fine key toward coarse
      if (enhKey !== coarseKey && gaussModel[coarseKey]) {
        var coarse = gaussModel[coarseKey];
        for (var c = 0; c < 6; c++) prior[c] = 0.92 * prior[c] + 0.08 * coarse[c];
      }

      var entropy = 0;
      for (var c = 0; c < 6; c++) if (prior[c] > 0.001) entropy -= prior[c] * Math.log(prior[c]);
      var cellFloor = entropy > 0.5 ? floor : floor * 0.1;
      var floored = prior.map(function(v) { return Math.max(v, cellFloor); });
      var sum = floored.reduce(function(a, b) { return a + b; }, 0);
      row.push(floored.map(function(v) { return v / sum; }));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
