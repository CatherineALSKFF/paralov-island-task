const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  var sigma = config.sigma || 0.05;
  var floor = config.FLOOR || 0.0001;
  var loessSigma = config.loessSigma || 0.15;

  var targetGrowth = growthRates[String(testRound)] || 0.15;
  var settPos = new Set();
  for (var s of settlements) settPos.add(s.y * W + s.x);

  var nearestDist = new Uint8Array(H * W).fill(99);
  for (var s of settlements) {
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

  // Build augmented buckets
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
      var baseN = b[base].count, sub1N = b[sub1]?b[sub1].count:0, sub2N = b[sub2]?b[sub2].count:0;
      var residN = baseN - sub1N - sub2N;
      if (residN < 5) continue;
      var dDist = [];
      for (var c = 0; c < 6; c++) {
        var total = b[base].sum[c], s1v = b[sub1]?b[sub1].sum[c]:0, s2v = b[sub2]?b[sub2].sum[c]:0;
        dDist.push(Math.max(0, (total - s1v - s2v) / residN));
      }
      augBuckets[rn][base + 'd'] = dDist;
    }
  }

  // LOESS with fit quality (R²) estimation
  var linCache = {};
  function fitLoessAdaptive(key) {
    if (linCache[key] !== undefined) return linCache[key];
    var points = [];
    for (var rn of allRounds) {
      if (augBuckets[rn] && augBuckets[rn][key]) {
        var g = growthRates[String(rn)] || 0.15;
        var diff = g - targetGrowth;
        var w = Math.exp(-diff * diff / (2 * loessSigma * loessSigma));
        points.push({ g: g, dist: augBuckets[rn][key], w: w });
      }
    }
    if (points.length < 3) { linCache[key] = null; return null; }

    // Fit LOESS for each class
    var linResult = [0,0,0,0,0,0];
    var meanResult = [0,0,0,0,0,0];
    var totalR2 = 0;
    
    for (var c = 0; c < 6; c++) {
      var sw = 0, swg = 0, swp = 0, swgg = 0, swgp = 0, swpp = 0;
      for (var pt of points) {
        sw += pt.w; swg += pt.w * pt.g; swp += pt.w * pt.dist[c];
        swgg += pt.w * pt.g * pt.g; swgp += pt.w * pt.g * pt.dist[c];
        swpp += pt.w * pt.dist[c] * pt.dist[c];
      }
      // Weighted mean
      meanResult[c] = swp / sw;
      
      var denom = sw * swgg - swg * swg;
      if (Math.abs(denom) < 1e-12) {
        linResult[c] = swp / sw;
      } else {
        var bCoef = (sw * swgp - swg * swp) / denom;
        var a = (swp - bCoef * swg) / sw;
        linResult[c] = Math.max(0, a + bCoef * targetGrowth);
      }
      
      // R² = 1 - SS_res / SS_tot (weighted)
      var ssTot = swpp - swp * swp / sw;
      if (ssTot > 1e-10) {
        var ssRes = 0;
        for (var pt of points) {
          var aPred = linResult[c]; // at this growth
          // Actually need prediction at pt.g, not targetGrowth
          if (Math.abs(denom) >= 1e-12) {
            var bCoef2 = (sw * swgp - swg * swp) / denom;
            var a2 = (swp - bCoef2 * swg) / sw;
            aPred = a2 + bCoef2 * pt.g;
          }
          var resid = pt.dist[c] - aPred;
          ssRes += pt.w * resid * resid;
        }
        var r2 = Math.max(0, 1 - ssRes / ssTot);
        totalR2 += r2;
      }
    }

    var avgR2 = totalR2 / 6;
    
    // Blend LOESS and mean based on R²
    // High R²: trust LOESS. Low R²: trust mean (more conservative)
    var loessW = 0.5 + 0.5 * avgR2;  // range: 0.5 to 1.0
    var result = [];
    for (var c = 0; c < 6; c++) {
      result.push(loessW * linResult[c] + (1 - loessW) * meanResult[c]);
    }
    
    var s = result.reduce(function(a, b) { return a + b; }, 0);
    linCache[key] = s > 0 ? result.map(function(v) { return Math.max(0, v) / s; }) : null;
    return linCache[key];
  }

  // Gaussian weighted mean model
  var roundWeights = {};
  var tw = 0;
  for (var rn of allRounds) {
    var diff = Math.abs((growthRates[String(rn)] || 0.15) - targetGrowth);
    roundWeights[rn] = Math.exp(-diff * diff / (2 * sigma * sigma));
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
    var lnDist = fitLoessAdaptive(fineKey) || fitLoessAdaptive(baseKey);
    var wmDist = meanModel[fineKey] ? meanModel[fineKey].slice() :
                 meanModel[baseKey] ? meanModel[baseKey].slice() : null;
    
    if (wmDist && lnDist) return wmDist.map(function(v, c) { return 0.15 * v + 0.85 * lnDist[c]; });
    if (lnDist) return lnDist;
    if (wmDist) return wmDist;
    var fb = baseKey;
    while (fb.length > 1) { fb = fb.slice(0, -1); if (meanModel[fb]) return meanModel[fb].slice(); }
    return [1/6,1/6,1/6,1/6,1/6,1/6];
  }

  var pred = [];
  for (var y = 0; y < H; y++) {
    var row = [];
    for (var x = 0; x < W; x++) {
      var enhKey = getEnhancedKey(y, x);
      var coarseKey = getFeatureKey(initGrid, settPos, y, x);
      var prior = lookupKey(enhKey, coarseKey);
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
