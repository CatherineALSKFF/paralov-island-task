const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  var sigma = config.sigma || 0.05;
  var floor = config.FLOOR || 0.0001;
  var linWeight = config.linWeight || 0.85;
  var loessSigma = config.loessSigma || 0.15;

  var targetGrowth = growthRates[String(testRound)] || 0.15;
  var testSettCount = settlements.length;
  var settPos = new Set();
  for (var s of settlements) settPos.add(s.y * W + s.x);

  // Infer training round settlement counts from bucket data
  var roundSettCounts = {};
  for (var rn of Object.keys(perRoundBuckets)) {
    var b = perRoundBuckets[rn];
    var sc = 0;
    for (var key in b) { if (key[0] === 'S') sc += b[key].count; }
    roundSettCounts[parseInt(rn)] = sc;
  }

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

  // Expected settlement count for this growth level
  var expectedSC = 0, scw = 0;
  for (var rn of allRounds) {
    var gDiff = Math.abs((growthRates[String(rn)] || 0.15) - targetGrowth);
    var w = Math.exp(-gDiff * gDiff / (2 * sigma * sigma));
    expectedSC += w * (roundSettCounts[rn] || 35);
    scw += w;
  }
  expectedSC = scw > 0 ? expectedSC / scw : 35;
  var settRatio = testSettCount / expectedSC;

  // Build augmented buckets + 'd' keys
  var augBuckets = {};
  for (var rn of allRounds) {
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

  // LOESS regression (growth-only, same as strategy_76)
  var linCache = {};
  function fitLoess(key) {
    if (linCache[key] !== undefined) return linCache[key];
    var points = [];
    for (var rn of allRounds) {
      if (augBuckets[rn] && augBuckets[rn][key]) {
        var g = growthRates[String(rn)] || 0.15;
        var diff = g - targetGrowth;
        var w = Math.exp(-diff * diff / (2 * loessSigma * loessSigma));
        points.push({ g: g, sc: roundSettCounts[rn] || 35, dist: augBuckets[rn][key], w: w });
      }
    }
    if (points.length < 3) { linCache[key] = null; return null; }
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
    linCache[key] = s > 0 ? result.map(function(v) { return v / s; }) : null;
    return linCache[key];
  }

  // Gaussian weighted mean model (growth-only)
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
      if (!meanModel[key]) meanModel[key] = [0, 0, 0, 0, 0, 0];
      var avg = augBuckets[rn][key];
      for (var c = 0; c < 6; c++) meanModel[key][c] += roundWeights[rn] * avg[c];
    }
  }

  // Compute settlement-count sensitivity per key:
  // For each key, correlate (settCount - expected) with (sett_prob - mean_sett_prob)
  var settSensitivity = {};
  for (var key of Object.keys(meanModel)) {
    var meanSP = meanModel[key][1]; // settlement class = 1
    var meanPP = meanModel[key][2]; // port class = 2
    var points = [];
    for (var rn of allRounds) {
      if (!augBuckets[rn] || !augBuckets[rn][key]) continue;
      var sc = roundSettCounts[rn] || 35;
      var sp = augBuckets[rn][key][1]; // settlement prob in this round
      var pp = augBuckets[rn][key][2]; // port prob
      var w = roundWeights[rn];
      points.push({ sc, sp, pp, w });
    }
    if (points.length < 4) continue;
    
    // Weighted regression: sp = a + b * sc
    var sw = 0, swsc = 0, swsp = 0, swscsc = 0, swscsp = 0;
    for (var pt of points) {
      sw += pt.w; swsc += pt.w * pt.sc; swsp += pt.w * pt.sp;
      swscsc += pt.w * pt.sc * pt.sc; swscsp += pt.w * pt.sc * pt.sp;
    }
    var den = sw * swscsc - swsc * swsc;
    if (Math.abs(den) > 1e-12) {
      var slope = (sw * swscsp - swsc * swsp) / den;
      settSensitivity[key] = { settSlope: slope };
      
      // Also for ports
      var swpp = 0, swscpp = 0;
      for (var pt of points) { swpp += pt.w * pt.pp; swscpp += pt.w * pt.sc * pt.pp; }
      settSensitivity[key].portSlope = (sw * swscpp - swsc * swpp) / den;
    }
  }

  function lookupKey(fineKey, baseKey) {
    var wmDist = meanModel[fineKey] ? meanModel[fineKey].slice() :
                 meanModel[baseKey] ? meanModel[baseKey].slice() : null;
    var lnDist = fitLoess(fineKey) || fitLoess(baseKey);

    var base;
    if (wmDist && lnDist) {
      base = wmDist.map(function(v, c) { return (1 - linWeight) * v + linWeight * lnDist[c]; });
    } else if (lnDist) {
      base = lnDist;
    } else if (wmDist) {
      base = wmDist;
    } else {
      var fb = baseKey;
      while (fb.length > 1) { fb = fb.slice(0, -1); if (meanModel[fb]) return meanModel[fb].slice(); }
      return [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
    }

    // Apply settlement-count correction
    var sens = settSensitivity[fineKey] || settSensitivity[baseKey];
    if (sens) {
      var scDelta = testSettCount - expectedSC;
      var settAdj = sens.settSlope * scDelta;
      var portAdj = sens.portSlope * scDelta;
      // Apply correction with damping
      var dampFactor = 0.5;
      base[1] = Math.max(0, base[1] + settAdj * dampFactor);
      base[2] = Math.max(0, base[2] + portAdj * dampFactor);
      // Take mass from plains (class 0)
      var totalAdj = (settAdj + portAdj) * dampFactor;
      base[0] = Math.max(0.01, base[0] - totalAdj * 0.6);
      base[4] = Math.max(0, base[4] - totalAdj * 0.4);
      // Renormalize
      var s = base.reduce(function(a, b) { return a + b; }, 0);
      if (s > 0) base = base.map(function(v) { return v / s; });
    }

    return base;
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
