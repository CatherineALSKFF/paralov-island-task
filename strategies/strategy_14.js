const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  var floor = config.FLOOR || 0.0001;
  var sigmaNarrow = config.SN || 0.055;
  var sigmaWide = config.SW || 0.18;
  var alpha = config.ALPHA || 1.2;
  var regWeight = config.REG || 0.25;
  var targetGrowth = growthRates[String(testRound)] || 0.15;
  var trainRounds = Object.keys(perRoundBuckets).map(Number).filter(function(n) { return n !== testRound; });

  var settPos = new Set();
  for (var i = 0; i < settlements.length; i++) {
    settPos.add(settlements[i].y * W + settlements[i].x);
  }

  // Detect rounds with anomalous entropy-to-growth ratio and downweight them
  var roundEntropy = {};
  for (var ri = 0; ri < trainRounds.length; ri++) {
    var r = trainRounds[ri];
    var bk = perRoundBuckets[r];
    if (!bk) continue;
    var totalEnt = 0, nKeys = 0;
    for (var key in bk) {
      if (key === 'O' || key === 'M') continue;
      var b = bk[key];
      if (b.count === 0) continue;
      var ent = 0;
      for (var c = 0; c < 6; c++) {
        var p = b.sum[c] / b.count;
        if (p > 0.001) ent -= p * Math.log(p);
      }
      totalEnt += ent;
      nKeys++;
    }
    roundEntropy[r] = nKeys > 0 ? totalEnt / nKeys : 0;
  }

  // Rounds where entropy is much lower than expected are suspicious
  var qualityWeight = {};
  for (var ri = 0; ri < trainRounds.length; ri++) {
    var r = trainRounds[ri];
    var g = growthRates[String(r)] || 0.15;
    var expectedEnt = g * 4.0;
    var actualEnt = roundEntropy[r] || 0;
    var entDiff = actualEnt - expectedEnt;
    if (entDiff < -0.15) {
      qualityWeight[r] = Math.exp(entDiff * 3);
    } else {
      qualityWeight[r] = 1.0;
    }
  }

  function gaussWeight(r, sigma) {
    var gDiff = (growthRates[String(r)] || 0.15) - targetGrowth;
    return Math.exp(-gDiff * gDiff / (2 * sigma * sigma));
  }

  function buildModel(weightFn) {
    var weights = {};
    var wTotal = 0;
    for (var ri = 0; ri < trainRounds.length; ri++) {
      var r = trainRounds[ri];
      weights[r] = weightFn(r) * qualityWeight[r];
      wTotal += weights[r];
    }
    if (wTotal > 0) {
      for (var ri = 0; ri < trainRounds.length; ri++) weights[trainRounds[ri]] /= wTotal;
    }

    var mSum = {}, mCount = {};
    for (var ri = 0; ri < trainRounds.length; ri++) {
      var r = trainRounds[ri];
      var bk = perRoundBuckets[r];
      if (!bk) continue;
      var w = weights[r];
      if (w < 1e-15) continue;
      for (var key in bk) {
        var b = bk[key];
        if (!mSum[key]) { mSum[key] = new Float64Array(6); mCount[key] = 0; }
        for (var c = 0; c < 6; c++) mSum[key][c] += w * (b.sum[c] / b.count);
        mCount[key] += w;
      }
    }
    var probs = {}, effN = {};
    for (var key in mSum) {
      var cnt = mCount[key];
      if (cnt > 0) {
        probs[key] = new Array(6);
        for (var c = 0; c < 6; c++) probs[key][c] = mSum[key][c] / cnt;
        effN[key] = cnt;
      }
    }
    return { probs: probs, effN: effN };
  }

  var narrow = buildModel(function(r) { return gaussWeight(r, sigmaNarrow); });
  var wide = buildModel(function(r) { return gaussWeight(r, sigmaWide); });

  function getSmoothedProb(key) {
    var nP = narrow.probs[key];
    var wP = wide.probs[key];
    var eN = narrow.effN[key] || 0;
    if (!nP && !wP) return null;
    if (!nP) return wP;
    if (!wP) return nP;
    var result = new Array(6);
    var denom = eN + alpha;
    for (var c = 0; c < 6; c++) {
      result[c] = (eN * nP[c] + alpha * wP[c]) / denom;
    }
    return result;
  }

  var pred = [];
  for (var y = 0; y < H; y++) {
    var row = [];
    for (var x = 0; x < W; x++) {
      var key = getFeatureKey(initGrid, settPos, y, x);
      var ck = key.length > 1 ? key.slice(0, -1) : key;
      var tk = key[0];
      var fine = getSmoothedProb(key);
      var coarse = getSmoothedProb(ck);
      var broad = getSmoothedProb(tk);

      var prior;
      if (fine && coarse) {
        var fineN = narrow.effN[key] || 0;
        var adaptReg = regWeight * Math.max(0.3, 1 - Math.min(fineN / 8, 1) * 0.7);
        prior = new Array(6);
        for (var c = 0; c < 6; c++) prior[c] = (1 - adaptReg) * fine[c] + adaptReg * coarse[c];
      } else if (fine) {
        prior = fine;
      } else if (coarse && broad) {
        prior = new Array(6);
        for (var c = 0; c < 6; c++) prior[c] = 0.8 * coarse[c] + 0.2 * broad[c];
      } else {
        prior = coarse || broad || [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      var ent = 0;
      for (var c = 0; c < 6; c++) {
        if (prior[c] > 1e-10) ent -= prior[c] * Math.log(prior[c]);
      }
      var aFloor = floor * (0.1 + 0.9 * ent / Math.log(6));

      var floored = new Array(6);
      var sum = 0;
      for (var c = 0; c < 6; c++) {
        floored[c] = Math.max(prior[c], aFloor);
        sum += floored[c];
      }
      var result = new Array(6);
      for (var c = 0; c < 6; c++) result[c] = floored[c] / sum;
      row.push(result);
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
