const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  var floor = config.FLOOR || 0.0001;
  var targetGrowth = growthRates[String(testRound)] || 0.15;
  var trainRounds = Object.keys(perRoundBuckets).map(Number).filter(function(n) { return n !== testRound; });

  var settPos = new Set();
  for (var i = 0; i < settlements.length; i++) {
    settPos.add(settlements[i].y * W + settlements[i].x);
  }

  // Gaussian weight
  function gaussWeight(r, sigma) {
    var gDiff = (growthRates[String(r)] || 0.15) - targetGrowth;
    return Math.exp(-gDiff * gDiff / (2 * sigma * sigma));
  }

  // Build per-round normalized model
  function buildModel(sigma) {
    var weights = {};
    var wTotal = 0;
    for (var ri = 0; ri < trainRounds.length; ri++) {
      var r = trainRounds[ri];
      weights[r] = gaussWeight(r, sigma);
      wTotal += weights[r];
    }
    if (wTotal > 0) for (var ri = 0; ri < trainRounds.length; ri++) weights[trainRounds[ri]] /= wTotal;

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

  // Ensemble of 5 configurations: narrow + wide + Dirichlet smoothing
  var configs = [
    { sn: 0.035, sw: 0.14, alpha: 0.8, reg: 0.20 },
    { sn: 0.045, sw: 0.16, alpha: 1.0, reg: 0.20 },
    { sn: 0.055, sw: 0.18, alpha: 1.2, reg: 0.18 },
    { sn: 0.065, sw: 0.20, alpha: 1.5, reg: 0.15 },
    { sn: 0.080, sw: 0.25, alpha: 2.0, reg: 0.12 },
  ];

  var narrowModels = [];
  var wideModels = [];
  for (var ci = 0; ci < configs.length; ci++) {
    narrowModels.push(buildModel(configs[ci].sn));
    wideModels.push(buildModel(configs[ci].sw));
  }

  function getSmoothedProb(key, ci) {
    var narrow = narrowModels[ci];
    var wide = wideModels[ci];
    var alpha = configs[ci].alpha;
    var nP = narrow.probs[key];
    var wP = wide.probs[key];
    var eN = narrow.effN[key] || 0;
    if (!nP && !wP) return null;
    if (!nP) return wP;
    if (!wP) return nP;
    var result = new Array(6);
    var denom = eN + alpha;
    for (var c = 0; c < 6; c++) result[c] = (eN * nP[c] + alpha * wP[c]) / denom;
    return result;
  }

  var pred = [];
  for (var y = 0; y < H; y++) {
    var row = [];
    for (var x = 0; x < W; x++) {
      var key = getFeatureKey(initGrid, settPos, y, x);
      var ck = key.length > 1 ? key.slice(0, -1) : key;
      var tk = key[0];

      var ensemble = new Float64Array(6);
      var ensN = 0;

      for (var ci = 0; ci < configs.length; ci++) {
        var reg = configs[ci].reg;
        var fine = getSmoothedProb(key, ci);
        var coarse = getSmoothedProb(ck, ci);

        var cellPred;
        if (fine && coarse) {
          var fineN = narrowModels[ci].effN[key] || 0;
          var adaptReg = reg * Math.max(0.3, 1 - Math.min(fineN / 8, 1) * 0.7);
          cellPred = new Array(6);
          for (var c = 0; c < 6; c++) cellPred[c] = (1 - adaptReg) * fine[c] + adaptReg * coarse[c];
        } else {
          cellPred = fine || coarse || getSmoothedProb(tk, ci) || null;
        }

        if (cellPred) {
          for (var c = 0; c < 6; c++) ensemble[c] += cellPred[c];
          ensN++;
        }
      }

      var prior;
      if (ensN > 0) {
        prior = new Array(6);
        for (var c = 0; c < 6; c++) prior[c] = ensemble[c] / ensN;
      } else {
        prior = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];
      }

      // Adaptive floor
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
