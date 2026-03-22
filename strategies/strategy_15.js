const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  var K = config.K || 4;
  var floor = config.FLOOR || 0.0001;
  var regWeight = config.REG || 0.25;
  var blendAll = config.BLEND || 0.15;
  var targetGrowth = growthRates[String(testRound)] || 0.15;
  var candidates = {};
  for (var rn in growthRates) {
    if (parseInt(rn) !== testRound) candidates[rn] = growthRates[rn];
  }
  var closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  var allRounds = Object.keys(perRoundBuckets).map(Number).filter(function(n) { return n !== testRound; });

  var settPos = new Set();
  for (var i = 0; i < settlements.length; i++) {
    settPos.add(settlements[i].y * W + settlements[i].x);
  }

  // Per-round normalized merge for K closest rounds
  function perRoundMerge(rounds) {
    var model = {};
    var nRounds = rounds.length;
    for (var ri = 0; ri < rounds.length; ri++) {
      var r = rounds[ri];
      var bk = perRoundBuckets[r] || perRoundBuckets[String(r)];
      if (!bk) continue;
      for (var key in bk) {
        var b = bk[key];
        if (b.count === 0) continue;
        if (!model[key]) model[key] = { sum: new Float64Array(6), n: 0 };
        for (var c = 0; c < 6; c++) model[key].sum[c] += b.sum[c] / b.count;
        model[key].n++;
      }
    }
    var probs = {};
    for (var key in model) {
      var m = model[key];
      if (m.n > 0) {
        probs[key] = new Array(6);
        for (var c = 0; c < 6; c++) probs[key][c] = m.sum[c] / m.n;
      }
    }
    return probs;
  }

  var kModel = perRoundMerge(closestRounds);
  var allModel = perRoundMerge(allRounds);

  var pred = [];
  for (var y = 0; y < H; y++) {
    var row = [];
    for (var x = 0; x < W; x++) {
      var key = getFeatureKey(initGrid, settPos, y, x);
      var ck = key.length > 1 ? key.slice(0, -1) : key;
      var tk = key[0];

      // Try K-model first, then all-model
      var fine = kModel[key] || null;
      var fineAll = allModel[key] || null;
      var coarse = kModel[ck] || allModel[ck] || null;
      var broad = kModel[tk] || allModel[tk] || null;

      // Blend K-model with all-model for robustness
      var primary;
      if (fine && fineAll) {
        primary = new Array(6);
        for (var c = 0; c < 6; c++) primary[c] = (1 - blendAll) * fine[c] + blendAll * fineAll[c];
      } else {
        primary = fine || fineAll || null;
      }

      var prior;
      if (primary && coarse) {
        prior = new Array(6);
        for (var c = 0; c < 6; c++) prior[c] = (1 - regWeight) * primary[c] + regWeight * coarse[c];
      } else if (primary) {
        prior = primary;
      } else if (coarse && broad) {
        prior = new Array(6);
        for (var c = 0; c < 6; c++) prior[c] = 0.8 * coarse[c] + 0.2 * broad[c];
      } else {
        prior = coarse || broad || [1/6,1/6,1/6,1/6,1/6,1/6];
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
