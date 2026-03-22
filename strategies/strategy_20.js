const { H, W, getFeatureKey, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  var K = 4;
  var floor = 0.0001;
  var bw = 0.05;
  var targetGrowth = growthRates[String(testRound)] || 0.15;
  var candidates = {};
  for (var rn in growthRates) {
    if (parseInt(rn) !== testRound) candidates[rn] = growthRates[rn];
  }
  var closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  var allRounds = Object.keys(perRoundBuckets).map(Number).filter(function(n) { return n !== testRound; });

  var settPos = new Set();
  for (var i = 0; i < settlements.length; i++) settPos.add(settlements[i].y * W + settlements[i].x);

  // Gaussian-weighted per-round merge (each round's average weighted by growth similarity)
  function gaussMerge(rounds) {
    var model = {};
    var keyW = {};
    var weights = {};
    var wSum = 0;
    for (var ri = 0; ri < rounds.length; ri++) {
      var r = rounds[ri];
      var g = growthRates[String(r)] || 0.15;
      var d = g - targetGrowth;
      weights[r] = Math.exp(-d * d / (2 * bw * bw));
      wSum += weights[r];
    }
    if (wSum > 0) for (var ri = 0; ri < rounds.length; ri++) weights[rounds[ri]] /= wSum;
    for (var ri = 0; ri < rounds.length; ri++) {
      var r = rounds[ri];
      var buckets = perRoundBuckets[String(r)];
      if (!buckets) continue;
      var w = weights[r];
      for (var key in buckets) {
        var v = buckets[key];
        if (v.count === 0) continue;
        if (!model[key]) { model[key] = [0,0,0,0,0,0]; keyW[key] = 0; }
        keyW[key] += w;
        for (var c = 0; c < 6; c++) model[key][c] += w * (v.sum[c] / v.count);
      }
    }
    for (var key in model) {
      var kw = keyW[key];
      for (var c = 0; c < 6; c++) model[key][c] /= kw;
    }
    return model;
  }

  var kModel = gaussMerge(closestRounds);
  var allModel = gaussMerge(allRounds);

  // Build initial predictions
  var pred = [];
  for (var y = 0; y < H; y++) {
    var row = [];
    for (var x = 0; x < W; x++) {
      var key = getFeatureKey(initGrid, settPos, y, x);
      var prior = kModel[key] ? kModel[key].slice() : allModel[key] ? allModel[key].slice() : null;
      if (!prior) {
        var fb = key.slice(0, -1);
        prior = kModel[fb] ? kModel[fb].slice() : allModel[fb] ? allModel[fb].slice() : [1/6,1/6,1/6,1/6,1/6,1/6];
      }
      var sum = 0;
      for (var c = 0; c < 6; c++) { prior[c] = Math.max(prior[c], floor); sum += prior[c]; }
      for (var c = 0; c < 6; c++) prior[c] /= sum;
      row.push(prior);
    }
    pred.push(row);
  }

  // Growth-rate calibration: scale settlement+port probabilities to match target growth
  var totalS = 0;
  for (var y = 0; y < H; y++)
    for (var x = 0; x < W; x++)
      totalS += pred[y][x][1] + pred[y][x][2];
  var predGrowth = totalS / (H * W);
  if (predGrowth > 0.001) {
    var ratio = targetGrowth / predGrowth;
    ratio = Math.max(0.5, Math.min(2.0, ratio));
    if (Math.abs(ratio - 1.0) > 0.02) {
      for (var y = 0; y < H; y++)
        for (var x = 0; x < W; x++) {
          var p = pred[y][x];
          p[1] *= ratio;
          p[2] *= ratio;
          var sum = 0;
          for (var c = 0; c < 6; c++) { p[c] = Math.max(p[c], floor); sum += p[c]; }
          for (var c = 0; c < 6; c++) p[c] /= sum;
        }
    }
  }

  return pred;
}

module.exports = { predict };
