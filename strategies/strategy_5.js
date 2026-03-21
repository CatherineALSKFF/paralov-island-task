const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRoundNums = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Gaussian-weighted round importance at multiple bandwidths
  var bandwidths = [0.04, 0.08, 0.15];
  var perBwWeights = bandwidths.map(function(bw) {
    var weights = {};
    var wSum = 0;
    for (var i = 0; i < allRoundNums.length; i++) {
      var r = allRoundNums[i];
      var diff = (growthRates[String(r)] || 0.15) - targetGrowth;
      var w = Math.exp(-diff * diff / (2 * bw * bw));
      weights[r] = w;
      wSum += w;
    }
    for (var i = 0; i < allRoundNums.length; i++) {
      weights[allRoundNums[i]] /= wSum;
    }
    return weights;
  });

  // Precompute weighted distributions per key per bandwidth
  var allKeys = {};
  for (var i = 0; i < allRoundNums.length; i++) {
    var buckets = perRoundBuckets[allRoundNums[i]];
    if (!buckets) continue;
    for (var k in buckets) allKeys[k] = true;
  }

  var bwModels = bandwidths.map(function(bw, bi) {
    var weights = perBwWeights[bi];
    var model = {};
    for (var key in allKeys) {
      var dist = [0,0,0,0,0,0];
      var tw = 0;
      var tc = 0;
      for (var i = 0; i < allRoundNums.length; i++) {
        var r = allRoundNums[i];
        var buckets = perRoundBuckets[r];
        if (!buckets || !buckets[key] || buckets[key].count === 0) continue;
        var b = buckets[key];
        var w = weights[r];
        for (var c = 0; c < 6; c++) dist[c] += w * (b.sum[c] / b.count);
        tw += w;
        tc += b.count;
      }
      if (tw > 1e-10) {
        for (var c = 0; c < 6; c++) dist[c] /= tw;
        model[key] = { dist: dist, count: tc };
      }
    }
    return model;
  });

  var pred = [];
  for (var y = 0; y < H; y++) {
    var row = [];
    for (var x = 0; x < W; x++) {
      var key = getFeatureKey(initGrid, settPos, y, x);
      var fbKey = key.slice(0, -1);

      // Ensemble across bandwidths
      var ensemble = [0,0,0,0,0,0];
      var ensembleN = 0;

      for (var bi = 0; bi < bwModels.length; bi++) {
        var model = bwModels[bi];
        var spec = model[key] || null;
        var coarse = model[fbKey] || null;

        var cellDist = null;
        if (spec && coarse) {
          // Count-adaptive blend: more specific data -> trust specific more
          var blendAlpha = Math.min(0.85, 0.3 + 0.55 * Math.min(1, spec.count / 20));
          cellDist = [0,0,0,0,0,0];
          for (var c = 0; c < 6; c++) {
            cellDist[c] = blendAlpha * spec.dist[c] + (1 - blendAlpha) * coarse.dist[c];
          }
        } else if (spec) {
          cellDist = spec.dist;
        } else if (coarse) {
          cellDist = coarse.dist;
        }

        if (cellDist) {
          for (var c = 0; c < 6; c++) ensemble[c] += cellDist[c];
          ensembleN++;
        }
      }

      var prior;
      if (ensembleN > 0) {
        prior = [0,0,0,0,0,0];
        for (var c = 0; c < 6; c++) prior[c] = ensemble[c] / ensembleN;
      } else {
        prior = [1/6,1/6,1/6,1/6,1/6,1/6];
      }

      // Adaptive floor: low entropy cells get tiny floor, high entropy get larger
      var entropy = 0;
      for (var c = 0; c < 6; c++) {
        if (prior[c] > 1e-10) entropy -= prior[c] * Math.log(prior[c]);
      }
      var adaptiveFloor = floor * (0.05 + 0.95 * entropy / Math.log(6));

      var floored = [0,0,0,0,0,0];
      var sum = 0;
      for (var c = 0; c < 6; c++) {
        floored[c] = Math.max(prior[c], adaptiveFloor);
        sum += floored[c];
      }
      var result = [0,0,0,0,0,0];
      for (var c = 0; c < 6; c++) result[c] = floored[c] / sum;

      row.push(result);
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };