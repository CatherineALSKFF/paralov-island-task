const { H, W, terrainToClass, getFeatureKey, mergeBuckets, selectClosestRounds } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const K = config.K || 3;
  const floor = config.FLOOR || 0.0001;
  const baseTemp = config.BASE_TEMP || 1.10;
  const dynTempAdd = config.DYN_TEMP || 0.08;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const candidates = { ...growthRates }; delete candidates[String(testRound)];
  const closestRounds = selectClosestRounds(candidates, targetGrowth, K);
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const adaptiveModel = mergeBuckets(perRoundBuckets, closestRounds);
  const allModel = mergeBuckets(perRoundBuckets, allRounds);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Precompute min distance to settlement for each cell
  const distMap = [];
  const settList = settlements.map(s => [s.y, s.x]);
  for (let y = 0; y < H; y++) {
    distMap.push(new Float64Array(W));
    for (let x = 0; x < W; x++) {
      let minD = 999;
      for (const [sy, sx] of settList) {
        const d = Math.max(Math.abs(y - sy), Math.abs(x - sx)); // Chebyshev
        if (d < minD) minD = d;
      }
      distMap[y][x] = minD;
    }
  }

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      const fb = key.slice(0, -1);

      let prior = adaptiveModel[key] || allModel[key] || adaptiveModel[fb] || allModel[fb] || null;
      if (!prior) {
        row.push([1/6,1/6,1/6,1/6,1/6,1/6]);
        continue;
      }
      prior = [...prior];

      // Position-aware temperature: dynamic cells (near settlements) get higher temp
      const dist = distMap[y][x];
      const isStatic = (key === 'O' || key === 'M');
      // Dynamic zone: within range 5 of a settlement
      const dynFactor = isStatic ? 0 : Math.max(0, 1 - dist / 8);
      const temp = baseTemp + dynTempAdd * dynFactor;

      if (temp > 1.01) {
        let s = 0;
        for (let c = 0; c < 6; c++) {
          prior[c] = Math.pow(Math.max(prior[c], 1e-10), 1 / temp);
          s += prior[c];
        }
        for (let c = 0; c < 6; c++) prior[c] /= s;
      }

      // Adaptive floor
      let entropy = 0;
      for (let c = 0; c < 6; c++) {
        if (prior[c] > 1e-10) entropy -= prior[c] * Math.log(prior[c]);
      }
      const eRatio = entropy / Math.log(6);
      const cellFloor = floor * (0.02 + 0.98 * eRatio * eRatio);

      const floored = prior.map(v => Math.max(v, cellFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}
module.exports = { predict };
