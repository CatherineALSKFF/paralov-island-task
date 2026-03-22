const { H, W, getFeatureKey } = require('./shared');

function getEnrichedKey(grid, settPos, y, x) {
  const v = grid[y][x];
  if (v === 10) return 'O';
  if (v === 5) return 'M';
  const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P';
  let nS = 0;
  for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
    if (!dy && !dx) continue;
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx)) nS++;
  }
  let coastal = false;
  for (const [dy, dx] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) coastal = true;
  }
  const sKey = nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3';
  let suffix = '';
  if (t !== 'S') {
    let minSD = 40;
    for (let dy = -7; dy <= 7; dy++) for (let dx = -7; dx <= 7; dx++) {
      const ny = y + dy, nx = x + dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx)) {
        minSD = Math.min(minSD, Math.max(Math.abs(dy), Math.abs(dx)));
      }
    }
    if (nS === 0) {
      suffix = minSD <= 5 ? 'n' : minSD <= 7 ? 'm' : 'f';
    } else if (nS <= 2) {
      suffix = nS === 1 ? 'a' : 'b';
    }
  }
  return t + sKey + (coastal ? 'c' : '') + suffix;
}

function buildRegModel(perRoundBuckets, growthRates, targetGrowth, sigma, allRounds, regWeight) {
  const rWeights = {};
  let totalW = 0;
  for (const rn of allRounds) {
    const g = growthRates[String(rn)] || 0.15;
    const d = (g - targetGrowth) / sigma;
    rWeights[rn] = Math.exp(-0.5 * d * d);
    totalW += rWeights[rn];
  }
  for (const rn of allRounds) rWeights[rn] /= totalW;

  const keyData = {};
  for (const rn of allRounds) {
    const rb = perRoundBuckets[String(rn)];
    if (!rb) continue;
    const g = growthRates[String(rn)] || 0.15;
    const w = rWeights[rn];
    for (const [key, val] of Object.entries(rb)) {
      if (!keyData[key]) keyData[key] = [];
      keyData[key].push({ g, dist: val.sum.map(s => s / val.count), w });
    }
  }

  const model = {};
  for (const [key, data] of Object.entries(keyData)) {
    const wMean = [0, 0, 0, 0, 0, 0];
    for (const d of data) for (let c = 0; c < 6; c++) wMean[c] += d.w * d.dist[c];

    let sumW = 0, sumWG = 0, sumWG2 = 0;
    for (const d of data) { sumW += d.w; sumWG += d.w * d.g; sumWG2 += d.w * d.g * d.g; }
    const gMean = sumWG / sumW;
    const gVar = sumWG2 / sumW - gMean * gMean;

    if (gVar > 1e-8 && data.length >= 4) {
      const regPred = [0, 0, 0, 0, 0, 0];
      for (let c = 0; c < 6; c++) {
        let sumWGY = 0, sumWY = 0;
        for (const d of data) { sumWGY += d.w * d.g * d.dist[c]; sumWY += d.w * d.dist[c]; }
        const cov = sumWGY / sumW - gMean * (sumWY / sumW);
        regPred[c] = Math.max((sumWY / sumW) + (cov / gVar) * (targetGrowth - gMean), 0);
      }
      const s = regPred.reduce((a, b) => a + b, 0);
      if (s > 0) for (let c = 0; c < 6; c++) regPred[c] /= s;
      model[key] = regPred.map((v, c) => regWeight * v + (1 - regWeight) * wMean[c]);
    } else {
      model[key] = [...wMean];
    }
  }
  return model;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.00001;
  const regWeight = config.regWeight || 0.9;
  const sigma = config.sigma || 0.10;
  const enrichBlend = config.enrichBlend || 0.3; // how much enriched key contributes
  const nPrior = config.N_PRIOR || 4;
  const vpCounts = config._vpCounts || null;
  const vpTotal = config._vpTotal || null;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  const model = buildRegModel(perRoundBuckets, growthRates, targetGrowth, sigma, allRounds, regWeight);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const eKey = getEnrichedKey(initGrid, settPos, y, x);
      const sKey = getFeatureKey(initGrid, settPos, y, x);

      // Standard key prediction (robust)
      let sPred = model[sKey] || null;
      if (!sPred) {
        const fb = sKey.slice(0, -1);
        sPred = model[fb] || null;
      }

      // Enriched key prediction (specific) — only if different from standard key
      let ePred = (eKey !== sKey) ? model[eKey] : null;

      let prior;
      if (ePred && sPred) {
        // Blend enriched with standard
        prior = ePred.map((v, c) => enrichBlend * v + (1 - enrichBlend) * sPred[c]);
      } else if (sPred) {
        prior = [...sPred];
      } else if (ePred) {
        prior = [...ePred];
      } else {
        prior = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6];
      }

      if (vpCounts && vpTotal && vpTotal[y][x] > 0) {
        const nObs = vpTotal[y][x];
        const updated = prior.map((p, c) => nPrior * p + vpCounts[y][x][c]);
        const total = nPrior + nObs;
        const floored = updated.map(v => Math.max(v / total, floor));
        const sum = floored.reduce((a, b) => a + b, 0);
        row.push(floored.map(v => v / sum));
      } else {
        const floored = prior.map(v => Math.max(v, floor));
        const sum = floored.reduce((a, b) => a + b, 0);
        row.push(floored.map(v => v / sum));
      }
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
