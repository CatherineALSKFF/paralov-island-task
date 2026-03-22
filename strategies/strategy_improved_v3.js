const { H, W, getFeatureKey, mergeBuckets } = require('./shared');

// Enriched feature key: adds distance-based suffix for nS=0 and nS=1-2 cells
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
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) coastal = true;
  }
  const sKey = nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3';
  let suffix = '';
  if (t !== 'S') {
    if (nS === 0) {
      let minSD = 40;
      for (let dy = -7; dy <= 7; dy++) for (let dx = -7; dx <= 7; dx++) {
        const ny = y + dy, nx = x + dx;
        if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny * W + nx))
          minSD = Math.min(minSD, Math.max(Math.abs(dy), Math.abs(dx)));
      }
      suffix = minSD <= 5 ? 'n' : minSD <= 7 ? 'm' : 'f';
    } else if (nS <= 2) {
      suffix = nS === 1 ? 'a' : 'b';
    }
  }
  return t + sKey + (coastal ? 'c' : '') + suffix;
}

// Key hierarchy: enriched -> basic -> terrain-only
function getKeyHierarchy(enrichedKey) {
  const keys = [enrichedKey];
  // Strip enriched suffix (a/b/f/m/n) to get basic key
  const last = enrichedKey[enrichedKey.length - 1];
  if ('abfmn'.includes(last)) {
    keys.push(enrichedKey.slice(0, -1)); // basic key (with coast if present)
  }
  // Strip coast flag
  if (keys[keys.length - 1].endsWith('c') && keys[keys.length - 1].length > 1) {
    keys.push(keys[keys.length - 1].slice(0, -1)); // no coast
  }
  // Terrain only
  const terrain = enrichedKey[0];
  if (terrain !== 'O' && terrain !== 'M') {
    const tOnly = terrain;
    if (keys[keys.length - 1] !== tOnly) keys.push(tOnly);
  }
  return keys;
}

// Gaussian-weighted merge: weight rounds by growth similarity, pool by cell count
function gaussianMerge(perRoundBuckets, growthRates, targetGrowth, sigma, excludeRound) {
  const rounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== excludeRound);
  const weights = {};
  for (const r of rounds) {
    const diff = (growthRates[String(r)] || 0.15) - targetGrowth;
    weights[r] = Math.exp(-diff * diff / (2 * sigma * sigma));
  }
  const model = {};
  for (const r of rounds) {
    const b = perRoundBuckets[String(r)];
    if (!b) continue;
    const w = weights[r];
    for (const [key, val] of Object.entries(b)) {
      if (!model[key]) model[key] = { count: 0, sum: new Float64Array(6) };
      model[key].count += w * val.count;
      for (let c = 0; c < 6; c++) model[key].sum[c] += w * val.sum[c];
    }
  }
  const out = {};
  for (const [k, v] of Object.entries(model)) {
    if (v.count > 0) out[k] = Array.from(v.sum).map(s => s / v.count);
  }
  return out;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const regWeight = 0.35;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  // Build ensemble of Gaussian-weighted models at multiple bandwidths
  const sigmas = [0.03, 0.05, 0.08, 0.15];
  const models = sigmas.map(s => gaussianMerge(perRoundBuckets, growthRates, targetGrowth, s, testRound));

  // Uniform fallback model
  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);
  const uniformModel = mergeBuckets(perRoundBuckets, allRounds);

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const enrichedKey = getEnrichedKey(initGrid, settPos, y, x);
      const hierarchy = getKeyHierarchy(enrichedKey);

      // Ensemble: average predictions from all sigma models
      const ens = new Float64Array(6);
      let ensN = 0;

      for (const model of models) {
        // Find best available key in hierarchy
        let bestProb = null;
        let bestLevel = -1;
        for (let i = 0; i < hierarchy.length; i++) {
          if (model[hierarchy[i]]) {
            if (bestProb === null) { bestProb = model[hierarchy[i]]; bestLevel = i; }
            break;
          }
        }
        if (!bestProb) {
          // Try uniform model
          for (const k of hierarchy) {
            if (uniformModel[k]) { bestProb = uniformModel[k]; break; }
          }
          if (!bestProb) continue;
        }

        // Find coarser key for regularization blend
        let coarseProb = null;
        for (let i = Math.max(1, bestLevel + 1); i < hierarchy.length; i++) {
          const p = model[hierarchy[i]] || uniformModel[hierarchy[i]];
          if (p) { coarseProb = p; break; }
        }

        let cellPred;
        if (coarseProb) {
          cellPred = bestProb.map((v, c) => (1 - regWeight) * v + regWeight * coarseProb[c]);
        } else {
          cellPred = bestProb;
        }

        for (let c = 0; c < 6; c++) ens[c] += cellPred[c];
        ensN++;
      }

      if (ensN === 0) {
        row.push([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]);
        continue;
      }

      // Average ensemble
      const dist = [];
      for (let c = 0; c < 6; c++) dist.push(ens[c] / ensN);

      // Adaptive floor: higher for uncertain cells
      let entropy = 0;
      for (let c = 0; c < 6; c++) {
        if (dist[c] > 1e-8) entropy -= dist[c] * Math.log(dist[c]);
      }
      const adaptiveFloor = floor * (1 + 4 * entropy / Math.log(6));

      const floored = dist.map(v => Math.max(v, adaptiveFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
