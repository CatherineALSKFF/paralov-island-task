const { H, W } = require('./shared');
const fs = require('fs');
const path = require('path');

const enrichedBuckets = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'enriched_buckets.json'), 'utf8'));

const _featureCache = {};
function computeRoundFeatures(roundNum) {
  if (_featureCache[roundNum]) return _featureCache[roundNum];
  const f = path.join(__dirname, '..', 'data', 'inits_R' + roundNum + '.json');
  if (!fs.existsSync(f)) return null;
  const raw = JSON.parse(fs.readFileSync(f, 'utf8'));
  let totS = 0, totF = 0, totM = 0, totO = 0, nSeeds = 0, totalCluster = 0;
  for (let s = 0; s < 5; s++) {
    const item = raw[s]; if (!item) continue;
    const grid = Array.isArray(item) && Array.isArray(item[0]) ? item : item.grid;
    if (!grid) continue;
    nSeeds++;
    const setts = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const v = grid[y][x];
      if (v === 1 || v === 2) { totS++; setts.push({y, x}); }
      else if (v === 4) totF++;
      else if (v === 5) totM++;
      else if (v === 10) totO++;
    }
    let cs = 0;
    for (let i = 0; i < setts.length; i++) {
      let md = Infinity;
      for (let j = 0; j < setts.length; j++) {
        if (i===j) continue;
        md = Math.min(md, Math.abs(setts[i].y-setts[j].y)+Math.abs(setts[i].x-setts[j].x));
      }
      cs += md;
    }
    totalCluster += setts.length > 1 ? cs / setts.length : 10;
  }
  _featureCache[roundNum] = { sett: totS/nSeeds, forest: totF/nSeeds, mtn: totM/nSeeds, ocean: totO/nSeeds, cluster: totalCluster/nSeeds };
  return _featureCache[roundNum];
}

function getEnrichedKey(grid, settPos, y, x) {
  const v = grid[y][x];
  if (v === 10) return 'O';
  if (v === 5) return 'M';
  const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P';
  let nS = 0, nO1 = 0;
  for (const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y+dy, nx = x+dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) nO1++;
  }
  for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
    if (!dy && !dx) continue;
    const ny = y+dy, nx = x+dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny*W+nx)) nS++;
  }
  const coastal = nO1 > 0;
  const sKey = nS === 0 ? '0' : nS <= 2 ? '1' : nS <= 5 ? '2' : '3';
  let suffix = '';
  if (t !== 'S') {
    let minSD = 40;
    for (let dy = -7; dy <= 7; dy++) for (let dx = -7; dx <= 7; dx++) {
      const ny = y+dy, nx = x+dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && settPos.has(ny*W+nx)) {
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

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = 0.00001;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const allRounds = Object.keys(enrichedBuckets).map(Number).filter(n => n !== testRound);
  const sigma_g = 0.05;
  const sigma_s = 10;
  const sigma_c = 3;
  
  // Compute test round features
  const testFeats = { sett: settlements.length, forest: 0, mtn: 0, ocean: 0 };
  let clusterSum = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const v = initGrid[y][x];
    if (v === 4) testFeats.forest++;
    else if (v === 5) testFeats.mtn++;
    else if (v === 10) testFeats.ocean++;
  }
  for (let i = 0; i < settlements.length; i++) {
    let md = Infinity;
    for (let j = 0; j < settlements.length; j++) {
      if (i===j) continue;
      md = Math.min(md, Math.abs(settlements[i].y-settlements[j].y)+Math.abs(settlements[i].x-settlements[j].x));
    }
    clusterSum += md;
  }
  testFeats.cluster = settlements.length > 1 ? clusterSum / settlements.length : 10;
  
  // Multi-feature weighted merging
  const roundWeights = {};
  let totalWeight = 0;
  for (const rn of allRounds) {
    const g = growthRates[String(rn)] || 0.15;
    const feats = computeRoundFeatures(rn);
    let dist = ((g - targetGrowth) / sigma_g) ** 2;
    if (feats) {
      dist += ((feats.sett - testFeats.sett) / sigma_s) ** 2;
      dist += ((feats.cluster - testFeats.cluster) / sigma_c) ** 2;
    }
    const w = Math.exp(-0.5 * dist);
    roundWeights[rn] = w;
    totalWeight += w;
  }
  for (const rn of allRounds) roundWeights[rn] /= totalWeight;
  
  // Build enriched model
  const eModel = {};
  for (const rn of allRounds) {
    const rb = enrichedBuckets[String(rn)];
    if (!rb) continue;
    const w = roundWeights[rn];
    for (const [key, val] of Object.entries(rb)) {
      if (!eModel[key]) eModel[key] = new Float64Array(6);
      const avg = val.sum.map(s => s / val.count);
      for (let c = 0; c < 6; c++) eModel[key][c] += w * avg[c];
    }
  }
  for (const key of Object.keys(eModel)) {
    const s = Array.from(eModel[key]).reduce((a,b) => a+b, 0);
    if (s > 0) for (let c = 0; c < 6; c++) eModel[key][c] /= s;
  }
  
  // Build standard model as fallback
  const sModel = {};
  for (const rn of allRounds) {
    const rb = perRoundBuckets[String(rn)];
    if (!rb) continue;
    const w = roundWeights[rn];
    for (const [key, val] of Object.entries(rb)) {
      if (!sModel[key]) sModel[key] = new Float64Array(6);
      const avg = val.sum.map(s => s / val.count);
      for (let c = 0; c < 6; c++) sModel[key][c] += w * avg[c];
    }
  }
  for (const key of Object.keys(sModel)) {
    const s = Array.from(sModel[key]).reduce((a,b) => a+b, 0);
    if (s > 0) for (let c = 0; c < 6; c++) sModel[key][c] /= s;
  }
  
  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const eKey = getEnrichedKey(initGrid, settPos, y, x);
      let prior = eModel[eKey] ? Array.from(eModel[eKey]) : null;
      if (!prior) {
        // Fallback to standard model
        const { getFeatureKey } = require('./shared');
        const sKey = getFeatureKey(initGrid, settPos, y, x);
        prior = sModel[sKey] ? Array.from(sModel[sKey]) : null;
        if (!prior) {
          const fb = sKey.slice(0, -1);
          prior = sModel[fb] ? Array.from(sModel[fb]) : null;
        }
      }
      if (!prior) prior = [1/6,1/6,1/6,1/6,1/6,1/6];
      
      const floored = prior.map(v => Math.max(v, floor));
      const sum = floored.reduce((a, b) => a + b, 0);
      row.push(floored.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}
module.exports = { predict };
