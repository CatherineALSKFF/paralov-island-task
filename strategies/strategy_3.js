const { H, W, getFeatureKey } = require('./shared');
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'data');

function getDistKey(grid, settPos, y, x) {
  const v = grid[y][x];
  if (v === 10) return 'O'; if (v === 5) return 'M';
  const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P';
  let nO1 = 0, nS1 = 0;
  for (const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y+dy, nx = x+dx;
    if (ny>=0 && ny<H && nx>=0 && nx<W) {
      if (grid[ny][nx] === 10) nO1++;
      if (settPos.has(ny*W+nx)) nS1++;
    }
  }
  const coastal = nO1 > 0;
  if (t === 'S') {
    let nS = 0;
    for (let dy=-3;dy<=3;dy++) for(let dx=-3;dx<=3;dx++) {
      if(!dy&&!dx) continue; const ny=y+dy,nx=x+dx;
      if(ny>=0&&ny<H&&nx>=0&&nx<W&&settPos.has(ny*W+nx)) nS++;
    }
    return 'S'+(nS<=1?'0':nS<=3?'1':'2')+(coastal?'c':'');
  }
  let minSD = 40;
  for (let dy=-10;dy<=10;dy++) for(let dx=-10;dx<=10;dx++) {
    const ny=y+dy,nx=x+dx;
    if(ny>=0&&ny<H&&nx>=0&&nx<W&&settPos.has(ny*W+nx))
      minSD = Math.min(minSD, Math.max(Math.abs(dy), Math.abs(dx)));
  }
  const dKey = minSD<=1?'1':minSD<=2?'2':minSD<=3?'3':minSD<=5?'4':minSD<=7?'5':'6';
  const suffix = (minSD<=1 && nS1>0) ? 't' : '';
  return t + dKey + (coastal?'c':'') + suffix;
}

let _distBuckets = null;
function getDistBuckets() {
  if (_distBuckets) return _distBuckets;
  _distBuckets = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'enriched_buckets_v8.json'), 'utf8'));
  return _distBuckets;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const LAMBDA = config.LAMBDA || 0.001;
  const SIGMA = config.SIGMA || 0.15;
  const temp = config.temp || 1.1;
  const linBlend = config.linBlend || 0.95;
  const eBlend = config.eBlend || 0.75;
  const targetGrowth = growthRates[String(testRound)] || 0.15;
  const distBuckets = getDistBuckets();
  const allRounds = Object.keys(distBuckets).map(Number).filter(n => n !== testRound);

  // Gaussian weights
  const rw = {};
  let tw = 0;
  for (const rn of allRounds) {
    const g = growthRates[String(rn)];
    if (g === undefined) { rw[rn] = 0.1; tw += 0.1; continue; }
    const d = g - targetGrowth;
    const w = Math.exp(-(d*d)/(2*SIGMA*SIGMA));
    rw[rn] = w; tw += w;
  }

  // Build weighted mean model from enriched buckets
  const meanModel = {};
  for (const rn of allRounds) {
    const w = rw[rn] / tw;
    const rb = distBuckets[String(rn)];
    if (!rb) continue;
    for (const [key, val] of Object.entries(rb)) {
      if (!meanModel[key]) meanModel[key] = [0,0,0,0,0,0];
      const avg = val.sum.map(v => v / val.count);
      for (let c = 0; c < 6; c++) meanModel[key][c] += w * avg[c];
    }
  }

  // Build linear regression model from enriched buckets
  const linModel = {};
  const allKeys = new Set();
  for (const rn of allRounds) {
    const rb = distBuckets[String(rn)];
    if (rb) for (const k of Object.keys(rb)) allKeys.add(k);
  }

  for (const key of allKeys) {
    const points = [];
    for (const rn of allRounds) {
      const rb = distBuckets[String(rn)];
      if (!rb || !rb[key]) continue;
      points.push({
        g: growthRates[String(rn)] || 0.15,
        p: rb[key].sum.map(v => v / rb[key].count),
        w: rw[rn] / tw
      });
    }
    if (points.length < 5) continue;

    let sw = 0, swG = 0, swG2 = 0;
    for (const pt of points) { sw += pt.w; swG += pt.w * pt.g; swG2 += pt.w * pt.g * pt.g; }
    const wMG = swG / sw, wVG = swG2 / sw - wMG * wMG;
    if (wVG < 1e-10) continue;

    const pred = [0,0,0,0,0,0];
    for (let c = 0; c < 6; c++) {
      let swP = 0, swGP = 0;
      for (const pt of points) { swP += pt.w * pt.p[c]; swGP += pt.w * pt.g * pt.p[c]; }
      const wMP = swP / sw;
      const slope = (swGP / sw - wMG * wMP) / wVG;
      pred[c] = Math.max(wMP + slope * (targetGrowth - wMG), 0);
    }
    linModel[key] = pred;
  }

  // Also build standard key regression as fallback
  const stdLinModel = {};
  const stdKeys = new Set();
  for (const rn of allRounds) {
    const rb = perRoundBuckets[String(rn)];
    if (rb) for (const k of Object.keys(rb)) stdKeys.add(k);
  }
  for (const key of stdKeys) {
    const points = [];
    for (const rn of allRounds) {
      const rb = perRoundBuckets[String(rn)];
      if (!rb || !rb[key]) continue;
      points.push({
        g: growthRates[String(rn)] || 0.15,
        p: rb[key].sum.map(v => v / rb[key].count),
        w: rw[rn] / tw
      });
    }
    if (points.length < 5) continue;
    let sw = 0, swG = 0, swG2 = 0;
    for (const pt of points) { sw += pt.w; swG += pt.w * pt.g; swG2 += pt.w * pt.g * pt.g; }
    const wMG = swG / sw, wVG = swG2 / sw - wMG * wMG;
    if (wVG < 1e-10) continue;
    const pred = [0,0,0,0,0,0];
    for (let c = 0; c < 6; c++) {
      let swP = 0, swGP = 0;
      for (const pt of points) { swP += pt.w * pt.p[c]; swGP += pt.w * pt.g * pt.p[c]; }
      const wMP = swP / sw;
      const slope = (swGP / sw - wMG * wMP) / wVG;
      pred[c] = Math.max(wMP + slope * (targetGrowth - wMG), 0);
    }
    stdLinModel[key] = pred;
  }

  // Standard weighted mean as additional fallback
  const stdMeanModel = {};
  for (const rn of allRounds) {
    const w = rw[rn] / tw;
    const rb = perRoundBuckets[String(rn)];
    if (!rb) continue;
    for (const [key, val] of Object.entries(rb)) {
      if (!stdMeanModel[key]) stdMeanModel[key] = [0,0,0,0,0,0];
      const avg = val.sum.map(v => v / val.count);
      for (let c = 0; c < 6; c++) stdMeanModel[key][c] += w * avg[c];
    }
  }

  function lookup(key, fallbackKey) {
    // Try enriched linear first, then enriched mean, then standard linear, then standard mean
    if (linModel[key]) return [...linModel[key]];
    if (meanModel[key]) return [...meanModel[key]];
    if (linModel[fallbackKey]) return [...linModel[fallbackKey]];

    // Standard key fallback
    const sKey = fallbackKey;
    if (stdLinModel[sKey]) return [...stdLinModel[sKey]];
    if (stdMeanModel[sKey]) return [...stdMeanModel[sKey]];
    let fb = sKey;
    while (fb.length > 1) { fb = fb.slice(0, -1); if (stdMeanModel[fb]) return [...stdMeanModel[fb]]; }
    return [1/6,1/6,1/6,1/6,1/6,1/6];
  }

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  const pred = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      const dKey = getDistKey(initGrid, settPos, y, x);
      const sKey = getFeatureKey(initGrid, settPos, y, x);

      // Get enriched key prediction (try key, then strip last char)
      let enrichedLin = linModel[dKey] ? [...linModel[dKey]] : null;
      if (!enrichedLin) {
        const fb = dKey.slice(0, -1);
        enrichedLin = linModel[fb] ? [...linModel[fb]] : null;
      }

      // Get standard key prediction
      let stdLin = stdLinModel[sKey] ? [...stdLinModel[sKey]] : null;
      if (!stdLin) {
        const fb = sKey.slice(0, -1);
        stdLin = stdLinModel[fb] ? [...stdLinModel[fb]] : null;
      }

      // Blend enriched and standard linear predictions
      let p;
      if (enrichedLin && stdLin) {
        p = enrichedLin.map((v, i) => eBlend * v + (1 - eBlend) * stdLin[i]);
      } else if (enrichedLin) {
        p = enrichedLin;
      } else if (stdLin) {
        p = stdLin;
      } else {
        p = lookup(dKey, sKey);
      }

      // Temperature scaling
      if (temp !== 1.0) {
        const logP = p.map(v => Math.log(Math.max(v, 1e-12)) / temp);
        const maxLP = Math.max(...logP);
        const expP = logP.map(v => Math.exp(v - maxLP));
        const s = expP.reduce((a, b) => a + b, 0);
        p = expP.map(v => v / s);
      }

      // Adaptive floor
      let ent = 0;
      for (let c = 0; c < 6; c++) if (p[c] > 0.001) ent -= p[c] * Math.log(p[c]);
      const cf = ent > 0.5 ? floor : floor * 0.1;

      const fl = p.map(v => Math.max(v, cf));
      const sum = fl.reduce((a, b) => a + b, 0);
      row.push(fl.map(v => v / sum));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
