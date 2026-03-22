const fs = require('fs');
const path = require('path');
const { H, W, terrainToClass, getFeatureKey } = require('./shared');

const DATA_DIR = path.join(__dirname, '..', 'data');
const ROUND_IDS = {
  1: '71451d74', 2: '76909e29', 3: 'f1dac9a9', 4: '8e839974',
  5: 'fd3c92ff', 6: 'ae78003a', 7: '36e581f1', 8: 'c5cdf100',
  9: '2a341ace', 10: '75e625c3', 11: '324fde07', 12: '795bfb1f',
  13: '7b4bda99', 14: 'd0a2c894', 15: 'cc5442dd',
};

let cachedDB = null, cachedTestRound = null;

function buildCellDB(growthRates, testRound) {
  if (cachedTestRound === testRound && cachedDB) return cachedDB;
  const db = {};
  const allRounds = Object.keys(ROUND_IDS).map(Number).filter(n => n !== testRound);
  for (const r of allRounds) {
    const prefix = ROUND_IDS[r];
    const growth = growthRates[String(r)] || 0.15;
    const initFile = path.join(DATA_DIR, `inits_R${r}.json`);
    if (!fs.existsSync(initFile)) continue;
    const initsRaw = JSON.parse(fs.readFileSync(initFile, 'utf8'));
    for (let seed = 0; seed < 5; seed++) {
      const gtFile = path.join(DATA_DIR, `gt_${prefix}_s${seed}.json`);
      if (!fs.existsSync(gtFile)) continue;
      const gtData = JSON.parse(fs.readFileSync(gtFile, 'utf8'));
      const gt = gtData.ground_truth || gtData.gt;
      if (!gt) continue;
      const initItem = initsRaw[seed];
      if (!initItem) continue;
      const grid = Array.isArray(initItem) && Array.isArray(initItem[0]) ? initItem : (initItem.grid || null);
      if (!grid) continue;
      const settPos = new Set();
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++)
          if (grid[y][x] === 1 || grid[y][x] === 2) settPos.add(y * W + x);
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++) {
          const key = getFeatureKey(grid, settPos, y, x);
          if (key === 'O' || key === 'M') continue;
          if (!db[key]) db[key] = [];
          db[key].push({ gt: gt[y][x], growth });
        }
    }
  }
  cachedDB = db; cachedTestRound = testRound;
  return db;
}

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const sigmas = [0.045, 0.12]; // narrow + wide
  const ensembleW = [0.70, 0.30]; // 70% narrow, 30% wide
  const floor = config.FLOOR || 1e-8;
  const tempCoeff = config.TEMP || 0.80;
  const maxRoundShare = config.CAP || 0.20; // max weight share per round
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);
  const db = buildCellDB(growthRates, testRound);

  // For each sigma, build key predictions with weight capping
  function buildKeyPreds(sigma) {
    const cache = {};
    function getKeyPred(key) {
      if (cache[key] !== undefined) return cache[key];
      const matches = db[key];
      if (!matches || matches.length === 0) { cache[key] = null; return null; }

      // Group by growth rate (= round)
      const roundAvgs = {}, roundCounts = {};
      for (const m of matches) {
        const g = m.growth;
        if (!roundAvgs[g]) { roundAvgs[g] = [0,0,0,0,0,0]; roundCounts[g] = 0; }
        for (let c = 0; c < 6; c++) roundAvgs[g][c] += m.gt[c];
        roundCounts[g]++;
      }

      // Compute weights with capping
      const growths = Object.keys(roundAvgs);
      const rawWeights = growths.map(g => {
        const d = parseFloat(g) - targetGrowth;
        return Math.exp(-0.5 * (d / sigma) * (d / sigma));
      });
      let rawTotal = rawWeights.reduce((a, b) => a + b, 0);

      // Cap: no round gets more than maxRoundShare of total
      const cappedWeights = [...rawWeights];
      if (rawTotal > 0) {
        let excess = 0, nCapped = 0;
        const cap = maxRoundShare * rawTotal;
        for (let i = 0; i < cappedWeights.length; i++) {
          if (cappedWeights[i] > cap) {
            excess += cappedWeights[i] - cap;
            cappedWeights[i] = cap;
            nCapped++;
          }
        }
        // Redistribute excess proportionally to uncapped rounds
        if (excess > 0 && nCapped < cappedWeights.length) {
          const uncappedTotal = cappedWeights.filter((w, i) => rawWeights[i] <= cap).reduce((a, b) => a + b, 0);
          if (uncappedTotal > 0) {
            for (let i = 0; i < cappedWeights.length; i++) {
              if (rawWeights[i] <= cap) cappedWeights[i] += excess * (cappedWeights[i] / uncappedTotal);
            }
          }
        }
      }

      const tw = cappedWeights.reduce((a, b) => a + b, 0);
      const avg = [0,0,0,0,0,0];
      const roundPreds = [], roundWs = [];
      for (let i = 0; i < growths.length; i++) {
        const cnt = roundCounts[growths[i]];
        const rAvg = roundAvgs[growths[i]].map(v => v / cnt);
        const w = cappedWeights[i];
        for (let c = 0; c < 6; c++) avg[c] += w * rAvg[c];
        roundPreds.push(rAvg);
        roundWs.push(w);
      }
      if (tw > 0) for (let c = 0; c < 6; c++) avg[c] /= tw;

      let dis = 0;
      if (roundPreds.length >= 2) {
        for (let c = 0; c < 6; c++) {
          let wVar = 0;
          for (let i = 0; i < roundPreds.length; i++) {
            const diff = roundPreds[i][c] - avg[c];
            wVar += roundWs[i] * diff * diff;
          }
          dis += Math.sqrt(wVar / tw);
        }
      }

      cache[key] = { avg, dis, nRounds: roundPreds.length };
      return cache[key];
    }
    return getKeyPred;
  }

  const keyPredFns = sigmas.map(s => buildKeyPreds(s));

  const predGrid = [];
  const isMO = [];

  for (let y = 0; y < H; y++) {
    predGrid.push(new Array(W));
    isMO.push(new Array(W));
    for (let x = 0; x < W; x++) {
      const key = getFeatureKey(initGrid, settPos, y, x);
      if (key === 'O' || key === 'M') {
        isMO[y][x] = true; predGrid[y][x] = null; continue;
      }
      isMO[y][x] = false;

      const fb = key.length > 1 ? key.slice(0, -1) : null;

      // Ensemble: weighted average of predictions from different sigma values
      let ensProbs = [0,0,0,0,0,0];
      let ensDis = 0;
      let ensTotal = 0;

      for (let e = 0; e < sigmas.length; e++) {
        const getKP = keyPredFns[e];
        const fineData = getKP(key);
        const coarseData = fb ? getKP(fb) : null;

        let probs;
        if (fineData) {
          probs = [...fineData.avg];
          if (coarseData) {
            const regW = 0.10;
            for (let c = 0; c < 6; c++) probs[c] = (1 - regW) * probs[c] + regW * coarseData.avg[c];
          }
          ensDis += ensembleW[e] * fineData.dis;
        } else if (coarseData) {
          probs = [...coarseData.avg];
          ensDis += ensembleW[e] * coarseData.dis;
        } else {
          probs = [1/6,1/6,1/6,1/6,1/6,1/6];
          ensDis += ensembleW[e] * 0.5;
        }

        for (let c = 0; c < 6; c++) ensProbs[c] += ensembleW[e] * probs[c];
        ensTotal += ensembleW[e];
      }

      for (let c = 0; c < 6; c++) ensProbs[c] /= ensTotal;
      ensDis /= ensTotal;

      // Temperature softening
      if (ensDis > 0.1 && tempCoeff > 0) {
        const temp = 1.0 + tempCoeff * Math.min(ensDis, 1.0);
        let s = 0;
        for (let c = 0; c < 6; c++) {
          ensProbs[c] = Math.pow(Math.max(ensProbs[c], 1e-12), 1 / temp);
          s += ensProbs[c];
        }
        for (let c = 0; c < 6; c++) ensProbs[c] /= s;
      }

      const aFloor = floor + 0.003 * Math.min(ensDis, 1.0);
      const floored = ensProbs.map(v => Math.max(v, aFloor));
      const sum = floored.reduce((a, b) => a + b, 0);
      predGrid[y][x] = floored.map(v => v / sum);
    }
  }

  // M/O hedging (use the narrow sigma predictor for surrogates)
  const getKP = keyPredFns[0];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (!isMO[y][x]) continue;
      const nAvg = [0,0,0,0,0,0]; let nW = 0;
      for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
        if (dy === 0 && dx === 0) continue;
        const ny = y+dy, nx = x+dx;
        if (ny<0||ny>=H||nx<0||nx>=W) continue;
        if (isMO[ny][nx]) continue;
        const p = predGrid[ny][nx]; if (!p) continue;
        const dist = Math.sqrt(dy*dy+dx*dx);
        const w = 1/(1+dist);
        for (let c=0;c<6;c++) nAvg[c] += w*p[c]; nW += w;
      }
      let neighborPred = nW>0 ? nAvg.map(v=>v/nW) : [1/6,1/6,1/6,1/6,1/6,1/6];

      let nS=0;
      for (let dy=-3;dy<=3;dy++) for (let dx=-3;dx<=3;dx++) {
        if(!dy&&!dx) continue;
        const ny=y+dy,nx=x+dx;
        if(ny>=0&&ny<H&&nx>=0&&nx<W&&settPos.has(ny*W+nx)) nS++;
      }
      let coast=false;
      for (const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
        const ny=y+dy,nx=x+dx;
        if(ny>=0&&ny<H&&nx>=0&&nx<W&&initGrid[ny][nx]===10) coast=true;
      }
      const sBucket = nS===0?'0':nS<=2?'1':nS<=5?'2':'3';
      const cSuffix = coast?'c':'';

      const surrAvg=[0,0,0,0,0,0]; let sC=0;
      for (const t of ['P','F','S']) {
        let d=getKP(t+sBucket+cSuffix);
        if(!d&&cSuffix) d=getKP(t+sBucket);
        if(d) { for(let c=0;c<6;c++) surrAvg[c]+=d.avg[c]; sC++; }
      }
      let surrPred = sC>0 ? surrAvg.map(v=>v/sC) : [1/6,1/6,1/6,1/6,1/6,1/6];

      const bl = new Array(6);
      for(let c=0;c<6;c++) bl[c] = 0.40*neighborPred[c]+0.35*surrPred[c]+0.25/6;
      const moFloor=0.02;
      const fl = bl.map(v=>Math.max(v,moFloor));
      const sum = fl.reduce((a,b)=>a+b,0);
      predGrid[y][x] = fl.map(v=>v/sum);
    }
  }

  return predGrid;
}

module.exports = { predict };
