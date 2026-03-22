#!/usr/bin/env node
/**
 * Quick LOO validation using GT data directly.
 * No replays needed - uses GT probability distributions as training data.
 * Tests different feature functions, weights, and crossWeights.
 */
const fs = require('fs');
const path = require('path');
const H = 40, W = 40, SEEDS = 5, C = 6;
const DATA_DIR = path.join(__dirname, 'data');
const log = m => { const t = new Date().toISOString().slice(11, 19); console.log(`[${t}] ${m}`); };

// ═══ FEATURE FUNCTIONS ═══
function cf(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS=0,co=0,fN=0,sR2=0,mN=0,pN=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
    if(!dy&&!dx)continue;
    const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;
    const nt=g[ny][nx];
    if(nt===1||nt===2)nS++;
    if(nt===10)co=1;
    if(nt===4)fN++;
    if(nt===5)mN++;
    if(nt===2)pN++;
  }
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){
    if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;
    const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;
    if(g[ny][nx]===1||g[ny][nx]===2)sR2++;
  }
  const sa=Math.min(nS,5);
  const sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3;
  const fb=fN<=1?0:fN<=3?1:2;
  const mb=mN>0?1:0;
  const pb=pN>0?1:0;
  const edge=(y<=0||y>=H-1||x<=0||x>=W-1)?1:0;

  return [
    `D0_${t}_${sa}_${co}_${sb2}_${fb}`,           // 0: Original D0
    `D0m_${t}_${sa}_${co}_${sb2}_${fb}_${mb}`,     // 1: D0 + mountain adj
    `D0p_${t}_${sa}_${co}_${sb2}_${fb}_${pb}`,     // 2: D0 + port adj
    `D0mp_${t}_${sa}_${co}_${sb2}_${fb}_${mb}_${pb}`, // 3: D0 + mountain + port
    `D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,      // 4: Standard D1
    `D2_${t}_${sa>0?1:0}_${co}`,                   // 5: Standard D2
    `D3_${t}_${co}`,                                // 6: Standard D3
    `D4_${t}`                                       // 7: Just terrain
  ];
}

// ═══ MODEL BUILDING FROM GT ═══
// Uses GT probabilities directly as soft counts (equivalent to infinite replays)
function buildModelFromGT(gts, inits, roundNames, featIdx=0, alpha=0.05) {
  const m = {};
  for (const rn of roundNames) {
    if (!gts[rn] || !inits[rn]) continue;
    for (let si = 0; si < SEEDS; si++) {
      if (!inits[rn][si] || !gts[rn][si]) continue;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cf(inits[rn][si], y, x); if (!keys) continue;
        const k = keys[featIdx];
        if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
        const p = gts[rn][si][y][x];
        for (let c = 0; c < C; c++) m[k].counts[c] += p[c];
        m[k].n++;
      }
    }
  }
  for (const k of Object.keys(m)) {
    const total = Array.from(m[k].counts).reduce((a,b) => a+b, 0) + C * alpha;
    m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / total);
  }
  return m;
}

// Build from replays (for comparison)
function buildModelFromReplays(replays, inits, roundNames, featIdx=0, alpha=0.05) {
  const m = {};
  function t2c(t) { return (t===10||t===11||t===0)?0:(t>=1&&t<=5)?t:0; }
  for (const rn of roundNames) {
    if (!replays[rn] || !inits[rn]) continue;
    for (const rep of replays[rn]) {
      const initGrid = inits[rn][rep.si];
      if (!initGrid) continue;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cf(initGrid, y, x); if (!keys) continue;
        const k = keys[featIdx];
        const fc = t2c(rep.finalGrid[y][x]);
        if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
        m[k].n++; m[k].counts[fc]++;
      }
    }
  }
  for (const k of Object.keys(m)) {
    const total = m[k].n + C * alpha;
    m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / total);
  }
  return m;
}

// ═══ PREDICTION ═══
function predictSingle(grid, model, cfg) {
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1,0,0,0,0,0]; continue; }
      if (t === 5) { pred[y][x] = [0,0,0,0,0,1]; continue; }
      const keys = cf(grid, y, x);
      if (!keys) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      const p = [0,0,0,0,0,0]; let wS = 0;
      for (let ki = 0; ki < cfg.keyIndices.length; ki++) {
        const keyIdx = cfg.keyIndices[ki];
        const d = model[keys[keyIdx]];
        if (d && d.n >= cfg.minN) {
          const w = cfg.ws[ki] * Math.pow(d.n, cfg.pow);
          for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w;
        }
      }
      if (wS === 0) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      let s = 0; for (let c = 0; c < C; c++) { p[c] /= wS; if (p[c] < cfg.fl) p[c] = cfg.fl; s += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

// Multi-model prediction: each model uses its own feature key
function predictMulti(grid, models, cfg) {
  const pred = [];
  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1,0,0,0,0,0]; continue; }
      if (t === 5) { pred[y][x] = [0,0,0,0,0,1]; continue; }
      const keys = cf(grid, y, x);
      if (!keys) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      const p = [0,0,0,0,0,0]; let wS = 0;
      for (let ki = 0; ki < models.length; ki++) {
        if (!models[ki]) continue;
        const keyIdx = cfg.keyIndices[ki];
        const d = models[ki][keys[keyIdx]];
        if (d && d.n >= cfg.minN) {
          const w = cfg.ws[ki] * Math.pow(d.n, cfg.pow);
          for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w;
        }
      }
      if (wS === 0) { pred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      let s = 0; for (let c = 0; c < C; c++) { p[c] /= wS; if (p[c] < cfg.fl) p[c] = cfg.fl; s += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

// ═══ SCORING ═══
function computeScore(pred, gt) {
  let totalEntropy = 0, totalWeightedKL = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const p = gt[y][x], q = pred[y][x];
    let entropy = 0;
    for (let c = 0; c < C; c++) if (p[c] > 0.001) entropy -= p[c] * Math.log(p[c]);
    if (entropy < 0.01) continue;
    let kl = 0;
    for (let c = 0; c < C; c++) if (p[c] > 0.001) kl += p[c] * Math.log(p[c] / Math.max(q[c], 1e-10));
    totalEntropy += entropy;
    totalWeightedKL += entropy * kl;
  }
  if (totalEntropy === 0) return 100;
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * totalWeightedKL / totalEntropy)));
}

// ═══ LOO VALIDATION ═══
function runLOO(allGTs, allInits, roundNames, modelBuilder, predictFn, cfg, label) {
  const scores = [];
  for (let holdout = 0; holdout < roundNames.length; holdout++) {
    const testRound = roundNames[holdout];
    const trainRounds = roundNames.filter((_, i) => i !== holdout);

    const model = modelBuilder(allGTs, allInits, trainRounds, cfg);

    const roundScores = [];
    for (let si = 0; si < SEEDS; si++) {
      if (!allInits[testRound] || !allInits[testRound][si]) continue;
      if (!allGTs[testRound] || !allGTs[testRound][si]) continue;
      const p = predictFn(allInits[testRound][si], model, cfg);
      roundScores.push(computeScore(p, allGTs[testRound][si]));
    }
    const avg = roundScores.reduce((a,b)=>a+b,0) / roundScores.length;
    scores.push({ round: testRound, score: avg, seeds: roundScores });
  }
  const overall = scores.reduce((a,b)=>a+b.score,0) / scores.length;
  return { overall, scores, label };
}

// ═══ SIMULATED VIEWPORT LOO ═══
function runLOOWithSimViewport(allGTs, allInits, allReplays, roundNames, cfg, cw) {
  function t2c(t) { return (t===10||t===11||t===0)?0:(t>=1&&t<=5)?t:0; }
  const scores = [];
  for (let holdout = 0; holdout < roundNames.length; holdout++) {
    const testRound = roundNames[holdout];
    const trainRounds = roundNames.filter((_, i) => i !== holdout);

    // Build cross-round model from GT of training rounds
    const crossModel = buildModelFromGT(allGTs, allInits, trainRounds, cfg.keyIndices[0], cfg.alpha || 0.05);

    // Simulate viewport using held-out round's replays
    const reps = allReplays[testRound];
    if (!reps || reps.length < 10) {
      // Fall back to cross-round only
      const roundScores = [];
      for (let si = 0; si < SEEDS; si++) {
        if (!allInits[testRound][si] || !allGTs[testRound][si]) continue;
        const p = predictSingle(allInits[testRound][si], crossModel, cfg);
        roundScores.push(computeScore(p, allGTs[testRound][si]));
      }
      scores.push({ round: testRound, score: roundScores.reduce((a,b)=>a+b,0)/roundScores.length, seeds: roundScores });
      continue;
    }

    // Build viewport model from replays (simulating viewport queries)
    // Use first 50 replays as viewport observations (full grid, not just 15x15)
    const vpReps = reps.slice(0, Math.min(50, reps.length));
    const viewportModel = {};
    const initGrid = allInits[testRound][0];
    for (const rep of vpReps) {
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const keys = cf(initGrid, y, x); if (!keys) continue;
        const k = keys[cfg.keyIndices[0]];
        const fc = t2c(rep.finalGrid[y][x]);
        if (!viewportModel[k]) viewportModel[k] = { n: 0, counts: new Float64Array(C) };
        viewportModel[k].n++; viewportModel[k].counts[fc]++;
      }
    }
    // Normalize viewport model
    for (const k of Object.keys(viewportModel)) {
      const total = viewportModel[k].n + C * 0.1;
      viewportModel[k].a = Array.from(viewportModel[k].counts).map(v => (v + 0.1) / total);
    }

    // Fuse
    const fusedModel = {};
    const allKeys = new Set([...Object.keys(crossModel), ...Object.keys(viewportModel)]);
    for (const k of allKeys) {
      const cm = crossModel[k]; const vm = viewportModel[k];
      if (cm && vm) {
        const prior = cm.a.map(p => p * cw);
        const post = prior.map((a, c) => a + vm.counts[c]);
        const total = post.reduce((a,b)=>a+b, 0);
        fusedModel[k] = { n: cm.n + vm.n, a: post.map(v => v / total) };
      } else if (vm) { fusedModel[k] = { n: vm.n, a: vm.a.slice() }; }
      else { fusedModel[k] = { n: cm.n, a: cm.a.slice() }; }
    }

    const roundScores = [];
    for (let si = 0; si < SEEDS; si++) {
      if (!allInits[testRound][si] || !allGTs[testRound][si]) continue;
      const p = predictSingle(allInits[testRound][si], fusedModel, cfg);
      roundScores.push(computeScore(p, allGTs[testRound][si]));
    }
    scores.push({ round: testRound, score: roundScores.reduce((a,b)=>a+b,0)/roundScores.length, seeds: roundScores });
  }
  const overall = scores.reduce((a,b)=>a+b.score,0) / scores.length;
  return { overall, scores };
}

// ═══ MAIN ═══
async function main() {
  log('╔══════════════════════════════════════════════════╗');
  log('║  Quick LOO Validation (GT-based)                 ║');
  log('╚══════════════════════════════════════════════════╝');

  // Load all data
  const allGTs = {}, allInits = {}, allReplays = {};
  const availableRounds = [];
  for (let r = 1; r <= 10; r++) {
    const rn = `R${r}`;
    const initFile = path.join(DATA_DIR, `inits_${rn}.json`);
    const gtFile = path.join(DATA_DIR, `gt_${rn}.json`);
    if (fs.existsSync(initFile) && fs.existsSync(gtFile)) {
      allInits[rn] = JSON.parse(fs.readFileSync(initFile));
      allGTs[rn] = JSON.parse(fs.readFileSync(gtFile));
      availableRounds.push(rn);
      // Try loading replays
      const repFile = path.join(DATA_DIR, `replays_${rn}.json`);
      if (fs.existsSync(repFile)) {
        allReplays[rn] = JSON.parse(fs.readFileSync(repFile));
      }
    }
  }
  log(`Loaded ${availableRounds.length} rounds: ${availableRounds.join(', ')}`);
  log(`Replays: ${Object.entries(allReplays).map(([k,v]) => `${k}=${v.length}`).join(', ') || 'none'}`);

  // ═══ PHASE 1: Base feature comparison ═══
  log('\n═══ PHASE 1: Feature function comparison (GT-based model) ═══');

  // Configs to test
  const growthRounds = availableRounds.filter(r => r !== 'R3'); // Exclude death round
  const allRoundsIncR3 = [...availableRounds];

  const baseCfg = { pow: 0.5, minN: 2, fl: 0.00005, alpha: 0.05 };

  const featureConfigs = [
    // Original D0 hierarchy (indices 0, 4, 5, 6, 7)
    { ...baseCfg, keyIndices: [0, 4, 5, 6, 7], ws: [1, 0.3, 0.15, 0.08, 0.02], label: 'D0→D4 (original)' },
    // D0 + mountain (indices 1, 4, 5, 6, 7)
    { ...baseCfg, keyIndices: [1, 4, 5, 6, 7], ws: [1, 0.3, 0.15, 0.08, 0.02], label: 'D0m→D4 (+ mountain)' },
    // D0 + port (indices 2, 4, 5, 6, 7)
    { ...baseCfg, keyIndices: [2, 4, 5, 6, 7], ws: [1, 0.3, 0.15, 0.08, 0.02], label: 'D0p→D4 (+ port)' },
    // D0 + mountain + port (indices 3, 4, 5, 6, 7)
    { ...baseCfg, keyIndices: [3, 4, 5, 6, 7], ws: [1, 0.3, 0.15, 0.08, 0.02], label: 'D0mp→D4 (+ mtn+port)' },
    // D0 only (no fallback)
    { ...baseCfg, keyIndices: [0], ws: [1], label: 'D0 only' },
    // D1 only
    { ...baseCfg, keyIndices: [4], ws: [1], label: 'D1 only' },
    // D0 + D1 only
    { ...baseCfg, keyIndices: [0, 4], ws: [1, 0.3], label: 'D0+D1 only' },
    // Higher fallback weights
    { ...baseCfg, keyIndices: [0, 4, 5, 6, 7], ws: [1, 0.5, 0.25, 0.12, 0.05], label: 'D0→D4 high-fallback' },
    // Lower fallback weights
    { ...baseCfg, keyIndices: [0, 4, 5, 6, 7], ws: [1, 0.15, 0.08, 0.04, 0.01], label: 'D0→D4 low-fallback' },
  ];

  const results1 = [];
  for (const cfg of featureConfigs) {
    const result = runLOO(allGTs, allInits, growthRounds,
      (gts, inits, trains, c) => buildModelFromGT(gts, inits, trains, c.keyIndices[0], c.alpha),
      predictSingle, cfg, cfg.label);
    results1.push({ ...result, cfg });
  }

  results1.sort((a, b) => b.overall - a.overall);
  log('\nResults (excluding R3):');
  for (const r of results1) {
    const breakdown = r.scores.map(s => `${s.round}=${s.score.toFixed(1)}`).join(', ');
    log(`  ${r.overall.toFixed(2)} : ${r.label}  [${breakdown}]`);
  }

  // Also test with R3 included
  log('\nWith R3 included:');
  try {
    const bestCfgForR3 = { ...results1[0].cfg };
    const withR3 = runLOO(allGTs, allInits, allRoundsIncR3,
      (gts, inits, trains, c) => buildModelFromGT(gts, inits, trains, c.keyIndices[0], c.alpha),
      predictSingle, bestCfgForR3, 'Best + R3');
    log(`  ${withR3.overall.toFixed(2)} : ${withR3.label}`);
    const breakdown = withR3.scores.map(s => `${s.round}=${s.score.toFixed(1)}`).join(', ');
    log(`  [${breakdown}]`);
  } catch(e) { log(`  R3 test failed: ${e.message}`); }

  // ═══ PHASE 2: Parameter tuning ═══
  log('\n═══ PHASE 2: Parameter tuning (best feature) ═══');
  const bestFeatCfg = results1[0].cfg;
  const paramConfigs = [
    // Alpha variations
    { ...bestFeatCfg, alpha: 0.01, label: 'alpha=0.01' },
    { ...bestFeatCfg, alpha: 0.02, label: 'alpha=0.02' },
    { ...bestFeatCfg, alpha: 0.05, label: 'alpha=0.05 (base)' },
    { ...bestFeatCfg, alpha: 0.1, label: 'alpha=0.1' },
    { ...bestFeatCfg, alpha: 0.2, label: 'alpha=0.2' },
    // Power variations
    { ...bestFeatCfg, pow: 0.3, label: 'pow=0.3' },
    { ...bestFeatCfg, pow: 0.5, label: 'pow=0.5 (base)' },
    { ...bestFeatCfg, pow: 0.7, label: 'pow=0.7' },
    { ...bestFeatCfg, pow: 1.0, label: 'pow=1.0' },
    // minN variations
    { ...bestFeatCfg, minN: 1, label: 'minN=1' },
    { ...bestFeatCfg, minN: 2, label: 'minN=2 (base)' },
    { ...bestFeatCfg, minN: 3, label: 'minN=3' },
    { ...bestFeatCfg, minN: 5, label: 'minN=5' },
    // Floor variations
    { ...bestFeatCfg, fl: 0.0001, label: 'fl=0.0001' },
    { ...bestFeatCfg, fl: 0.00005, label: 'fl=0.00005 (base)' },
    { ...bestFeatCfg, fl: 0.00001, label: 'fl=0.00001' },
    { ...bestFeatCfg, fl: 0.000001, label: 'fl=0.000001' },
  ];

  const results2 = [];
  for (const cfg of paramConfigs) {
    const result = runLOO(allGTs, allInits, growthRounds,
      (gts, inits, trains, c) => buildModelFromGT(gts, inits, trains, c.keyIndices[0], c.alpha),
      predictSingle, cfg, cfg.label);
    results2.push({ ...result, cfg });
  }

  results2.sort((a, b) => b.overall - a.overall);
  log('\nParameter tuning results:');
  for (const r of results2.slice(0, 10)) {
    log(`  ${r.overall.toFixed(2)} : ${r.label}`);
  }

  // ═══ PHASE 3: Cross-weight search (if replays available) ═══
  const roundsWithReplays = Object.keys(allReplays).filter(rn => allReplays[rn].length >= 10);
  if (roundsWithReplays.length >= 2) {
    log('\n═══ PHASE 3: Cross-weight search (with simulated viewport) ═══');
    const bestParams = results2[0].cfg;
    const crossWeights = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100];
    const cwResults = [];

    for (const cw of crossWeights) {
      const testRounds = growthRounds.filter(r => roundsWithReplays.includes(r));
      const result = runLOOWithSimViewport(allGTs, allInits, allReplays, testRounds, bestParams, cw);
      cwResults.push({ cw, overall: result.overall, scores: result.scores });
      const breakdown = result.scores.map(s => `${s.round}=${s.score.toFixed(1)}`).join(', ');
      log(`  cw=${cw}: ${result.overall.toFixed(2)}  [${breakdown}]`);
    }

    cwResults.sort((a, b) => b.overall - a.overall);
    log('\nBest crossWeights:');
    for (const r of cwResults.slice(0, 5)) {
      log(`  cw=${r.cw}: ${r.overall.toFixed(2)}`);
    }
  } else {
    log('\nSkipping cross-weight search (need replays for at least 2 rounds)');
    log(`Currently have replays for: ${roundsWithReplays.join(', ') || 'none'}`);
  }

  // ═══ PHASE 4: GT vs Replay model comparison ═══
  if (roundsWithReplays.length >= 2) {
    log('\n═══ PHASE 4: GT-based vs Replay-based model ═══');
    const bestCfg = results2[0];
    const replayRounds = growthRounds.filter(r => roundsWithReplays.includes(r));

    const bestCfg4 = results2[0].cfg;
    // GT-based LOO
    const gtResult = runLOO(allGTs, allInits, replayRounds,
      (gts, inits, trains, c) => buildModelFromGT(gts, inits, trains, c.keyIndices[0], c.alpha),
      predictSingle, bestCfg4, 'GT-based');

    // Replay-based LOO
    const repResult = runLOO(allGTs, allInits, replayRounds,
      (gts, inits, trains, c) => buildModelFromReplays(allReplays, inits, trains, c.keyIndices[0], c.alpha),
      predictSingle, bestCfg4, 'Replay-based');

    log(`  GT-based:     ${gtResult.overall.toFixed(2)}`);
    log(`  Replay-based: ${repResult.overall.toFixed(2)}`);
  }

  // ═══ SUMMARY ═══
  log('\n╔══════════════════════════════════════════════════╗');
  log('║  VALIDATION SUMMARY                              ║');
  log('╚══════════════════════════════════════════════════╝');
  log(`Best feature: ${results1[0].label} (LOO=${results1[0].overall.toFixed(2)})`);
  log(`Best params:  ${results2[0].label} (LOO=${results2[0].overall.toFixed(2)})`);
  log(`Best config breakdown:`);
  for (const s of results2[0].scores) {
    log(`  ${s.round}: ${s.score.toFixed(2)} [${s.seeds.map(v => v.toFixed(1)).join(', ')}]`);
  }

  // Save best config
  const bestConfig = { ...results2[0].cfg };
  fs.writeFileSync(path.join(DATA_DIR, 'best_config.json'), JSON.stringify(bestConfig, null, 2));
  log('\nBest config saved to data/best_config.json');
}

main().catch(e => { console.error('Fatal:', e.message, e.stack); process.exit(1); });
