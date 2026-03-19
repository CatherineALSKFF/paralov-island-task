/**
 * Mega Solver — All-in-one Astar Island prediction engine.
 * Paste into browser console on app.ainm.no (must be logged in).
 *
 * This script:
 * 1. Fetches round data
 * 2. Builds Norse civilization simulator
 * 3. Runs per-regime Monte Carlo (7 regimes × 5 seeds × 80 sims)
 * 4. Finds optimal blend weights via minimax
 * 5. Generates final predictions
 * 6. Stores everything in window._mega for inspection
 *
 * Does NOT submit — call window._mega.submitAll() when ready.
 */
(async function MegaSolver() {
  const BASE = 'https://api.ainm.no';
  const FLOOR = 0.01;
  const MS = (s) => `${((Date.now() - _t0) / 1000).toFixed(1)}s`;
  const _t0 = Date.now();

  if (!window._mega) window._mega = {};
  const M = window._mega;
  function log(msg) { console.log(`[MEGA ${MS()}] ${msg}`); }

  // ═══════════════════════════════════════════════════════════════════
  // PHASE 1: FETCH ROUND DATA
  // ═══════════════════════════════════════════════════════════════════
  log('Phase 1: Fetching round data...');
  const roundsResp = await fetch(`${BASE}/astar-island/rounds`, {credentials:'include'});
  const rounds = await roundsResp.json();
  const active = rounds.find(r => r.status === 'active');
  if (!active) { log('No active round!'); return; }

  const ROUND_ID = active.id;
  log(`Round: ${active.round_number} (${ROUND_ID})`);

  const detResp = await fetch(`${BASE}/astar-island/rounds/${ROUND_ID}`, {credentials:'include'});
  const detail = await detResp.json();
  M.detail = detail;
  const H = detail.map_height, W = detail.map_width, SEEDS = detail.seeds_count;
  log(`Map: ${W}x${H}, Seeds: ${SEEDS}, Closes: ${detail.closes_at}`);

  const budgetResp = await fetch(`${BASE}/astar-island/budget`, {credentials:'include'});
  const budget = await budgetResp.json();
  log(`Budget: ${budget.queries_used}/${budget.queries_max}`);

  // ═══════════════════════════════════════════════════════════════════
  // PHASE 2: BUILD SIMULATOR
  // ═══════════════════════════════════════════════════════════════════
  log('Phase 2: Building simulator...');

  // Mulberry32 seeded RNG
  function makeRng(seed) {
    let t = seed | 0;
    return function() {
      t = (t + 0x6D2B79F5) | 0;
      let x = Math.imul(t ^ (t >>> 15), 1 | t);
      x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Settlement class
  class Settlement {
    constructor(x, y, hasPort, alive) {
      this.x = x; this.y = y;
      this.pop = 1.0; this.food = 0.5; this.wealth = 0;
      this.hasPort = !!hasPort; this.alive = alive !== false;
    }
  }

  // 7 parameter regimes: Conservative → Moderate → Aggressive + Ultra-harsh + Ultra-prosperous
  const PARAM_SETS = [
    // 0: Conservative
    { foodForest:0.3, foodPlains:0.08, growthTh:1.8, expandTh:4, expandPopTh:2,
      expandChance:0.04, expandDist:3, portChance:0.06, longshipChance:0.05,
      raidRange:3, longRaidRange:7, raidChance:0.15, despRaid:0.2, raidStr:0.5,
      loot:0.3, conquerChance:0.08, tradeRange:5, tradeFood:0.2, tradeWealth:0.15,
      winterBase:1, winterVar:0.6, collapseTh:-0.5, collapseChance:0.4,
      forestReclaim:0.08, ruinDecay:0.05, rebuildChance:0.05, rebuildRange:3 },
    // 1: Mild-conservative
    { foodForest:0.35, foodPlains:0.1, growthTh:1.5, expandTh:3.5, expandPopTh:1.8,
      expandChance:0.06, expandDist:3, portChance:0.08, longshipChance:0.06,
      raidRange:3, longRaidRange:7, raidChance:0.18, despRaid:0.22, raidStr:0.5,
      loot:0.3, conquerChance:0.1, tradeRange:5, tradeFood:0.25, tradeWealth:0.18,
      winterBase:0.9, winterVar:0.5, collapseTh:-0.6, collapseChance:0.35,
      forestReclaim:0.07, ruinDecay:0.04, rebuildChance:0.06, rebuildRange:3 },
    // 2: Balanced
    { foodForest:0.4, foodPlains:0.12, growthTh:1.3, expandTh:3, expandPopTh:1.5,
      expandChance:0.08, expandDist:3, portChance:0.1, longshipChance:0.08,
      raidRange:4, longRaidRange:8, raidChance:0.2, despRaid:0.25, raidStr:0.55,
      loot:0.35, conquerChance:0.12, tradeRange:6, tradeFood:0.3, tradeWealth:0.2,
      winterBase:0.85, winterVar:0.45, collapseTh:-0.8, collapseChance:0.3,
      forestReclaim:0.06, ruinDecay:0.03, rebuildChance:0.07, rebuildRange:3 },
    // 3: Moderate-aggressive
    { foodForest:0.45, foodPlains:0.14, growthTh:1.1, expandTh:2.5, expandPopTh:1.3,
      expandChance:0.1, expandDist:4, portChance:0.12, longshipChance:0.1,
      raidRange:4, longRaidRange:9, raidChance:0.22, despRaid:0.28, raidStr:0.6,
      loot:0.4, conquerChance:0.15, tradeRange:6, tradeFood:0.35, tradeWealth:0.25,
      winterBase:0.75, winterVar:0.4, collapseTh:-1, collapseChance:0.25,
      forestReclaim:0.05, ruinDecay:0.025, rebuildChance:0.08, rebuildRange:4 },
    // 4: Aggressive
    { foodForest:0.5, foodPlains:0.16, growthTh:0.9, expandTh:2, expandPopTh:1,
      expandChance:0.12, expandDist:4, portChance:0.15, longshipChance:0.12,
      raidRange:5, longRaidRange:10, raidChance:0.25, despRaid:0.3, raidStr:0.65,
      loot:0.45, conquerChance:0.18, tradeRange:7, tradeFood:0.4, tradeWealth:0.3,
      winterBase:0.65, winterVar:0.35, collapseTh:-1.2, collapseChance:0.2,
      forestReclaim:0.04, ruinDecay:0.02, rebuildChance:0.1, rebuildRange:4 },
    // 5: Ultra-harsh (most die)
    { foodForest:0.2, foodPlains:0.05, growthTh:2.5, expandTh:5, expandPopTh:3,
      expandChance:0.02, expandDist:2, portChance:0.03, longshipChance:0.03,
      raidRange:4, longRaidRange:8, raidChance:0.25, despRaid:0.3, raidStr:0.7,
      loot:0.4, conquerChance:0.12, tradeRange:4, tradeFood:0.15, tradeWealth:0.1,
      winterBase:1.5, winterVar:0.8, collapseTh:-0.3, collapseChance:0.55,
      forestReclaim:0.12, ruinDecay:0.08, rebuildChance:0.03, rebuildRange:2 },
    // 6: Ultra-prosperous (most thrive)
    { foodForest:0.5, foodPlains:0.18, growthTh:1.0, expandTh:2, expandPopTh:1.2,
      expandChance:0.12, expandDist:4, portChance:0.12, longshipChance:0.1,
      raidRange:2, longRaidRange:5, raidChance:0.08, despRaid:0.1, raidStr:0.3,
      loot:0.2, conquerChance:0.05, tradeRange:6, tradeFood:0.35, tradeWealth:0.25,
      winterBase:0.6, winterVar:0.3, collapseTh:-1.5, collapseChance:0.15,
      forestReclaim:0.04, ruinDecay:0.02, rebuildChance:0.1, rebuildRange:4 },
  ];

  // Core simulation: runs 50 years of Norse civilization
  function simulate(initialGrid, initialSettlements, rng, P) {
    const grid = initialGrid.map(r => [...r]);
    const settles = initialSettlements.map(s =>
      new Settlement(s.x, s.y, s.has_port, s.alive));

    for (let year = 0; year < 50; year++) {
      // === GROWTH PHASE ===
      for (const s of settles) {
        if (!s.alive) continue;
        let foodGain = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            const ny = s.y+dy, nx = s.x+dx;
            if (ny<0||ny>=H||nx<0||nx>=W) continue;
            const t = grid[ny][nx];
            if (t === 4) foodGain += P.foodForest;
            else if (t === 11 || t === 0) foodGain += P.foodPlains;
          }
        }
        s.food += foodGain;
        if (s.food > P.growthTh) {
          s.pop += 0.1 * (1 + s.wealth * 0.05);
          s.food -= P.growthTh * 0.5;
        }

        // Expand
        if (s.pop >= P.expandPopTh && s.food > P.expandTh && rng() < P.expandChance) {
          const candidates = [];
          for (let dy = -P.expandDist; dy <= P.expandDist; dy++) {
            for (let dx = -P.expandDist; dx <= P.expandDist; dx++) {
              if (dy===0 && dx===0) continue;
              const ny = s.y+dy, nx = s.x+dx;
              if (ny<0||ny>=H||nx<0||nx>=W) continue;
              const t = grid[ny][nx];
              if (t===0 || t===11 || t===4) {
                // Check spacing: no settlement within 2 tiles
                let tooClose = false;
                for (const o of settles) {
                  if (!o.alive) continue;
                  if (Math.abs(o.x-nx)+Math.abs(o.y-ny) < 2) { tooClose=true; break; }
                }
                if (!tooClose) candidates.push({x:nx,y:ny});
              }
            }
          }
          if (candidates.length > 0) {
            const c = candidates[Math.floor(rng()*candidates.length)];
            grid[c.y][c.x] = 1;
            const ns = new Settlement(c.x, c.y, false, true);
            ns.pop = 0.5; ns.food = s.food * 0.3;
            settles.push(ns);
            s.food *= 0.5; s.pop *= 0.8;
          }
        }

        // Port upgrade
        if (!s.hasPort && rng() < P.portChance) {
          for (const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
            const ny=s.y+dy, nx=s.x+dx;
            if (ny>=0&&ny<H&&nx>=0&&nx<W && grid[ny][nx]===10) {
              s.hasPort = true; grid[s.y][s.x] = 2;
              break;
            }
          }
        }
      }

      // === CONFLICT PHASE ===
      const alive = settles.filter(s => s.alive);
      for (const attacker of alive) {
        if (!attacker.alive) continue;
        const range = (attacker.hasPort && rng() < P.longshipChance)
          ? P.longRaidRange : P.raidRange;
        if (rng() < (attacker.food < 0.3 ? P.despRaid : P.raidChance)) {
          const targets = alive.filter(t =>
            t !== attacker && t.alive &&
            Math.abs(t.x-attacker.x)+Math.abs(t.y-attacker.y) <= range);
          if (targets.length > 0) {
            const target = targets[Math.floor(rng()*targets.length)];
            const atkPow = attacker.pop * P.raidStr * (1 + attacker.wealth*0.05);
            const defPow = target.pop * (1 + target.wealth*0.03);
            if (atkPow > defPow * (0.8 + rng()*0.4)) {
              const stolen = target.food * P.loot;
              attacker.food += stolen; target.food -= stolen;
              attacker.wealth += target.wealth * P.loot * 0.5;
              target.wealth = Math.max(0, target.wealth - target.wealth * P.loot * 0.5);
              if (rng() < P.conquerChance) {
                target.alive = false;
                grid[target.y][target.x] = 3; // ruin
              }
            }
          }
        }
      }

      // === TRADE PHASE ===
      for (const s of settles) {
        if (!s.alive || !s.hasPort) continue;
        const partners = settles.filter(t =>
          t !== s && t.alive && t.hasPort &&
          Math.abs(t.x-s.x)+Math.abs(t.y-s.y) <= P.tradeRange);
        for (const p of partners) {
          s.food += P.tradeFood * 0.5;
          p.food += P.tradeFood * 0.5;
          s.wealth += P.tradeWealth;
          p.wealth += P.tradeWealth;
        }
      }

      // === WINTER PHASE ===
      const severity = P.winterBase + (rng()-0.5)*P.winterVar;
      for (const s of settles) {
        if (!s.alive) continue;
        s.food -= severity * (0.8 + s.pop * 0.2);
        s.pop = Math.max(0.1, s.pop - severity * 0.05);
        if (s.food < P.collapseTh && rng() < P.collapseChance) {
          s.alive = false;
          grid[s.y][s.x] = 3; // ruin
        }
      }

      // === ENVIRONMENT PHASE ===
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          if (grid[y][x] === 3) {
            // Ruin → forest reclaim
            if (rng() < P.forestReclaim) grid[y][x] = 4;
            // Ruin → rebuild
            const nearAlive = settles.some(s => s.alive &&
              Math.abs(s.x-x)+Math.abs(s.y-y) <= P.rebuildRange);
            if (nearAlive && rng() < P.rebuildChance) {
              grid[y][x] = 1;
              settles.push(new Settlement(x, y, false, true));
            }
          }
          // Empty near ruin → forest growth
          if ((grid[y][x]===0 || grid[y][x]===11) && rng() < P.ruinDecay) {
            let nearRuin = false;
            for (const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
              const ny=y+dy, nx=x+dx;
              if (ny>=0&&ny<H&&nx>=0&&nx<W && grid[ny][nx]===3) nearRuin=true;
            }
            if (nearRuin) grid[y][x] = 4;
          }
        }
      }
    }
    return grid;
  }

  M.simulate = simulate;
  M.makeRng = makeRng;
  M.PARAM_SETS = PARAM_SETS;
  log('Simulator built (7 param regimes).');

  // ═══════════════════════════════════════════════════════════════════
  // PHASE 3: BUILD SCORER
  // ═══════════════════════════════════════════════════════════════════
  log('Phase 3: Building scorer...');

  function computeScore(prediction, groundTruth) {
    let totalWeight = 0, totalKL = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const p = groundTruth[y][x];
        const q = prediction[y][x];
        // Entropy of ground truth
        let entropy = 0;
        for (let c = 0; c < 6; c++) {
          if (p[c] > 0) entropy -= p[c] * Math.log2(p[c]);
        }
        if (entropy < 0.001) continue; // skip near-static cells
        // KL divergence
        let kl = 0;
        for (let c = 0; c < 6; c++) {
          if (p[c] > 0) {
            const qi = Math.max(q[c], 1e-10);
            kl += p[c] * Math.log(p[c] / qi);
          }
        }
        totalKL += entropy * kl;
        totalWeight += entropy;
      }
    }
    const wkl = totalWeight > 0 ? totalKL / totalWeight : 0;
    return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl)));
  }

  function countsToDistribution(counts, nSims) {
    const alpha = 1.0; // Dirichlet smoothing
    const dist = [];
    for (let y = 0; y < H; y++) {
      dist[y] = [];
      for (let x = 0; x < W; x++) {
        const probs = new Array(6);
        let sum = 0;
        for (let c = 0; c < 6; c++) {
          probs[c] = counts[y][x][c] + alpha;
          sum += probs[c];
        }
        for (let c = 0; c < 6; c++) {
          probs[c] = Math.max(FLOOR, probs[c] / sum);
        }
        // Renormalize
        let s2 = 0;
        for (let c = 0; c < 6; c++) s2 += probs[c];
        for (let c = 0; c < 6; c++) probs[c] = parseFloat((probs[c]/s2).toFixed(4));
        // Fix rounding
        let diff = 1.0 - probs.reduce((a,b)=>a+b,0);
        probs[0] = parseFloat((probs[0]+diff).toFixed(4));
        dist[y][x] = probs;
      }
    }
    return dist;
  }

  M.computeScore = computeScore;
  M.countsToDistribution = countsToDistribution;
  log('Scorer built.');

  // ═══════════════════════════════════════════════════════════════════
  // PHASE 4: RUN PER-REGIME MONTE CARLO
  // ═══════════════════════════════════════════════════════════════════
  log('Phase 4: Running per-regime Monte Carlo...');
  const NR = PARAM_SETS.length; // 7
  const NSIM = 80; // per regime per seed

  M.regimeCounts = {};
  M.regimeDistributions = {};
  let totalSims = 0;

  for (let r = 0; r < NR; r++) {
    M.regimeCounts[r] = {};
    const P = PARAM_SETS[r];

    for (let s = 0; s < SEEDS; s++) {
      const counts = [];
      for (let y = 0; y < H; y++) {
        counts[y] = [];
        for (let x = 0; x < W; x++) {
          counts[y][x] = new Float32Array(6);
        }
      }

      const initState = detail.initial_states[s];

      for (let i = 0; i < NSIM; i++) {
        const rng = makeRng(s * 100000 + r * 10000 + i);
        const gCopy = initState.grid.map(row => [...row]);
        const sCopy = initState.settlements.map(st => ({...st}));
        const finalGrid = simulate(gCopy, sCopy, rng, P);

        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const t = finalGrid[y][x];
            let c;
            if (t===10||t===11||t===0) c=0;
            else if (t===1) c=1;
            else if (t===2) c=2;
            else if (t===3) c=3;
            else if (t===4) c=4;
            else if (t===5) c=5;
            else c=0;
            counts[y][x][c]++;
          }
        }
        totalSims++;
      }
      M.regimeCounts[r][s] = counts;
    }
    // Convert to distributions
    M.regimeDistributions[r] = {};
    for (let s = 0; s < SEEDS; s++) {
      M.regimeDistributions[r][s] = countsToDistribution(M.regimeCounts[r][s], NSIM);
    }
    log(`  Regime ${r} done (${totalSims} sims so far)`);
    await new Promise(ok => setTimeout(ok, 0)); // yield
  }
  log(`Phase 4 complete: ${totalSims} total simulations.`);

  // ═══════════════════════════════════════════════════════════════════
  // PHASE 5: BLEND OPTIMIZATION (minimax across regimes)
  // ═══════════════════════════════════════════════════════════════════
  log('Phase 5: Finding optimal blend weights...');

  function blendPredictions(weights, seed) {
    const wSum = weights.reduce((a,b)=>a+b, 0);
    const pred = [];
    for (let y = 0; y < H; y++) {
      pred[y] = [];
      for (let x = 0; x < W; x++) {
        const probs = new Array(6).fill(0);
        for (let r = 0; r < NR; r++) {
          const w = weights[r] / wSum;
          const rp = M.regimeDistributions[r][seed][y][x];
          for (let c = 0; c < 6; c++) {
            probs[c] += w * rp[c];
          }
        }
        // Apply floor and normalize
        for (let c = 0; c < 6; c++) probs[c] = Math.max(FLOOR, probs[c]);
        let s = 0;
        for (let c = 0; c < 6; c++) s += probs[c];
        for (let c = 0; c < 6; c++) probs[c] = parseFloat((probs[c]/s).toFixed(4));
        let diff = 1.0 - probs.reduce((a,b)=>a+b,0);
        probs[0] = parseFloat((probs[0]+diff).toFixed(4));
        pred[y][x] = probs;
      }
    }
    return pred;
  }

  // Test each regime as if it were the "ground truth" — score our blend against it
  function evaluateBlend(weights) {
    const scores = {}; // scores[seed][gtRegime] = score
    let worstOverall = 100;
    let sumOverall = 0;
    let count = 0;

    for (let s = 0; s < SEEDS; s++) {
      scores[s] = [];
      const pred = blendPredictions(weights, s);
      for (let gt = 0; gt < NR; gt++) {
        const gtDist = M.regimeDistributions[gt][s];
        const sc = computeScore(pred, gtDist);
        scores[s].push(sc);
        if (sc < worstOverall) worstOverall = sc;
        sumOverall += sc;
        count++;
      }
    }
    return {
      scores,
      worst: worstOverall,
      avg: sumOverall / count
    };
  }

  // Test many weight combinations
  const blendTests = [
    { name: 'Equal',          w: [1,1,1,1,1,1,1] },
    { name: 'Con-heavy',      w: [5,3,2,1,1,2,1] },
    { name: 'Con-heavy-v2',   w: [4,3,2,1,1,3,1] },
    { name: 'Balanced-center',w: [2,3,4,3,2,1,1] },
    { name: 'Center-heavy',   w: [1,2,5,2,1,1,1] },
    { name: 'MildCon-heavy',  w: [3,5,3,2,1,2,1] },
    { name: 'MildCon-v2',     w: [2,5,4,2,1,1,1] },
    { name: 'Spread-con',     w: [4,4,3,2,1,2,1] },
    { name: 'V-shaped',       w: [4,2,1,1,2,4,1] },
    { name: 'Harsh-hedge',    w: [3,2,2,1,1,4,1] },
    { name: 'Ultra-flat',     w: [2,2,2,2,2,2,2] },
    { name: 'Con-dominant',   w: [8,3,2,1,1,2,1] },
    { name: 'Two-peak',       w: [5,2,1,1,2,5,1] },
    { name: 'Gentle-con',     w: [3,3,2,2,1,1,1] },
  ];

  M.blendResults = [];
  for (const bt of blendTests) {
    const result = evaluateBlend(bt.w);
    M.blendResults.push({ ...bt, ...result });
    log(`  ${bt.name.padEnd(18)} AVG=${result.avg.toFixed(1)} WORST=${result.worst.toFixed(1)}`);
  }

  // Sort by worst-case score (minimax)
  M.blendResults.sort((a,b) => b.worst - a.worst);
  const best = M.blendResults[0];
  log(`Best blend: ${best.name} (WORST=${best.worst.toFixed(1)}, AVG=${best.avg.toFixed(1)})`);

  // ═══════════════════════════════════════════════════════════════════
  // PHASE 6: GENERATE FINAL PREDICTIONS
  // ═══════════════════════════════════════════════════════════════════
  log('Phase 6: Generating final predictions with best blend...');
  M.bestWeights = best.w;
  M.finalPreds = {};

  for (let s = 0; s < SEEDS; s++) {
    M.finalPreds[s] = blendPredictions(best.w, s);
    // Self-score against each regime
    const selfScores = [];
    for (let gt = 0; gt < NR; gt++) {
      selfScores.push(computeScore(M.finalPreds[s], M.regimeDistributions[gt][s]).toFixed(1));
    }
    log(`  Seed ${s} self-scores: [${selfScores.join(', ')}]`);
  }

  // ═══════════════════════════════════════════════════════════════════
  // SUBMISSION FUNCTION (call manually)
  // ═══════════════════════════════════════════════════════════════════
  M.submitAll = async function() {
    log('Submitting all seeds...');
    const results = [];
    for (let s = 0; s < SEEDS; s++) {
      const resp = await fetch(`${BASE}/astar-island/submit`, {
        method: 'POST', credentials: 'include',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          round_id: ROUND_ID,
          seed_index: s,
          prediction: M.finalPreds[s]
        })
      });
      const r = await resp.json();
      results.push({seed:s, status:resp.status, result:r});
      log(`  Seed ${s}: ${resp.status} — ${r.status || JSON.stringify(r)}`);
      await new Promise(ok => setTimeout(ok, 600));
    }
    M.submissions = results;
    log(`Submitted ${results.filter(r=>r.status===200).length}/5 seeds.`);
    return results;
  };

  // Also expose a function to test custom blend weights
  M.testBlend = function(weights) {
    return evaluateBlend(weights);
  };
  M.blendPredictions = blendPredictions;

  log('=== MEGA SOLVER READY ===');
  log(`Best blend: ${best.name} weights=[${best.w}]`);
  log(`Worst-case score: ${best.worst.toFixed(1)}, Average: ${best.avg.toFixed(1)}`);
  log('Call window._mega.submitAll() to submit, or window._mega.testBlend([...]) to test.');
  return M;
})();
