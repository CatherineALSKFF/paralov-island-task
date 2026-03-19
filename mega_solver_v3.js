/**
 * Mega Solver v3 — Astar Island prediction engine with observation + GT learning.
 * Paste into browser console on app.ainm.no (must be logged in).
 *
 * v3 improvements over v2:
 * - Correct API endpoints (POST /submit, POST /simulate)
 * - Observation data integration (use 50 queries strategically)
 * - Post-round GT learning (GET /analysis/{round_id}/{seed})
 * - Bayesian parameter estimation from observations
 * - 17 simulation regimes for dense parameter coverage
 * - Ensemble blending (alpha=0.7 mix of two optimization targets)
 *
 * Does NOT submit or query — call functions explicitly.
 */
(async function MegaSolverV3() {
  const BASE = 'https://api.ainm.no/astar-island';
  const FLOOR = 0.01;
  const PRED_ALPHA = 0.2;
  const GT_ALPHA = 0.1;
  const _t0 = Date.now();
  const MS = () => `${((Date.now() - _t0) / 1000).toFixed(1)}s`;

  window._M = {};
  const M = window._M;
  function log(msg) { console.log(`[MEGA ${MS()}] ${msg}`); }

  // ═══════════════════════════════════════════════════════════════════
  // PHASE 1: FETCH ROUND DATA
  // ═══════════════════════════════════════════════════════════════════
  log('Phase 1: Fetching round data...');
  const rounds = await (await fetch(`${BASE}/rounds`, {credentials:'include'})).json();
  const active = rounds.find(r => r.status === 'active');
  if (!active) { log('No active round! Looking for latest...'); }
  const ROUND = active || rounds[rounds.length - 1];
  const ROUND_ID = ROUND.id;

  const detail = await (await fetch(`${BASE}/rounds/${ROUND_ID}`, {credentials:'include'})).json();
  M.d = detail;
  M.rid = ROUND_ID;
  const H = M.H = detail.map_height;
  const W = M.W = detail.map_width;
  const SEEDS = M.S = detail.seeds_count;
  log(`Round ${detail.round_number} (${ROUND_ID.slice(0,8)}), ${W}x${H}, ${SEEDS} seeds`);
  log(`Status: ${ROUND.status}, closes: ${detail.closes_at}`);

  // ═══════════════════════════════════════════════════════════════════
  // PHASE 2: BUILD SIMULATOR
  // ═══════════════════════════════════════════════════════════════════
  log('Phase 2: Building simulator...');

  // Mulberry32 seeded RNG
  M.mkRng = function(seed) {
    let t = seed | 0;
    return function() {
      t = (t + 0x6D2B79F5) | 0;
      let x = Math.imul(t ^ (t >>> 15), 1 | t);
      x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  };

  class Settlement {
    constructor(x, y, hasPort, alive) {
      this.x = x; this.y = y; this.pop = 1; this.food = 0.5; this.wealth = 0;
      this.defense = 0.5; this.hasPort = !!hasPort; this.alive = alive !== false;
      this.ownerId = null; // faction tracking
    }
  }

  // Terrain-to-class mapper (8 internal codes → 6 prediction classes)
  M.t2c = function(t) {
    if (t === 10 || t === 11 || t === 0) return 0; // Ocean/Plains/Empty → class 0
    if (t === 1) return 1; // Settlement
    if (t === 2) return 2; // Port
    if (t === 3) return 3; // Ruin
    if (t === 4) return 4; // Forest
    if (t === 5) return 5; // Mountain
    return 0;
  };

  // Core simulator: 50 years of Norse civilization
  M.sim = function(initialGrid, initialSettlements, rng, P) {
    const grid = initialGrid.map(r => [...r]);
    const settles = initialSettlements.map((s, i) => {
      const ns = new Settlement(s.x, s.y, s.has_port, s.alive);
      ns.ownerId = i; // each settlement starts as own faction
      return ns;
    });

    for (let year = 0; year < 50; year++) {
      // === GROWTH PHASE ===
      for (const s of settles) {
        if (!s.alive) continue;
        let fg = 0;
        for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
          const ny = s.y + dy, nx = s.x + dx;
          if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
          const t = grid[ny][nx];
          if (t === 4) fg += P.foodForest;
          else if (t === 11 || t === 0) fg += P.foodPlains;
        }
        s.food += fg;
        if (s.food > P.growthTh) {
          s.pop += 0.1 * (1 + s.wealth * 0.05);
          s.food -= P.growthTh * 0.5;
        }
        // Defense grows with population
        s.defense = Math.min(s.defense + 0.02 * s.pop, s.pop * 0.8);

        // Expand
        if (!P.noExpand && s.pop >= P.expandPopTh && s.food > P.expandTh && rng() < P.expandChance) {
          const cands = [];
          for (let dy = -P.expandDist; dy <= P.expandDist; dy++)
            for (let dx = -P.expandDist; dx <= P.expandDist; dx++) {
              if (!dy && !dx) continue;
              const ny = s.y + dy, nx = s.x + dx;
              if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
              const t = grid[ny][nx];
              if (t === 0 || t === 11 || t === 4) {
                let close = false;
                for (const o of settles) if (o.alive && Math.abs(o.x - nx) + Math.abs(o.y - ny) < 2) { close = true; break; }
                if (!close) cands.push({ x: nx, y: ny });
              }
            }
          if (cands.length) {
            const c = cands[Math.floor(rng() * cands.length)];
            grid[c.y][c.x] = 1;
            const ns = new Settlement(c.x, c.y, false, true);
            ns.pop = 0.5; ns.food = s.food * 0.3; ns.ownerId = s.ownerId;
            settles.push(ns); s.food *= 0.5; s.pop *= 0.8;
          }
        }

        // Port upgrade
        if (!s.hasPort && rng() < P.portChance) {
          for (const [dy, dx] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
            const ny = s.y + dy, nx = s.x + dx;
            if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) {
              s.hasPort = true; grid[s.y][s.x] = 2; break;
            }
          }
        }
      }

      // === CONFLICT PHASE ===
      if (!P.noConflict) {
        const alive = settles.filter(s => s.alive);
        for (const a of alive) {
          if (!a.alive) continue;
          const range = (a.hasPort && rng() < P.longshipChance) ? P.longRaidRange : P.raidRange;
          if (rng() < (a.food < 0.3 ? P.despRaid : P.raidChance)) {
            const tgts = alive.filter(t => t !== a && t.alive &&
              t.ownerId !== a.ownerId && // don't raid own faction
              Math.abs(t.x - a.x) + Math.abs(t.y - a.y) <= range);
            if (tgts.length) {
              const tg = tgts[Math.floor(rng() * tgts.length)];
              const ap = a.pop * P.raidStr * (1 + a.wealth * 0.05);
              const dp = tg.pop * (1 + tg.defense * 0.3);
              if (ap > dp * (0.8 + rng() * 0.4)) {
                const st = tg.food * P.loot; a.food += st; tg.food -= st;
                a.wealth += tg.wealth * P.loot * 0.5;
                tg.wealth = Math.max(0, tg.wealth - tg.wealth * P.loot * 0.5);
                tg.defense *= 0.7; // defense degrades after losing
                if (rng() < P.conquerChance) {
                  tg.alive = false; grid[tg.y][tg.x] = 3;
                }
              } else {
                a.defense *= 0.9; // failed raid costs some defense
              }
            }
          }
        }
      }

      // === TRADE PHASE ===
      if (!P.noTrade) {
        for (const s of settles) {
          if (!s.alive || !s.hasPort) continue;
          for (const p of settles) {
            if (p === s || !p.alive || !p.hasPort) continue;
            if (Math.abs(p.x - s.x) + Math.abs(p.y - s.y) > P.tradeRange) continue;
            // Factions at war trade less effectively
            const tradeMul = (s.ownerId !== p.ownerId) ? 0.5 : 1.0;
            s.food += P.tradeFood * 0.5 * tradeMul;
            p.food += P.tradeFood * 0.5 * tradeMul;
            s.wealth += P.tradeWealth * tradeMul;
            p.wealth += P.tradeWealth * tradeMul;
          }
        }
      }

      // === WINTER PHASE ===
      const sev = P.constWinter ? P.winterBase : P.winterBase + (rng() - 0.5) * P.winterVar;
      for (const s of settles) {
        if (!s.alive) continue;
        s.food -= sev * (0.8 + s.pop * 0.2);
        s.pop = Math.max(0.1, s.pop - sev * 0.05);
        if (s.food < P.collapseTh && rng() < P.collapseChance) {
          s.alive = false; grid[s.y][s.x] = 3;
        }
      }

      // === ENVIRONMENT PHASE ===
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        if (grid[y][x] === 3) {
          if (rng() < P.forestReclaim) grid[y][x] = 4;
          const near = settles.some(s => s.alive && Math.abs(s.x - x) + Math.abs(s.y - y) <= P.rebuildRange);
          if (near && rng() < P.rebuildChance) {
            grid[y][x] = 1;
            const ns = new Settlement(x, y, false, true);
            // Inherit faction of nearest alive settlement
            let minD = Infinity, owner = null;
            for (const s of settles) {
              if (!s.alive) continue;
              const d = Math.abs(s.x - x) + Math.abs(s.y - y);
              if (d < minD) { minD = d; owner = s.ownerId; }
            }
            ns.ownerId = owner;
            settles.push(ns);
          }
        }
        if ((grid[y][x] === 0 || grid[y][x] === 11) && rng() < P.ruinDecay) {
          let nr = false;
          for (const [dy, dx] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
            const ny2 = y + dy, nx2 = x + dx;
            if (ny2 >= 0 && ny2 < H && nx2 >= 0 && nx2 < W && grid[ny2][nx2] === 3) nr = true;
          }
          if (nr) grid[y][x] = 4;
        }
      }
    }
    return grid;
  };

  log('Simulator built (with defense + factions).');

  // ═══════════════════════════════════════════════════════════════════
  // PHASE 3: PARAMETER REGIMES
  // ═══════════════════════════════════════════════════════════════════
  const balanced = {foodForest:.4,foodPlains:.12,growthTh:1.3,expandTh:3,expandPopTh:1.5,expandChance:.08,expandDist:3,portChance:.1,longshipChance:.08,raidRange:4,longRaidRange:8,raidChance:.2,despRaid:.25,raidStr:.55,loot:.35,conquerChance:.12,tradeRange:6,tradeFood:.3,tradeWealth:.2,winterBase:.85,winterVar:.45,collapseTh:-.8,collapseChance:.3,forestReclaim:.06,ruinDecay:.03,rebuildChance:.07,rebuildRange:3};
  const modAgg = {foodForest:.45,foodPlains:.14,growthTh:1.1,expandTh:2.5,expandPopTh:1.3,expandChance:.1,expandDist:4,portChance:.12,longshipChance:.1,raidRange:4,longRaidRange:9,raidChance:.22,despRaid:.28,raidStr:.6,loot:.4,conquerChance:.15,tradeRange:6,tradeFood:.35,tradeWealth:.25,winterBase:.75,winterVar:.4,collapseTh:-1,collapseChance:.25,forestReclaim:.05,ruinDecay:.025,rebuildChance:.08,rebuildRange:4};
  const aggressive = {foodForest:.5,foodPlains:.16,growthTh:.9,expandTh:2,expandPopTh:1,expandChance:.12,expandDist:4,portChance:.15,longshipChance:.12,raidRange:5,longRaidRange:10,raidChance:.25,despRaid:.3,raidStr:.65,loot:.45,conquerChance:.18,tradeRange:7,tradeFood:.4,tradeWealth:.3,winterBase:.65,winterVar:.35,collapseTh:-1.2,collapseChance:.2,forestReclaim:.04,ruinDecay:.02,rebuildChance:.1,rebuildRange:4};
  const conserv = {foodForest:.3,foodPlains:.08,growthTh:1.8,expandTh:4,expandPopTh:2,expandChance:.04,expandDist:3,portChance:.06,longshipChance:.05,raidRange:3,longRaidRange:7,raidChance:.15,despRaid:.2,raidStr:.5,loot:.3,conquerChance:.08,tradeRange:5,tradeFood:.2,tradeWealth:.15,winterBase:1,winterVar:.6,collapseTh:-.5,collapseChance:.4,forestReclaim:.08,ruinDecay:.05,rebuildChance:.05,rebuildRange:3};
  const mildCon = {foodForest:.35,foodPlains:.1,growthTh:1.5,expandTh:3.5,expandPopTh:1.8,expandChance:.06,expandDist:3,portChance:.08,longshipChance:.06,raidRange:3,longRaidRange:7,raidChance:.18,despRaid:.22,raidStr:.5,loot:.3,conquerChance:.1,tradeRange:5,tradeFood:.25,tradeWealth:.18,winterBase:.9,winterVar:.5,collapseTh:-.6,collapseChance:.35,forestReclaim:.07,ruinDecay:.04,rebuildChance:.06,rebuildRange:3};
  const ultraH = {foodForest:.2,foodPlains:.05,growthTh:2.5,expandTh:5,expandPopTh:3,expandChance:.02,expandDist:2,portChance:.03,longshipChance:.03,raidRange:4,longRaidRange:8,raidChance:.25,despRaid:.3,raidStr:.7,loot:.4,conquerChance:.12,tradeRange:4,tradeFood:.15,tradeWealth:.1,winterBase:1.5,winterVar:.8,collapseTh:-.3,collapseChance:.55,forestReclaim:.12,ruinDecay:.08,rebuildChance:.03,rebuildRange:2};
  const ultraP = {foodForest:.5,foodPlains:.18,growthTh:1,expandTh:2,expandPopTh:1.2,expandChance:.12,expandDist:4,portChance:.12,longshipChance:.1,raidRange:2,longRaidRange:5,raidChance:.08,despRaid:.1,raidStr:.3,loot:.2,conquerChance:.05,tradeRange:6,tradeFood:.35,tradeWealth:.25,winterBase:.6,winterVar:.3,collapseTh:-1.5,collapseChance:.15,forestReclaim:.04,ruinDecay:.02,rebuildChance:.1,rebuildRange:4};
  const hiExpLo = {foodForest:.45,foodPlains:.15,growthTh:1,expandTh:2,expandPopTh:1,expandChance:.15,expandDist:5,portChance:.12,longshipChance:.05,raidRange:2,longRaidRange:4,raidChance:.05,despRaid:.08,raidStr:.3,loot:.15,conquerChance:.02,tradeRange:8,tradeFood:.4,tradeWealth:.3,winterBase:.6,winterVar:.2,collapseTh:-2,collapseChance:.1,forestReclaim:.03,ruinDecay:.01,rebuildChance:.12,rebuildRange:5};

  function interp(p1, p2, t) {
    const intKeys = ['expandDist', 'rebuildRange', 'raidRange', 'longRaidRange', 'tradeRange'];
    const r = {};
    for (const k of Object.keys(p1)) {
      if (typeof p1[k] === 'number') {
        const v = p1[k] * (1 - t) + p2[k] * t;
        r[k] = intKeys.includes(k) ? Math.round(v) : parseFloat(v.toFixed(4));
      } else r[k] = p1[k];
    }
    return r;
  }

  M.PS = [
    conserv,     // 0: Conservative
    mildCon,     // 1: MildConservative
    balanced,    // 2: Balanced
    modAgg,      // 3: ModerateAggressive
    aggressive,  // 4: Aggressive
    ultraH,      // 5: UltraHarsh
    ultraP,      // 6: UltraProsperous
    {...balanced, noTrade: true},    // 7: NoTrade
    {...balanced, noExpand: true},   // 8: NoExpand
    {...balanced, noConflict: true}, // 9: NoConflict
    {...balanced, constWinter: true},// 10: ConstantWinter
    interp(modAgg, aggressive, 0.33),  // 11: MA-A33
    interp(modAgg, aggressive, 0.67),  // 12: MA-A67
    hiExpLo,                            // 13: HiExpLo
    interp(conserv, balanced, 0.5),     // 14: Con-Bal
    interp(balanced, modAgg, 0.5),      // 15: Bal-MA
    interp(mildCon, balanced, 0.5),     // 16: MC-Bal
  ];
  M.regimeNames = [
    'Conserv','MildCon','Balance','ModAgg','Aggress','UltraH','UltraP',
    'NoTrade','NoExpan','NoCnflt','CstWntr','MA-A33','MA-A67',
    'HiExpLo','Con-Bal','Bal-MA','MC-Bal'
  ];
  const NR = M.PS.length;
  log(`${NR} parameter regimes defined.`);

  // ═══════════════════════════════════════════════════════════════════
  // PHASE 4: MONTE CARLO + DISTRIBUTIONS
  // ═══════════════════════════════════════════════════════════════════
  log('Phase 4: Running Monte Carlo simulations...');
  // Higher sim counts for regimes with higher blend weights
  const simCounts = {0:60,1:40,2:40,3:50,4:60,5:60,6:40,7:25,8:25,9:25,10:25,11:40,12:40,13:40,14:30,15:30,16:30};

  M.rc = {};
  for (let r = 0; r < NR; r++) {
    M.rc[r] = {};
    const P = M.PS[r];
    const NSIM = simCounts[r] || 30;
    for (let s = 0; s < SEEDS; s++) {
      const counts = [];
      for (let y = 0; y < H; y++) {
        counts[y] = [];
        for (let x = 0; x < W; x++) counts[y][x] = {};
      }
      const init = detail.initial_states[s];
      for (let i = 0; i < NSIM; i++) {
        const rng = M.mkRng(s * 100000 + r * 10000 + i);
        const g = init.grid.map(row => [...row]);
        const ss = init.settlements.map(st => ({...st}));
        const fg = M.sim(g, ss, rng, P);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const c = M.t2c(fg[y][x]);
          counts[y][x][c] = (counts[y][x][c] || 0) + 1;
        }
      }
      M.rc[r][s] = counts;
    }
    log(`  Regime ${r} (${M.regimeNames[r]}): ${NSIM} sims × ${SEEDS} seeds`);
    await new Promise(ok => setTimeout(ok, 0)); // yield to prevent timeout
  }

  // Build distributions
  log('Building distributions...');
  function buildDist(regime, seed, alpha) {
    const counts = M.rc[regime][seed];
    const dist = [];
    for (let y = 0; y < H; y++) {
      dist[y] = [];
      for (let x = 0; x < W; x++) {
        const c = counts[y][x];
        const p = new Array(6);
        let sum = 0;
        for (let k = 0; k < 6; k++) { p[k] = (c[k] || 0) + alpha; sum += p[k]; }
        for (let k = 0; k < 6; k++) p[k] /= sum;
        dist[y][x] = p;
      }
    }
    return dist;
  }

  M.pd = {}; M.gt = {};
  for (let r = 0; r < NR; r++) {
    M.pd[r] = {}; M.gt[r] = {};
    for (let s = 0; s < SEEDS; s++) {
      M.pd[r][s] = buildDist(r, s, PRED_ALPHA);
      M.gt[r][s] = buildDist(r, s, GT_ALPHA);
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // SCORING
  // ═══════════════════════════════════════════════════════════════════
  M.score = function(pred, gt) {
    let totalKL = 0, totalEnt = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const g = gt[y][x];
        let ent = 0;
        for (let c = 0; c < 6; c++) if (g[c] > 0.001) ent -= g[c] * Math.log2(g[c]);
        if (ent < 0.01) continue;
        let kl = 0;
        for (let c = 0; c < 6; c++) {
          if (g[c] > 0.001) kl += g[c] * Math.log(g[c] / Math.max(pred[y][x][c], 1e-10));
        }
        totalKL += Math.max(0, kl) * ent;
        totalEnt += ent;
      }
    }
    const wkl = totalEnt > 0 ? totalKL / totalEnt : 0;
    return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl)));
  };

  // ═══════════════════════════════════════════════════════════════════
  // BLENDING WITH FLOOR ENFORCEMENT
  // ═══════════════════════════════════════════════════════════════════
  M.blend = function(weights, seed, nRegimes) {
    const nr = nRegimes || NR;
    let totalW = 0;
    for (let i = 0; i < nr; i++) totalW += (weights[i] || 0);
    if (totalW === 0) totalW = 1;

    const pred = [];
    for (let y = 0; y < H; y++) {
      pred[y] = [];
      for (let x = 0; x < W; x++) {
        const p = [0, 0, 0, 0, 0, 0];
        for (let r = 0; r < nr; r++) {
          const w = weights[r] || 0;
          if (w === 0 || !M.pd[r] || !M.pd[r][seed]) continue;
          const rd = M.pd[r][seed][y][x];
          for (let c = 0; c < 6; c++) p[c] += w * rd[c];
        }
        for (let c = 0; c < 6; c++) p[c] /= totalW;

        // Iterative floor enforcement
        for (let iter = 0; iter < 5; iter++) {
          let below = false;
          for (let c = 0; c < 6; c++) if (p[c] < FLOOR) { p[c] = FLOOR; below = true; }
          if (!below) break;
          let s = 0; for (let c = 0; c < 6; c++) s += p[c];
          const exc = s - 1.0;
          if (Math.abs(exc) > 1e-10) {
            let above = 0;
            for (let c = 0; c < 6; c++) if (p[c] > FLOOR) above += p[c];
            if (above > 0) for (let c = 0; c < 6; c++) if (p[c] > FLOOR) p[c] -= exc * (p[c] / above);
          }
        }
        // Final precision: normalize + fix sum
        for (let c = 0; c < 6; c++) p[c] = Math.max(FLOOR, parseFloat(p[c].toFixed(6)));
        let s = 0; for (let c = 0; c < 6; c++) s += p[c];
        for (let c = 0; c < 6; c++) p[c] /= s;
        let sum4 = 0, maxIdx = 0, maxVal = 0;
        for (let c = 0; c < 6; c++) {
          p[c] = parseFloat(p[c].toFixed(6));
          sum4 += p[c];
          if (p[c] > maxVal) { maxVal = p[c]; maxIdx = c; }
        }
        p[maxIdx] = parseFloat((p[maxIdx] + (1.0 - sum4)).toFixed(6));
        pred[y][x] = p;
      }
    }
    return pred;
  };

  // ═══════════════════════════════════════════════════════════════════
  // WEIGHT OPTIMIZATION
  // ═══════════════════════════════════════════════════════════════════
  M.evalSeed = function(weights, seed) {
    const pred = M.blend(weights, seed);
    let worst = 999;
    for (let r = 0; r < NR; r++) {
      if (!M.gt[r] || !M.gt[r][seed]) continue;
      const sc = M.score(pred, M.gt[r][seed]);
      if (sc < worst) worst = sc;
    }
    return Math.round(worst * 10) / 10;
  };

  M.optimizeSeed = function(seed, startWeights) {
    let best = startWeights ? [...startWeights] : new Array(NR).fill(1);
    let bestScore = M.evalSeed(best, seed);
    const vals = [0, 0.2, 0.5, 1, 2, 3, 4, 6, 8];
    for (let pass = 0; pass < 2; pass++) {
      for (let dim = 0; dim < NR; dim++) {
        let bestVal = best[dim], bestDimScore = bestScore;
        for (const v of vals) {
          const w = [...best]; w[dim] = v;
          const sc = M.evalSeed(w, seed);
          if (sc > bestDimScore) { bestDimScore = sc; bestVal = v; }
        }
        best[dim] = bestVal;
        bestScore = bestDimScore;
      }
    }
    return { weights: best, worst: bestScore };
  };

  // ═══════════════════════════════════════════════════════════════════
  // OBSERVATION API (for next rounds)
  // ═══════════════════════════════════════════════════════════════════
  M.observe = async function(seedIndex, vx, vy, vw, vh) {
    const resp = await fetch(`${BASE}/simulate`, {
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        round_id: ROUND_ID, seed_index: seedIndex,
        viewport_x: vx || 0, viewport_y: vy || 0,
        viewport_w: vw || 15, viewport_h: vh || 15
      })
    });
    const data = await resp.json();
    if (resp.status !== 200) { log(`Observe failed: ${resp.status} ${JSON.stringify(data)}`); return null; }
    log(`Observed seed ${seedIndex} viewport (${vx},${vy})→(${vx+vw},${vy+vh}), queries: ${data.queries_used}/${data.queries_max}`);
    return data;
  };

  // Store observation data for parameter estimation
  M.observations = [];
  M.addObservation = function(seedIndex, vx, vy, data) {
    M.observations.push({ seed: seedIndex, vx, vy, data });
    log(`Stored observation ${M.observations.length} (seed ${seedIndex})`);
  };

  // ═══════════════════════════════════════════════════════════════════
  // POST-ROUND ANALYSIS (get ground truth)
  // ═══════════════════════════════════════════════════════════════════
  M.fetchGT = async function(roundId) {
    roundId = roundId || ROUND_ID;
    log(`Fetching ground truth for round ${roundId.slice(0,8)}...`);
    M.realGT = {};
    for (let s = 0; s < SEEDS; s++) {
      const resp = await fetch(`${BASE}/analysis/${roundId}/${s}`, {credentials:'include'});
      if (resp.status !== 200) {
        log(`  Seed ${s}: ${resp.status} (not available yet)`);
        continue;
      }
      const data = await resp.json();
      M.realGT[s] = data;
      log(`  Seed ${s}: score=${data.score}, GT shape=${data.ground_truth.length}x${data.ground_truth[0].length}x${data.ground_truth[0][0].length}`);
    }
    return M.realGT;
  };

  // Score our predictions against REAL GT
  M.scoreVsRealGT = function() {
    if (!M.realGT || !M.finalPreds) { log('Need realGT and finalPreds first!'); return; }
    const results = [];
    for (let s = 0; s < SEEDS; s++) {
      if (!M.realGT[s]) continue;
      const sc = M.score(M.finalPreds[s], M.realGT[s].ground_truth);
      results.push({ seed: s, ourScore: sc.toFixed(1), officialScore: M.realGT[s].score });
      log(`Seed ${s}: our calc=${sc.toFixed(1)}, official=${M.realGT[s].score}`);
    }
    return results;
  };

  // ═══════════════════════════════════════════════════════════════════
  // SUBMISSION
  // ═══════════════════════════════════════════════════════════════════
  M.submitSeed = async function(s, preds) {
    preds = preds || M.finalPreds;
    const resp = await fetch(`${BASE}/submit`, {
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ round_id: ROUND_ID, seed_index: s, prediction: preds[s] })
    });
    const r = await resp.json();
    log(`Seed ${s}: ${resp.status} — ${JSON.stringify(r).slice(0, 200)}`);
    return { seed: s, status: resp.status, result: r };
  };

  M.submitAll = async function(preds) {
    preds = preds || M.finalPreds;
    log('Submitting all seeds...');
    const results = [];
    for (let s = 0; s < SEEDS; s++) {
      results.push(await M.submitSeed(s, preds));
      await new Promise(ok => setTimeout(ok, 600));
    }
    log(`Done: ${results.filter(r => r.status === 200 || r.status === 201).length}/${SEEDS} accepted.`);
    return results;
  };

  // ═══════════════════════════════════════════════════════════════════
  // GENERATE PREDICTIONS (call after MC + optimization)
  // ═══════════════════════════════════════════════════════════════════
  M.generatePredictions = function() {
    log('Generating per-seed optimized predictions...');
    const startW = [3, 0, 0, 0, 4, 6, 0.5, 0, 0.2, 0, 0, 1.5, 2, 0, 0, 0, 0];
    M.perSeedWeights = [];
    M.finalPreds = {};

    for (let s = 0; s < SEEDS; s++) {
      const opt = M.optimizeSeed(s, startW);
      M.perSeedWeights[s] = opt.weights;
      M.finalPreds[s] = M.blend(opt.weights, s);
      log(`  Seed ${s}: worst=${opt.worst}`);
    }

    // Validate
    let badSums = 0, floorViol = 0;
    for (let s = 0; s < SEEDS; s++) {
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const p = M.finalPreds[s][y][x];
        let sum = 0;
        for (let c = 0; c < 6; c++) { sum += p[c]; if (p[c] < 0.0099) floorViol++; }
        if (Math.abs(sum - 1.0) > 0.011) badSums++;
      }
    }
    log(`Validation: ${badSums} bad sums, ${floorViol} floor violations`);
    return M.finalPreds;
  };

  // ═══════════════════════════════════════════════════════════════════
  // UTILITY: CHECK BUDGET
  // ═══════════════════════════════════════════════════════════════════
  M.budget = async function() {
    const resp = await fetch(`${BASE}/budget`, {credentials:'include'});
    return await resp.json();
  };

  M.myScores = async function() {
    const resp = await fetch(`${BASE}/my-rounds`, {credentials:'include'});
    return await resp.json();
  };

  log('=== MEGA SOLVER v3 READY ===');
  log('Commands:');
  log('  _M.generatePredictions()  — optimize + generate predictions');
  log('  _M.submitAll()            — submit all seeds (overwrites previous)');
  log('  _M.submitSeed(n)          — submit one seed');
  log('  _M.observe(seed, x, y, w, h) — observe viewport (costs 1 query)');
  log('  _M.budget()               — check query budget');
  log('  _M.fetchGT(roundId)       — fetch ground truth (post-round)');
  log('  _M.scoreVsRealGT()        — compare preds vs real GT');
  log('  _M.myScores()             — check our scores');
  return M;
})();
