/**
 * Mega Solver v4 — Astar Island prediction engine
 * Paste into browser console on app.ainm.no (must be logged in).
 *
 * v4 Critical fixes over v3:
 * - CONQUEST: conquered settlements change allegiance (NOT killed!)
 * - Population dispersal on collapse
 * - Ruin → plains decay (not just forest)
 * - Coastal ruin → port restoration
 * - Tech level tracking (affects expansion, trade, combat)
 * - Longship as separate property (not just hasPort proxy)
 * - Observation-based parameter estimation (ABC approach)
 * - Smart query planner
 * - Hybrid predictions (observations + MC)
 * - Higher sim counts (200+)
 *
 * *** DOES NOT submit or query — all API calls require explicit invocation ***
 */
(async function MegaSolverV4() {
  const BASE = 'https://api.ainm.no/astar-island';
  const FLOOR = 0.01;
  const PRED_ALPHA = 0.15;
  const GT_ALPHA = 0.08;
  const _t0 = Date.now();
  const MS = () => `${((Date.now() - _t0) / 1000).toFixed(1)}s`;

  window._M = {};
  const M = window._M;
  function log(msg) { console.log(`[MEGA4 ${MS()}] ${msg}`); }

  // ═══════════════════════════════════════════════════════════════════════
  // PHASE 1: FETCH ROUND DATA
  // ═══════════════════════════════════════════════════════════════════════
  log('Phase 1: Fetching round data...');
  const rounds = await (await fetch(`${BASE}/rounds`, {credentials:'include'})).json();
  const active = rounds.find(r => r.status === 'active');
  const scoring = rounds.find(r => r.status === 'scoring');
  const completed = rounds.filter(r => r.status === 'completed').sort((a,b) => b.round_number - a.round_number);
  const ROUND = active || scoring || completed[0] || rounds[rounds.length - 1];
  const ROUND_ID = ROUND.id;

  const detail = await (await fetch(`${BASE}/rounds/${ROUND_ID}`, {credentials:'include'})).json();
  M.d = detail;
  M.rid = ROUND_ID;
  M.rounds = rounds;
  const H = M.H = detail.map_height;
  const W = M.W = detail.map_width;
  const SEEDS = M.S = detail.seeds_count;
  log(`Round ${detail.round_number} (${ROUND_ID.slice(0,8)}), ${W}x${H}, ${SEEDS} seeds`);
  log(`Status: ${ROUND.status}, closes: ${ROUND.closes_at}`);
  if (active) log('>>> ROUND IS ACTIVE <<<');
  if (scoring) log('>>> ROUND IS SCORING — GT may be available soon <<<');
  if (!active && completed.length) log(`Latest completed: round ${completed[0].round_number}`);

  // ═══════════════════════════════════════════════════════════════════════
  // PHASE 2: SIMULATOR (v4 — major accuracy improvements)
  // ═══════════════════════════════════════════════════════════════════════
  log('Phase 2: Building v4 simulator...');

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
      this.x = x; this.y = y;
      this.pop = 1; this.food = 0.5; this.wealth = 0;
      this.defense = 0.5;
      this.tech = 0; // tech level: affects expansion efficiency, trade, combat
      this.hasPort = !!hasPort;
      this.hasLongship = false; // separate from port status
      this.alive = alive !== false;
      this.ownerId = null; // faction allegiance
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

  // Check if cell is adjacent to ocean (for port eligibility)
  function isCoastal(grid, x, y) {
    for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
      const ny = y + dy, nx = x + dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) return true;
    }
    return false;
  }

  // Core simulator: 50 years of Norse civilization (v4 accuracy)
  M.sim = function(initialGrid, initialSettlements, rng, P) {
    const grid = initialGrid.map(r => [...r]);
    const settles = initialSettlements.map((s, i) => {
      const ns = new Settlement(s.x, s.y, s.has_port, s.alive);
      ns.ownerId = i; // each settlement starts as own faction
      return ns;
    });

    // Fisher-Yates shuffle for realistic phase ordering
    function shuffle(arr) {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
      return arr;
    }

    for (let year = 0; year < 50; year++) {
      const alive = () => shuffle(settles.filter(s => s.alive));

      // ═══ GROWTH PHASE ═══
      for (const s of alive()) {
        // Food from adjacent terrain (parameterized radius)
        const fr = P.foodRadius || 1; // docs say "adjacent" — likely radius 1
        let fg = 0;
        for (let dy = -fr; dy <= fr; dy++) for (let dx = -fr; dx <= fr; dx++) {
          if (!dy && !dx) continue; // skip self
          const ny = s.y + dy, nx = s.x + dx;
          if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
          const t = grid[ny][nx];
          if (t === 4) fg += P.foodForest;
          else if (t === 11 || t === 0) fg += P.foodPlains;
        }
        s.food += fg;

        // Population growth when well-fed
        if (s.food > P.growthTh) {
          const techBonus = 1 + s.tech * 0.1;
          s.pop += 0.1 * (1 + s.wealth * 0.05) * techBonus;
          s.food -= P.growthTh * 0.5;
        }

        // Defense grows with population and tech
        s.defense = Math.min(s.defense + 0.02 * s.pop * (1 + s.tech * 0.05), s.pop * 0.8);

        // Tech slowly grows for established settlements
        if (s.pop > 1.0 && s.wealth > 0.1) {
          s.tech = Math.min(s.tech + P.techGrowth * (1 + s.wealth * 0.02), P.techMax || 5);
        }

        // Build longships (requires port + resources)
        if (s.hasPort && !s.hasLongship && s.wealth > P.longshipCost && rng() < P.longshipChance) {
          s.hasLongship = true;
          s.wealth -= P.longshipCost * 0.5;
        }

        // Expand: found new settlement on nearby land
        if (!P.noExpand && s.pop >= P.expandPopTh && s.food > P.expandTh && rng() < P.expandChance * (1 + s.tech * 0.05)) {
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
            ns.pop = 0.5; ns.food = s.food * 0.3;
            ns.tech = s.tech * 0.5; // inherit some tech
            ns.ownerId = s.ownerId; // same faction
            settles.push(ns);
            s.food *= 0.5; s.pop *= 0.8;
          }
        }

        // Port upgrade (settlement near coast)
        if (!s.hasPort && rng() < P.portChance) {
          if (isCoastal(grid, s.x, s.y)) {
            s.hasPort = true; grid[s.y][s.x] = 2;
          }
        }
      }

      // ═══ CONFLICT PHASE ═══
      if (!P.noConflict) {
        const aliveList = alive();
        for (const a of aliveList) {
          if (!a.alive) continue;
          // Longship extends raid range (separate from port)
          const range = a.hasLongship ? P.longRaidRange : P.raidRange;
          const isDesp = a.food < 0.3;
          if (rng() < (isDesp ? P.despRaid : P.raidChance)) {
            const tgts = aliveList.filter(t => t !== a && t.alive &&
              t.ownerId !== a.ownerId && // don't raid own faction
              Math.abs(t.x - a.x) + Math.abs(t.y - a.y) <= range);
            if (tgts.length) {
              const tg = tgts[Math.floor(rng() * tgts.length)];
              const techAdv = (a.tech - tg.tech) * 0.1;
              const ap = a.pop * P.raidStr * (1 + a.wealth * 0.05 + techAdv);
              const dp = tg.pop * (1 + tg.defense * 0.3 + tg.tech * 0.05);
              if (ap > dp * (0.8 + rng() * 0.4)) {
                // Successful raid: loot resources
                const st = tg.food * P.loot; a.food += st; tg.food -= st;
                a.wealth += tg.wealth * P.loot * 0.5;
                tg.wealth = Math.max(0, tg.wealth - tg.wealth * P.loot * 0.5);
                tg.defense *= 0.7; // defense degrades after losing

                // CONQUEST: settlement changes allegiance (v4 fix — NOT killed!)
                if (rng() < P.conquerChance) {
                  tg.ownerId = a.ownerId; // change faction allegiance
                  tg.defense *= 0.5; // weakened after conquest
                  // Small chance of destruction instead of conquest
                  if (rng() < P.destroyOnConquest) {
                    tg.alive = false; grid[tg.y][tg.x] = 3;
                  }
                }
              } else {
                a.defense *= 0.9; // failed raid costs some defense
              }
            }
          }
        }
      }

      // ═══ TRADE PHASE ═══
      if (!P.noTrade) {
        const ports = alive().filter(s => s.hasPort);
        for (let i = 0; i < ports.length; i++) {
          for (let j = i + 1; j < ports.length; j++) {
            const s = ports[i], p = ports[j];
            if (Math.abs(p.x - s.x) + Math.abs(p.y - s.y) > P.tradeRange) continue;
            // Factions at war trade less (different factions = reduced)
            const sameFaction = s.ownerId === p.ownerId;
            const tradeMul = sameFaction ? 1.0 : 0.5;
            const techMul = 1 + (s.tech + p.tech) * 0.05;
            s.food += P.tradeFood * 0.5 * tradeMul * techMul;
            p.food += P.tradeFood * 0.5 * tradeMul * techMul;
            s.wealth += P.tradeWealth * tradeMul * techMul;
            p.wealth += P.tradeWealth * tradeMul * techMul;
            // Tech diffusion through trade
            if (s.tech > p.tech + 0.1) p.tech += (s.tech - p.tech) * P.techDiffusion * tradeMul;
            if (p.tech > s.tech + 0.1) s.tech += (p.tech - s.tech) * P.techDiffusion * tradeMul;
          }
        }
      }

      // ═══ WINTER PHASE ═══
      const sev = P.constWinter ? P.winterBase : P.winterBase + (rng() - 0.5) * P.winterVar;
      for (const s of alive()) {
        s.food -= sev * (0.8 + s.pop * 0.2);
        s.pop = Math.max(0.1, s.pop - sev * 0.05);
        // Collapse check
        if (s.food < P.collapseTh && rng() < P.collapseChance) {
          s.alive = false; grid[s.y][s.x] = 3;
          // POPULATION DISPERSAL (v4 fix): distribute pop to nearby friendly settlements
          const nearby = settles.filter(n => n.alive && n.ownerId === s.ownerId &&
            Math.abs(n.x - s.x) + Math.abs(n.y - s.y) <= P.dispersalRange && n !== s);
          if (nearby.length > 0) {
            const popShare = s.pop * P.dispersalFraction / nearby.length;
            const foodShare = Math.max(0, s.food * 0.5) / nearby.length;
            for (const n of nearby) {
              n.pop += popShare;
              n.food += foodShare;
            }
          }
        }
      }

      // ═══ ENVIRONMENT PHASE ═══
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        if (grid[y][x] === 3) { // Ruin
          // Nearby thriving settlements can reclaim ruins
          const nearSettles = settles.filter(s => s.alive &&
            Math.abs(s.x - x) + Math.abs(s.y - y) <= P.rebuildRange);
          const thrivingNear = nearSettles.filter(s => s.pop > 1.0 && s.food > 0.5);

          if (thrivingNear.length && rng() < P.rebuildChance) {
            // COASTAL PORT RESTORATION (v4 fix)
            if (isCoastal(grid, x, y) && rng() < P.portRestoreChance) {
              grid[y][x] = 2; // restore as port
              const ns = new Settlement(x, y, true, true);
              const patron = thrivingNear[Math.floor(rng() * thrivingNear.length)];
              ns.pop = patron.pop * 0.3;
              ns.food = patron.food * 0.2;
              ns.tech = patron.tech * 0.5;
              ns.ownerId = patron.ownerId;
              ns.hasPort = true;
              settles.push(ns);
              patron.pop *= 0.85; patron.food *= 0.7;
            } else {
              grid[y][x] = 1; // rebuild as settlement
              const ns = new Settlement(x, y, false, true);
              const patron = thrivingNear[Math.floor(rng() * thrivingNear.length)];
              ns.pop = patron.pop * 0.3;
              ns.food = patron.food * 0.2;
              ns.tech = patron.tech * 0.5;
              ns.ownerId = patron.ownerId;
              settles.push(ns);
              patron.pop *= 0.85; patron.food *= 0.7;
            }
          } else if (rng() < P.forestReclaim) {
            grid[y][x] = 4; // forest reclaims ruin
          } else if (rng() < P.plainsReclaim) {
            // RUIN → PLAINS DECAY (v4 fix)
            grid[y][x] = 11; // fade back to plains
          }
        }
      }
    }
    return { grid, settlements: settles };
  };

  // Backward-compatible wrapper: returns just grid (for MC counting)
  M.simGrid = function(initialGrid, initialSettlements, rng, P) {
    return M.sim(initialGrid, initialSettlements, rng, P).grid;
  };

  // Full sim with stats: returns grid + settlement stats (for param estimation)
  M.simFull = function(initialGrid, initialSettlements, rng, P) {
    const result = M.sim(initialGrid, initialSettlements, rng, P);
    const alive = result.settlements.filter(s => s.alive);
    const avg = (arr) => arr.length ? arr.reduce((s,x) => s+x, 0) / arr.length : 0;
    return {
      grid: result.grid,
      settlements: result.settlements,
      stats: {
        aliveCount: alive.length,
        totalCount: result.settlements.length,
        avgPop: avg(alive.map(s => s.pop)),
        avgFood: avg(alive.map(s => s.food)),
        avgWealth: avg(alive.map(s => s.wealth)),
        avgDefense: avg(alive.map(s => s.defense)),
        avgTech: avg(alive.map(s => s.tech)),
        portCount: alive.filter(s => s.hasPort).length,
        longshipCount: alive.filter(s => s.hasLongship).length,
        factionCount: new Set(alive.map(s => s.ownerId)).size,
        survivalRate: alive.length / Math.max(1, result.settlements.length)
      }
    };
  };

  log('Simulator v4 built (conquest-fix, dispersal, tech, longships, port-restore, plains-decay, shuffle).');

  // ═══════════════════════════════════════════════════════════════════════
  // PHASE 3: PARAMETER REGIMES (v4 — expanded with new params)
  // ═══════════════════════════════════════════════════════════════════════
  // New params in v4: techGrowth, techMax, techDiffusion, longshipCost,
  //                   destroyOnConquest, dispersalRange, dispersalFraction,
  //                   portRestoreChance, plainsReclaim

  const baseNew = {techGrowth:0.05, techMax:5, techDiffusion:0.1, longshipCost:0.5,
                   destroyOnConquest:0.15, dispersalRange:4, dispersalFraction:0.5,
                   portRestoreChance:0.4, plainsReclaim:0.03, foodRadius:1};

  function regime(base, overrides) { return {...baseNew, ...base, ...overrides}; }

  const conserv = regime({foodForest:.3,foodPlains:.08,growthTh:1.8,expandTh:4,expandPopTh:2,expandChance:.04,expandDist:3,portChance:.06,longshipChance:.05,raidRange:3,longRaidRange:7,raidChance:.15,despRaid:.2,raidStr:.5,loot:.3,conquerChance:.08,tradeRange:5,tradeFood:.2,tradeWealth:.15,winterBase:1,winterVar:.6,collapseTh:-.5,collapseChance:.4,forestReclaim:.08,ruinDecay:.05,rebuildChance:.05,rebuildRange:3});
  const mildCon = regime({foodForest:.35,foodPlains:.1,growthTh:1.5,expandTh:3.5,expandPopTh:1.8,expandChance:.06,expandDist:3,portChance:.08,longshipChance:.06,raidRange:3,longRaidRange:7,raidChance:.18,despRaid:.22,raidStr:.5,loot:.3,conquerChance:.1,tradeRange:5,tradeFood:.25,tradeWealth:.18,winterBase:.9,winterVar:.5,collapseTh:-.6,collapseChance:.35,forestReclaim:.07,ruinDecay:.04,rebuildChance:.06,rebuildRange:3});
  const balanced = regime({foodForest:.4,foodPlains:.12,growthTh:1.3,expandTh:3,expandPopTh:1.5,expandChance:.08,expandDist:3,portChance:.1,longshipChance:.08,raidRange:4,longRaidRange:8,raidChance:.2,despRaid:.25,raidStr:.55,loot:.35,conquerChance:.12,tradeRange:6,tradeFood:.3,tradeWealth:.2,winterBase:.85,winterVar:.45,collapseTh:-.8,collapseChance:.3,forestReclaim:.06,ruinDecay:.03,rebuildChance:.07,rebuildRange:3});
  const modAgg = regime({foodForest:.45,foodPlains:.14,growthTh:1.1,expandTh:2.5,expandPopTh:1.3,expandChance:.1,expandDist:4,portChance:.12,longshipChance:.1,raidRange:4,longRaidRange:9,raidChance:.22,despRaid:.28,raidStr:.6,loot:.4,conquerChance:.15,tradeRange:6,tradeFood:.35,tradeWealth:.25,winterBase:.75,winterVar:.4,collapseTh:-1,collapseChance:.25,forestReclaim:.05,ruinDecay:.025,rebuildChance:.08,rebuildRange:4});
  const aggressive = regime({foodForest:.5,foodPlains:.16,growthTh:.9,expandTh:2,expandPopTh:1,expandChance:.12,expandDist:4,portChance:.15,longshipChance:.12,raidRange:5,longRaidRange:10,raidChance:.25,despRaid:.3,raidStr:.65,loot:.45,conquerChance:.18,tradeRange:7,tradeFood:.4,tradeWealth:.3,winterBase:.65,winterVar:.35,collapseTh:-1.2,collapseChance:.2,forestReclaim:.04,ruinDecay:.02,rebuildChance:.1,rebuildRange:4});
  const ultraH = regime({foodForest:.2,foodPlains:.05,growthTh:2.5,expandTh:5,expandPopTh:3,expandChance:.02,expandDist:2,portChance:.03,longshipChance:.03,raidRange:4,longRaidRange:8,raidChance:.25,despRaid:.3,raidStr:.7,loot:.4,conquerChance:.12,tradeRange:4,tradeFood:.15,tradeWealth:.1,winterBase:1.5,winterVar:.8,collapseTh:-.3,collapseChance:.55,forestReclaim:.12,ruinDecay:.08,rebuildChance:.03,rebuildRange:2}, {destroyOnConquest:.3});
  const ultraP = regime({foodForest:.5,foodPlains:.18,growthTh:1,expandTh:2,expandPopTh:1.2,expandChance:.12,expandDist:4,portChance:.12,longshipChance:.1,raidRange:2,longRaidRange:5,raidChance:.08,despRaid:.1,raidStr:.3,loot:.2,conquerChance:.05,tradeRange:6,tradeFood:.35,tradeWealth:.25,winterBase:.6,winterVar:.3,collapseTh:-1.5,collapseChance:.15,forestReclaim:.04,ruinDecay:.02,rebuildChance:.1,rebuildRange:4}, {destroyOnConquest:.05});
  const hiExpLo = regime({foodForest:.45,foodPlains:.15,growthTh:1,expandTh:2,expandPopTh:1,expandChance:.15,expandDist:5,portChance:.12,longshipChance:.05,raidRange:2,longRaidRange:4,raidChance:.05,despRaid:.08,raidStr:.3,loot:.15,conquerChance:.02,tradeRange:8,tradeFood:.4,tradeWealth:.3,winterBase:.6,winterVar:.2,collapseTh:-2,collapseChance:.1,forestReclaim:.03,ruinDecay:.01,rebuildChance:.12,rebuildRange:5}, {destroyOnConquest:.05});

  function interp(p1, p2, t) {
    const intKeys = ['expandDist', 'rebuildRange', 'raidRange', 'longRaidRange', 'tradeRange', 'dispersalRange'];
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
    interp(balanced, hiExpLo, 0.5),     // 17: Bal-HiExp
    interp(modAgg, hiExpLo, 0.5),       // 18: MA-HiExp
    interp(aggressive, hiExpLo, 0.5),   // 19: Agg-HiExp
    // Wide-radius food variants (foodRadius=2 → more food → different dynamics)
    regime({...balanced, foodRadius:2, foodForest:.2, foodPlains:.06}),  // 20: Bal-WR
    regime({...modAgg, foodRadius:2, foodForest:.22, foodPlains:.07}),   // 21: MA-WR
    regime({...aggressive, foodRadius:2, foodForest:.25, foodPlains:.08}), // 22: Agg-WR
  ];
  M.regimeNames = [
    'Conserv','MildCon','Balance','ModAgg','Aggress','UltraH','UltraP',
    'NoTrade','NoExpan','NoCnflt','CstWntr','MA-A33','MA-A67',
    'HiExpLo','Con-Bal','Bal-MA','MC-Bal','Bal-HiX','MA-HiX','Agg-HiX',
    'Bal-WR','MA-WR','Agg-WR'
  ];
  const NR = M.PS.length;
  log(`${NR} parameter regimes defined.`);

  // ═══════════════════════════════════════════════════════════════════════
  // PHASE 4: MONTE CARLO SIMULATION
  // ═══════════════════════════════════════════════════════════════════════
  log('Phase 4: Running Monte Carlo simulations...');

  // Sim counts: core regimes get more sims
  const simCounts = {};
  for (let r = 0; r < NR; r++) {
    if (r <= 6) simCounts[r] = 80;      // core 7 regimes
    else if (r <= 10) simCounts[r] = 40; // mechanical variants
    else simCounts[r] = 50;              // interpolated
  }

  M.rc = {};
  for (let r = 0; r < NR; r++) {
    M.rc[r] = {};
    const P = M.PS[r];
    const NSIM = simCounts[r] || 50;
    for (let s = 0; s < SEEDS; s++) {
      const counts = [];
      for (let y = 0; y < H; y++) {
        counts[y] = [];
        for (let x = 0; x < W; x++) counts[y][x] = {};
      }
      const init = detail.initial_states[s];
      for (let i = 0; i < NSIM; i++) {
        const rng = M.mkRng(s * 100000 + r * 10000 + i * 7 + 42);
        const fg = M.simGrid(init.grid, init.settlements, rng, P);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const c = M.t2c(fg[y][x]);
          counts[y][x][c] = (counts[y][x][c] || 0) + 1;
        }
      }
      M.rc[r][s] = counts;
    }
    log(`  Regime ${r} (${M.regimeNames[r]}): ${NSIM} sims x ${SEEDS} seeds`);
    await new Promise(ok => setTimeout(ok, 0)); // yield
  }

  // Build distributions with Dirichlet smoothing
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

  log('Building distributions...');
  M.pd = {}; M.gt = {};
  for (let r = 0; r < NR; r++) {
    M.pd[r] = {}; M.gt[r] = {};
    for (let s = 0; s < SEEDS; s++) {
      M.pd[r][s] = buildDist(r, s, PRED_ALPHA);
      M.gt[r][s] = buildDist(r, s, GT_ALPHA);
    }
  }
  log('Distributions built.');

  // ═══════════════════════════════════════════════════════════════════════
  // SCORING ENGINE
  // ═══════════════════════════════════════════════════════════════════════
  M.score = function(pred, gt) {
    let totalKL = 0, totalEnt = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const g = gt[y][x];
        let ent = 0;
        for (let c = 0; c < 6; c++) if (g[c] > 1e-6) ent -= g[c] * Math.log(g[c]);
        if (ent < 0.01) continue;
        let kl = 0;
        for (let c = 0; c < 6; c++) {
          if (g[c] > 1e-6) kl += g[c] * Math.log(g[c] / Math.max(pred[y][x][c], 1e-10));
        }
        totalKL += Math.max(0, kl) * ent;
        totalEnt += ent;
      }
    }
    const wkl = totalEnt > 0 ? totalKL / totalEnt : 0;
    return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl)));
  };

  // Score with detailed breakdown
  M.scoreDetailed = function(pred, gt) {
    let totalKL = 0, totalEnt = 0;
    let worstCell = null, worstKL = 0;
    let dynamicCells = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const g = gt[y][x];
        let ent = 0;
        for (let c = 0; c < 6; c++) if (g[c] > 1e-6) ent -= g[c] * Math.log(g[c]);
        if (ent < 0.01) continue;
        dynamicCells++;
        let kl = 0;
        for (let c = 0; c < 6; c++) {
          if (g[c] > 1e-6) kl += g[c] * Math.log(g[c] / Math.max(pred[y][x][c], 1e-10));
        }
        kl = Math.max(0, kl);
        totalKL += kl * ent;
        totalEnt += ent;
        if (kl > worstKL) { worstKL = kl; worstCell = {y, x, kl, ent, gt: g, pred: pred[y][x]}; }
      }
    }
    const wkl = totalEnt > 0 ? totalKL / totalEnt : 0;
    return {
      score: Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl))),
      weightedKL: wkl, dynamicCells, worstCell
    };
  };

  // ═══════════════════════════════════════════════════════════════════════
  // BLENDING WITH FLOOR ENFORCEMENT
  // ═══════════════════════════════════════════════════════════════════════
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

  // ═══════════════════════════════════════════════════════════════════════
  // WEIGHT OPTIMIZATION (coordinate descent)
  // ═══════════════════════════════════════════════════════════════════════
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

  // Evaluate against specific GT regimes (for focused optimization)
  M.evalSeedVsRegimes = function(weights, seed, regimeList) {
    const pred = M.blend(weights, seed);
    let worst = 999;
    for (const r of regimeList) {
      if (!M.gt[r] || !M.gt[r][seed]) continue;
      const sc = M.score(pred, M.gt[r][seed]);
      if (sc < worst) worst = sc;
    }
    return Math.round(worst * 10) / 10;
  };

  M.optimizeSeed = function(seed, startWeights, regimeList) {
    let best = startWeights ? [...startWeights] : new Array(NR).fill(1);
    const evalFn = regimeList
      ? (w) => M.evalSeedVsRegimes(w, seed, regimeList)
      : (w) => M.evalSeed(w, seed);
    let bestScore = evalFn(best);
    const vals = [0, 0.2, 0.5, 1, 2, 3, 4, 6, 8, 10];
    for (let pass = 0; pass < 3; pass++) {
      for (let dim = 0; dim < NR; dim++) {
        let bestVal = best[dim], bestDimScore = bestScore;
        for (const v of vals) {
          const w = [...best]; w[dim] = v;
          const sc = evalFn(w);
          if (sc > bestDimScore) { bestDimScore = sc; bestVal = v; }
        }
        best[dim] = bestVal;
        bestScore = bestDimScore;
      }
    }
    return { weights: best, worst: bestScore };
  };

  // ═══════════════════════════════════════════════════════════════════════
  // OBSERVATION API (NO auto-query — explicit only)
  // ═══════════════════════════════════════════════════════════════════════
  M.observations = [];

  M.observe = async function(seedIndex, vx, vy, vw, vh) {
    log(`*** QUERYING API — seed ${seedIndex}, viewport (${vx},${vy},${vw},${vh}) ***`);
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
    if (resp.status !== 200) { log(`Observe FAILED: ${resp.status} ${JSON.stringify(data)}`); return null; }
    // Store observation
    M.observations.push({ seed: seedIndex, vx: vx||0, vy: vy||0, vw: vw||15, vh: vh||15, data });
    log(`Observed seed ${seedIndex} viewport (${vx},${vy})->(${(vx||0)+(vw||15)},${(vy||0)+(vh||15)}), queries: ${data.queries_used}/${data.queries_max}`);
    log(`  Settlements in view: ${data.settlements.length}`);
    return data;
  };

  M.budget = async function() {
    const resp = await fetch(`${BASE}/budget`, {credentials:'include'});
    return await resp.json();
  };

  // ═══════════════════════════════════════════════════════════════════════
  // SMART QUERY PLANNER
  // ═══════════════════════════════════════════════════════════════════════
  M.planQueries = function(totalBudget) {
    totalBudget = totalBudget || 50;
    log('=== QUERY PLAN ===');

    // Analyze initial states to find settlement-dense regions
    const settlementHeatmap = Array(H).fill(0).map(() => Array(W).fill(0));
    for (let s = 0; s < SEEDS; s++) {
      for (const st of detail.initial_states[s].settlements) {
        for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
          const ny = st.y + dy, nx = st.x + dx;
          if (ny >= 0 && ny < H && nx >= 0 && nx < W) settlementHeatmap[ny][nx]++;
        }
      }
    }

    // Find best viewport positions (max settlement coverage)
    const viewports = [];
    for (let vy = 0; vy <= H - 15; vy += 5) {
      for (let vx = 0; vx <= W - 15; vx += 5) {
        let heat = 0;
        for (let y = vy; y < vy + 15 && y < H; y++)
          for (let x = vx; x < vx + 15 && x < W; x++)
            heat += settlementHeatmap[y][x];
        viewports.push({vx, vy, heat});
      }
    }
    viewports.sort((a, b) => b.heat - a.heat);

    // Phase 1: Full map coverage on 2 seeds (18 queries)
    const phase1 = [];
    for (const s of [0, 1]) {
      // Cover 40x40 with overlapping 15x15 viewports
      for (const [vx, vy] of [[0,0],[13,0],[25,0],[0,13],[13,13],[25,13],[0,25],[13,25],[25,25]]) {
        phase1.push({seed: s, vx, vy, vw: 15, vh: 15});
      }
    }

    // Phase 2: Dense observation of settlement-heavy regions (24 queries)
    const topViewports = viewports.slice(0, 4);
    const phase2 = [];
    for (const vp of topViewports) {
      for (const s of [0, 1, 2]) { // 3 seeds × 4 viewports × 1 each = 12
        phase2.push({seed: s, vx: vp.vx, vy: vp.vy, vw: 15, vh: 15});
      }
    }
    // Repeat top 2 viewports on remaining seeds
    for (const vp of topViewports.slice(0, 2)) {
      for (const s of [3, 4]) {
        phase2.push({seed: s, vx: vp.vx, vy: vp.vy, vw: 15, vh: 15});
      }
      // Extra observations on seed 0 for high-heat areas
      phase2.push({seed: 0, vx: vp.vx, vy: vp.vy, vw: 15, vh: 15});
      phase2.push({seed: 1, vx: vp.vx, vy: vp.vy, vw: 15, vh: 15});
    }

    // Phase 3: Reserve (8 queries for refinement)
    const phase3 = [];

    const plan = {
      phase1: phase1.slice(0, 18),
      phase2: phase2.slice(0, 24),
      phase3Reserve: totalBudget - 18 - 24,
      total: Math.min(totalBudget, phase1.length + phase2.length),
      topViewports: topViewports.map(v => `(${v.vx},${v.vy}) heat=${v.heat}`)
    };

    log(`Phase 1: ${plan.phase1.length} queries (full map coverage, 2 seeds)`);
    log(`Phase 2: ${plan.phase2.length} queries (settlement-dense regions, all seeds)`);
    log(`Phase 3: ${plan.phase3Reserve} queries reserved for refinement`);
    log(`Top viewports: ${plan.topViewports.join(', ')}`);

    M.queryPlan = plan;
    return plan;
  };

  // Execute query plan (one phase at a time, with user confirmation)
  M.executePhase = async function(phaseNum) {
    const plan = M.queryPlan;
    if (!plan) { log('Run M.planQueries() first!'); return; }
    const queries = phaseNum === 1 ? plan.phase1 : phaseNum === 2 ? plan.phase2 : [];
    log(`Executing phase ${phaseNum}: ${queries.length} queries...`);
    const results = [];
    for (const q of queries) {
      const data = await M.observe(q.seed, q.vx, q.vy, q.vw, q.vh);
      results.push(data);
      await new Promise(ok => setTimeout(ok, 250)); // rate limit: max 5/sec
    }
    log(`Phase ${phaseNum} complete: ${results.filter(Boolean).length}/${queries.length} successful`);
    return results;
  };

  // ═══════════════════════════════════════════════════════════════════════
  // PARAMETER ESTIMATION FROM OBSERVATIONS (ABC approach)
  // ═══════════════════════════════════════════════════════════════════════
  M.extractStats = function(obsData) {
    // Extract summary statistics from an observation
    const g = obsData.grid;
    const setts = obsData.settlements;
    const h = g.length, w = g[0].length;
    let nSettle = 0, nPort = 0, nRuin = 0, nForest = 0, nEmpty = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const t = g[y][x];
      if (t === 1) nSettle++;
      else if (t === 2) nPort++;
      else if (t === 3) nRuin++;
      else if (t === 4) nForest++;
      else if (t === 0 || t === 11) nEmpty++;
    }
    const aliveS = setts.filter(s => s.alive);
    const avgPop = aliveS.length ? aliveS.reduce((s, x) => s + x.population, 0) / aliveS.length : 0;
    const avgFood = aliveS.length ? aliveS.reduce((s, x) => s + x.food, 0) / aliveS.length : 0;
    const avgWealth = aliveS.length ? aliveS.reduce((s, x) => s + x.wealth, 0) / aliveS.length : 0;
    const avgDefense = aliveS.length ? aliveS.reduce((s, x) => s + x.defense, 0) / aliveS.length : 0;
    const factions = new Set(aliveS.map(s => s.owner_id));
    const nPorts = aliveS.filter(s => s.has_port).length;

    return {
      nSettle, nPort, nRuin, nForest, nEmpty,
      aliveCount: aliveS.length,
      avgPop, avgFood, avgWealth, avgDefense,
      nFactions: factions.size,
      nPortSettles: nPorts,
      survivalRate: aliveS.length / Math.max(1, setts.length),
      ruinRatio: nRuin / Math.max(1, nSettle + nPort + nRuin)
    };
  };

  // Compare our simulated stats with observed stats to find best regime
  // Uses BOTH terrain counts AND settlement stats (pop, food, wealth, defense)
  M.estimateParams = function() {
    if (!M.observations.length) { log('No observations! Run queries first.'); return; }

    // Compute average stats across all observations
    const allStats = M.observations.map(o => M.extractStats(o.data));
    const avgObs = {};
    const keys = Object.keys(allStats[0]);
    for (const k of keys) {
      avgObs[k] = allStats.reduce((s, st) => s + st[k], 0) / allStats.length;
    }

    log('Observed averages:');
    log(`  Settlements: ${avgObs.nSettle.toFixed(1)}, Ports: ${avgObs.nPort.toFixed(1)}, Ruins: ${avgObs.nRuin.toFixed(1)}`);
    log(`  AvgPop: ${avgObs.avgPop.toFixed(2)}, AvgFood: ${avgObs.avgFood.toFixed(2)}, AvgWealth: ${avgObs.avgWealth.toFixed(2)}, AvgDef: ${avgObs.avgDefense.toFixed(2)}`);
    log(`  Factions: ${avgObs.nFactions.toFixed(1)}, SurvivalRate: ${avgObs.survivalRate.toFixed(2)}`);

    // For each regime, simulate with FULL stats and compare
    const regimeScores = [];
    for (let r = 0; r < NR; r++) {
      const simTerrainStats = [];
      const simSettleStats = [];

      // Simulate a few runs using simFull for stat comparison
      for (let i = 0; i < 8; i++) {
        for (const obs of M.observations.slice(0, Math.min(5, M.observations.length))) {
          const init = detail.initial_states[obs.seed];
          const rng = M.mkRng(obs.seed * 100000 + r * 10000 + i * 31 + 99);
          const result = M.simFull(init.grid, init.settlements, rng, M.PS[r]);

          // Extract viewport terrain counts
          const vx = obs.vx, vy = obs.vy, vw = obs.vw, vh = obs.vh;
          let ns = 0, np = 0, nr2 = 0;
          for (let y = vy; y < vy + vh && y < H; y++) {
            for (let x = vx; x < vx + vw && x < W; x++) {
              if (result.grid[y][x] === 1) ns++;
              else if (result.grid[y][x] === 2) np++;
              else if (result.grid[y][x] === 3) nr2++;
            }
          }
          simTerrainStats.push({ nSettle: ns, nPort: np, nRuin: nr2 });

          // Collect settlement stats for viewport area
          const viewportSettles = result.settlements.filter(s => s.alive &&
            s.x >= vx && s.x < vx + vw && s.y >= vy && s.y < vy + vh);
          if (viewportSettles.length) {
            const avg = arr => arr.reduce((s,x) => s+x, 0) / arr.length;
            simSettleStats.push({
              avgPop: avg(viewportSettles.map(s => s.pop)),
              avgFood: avg(viewportSettles.map(s => s.food)),
              avgWealth: avg(viewportSettles.map(s => s.wealth)),
              avgDefense: avg(viewportSettles.map(s => s.defense)),
              factions: new Set(viewportSettles.map(s => s.ownerId)).size
            });
          }
        }
      }

      // Average simulated terrain stats
      const avgSim = {nSettle: 0, nPort: 0, nRuin: 0};
      for (const st of simTerrainStats) {
        avgSim.nSettle += st.nSettle; avgSim.nPort += st.nPort; avgSim.nRuin += st.nRuin;
      }
      const n = simTerrainStats.length;
      avgSim.nSettle /= n; avgSim.nPort /= n; avgSim.nRuin /= n;

      // Average simulated settlement stats
      const avgSimStats = {avgPop: 0, avgFood: 0, avgWealth: 0, avgDefense: 0, factions: 0};
      if (simSettleStats.length) {
        for (const st of simSettleStats) {
          avgSimStats.avgPop += st.avgPop;
          avgSimStats.avgFood += st.avgFood;
          avgSimStats.avgWealth += st.avgWealth;
          avgSimStats.avgDefense += st.avgDefense;
          avgSimStats.factions += st.factions;
        }
        const ns = simSettleStats.length;
        avgSimStats.avgPop /= ns; avgSimStats.avgFood /= ns;
        avgSimStats.avgWealth /= ns; avgSimStats.avgDefense /= ns;
        avgSimStats.factions /= ns;
      }

      // Combined distance metric: terrain counts + settlement stats
      const terrainDist = Math.sqrt(
        Math.pow((avgSim.nSettle - avgObs.nSettle) / Math.max(1, avgObs.nSettle), 2) +
        Math.pow((avgSim.nPort - avgObs.nPort) / Math.max(1, avgObs.nPort), 2) +
        Math.pow((avgSim.nRuin - avgObs.nRuin) / Math.max(1, avgObs.nRuin), 2)
      );
      const statsDist = (avgObs.avgPop > 0.01) ? Math.sqrt(
        Math.pow((avgSimStats.avgPop - avgObs.avgPop) / Math.max(0.1, avgObs.avgPop), 2) +
        Math.pow((avgSimStats.avgFood - avgObs.avgFood) / Math.max(0.1, Math.abs(avgObs.avgFood) + 0.1), 2) +
        Math.pow((avgSimStats.avgWealth - avgObs.avgWealth) / Math.max(0.1, avgObs.avgWealth + 0.1), 2) +
        Math.pow((avgSimStats.avgDefense - avgObs.avgDefense) / Math.max(0.1, avgObs.avgDefense + 0.1), 2)
      ) : 0;
      const dist = terrainDist * 0.6 + statsDist * 0.4; // weighted combination

      regimeScores.push({regime: r, name: M.regimeNames[r], dist, terrainDist, statsDist, avgSim, avgSimStats});
      log(`  Regime ${r} (${M.regimeNames[r]}): dist=${dist.toFixed(3)} terrain=${terrainDist.toFixed(3)} stats=${statsDist.toFixed(3)} [S:${avgSim.nSettle.toFixed(1)} P:${avgSim.nPort.toFixed(1)} R:${avgSim.nRuin.toFixed(1)} pop:${avgSimStats.avgPop.toFixed(2)}]`);
    }

    regimeScores.sort((a, b) => a.dist - b.dist);
    log('Best matching regimes:');
    for (const r of regimeScores.slice(0, 5)) {
      log(`  #${r.regime} ${r.name}: dist=${r.dist.toFixed(3)}`);
    }

    M.paramEstimate = {
      bestRegimes: regimeScores.slice(0, 5).map(r => r.regime),
      allScores: regimeScores,
      observedStats: avgObs
    };
    return M.paramEstimate;
  };

  // ═══════════════════════════════════════════════════════════════════════
  // HYBRID PREDICTIONS (observations + MC)
  // ═══════════════════════════════════════════════════════════════════════
  M.buildObservationDist = function() {
    // Build empirical distributions from observation data
    const obsCounts = {};
    for (let s = 0; s < SEEDS; s++) {
      obsCounts[s] = Array(H).fill(null).map(() => Array(W).fill(null).map(() => ({})));
    }
    let totalObs = 0;
    for (const obs of M.observations) {
      const g = obs.data.grid;
      const s = obs.seed;
      for (let dy = 0; dy < obs.vh; dy++) {
        for (let dx = 0; dx < obs.vw; dx++) {
          const y = obs.vy + dy, x = obs.vx + dx;
          if (y >= H || x >= W || dy >= g.length || dx >= g[0].length) continue;
          const c = M.t2c(g[dy][dx]);
          obsCounts[s][y][x][c] = (obsCounts[s][y][x][c] || 0) + 1;
          totalObs++;
        }
      }
    }

    // Convert to distributions
    M.obsDist = {};
    M.obsCount = {};
    for (let s = 0; s < SEEDS; s++) {
      M.obsDist[s] = [];
      M.obsCount[s] = [];
      for (let y = 0; y < H; y++) {
        M.obsDist[s][y] = [];
        M.obsCount[s][y] = [];
        for (let x = 0; x < W; x++) {
          const c = obsCounts[s][y][x];
          let total = 0;
          for (const k in c) total += c[k];
          M.obsCount[s][y][x] = total;
          if (total > 0) {
            const p = new Array(6);
            let sum = 0;
            for (let k = 0; k < 6; k++) { p[k] = (c[k] || 0) + PRED_ALPHA; sum += p[k]; }
            for (let k = 0; k < 6; k++) p[k] /= sum;
            M.obsDist[s][y][x] = p;
          } else {
            M.obsDist[s][y][x] = null;
          }
        }
      }
    }
    log(`Built observation distributions from ${M.observations.length} observations (${totalObs} cell-obs)`);
    return M.obsDist;
  };

  // Hybrid: blend MC predictions with observation distributions
  M.hybridBlend = function(mcPred, seed, obsWeight) {
    obsWeight = obsWeight || 0.6; // how much to trust observations vs MC
    if (!M.obsDist || !M.obsDist[seed]) {
      log('No observation distributions! Run M.buildObservationDist() first.');
      return mcPred;
    }

    const pred = [];
    for (let y = 0; y < H; y++) {
      pred[y] = [];
      for (let x = 0; x < W; x++) {
        const mc = mcPred[y][x];
        const obs = M.obsDist[seed][y][x];
        const nObs = M.obsCount[seed][y][x];

        if (obs && nObs >= 2) {
          // Blend observation distribution with MC, weighted by observation count
          const w = Math.min(obsWeight, nObs / (nObs + 5)); // more obs = more weight
          const p = new Array(6);
          for (let c = 0; c < 6; c++) p[c] = w * obs[c] + (1 - w) * mc[c];
          // Floor + normalize
          for (let c = 0; c < 6; c++) p[c] = Math.max(FLOOR, p[c]);
          let s = 0; for (let c = 0; c < 6; c++) s += p[c];
          for (let c = 0; c < 6; c++) p[c] /= s;
          pred[y][x] = p;
        } else {
          pred[y][x] = [...mc];
        }
      }
    }
    return pred;
  };

  // ═══════════════════════════════════════════════════════════════════════
  // OBSERVATION VS MC COMPARISON (live parameter estimation)
  // ═══════════════════════════════════════════════════════════════════════
  M.compareObsVsMC = function() {
    if (!M.observations.length) { log('No observations!'); return; }
    log('=== OBSERVATION vs MC COMPARISON ===');

    // For each observation, compute likelihood under each regime
    const regimeLikelihoods = new Array(NR).fill(0);

    for (const obs of M.observations) {
      const g = obs.data.grid;
      const s = obs.seed;

      for (let r = 0; r < NR; r++) {
        let logLik = 0;
        let cells = 0;
        for (let dy = 0; dy < obs.vh; dy++) {
          for (let dx = 0; dx < obs.vw; dx++) {
            const y = obs.vy + dy, x = obs.vx + dx;
            if (y >= H || x >= W || dy >= g.length || dx >= g[0].length) continue;
            const obsClass = M.t2c(g[dy][dx]);
            const pred = M.pd[r][s][y][x];
            // Log-likelihood of observed class under this regime's distribution
            logLik += Math.log(Math.max(pred[obsClass], 1e-10));
            cells++;
          }
        }
        regimeLikelihoods[r] += logLik;
      }
    }

    // Rank regimes by total log-likelihood
    const ranked = regimeLikelihoods.map((ll, r) => ({regime: r, name: M.regimeNames[r], logLik: ll}));
    ranked.sort((a, b) => b.logLik - a.logLik);

    log('Regime rankings by observation likelihood:');
    for (const r of ranked) {
      log(`  ${r.name.padEnd(8)}: logLik=${r.logLik.toFixed(1)}`);
    }

    // The regime with highest log-likelihood best explains the observations
    M.bestRegimeFromObs = ranked[0].regime;
    log(`\nBest regime from observations: ${ranked[0].name} (regime ${ranked[0].regime})`);

    // Also check settlement stats from observations vs MC
    if (M.observations[0].data.settlements) {
      log('\nSettlement stats comparison:');
      const obsPops = [], obsFoods = [], obsWealth = [], obsDef = [];
      for (const obs of M.observations) {
        for (const st of obs.data.settlements) {
          if (st.alive) {
            obsPops.push(st.population);
            obsFoods.push(st.food);
            obsWealth.push(st.wealth);
            obsDef.push(st.defense);
          }
        }
      }
      if (obsPops.length) {
        const avg = arr => arr.reduce((s,x) => s+x, 0) / arr.length;
        log(`  Observed: pop=${avg(obsPops).toFixed(2)} food=${avg(obsFoods).toFixed(2)} wealth=${avg(obsWealth).toFixed(2)} def=${avg(obsDef).toFixed(2)} (n=${obsPops.length})`);
      }
    }

    return ranked;
  };

  // ═══════════════════════════════════════════════════════════════════════
  // POST-ROUND GT ANALYSIS
  // ═══════════════════════════════════════════════════════════════════════
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
      log(`  Seed ${s}: score=${data.score}, GT ${data.ground_truth.length}x${data.ground_truth[0].length}x${data.ground_truth[0][0].length}`);
    }
    return M.realGT;
  };

  // Score our predictions against REAL GT with detailed analysis
  M.scoreVsRealGT = function(preds) {
    preds = preds || M.finalPreds;
    if (!M.realGT || !preds) { log('Need realGT and preds!'); return; }
    const results = [];
    for (let s = 0; s < SEEDS; s++) {
      if (!M.realGT[s] || !preds[s]) continue;
      const detail = M.scoreDetailed(preds[s], M.realGT[s].ground_truth);
      results.push({
        seed: s,
        ourScore: detail.score.toFixed(1),
        officialScore: M.realGT[s].score,
        weightedKL: detail.weightedKL.toFixed(4),
        dynamicCells: detail.dynamicCells,
        worstCell: detail.worstCell ? `(${detail.worstCell.y},${detail.worstCell.x}) kl=${detail.worstCell.kl.toFixed(3)}` : 'none'
      });
      log(`Seed ${s}: calc=${detail.score.toFixed(1)}, official=${M.realGT[s].score}, wKL=${detail.weightedKL.toFixed(4)}, dynCells=${detail.dynamicCells}`);
    }
    const avgScore = results.reduce((s, r) => s + parseFloat(r.ourScore), 0) / results.length;
    log(`Average score: ${avgScore.toFixed(1)}`);
    return results;
  };

  // Analyze GT to find regime-matching patterns
  M.analyzeGT = function() {
    if (!M.realGT) { log('Fetch GT first!'); return; }
    log('=== GT ANALYSIS ===');

    // For each regime, score against real GT
    const regimeVsGT = [];
    for (let r = 0; r < NR; r++) {
      let totalScore = 0;
      let seedScores = [];
      for (let s = 0; s < SEEDS; s++) {
        if (!M.realGT[s]) continue;
        const pred = M.pd[r][s];
        const sc = M.score(pred, M.realGT[s].ground_truth);
        totalScore += sc;
        seedScores.push(sc);
      }
      const avgSc = seedScores.length ? totalScore / seedScores.length : 0;
      regimeVsGT.push({regime: r, name: M.regimeNames[r], avg: avgSc, seeds: seedScores});
      log(`  Regime ${r} (${M.regimeNames[r]}): avg=${avgSc.toFixed(1)} seeds=[${seedScores.map(s=>s.toFixed(1)).join(',')}]`);
    }

    regimeVsGT.sort((a, b) => b.avg - a.avg);
    log('Best regimes for real GT:');
    for (const r of regimeVsGT.slice(0, 5)) {
      log(`  #${r.regime} ${r.name}: avg=${r.avg.toFixed(1)}`);
    }

    M.gtAnalysis = regimeVsGT;
    return regimeVsGT;
  };

  // Run additional sims for best-matching regimes (boost accuracy)
  M.boostRegime = async function(r, extraSims) {
    extraSims = extraSims || 150; // boost to ~200+ total
    const P = M.PS[r];
    const existingCount = simCounts[r] || 50;
    log(`Boosting regime ${r} (${M.regimeNames[r]}): +${extraSims} sims (total: ${existingCount + extraSims})`);

    for (let s = 0; s < SEEDS; s++) {
      for (let i = 0; i < extraSims; i++) {
        const rng = M.mkRng(s * 100000 + r * 10000 + (existingCount + i) * 7 + 137);
        const init = detail.initial_states[s];
        const fg = M.simGrid(init.grid, init.settlements, rng, P);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const c = M.t2c(fg[y][x]);
          M.rc[r][s][y][x][c] = (M.rc[r][s][y][x][c] || 0) + 1;
        }
      }
      await new Promise(ok => setTimeout(ok, 0));
    }

    simCounts[r] = existingCount + extraSims;

    // Rebuild distributions for this regime
    for (let s = 0; s < SEEDS; s++) {
      M.pd[r][s] = buildDist(r, s, PRED_ALPHA);
      M.gt[r][s] = buildDist(r, s, GT_ALPHA);
    }
    log(`  Regime ${r} boosted to ${simCounts[r]} sims. Distributions rebuilt.`);
  };

  // ═══════════════════════════════════════════════════════════════════════
  // GENERATE PREDICTIONS
  // ═══════════════════════════════════════════════════════════════════════
  M.generatePredictions = function(regimeList) {
    log('Generating per-seed optimized predictions...');
    const startW = new Array(NR).fill(1);
    M.perSeedWeights = [];
    M.finalPreds = {};

    for (let s = 0; s < SEEDS; s++) {
      const opt = M.optimizeSeed(s, startW, regimeList);
      M.perSeedWeights[s] = opt.weights;
      M.finalPreds[s] = M.blend(opt.weights, s);
      log(`  Seed ${s}: worst=${opt.worst}`);
    }

    // Validate
    let badSums = 0, floorViol = 0, minP = 1;
    for (let s = 0; s < SEEDS; s++) {
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const p = M.finalPreds[s][y][x];
        let sum = 0;
        for (let c = 0; c < 6; c++) {
          sum += p[c];
          if (p[c] < 0.0099) floorViol++;
          if (p[c] < minP) minP = p[c];
        }
        if (Math.abs(sum - 1.0) > 0.011) badSums++;
      }
    }
    log(`Validation: ${badSums} bad sums, ${floorViol} floor violations, minP=${minP.toFixed(4)}`);
    return M.finalPreds;
  };

  // Generate predictions focused on best-matching regimes from GT analysis
  M.generateFromGTAnalysis = function() {
    if (!M.gtAnalysis) { log('Run M.analyzeGT() first!'); return; }
    // Use top 5 regimes by GT match
    const topRegimes = M.gtAnalysis.slice(0, 7).map(r => r.regime);
    log(`Generating predictions focused on top GT-matching regimes: ${topRegimes.join(',')}`);
    return M.generatePredictions(topRegimes);
  };

  // ═══════════════════════════════════════════════════════════════════════
  // SUBMISSION (explicit only — NEVER auto-submits)
  // ═══════════════════════════════════════════════════════════════════════
  M.submitSeed = async function(s, preds) {
    preds = preds || M.finalPreds;
    log(`*** SUBMITTING seed ${s} ***`);
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
    log('*** SUBMITTING ALL SEEDS ***');
    const results = [];
    for (let s = 0; s < SEEDS; s++) {
      results.push(await M.submitSeed(s, preds));
      await new Promise(ok => setTimeout(ok, 600));
    }
    log(`Done: ${results.filter(r => r.status === 200 || r.status === 201).length}/${SEEDS} accepted.`);
    return results;
  };

  M.myScores = async function() {
    const resp = await fetch(`${BASE}/my-rounds`, {credentials:'include'});
    return await resp.json();
  };

  M.leaderboard = async function() {
    const resp = await fetch(`${BASE}/leaderboard`, {credentials:'include'});
    return await resp.json();
  };

  // ═══════════════════════════════════════════════════════════════════════
  // WORKFLOW: Complete next-round strategy
  // ═══════════════════════════════════════════════════════════════════════
  M.workflow = function() {
    log('=== RECOMMENDED WORKFLOW ===');
    log('');
    log('STEP 1: Check round status');
    log('  await _M.myScores()');
    log('  await _M.leaderboard()');
    log('');
    log('STEP 2: If previous round completed, fetch GT and calibrate');
    log('  await _M.fetchGT("previous-round-id")');
    log('  _M.analyzeGT()');
    log('  await _M.boostRegime(bestRegimeIndex, 150)');
    log('');
    log('STEP 3: Plan and execute observations (with user approval)');
    log('  _M.planQueries()');
    log('  await _M.executePhase(1)  // 18 queries: full map coverage');
    log('  _M.estimateParams()       // find best-matching regimes');
    log('  await _M.executePhase(2)  // 24 queries: settlement-dense regions');
    log('');
    log('STEP 4: Build hybrid predictions');
    log('  _M.buildObservationDist()');
    log('  _M.generatePredictions()  // MC-based');
    log('  // Then blend with observations: _M.hybridBlend(pred, seed)');
    log('');
    log('STEP 5: Submit (with user approval)');
    log('  await _M.submitAll()');
    log('');
    log('STEP 6: Re-submit if improved');
    log('  // Iterate: boost regimes, re-generate, re-submit');
  };

  // ═══════════════════════════════════════════════════════════════════════
  // SELF-TEST: Run simulator and analyze output quality
  // ═══════════════════════════════════════════════════════════════════════
  M.selfTest = function(seed, regimeIdx) {
    seed = seed || 0;
    regimeIdx = regimeIdx || 2; // balanced
    const P = M.PS[regimeIdx];
    const init = detail.initial_states[seed];
    log(`=== SELF-TEST: Regime ${regimeIdx} (${M.regimeNames[regimeIdx]}), Seed ${seed} ===`);
    log(`Initial: ${init.settlements.length} settlements`);

    // Run 5 quick sims and analyze
    const results = [];
    for (let i = 0; i < 5; i++) {
      const rng = M.mkRng(seed * 100000 + regimeIdx * 10000 + i * 7 + 42);
      const fg = M.simGrid(init.grid, init.settlements, rng, P);
      let ns = 0, np = 0, nr = 0, nf = 0, ne = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const t = fg[y][x];
        if (t === 1) ns++;
        else if (t === 2) np++;
        else if (t === 3) nr++;
        else if (t === 4) nf++;
        else if (t === 0 || t === 11) ne++;
      }
      results.push({settles: ns, ports: np, ruins: nr, forests: nf, empty: ne});
    }

    // Statistics
    const avg = (arr) => arr.reduce((s,x) => s+x, 0) / arr.length;
    const std = (arr) => { const m = avg(arr); return Math.sqrt(arr.reduce((s,x) => s+(x-m)**2, 0) / arr.length); };
    log(`  Settlements: avg=${avg(results.map(r=>r.settles)).toFixed(1)} ±${std(results.map(r=>r.settles)).toFixed(1)}`);
    log(`  Ports:       avg=${avg(results.map(r=>r.ports)).toFixed(1)} ±${std(results.map(r=>r.ports)).toFixed(1)}`);
    log(`  Ruins:       avg=${avg(results.map(r=>r.ruins)).toFixed(1)} ±${std(results.map(r=>r.ruins)).toFixed(1)}`);
    log(`  Forests:     avg=${avg(results.map(r=>r.forests)).toFixed(1)} ±${std(results.map(r=>r.forests)).toFixed(1)}`);
    log(`  Empty:       avg=${avg(results.map(r=>r.empty)).toFixed(1)} ±${std(results.map(r=>r.empty)).toFixed(1)}`);
    return results;
  };

  // Run self-test across all regimes for one seed
  M.selfTestAll = function(seed) {
    seed = seed || 0;
    log(`=== FULL SELF-TEST: Seed ${seed}, all ${NR} regimes ===`);
    const summary = [];
    for (let r = 0; r < NR; r++) {
      const results = M.selfTest(seed, r);
      const avg = (arr) => arr.reduce((s,x) => s+x, 0) / arr.length;
      summary.push({
        regime: r, name: M.regimeNames[r],
        settles: avg(results.map(x=>x.settles)).toFixed(1),
        ports: avg(results.map(x=>x.ports)).toFixed(1),
        ruins: avg(results.map(x=>x.ruins)).toFixed(1)
      });
    }
    log('\n=== SUMMARY ===');
    for (const s of summary) {
      log(`  ${s.name.padEnd(8)}: S=${s.settles} P=${s.ports} R=${s.ruins}`);
    }
    return summary;
  };

  // ═══════════════════════════════════════════════════════════════════════
  // SENSITIVITY ANALYSIS: How much does score change with parameter tweaks?
  // ═══════════════════════════════════════════════════════════════════════
  M.sensitivity = function(seed, baseRegime) {
    seed = seed || 0;
    baseRegime = baseRegime || 2;
    log(`=== SENSITIVITY ANALYSIS: Seed ${seed}, Base regime ${baseRegime} ===`);

    const P = M.PS[baseRegime];
    const basePred = M.pd[baseRegime][seed];

    // Test scoring sensitivity to each regime
    const crossScores = [];
    for (let r = 0; r < NR; r++) {
      const gt = M.gt[r][seed];
      const sc = M.score(basePred, gt);
      crossScores.push({regime: r, name: M.regimeNames[r], score: sc});
    }
    crossScores.sort((a,b) => a.score - b.score);
    log('Cross-regime scores (worst-case tells us sensitivity):');
    for (const cs of crossScores) {
      log(`  vs ${cs.name.padEnd(8)}: ${cs.score.toFixed(1)}`);
    }

    // Find which cells contribute most to KL divergence
    const worstRegime = crossScores[0].regime;
    const worstGT = M.gt[worstRegime][seed];
    const cellKLs = [];
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const g = worstGT[y][x];
        let ent = 0;
        for (let c = 0; c < 6; c++) if (g[c] > 1e-6) ent -= g[c] * Math.log(g[c]);
        if (ent < 0.01) continue;
        let kl = 0;
        for (let c = 0; c < 6; c++) {
          if (g[c] > 1e-6) kl += g[c] * Math.log(g[c] / Math.max(basePred[y][x][c], 1e-10));
        }
        if (kl > 0) cellKLs.push({y, x, kl, ent, contrib: kl * ent, gt: g, pred: basePred[y][x]});
      }
    }
    cellKLs.sort((a,b) => b.contrib - a.contrib);
    log(`\nTop 10 worst cells (vs ${M.regimeNames[worstRegime]}):`);
    for (const c of cellKLs.slice(0, 10)) {
      const gtStr = c.gt.map(v => v.toFixed(2)).join(',');
      const prStr = c.pred.map(v => v.toFixed(2)).join(',');
      log(`  (${c.y},${c.x}): contrib=${c.contrib.toFixed(4)} kl=${c.kl.toFixed(3)} ent=${c.ent.toFixed(3)}`);
      log(`    GT:   [${gtStr}]`);
      log(`    Pred: [${prStr}]`);
    }

    return {crossScores, worstCells: cellKLs.slice(0, 20)};
  };

  // ═══════════════════════════════════════════════════════════════════════
  // DEEP CALIBRATION: Use real GT to find optimal parameters
  // ═══════════════════════════════════════════════════════════════════════
  M.deepCalibrate = async function() {
    if (!M.realGT) { log('Fetch real GT first with M.fetchGT()!'); return; }
    log('=== DEEP CALIBRATION ===');

    // Step 1: Score all current regimes against real GT
    log('Step 1: Scoring all regimes against real GT...');
    const regimeScores = M.analyzeGT();

    // Step 2: Find top 3 regimes
    const top3 = regimeScores.slice(0, 3);
    log(`\nTop 3 regimes: ${top3.map(r => `${r.name}(${r.avg.toFixed(1)})`).join(', ')}`);

    // Step 3: Boost top 3 regimes with more simulations
    log('\nStep 3: Boosting top regimes...');
    for (const r of top3) {
      const current = simCounts[r.regime] || 50;
      if (current < 200) {
        await M.boostRegime(r.regime, 200 - current);
      }
    }

    // Step 4: Create interpolated regimes between top matches
    log('\nStep 4: Creating targeted interpolations...');
    const r1 = top3[0].regime, r2 = top3[1].regime;
    const extraRegimes = [];
    for (const t of [0.25, 0.5, 0.75]) {
      const newP = interp(M.PS[r1], M.PS[r2], t);
      const newIdx = M.PS.length;
      M.PS.push(newP);
      M.regimeNames.push(`Cal${t*100}`);
      simCounts[newIdx] = 100;
      extraRegimes.push(newIdx);

      // Run MC for new regime
      M.rc[newIdx] = {};
      for (let s = 0; s < SEEDS; s++) {
        const counts = [];
        for (let y = 0; y < H; y++) {
          counts[y] = [];
          for (let x = 0; x < W; x++) counts[y][x] = {};
        }
        for (let i = 0; i < 100; i++) {
          const rng = M.mkRng(s * 100000 + newIdx * 10000 + i * 7 + 42);
          const fg = M.simGrid(detail.initial_states[s].grid, detail.initial_states[s].settlements, rng, newP);
          for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const c = M.t2c(fg[y][x]);
            counts[y][x][c] = (counts[y][x][c] || 0) + 1;
          }
        }
        M.rc[newIdx][s] = counts;
      }
      M.pd[newIdx] = {}; M.gt[newIdx] = {};
      for (let s = 0; s < SEEDS; s++) {
        M.pd[newIdx][s] = buildDist(newIdx, s, PRED_ALPHA);
        M.gt[newIdx][s] = buildDist(newIdx, s, GT_ALPHA);
      }

      // Score against real GT
      let totalSc = 0;
      for (let s = 0; s < SEEDS; s++) {
        if (!M.realGT[s]) continue;
        totalSc += M.score(M.pd[newIdx][s], M.realGT[s].ground_truth);
      }
      const avgSc = totalSc / SEEDS;
      log(`  Cal${t*100}: avg score vs GT = ${avgSc.toFixed(1)}`);
      await new Promise(ok => setTimeout(ok, 0));
    }

    // Step 5: Re-optimize with expanded regime set
    log('\nStep 5: Re-optimizing with calibrated regimes...');
    // Include all original + calibrated regimes
    return M.generatePredictions();
  };

  // ═══════════════════════════════════════════════════════════════════════
  // PARAMETER GRID SEARCH: Fine-tune individual parameters against GT
  // ═══════════════════════════════════════════════════════════════════════
  M.paramSearch = async function(baseRegime, paramName, values) {
    if (!M.realGT) { log('Need real GT!'); return; }
    baseRegime = baseRegime || 2;
    log(`=== PARAM SEARCH: ${paramName} on regime ${baseRegime} ===`);

    const results = [];
    for (const val of values) {
      const P = {...M.PS[baseRegime], [paramName]: val};
      let totalSc = 0;
      for (let s = 0; s < SEEDS; s++) {
        if (!M.realGT[s]) continue;
        // Quick MC: 30 sims
        const counts = [];
        for (let y = 0; y < H; y++) {
          counts[y] = [];
          for (let x = 0; x < W; x++) counts[y][x] = {};
        }
        for (let i = 0; i < 30; i++) {
          const rng = M.mkRng(s * 100000 + 999 * 10000 + i);
          const fg = M.simGrid(detail.initial_states[s].grid, detail.initial_states[s].settlements, rng, P);
          for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const c = M.t2c(fg[y][x]);
            counts[y][x][c] = (counts[y][x][c] || 0) + 1;
          }
        }
        // Build dist and score
        const dist = [];
        for (let y = 0; y < H; y++) {
          dist[y] = [];
          for (let x = 0; x < W; x++) {
            const cts = counts[y][x];
            const p = new Array(6);
            let sm = 0;
            for (let k = 0; k < 6; k++) { p[k] = (cts[k] || 0) + PRED_ALPHA; sm += p[k]; }
            for (let k = 0; k < 6; k++) p[k] /= sm;
            dist[y][x] = p;
          }
        }
        totalSc += M.score(dist, M.realGT[s].ground_truth);
      }
      const avgSc = totalSc / SEEDS;
      results.push({val, score: avgSc});
      log(`  ${paramName}=${val}: score=${avgSc.toFixed(1)}`);
    }

    results.sort((a,b) => b.score - a.score);
    log(`\nBest ${paramName}: ${results[0].val} (score=${results[0].score.toFixed(1)})`);
    return results;
  };

  // Convenience: search key parameters
  M.searchKeyParams = async function(baseRegime) {
    baseRegime = baseRegime || 2;
    log('=== SEARCHING KEY PARAMETERS ===');
    const searches = [
      ['foodRadius', [1, 2, 3]],
      ['expandChance', [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]],
      ['raidChance', [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]],
      ['winterBase', [0.5, 0.65, 0.75, 0.85, 1.0, 1.2, 1.5]],
      ['conquerChance', [0.02, 0.05, 0.08, 0.12, 0.18, 0.25]],
      ['collapseChance', [0.1, 0.2, 0.3, 0.4, 0.5]],
      ['portChance', [0.03, 0.06, 0.1, 0.15, 0.2]],
      ['destroyOnConquest', [0, 0.05, 0.1, 0.15, 0.25, 0.4]],
    ];
    const bestParams = {};
    for (const [param, vals] of searches) {
      const res = await M.paramSearch(baseRegime, param, vals);
      bestParams[param] = res[0].val;
      await new Promise(ok => setTimeout(ok, 0));
    }
    log('\n=== BEST PARAMETERS FOUND ===');
    for (const [k, v] of Object.entries(bestParams)) {
      log(`  ${k}: ${v}`);
    }
    M.bestParams = bestParams;
    return bestParams;
  };

  // Create a custom regime from searched parameters and run full MC
  M.createCalibratedRegime = async function(baseRegime, overrides) {
    baseRegime = baseRegime || 2;
    overrides = overrides || M.bestParams || {};
    const P = {...M.PS[baseRegime], ...overrides};
    const newIdx = M.PS.length;
    M.PS.push(P);
    M.regimeNames.push('Calibrated');
    simCounts[newIdx] = 250;

    log(`Creating calibrated regime ${newIdx} with 250 sims...`);
    M.rc[newIdx] = {};
    for (let s = 0; s < SEEDS; s++) {
      const counts = [];
      for (let y = 0; y < H; y++) {
        counts[y] = [];
        for (let x = 0; x < W; x++) counts[y][x] = {};
      }
      for (let i = 0; i < 250; i++) {
        const rng = M.mkRng(s * 100000 + newIdx * 10000 + i * 7 + 42);
        const fg = M.simGrid(detail.initial_states[s].grid, detail.initial_states[s].settlements, rng, P);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const c = M.t2c(fg[y][x]);
          counts[y][x][c] = (counts[y][x][c] || 0) + 1;
        }
      }
      M.rc[newIdx][s] = counts;
      await new Promise(ok => setTimeout(ok, 0));
    }
    M.pd[newIdx] = {}; M.gt[newIdx] = {};
    for (let s = 0; s < SEEDS; s++) {
      M.pd[newIdx][s] = buildDist(newIdx, s, PRED_ALPHA);
      M.gt[newIdx][s] = buildDist(newIdx, s, GT_ALPHA);
    }

    // Score against real GT if available
    if (M.realGT) {
      let totalSc = 0;
      for (let s = 0; s < SEEDS; s++) {
        if (!M.realGT[s]) continue;
        const sc = M.score(M.pd[newIdx][s], M.realGT[s].ground_truth);
        log(`  Seed ${s}: ${sc.toFixed(1)}`);
        totalSc += sc;
      }
      log(`  Calibrated regime avg: ${(totalSc/SEEDS).toFixed(1)}`);
    }

    log(`Calibrated regime ${newIdx} ready.`);
    return newIdx;
  };

  // ═══════════════════════════════════════════════════════════════════════
  // PERFECT PREDICTION: Direct single-regime prediction (no blending)
  // When we find the right parameters, use ONLY that regime
  // ═══════════════════════════════════════════════════════════════════════
  M.singleRegimePreds = function(regimeIdx) {
    log(`Generating single-regime predictions from regime ${regimeIdx} (${M.regimeNames[regimeIdx]})...`);
    M.finalPreds = {};
    for (let s = 0; s < SEEDS; s++) {
      const raw = M.pd[regimeIdx][s];
      const pred = [];
      for (let y = 0; y < H; y++) {
        pred[y] = [];
        for (let x = 0; x < W; x++) {
          const p = [...raw[y][x]];
          // Floor enforcement
          for (let c = 0; c < 6; c++) p[c] = Math.max(FLOOR, p[c]);
          let sum = 0; for (let c = 0; c < 6; c++) sum += p[c];
          for (let c = 0; c < 6; c++) p[c] /= sum;
          // Precision fix
          let sum2 = 0, maxIdx = 0, maxVal = 0;
          for (let c = 0; c < 6; c++) {
            p[c] = parseFloat(p[c].toFixed(6));
            sum2 += p[c];
            if (p[c] > maxVal) { maxVal = p[c]; maxIdx = c; }
          }
          p[maxIdx] = parseFloat((p[maxIdx] + (1.0 - sum2)).toFixed(6));
          pred[y][x] = p;
        }
      }
      M.finalPreds[s] = pred;
    }

    // Score against real GT if available
    if (M.realGT) {
      for (let s = 0; s < SEEDS; s++) {
        if (!M.realGT[s]) continue;
        const sc = M.score(M.finalPreds[s], M.realGT[s].ground_truth);
        log(`  Seed ${s}: ${sc.toFixed(1)} vs real GT`);
      }
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

  log('=== MEGA SOLVER v4 READY ===');
  log(`${NR} regimes × ${SEEDS} seeds, v4 simulator (conquest-fix, dispersal, tech, ports)`);
  log('');
  log('Key commands:');
  log('  _M.workflow()              — show full workflow');
  log('  _M.generatePredictions()   — optimize + generate predictions');
  log('  _M.submitAll()             — submit all seeds (NEEDS USER APPROVAL)');
  log('  _M.observe(seed, x, y, w, h) — observe viewport (NEEDS USER APPROVAL)');
  log('  _M.planQueries()           — plan observation strategy');
  log('  _M.budget()                — check query budget');
  log('  _M.fetchGT(roundId)        — fetch ground truth (post-round)');
  log('  _M.scoreVsRealGT()         — compare preds vs real GT');
  log('  _M.analyzeGT()             — find best regime for real GT');
  log('  _M.boostRegime(r, n)       — add n sims to regime r');
  log('  _M.myScores()              — check our scores');
  log('  _M.leaderboard()           — check leaderboard');
  return M;
})();
