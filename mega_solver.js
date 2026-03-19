/**
 * Mega Solver v2 — All-in-one Astar Island prediction engine.
 * Paste into browser console on app.ainm.no (must be logged in).
 *
 * Architecture:
 * - 13 simulation regimes (7 param + 4 mechanical + 2 intermediate)
 * - Per-seed minimax weight optimization
 * - Dirichlet-smoothed distributions (alpha=0.2)
 * - Entropy-weighted KL divergence scoring (exact competition formula)
 * - Strict probability floor enforcement (>=0.01, sum=1.0)
 *
 * Does NOT submit — call window._M.submitAll() when ready.
 */
(async function MegaSolver() {
  const BASE = 'https://api.ainm.no';
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
  const roundsResp = await fetch(`${BASE}/astar-island/rounds`, {credentials:'include'});
  const rounds = await roundsResp.json();
  const active = rounds.find(r => r.status === 'active');
  if (!active) { log('No active round!'); return; }

  const ROUND_ID = active.id;
  const detResp = await fetch(`${BASE}/astar-island/rounds/${ROUND_ID}`, {credentials:'include'});
  const detail = await detResp.json();
  M.d = detail;
  M.rid = ROUND_ID;
  const H = M.H = detail.map_height;
  const W = M.W = detail.map_width;
  const SEEDS = M.S = detail.seeds_count;
  log(`Round ${detail.round_number} (${ROUND_ID}), ${W}x${H}, ${SEEDS} seeds, closes: ${detail.closes_at}`);

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
      this.hasPort = !!hasPort; this.alive = alive !== false;
    }
  }

  // 13 parameter regimes
  const balanced = {foodForest:.4,foodPlains:.12,growthTh:1.3,expandTh:3,expandPopTh:1.5,expandChance:.08,expandDist:3,portChance:.1,longshipChance:.08,raidRange:4,longRaidRange:8,raidChance:.2,despRaid:.25,raidStr:.55,loot:.35,conquerChance:.12,tradeRange:6,tradeFood:.3,tradeWealth:.2,winterBase:.85,winterVar:.45,collapseTh:-.8,collapseChance:.3,forestReclaim:.06,ruinDecay:.03,rebuildChance:.07,rebuildRange:3};
  const modAgg = {foodForest:.45,foodPlains:.14,growthTh:1.1,expandTh:2.5,expandPopTh:1.3,expandChance:.1,expandDist:4,portChance:.12,longshipChance:.1,raidRange:4,longRaidRange:9,raidChance:.22,despRaid:.28,raidStr:.6,loot:.4,conquerChance:.15,tradeRange:6,tradeFood:.35,tradeWealth:.25,winterBase:.75,winterVar:.4,collapseTh:-1,collapseChance:.25,forestReclaim:.05,ruinDecay:.025,rebuildChance:.08,rebuildRange:4};
  const aggressive = {foodForest:.5,foodPlains:.16,growthTh:.9,expandTh:2,expandPopTh:1,expandChance:.12,expandDist:4,portChance:.15,longshipChance:.12,raidRange:5,longRaidRange:10,raidChance:.25,despRaid:.3,raidStr:.65,loot:.45,conquerChance:.18,tradeRange:7,tradeFood:.4,tradeWealth:.3,winterBase:.65,winterVar:.35,collapseTh:-1.2,collapseChance:.2,forestReclaim:.04,ruinDecay:.02,rebuildChance:.1,rebuildRange:4};

  function interp(p1, p2, t) {
    const r = {};
    for (const k of Object.keys(p1)) {
      r[k] = typeof p1[k] === 'number' ? p1[k]*(1-t) + p2[k]*t : p1[k];
    }
    return r;
  }

  M.PS = [
    // 0: Conservative
    {foodForest:.3,foodPlains:.08,growthTh:1.8,expandTh:4,expandPopTh:2,expandChance:.04,expandDist:3,portChance:.06,longshipChance:.05,raidRange:3,longRaidRange:7,raidChance:.15,despRaid:.2,raidStr:.5,loot:.3,conquerChance:.08,tradeRange:5,tradeFood:.2,tradeWealth:.15,winterBase:1,winterVar:.6,collapseTh:-.5,collapseChance:.4,forestReclaim:.08,ruinDecay:.05,rebuildChance:.05,rebuildRange:3},
    // 1: MildConservative
    {foodForest:.35,foodPlains:.1,growthTh:1.5,expandTh:3.5,expandPopTh:1.8,expandChance:.06,expandDist:3,portChance:.08,longshipChance:.06,raidRange:3,longRaidRange:7,raidChance:.18,despRaid:.22,raidStr:.5,loot:.3,conquerChance:.1,tradeRange:5,tradeFood:.25,tradeWealth:.18,winterBase:.9,winterVar:.5,collapseTh:-.6,collapseChance:.35,forestReclaim:.07,ruinDecay:.04,rebuildChance:.06,rebuildRange:3},
    // 2: Balanced
    balanced,
    // 3: ModerateAggressive
    modAgg,
    // 4: Aggressive
    aggressive,
    // 5: UltraHarsh
    {foodForest:.2,foodPlains:.05,growthTh:2.5,expandTh:5,expandPopTh:3,expandChance:.02,expandDist:2,portChance:.03,longshipChance:.03,raidRange:4,longRaidRange:8,raidChance:.25,despRaid:.3,raidStr:.7,loot:.4,conquerChance:.12,tradeRange:4,tradeFood:.15,tradeWealth:.1,winterBase:1.5,winterVar:.8,collapseTh:-.3,collapseChance:.55,forestReclaim:.12,ruinDecay:.08,rebuildChance:.03,rebuildRange:2},
    // 6: UltraProsperous
    {foodForest:.5,foodPlains:.18,growthTh:1,expandTh:2,expandPopTh:1.2,expandChance:.12,expandDist:4,portChance:.12,longshipChance:.1,raidRange:2,longRaidRange:5,raidChance:.08,despRaid:.1,raidStr:.3,loot:.2,conquerChance:.05,tradeRange:6,tradeFood:.35,tradeWealth:.25,winterBase:.6,winterVar:.3,collapseTh:-1.5,collapseChance:.15,forestReclaim:.04,ruinDecay:.02,rebuildChance:.1,rebuildRange:4},
    // 7: NoTrade
    {...balanced, noTrade: true},
    // 8: NoExpand
    {...balanced, noExpand: true},
    // 9: NoConflict
    {...balanced, noConflict: true},
    // 10: ConstantWinter
    {...balanced, constWinter: true},
    // 11: ModAgg→Aggressive 33%
    interp(modAgg, aggressive, 0.33),
    // 12: ModAgg→Aggressive 67%
    interp(modAgg, aggressive, 0.67),
  ];
  M.regimeNames = ['Conserv','MildCon','Balance','ModAgg','Aggress','UltraH','UltraP','NoTrade','NoExpan','NoCnflt','CstWntr','MA-A33','MA-A67'];

  // Core simulator: 50 years of Norse civilization
  M.sim = function(initialGrid, initialSettlements, rng, P) {
    const grid = initialGrid.map(r => [...r]);
    const settles = initialSettlements.map(s => new Settlement(s.x, s.y, s.has_port, s.alive));

    for (let year = 0; year < 50; year++) {
      // GROWTH PHASE
      for (const s of settles) {
        if (!s.alive) continue;
        let fg = 0;
        for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
          const ny = s.y+dy, nx = s.x+dx;
          if (ny<0||ny>=H||nx<0||nx>=W) continue;
          const t = grid[ny][nx];
          if (t === 4) fg += P.foodForest;
          else if (t === 11 || t === 0) fg += P.foodPlains;
        }
        s.food += fg;
        if (s.food > P.growthTh) { s.pop += 0.1*(1+s.wealth*0.05); s.food -= P.growthTh*0.5; }

        // Expand
        if (!P.noExpand && s.pop >= P.expandPopTh && s.food > P.expandTh && rng() < P.expandChance) {
          const cands = [];
          for (let dy = -P.expandDist; dy <= P.expandDist; dy++) for (let dx = -P.expandDist; dx <= P.expandDist; dx++) {
            if (!dy && !dx) continue;
            const ny = s.y+dy, nx = s.x+dx;
            if (ny<0||ny>=H||nx<0||nx>=W) continue;
            const t = grid[ny][nx];
            if (t===0||t===11||t===4) {
              let close = false;
              for (const o of settles) if (o.alive && Math.abs(o.x-nx)+Math.abs(o.y-ny)<2) { close=true; break; }
              if (!close) cands.push({x:nx, y:ny});
            }
          }
          if (cands.length) {
            const c = cands[Math.floor(rng()*cands.length)];
            grid[c.y][c.x] = 1;
            const ns = new Settlement(c.x, c.y, false, true);
            ns.pop = 0.5; ns.food = s.food * 0.3;
            settles.push(ns); s.food *= 0.5; s.pop *= 0.8;
          }
        }

        // Port upgrade
        if (!s.hasPort && rng() < P.portChance) {
          for (const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
            const ny = s.y+dy, nx = s.x+dx;
            if (ny>=0 && ny<H && nx>=0 && nx<W && grid[ny][nx] === 10) {
              s.hasPort = true; grid[s.y][s.x] = 2; break;
            }
          }
        }
      }

      // CONFLICT PHASE
      if (!P.noConflict) {
        const alive = settles.filter(s => s.alive);
        for (const a of alive) {
          if (!a.alive) continue;
          const range = (a.hasPort && rng() < P.longshipChance) ? P.longRaidRange : P.raidRange;
          if (rng() < (a.food < 0.3 ? P.despRaid : P.raidChance)) {
            const tgts = alive.filter(t => t!==a && t.alive && Math.abs(t.x-a.x)+Math.abs(t.y-a.y)<=range);
            if (tgts.length) {
              const tg = tgts[Math.floor(rng()*tgts.length)];
              const ap = a.pop*P.raidStr*(1+a.wealth*0.05), dp = tg.pop*(1+tg.wealth*0.03);
              if (ap > dp*(0.8+rng()*0.4)) {
                const st = tg.food*P.loot; a.food += st; tg.food -= st;
                a.wealth += tg.wealth*P.loot*0.5; tg.wealth = Math.max(0, tg.wealth-tg.wealth*P.loot*0.5);
                if (rng() < P.conquerChance) { tg.alive = false; grid[tg.y][tg.x] = 3; }
              }
            }
          }
        }
      }

      // TRADE PHASE
      if (!P.noTrade) {
        for (const s of settles) {
          if (!s.alive || !s.hasPort) continue;
          for (const p of settles) {
            if (p===s || !p.alive || !p.hasPort || Math.abs(p.x-s.x)+Math.abs(p.y-s.y)>P.tradeRange) continue;
            s.food += P.tradeFood*0.5; p.food += P.tradeFood*0.5;
            s.wealth += P.tradeWealth; p.wealth += P.tradeWealth;
          }
        }
      }

      // WINTER PHASE
      const sev = P.constWinter ? P.winterBase : P.winterBase + (rng()-0.5)*P.winterVar;
      for (const s of settles) {
        if (!s.alive) continue;
        s.food -= sev*(0.8+s.pop*0.2);
        s.pop = Math.max(0.1, s.pop - sev*0.05);
        if (s.food < P.collapseTh && rng() < P.collapseChance) { s.alive = false; grid[s.y][s.x] = 3; }
      }

      // ENVIRONMENT PHASE
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        if (grid[y][x] === 3) {
          if (rng() < P.forestReclaim) grid[y][x] = 4;
          const near = settles.some(s => s.alive && Math.abs(s.x-x)+Math.abs(s.y-y)<=P.rebuildRange);
          if (near && rng() < P.rebuildChance) { grid[y][x] = 1; settles.push(new Settlement(x, y, false, true)); }
        }
        if ((grid[y][x]===0 || grid[y][x]===11) && rng() < P.ruinDecay) {
          let nr = false;
          for (const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
            const ny2 = y+dy, nx2 = x+dx;
            if (ny2>=0 && ny2<H && nx2>=0 && nx2<W && grid[ny2][nx2]===3) nr = true;
          }
          if (nr) grid[y][x] = 4;
        }
      }
    }
    return grid;
  };

  function terrainToClass(t) {
    if (t===10||t===11||t===0) return 0;
    if (t===1) return 1; if (t===2) return 2;
    if (t===3) return 3; if (t===4) return 4;
    if (t===5) return 5; return 0;
  }
  M.t2c = terrainToClass;
  log('Simulator built (13 regimes).');

  // ═══════════════════════════════════════════════════════════════════
  // PHASE 3: MONTE CARLO SIMULATION
  // ═══════════════════════════════════════════════════════════════════
  log('Phase 3: Running Monte Carlo...');
  const simCounts = {0:60,1:60,2:60,3:60,4:50,5:60,6:50,7:30,8:30,9:30,10:30,11:20,12:20};
  M.rc = {};

  for (let r = 0; r < 13; r++) {
    M.rc[r] = {};
    const P = M.PS[r];
    const NSIM = simCounts[r];
    for (let s = 0; s < SEEDS; s++) {
      const counts = [];
      for (let y = 0; y < H; y++) {
        counts[y] = [];
        for (let x = 0; x < W; x++) counts[y][x] = {0:0,1:0,2:0,3:0,4:0,5:0};
      }
      const init = detail.initial_states[s];
      for (let i = 0; i < NSIM; i++) {
        const rng = M.mkRng(s*100000 + r*10000 + i);
        const g = init.grid.map(row => [...row]);
        const ss = init.settlements.map(st => ({...st}));
        const fg = M.sim(g, ss, rng, P);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          counts[y][x][terrainToClass(fg[y][x])]++;
        }
      }
      M.rc[r][s] = counts;
    }
    log(`  Regime ${r} (${M.regimeNames[r]}) done — ${NSIM} sims × ${SEEDS} seeds`);
    await new Promise(ok => setTimeout(ok, 0));
  }

  // ═══════════════════════════════════════════════════════════════════
  // PHASE 4: BUILD DISTRIBUTIONS & SCORING
  // ═══════════════════════════════════════════════════════════════════
  log('Phase 4: Building distributions...');

  function buildDist(regime, seed, alpha) {
    const counts = M.rc[regime][seed];
    const dist = [];
    for (let y = 0; y < H; y++) {
      dist[y] = [];
      for (let x = 0; x < W; x++) {
        const c = counts[y][x];
        const p = new Array(6);
        let sum = 0;
        for (let k = 0; k < 6; k++) { p[k] = c[k] + alpha; sum += p[k]; }
        for (let k = 0; k < 6; k++) p[k] /= sum;
        for (let k = 0; k < 6; k++) if (p[k] < FLOOR) p[k] = FLOOR;
        let s2 = 0;
        for (let k = 0; k < 6; k++) s2 += p[k];
        for (let k = 0; k < 6; k++) p[k] /= s2;
        dist[y][x] = p;
      }
    }
    return dist;
  }

  M.pd = {}; M.gt = {};
  for (let r = 0; r < 13; r++) {
    M.pd[r] = {}; M.gt[r] = {};
    for (let s = 0; s < SEEDS; s++) {
      M.pd[r][s] = buildDist(r, s, PRED_ALPHA);
      M.gt[r][s] = buildDist(r, s, GT_ALPHA);
    }
  }

  // Exact competition scoring formula
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

  // Blend with strict floor enforcement
  M.blend = function(weights, seed) {
    let totalW = 0;
    for (let r = 0; r < 13; r++) totalW += (weights[r] || 0);
    const pred = [];
    for (let y = 0; y < H; y++) {
      pred[y] = [];
      for (let x = 0; x < W; x++) {
        const p = [0,0,0,0,0,0];
        for (let r = 0; r < 13; r++) {
          const w = weights[r] || 0;
          if (w === 0) continue;
          const rd = M.pd[r][seed][y][x];
          for (let c = 0; c < 6; c++) p[c] += w * rd[c];
        }
        let s = 0;
        for (let c = 0; c < 6; c++) s += p[c];
        for (let c = 0; c < 6; c++) p[c] /= s;
        // Iterative floor enforcement
        for (let iter = 0; iter < 5; iter++) {
          let below = false;
          for (let c = 0; c < 6; c++) if (p[c] < FLOOR) { p[c] = FLOOR; below = true; }
          if (!below) break;
          s = 0;
          for (let c = 0; c < 6; c++) s += p[c];
          const exc = s - 1.0;
          if (Math.abs(exc) > 1e-10) {
            let above = 0;
            for (let c = 0; c < 6; c++) if (p[c] > FLOOR) above += p[c];
            for (let c = 0; c < 6; c++) if (p[c] > FLOOR) p[c] -= exc * (p[c] / above);
          }
        }
        // Final precision fix
        for (let c = 0; c < 6; c++) p[c] = Math.max(FLOOR, parseFloat(p[c].toFixed(6)));
        s = 0;
        for (let c = 0; c < 6; c++) s += p[c];
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

  log('Distributions and scoring built.');

  // ═══════════════════════════════════════════════════════════════════
  // PHASE 5: MINIMAX WEIGHT OPTIMIZATION
  // ═══════════════════════════════════════════════════════════════════
  log('Phase 5: Optimizing blend weights...');

  // Per-seed optimized weights (from coordinate descent minimax)
  M.perSeedWeights = [
    [3, 0, 0, 0, 4, 6, 1, 0.2, 0.2, 0, 0, 1.5, 2],    // seed 0
    [5, 0, 0, 0.2, 4, 6, 1, 0.2, 0.2, 0, 0, 1.5, 2],  // seed 1
    [3, 0, 0, 0, 4, 6, 0.5, 0.2, 0.2, 0, 0, 2, 2],    // seed 2
    [5, 0, 0, 0.2, 4, 6, 1, 0.2, 0.2, 0, 0, 1.5, 2],  // seed 3
    [3, 0, 0, 0, 4, 6, 1, 0.2, 0.2, 0, 0, 1.5, 2],    // seed 4
  ];

  // Generate final predictions
  M.finalPreds = {};
  for (let s = 0; s < SEEDS; s++) {
    M.finalPreds[s] = M.blend(M.perSeedWeights[s], s);
    const pred = M.finalPreds[s];
    const scores = [];
    for (let r = 0; r < 13; r++) scores.push(M.score(pred, M.gt[r][s]).toFixed(1));
    const nums = scores.map(Number);
    log(`  Seed ${s}: worst=${Math.min(...nums).toFixed(1)} avg=${(nums.reduce((a,b)=>a+b,0)/13).toFixed(1)}`);
  }

  // ═══════════════════════════════════════════════════════════════════
  // SUBMISSION FUNCTIONS
  // ═══════════════════════════════════════════════════════════════════
  M.submitAll = async function() {
    log('Submitting all seeds...');
    const results = [];
    for (let s = 0; s < SEEDS; s++) {
      const resp = await fetch(`${BASE}/astar-island/submit`, {
        method: 'POST', credentials: 'include',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ round_id: ROUND_ID, seed_index: s, prediction: M.finalPreds[s] })
      });
      const r = await resp.json();
      results.push({seed: s, status: resp.status, result: r});
      log(`  Seed ${s}: ${resp.status} — ${JSON.stringify(r).slice(0, 200)}`);
      await new Promise(ok => setTimeout(ok, 800));
    }
    M.submissions = results;
    log(`Done: ${results.filter(r => r.status===200||r.status===201).length}/${SEEDS} submitted.`);
    return results;
  };

  M.submitSeed = async function(s) {
    const resp = await fetch(`${BASE}/astar-island/submit`, {
      method: 'POST', credentials: 'include',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ round_id: ROUND_ID, seed_index: s, prediction: M.finalPreds[s] })
    });
    const r = await resp.json();
    log(`Seed ${s}: ${resp.status} — ${JSON.stringify(r).slice(0, 200)}`);
    return {seed: s, status: resp.status, result: r};
  };

  // Utility: evaluate any weight config
  M.evalW = function(weights) {
    let worst = 100, total = 0, count = 0;
    const perSeed = [];
    for (let s = 0; s < SEEDS; s++) {
      const pred = M.blend(weights, s);
      let sw = 100, ss = 0;
      for (let r = 0; r < 13; r++) {
        const sc = M.score(pred, M.gt[r][s]);
        if (sc < sw) sw = sc; if (sc < worst) worst = sc;
        ss += sc; total += sc; count++;
      }
      perSeed.push({seed: s, worst: sw.toFixed(1), avg: (ss/13).toFixed(1)});
    }
    return {globalWorst: worst.toFixed(1), globalAvg: (total/count).toFixed(1), perSeed};
  };

  log('=== MEGA SOLVER v2 READY ===');
  log('Call window._M.submitAll() to submit all seeds');
  log('Call window._M.submitSeed(N) to submit one seed');
  log('Call window._M.evalW([...weights]) to test a weight config');
  return M;
})();
