#!/usr/bin/env node
/**
 * Node.js test harness for Astar Island solver v4
 * Runs simulator locally without browser, validates scoring pipeline.
 *
 * Tests:
 * 1. Self-consistency (split-half: score should approach 100 with more sims)
 * 2. Cross-regime sensitivity (how different are regimes from each other)
 * 3. Convergence (how many sims needed for stable predictions)
 * 4. Simulator output sanity checks
 */

const H = 40, W = 40, FLOOR = 0.01, PRED_ALPHA = 0.15, GT_ALPHA = 0.08;

// ═══════════════════════════════════════════════════════════════════════
// RNG (same as v4 — Mulberry32)
// ═══════════════════════════════════════════════════════════════════════
function mkRng(seed) {
  let t = seed | 0;
  return function() {
    t = (t + 0x6D2B79F5) | 0;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

// ═══════════════════════════════════════════════════════════════════════
// SETTLEMENT
// ═══════════════════════════════════════════════════════════════════════
function Settle(x, y, hp, al) {
  this.x = x; this.y = y; this.pop = 1; this.food = 0.5; this.wealth = 0;
  this.defense = 0.5; this.tech = 0; this.hasPort = !!hp; this.hasLongship = false;
  this.alive = al !== false; this.ownerId = null;
}

// ═══════════════════════════════════════════════════════════════════════
// TERRAIN CODES: 0=empty/plains, 1=settlement, 2=port, 3=ruin, 4=forest, 5=mountain, 10=ocean, 11=plains
// ═══════════════════════════════════════════════════════════════════════
function t2c(t) {
  if (t === 10 || t === 11 || t === 0) return 0; // plains/ocean/empty → class 0
  if (t === 1) return 1; // settlement
  if (t === 2) return 2; // port
  if (t === 3) return 3; // ruin
  if (t === 4) return 4; // forest
  if (t === 5) return 5; // mountain
  return 0;
}

function isCoastal(grid, x, y) {
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) return true;
  }
  return false;
}

// ═══════════════════════════════════════════════════════════════════════
// SIMULATOR (exact copy of v4)
// ═══════════════════════════════════════════════════════════════════════
function sim(initialGrid, initialSettlements, rng, P) {
  const grid = initialGrid.map(r => [...r]);
  const settles = initialSettlements.map((s, i) => {
    const ns = new Settle(s.x, s.y, s.has_port, s.alive);
    ns.ownerId = i;
    return ns;
  });

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
      const fr = P.foodRadius || 1;
      let fg = 0;
      for (let dy = -fr; dy <= fr; dy++) for (let dx = -fr; dx <= fr; dx++) {
        if (!dy && !dx) continue;
        const ny = s.y + dy, nx = s.x + dx;
        if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
        const t = grid[ny][nx];
        if (t === 4) fg += P.foodForest;
        else if (t === 11 || t === 0) fg += P.foodPlains;
      }
      s.food += fg;
      if (s.food > P.growthTh) {
        const techBonus = 1 + s.tech * 0.1;
        s.pop += 0.1 * (1 + s.wealth * 0.05) * techBonus;
        s.food -= P.growthTh * 0.5;
      }
      s.defense = Math.min(s.defense + 0.02 * s.pop * (1 + s.tech * 0.05), s.pop * 0.8);
      if (s.pop > 1.0 && s.wealth > 0.1) {
        s.tech = Math.min(s.tech + P.techGrowth * (1 + s.wealth * 0.02), P.techMax || 5);
      }
      if (s.hasPort && !s.hasLongship && s.wealth > P.longshipCost && rng() < P.longshipChance) {
        s.hasLongship = true;
        s.wealth -= P.longshipCost * 0.5;
      }
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
              if (!close) cands.push({x: nx, y: ny});
            }
          }
        if (cands.length) {
          const c = cands[Math.floor(rng() * cands.length)];
          grid[c.y][c.x] = 1;
          const ns = new Settle(c.x, c.y, false, true);
          ns.pop = 0.5; ns.food = s.food * 0.3;
          ns.tech = s.tech * 0.5; ns.ownerId = s.ownerId;
          settles.push(ns);
          s.food *= 0.5; s.pop *= 0.8;
        }
      }
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
        const range = a.hasLongship ? P.longRaidRange : P.raidRange;
        const isDesp = a.food < 0.3;
        if (rng() < (isDesp ? P.despRaid : P.raidChance)) {
          const tgts = aliveList.filter(t => t !== a && t.alive &&
            t.ownerId !== a.ownerId &&
            Math.abs(t.x - a.x) + Math.abs(t.y - a.y) <= range);
          if (tgts.length) {
            const tg = tgts[Math.floor(rng() * tgts.length)];
            const techAdv = (a.tech - tg.tech) * 0.1;
            const ap = a.pop * P.raidStr * (1 + a.wealth * 0.05 + techAdv);
            const dp = tg.pop * (1 + tg.defense * 0.3 + tg.tech * 0.05);
            if (ap > dp * (0.8 + rng() * 0.4)) {
              const st = tg.food * P.loot; a.food += st; tg.food -= st;
              a.wealth += tg.wealth * P.loot * 0.5;
              tg.wealth = Math.max(0, tg.wealth - tg.wealth * P.loot * 0.5);
              tg.defense *= 0.7;
              if (rng() < P.conquerChance) {
                tg.ownerId = a.ownerId;
                tg.defense *= 0.5;
                if (rng() < P.destroyOnConquest) {
                  tg.alive = false; grid[tg.y][tg.x] = 3;
                }
              }
            } else {
              a.defense *= 0.9;
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
          const s2 = ports[i], p = ports[j];
          if (Math.abs(p.x - s2.x) + Math.abs(p.y - s2.y) > P.tradeRange) continue;
          const sameFaction = s2.ownerId === p.ownerId;
          const tradeMul = sameFaction ? 1.0 : 0.5;
          const techMul = 1 + (s2.tech + p.tech) * 0.05;
          s2.food += P.tradeFood * 0.5 * tradeMul * techMul;
          p.food += P.tradeFood * 0.5 * tradeMul * techMul;
          s2.wealth += P.tradeWealth * tradeMul * techMul;
          p.wealth += P.tradeWealth * tradeMul * techMul;
          if (s2.tech > p.tech + 0.1) p.tech += (s2.tech - p.tech) * P.techDiffusion * tradeMul;
          if (p.tech > s2.tech + 0.1) s2.tech += (p.tech - s2.tech) * P.techDiffusion * tradeMul;
        }
      }
    }

    // ═══ WINTER PHASE ═══
    const sev = P.constWinter ? P.winterBase : P.winterBase + (rng() - 0.5) * P.winterVar;
    for (const s of alive()) {
      s.food -= sev * (0.8 + s.pop * 0.2);
      s.pop = Math.max(0.1, s.pop - sev * 0.05);
      if (s.food < P.collapseTh && rng() < P.collapseChance) {
        s.alive = false; grid[s.y][s.x] = 3;
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
      if (grid[y][x] === 3) {
        const nearSettles = settles.filter(s => s.alive &&
          Math.abs(s.x - x) + Math.abs(s.y - y) <= P.rebuildRange);
        const thrivingNear = nearSettles.filter(s => s.pop > 1.0 && s.food > 0.5);
        if (thrivingNear.length && rng() < P.rebuildChance) {
          if (isCoastal(grid, x, y) && rng() < P.portRestoreChance) {
            grid[y][x] = 2;
            const ns = new Settle(x, y, true, true);
            const patron = thrivingNear[Math.floor(rng() * thrivingNear.length)];
            ns.pop = patron.pop * 0.3; ns.food = patron.food * 0.2;
            ns.tech = patron.tech * 0.5; ns.ownerId = patron.ownerId;
            ns.hasPort = true; settles.push(ns);
            patron.pop *= 0.85; patron.food *= 0.7;
          } else {
            grid[y][x] = 1;
            const ns = new Settle(x, y, false, true);
            const patron = thrivingNear[Math.floor(rng() * thrivingNear.length)];
            ns.pop = patron.pop * 0.3; ns.food = patron.food * 0.2;
            ns.tech = patron.tech * 0.5; ns.ownerId = patron.ownerId;
            settles.push(ns);
            patron.pop *= 0.85; patron.food *= 0.7;
          }
        } else if (rng() < P.forestReclaim) {
          grid[y][x] = 4;
        } else if (rng() < P.plainsReclaim) {
          grid[y][x] = 11;
        }
      }
    }
  }
  return grid;
}

// ═══════════════════════════════════════════════════════════════════════
// PARAMETER REGIMES (exact copy of v4)
// ═══════════════════════════════════════════════════════════════════════
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

const PS = [
  conserv, mildCon, balanced, modAgg, aggressive, ultraH, ultraP,
  {...balanced, noTrade: true}, {...balanced, noExpand: true},
  {...balanced, noConflict: true}, {...balanced, constWinter: true},
  interp(modAgg, aggressive, 0.33), interp(modAgg, aggressive, 0.67),
  hiExpLo, interp(conserv, balanced, 0.5), interp(balanced, modAgg, 0.5),
  interp(mildCon, balanced, 0.5), interp(balanced, hiExpLo, 0.5),
  interp(modAgg, hiExpLo, 0.5), interp(aggressive, hiExpLo, 0.5),
  regime({...balanced, foodRadius:2, foodForest:.2, foodPlains:.06}),
  regime({...modAgg, foodRadius:2, foodForest:.22, foodPlains:.07}),
  regime({...aggressive, foodRadius:2, foodForest:.25, foodPlains:.08}),
];
const regimeNames = [
  'Conserv','MildCon','Balance','ModAgg','Aggress','UltraH','UltraP',
  'NoTrade','NoExpan','NoCnflt','CstWntr','MA-A33','MA-A67',
  'HiExpLo','Con-Bal','Bal-MA','MC-Bal','Bal-HiX','MA-HiX','Agg-HiX',
  'Bal-WR','MA-WR','Agg-WR'
];
const NR = PS.length;

// ═══════════════════════════════════════════════════════════════════════
// GENERATE REALISTIC INITIAL STATE
// ═══════════════════════════════════════════════════════════════════════
function generateInitialState(seed) {
  const rng = mkRng(seed * 999 + 12345);
  const grid = [];

  // Create a realistic Norse island
  for (let y = 0; y < H; y++) {
    grid[y] = [];
    for (let x = 0; x < W; x++) {
      // Ocean border (rough coastline)
      const distEdge = Math.min(x, y, W-1-x, H-1-y);
      const noise = (rng() - 0.5) * 3;
      if (distEdge + noise < 2) {
        grid[y][x] = 10; // ocean
      } else if (distEdge + noise < 4 && rng() < 0.3) {
        grid[y][x] = 10; // ocean inlets
      } else {
        // Land types
        const r = rng();
        if (r < 0.12) grid[y][x] = 5;      // mountain (12%)
        else if (r < 0.35) grid[y][x] = 4;  // forest (23%)
        else grid[y][x] = 11;                // plains (65%)
      }
    }
  }

  // Place initial settlements (5-8 settlements, some coastal)
  const settlements = [];
  const nSettles = 5 + Math.floor(rng() * 4);
  for (let i = 0; i < nSettles; i++) {
    let x, y, attempts = 0;
    do {
      x = 5 + Math.floor(rng() * (W - 10));
      y = 5 + Math.floor(rng() * (H - 10));
      attempts++;
    } while (attempts < 100 && (grid[y][x] === 10 || grid[y][x] === 5 ||
             settlements.some(s => Math.abs(s.x - x) + Math.abs(s.y - y) < 4)));

    const coastal = isCoastal(grid, x, y);
    const hasPort = coastal && rng() < 0.5;
    grid[y][x] = hasPort ? 2 : 1;
    settlements.push({x, y, has_port: hasPort, alive: true, population: 1, food: 0.5, wealth: 0, defense: 0.5});
  }

  return {grid, settlements};
}

// ═══════════════════════════════════════════════════════════════════════
// SCORING (exact copy of v4)
// ═══════════════════════════════════════════════════════════════════════
function buildDist(counts, nsim, alpha) {
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

function score(pred, gt) {
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
}

function scoreDetailed(pred, gt) {
  let totalKL = 0, totalEnt = 0, dynamicCells = 0, worstCell = null, worstKL = 0;
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
      if (kl > worstKL) { worstKL = kl; worstCell = {y, x, kl, ent}; }
    }
  }
  const wkl = totalEnt > 0 ? totalKL / totalEnt : 0;
  return {
    score: Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl))),
    weightedKL: wkl, dynamicCells, worstCell
  };
}

// Apply floor enforcement
function applyFloor(dist) {
  const pred = [];
  for (let y = 0; y < H; y++) {
    pred[y] = [];
    for (let x = 0; x < W; x++) {
      const p = [...dist[y][x]];
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
      for (let c = 0; c < 6; c++) p[c] = Math.max(FLOOR, parseFloat(p[c].toFixed(6)));
      let s = 0; for (let c = 0; c < 6; c++) s += p[c];
      for (let c = 0; c < 6; c++) p[c] /= s;
      let sum4 = 0, maxIdx = 0, maxVal = 0;
      for (let c = 0; c < 6; c++) {
        p[c] = parseFloat(p[c].toFixed(6));
        sum4 += p[c]; if (p[c] > maxVal) { maxVal = p[c]; maxIdx = c; }
      }
      p[maxIdx] = parseFloat((p[maxIdx] + (1.0 - sum4)).toFixed(6));
      pred[y][x] = p;
    }
  }
  return pred;
}

// ═══════════════════════════════════════════════════════════════════════
// MC RUNNER
// ═══════════════════════════════════════════════════════════════════════
function runMC(init, regimeIdx, nsim, seedOffset) {
  const P = PS[regimeIdx];
  const counts = [];
  for (let y = 0; y < H; y++) {
    counts[y] = [];
    for (let x = 0; x < W; x++) counts[y][x] = {};
  }
  for (let i = 0; i < nsim; i++) {
    const rng = mkRng((seedOffset || 0) * 100000 + regimeIdx * 10000 + i * 7 + 42);
    const fg = sim(init.grid, init.settlements, rng, P);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const c = t2c(fg[y][x]);
      counts[y][x][c] = (counts[y][x][c] || 0) + 1;
    }
  }
  return counts;
}

// ═══════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════
console.log('=== ASTAR ISLAND SOLVER v4 — NODE.JS TEST HARNESS ===\n');
console.log(`${NR} regimes defined\n`);

// Generate 3 synthetic initial states (simulating 3 seeds)
const NSEEDS = 3;
const inits = [];
for (let s = 0; s < NSEEDS; s++) {
  inits[s] = generateInitialState(s);
  const g = inits[s].grid;
  let ocean = 0, forest = 0, mountain = 0, plains = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (g[y][x] === 10) ocean++;
    else if (g[y][x] === 4) forest++;
    else if (g[y][x] === 5) mountain++;
    else if (g[y][x] === 11 || g[y][x] === 0) plains++;
  }
  console.log(`Seed ${s}: ${inits[s].settlements.length} settlements, ocean=${ocean} forest=${forest} mount=${mountain} plains=${plains}`);
}

// ═══════════════════════════════════════════════════════════════════════
// TEST 1: Single-sim sanity check
// ═══════════════════════════════════════════════════════════════════════
console.log('\n═══ TEST 1: SINGLE-SIM SANITY CHECK ═══');
for (let r = 0; r < 7; r++) { // core regimes only
  const rng = mkRng(42 + r);
  const grid = sim(inits[0].grid, inits[0].settlements, rng, PS[r]);
  let ns = 0, np = 0, nr = 0, nf = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (grid[y][x] === 1) ns++;
    else if (grid[y][x] === 2) np++;
    else if (grid[y][x] === 3) nr++;
    else if (grid[y][x] === 4) nf++;
  }
  console.log(`  ${regimeNames[r].padEnd(8)}: S=${String(ns).padStart(3)} P=${String(np).padStart(2)} R=${String(nr).padStart(3)} F=${nf}`);
}

// ═══════════════════════════════════════════════════════════════════════
// TEST 2: SELF-CONSISTENCY (split-half scoring)
// If we run N sims, use first N/2 as prediction and second N/2 as GT,
// the score should be high (near 100). This tests our pipeline correctness.
// ═══════════════════════════════════════════════════════════════════════
console.log('\n═══ TEST 2: SELF-CONSISTENCY (split-half) ═══');
console.log('  (Using same regime for pred and GT, split sims in half)');

for (const nsimTotal of [20, 50, 100, 200]) {
  const half = nsimTotal / 2;
  let totalScore = 0;
  let nTests = 0;

  for (const r of [2, 3, 4]) { // balanced, modAgg, aggressive
    for (let s = 0; s < NSEEDS; s++) {
      const P = PS[r];

      // First half: prediction
      const predCounts = [];
      for (let y = 0; y < H; y++) { predCounts[y] = []; for (let x = 0; x < W; x++) predCounts[y][x] = {}; }
      for (let i = 0; i < half; i++) {
        const rng = mkRng(s * 100000 + r * 10000 + i * 7 + 42);
        const fg = sim(inits[s].grid, inits[s].settlements, rng, P);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const c = t2c(fg[y][x]); predCounts[y][x][c] = (predCounts[y][x][c] || 0) + 1;
        }
      }

      // Second half: GT
      const gtCounts = [];
      for (let y = 0; y < H; y++) { gtCounts[y] = []; for (let x = 0; x < W; x++) gtCounts[y][x] = {}; }
      for (let i = half; i < nsimTotal; i++) {
        const rng = mkRng(s * 100000 + r * 10000 + i * 7 + 42);
        const fg = sim(inits[s].grid, inits[s].settlements, rng, P);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const c = t2c(fg[y][x]); gtCounts[y][x][c] = (gtCounts[y][x][c] || 0) + 1;
        }
      }

      const predDist = applyFloor(buildDist(predCounts, half, PRED_ALPHA));
      const gtDist = buildDist(gtCounts, half, GT_ALPHA);
      const sc = score(predDist, gtDist);
      totalScore += sc;
      nTests++;
    }
  }

  console.log(`  ${String(nsimTotal).padStart(3)} sims (${half}/${half} split): avg self-score = ${(totalScore / nTests).toFixed(1)}`);
}

// ═══════════════════════════════════════════════════════════════════════
// TEST 3: CROSS-REGIME SCORING
// How badly do we score when using regime A to predict regime B?
// This shows the "penalty" of wrong parameter guess.
// ═══════════════════════════════════════════════════════════════════════
console.log('\n═══ TEST 3: CROSS-REGIME SCORING MATRIX ═══');
console.log('  (Row=prediction regime, Col=GT regime)');

const coreRegimes = [0, 1, 2, 3, 4, 5, 6]; // 7 core regimes
const NSIM_TEST = 60;

// Pre-compute MC for core regimes
const mcData = {};
for (const r of coreRegimes) {
  mcData[r] = {};
  for (let s = 0; s < NSEEDS; s++) {
    mcData[r][s] = runMC(inits[s], r, NSIM_TEST, s);
  }
  process.stdout.write(`  Computing regime ${r} (${regimeNames[r]})...\r`);
}

// Build distributions
const predDists = {}, gtDists = {};
for (const r of coreRegimes) {
  predDists[r] = {};
  gtDists[r] = {};
  for (let s = 0; s < NSEEDS; s++) {
    predDists[r][s] = applyFloor(buildDist(mcData[r][s], NSIM_TEST, PRED_ALPHA));
    gtDists[r][s] = buildDist(mcData[r][s], NSIM_TEST, GT_ALPHA);
  }
}

// Score matrix
console.log(`\n  ${'Pred\\GT'.padEnd(8)} | ${coreRegimes.map(r => regimeNames[r].padEnd(7)).join(' | ')}`);
console.log(`  ${''.padEnd(8)}-+-${coreRegimes.map(() => ''.padEnd(7, '-')).join('-+-')}`);

for (const pr of coreRegimes) {
  const row = [];
  for (const gr of coreRegimes) {
    let totalSc = 0;
    for (let s = 0; s < NSEEDS; s++) {
      totalSc += score(predDists[pr][s], gtDists[gr][s]);
    }
    row.push((totalSc / NSEEDS).toFixed(1).padStart(7));
  }
  console.log(`  ${regimeNames[pr].padEnd(8)} | ${row.join(' | ')}`);
}

// ═══════════════════════════════════════════════════════════════════════
// TEST 4: WORST-CASE ANALYSIS
// If we use each regime, what's our worst score across all possible GTs?
// ═══════════════════════════════════════════════════════════════════════
console.log('\n═══ TEST 4: WORST-CASE & BEST-CASE ANALYSIS ═══');
for (const pr of coreRegimes) {
  let worst = 999, best = -1, worstR = -1, bestR = -1;
  for (const gr of coreRegimes) {
    let totalSc = 0;
    for (let s = 0; s < NSEEDS; s++) {
      totalSc += score(predDists[pr][s], gtDists[gr][s]);
    }
    const avg = totalSc / NSEEDS;
    if (avg < worst) { worst = avg; worstR = gr; }
    if (avg > best) { best = avg; bestR = gr; }
  }
  console.log(`  ${regimeNames[pr].padEnd(8)}: worst=${worst.toFixed(1)} (vs ${regimeNames[worstR]})  best=${best.toFixed(1)} (vs ${regimeNames[bestR]})`);
}

// ═══════════════════════════════════════════════════════════════════════
// TEST 5: BLENDED PREDICTION
// Optimize weights across regimes and see if blending helps
// ═══════════════════════════════════════════════════════════════════════
console.log('\n═══ TEST 5: BLENDED vs SINGLE REGIME ═══');

function blend(weights, seed, predDists, nRegimes) {
  const nr = nRegimes || coreRegimes.length;
  let totalW = 0;
  for (let i = 0; i < nr; i++) totalW += (weights[i] || 0);
  if (totalW === 0) totalW = 1;

  const pred = [];
  for (let y = 0; y < H; y++) {
    pred[y] = [];
    for (let x = 0; x < W; x++) {
      const p = [0, 0, 0, 0, 0, 0];
      for (let i = 0; i < nr; i++) {
        const w = weights[i] || 0;
        const r = coreRegimes[i];
        if (w === 0 || !predDists[r] || !predDists[r][seed]) continue;
        const rd = predDists[r][seed][y][x];
        for (let c = 0; c < 6; c++) p[c] += w * rd[c];
      }
      for (let c = 0; c < 6; c++) p[c] /= totalW;
      // Floor
      for (let c = 0; c < 6; c++) p[c] = Math.max(FLOOR, p[c]);
      let s = 0; for (let c = 0; c < 6; c++) s += p[c];
      for (let c = 0; c < 6; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

// Equal blend vs single best
const equalWeights = new Array(coreRegimes.length).fill(1);
for (const gt of coreRegimes) {
  let singleBest = -1, singleBestScore = -1;
  let blendedScore = 0;

  for (let s = 0; s < NSEEDS; s++) {
    // Equal blend
    const blendPred = blend(equalWeights, s, predDists, coreRegimes.length);
    blendedScore += score(blendPred, gtDists[gt][s]);

    // Best single
    for (const pr of coreRegimes) {
      const sc = score(predDists[pr][s], gtDists[gt][s]);
      if (sc > singleBestScore) { singleBestScore = sc; singleBest = pr; }
    }
  }

  console.log(`  GT=${regimeNames[gt].padEnd(8)}: equalBlend=${(blendedScore/NSEEDS).toFixed(1)}  bestSingle=${singleBestScore.toFixed(1)} (${regimeNames[singleBest]})`);
}

// ═══════════════════════════════════════════════════════════════════════
// TEST 6: CONVERGENCE — How many sims to approach score 100?
// ═══════════════════════════════════════════════════════════════════════
console.log('\n═══ TEST 6: CONVERGENCE TEST ═══');
console.log('  (Self-score with increasing sims — regime 2/Balanced, seed 0)');

const convergenceR = 2;
const convergenceSeed = 0;
// Use 400 sims as "ground truth"
const gtCounts400 = [];
for (let y = 0; y < H; y++) { gtCounts400[y] = []; for (let x = 0; x < W; x++) gtCounts400[y][x] = {}; }
for (let i = 0; i < 400; i++) {
  const rng = mkRng(convergenceSeed * 100000 + convergenceR * 10000 + (1000 + i) * 7 + 42);
  const fg = sim(inits[convergenceSeed].grid, inits[convergenceSeed].settlements, rng, PS[convergenceR]);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const c = t2c(fg[y][x]); gtCounts400[y][x][c] = (gtCounts400[y][x][c] || 0) + 1;
  }
}
const gt400 = buildDist(gtCounts400, 400, GT_ALPHA);

for (const nsim of [10, 20, 30, 50, 80, 100, 150, 200, 300]) {
  const counts = [];
  for (let y = 0; y < H; y++) { counts[y] = []; for (let x = 0; x < W; x++) counts[y][x] = {}; }
  for (let i = 0; i < nsim; i++) {
    const rng = mkRng(convergenceSeed * 100000 + convergenceR * 10000 + i * 7 + 42);
    const fg = sim(inits[convergenceSeed].grid, inits[convergenceSeed].settlements, rng, PS[convergenceR]);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const c = t2c(fg[y][x]); counts[y][x][c] = (counts[y][x][c] || 0) + 1;
    }
  }
  const predDist = applyFloor(buildDist(counts, nsim, PRED_ALPHA));
  const sc = score(predDist, gt400);
  const detail = scoreDetailed(predDist, gt400);
  console.log(`  ${String(nsim).padStart(3)} sims: score=${sc.toFixed(1)}  wKL=${detail.weightedKL.toFixed(4)}  dynCells=${detail.dynamicCells}${detail.worstCell ? `  worst=(${detail.worstCell.y},${detail.worstCell.x})` : ''}`);
}

// ═══════════════════════════════════════════════════════════════════════
// TEST 7: ALPHA SENSITIVITY
// How does Dirichlet smoothing alpha affect scores?
// ═══════════════════════════════════════════════════════════════════════
console.log('\n═══ TEST 7: DIRICHLET ALPHA SENSITIVITY ═══');
const nsimAlpha = 100;
const alphaCountsPred = [];
for (let y = 0; y < H; y++) { alphaCountsPred[y] = []; for (let x = 0; x < W; x++) alphaCountsPred[y][x] = {}; }
for (let i = 0; i < nsimAlpha; i++) {
  const rng = mkRng(0 * 100000 + 2 * 10000 + i * 7 + 42);
  const fg = sim(inits[0].grid, inits[0].settlements, rng, PS[2]);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const c = t2c(fg[y][x]); alphaCountsPred[y][x][c] = (alphaCountsPred[y][x][c] || 0) + 1;
  }
}

for (const alpha of [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]) {
  const predDist = applyFloor(buildDist(alphaCountsPred, nsimAlpha, alpha));
  const sc = score(predDist, gt400);
  console.log(`  alpha=${String(alpha).padStart(4)}: score=${sc.toFixed(1)}`);
}

console.log('\n═══ ALL TESTS COMPLETE ═══');
console.log('\nKey insights:');
console.log('- Self-consistency score tells you pipeline correctness');
console.log('- Cross-regime matrix shows penalty for wrong parameters');
console.log('- Convergence test shows how many sims you need');
console.log('- Blended vs single shows when to blend vs commit to one regime');
