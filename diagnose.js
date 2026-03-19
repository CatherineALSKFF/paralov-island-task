#!/usr/bin/env node
/**
 * Diagnose why self-consistency score plateaus at ~89
 *
 * Hypothesis testing:
 * H1: Simulation variance is too high (cells too noisy)
 * H2: Dirichlet alpha interaction (pred vs GT alpha mismatch)
 * H3: Floor enforcement stealing probability mass
 * H4: Too few dynamic cells / wrong entropy threshold
 * H5: Need much more sims (1000+)
 */

const H = 40, W = 40, FLOOR = 0.01, PRED_ALPHA = 0.15, GT_ALPHA = 0.08;

function mkRng(seed) {
  let t = seed | 0;
  return function() {
    t = (t + 0x6D2B79F5) | 0;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function Settle(x, y, hp, al) {
  this.x = x; this.y = y; this.pop = 1; this.food = 0.5; this.wealth = 0;
  this.defense = 0.5; this.tech = 0; this.hasPort = !!hp; this.hasLongship = false;
  this.alive = al !== false; this.ownerId = null;
}

function t2c(t) {
  if (t === 10 || t === 11 || t === 0) return 0;
  if (t === 1) return 1; if (t === 2) return 2; if (t === 3) return 3;
  if (t === 4) return 4; if (t === 5) return 5; return 0;
}

function isCoastal(grid, x, y) {
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) return true;
  }
  return false;
}

// Exact v4 simulator (copy from node_test.js)
function sim(initialGrid, initialSettlements, rng, P) {
  const grid = initialGrid.map(r => [...r]);
  const settles = initialSettlements.map((s, i) => {
    const ns = new Settle(s.x, s.y, s.has_port, s.alive);
    ns.ownerId = i; return ns;
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
        s.pop += 0.1 * (1 + s.wealth * 0.05) * (1 + s.tech * 0.1);
        s.food -= P.growthTh * 0.5;
      }
      s.defense = Math.min(s.defense + 0.02 * s.pop * (1 + s.tech * 0.05), s.pop * 0.8);
      if (s.pop > 1.0 && s.wealth > 0.1) {
        s.tech = Math.min(s.tech + P.techGrowth * (1 + s.wealth * 0.02), P.techMax || 5);
      }
      if (s.hasPort && !s.hasLongship && s.wealth > P.longshipCost && rng() < P.longshipChance) {
        s.hasLongship = true; s.wealth -= P.longshipCost * 0.5;
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
          ns.pop = 0.5; ns.food = s.food * 0.3; ns.tech = s.tech * 0.5; ns.ownerId = s.ownerId;
          settles.push(ns); s.food *= 0.5; s.pop *= 0.8;
        }
      }
      if (!s.hasPort && rng() < P.portChance) {
        if (isCoastal(grid, s.x, s.y)) { s.hasPort = true; grid[s.y][s.x] = 2; }
      }
    }
    if (!P.noConflict) {
      const aliveList = alive();
      for (const a of aliveList) {
        if (!a.alive) continue;
        const range = a.hasLongship ? P.longRaidRange : P.raidRange;
        if (rng() < (a.food < 0.3 ? P.despRaid : P.raidChance)) {
          const tgts = aliveList.filter(t => t !== a && t.alive && t.ownerId !== a.ownerId && Math.abs(t.x - a.x) + Math.abs(t.y - a.y) <= range);
          if (tgts.length) {
            const tg = tgts[Math.floor(rng() * tgts.length)];
            const ap = a.pop * P.raidStr * (1 + a.wealth * 0.05 + (a.tech - tg.tech) * 0.1);
            const dp = tg.pop * (1 + tg.defense * 0.3 + tg.tech * 0.05);
            if (ap > dp * (0.8 + rng() * 0.4)) {
              const st = tg.food * P.loot; a.food += st; tg.food -= st;
              a.wealth += tg.wealth * P.loot * 0.5;
              tg.wealth = Math.max(0, tg.wealth - tg.wealth * P.loot * 0.5);
              tg.defense *= 0.7;
              if (rng() < P.conquerChance) {
                tg.ownerId = a.ownerId; tg.defense *= 0.5;
                if (rng() < P.destroyOnConquest) { tg.alive = false; grid[tg.y][tg.x] = 3; }
              }
            } else { a.defense *= 0.9; }
          }
        }
      }
    }
    if (!P.noTrade) {
      const ports = alive().filter(s => s.hasPort);
      for (let i = 0; i < ports.length; i++) for (let j = i + 1; j < ports.length; j++) {
        const s2 = ports[i], p = ports[j];
        if (Math.abs(p.x - s2.x) + Math.abs(p.y - s2.y) > P.tradeRange) continue;
        const tradeMul = s2.ownerId === p.ownerId ? 1.0 : 0.5;
        const techMul = 1 + (s2.tech + p.tech) * 0.05;
        s2.food += P.tradeFood * 0.5 * tradeMul * techMul;
        p.food += P.tradeFood * 0.5 * tradeMul * techMul;
        s2.wealth += P.tradeWealth * tradeMul * techMul;
        p.wealth += P.tradeWealth * tradeMul * techMul;
        if (s2.tech > p.tech + 0.1) p.tech += (s2.tech - p.tech) * P.techDiffusion * tradeMul;
        if (p.tech > s2.tech + 0.1) s2.tech += (p.tech - s2.tech) * P.techDiffusion * tradeMul;
      }
    }
    const sev = P.constWinter ? P.winterBase : P.winterBase + (rng() - 0.5) * P.winterVar;
    for (const s of alive()) {
      s.food -= sev * (0.8 + s.pop * 0.2);
      s.pop = Math.max(0.1, s.pop - sev * 0.05);
      if (s.food < P.collapseTh && rng() < P.collapseChance) {
        s.alive = false; grid[s.y][s.x] = 3;
        const nearby = settles.filter(n => n.alive && n.ownerId === s.ownerId &&
          Math.abs(n.x - s.x) + Math.abs(n.y - s.y) <= P.dispersalRange && n !== s);
        if (nearby.length > 0) {
          const ps = s.pop * P.dispersalFraction / nearby.length;
          const fs = Math.max(0, s.food * 0.5) / nearby.length;
          for (const n of nearby) { n.pop += ps; n.food += fs; }
        }
      }
    }
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (grid[y][x] === 3) {
        const nearSettles = settles.filter(s => s.alive && Math.abs(s.x - x) + Math.abs(s.y - y) <= P.rebuildRange);
        const thrivingNear = nearSettles.filter(s => s.pop > 1.0 && s.food > 0.5);
        if (thrivingNear.length && rng() < P.rebuildChance) {
          if (isCoastal(grid, x, y) && rng() < P.portRestoreChance) {
            grid[y][x] = 2;
            const ns = new Settle(x, y, true, true);
            const patron = thrivingNear[Math.floor(rng() * thrivingNear.length)];
            ns.pop = patron.pop * 0.3; ns.food = patron.food * 0.2; ns.tech = patron.tech * 0.5; ns.ownerId = patron.ownerId; ns.hasPort = true;
            settles.push(ns); patron.pop *= 0.85; patron.food *= 0.7;
          } else {
            grid[y][x] = 1;
            const ns = new Settle(x, y, false, true);
            const patron = thrivingNear[Math.floor(rng() * thrivingNear.length)];
            ns.pop = patron.pop * 0.3; ns.food = patron.food * 0.2; ns.tech = patron.tech * 0.5; ns.ownerId = patron.ownerId;
            settles.push(ns); patron.pop *= 0.85; patron.food *= 0.7;
          }
        } else if (rng() < P.forestReclaim) { grid[y][x] = 4; }
        else if (rng() < P.plainsReclaim) { grid[y][x] = 11; }
      }
    }
  }
  return grid;
}

// Balanced regime
const baseNew = {techGrowth:0.05, techMax:5, techDiffusion:0.1, longshipCost:0.5,
                 destroyOnConquest:0.15, dispersalRange:4, dispersalFraction:0.5,
                 portRestoreChance:0.4, plainsReclaim:0.03, foodRadius:1};
const balanced = {...baseNew, foodForest:.4,foodPlains:.12,growthTh:1.3,expandTh:3,expandPopTh:1.5,expandChance:.08,expandDist:3,portChance:.1,longshipChance:.08,raidRange:4,longRaidRange:8,raidChance:.2,despRaid:.25,raidStr:.55,loot:.35,conquerChance:.12,tradeRange:6,tradeFood:.3,tradeWealth:.2,winterBase:.85,winterVar:.45,collapseTh:-.8,collapseChance:.3,forestReclaim:.06,ruinDecay:.03,rebuildChance:.07,rebuildRange:3};
const aggressive = {...baseNew, foodForest:.5,foodPlains:.16,growthTh:.9,expandTh:2,expandPopTh:1,expandChance:.12,expandDist:4,portChance:.15,longshipChance:.12,raidRange:5,longRaidRange:10,raidChance:.25,despRaid:.3,raidStr:.65,loot:.45,conquerChance:.18,tradeRange:7,tradeFood:.4,tradeWealth:.3,winterBase:.65,winterVar:.35,collapseTh:-1.2,collapseChance:.2,forestReclaim:.04,ruinDecay:.02,rebuildChance:.1,rebuildRange:4};

// Generate initial state
function generateInitialState(seed) {
  const rng = mkRng(seed * 999 + 12345);
  const grid = [];
  for (let y = 0; y < H; y++) {
    grid[y] = [];
    for (let x = 0; x < W; x++) {
      const distEdge = Math.min(x, y, W-1-x, H-1-y);
      const noise = (rng() - 0.5) * 3;
      if (distEdge + noise < 2) grid[y][x] = 10;
      else if (distEdge + noise < 4 && rng() < 0.3) grid[y][x] = 10;
      else {
        const r = rng();
        if (r < 0.12) grid[y][x] = 5;
        else if (r < 0.35) grid[y][x] = 4;
        else grid[y][x] = 11;
      }
    }
  }
  const settlements = [];
  const nSettles = 5 + Math.floor(rng() * 4);
  for (let i = 0; i < nSettles; i++) {
    let x, y, attempts = 0;
    do {
      x = 5 + Math.floor(rng() * (W - 10)); y = 5 + Math.floor(rng() * (H - 10)); attempts++;
    } while (attempts < 100 && (grid[y][x] === 10 || grid[y][x] === 5 || settlements.some(s => Math.abs(s.x - x) + Math.abs(s.y - y) < 4)));
    const coastal = isCoastal(grid, x, y);
    const hasPort = coastal && rng() < 0.5;
    grid[y][x] = hasPort ? 2 : 1;
    settlements.push({x, y, has_port: hasPort, alive: true});
  }
  return {grid, settlements};
}

const init = generateInitialState(0);

// ═══════════════════════════════════════════════════════════════════════
// DIAGNOSIS 1: Per-cell variance analysis
// ═══════════════════════════════════════════════════════════════════════
console.log('═══ DIAGNOSIS 1: PER-CELL VARIANCE ═══');
console.log('Running 500 sims with balanced regime...');
const P = balanced;
const allCounts = [];
for (let y = 0; y < H; y++) { allCounts[y] = []; for (let x = 0; x < W; x++) allCounts[y][x] = {}; }

for (let i = 0; i < 500; i++) {
  const rng = mkRng(i * 7 + 42);
  const fg = sim(init.grid, init.settlements, rng, P);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const c = t2c(fg[y][x]);
    allCounts[y][x][c] = (allCounts[y][x][c] || 0) + 1;
  }
}

// Find cells with highest entropy (most uncertain)
const cellEntropies = [];
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  const c = allCounts[y][x];
  let total = 0;
  for (const k in c) total += c[k];
  if (total === 0) continue;
  let ent = 0;
  const probs = {};
  for (const k in c) {
    const p = c[k] / total;
    probs[k] = p;
    if (p > 0) ent -= p * Math.log(p);
  }
  if (ent > 0.01) {
    cellEntropies.push({y, x, ent, probs, total});
  }
}
cellEntropies.sort((a, b) => b.ent - a.ent);

console.log(`\nDynamic cells (entropy > 0.01): ${cellEntropies.length} / ${H*W}`);
console.log(`\nTop 20 highest-entropy cells (500 sims):`);
for (const ce of cellEntropies.slice(0, 20)) {
  const probStr = Object.entries(ce.probs).map(([k, v]) => `c${k}:${(v*100).toFixed(1)}%`).join(' ');
  console.log(`  (${ce.y},${ce.x}): H=${ce.ent.toFixed(3)} [${probStr}]`);
}

// ═══════════════════════════════════════════════════════════════════════
// DIAGNOSIS 2: Scoring with ZERO alpha (no smoothing)
// ═══════════════════════════════════════════════════════════════════════
console.log('\n═══ DIAGNOSIS 2: ALPHA EFFECT ON SELF-SCORE ═══');
console.log('(500 sims split 250/250, varying both pred and GT alpha)');

const half1 = [], half2 = [];
for (let y = 0; y < H; y++) {
  half1[y] = []; half2[y] = [];
  for (let x = 0; x < W; x++) { half1[y][x] = {}; half2[y][x] = {}; }
}
for (let i = 0; i < 250; i++) {
  const rng = mkRng(i * 7 + 42);
  const fg = sim(init.grid, init.settlements, rng, P);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const c = t2c(fg[y][x]); half1[y][x][c] = (half1[y][x][c] || 0) + 1;
  }
}
for (let i = 250; i < 500; i++) {
  const rng = mkRng(i * 7 + 42);
  const fg = sim(init.grid, init.settlements, rng, P);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const c = t2c(fg[y][x]); half2[y][x][c] = (half2[y][x][c] || 0) + 1;
  }
}

function buildDistRaw(counts, alpha) {
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

function applyFloorDist(dist) {
  const pred = [];
  for (let y = 0; y < H; y++) {
    pred[y] = [];
    for (let x = 0; x < W; x++) {
      const p = [...dist[y][x]];
      for (let c = 0; c < 6; c++) p[c] = Math.max(FLOOR, p[c]);
      let s = 0; for (let c = 0; c < 6; c++) s += p[c];
      for (let c = 0; c < 6; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

function scoreRaw(pred, gt) {
  let totalKL = 0, totalEnt = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
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
  const wkl = totalEnt > 0 ? totalKL / totalEnt : 0;
  return {score: Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl))), wkl};
}

console.log('  PredAlpha  GTAlpha  Floor  Score    wKL');
for (const predA of [0.01, 0.05, 0.1, 0.15, 0.3, 0.5, 1.0]) {
  for (const gtA of [0.01, 0.08, 0.15]) {
    const predDist = buildDistRaw(half1, predA);
    const gtDist = buildDistRaw(half2, gtA);

    // Without floor
    const noFloor = scoreRaw(predDist, gtDist);
    // With floor
    const withFloor = scoreRaw(applyFloorDist(predDist), gtDist);

    console.log(`  ${String(predA).padStart(8)}  ${String(gtA).padStart(7)}  no:${noFloor.score.toFixed(1).padStart(5)}  yes:${withFloor.score.toFixed(1).padStart(5)}  wKL=${noFloor.wkl.toFixed(5)}`);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// DIAGNOSIS 3: Tautological score (same sims for pred and GT)
// Should be EXACTLY 100 if pipeline is correct
// ═══════════════════════════════════════════════════════════════════════
console.log('\n═══ DIAGNOSIS 3: TAUTOLOGICAL SCORE (same data for pred and GT) ═══');
for (const alpha of [0.01, 0.05, 0.1, 0.15, 0.5]) {
  const dist = buildDistRaw(allCounts, alpha);
  const floored = applyFloorDist(dist);
  const sc1 = scoreRaw(dist, dist);
  const sc2 = scoreRaw(floored, dist);
  const sc3 = scoreRaw(floored, floored);
  console.log(`  alpha=${String(alpha).padStart(4)}: raw→raw=${sc1.score.toFixed(2)}  floor→raw=${sc2.score.toFixed(2)}  floor→floor=${sc3.score.toFixed(2)}`);
}

// ═══════════════════════════════════════════════════════════════════════
// DIAGNOSIS 4: What the real competition GT looks like
// The platform probably uses 1000+ sims. Simulate that.
// Score our N-sim prediction against 2000-sim "real GT"
// ═══════════════════════════════════════════════════════════════════════
console.log('\n═══ DIAGNOSIS 4: SCORE vs "REAL GT" (2000-sim ground truth, no smoothing) ═══');
console.log('Building 2000-sim ground truth...');

const gtCounts2000 = [];
for (let y = 0; y < H; y++) { gtCounts2000[y] = []; for (let x = 0; x < W; x++) gtCounts2000[y][x] = {}; }
for (let i = 0; i < 2000; i++) {
  const rng = mkRng(i * 13 + 9999);
  const fg = sim(init.grid, init.settlements, rng, P);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const c = t2c(fg[y][x]); gtCounts2000[y][x][c] = (gtCounts2000[y][x][c] || 0) + 1;
  }
}

// The REAL GT is probably the empirical distribution without Dirichlet smoothing,
// but with a minimum probability floor from their end
const realGT = buildDistRaw(gtCounts2000, 0.001); // near-zero smoothing for 2000 sims

console.log('\nScore of N-sim predictions vs 2000-sim GT:');
console.log('  NSim  PredAlpha  Floor?  Score    wKL');

for (const nsim of [30, 50, 80, 100, 150, 200, 300, 500]) {
  const counts = [];
  for (let y = 0; y < H; y++) { counts[y] = []; for (let x = 0; x < W; x++) counts[y][x] = {}; }
  for (let i = 0; i < nsim; i++) {
    const rng = mkRng(i * 7 + 42);
    const fg = sim(init.grid, init.settlements, rng, P);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const c = t2c(fg[y][x]); counts[y][x][c] = (counts[y][x][c] || 0) + 1;
    }
  }

  for (const predA of [0.05, 0.15, 0.5]) {
    const predDist = buildDistRaw(counts, predA);
    const noFloor = scoreRaw(predDist, realGT);
    const withFloor = scoreRaw(applyFloorDist(predDist), realGT);
    console.log(`  ${String(nsim).padStart(4)}  ${String(predA).padStart(9)}  no:${noFloor.score.toFixed(1).padStart(5)} yes:${withFloor.score.toFixed(1).padStart(5)}  wKL=${noFloor.wkl.toFixed(5)}`);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// DIAGNOSIS 5: What if we use 2000 sims for prediction too?
// This tests the THEORETICAL maximum score with our sim
// ═══════════════════════════════════════════════════════════════════════
console.log('\n═══ DIAGNOSIS 5: THEORETICAL MAX (2000-sim pred vs 2000-sim GT, different seeds) ═══');
const predCounts2000 = [];
for (let y = 0; y < H; y++) { predCounts2000[y] = []; for (let x = 0; x < W; x++) predCounts2000[y][x] = {}; }
for (let i = 0; i < 2000; i++) {
  const rng = mkRng(i * 7 + 42); // different seed series from GT
  const fg = sim(init.grid, init.settlements, rng, P);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const c = t2c(fg[y][x]); predCounts2000[y][x][c] = (predCounts2000[y][x][c] || 0) + 1;
  }
}

for (const predA of [0.001, 0.01, 0.05, 0.1, 0.15, 0.5]) {
  const predDist = applyFloorDist(buildDistRaw(predCounts2000, predA));
  const sc = scoreRaw(predDist, realGT);
  console.log(`  pred_alpha=${String(predA).padStart(5)}: score=${sc.score.toFixed(2)}  wKL=${sc.wkl.toFixed(6)}`);
}

// ═══════════════════════════════════════════════════════════════════════
// DIAGNOSIS 6: Per-cell KL contribution breakdown
// ═══════════════════════════════════════════════════════════════════════
console.log('\n═══ DIAGNOSIS 6: PER-CELL KL CONTRIBUTION (200 pred vs 2000 GT) ═══');
const pred200 = [];
for (let y = 0; y < H; y++) { pred200[y] = []; for (let x = 0; x < W; x++) pred200[y][x] = {}; }
for (let i = 0; i < 200; i++) {
  const rng = mkRng(i * 7 + 42);
  const fg = sim(init.grid, init.settlements, rng, P);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const c = t2c(fg[y][x]); pred200[y][x][c] = (pred200[y][x][c] || 0) + 1;
  }
}
const predDist200 = applyFloorDist(buildDistRaw(pred200, 0.15));

const cellContribs = [];
let totalWeightedKL = 0, totalEntropy = 0;
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  const g = realGT[y][x];
  let ent = 0;
  for (let c = 0; c < 6; c++) if (g[c] > 1e-6) ent -= g[c] * Math.log(g[c]);
  if (ent < 0.01) continue;
  let kl = 0;
  for (let c = 0; c < 6; c++) {
    if (g[c] > 1e-6) kl += g[c] * Math.log(g[c] / Math.max(predDist200[y][x][c], 1e-10));
  }
  kl = Math.max(0, kl);
  const contrib = kl * ent;
  totalWeightedKL += contrib;
  totalEntropy += ent;
  cellContribs.push({y, x, kl, ent, contrib, gt: g, pred: predDist200[y][x]});
}
cellContribs.sort((a, b) => b.contrib - a.contrib);

const wkl = totalEntropy > 0 ? totalWeightedKL / totalEntropy : 0;
console.log(`Total: wKL=${wkl.toFixed(5)}, dynamic_cells=${cellContribs.length}, score=${(100*Math.exp(-3*wkl)).toFixed(1)}`);
console.log(`\nTop 15 worst cells (highest KL×entropy contribution):`);
for (const cc of cellContribs.slice(0, 15)) {
  const gtStr = cc.gt.map(v => (v*100).toFixed(1) + '%').join(' ');
  const prStr = cc.pred.map(v => (v*100).toFixed(1) + '%').join(' ');
  const pctContrib = (cc.contrib / totalWeightedKL * 100).toFixed(1);
  console.log(`  (${cc.y},${cc.x}): KL=${cc.kl.toFixed(4)} H=${cc.ent.toFixed(3)} contrib=${cc.contrib.toFixed(5)} (${pctContrib}% of total)`);
  console.log(`    GT:   [${gtStr}]`);
  console.log(`    Pred: [${prStr}]`);
}

// Cumulative contribution
let cumContrib = 0;
for (let i = 0; i < cellContribs.length; i++) {
  cumContrib += cellContribs[i].contrib;
  if (i === 4 || i === 9 || i === 19 || i === 29 || i === cellContribs.length - 1) {
    console.log(`  Top ${i+1} cells: ${(cumContrib/totalWeightedKL*100).toFixed(1)}% of total KL`);
  }
}

console.log('\n═══ DIAGNOSIS COMPLETE ═══');
