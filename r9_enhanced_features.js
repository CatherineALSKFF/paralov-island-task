#!/usr/bin/env node
/**
 * R9 ENHANCED FEATURES — More granular terrain features for better predictions
 * Adds: ruin count, port vs settlement distinction, terrain diversity, distance features
 *
 * Usage: node r9_enhanced_features.js <JWT>
 */
const fs = require('fs'), path = require('path'), https = require('https');
const H = 40, W = 40, SEEDS = 5, C = 6;
const DD = path.join(__dirname, 'data');
const TOKEN = process.argv[2] || '';
const BASE = 'https://api.ainm.no/astar-island';

if (!TOKEN) { console.log('Usage: node r9_enhanced_features.js <JWT>'); process.exit(1); }

function api(m, p, b) { return new Promise((res, rej) => {
  const u = new URL(BASE + p); const pl = b ? JSON.stringify(b) : null;
  const o = { hostname: u.hostname, path: u.pathname + u.search, method: m,
    headers: { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' } };
  if (pl) o.headers['Content-Length'] = Buffer.byteLength(pl);
  const r = https.request(o, re => { let d = ''; re.on('data', c => d += c);
    re.on('end', () => { try { res({ ok: re.statusCode < 300, status: re.statusCode, data: JSON.parse(d) }); } catch { res({ ok: false, status: re.statusCode, data: d }); } });
  }); r.on('error', rej); if (pl) r.write(pl); r.end(); }); }
const GET = p => api('GET', p), POST = (p, b) => api('POST', p, b);
const sleep = ms => new Promise(r => setTimeout(r, ms));
function t2c(t) { return (t === 10 || t === 11 || t === 0) ? 0 : (t >= 1 && t <= 5) ? t : 0; }

// Enhanced feature extraction
function cfEnhanced(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;

  let nSet = 0, nPort = 0, nRuin = 0, nForest = 0, nMtn = 0, co = 0;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    if (dy === 0 && dx === 0) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    const nt = g[ny][nx];
    if (nt === 1) nSet++;
    if (nt === 2) nPort++;
    if (nt === 3) nRuin++;
    if (nt === 4) nForest++;
    if (nt === 5) nMtn++;
    if (nt === 10) co = 1;
  }

  // Ring 2 counts
  let sR2 = 0, fR2 = 0;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    if (Math.abs(dy) <= 1 && Math.abs(dx) <= 1) continue;
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    if (g[ny][nx] === 1 || g[ny][nx] === 2) sR2++;
    if (g[ny][nx] === 4) fR2++;
  }

  // Bin features
  const ns = nSet + nPort;  // Total settlements/ports
  const sa = Math.min(ns, 5);
  const sb2 = sR2 === 0 ? 0 : sR2 <= 2 ? 1 : sR2 <= 4 ? 2 : 3;
  const fb = nForest <= 1 ? 0 : nForest <= 3 ? 1 : 2;
  const rb = nRuin === 0 ? 0 : 1;  // Has ruin nearby
  const pb = nPort === 0 ? 0 : 1;  // Has port nearby
  const mb = nMtn === 0 ? 0 : nMtn <= 2 ? 1 : 2; // Mountain proximity
  const fb2 = fR2 <= 2 ? 0 : fR2 <= 5 ? 1 : 2; // Forest ring 2

  return {
    // Level 0: Most specific — original + ruin + port + mtn
    e0: `E0_${t}_${sa}_${co}_${sb2}_${fb}_${rb}_${pb}_${mb}`,
    // Level 1: Drop mountain proximity
    e1: `E1_${t}_${sa}_${co}_${sb2}_${fb}_${rb}_${pb}`,
    // Level 2: Drop port distinction
    e2: `E2_${t}_${sa}_${co}_${sb2}_${fb}_${rb}`,
    // Level 3: Original D0 equivalent
    e3: `E3_${t}_${sa}_${co}_${sb2}_${fb}`,
    // Level 4: D1 equivalent (coarser sa)
    e4: `E4_${t}_${Math.min(sa, 3)}_${co}_${sb2}`,
    // Level 5: D2 equivalent
    e5: `E5_${t}_${sa > 0 ? 1 : 0}_${co}`,
    // Level 6: D3 equivalent
    e6: `E6_${t}_${co}`,
    // Level 7: D4 equivalent
    e7: `E7_${t}`,
  };
}

// Also keep the original feature extractor for comparison
function cfOriginal(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS = 0, co = 0, fN = 0, sR2 = 0;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    if (dy === 0 && dx === 0) continue; const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue; const nt = g[ny][nx];
    if (nt === 1 || nt === 2) nS++; if (nt === 10) co = 1; if (nt === 4) fN++; }
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    if (Math.abs(dy) <= 1 && Math.abs(dx) <= 1) continue; const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
    if (g[ny][nx] === 1 || g[ny][nx] === 2) sR2++; }
  const sa = Math.min(nS, 5), sb2 = sR2 === 0 ? 0 : sR2 <= 2 ? 1 : sR2 <= 4 ? 2 : 3;
  const fb = fN <= 1 ? 0 : fN <= 3 ? 1 : 2;
  return { d0: `D0_${t}_${sa}_${co}_${sb2}_${fb}`, d1: `D1_${t}_${Math.min(sa, 3)}_${co}_${sb2}`,
    d2: `D2_${t}_${sa > 0 ? 1 : 0}_${co}`, d3: `D3_${t}_${co}`, d4: `D4_${t}` };
}

function buildEnhancedModel(G, R, I, trainRounds, gtW, alpha) {
  const levels = ['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7'];
  const model = {};

  for (const level of levels) {
    const m = {};
    for (const rn of trainRounds) {
      if (!G[rn] || !I[rn]) continue;
      for (let si = 0; si < SEEDS; si++) {
        if (!I[rn][si] || !G[rn][si]) continue;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = cfEnhanced(I[rn][si], y, x); if (!keys) continue; const k = keys[level];
          if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
          const p = G[rn][si][y][x];
          for (let c = 0; c < C; c++) m[k].counts[c] += p[c] * gtW; m[k].n += gtW;
        }
      }
    }
    for (const rn of trainRounds) {
      if (!R[rn] || !I[rn]) continue;
      for (const rep of R[rn]) { const g = I[rn][rep.si]; if (!g) continue;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const keys = cfEnhanced(g, y, x); if (!keys) continue; const k = keys[level];
          const fc = t2c(rep.finalGrid[y][x]);
          if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) }; m[k].n++; m[k].counts[fc]++;
        }
      }
    }
    for (const k of Object.keys(m)) {
      const tot = Array.from(m[k].counts).reduce((a, b) => a + b, 0) + C * alpha;
      m[k].a = Array.from(m[k].counts).map(v => (v + alpha) / tot);
    }
    for (const [k, v] of Object.entries(m)) { if (!model[k]) model[k] = v; }
  }
  return model;
}

function predictEnhanced(grid, model, temp) {
  const levels = ['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7'];
  const ws = [1.0, 0.5, 0.3, 0.2, 0.12, 0.06, 0.03, 0.01];
  const pred = [];

  for (let y = 0; y < H; y++) { pred[y] = [];
    for (let x = 0; x < W; x++) {
      const t = grid[y][x];
      if (t === 10) { pred[y][x] = [1, 0, 0, 0, 0, 0]; continue; }
      if (t === 5) { pred[y][x] = [0, 0, 0, 0, 0, 1]; continue; }
      const keys = cfEnhanced(grid, y, x);
      if (!keys) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }

      const p = [0, 0, 0, 0, 0, 0]; let wS = 0;
      for (let li = 0; li < levels.length; li++) {
        const d = model[keys[levels[li]]];
        if (d && d.n >= 1) {
          const w = ws[li] * Math.pow(d.n, 0.5);
          for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w;
        }
      }
      if (wS === 0) { pred[y][x] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]; continue; }
      let s = 0;
      for (let c = 0; c < C; c++) { p[c] = Math.pow(Math.max(p[c] / wS, 1e-10), 1 / temp);
        if (p[c] < 0.00005) p[c] = 0.00005; s += p[c]; }
      for (let c = 0; c < C; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

function scoreVsGT(pred, gt) {
  let totalEntropy = 0, totalWeightedKL = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const gtP = gt[y][x], prP = pred[y][x];
    let entropy = 0;
    for (let c = 0; c < C; c++) if (gtP[c] > 0) entropy -= gtP[c] * Math.log(gtP[c]);
    if (entropy < 0.01) continue;
    let kl = 0;
    for (let c = 0; c < C; c++) if (gtP[c] > 0) kl += gtP[c] * Math.log(gtP[c] / Math.max(prP[c], 1e-10));
    totalEntropy += entropy;
    totalWeightedKL += entropy * kl;
  }
  if (totalEntropy === 0) return 100;
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * totalWeightedKL / totalEntropy)));
}

function fuseVP(model, vpObs, inits, cw) {
  const fused = {};
  for (const [k, v] of Object.entries(model)) fused[k] = { n: v.n, a: [...v.a] };
  if (vpObs.length === 0) return fused;

  // Fuse at e3 level (equivalent to D0)
  const vpKeys = {};
  for (const obs of vpObs) {
    const si = obs.si !== undefined ? obs.si : 0;
    for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
      const gy = obs.vy + dy, gx = obs.vx + dx;
      if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
      const keys = cfEnhanced(inits[si], gy, gx); if (!keys) continue;
      // Fuse at multiple levels
      for (const level of ['e3', 'e2', 'e1']) {
        const k = keys[level], fc = t2c(obs.grid[dy][dx]);
        if (!vpKeys[k]) vpKeys[k] = { n: 0, counts: new Float64Array(C) };
        vpKeys[k].n++; vpKeys[k].counts[fc]++;
      }
    }
  }

  for (const [k, vm] of Object.entries(vpKeys)) {
    const bm = fused[k];
    if (bm) {
      const pa = bm.a.map(p => p * cw), post = pa.map((a, c) => a + vm.counts[c]);
      const tot = post.reduce((a, b) => a + b, 0);
      fused[k] = { n: bm.n + vm.n, a: post.map(v => v / tot) };
    } else {
      const tot = vm.n + C * 0.1;
      fused[k] = { n: vm.n, a: Array.from(vm.counts).map(v => (v + 0.1) / tot) };
    }
  }
  return fused;
}

async function main() {
  console.log('=== R9 ENHANCED FEATURES ===');
  console.log('Time:', new Date().toISOString());

  // Load data
  const I = {}, G = {}, R = {};
  const allRounds = [];
  for (let r = 1; r <= 20; r++) {
    const rn = `R${r}`;
    if (fs.existsSync(path.join(DD, `inits_${rn}.json`))) I[rn] = JSON.parse(fs.readFileSync(path.join(DD, `inits_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `gt_${rn}.json`))) G[rn] = JSON.parse(fs.readFileSync(path.join(DD, `gt_${rn}.json`)));
    if (fs.existsSync(path.join(DD, `replays_${rn}.json`))) R[rn] = JSON.parse(fs.readFileSync(path.join(DD, `replays_${rn}.json`)));
    if (I[rn] && G[rn]) allRounds.push(rn);
  }
  const trainRounds = allRounds.filter(r => r !== 'R3');
  console.log('Training:', trainRounds.join(', '));

  // LOO validation — compare original vs enhanced
  console.log('\n=== LOO: ORIGINAL vs ENHANCED ===');

  for (const temp of [1.1, 1.15, 1.2]) {
    // Original model
    let origTotal = 0, enhTotal = 0, count = 0;
    for (const testRound of trainRounds) {
      const train = trainRounds.filter(r => r !== testRound);

      // Original
      const origModel = {};
      for (const level of ['d0', 'd1', 'd2', 'd3', 'd4']) {
        const m = {};
        for (const rn of train) {
          if (!G[rn] || !I[rn]) continue;
          for (let si = 0; si < SEEDS; si++) {
            if (!I[rn][si] || !G[rn][si]) continue;
            for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
              const keys = cfOriginal(I[rn][si], y, x); if (!keys) continue; const k = keys[level];
              if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) };
              const p = G[rn][si][y][x];
              for (let c = 0; c < C; c++) m[k].counts[c] += p[c] * 20; m[k].n += 20;
            }
          }
        }
        for (const rn of train) {
          if (!R[rn] || !I[rn]) continue;
          for (const rep of R[rn]) { const g = I[rn][rep.si]; if (!g) continue;
            for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
              const keys = cfOriginal(g, y, x); if (!keys) continue; const k = keys[level];
              const fc = t2c(rep.finalGrid[y][x]);
              if (!m[k]) m[k] = { n: 0, counts: new Float64Array(C) }; m[k].n++; m[k].counts[fc]++;
            }
          }
        }
        for (const k of Object.keys(m)) {
          const tot = Array.from(m[k].counts).reduce((a, b) => a + b, 0) + C * 0.05;
          m[k].a = Array.from(m[k].counts).map(v => (v + 0.05) / tot);
        }
        for (const [k, v] of Object.entries(m)) { if (!origModel[k]) origModel[k] = v; }
      }

      // Enhanced
      const enhModel = buildEnhancedModel(G, R, I, train, 20, 0.05);

      let origRound = 0, enhRound = 0;
      for (let si = 0; si < SEEDS; si++) {
        if (!I[testRound][si] || !G[testRound][si]) continue;

        // Original prediction
        const origPred = [];
        for (let y = 0; y < H; y++) { origPred[y] = [];
          for (let x = 0; x < W; x++) {
            const t2 = I[testRound][si][y][x];
            if (t2 === 10) { origPred[y][x] = [1,0,0,0,0,0]; continue; }
            if (t2 === 5) { origPred[y][x] = [0,0,0,0,0,1]; continue; }
            const keys = cfOriginal(I[testRound][si], y, x);
            if (!keys) { origPred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
            const lvls = ['d0','d1','d2','d3','d4'], ws = [1.0,0.3,0.15,0.08,0.02];
            const p = [0,0,0,0,0,0]; let wS = 0;
            for (let li = 0; li < lvls.length; li++) {
              const d = origModel[keys[lvls[li]]];
              if (d && d.n >= 1) { const w = ws[li]*Math.pow(d.n,0.5);
                for (let c = 0; c < C; c++) p[c] += w * d.a[c]; wS += w; } }
            if (wS === 0) { origPred[y][x] = [1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
            let s = 0;
            for (let c = 0; c < C; c++) { p[c] = Math.pow(Math.max(p[c]/wS, 1e-10), 1/temp);
              if (p[c] < 0.00005) p[c] = 0.00005; s += p[c]; }
            for (let c = 0; c < C; c++) p[c] /= s;
            origPred[y][x] = p;
          }
        }
        origRound += scoreVsGT(origPred, G[testRound][si]);

        // Enhanced prediction
        const enhPred = predictEnhanced(I[testRound][si], enhModel, temp);
        enhRound += scoreVsGT(enhPred, G[testRound][si]);
      }
      origTotal += origRound / SEEDS;
      enhTotal += enhRound / SEEDS;
      count++;
    }

    console.log(`temp=${temp}: original=${(origTotal/count).toFixed(3)}, enhanced=${(enhTotal/count).toFixed(3)}, diff=${((enhTotal-origTotal)/count).toFixed(3)}`);
  }

  // If enhanced is better, submit to R9
  console.log('\n=== SUBMITTING ENHANCED MODEL TO R9 ===');

  const { data: rounds } = await GET('/rounds');
  const r9 = rounds.find(r => r.round_number === 9);
  if (!r9 || r9.status !== 'active') { console.log('R9 not active!'); return; }

  const { data: rd } = await GET('/rounds/' + r9.id);
  const inits = rd.initial_states.map(is => is.grid);

  const vpFile = path.join(DD, `viewport_${r9.id.slice(0,8)}.json`);
  let vpObs = [];
  if (fs.existsSync(vpFile)) vpObs = JSON.parse(fs.readFileSync(vpFile));
  console.log('VP:', vpObs.length);

  // Build and submit with different temps
  for (const temp of [1.1, 1.15, 1.2]) {
    const enhModel = buildEnhancedModel(G, R, I, trainRounds, 20, 0.05);
    const model = fuseVP(enhModel, vpObs, inits, 20);

    console.log(`\nSubmitting enhanced model temp=${temp}:`);
    for (let si = 0; si < SEEDS; si++) {
      let pred = predictEnhanced(inits[si], model, temp);

      // Apply per-cell corrections
      const obsBySeed = vpObs.filter(obs => (obs.si !== undefined ? obs.si : 0) === si);
      const cells = {};
      for (const obs of obsBySeed) {
        for (let dy = 0; dy < obs.grid.length; dy++) for (let dx = 0; dx < obs.grid[0].length; dx++) {
          const gy = obs.vy + dy, gx = obs.vx + dx;
          if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
          if (inits[si][gy][gx] === 10 || inits[si][gy][gx] === 5) continue;
          const k = `${gy},${gx}`, fc = t2c(obs.grid[dy][dx]);
          if (!cells[k]) cells[k] = { n: 0, counts: new Float64Array(C) };
          cells[k].n++; cells[k].counts[fc]++;
        }
      }
      for (const [key, cell] of Object.entries(cells)) {
        const [y, x] = key.split(',').map(Number);
        const pw = cell.n >= 5 ? 2 : cell.n >= 3 ? 4 : cell.n >= 2 ? 7 : 15;
        const prior = pred[y][x], posterior = new Array(C); let total = 0;
        for (let c = 0; c < C; c++) { posterior[c] = prior[c] * pw + cell.counts[c]; total += posterior[c]; }
        if (total > 0) { for (let c = 0; c < C; c++) posterior[c] /= total; pred[y][x] = posterior; }
      }

      // Validate
      let valid = true;
      for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
        const s = pred[y][x].reduce((a, b) => a + b, 0);
        if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false;
      }
      if (!valid) { console.log(`  Seed ${si}: VALIDATION FAILED`); continue; }

      const res = await POST('/submit', { round_id: r9.id, seed_index: si, prediction: pred });
      const score = res.data && res.data.score !== undefined ? res.data.score : '?';
      console.log(`  Seed ${si}: ${res.ok ? 'OK' : 'FAIL'} score=${score}`);
      await sleep(600);
    }
  }

  console.log('\n=== DONE ===');
}

main().catch(e => console.error('Error:', e.message, e.stack));
