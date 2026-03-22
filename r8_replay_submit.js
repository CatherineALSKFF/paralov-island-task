#!/usr/bin/env node
// R8 REPLAY-BASED SUBMISSION — Pure empirical distributions from replays
// This is the approach that scored 93.7 on R1 with 150 replays/seed
const fs = require('fs'), path = require('path'), https = require('https');
const H = 40, W = 40, SEEDS = 5, C = 6;
const DD = path.join(__dirname, 'data');
const TOKEN = process.argv[2] || process.argv[1] || '';
const MIN_REPLAYS = parseInt(process.argv[3]) || 50;
const BASE = 'https://api.ainm.no/astar-island';
const R8 = 'c5cdf100-a876-4fb7-b5d8-757162c97989';

function api(m, p, b) {
  return new Promise((res, rej) => {
    const u = new URL(BASE + p); const pl = b ? JSON.stringify(b) : null;
    const o = { hostname: u.hostname, path: u.pathname + u.search, method: m,
      headers: { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' } };
    if (pl) o.headers['Content-Length'] = Buffer.byteLength(pl);
    const r = https.request(o, re => {
      let d = ''; re.on('data', c => d += c);
      re.on('end', () => { try { res({ ok: re.statusCode < 300, status: re.statusCode, data: JSON.parse(d) }); } catch { res({ ok: false, status: re.statusCode, data: d }); } });
    }); r.on('error', rej); if (pl) r.write(pl); r.end();
  });
}
const GET = p => api('GET', p), POST = (p, b) => api('POST', p, b);
const sleep = ms => new Promise(r => setTimeout(r, ms));

// Terrain type to class index mapping
function t2c(t) { return (t === 10 || t === 11 || t === 0) ? 0 : (t >= 1 && t <= 5) ? t : 0; }

async function main() {
  console.log('=== R8 REPLAY-BASED SUBMISSION ===');
  console.log('Time:', new Date().toISOString());
  
  // Load R8 replay data
  const replayFile = path.join(DD, 'replays_R8.json');
  if (!fs.existsSync(replayFile)) {
    console.log('ERROR: No R8 replays found! Run collect_r8_replays.js first.');
    return;
  }
  
  const replays = JSON.parse(fs.readFileSync(replayFile));
  console.log('Total R8 replays:', replays.length);
  
  // Group by seed
  const bySeed = {};
  for (const r of replays) {
    if (!bySeed[r.si]) bySeed[r.si] = [];
    bySeed[r.si].push(r);
  }
  
  for (let si = 0; si < SEEDS; si++) {
    const count = bySeed[si] ? bySeed[si].length : 0;
    console.log(`  Seed ${si}: ${count} replays`);
  }
  
  // Load initial states for static cell detection
  const { data: rd } = await GET('/rounds/' + R8);
  const inits = rd.initial_states.map(is => is.grid);
  
  // Build predictions from replay data
  for (let si = 0; si < SEEDS; si++) {
    const seedReplays = bySeed[si] || [];
    const N = seedReplays.length;
    
    if (N < MIN_REPLAYS) {
      console.log(`Seed ${si}: Only ${N} replays, need ${MIN_REPLAYS}. SKIPPING.`);
      continue;
    }
    
    // Count occurrences per cell
    const counts = [];
    for (let y = 0; y < H; y++) {
      counts[y] = [];
      for (let x = 0; x < W; x++) {
        counts[y][x] = new Float64Array(C);
      }
    }
    
    for (const rep of seedReplays) {
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const fc = t2c(rep.finalGrid[y][x]);
          counts[y][x][fc]++;
        }
      }
    }
    
    // Build prediction with per-cell adaptive Dirichlet smoothing
    const pred = [];
    let staticCells = 0, dynamicCells = 0;
    
    for (let y = 0; y < H; y++) {
      pred[y] = [];
      for (let x = 0; x < W; x++) {
        const t = inits[si][y][x];
        
        // Static cells: ocean = 100% class 0, mountain = 100% class 5
        if (t === 10) {
          pred[y][x] = [1.0, 0, 0, 0, 0, 0];
          staticCells++;
          continue;
        }
        if (t === 5) {
          pred[y][x] = [0, 0, 0, 0, 0, 1.0];
          staticCells++;
          continue;
        }
        
        dynamicCells++;
        const cellCounts = counts[y][x];
        
        // Count unique classes observed
        let nClasses = 0;
        for (let c = 0; c < C; c++) if (cellCounts[c] > 0) nClasses++;
        
        // Per-cell adaptive alpha (v5.1 approach)
        let alpha;
        if (nClasses <= 1) {
          alpha = 0.001; // Very static, barely needs smoothing
        } else if (nClasses === 2) {
          // Calculate entropy for 2-class
          const total = N;
          let ent = 0;
          for (let c = 0; c < C; c++) {
            if (cellCounts[c] > 0) {
              const p = cellCounts[c] / total;
              ent -= p * Math.log(p);
            }
          }
          alpha = 0.003 + 0.002 * Math.min(ent, 1);
        } else {
          // High entropy, more smoothing
          const total = N;
          let ent = 0;
          for (let c = 0; c < C; c++) {
            if (cellCounts[c] > 0) {
              const p = cellCounts[c] / total;
              ent -= p * Math.log(p);
            }
          }
          // Scale alpha based on entropy
          const baseAlpha = Math.max(0.02, 0.15 * Math.sqrt(150 / N));
          alpha = baseAlpha * Math.min(ent / 1.5, 1.0);
        }
        
        // Compute prediction
        const total = N + C * alpha;
        const p = new Array(C);
        let sum = 0;
        for (let c = 0; c < C; c++) {
          p[c] = (cellCounts[c] + alpha) / total;
          sum += p[c];
        }
        // Normalize
        for (let c = 0; c < C; c++) p[c] /= sum;
        pred[y][x] = p;
      }
    }
    
    console.log(`Seed ${si}: ${N} replays, ${dynamicCells} dynamic, ${staticCells} static`);
    
    // Validate
    let valid = true;
    for (let y = 0; y < H && valid; y++) for (let x = 0; x < W && valid; x++) {
      const s = pred[y][x].reduce((a, b) => a + b, 0);
      if (Math.abs(s - 1) > 0.02 || pred[y][x].some(v => v < 0)) valid = false;
    }
    if (!valid) { console.log(`Seed ${si}: VALIDATION FAILED`); continue; }
    
    // Submit
    const res = await POST('/submit', { round_id: R8, seed_index: si, prediction: pred });
    console.log(`Seed ${si}: ${res.ok ? 'ACCEPTED' : 'FAILED'} ${JSON.stringify(res.data).slice(0, 100)}`);
    await sleep(600);
  }
  
  console.log('\n=== R8 REPLAY-BASED SUBMISSION DONE ===');
  console.log(`Approach: Pure replay empirical distributions with per-cell adaptive Dirichlet`);
}

main().catch(e => console.error('Error:', e.message, e.stack));
