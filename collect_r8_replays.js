#!/usr/bin/env node
// Collect R8 replays — UNLIMITED and FREE
const https = require('https'), fs = require('fs'), path = require('path');
const TOKEN = process.argv[1] || '';
const TARGET = parseInt(process.argv[2]) || 500;
const BASE = 'https://api.ainm.no/astar-island';
const R8 = 'c5cdf100-a876-4fb7-b5d8-757162c97989';
const DD = path.join(__dirname, 'data');
const SEEDS = 5;

function api(m, p, b) {
  return new Promise((res, rej) => {
    const u = new URL(BASE + p); const pl = b ? JSON.stringify(b) : null;
    const o = { hostname: u.hostname, path: u.pathname + u.search, method: m,
      headers: { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' } };
    if (pl) o.headers['Content-Length'] = Buffer.byteLength(pl);
    const r = https.request(o, re => {
      let d = ''; re.on('data', c => d += c);
      re.on('end', () => { try { res({ ok: re.statusCode < 300, status: re.statusCode, data: JSON.parse(d) }); } catch { res({ ok: false, status: re.statusCode, data: d }); } });
    }); r.on('error', e => { res({ ok: false, status: 0, data: e.message }); }); if (pl) r.write(pl); r.end();
  });
}
const POST = (p, b) => api('POST', p, b);
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function main() {
  console.log(`Collecting R8 replays: target ${TARGET}/seed (${TARGET * SEEDS} total)`);
  const f = path.join(DD, 'replays_R8.json');
  let all = fs.existsSync(f) ? JSON.parse(fs.readFileSync(f)) : [];
  
  // Count per seed
  const perSeed = [0,0,0,0,0];
  for (const r of all) perSeed[r.si]++;
  console.log('Existing:', perSeed.join(', '), '=', all.length, 'total');
  
  let errors = 0, consecutive_errors = 0;
  const BATCH = 10; // 10 concurrent requests
  
  while (true) {
    // Find seed with fewest replays
    let minCount = Infinity, targetDone = true;
    for (let si = 0; si < SEEDS; si++) {
      if (perSeed[si] < TARGET) targetDone = false;
      if (perSeed[si] < minCount) minCount = perSeed[si];
    }
    if (targetDone) break;
    
    // Launch batch
    const batch = [];
    for (let i = 0; i < BATCH; i++) {
      // Pick seed with fewest replays (round robin if tied)
      let bestSi = 0, bestCount = perSeed[0];
      for (let si = 1; si < SEEDS; si++) {
        if (perSeed[si] < bestCount) { bestSi = si; bestCount = perSeed[si]; }
      }
      const si = (bestSi + i) % SEEDS;
      if (perSeed[si] >= TARGET) continue;
      
      batch.push((async () => {
        try {
          const res = await POST('/replay', { round_id: R8, seed_index: si });
          if (res.ok && res.data && res.data.frames) {
            const frames = res.data.frames;
            return { si, finalGrid: frames[frames.length - 1].grid };
          }
          return null;
        } catch { return null; }
      })());
    }
    
    const results = await Promise.all(batch);
    let added = 0;
    for (const r of results) {
      if (r) { all.push(r); perSeed[r.si]++; added++; consecutive_errors = 0; }
      else { errors++; consecutive_errors++; }
    }
    
    if (consecutive_errors > 50) {
      console.log('Too many consecutive errors, stopping');
      break;
    }
    
    // Save every 50 replays
    if (all.length % 50 < BATCH) {
      fs.writeFileSync(f, JSON.stringify(all));
      console.log(`${new Date().toISOString().slice(11,19)} | ${perSeed.join(', ')} = ${all.length} total (${errors} err)`);
    }
    
    await sleep(100); // Small delay between batches
  }
  
  fs.writeFileSync(f, JSON.stringify(all));
  console.log(`\nDone: ${all.length} total replays`);
  console.log('Per seed:', perSeed.join(', '));
}
main().catch(e => console.error('Error:', e.message));
