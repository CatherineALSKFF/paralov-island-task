#!/usr/bin/env node
/**
 * Massive replay collector with disk caching.
 * Collects 500 replays per completed round and saves to disk.
 * Run in background while building the optimized pipeline.
 *
 * Usage: node collect_replays_massive.js <JWT> [replays_per_round]
 */
const https = require('https');
const fs = require('fs');
const path = require('path');
const BASE = 'https://api.ainm.no/astar-island';
const TOKEN = process.argv[2] || '';
const TARGET_COUNT = parseInt(process.argv[3]) || 500;
const DATA_DIR = path.join(__dirname, 'data');
const SEEDS = 5;

if (!TOKEN) { console.log('Usage: node collect_replays_massive.js <JWT> [replays_per_round]'); process.exit(1); }
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

function api(method, pth, body) {
  return new Promise((resolve, reject) => {
    const url = new URL(BASE + pth);
    const payload = body ? JSON.stringify(body) : null;
    const opts = { hostname: url.hostname, path: url.pathname + url.search, method,
      headers: { 'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json' } };
    if (payload) opts.headers['Content-Length'] = Buffer.byteLength(payload);
    const req = https.request(opts, res => {
      let data = ''; res.on('data', c => data += c);
      res.on('end', () => { try { resolve({ ok: res.statusCode < 300, status: res.statusCode, data: JSON.parse(data) }); } catch { resolve({ ok: false, status: res.statusCode, data }); } });
    }); req.on('error', reject); if (payload) req.write(payload); req.end();
  });
}
const GET = p => api('GET', p), POST = (p, b) => api('POST', p, b);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const log = m => { const t = new Date().toISOString().slice(11, 19); console.log(`[${t}] ${m}`); };

async function collectReplays(roundId, roundName, count, concurrency = 10) {
  const dataFile = path.join(DATA_DIR, `replays_${roundName}.json`);

  // Load existing data
  let existing = [];
  if (fs.existsSync(dataFile)) {
    try {
      existing = JSON.parse(fs.readFileSync(dataFile, 'utf8'));
      log(`  Loaded ${existing.length} existing replays from disk`);
    } catch (e) {
      log(`  Warning: Could not load existing data: ${e.message}`);
    }
  }

  const needed = count - existing.length;
  if (needed <= 0) {
    log(`  Already have ${existing.length}/${count} replays. Skipping.`);
    return existing;
  }

  log(`  Need ${needed} more replays (have ${existing.length}/${count})`);
  const results = [...existing];
  let collected = 0, errors = 0, consecutive_errors = 0;

  while (collected < needed) {
    const batch = [];
    const batchSize = Math.min(concurrency, needed - collected);
    for (let i = 0; i < batchSize; i++) {
      const si = (collected + i) % SEEDS;
      batch.push((async () => {
        try {
          const res = await POST('/replay', { round_id: roundId, seed_index: si });
          if (!res.ok || !res.data.frames) { errors++; consecutive_errors++; return null; }
          consecutive_errors = 0;
          const frames = res.data.frames;
          return { si, finalGrid: frames[frames.length - 1].grid };
        } catch (e) { errors++; consecutive_errors++; return null; }
      })());
    }
    const batchResults = await Promise.all(batch);
    for (const r of batchResults) {
      if (r) { results.push(r); collected++; }
    }

    // Save progress every 50 replays
    if (collected % 50 < concurrency) {
      fs.writeFileSync(dataFile, JSON.stringify(results));
      log(`  Progress: ${results.length}/${count} replays (${errors} errors, ${(collected/needed*100).toFixed(0)}%)`);
    }

    // Back off if too many consecutive errors
    if (consecutive_errors > 20) {
      log(`  Too many consecutive errors (${consecutive_errors}). Waiting 5s...`);
      await sleep(5000);
      consecutive_errors = 0;
    }

    await sleep(150); // Small delay between batches
  }

  // Final save
  fs.writeFileSync(dataFile, JSON.stringify(results));
  log(`  ✅ ${results.length} replays saved to ${dataFile} (${errors} total errors)`);
  return results;
}

async function main() {
  log(`╔═════════════════════════════════════════════╗`);
  log(`║  Massive Replay Collector (${TARGET_COUNT}/round)     ║`);
  log(`╚═════════════════════════════════════════════╝`);

  // Get all completed rounds
  const { data: rounds } = await GET('/rounds');
  const completed = rounds.filter(r => r.status === 'completed');

  log(`\nFound ${completed.length} completed rounds`);

  // Also save initial states
  for (const r of completed) {
    const rn = `R${r.round_number}`;
    const initFile = path.join(DATA_DIR, `inits_${rn}.json`);
    if (!fs.existsSync(initFile)) {
      log(`\nLoading initial states for ${rn}...`);
      const { data } = await GET('/rounds/' + r.id);
      const inits = data.initial_states.map(is => is.grid);
      fs.writeFileSync(initFile, JSON.stringify(inits));
      log(`  Saved ${inits.length} initial states`);
    }

    // Save GT if available
    const gtFile = path.join(DATA_DIR, `gt_${rn}.json`);
    if (!fs.existsSync(gtFile)) {
      log(`Loading GT for ${rn}...`);
      const gts = [];
      let gotAll = true;
      for (let si = 0; si < SEEDS; si++) {
        const res = await GET('/analysis/' + r.id + '/' + si);
        if (res.ok && res.data && res.data.ground_truth) {
          gts[si] = res.data.ground_truth;
        } else {
          gotAll = false;
          log(`  Warning: No GT for ${rn} seed ${si}`);
        }
      }
      if (gotAll) {
        fs.writeFileSync(gtFile, JSON.stringify(gts));
        log(`  Saved GT for ${rn}`);
      }
    }
  }

  // Collect replays for each round
  for (const r of completed) {
    const rn = `R${r.round_number}`;
    log(`\n═══ Collecting ${rn} (${r.id.slice(0,8)}...) ═══`);
    await collectReplays(r.id, rn, TARGET_COUNT);
  }

  log(`\n╔═════════════════════════════════════════════╗`);
  log(`║  ✅ Collection complete!                      ║`);
  log(`╚═════════════════════════════════════════════╝`);

  // Summary
  for (const r of completed) {
    const rn = `R${r.round_number}`;
    const dataFile = path.join(DATA_DIR, `replays_${rn}.json`);
    if (fs.existsSync(dataFile)) {
      const data = JSON.parse(fs.readFileSync(dataFile, 'utf8'));
      log(`  ${rn}: ${data.length} replays`);
    }
  }
}

main().catch(e => { console.error('Fatal:', e.message, e.stack); process.exit(1); });
