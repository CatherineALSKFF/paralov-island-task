/**
 * Mega Solver v5.1 — Replay-Based Prediction Engine (Per-Cell Adaptive Alpha)
 * Paste into browser console on app.ainm.no (must be logged in).
 *
 * STRATEGY: Replay API = unlimited free runs of the REAL simulator.
 * Aggregate N replays per seed → empirical probability distributions.
 * Per-cell adaptive alpha smoothing squeezes out every fraction of a point.
 *
 * Usage:
 *   1. Paste this script in browser console
 *   2. await _M.init()
 *   3. await _M.collect(10000)  — collect replays (target per seed)
 *   4. await _M.scoreVsGT()    — score against GT (completed rounds)
 *   5. _M.submitAll() → _M.confirmSubmitAll()  — submit (with confirmation)
 *
 * Advanced:
 *   _M.scoreAdaptive()    — score with per-cell adaptive alpha
 *   _M.collectForever()   — continuous collection (Ctrl+C safe)
 *   _M.exportData()       — export data for Node.js script
 *   _M.importData(json)   — import data from Node.js script
 *
 * *** DOES NOT submit without explicit confirmation ***
 */
(function MegaSolverV5() {
  'use strict';
  var BASE = 'https://api.ainm.no/astar-island';
  var H = 40, W = 40, SEEDS = 5;
  var _t0 = Date.now();

  window._M = {};
  var M = window._M;

  function log(msg) { console.log('[V5 ' + ((Date.now()-_t0)/1000).toFixed(1) + 's] ' + msg); }
  function t2c(t) { return (t===10||t===11||t===0)?0:((t>=1&&t<=5)?t:0); }

  // ═══════════════════════════════════════════════════════════════════════
  // PHASE 1: DETECT ROUND
  // ═══════════════════════════════════════════════════════════════════════
  M.init = async function() {
    log('Initializing...');
    var rounds = await (await fetch(BASE+'/rounds', {credentials:'include'})).json();
    var active = rounds.find(function(r){return r.status==='active';});
    var scoring = rounds.find(function(r){return r.status==='scoring';});
    var completed = rounds.filter(function(r){return r.status==='completed';}).sort(function(a,b){return b.round_number-a.round_number;});
    M.round = active || scoring || completed[0] || rounds[rounds.length-1];
    M.rid = M.round.id;

    var detail = await (await fetch(BASE+'/rounds/'+M.rid, {credentials:'include'})).json();
    M.detail = detail;
    M.initialStates = detail.initial_states || detail.seeds;

    log('Round ' + detail.round_number + ' (' + M.round.status + ') — ' + M.rid.slice(0,8));
    log('Map: ' + detail.map_width + 'x' + detail.map_height + ', ' + detail.seeds_count + ' seeds');

    // Load accumulated data from localStorage
    var saved = localStorage.getItem('pred_acc_' + M.rid);
    if (saved) {
      M.acc = JSON.parse(saved);
      var counts = [];
      for (var s = 0; s < SEEDS; s++) counts.push(M.acc[s] ? M.acc[s].count : 0);
      log('Loaded saved replays: [' + counts.join(',') + ']');
    } else {
      M.acc = {};
      log('No saved replays for this round');
    }

    // Also check legacy key (from Round 1 testing)
    if (!saved) {
      var legacy = localStorage.getItem('pred_acc');
      if (legacy) {
        var legAcc = JSON.parse(legacy);
        // Check if it's for our round
        if (legAcc[0] && legAcc[0].count > 0) {
          M.acc = legAcc;
          log('Loaded legacy replays');
        }
      }
    }

    return M;
  };

  // ═══════════════════════════════════════════════════════════════════════
  // PHASE 2: COLLECT REPLAYS
  // ═══════════════════════════════════════════════════════════════════════
  function initSeed(s) {
    if (M.acc[s]) return;
    M.acc[s] = {count: 0, grid: []};
    for (var y = 0; y < H; y++) {
      M.acc[s].grid[y] = [];
      for (var x = 0; x < W; x++) M.acc[s].grid[y][x] = [0,0,0,0,0,0];
    }
  }

  function addReplay(seed, data) {
    if (!data || !data.frames || data.frames.length < 2) return false;
    initSeed(seed);
    var f = data.frames[data.frames.length - 1];
    for (var y = 0; y < H; y++) {
      for (var x = 0; x < W; x++) {
        M.acc[seed].grid[y][x][t2c(f.grid[y][x])]++;
      }
    }
    M.acc[seed].count++;
    return true;
  }

  function fetchReplay(seed) {
    return fetch(BASE + '/replay', {
      method: 'POST', credentials: 'include',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({round_id: M.rid, seed_index: seed})
    }).then(function(r) { return r.json(); }).catch(function() { return null; });
  }

  M.collect = function(target, concurrency) {
    target = target || 500;
    concurrency = concurrency || 5;
    log('Collecting replays (target ' + target + '/seed, concurrency ' + concurrency + ')...');

    return new Promise(function(resolve) {
      // Round-robin queue
      var q = [];
      var np = {};
      for (var s = 0; s < SEEDS; s++) np[s] = target - (M.acc[s] ? M.acc[s].count : 0);
      var mx = Math.max.apply(null, [0,1,2,3,4].map(function(s){return Math.max(0,np[s]);}));
      for (var i = 0; i < mx; i++) {
        for (var s = 0; s < SEEDS; s++) {
          if (np[s] > i) q.push(s);
        }
      }

      if (q.length === 0) {
        log('All seeds already at target!');
        resolve(M.acc);
        return;
      }

      var qi = 0, inf = 0, dn = 0;
      function go() {
        while (inf < concurrency && qi < q.length) {
          var seed = q[qi++]; inf++;
          fetchReplay(seed).then(function(d) {
            if (d && d.seed_index >= 0) addReplay(d.seed_index, d);
            inf--; dn++;
            if (dn % 10 === 0) {
              M.save();
              var cts = [0,1,2,3,4].map(function(s){return M.acc[s]?M.acc[s].count:0;});
              log('Progress: ' + dn + '/' + q.length + ' [' + cts.join(',') + ']');
              document.title = dn + '/' + q.length + ' [' + cts.join(',') + ']';
            }
            if (qi < q.length) go();
            else if (inf === 0) {
              M.save();
              var cts = [0,1,2,3,4].map(function(s){return M.acc[s]?M.acc[s].count:0;});
              log('DONE! [' + cts.join(',') + ']');
              document.title = 'DONE [' + cts.join(',') + ']';
              resolve(M.acc);
            }
          }).catch(function() { inf--; go(); });
        }
      }

      log('Queued ' + q.length + ' replays...');
      go();
    });
  };

  M.save = function() {
    localStorage.setItem('pred_acc_' + M.rid, JSON.stringify(M.acc));
    // Also save to legacy key for compatibility
    localStorage.setItem('pred_acc', JSON.stringify(M.acc));
  };

  // ═══════════════════════════════════════════════════════════════════════
  // PHASE 3: BUILD PREDICTIONS (per-cell adaptive alpha)
  // ═══════════════════════════════════════════════════════════════════════

  // Compute per-cell alpha based on observed dynamics
  function cellAlpha(counts, N, baseAlpha) {
    // Count distinct classes observed
    var nClasses = 0;
    var maxCount = 0;
    for (var c = 0; c < 6; c++) {
      if (counts[c] > 0) nClasses++;
      if (counts[c] > maxCount) maxCount = counts[c];
    }
    // Empirical entropy
    var ent = 0;
    for (var c = 0; c < 6; c++) {
      if (counts[c] > 0) {
        var p = counts[c] / N;
        ent -= p * Math.log(p);
      }
    }
    // Adaptive alpha based on cell dynamics:
    // - 1 class observed = static → tiny alpha (but nonzero for safety)
    // - 2 classes = mostly static → small alpha
    // - 3+ classes = dynamic → scale by entropy
    if (nClasses <= 1) return 0.001;
    if (nClasses === 2 && maxCount > N * 0.95) return 0.003;
    if (nClasses === 2) return Math.max(0.005, baseAlpha * 0.3);
    // Dynamic cells: scale alpha by entropy / max_entropy
    var maxEnt = Math.log(6); // ~1.79
    var entRatio = Math.max(0.1, ent / maxEnt);
    return baseAlpha * entRatio;
  }

  M.buildPrediction = function(seed, alpha, adaptive) {
    var a = M.acc[seed];
    if (!a || a.count === 0) { log('No data for seed ' + seed); return null; }
    var N = a.count;
    // Base alpha scales with 1/sqrt(N)
    var baseAlpha = (alpha !== undefined) ? alpha : Math.max(0.02, 0.15 * Math.sqrt(150 / N));
    // Default to adaptive mode
    if (adaptive === undefined) adaptive = true;

    var pred = [];
    var alphaStats = {min: 999, max: 0, sum: 0, n: 0};
    for (var y = 0; y < H; y++) {
      pred[y] = [];
      for (var x = 0; x < W; x++) {
        var al = adaptive ? cellAlpha(a.grid[y][x], N, baseAlpha) : baseAlpha;
        alphaStats.min = Math.min(alphaStats.min, al);
        alphaStats.max = Math.max(alphaStats.max, al);
        alphaStats.sum += al; alphaStats.n++;

        var p = [0,0,0,0,0,0];
        var total = N + 6 * al;
        for (var c = 0; c < 6; c++) p[c] = (a.grid[y][x][c] + al) / total;

        // Precision: round to 6 decimals, fix sum to exactly 1.0
        var sum = 0, maxIdx = 0, maxVal = 0;
        for (var c = 0; c < 6; c++) {
          p[c] = parseFloat(p[c].toFixed(6));
          sum += p[c];
          if (p[c] > maxVal) { maxVal = p[c]; maxIdx = c; }
        }
        p[maxIdx] = parseFloat((p[maxIdx] + (1.0 - sum)).toFixed(6));
        pred[y][x] = p;
      }
    }
    var avgAlpha = (alphaStats.sum / alphaStats.n).toFixed(5);
    log('Built prediction for seed ' + seed + ' (N=' + N + ', adaptive=' + adaptive +
        ', alpha range ' + alphaStats.min.toFixed(4) + '-' + alphaStats.max.toFixed(4) +
        ', avg=' + avgAlpha + ')');
    return pred;
  };

  M.buildPredictions = function(alpha, adaptive) {
    M.predictions = {};
    for (var s = 0; s < SEEDS; s++) {
      M.predictions[s] = M.buildPrediction(s, alpha, adaptive);
    }
    log('All predictions built');
    return M.predictions;
  };

  // ═══════════════════════════════════════════════════════════════════════
  // PHASE 4: SCORE (against GT if available)
  // ═══════════════════════════════════════════════════════════════════════

  // Score helper: compute wkl for a seed with given alpha strategy
  function computeWKL(gt, acc, N, alphaOrAdaptive) {
    var adaptive = (alphaOrAdaptive === true || alphaOrAdaptive === undefined);
    var fixedAlpha = (typeof alphaOrAdaptive === 'number') ? alphaOrAdaptive : null;
    var baseAlpha = fixedAlpha || Math.max(0.02, 0.15 * Math.sqrt(150 / N));

    var totalKL = 0, totalEnt = 0;
    for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) {
      var g = gt[y][x];
      var ent = 0;
      for (var c = 0; c < 6; c++) if (g[c] > 1e-6) ent -= g[c] * Math.log(g[c]);
      if (ent < 0.01) continue;

      var al = (adaptive && !fixedAlpha) ? cellAlpha(acc.grid[y][x], N, baseAlpha) : baseAlpha;

      var p = [0,0,0,0,0,0];
      var total = N + 6 * al;
      for (var c = 0; c < 6; c++) p[c] = (acc.grid[y][x][c] + al) / total;

      var kl = 0;
      for (var c = 0; c < 6; c++) {
        if (g[c] > 1e-6) kl += g[c] * Math.log(g[c] / Math.max(p[c], 1e-15));
      }
      totalKL += Math.max(0, kl) * ent;
      totalEnt += ent;
    }
    return totalEnt > 0 ? totalKL / totalEnt : 0;
  }

  M.scoreVsGT = async function(alpha) {
    log('Scoring against GT (adaptive per-cell alpha)...');
    var scores = [];
    for (var seed = 0; seed < SEEDS; seed++) {
      try {
        var resp = await fetch(BASE + '/analysis/' + M.rid + '/' + seed, {credentials:'include'});
        if (resp.status !== 200) { scores.push({seed:seed, score:0, msg:'no GT'}); continue; }
        var data = await resp.json();
        var gt = data.ground_truth;
        var a = M.acc[seed];
        if (!a || a.count === 0) { scores.push({seed:seed, score:0, msg:'no replays'}); continue; }
        var N = a.count;

        // Score with adaptive alpha (default)
        var wkl = computeWKL(gt, a, N, alpha !== undefined ? alpha : true);
        var score = Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl)));

        // Also score with fixed alpha for comparison
        var fixedAl = Math.max(0.02, 0.15 * Math.sqrt(150 / N));
        var wklFixed = computeWKL(gt, a, N, fixedAl);
        var scoreFixed = Math.max(0, Math.min(100, 100 * Math.exp(-3 * wklFixed)));

        scores.push({seed:seed, score:+score.toFixed(2), scoreFixed:+scoreFixed.toFixed(2),
                      wkl:+wkl.toFixed(6), N:N});
        log('  Seed ' + seed + ': ' + score.toFixed(2) + ' adaptive | ' +
            scoreFixed.toFixed(2) + ' fixed (N=' + N + ')');
      } catch(e) {
        scores.push({seed:seed, score:0, msg:e.message});
      }
    }
    var avg = scores.reduce(function(s,v){return s+(v.score||0);},0) / SEEDS;
    var avgFixed = scores.reduce(function(s,v){return s+(v.scoreFixed||0);},0) / SEEDS;
    log('AVERAGE: ' + avg.toFixed(2) + ' adaptive | ' + avgFixed.toFixed(2) + ' fixed');
    scores.push({avg:+avg.toFixed(2), avgFixed:+avgFixed.toFixed(2)});
    M.scores = scores;
    return scores;
  };

  // Sweep alpha to find optimal (for fixed-alpha mode)
  M.tuneAlpha = async function(seed) {
    seed = seed || 0;
    log('Tuning alpha for seed ' + seed + '...');
    var resp = await fetch(BASE + '/analysis/' + M.rid + '/' + seed, {credentials:'include'});
    var data = await resp.json();
    var gt = data.ground_truth;
    var a = M.acc[seed];
    var N = a.count;

    var results = [];
    [0.001, 0.003, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5].forEach(function(al) {
      var wkl = computeWKL(gt, a, N, al);
      results.push({alpha: al, score: +(100 * Math.exp(-3 * wkl)).toFixed(3)});
    });

    // Also test adaptive
    var wklAdapt = computeWKL(gt, a, N, true);
    results.push({alpha: 'adaptive', score: +(100 * Math.exp(-3 * wklAdapt)).toFixed(3)});

    results.sort(function(a,b){return b.score - a.score;});
    log('Best: ' + results[0].alpha + ' → ' + results[0].score);
    log('All: ' + results.map(function(r){return r.alpha+'→'+r.score;}).join(', '));
    M.bestAlpha = results[0].alpha;
    return results;
  };

  // ═══════════════════════════════════════════════════════════════════════
  // PHASE 4b: CONTINUOUS COLLECTION
  // ═══════════════════════════════════════════════════════════════════════
  M.collectForever = function(concurrency) {
    concurrency = concurrency || 10;
    M._collectStop = false;
    log('Continuous collection started (concurrency ' + concurrency + '). Call _M.stopCollect() to stop.');

    var inf = 0, total = 0;
    function nextSeed() {
      // Pick seed with fewest replays
      var minN = Infinity, minS = 0;
      for (var s = 0; s < SEEDS; s++) {
        var n = M.acc[s] ? M.acc[s].count : 0;
        if (n < minN) { minN = n; minS = s; }
      }
      return minS;
    }

    function go() {
      if (M._collectStop) { log('Collection stopped. Total this session: ' + total); return; }
      while (inf < concurrency) {
        var seed = nextSeed(); inf++;
        fetchReplay(seed).then(function(d) {
          if (d && d.seed_index >= 0) addReplay(d.seed_index, d);
          inf--; total++;
          if (total % 25 === 0) {
            M.save();
            var cts = [0,1,2,3,4].map(function(s){return M.acc[s]?M.acc[s].count:0;});
            log('Continuous: +' + total + ' [' + cts.join(',') + ']');
            document.title = 'C:' + total + ' [' + cts.join(',') + ']';
          }
          go();
        }).catch(function() { inf--; setTimeout(go, 100); });
      }
    }
    go();
  };

  M.stopCollect = function() {
    M._collectStop = true;
    M.save();
    log('Stopping collection...');
  };

  // ═══════════════════════════════════════════════════════════════════════
  // PHASE 4c: DATA EXPORT / IMPORT (for Node.js ↔ browser sync)
  // ═══════════════════════════════════════════════════════════════════════
  M.exportData = function() {
    var data = JSON.stringify(M.acc);
    log('Data exported (' + (data.length / 1024).toFixed(0) + ' KB). Copy from console or use:');
    log('  copy(_M.exportData())  — copies to clipboard');
    return data;
  };

  M.importData = function(json) {
    var imported = typeof json === 'string' ? JSON.parse(json) : json;
    for (var s = 0; s < SEEDS; s++) {
      if (!imported[s] || imported[s].count === 0) continue;
      if (!M.acc[s] || M.acc[s].count === 0) {
        M.acc[s] = imported[s];
        log('  Seed ' + s + ': imported ' + imported[s].count + ' replays (new)');
      } else {
        // Merge: add counts together
        for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) {
          for (var c = 0; c < 6; c++) {
            M.acc[s].grid[y][x][c] += imported[s].grid[y][x][c];
          }
        }
        M.acc[s].count += imported[s].count;
        log('  Seed ' + s + ': merged +' + imported[s].count + ' → total ' + M.acc[s].count);
      }
    }
    M.save();
    log('Import complete');
  };

  // ═══════════════════════════════════════════════════════════════════════
  // PHASE 5: SUBMIT
  // ═══════════════════════════════════════════════════════════════════════
  M.submit = async function(seedIdx, alpha, adaptive) {
    if (!M.rid) { log('ERROR: call _M.init() first'); return; }
    if (M.round.status !== 'active') {
      log('WARNING: Round is ' + M.round.status + ', not active!');
    }

    var pred = M.buildPrediction(seedIdx, alpha, adaptive);
    if (!pred) return;

    // Validate
    var valid = true;
    for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) {
      var sum = 0;
      for (var c = 0; c < 6; c++) {
        if (pred[y][x][c] < 0) { valid = false; log('NEGATIVE at '+y+','+x+','+c); }
        sum += pred[y][x][c];
      }
      if (Math.abs(sum - 1.0) > 0.01) { valid = false; log('SUM != 1 at '+y+','+x+': '+sum); }
    }
    if (!valid) { log('VALIDATION FAILED — not submitting'); return; }

    log('Ready to submit seed ' + seedIdx + ' (N=' + M.acc[seedIdx].count + ')');
    log('*** CONFIRM: Type _M.confirmSubmit(' + seedIdx + ') to actually submit ***');

    M._pendingSubmit = {seed: seedIdx, pred: pred};
  };

  M.confirmSubmit = async function(seedIdx) {
    if (!M._pendingSubmit || M._pendingSubmit.seed !== seedIdx) {
      log('No pending submission for seed ' + seedIdx + '. Call _M.submit(' + seedIdx + ') first.');
      return;
    }

    var pred = M._pendingSubmit.pred;
    log('Submitting seed ' + seedIdx + '...');

    var resp = await fetch(BASE + '/submit', {
      method: 'POST', credentials: 'include',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        round_id: M.rid,
        seed_index: seedIdx,
        prediction: pred
      })
    });
    var result = await resp.json();
    log('RESPONSE: ' + JSON.stringify(result));
    M._pendingSubmit = null;
    return result;
  };

  M.submitAll = function(alpha, adaptive) {
    var cts = [0,1,2,3,4].map(function(s){return M.acc[s]?M.acc[s].count:0;});
    log('=== SUBMIT ALL 5 SEEDS (adaptive per-cell alpha) ===');
    log('Replay counts: [' + cts.join(',') + ']');
    log('Call _M.confirmSubmitAll() to confirm');
    M._pendingSubmitAll = {alpha: alpha, adaptive: adaptive};
  };

  M.confirmSubmitAll = async function() {
    if (!M._pendingSubmitAll) {
      log('No pending submitAll. Call _M.submitAll() first.');
      return;
    }
    var alpha = M._pendingSubmitAll.alpha;
    var adaptive = M._pendingSubmitAll.adaptive;
    var results = [];
    for (var s = 0; s < SEEDS; s++) {
      var pred = M.buildPrediction(s, alpha, adaptive);
      if (!pred) { log('Skipping seed ' + s + ' — no data'); continue; }

      // Validate before submitting
      var valid = true;
      for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) {
        var sum = 0;
        for (var c = 0; c < 6; c++) {
          if (pred[y][x][c] < 0) { valid = false; break; }
          sum += pred[y][x][c];
        }
        if (Math.abs(sum - 1.0) > 0.01) valid = false;
        if (!valid) break;
      }
      if (!valid) { log('VALIDATION FAILED for seed ' + s + ' — skipping'); continue; }

      log('Submitting seed ' + s + '...');
      var resp = await fetch(BASE + '/submit', {
        method: 'POST', credentials: 'include',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({round_id: M.rid, seed_index: s, prediction: pred})
      });
      var result = await resp.json();
      log('  Seed ' + s + ': ' + JSON.stringify(result));
      results.push(result);
    }
    M._pendingSubmitAll = null;
    log('All seeds submitted!');
    return results;
  };

  // ═══════════════════════════════════════════════════════════════════════
  // PHASE 6: STATUS & UTILITIES
  // ═══════════════════════════════════════════════════════════════════════
  M.status = function() {
    var counts = [];
    for (var s = 0; s < SEEDS; s++) counts.push(M.acc[s] ? M.acc[s].count : 0);
    log('Replays: [' + counts.join(',') + '] = ' + counts.reduce(function(a,b){return a+b;},0) + ' total');
    log('Round: ' + (M.round ? M.round.status : 'not initialized'));
    return counts;
  };

  M.leaderboard = async function() {
    var lb = await (await fetch(BASE + '/leaderboard', {credentials:'include'})).json();
    var entries = Array.isArray(lb) ? lb : (lb.entries || lb.leaderboard || []);
    log('=== LEADERBOARD (top 15) ===');
    for (var i = 0; i < Math.min(15, entries.length); i++) {
      var e = entries[i];
      log('  #' + (i+1) + ': ' + (e.team_name || e.username || e.name) + ' — ' + (e.score || e.total_score));
    }
    return entries;
  };

  M.myScores = async function() {
    var my = await (await fetch(BASE + '/my-rounds', {credentials:'include'})).json();
    log('Our rounds: ' + JSON.stringify(my).slice(0, 500));
    return my;
  };

  // ═══════════════════════════════════════════════════════════════════════
  // PHASE 7: AUTH HELPER (for Node.js script)
  // ═══════════════════════════════════════════════════════════════════════
  M.getToken = function() {
    // Extract JWT from cookie for use with Node.js script
    var cookies = document.cookie.split(';');
    for (var i = 0; i < cookies.length; i++) {
      var c = cookies[i].trim();
      if (c.startsWith('access_token=')) {
        var token = c.substring('access_token='.length);
        log('Token extracted! Use with Node.js:');
        log('  node collect_node.js --token ' + token.slice(0,20) + '...');
        return token;
      }
    }
    log('No access_token cookie found. Try logging in again.');
    return null;
  };

  // ═══════════════════════════════════════════════════════════════════════
  // AUTO-INIT
  // ═══════════════════════════════════════════════════════════════════════
  log('Mega Solver v5.1 loaded! (per-cell adaptive alpha)');
  log('');
  log('Quick start:');
  log('  await _M.init()');
  log('  _M.collectForever(10)     — continuous collection');
  log('  await _M.scoreVsGT()      — score with adaptive alpha');
  log('  _M.submitAll()            — prepare submission');
  log('  _M.confirmSubmitAll()     — submit all 5 seeds');
  log('');
  log('For Node.js: _M.getToken() — extract auth token');

})();
