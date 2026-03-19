// ROUND 2 EMERGENCY — Paste in browser console on app.ainm.no
// Step 1: Fires ALL remaining queries
// Step 2: Builds optimized predictions using query data + initial state priors
// Step 3: Waits for your confirm before submitting
(async function() {
  var B = 'https://api.ainm.no/astar-island';
  var R = '76909e29-f664-4b2f-b16b-61b7507277e9';
  var H = 40, W = 40, SEEDS = 5;
  function t2c(t) { return (t===10||t===11||t===0) ? 0 : ((t>=1&&t<=5) ? t : 0); }
  function log(m) { console.log('[R2] ' + m); }

  // ============ PHASE 1: CHECK STATUS ============
  log('Checking budget...');
  var bud = await (await fetch(B+'/budget', {credentials:'include'})).json();
  var left = bud.queries_max - bud.queries_used;
  log('Budget: ' + bud.queries_used + '/' + bud.queries_max + ' used, ' + left + ' remaining');

  log('Getting round details + initial states...');
  var det = await (await fetch(B+'/rounds/'+R, {credentials:'include'})).json();
  var initStates = det.initial_states;
  log('Got ' + initStates.length + ' initial states');

  // ============ PHASE 2: FIRE ALL QUERIES ============
  // 9 viewports cover the full 40x40 map with 15x15 windows
  var VPS = [
    {x:0,y:0},{x:13,y:0},{x:25,y:0},
    {x:0,y:13},{x:13,y:13},{x:25,y:13},
    {x:0,y:25},{x:13,y:25},{x:25,y:25}
  ];

  // Init accumulators
  var acc = {};
  for (var s = 0; s < SEEDS; s++) {
    acc[s] = {cnt: [], sam: []};
    for (var y = 0; y < H; y++) {
      acc[s].cnt[y] = []; acc[s].sam[y] = [];
      for (var x = 0; x < W; x++) {
        acc[s].cnt[y][x] = [0,0,0,0,0,0];
        acc[s].sam[y][x] = 0;
      }
    }
  }

  // Build query plan: ROUND-ROBIN across seeds, center-first viewports
  // Priority: center first (most dynamic), then corners, then edges
  var VP_PRIORITY = [
    {x:13,y:13}, // center (most dynamic cells)
    {x:0,y:0},{x:25,y:0},{x:0,y:25},{x:25,y:25}, // corners
    {x:13,y:0},{x:25,y:13},{x:0,y:13},{x:13,y:25}  // edges
  ];
  var plan = [];
  for (var v = 0; v < VP_PRIORITY.length; v++)
    for (var s = 0; s < SEEDS; s++)
      plan.push({s: s, x: VP_PRIORITY[v].x, y: VP_PRIORITY[v].y});

  plan = plan.slice(0, left); // don't exceed budget
  // Report coverage per seed
  var seedCov = [0,0,0,0,0];
  for (var i = 0; i < plan.length; i++) seedCov[plan[i].s]++;
  log('Query plan: ' + plan.length + ' queries, per-seed: [' + seedCov.join(',') + ']');
  log('Firing ' + plan.length + ' queries...');

  var fired = 0, errs = 0;
  for (var i = 0; i < plan.length; i++) {
    // Rate limit: max 4 per second
    if (fired > 0 && fired % 4 === 0)
      await new Promise(function(r) { setTimeout(r, 1100); });

    var p = plan[i];
    try {
      var resp = await fetch(B + '/simulate', {
        method: 'POST', credentials: 'include',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          round_id: R, seed_index: p.s,
          viewport_x: p.x, viewport_y: p.y,
          viewport_w: 15, viewport_h: 15
        })
      });
      var data = await resp.json();
      fired++;

      if (data.grid) {
        for (var gy = 0; gy < data.grid.length; gy++) {
          for (var gx = 0; gx < data.grid[gy].length; gx++) {
            var mY = p.y + gy, mX = p.x + gx;
            if (mY < H && mX < W) {
              acc[p.s].cnt[mY][mX][t2c(data.grid[gy][gx])]++;
              acc[p.s].sam[mY][mX]++;
            }
          }
        }
      } else {
        errs++;
        log('  Error on query ' + fired + ': ' + JSON.stringify(data).slice(0, 100));
      }

      if (fired % 5 === 0) {
        document.title = 'Querying: ' + fired + '/' + plan.length;
        log('  Progress: ' + fired + '/' + plan.length + ' (errors: ' + errs + ')');
      }
    } catch(e) {
      errs++; fired++;
      log('  Fetch error: ' + e.message);
    }
  }

  log('=== QUERIES DONE: ' + fired + ' fired, ' + errs + ' errors ===');

  // Save raw data
  localStorage.setItem('r2_query_acc', JSON.stringify(acc));
  localStorage.setItem('r2_init_states', JSON.stringify(initStates));

  // ============ PHASE 3: BUILD PREDICTIONS ============
  log('Building predictions with query data + initial state priors...');

  // Learn transition probabilities from ALL 5 Round 1 GT seeds
  var transitions = {};
  var transCounts = {};
  try {
    var r1id = '71451d74-be9f-471f-aacd-a41f3b68a9cd';
    var r1detail = await (await fetch(B+'/rounds/'+r1id, {credentials:'include'})).json();
    var r1Init = r1detail.initial_states;

    for (var gs = 0; gs < SEEDS; gs++) {
      var r1analysis = await (await fetch(B+'/analysis/'+r1id+'/'+gs, {credentials:'include'})).json();
      var r1GT = r1analysis.ground_truth;
      var r1Grid = r1Init[gs].grid;
      for (var y = 0; y < H; y++) {
        for (var x = 0; x < W; x++) {
          var key = r1Grid[y][x]; // raw terrain type
          if (!transitions[key]) { transitions[key] = [0,0,0,0,0,0]; transCounts[key] = 0; }
          for (var c = 0; c < 6; c++) transitions[key][c] += r1GT[y][x][c];
          transCounts[key]++;
        }
      }
      log('  Loaded R1 GT seed ' + gs + ' (' + transCounts[10] + ' ocean, ' + (transCounts[11]||0) + ' plains cells so far)');
    }
    for (var key in transitions) {
      var n = transCounts[key];
      for (var c = 0; c < 6; c++) transitions[key][c] /= n;
    }
    window._R1Trans = transitions;
    log('Learned transitions from 5 R1 GT seeds: ' + Object.keys(transitions).length + ' terrain types');
    for (var k in transitions) log('  Type ' + k + ': ' + transitions[k].map(function(v){return v.toFixed(3);}).join(','));
  } catch(e) {
    log('Could not load Round 1 GT: ' + e.message);
  }

  // Build predictions for each seed
  var predictions = {};
  for (var seed = 0; seed < SEEDS; seed++) {
    var pred = [];
    var initGrid = initStates[seed].grid;

    for (var y = 0; y < H; y++) {
      pred[y] = [];
      for (var x = 0; x < W; x++) {
        var N = acc[seed].sam[y][x]; // number of query samples for this cell
        var p = [0,0,0,0,0,0];

        if (N > 0) {
          // We have real data! Use it with Dirichlet smoothing.
          // For low N (1-4), use learned R1 transitions as pseudo-counts
          var initTerrain = initGrid[y][x];
          var prior = window._R1Trans && window._R1Trans[initTerrain]
            ? window._R1Trans[initTerrain] : [0.167, 0.167, 0.167, 0.167, 0.167, 0.167];

          // Bayesian approach: prior (from R1 transitions) + observed counts
          // prior_strength controls how much we trust the prior vs observed data
          var priorStrength = 2.0; // equivalent to 2 pseudo-observations from prior
          var total = N + priorStrength;
          for (var c = 0; c < 6; c++) {
            p[c] = (acc[seed].cnt[y][x][c] + priorStrength * prior[c]) / total;
          }
        } else {
          // No query data — use R1 transition probabilities as prediction
          var initTerrain = initGrid[y][x];
          if (window._R1Trans && window._R1Trans[initTerrain]) {
            p = window._R1Trans[initTerrain].slice();
          } else {
            // Ultimate fallback: initial state with smoothing
            var ic = t2c(initTerrain);
            var alpha = 0.05;
            for (var c = 0; c < 6; c++) p[c] = alpha / (1 + 6 * alpha);
            p[ic] = (1 + alpha) / (1 + 6 * alpha);
          }
        }

        // Ensure valid: non-negative, sum to 1.0
        var sum = 0, maxI = 0, maxV = 0;
        for (var c = 0; c < 6; c++) {
          if (p[c] < 0) p[c] = 0;
          p[c] = parseFloat(p[c].toFixed(6));
          sum += p[c];
          if (p[c] > maxV) { maxV = p[c]; maxI = c; }
        }
        p[maxI] = parseFloat((p[maxI] + (1.0 - sum)).toFixed(6));
        pred[y][x] = p;
      }
    }
    predictions[seed] = pred;
    log('Built prediction for seed ' + seed);
  }

  window._R2Preds = predictions;
  window._R2Acc = acc;
  log('');
  log('=== ALL PREDICTIONS BUILT ===');
  log('To submit: run _R2Submit()');
  log('');

  // ============ PHASE 4: SUBMIT FUNCTION ============
  window._R2Submit = async function() {
    log('=== SUBMITTING ALL 5 SEEDS ===');
    var results = [];
    for (var s = 0; s < SEEDS; s++) {
      var pred = window._R2Preds[s];
      if (!pred) { log('No prediction for seed ' + s); continue; }

      // Validate
      var valid = true;
      for (var y = 0; y < H && valid; y++) {
        for (var x = 0; x < W && valid; x++) {
          var sum = 0;
          for (var c = 0; c < 6; c++) {
            if (pred[y][x][c] < 0) valid = false;
            sum += pred[y][x][c];
          }
          if (Math.abs(sum - 1.0) > 0.01) valid = false;
        }
      }
      if (!valid) { log('VALIDATION FAILED for seed ' + s + ' — skipping!'); continue; }

      log('Submitting seed ' + s + '...');
      var resp = await fetch(B + '/submit', {
        method: 'POST', credentials: 'include',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({round_id: R, seed_index: s, prediction: pred})
      });
      var res = await resp.json();
      log('  Seed ' + s + ': ' + JSON.stringify(res));
      results.push({seed: s, result: res});
      document.title = 'Submitted seed ' + s;
    }
    log('=== ALL DONE ===');
    return results;
  };

  document.title = 'READY — Run _R2Submit() to submit';
  return 'DONE: ' + fired + ' queries fired, predictions built. Run _R2Submit() in console to submit.';
})();
