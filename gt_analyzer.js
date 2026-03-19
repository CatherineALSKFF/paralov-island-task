/**
 * GT Analyzer — Paste into browser console AFTER Round 1 completes
 * Fetches real ground truth, analyzes our v1 predictions vs GT,
 * identifies exactly where we lost points and what to fix.
 *
 * Usage: paste this after loader.js or mega_solver_v4.js is loaded
 *   1. await analyzeRound1GT()
 *   2. Review output
 *
 * *** NO submissions, NO queries — read-only analysis ***
 */
async function analyzeRound1GT() {
  var BASE = 'https://api.ainm.no/astar-island';
  var ROUND1_ID = '71451d74-be9f-471f-aacd-a41f3b68a9cd';
  var SEEDS = 5, H = 40, W = 40;

  console.log('=== ROUND 1 GT ANALYSIS ===');

  // Step 1: Check round status
  console.log('\n[1] Checking round status...');
  var rounds = await (await fetch(BASE+'/rounds',{credentials:'include'})).json();
  var r1 = rounds.find(function(r){return r.id === ROUND1_ID;});
  console.log('Round 1 status: ' + (r1 ? r1.status : 'NOT FOUND'));
  if (r1 && r1.status !== 'completed') {
    console.log('Round 1 not yet completed! Status: ' + r1.status);
    console.log('Wait for scoring to finish, then re-run this script.');
    return {status: r1 ? r1.status : 'unknown'};
  }

  // Step 2: Fetch GT for all seeds
  console.log('\n[2] Fetching ground truth...');
  var gt = {};
  var scores = {};
  for (var s = 0; s < SEEDS; s++) {
    try {
      var resp = await fetch(BASE+'/analysis/'+ROUND1_ID+'/'+s, {credentials:'include'});
      if (resp.status !== 200) {
        console.log('  Seed ' + s + ': HTTP ' + resp.status + ' (GT not available)');
        continue;
      }
      var data = await resp.json();
      gt[s] = data.ground_truth;
      scores[s] = data.score;
      console.log('  Seed ' + s + ': score=' + data.score +
                  ', GT shape=' + data.ground_truth.length + 'x' + data.ground_truth[0].length +
                  'x' + data.ground_truth[0][0].length);
    } catch(e) {
      console.log('  Seed ' + s + ': ERROR — ' + e.message);
    }
  }

  var seedsWithGT = Object.keys(gt).map(Number);
  if (seedsWithGT.length === 0) {
    console.log('No GT available yet!');
    return {status: 'no_gt'};
  }

  // Step 3: Analyze GT characteristics
  console.log('\n[3] Analyzing GT characteristics...');
  for (var s of seedsWithGT) {
    var g = gt[s];
    var dynamicCells = 0, totalEnt = 0, maxEnt = 0;
    var classDistrib = [0,0,0,0,0,0]; // how many cells have each class as dominant
    var zeroClassCount = [0,0,0,0,0,0,0]; // how many cells have 0,1,2,...,6 zero-prob classes
    var floorNeeded = 0; // cells where GT has classes < 0.01

    for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) {
      var cell = g[y][x];
      var ent = 0;
      var dominant = 0, maxP = 0;
      var zeroCount = 0;
      var belowFloor = false;
      for (var c = 0; c < 6; c++) {
        if (cell[c] > maxP) { maxP = cell[c]; dominant = c; }
        if (cell[c] > 1e-6) ent -= cell[c] * Math.log(cell[c]);
        if (cell[c] < 0.001) zeroCount++;
        if (cell[c] > 0 && cell[c] < 0.01) belowFloor = true;
      }
      classDistrib[dominant]++;
      zeroClassCount[zeroCount]++;
      if (belowFloor) floorNeeded++;
      if (ent > 0.01) { dynamicCells++; totalEnt += ent; }
      if (ent > maxEnt) maxEnt = ent;
    }

    console.log('\n  Seed ' + s + ' (official score: ' + scores[s] + '):');
    console.log('    Dynamic cells (ent>0.01): ' + dynamicCells + ' / ' + (H*W));
    console.log('    Avg entropy (dynamic): ' + (dynamicCells > 0 ? (totalEnt/dynamicCells).toFixed(3) : 'n/a'));
    console.log('    Max entropy: ' + maxEnt.toFixed(3));
    console.log('    Dominant class distribution: plains=' + classDistrib[0] +
                ' settle=' + classDistrib[1] + ' port=' + classDistrib[2] +
                ' ruin=' + classDistrib[3] + ' forest=' + classDistrib[4] +
                ' mountain=' + classDistrib[5]);
    console.log('    GT cells with below-0.01 non-zero probs: ' + floorNeeded);
    console.log('    Zero-class counts per cell: ' + zeroClassCount.map(function(v,i){return i+'zeros:'+v;}).join(' '));
  }

  // Step 4: Compute theoretical floor penalty
  console.log('\n[4] Theoretical floor penalty analysis...');
  for (var s of seedsWithGT) {
    var g = gt[s];
    // Score a PERFECT prediction (GT itself) vs GT — should be 100
    var perfectScore = computeScore(g, g);
    // Score a floored GT vs GT — shows floor penalty
    var flooredGT = applyFloor(g);
    var flooredScore = computeScore(flooredGT, g);
    console.log('  Seed ' + s + ': perfect=' + perfectScore.toFixed(2) +
                ', floored-perfect=' + flooredScore.toFixed(2) +
                ', floor penalty=' + (perfectScore - flooredScore).toFixed(2));
  }

  // Step 5: Our submission analysis
  console.log('\n[5] Checking our submission scores...');
  try {
    var myRounds = await (await fetch(BASE+'/my-rounds',{credentials:'include'})).json();
    console.log('  Our rounds: ' + JSON.stringify(myRounds).slice(0, 500));
  } catch(e) {
    console.log('  Could not fetch our scores: ' + e.message);
  }

  // Step 6: Leaderboard
  console.log('\n[6] Leaderboard...');
  try {
    var lb = await (await fetch(BASE+'/leaderboard',{credentials:'include'})).json();
    console.log('  Top 10:');
    var entries = Array.isArray(lb) ? lb : (lb.entries || lb.leaderboard || []);
    for (var i = 0; i < Math.min(10, entries.length); i++) {
      var e = entries[i];
      console.log('    #' + (i+1) + ': ' + (e.team_name || e.username || e.name || 'unknown') +
                  ' — score=' + (e.score || e.total_score || 'n/a'));
    }
  } catch(e) {
    console.log('  Could not fetch leaderboard: ' + e.message);
  }

  // Step 7: Detailed GT data dump (for offline analysis)
  console.log('\n[7] GT data available in window._GT');
  window._GT = {gt: gt, scores: scores, roundId: ROUND1_ID};

  // Return summary
  return {
    scores: scores,
    seedsAvailable: seedsWithGT,
    gt: gt
  };

  // Helper functions
  function computeScore(pred, gt) {
    var totalKL = 0, totalEnt = 0;
    for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) {
      var g = gt[y][x];
      var ent = 0;
      for (var c = 0; c < 6; c++) if (g[c] > 1e-6) ent -= g[c] * Math.log(g[c]);
      if (ent < 0.01) continue;
      var kl = 0;
      for (var c = 0; c < 6; c++) {
        if (g[c] > 1e-6) kl += g[c] * Math.log(g[c] / Math.max(pred[y][x][c], 1e-10));
      }
      totalKL += Math.max(0, kl) * ent;
      totalEnt += ent;
    }
    var wkl = totalEnt > 0 ? totalKL / totalEnt : 0;
    return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl)));
  }

  function applyFloor(dist) {
    var FLOOR = 0.01;
    var result = [];
    for (var y = 0; y < H; y++) {
      result[y] = [];
      for (var x = 0; x < W; x++) {
        var p = dist[y][x].slice();
        for (var c = 0; c < 6; c++) p[c] = Math.max(FLOOR, p[c]);
        var s = 0; for (var c = 0; c < 6; c++) s += p[c];
        for (var c = 0; c < 6; c++) p[c] /= s;
        result[y][x] = p;
      }
    }
    return result;
  }
}

// Auto-run when pasted
console.log('GT Analyzer loaded. Run: await analyzeRound1GT()');
console.log('Or wait — checking status automatically...');
analyzeRound1GT().then(function(r) {
  if (r.status === 'no_gt') console.log('GT not available yet. Re-run later.');
  else if (r.scores) console.log('Analysis complete! GT stored in window._GT');
});
