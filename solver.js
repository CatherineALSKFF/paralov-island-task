/**
 * Astar Island Round Solver — runs entirely in browser console.
 *
 * USAGE: Copy-paste into browser console on app.ainm.no (must be logged in).
 *
 * This script:
 * 1. Fetches active round + initial states
 * 2. Runs queries in batches of 5, logging results as it goes
 * 3. Builds predictions from observations + initial state heuristics
 * 4. Submits all 5 seeds
 *
 * All results are stored in window._solver so nothing is lost if something breaks.
 */

(async function AstarSolver() {
  const BASE = 'https://api.ainm.no';
  const FLOOR = 0.01;
  const DELAY = 250; // ms between API calls (5 req/s limit)

  // Persistent state — survives even if the script errors partway through
  if (!window._solver) window._solver = {};
  const S = window._solver;

  function log(msg) { console.log(`[ASTAR] ${msg}`); }

  // ─── PHASE 1: GET ROUND DATA ───
  log('Phase 1: Fetching round data...');
  const roundsResp = await fetch(`${BASE}/astar-island/rounds`, {credentials: 'include'});
  const rounds = await roundsResp.json();
  const active = rounds.find(r => r.status === 'active');
  if (!active) { log('ERROR: No active round!'); return; }

  const ROUND_ID = active.id;
  log(`Active round: ${active.round_number} (${ROUND_ID})`);

  const detailResp = await fetch(`${BASE}/astar-island/rounds/${ROUND_ID}`, {credentials: 'include'});
  const detail = await detailResp.json();
  S.detail = detail;
  const H = detail.map_height;
  const W = detail.map_width;
  const SEEDS = detail.seeds_count;
  log(`Map: ${W}x${H}, Seeds: ${SEEDS}`);

  // ─── PHASE 2: CHECK BUDGET ───
  const budgetResp = await fetch(`${BASE}/astar-island/budget`, {credentials: 'include'});
  const budget = await budgetResp.json();
  S.budget = budget;
  log(`Budget: ${budget.queries_used}/${budget.queries_max} used`);
  const queriesLeft = budget.queries_max - budget.queries_used;
  if (queriesLeft === 0) {
    log('WARNING: No queries left! Will predict from initial state only.');
  }

  // ─── PHASE 3: PLAN VIEWPORTS ───
  // 3x3 tiling covers 40x40 map with 15x15 viewports
  const TILE_POSITIONS = [
    [0, 0],   [0, 13],  [0, 25],
    [13, 0],  [13, 13], [13, 25],
    [25, 0],  [25, 13], [25, 25]
  ];

  // Allocate queries: 9 tiles per seed to cover full map, +1 extra on center
  const queriesPerSeed = Math.floor(queriesLeft / SEEDS);
  log(`Queries per seed: ${queriesPerSeed}`);

  // ─── PHASE 4: RUN QUERIES ───
  if (!S.observations) S.observations = {};

  if (queriesLeft > 0) {
    log('Phase 4: Running simulation queries...');
    let totalRun = 0;

    for (let seed = 0; seed < SEEDS; seed++) {
      if (!S.observations[seed]) S.observations[seed] = [];
      const maxQ = Math.min(queriesPerSeed, TILE_POSITIONS.length);

      for (let qi = 0; qi < maxQ; qi++) {
        if (totalRun >= queriesLeft) break;
        const [vx, vy] = qi < TILE_POSITIONS.length
          ? TILE_POSITIONS[qi]
          : [12, 12]; // extra query on center

        try {
          const resp = await fetch(`${BASE}/astar-island/simulate`, {
            method: 'POST',
            credentials: 'include',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              round_id: ROUND_ID,
              seed_index: seed,
              viewport_x: vx,
              viewport_y: vy,
              viewport_w: 15,
              viewport_h: 15
            })
          });
          const result = await resp.json();

          if (resp.ok) {
            S.observations[seed].push({
              vx, vy, vw: 15, vh: 15,
              grid: result.grid,
              settlements: result.settlements
            });
            totalRun++;
            log(`  Seed ${seed}, viewport (${vx},${vy}): OK [${result.queries_used}/${result.queries_max}]`);
          } else {
            log(`  Seed ${seed}, viewport (${vx},${vy}): ERROR ${resp.status} - ${JSON.stringify(result)}`);
            if (resp.status === 429) {
              log('  Rate limited or budget exhausted. Stopping queries.');
              break;
            }
          }
        } catch (e) {
          log(`  Seed ${seed}, viewport (${vx},${vy}): EXCEPTION ${e.message}`);
        }
        await new Promise(r => setTimeout(r, DELAY));
      }
    }
    log(`Phase 4 complete: ${totalRun} queries executed.`);
  }

  // ─── PHASE 5: BUILD PREDICTIONS ───
  log('Phase 5: Building predictions...');

  function buildPrediction(seedIdx) {
    const state = detail.initial_states[seedIdx];
    const grid = state.grid;
    const settlements = state.settlements;
    const obs = S.observations[seedIdx] || [];

    // Build observation count map: for each cell, count terrain classes seen
    // counts[y][x] = [c0, c1, c2, c3, c4, c5]
    const counts = Array.from({length: H}, () =>
      Array.from({length: W}, () => new Float32Array(6))
    );
    let totalObs = 0;

    for (const o of obs) {
      if (!o.grid) continue;
      for (let gy = 0; gy < o.grid.length; gy++) {
        for (let gx = 0; gx < o.grid[gy].length; gx++) {
          const mapY = o.vy + gy;
          const mapX = o.vx + gx;
          if (mapY >= H || mapX >= W) continue;
          const terrainCode = o.grid[gy][gx];
          // Map terrain code to class
          let cls;
          if (terrainCode === 10 || terrainCode === 11 || terrainCode === 0) cls = 0;
          else if (terrainCode === 1) cls = 1;
          else if (terrainCode === 2) cls = 2;
          else if (terrainCode === 3) cls = 3;
          else if (terrainCode === 4) cls = 4;
          else if (terrainCode === 5) cls = 5;
          else cls = 0;
          counts[mapY][mapX][cls]++;
          totalObs++;
        }
      }
    }
    log(`  Seed ${seedIdx}: ${obs.length} observations, ${totalObs} cell samples`);

    // Precompute per-cell features
    const distToSettlement = Array.from({length: H}, () => new Float32Array(W).fill(999));
    const nearCoast = Array.from({length: H}, () => new Uint8Array(W));
    const neighborDensity = Array.from({length: H}, () => new Float32Array(W));

    for (const s of settlements) {
      for (let y = Math.max(0, s.y - 10); y < Math.min(H, s.y + 11); y++) {
        for (let x = Math.max(0, s.x - 10); x < Math.min(W, s.x + 11); x++) {
          const d = Math.abs(y - s.y) + Math.abs(x - s.x);
          if (d < distToSettlement[y][x]) distToSettlement[y][x] = d;
          if (d <= 6) neighborDensity[y][x]++;
        }
      }
    }

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
          const ny = y+dy, nx = x+dx;
          if (ny>=0 && ny<H && nx>=0 && nx<W && grid[ny][nx] === 10) {
            nearCoast[y][x] = 1;
          }
        }
      }
    }

    // Count forests adjacent to each settlement
    const settleForestCount = {};
    for (const s of settlements) {
      let fc = 0;
      for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]) {
        const ny = s.y+dy, nx = s.x+dx;
        if (ny>=0 && ny<H && nx>=0 && nx<W && grid[ny][nx] === 4) fc++;
      }
      settleForestCount[`${s.x},${s.y}`] = fc;
    }

    // Build prediction
    const pred = [];
    for (let y = 0; y < H; y++) {
      const row = [];
      for (let x = 0; x < W; x++) {
        const totalSamples = counts[y][x].reduce((a, b) => a + b, 0);
        let probs;

        if (totalSamples >= 1) {
          // HAVE OBSERVATIONS — use Bayesian with Dirichlet prior
          const alpha = getHeuristicPrior(grid[y][x], distToSettlement[y][x],
            nearCoast[y][x], neighborDensity[y][x], settleForestCount[`${x},${y}`] || 0);
          // Posterior = alpha + counts
          probs = alpha.map((a, i) => a + counts[y][x][i]);
        } else {
          // NO OBSERVATIONS — use heuristic prior only
          probs = getHeuristicPrior(grid[y][x], distToSettlement[y][x],
            nearCoast[y][x], neighborDensity[y][x], settleForestCount[`${x},${y}`] || 0);
        }

        // Apply floor and normalize
        probs = probs.map(p => Math.max(p, FLOOR));
        const sum = probs.reduce((a, b) => a + b, 0);
        probs = probs.map(p => parseFloat((p / sum).toFixed(4)));
        // Fix rounding
        const roundDiff = 1.0 - probs.reduce((a, b) => a + b, 0);
        probs[0] = parseFloat((probs[0] + roundDiff).toFixed(4));

        row.push(probs);
      }
      pred.push(row);
    }
    return pred;
  }

  function getHeuristicPrior(terrain, dist, isCoast, density, forestAdj) {
    // Returns Dirichlet-like prior weights [empty, settlement, port, ruin, forest, mountain]
    // Higher values = stronger prior belief

    if (terrain === 10) {
      // Ocean — never changes
      return [50, 0.1, 0.1, 0.1, 0.1, 0.5];
    }
    if (terrain === 5) {
      // Mountain — never changes
      return [0.5, 0.1, 0.1, 0.1, 0.1, 50];
    }

    const survivalBoost = Math.min(forestAdj * 0.5, 2.0);
    const conflictPenalty = Math.min(density * 0.3, 2.0);

    if (terrain === 1) {
      // Settlement
      const portChance = isCoast ? 1.5 : 0.2;
      return [
        1.0,                                         // empty
        4.0 + survivalBoost,                         // settlement (survive)
        portChance,                                  // port
        3.0 - survivalBoost + conflictPenalty,       // ruin
        1.2,                                         // forest reclaim
        0.1                                          // mountain
      ];
    }
    if (terrain === 2) {
      // Port — trades, wealthier
      return [
        0.8,                                         // empty
        1.5,                                         // settlement
        5.0 + survivalBoost,                         // port (survive)
        2.5 - survivalBoost + conflictPenalty,       // ruin
        1.0,                                         // forest
        0.1                                          // mountain
      ];
    }
    if (terrain === 3) {
      // Ruin — could be rebuilt or reclaimed
      return [2.0, 1.5, 0.5, 3.0, 2.5, 0.1];
    }
    if (terrain === 4) {
      // Forest
      if (dist <= 2) return [1.5, 1.2, 0.3, 0.6, 6.0, 0.1];
      if (dist <= 5) return [0.8, 0.5, 0.2, 0.3, 9.0, 0.1];
      return [0.3, 0.1, 0.1, 0.1, 12.0, 0.2];
    }

    // Plains (11) or Empty (0)
    if (dist <= 2) {
      const portC = isCoast ? 0.8 : 0.1;
      return [4.0, 2.5, portC, 1.2, 1.8, 0.1];
    }
    if (dist <= 4) {
      return [6.0, 1.2, isCoast ? 0.4 : 0.1, 0.6, 2.2, 0.1];
    }
    if (dist <= 7) {
      return [8.0, 0.5, 0.1, 0.3, 2.5, 0.1];
    }
    // Far from everything
    return [10.0, 0.2, 0.1, 0.2, 1.8, 0.2];
  }

  // ─── PHASE 6: SUBMIT ALL SEEDS ───
  log('Phase 6: Submitting predictions...');
  S.submissions = [];

  for (let seed = 0; seed < SEEDS; seed++) {
    const prediction = buildPrediction(seed);
    S[`pred_${seed}`] = prediction; // Save in case we need to debug

    const resp = await fetch(`${BASE}/astar-island/submit`, {
      method: 'POST',
      credentials: 'include',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        round_id: ROUND_ID,
        seed_index: seed,
        prediction: prediction
      })
    });
    const result = await resp.json();
    S.submissions.push({seed, status: resp.status, result});
    log(`  Seed ${seed}: ${resp.status} — ${result.status || JSON.stringify(result)}`);
    await new Promise(r => setTimeout(r, 600));
  }

  log('=== ALL DONE ===');
  log(`Submitted ${S.submissions.filter(s => s.status === 200).length}/5 seeds successfully.`);
  log('Results stored in window._solver');
  return S.submissions;
})();
