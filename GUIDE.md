# Astar Island Solver Guide — Team Paralov (studs.gg)

## NM i AI 2026 — Norwegian AI Championship

### Challenge Overview
Astar Island is a simulation-based prediction challenge. You're given a 40x40 grid map with terrain types (ocean, plains, forest, mountain) and initial settlements. The simulator runs 50 years of civilization dynamics — settlements grow, form ports, fight wars, create ruins, expand through forests. Your job: predict the **probability distribution** over 6 terrain classes for every cell at year 50.

**Scoring**: `score = 100 * exp(-3 * weighted_KL)` where weighted_KL is the entropy-weighted KL divergence between your prediction and the ground truth distribution. Only dynamic cells (entropy > 0.01) count.

### Our Approach: Three-Layer Prediction System

We built a **three-layer system** that combines learned transition patterns from past rounds with direct observations of the active round:

#### Layer 1: DIRECT Model (Initial → Final Mapping)
Instead of simulating step-by-step (which accumulates errors over 50 steps), we learn a **direct mapping** from initial cell features to final terrain distribution.

**Features extracted per cell:**
- **Terrain type** (plains=11, forest=4, settlement=1, port=2, ruin=3)
- **Neighbor settlements** (count in radius-1, binned)
- **Coastal flag** (adjacent to ocean)
- **Settlement density R2** (count in radius-2, excluding R1, binned into 4 levels)
- **Settlement density R3** (count in radius-3, excluding R2, binned into 3 levels)
- **Forest density** (forest neighbors, binned into 3 levels)

**Hierarchical key system** (5 levels, D0 most specific → D4 least):
```
D0: terrain_settleR1_coastal_settleBinR2_forestBin_settleBinR3
D1: terrain_settleR1_coastal_settleBinR2_forestBin
D2: terrain_settleR1capped_coastal_settleBinR2
D3: terrain_hasSettle_coastal
D4: terrain
```

The model falls back from D0 → D4 based on minimum sample thresholds (30, 20, 10, 5, any). This ensures we always have enough data for reliable distribution estimates.

**Data source**: Replay API on completed rounds (R1 + R2). Each replay runs the real server-side simulator with a random seed, giving us frame 0 (initial) and frame 50 (final). We aggregate hundreds of replays to build robust distributions per feature key.

#### Layer 2: Query Observations (Ground Truth Snapshots)
For the active round (R3), replays aren't available. We use the 50 allocated `/simulate` queries strategically:

- **9 tiled viewports per seed** at year 50, covering the entire 40x40 map
- Positions: `(0,0), (0,13), (0,25), (13,0), (13,13), (13,25), (25,0), (25,13), (25,25)`
- Each viewport is 15x15, giving us at least 1 real observation per cell
- Overlap regions get 2+ observations
- Total: 45 queries (9 per seed x 5 seeds), leaving 4-5 spare

Each query runs the actual R3 simulator with a new random seed, giving us one sample from the true distribution.

#### Layer 3: Ultra-Blend (Adaptive Fusion)
We blend the DIRECT model with query observations using **entropy-adaptive weights**:

- **Static cells** (DIRECT entropy < 0.15, observation agrees): Heavy query weight (wD=1, wQ=8) — the observation confirms the prediction
- **Surprise cells** (DIRECT entropy < 0.15, observation disagrees): Moderate blend (wD=2, wQ=3) — the cell is more dynamic than expected
- **Semi-dynamic cells** (entropy 0.15-0.5): Equal weight (wD=2, wQ=2)
- **Highly dynamic cells** (entropy > 0.5): DIRECT model dominates (wD=3, wQ=1) — one observation is too noisy for high-entropy cells

Per-cell Dirichlet smoothing with adaptive alpha: 0.0002 (1 class) to 0.002 (3+ classes).

### Replay Collection Pipeline

```
Replay API (POST /replay) → unlimited, completed rounds only
├── R1: 71451d74-be9f-471f-aacd-a41f3b68a9cd (completed)
├── R2: 76909e29-f664-4b2f-b16b-61b7507277e9 (completed)
└── Each returns 51 frames (year 0..50) of the 40x40 grid
```

- Collected 500+ replays interleaved across R1 and R2
- 5 concurrent requests to avoid rate limiting
- Each replay processed into DIRECT model (initial→final) and transition table (step-by-step)
- All run in-browser via Chrome console (auth cookie handles authentication)

### Step-by-Step MC Simulator (Backup)

We also built a step-by-step Monte Carlo simulator using learned transition probabilities:

- **Transition table**: 5-level hierarchical keys (L0-L4) with terrain, phase, neighbors, coastal, density
- **Phases**: 0-4 (years 0-9, 10-19, 20-29, 30-39, 40-50)
- Run 500-1000 simulations per seed, aggregate final states
- Validated at ~84-88 on R1/R2 GT (worse than DIRECT due to error accumulation)
- Used as a sanity check, not in final predictions

### Terrain Code Mapping

```
GT Class 0 ("plains") = ocean(10) + plains(11) + empty(0)
GT Class 1 = settlement(1)
GT Class 2 = port(2)
GT Class 3 = ruin(3)
GT Class 4 = forest(4)
GT Class 5 = mountain(5)
```

- Ocean and mountain cells are always static (100% one class)
- Dynamic cells are near settlements: plains→settlement, forest→settlement, settlement→ruin, etc.

### Key Files

| File | Purpose |
|------|---------|
| `solver_r3.js` | Full Node.js solver with CLI (train, predict, replay, score) |
| `mega_solver_v5.js` | Browser-based replay aggregation with per-cell adaptive alpha |
| `collect_node.js` | Standalone Node.js replay collector |
| `r2_ultimate.js` | R2 browser solver (transitions + GT matching) |
| `mega_solver.js` | Original agent-based simulator (13 parameter regimes) |
| `diagnose.js` | Self-consistency analysis and alpha tuning |
| `gt_analyzer.js` | Post-round GT analysis tool |

### Score Progression

| Version | Approach | R1 Score | R2 Score |
|---------|----------|----------|----------|
| V1 (mega_solver) | Agent-based sim, 13 regimes | 46.38 | — |
| V5 (replay-based) | Pure replay aggregation, 150 replays | 93.67 | — |
| V5.1 (adaptive alpha) | Per-cell adaptive Dirichlet | ~94+ | 96.4 |
| R3 DIRECT | Initial→final mapping, 421+ replays | 88.24 | 90.22 |
| R3 Ultra-Blend | DIRECT + 45 query observations | **~90-93 est** | — |

### Critical Insights

1. **Replays are gold**: The replay API runs the REAL server simulator. Aggregating many replays gives near-perfect distributions for completed rounds. No amount of hand-crafted simulation can match it.

2. **Direct > Step-by-step**: Mapping initial features directly to final state avoids 50 steps of error accumulation. DIRECT scored 88+ vs MC's 84.

3. **Queries are strategic, not brute-force**: With only 50 queries per round, tile the map at year 50 for complete single-sample coverage. Don't waste queries on intermediate years unless you have a specific calibration plan.

4. **Adaptive blending matters**: Static cells need strong observation weight; dynamic cells need the learned prior. One-size-fits-all blending is suboptimal.

5. **Round weights are the secret weapon**: R3 weight 1.1576 means raw 86.4 → weighted 100. Focus on later rounds which carry higher weight.

6. **Re-submission overwrites**: You can iterate! Submit early with a baseline, then keep improving and resubmitting until the round closes.

### API Reference

```
Base: https://api.ainm.no/astar-island

GET  /rounds                     — List all rounds
GET  /rounds/{id}                — Round details + initial_states
POST /replay                     — Full 51-frame replay (completed rounds only)
POST /simulate                   — 15x15 viewport observation (costs 1 query)
POST /submit                     — Submit prediction (overwrites previous)
GET  /analysis/{round_id}/{seed} — Ground truth (after round completes)
GET  /leaderboard                — Public leaderboard
GET  /my-rounds                  — Your submission status
```

### Team
**Paralov — studs.gg** (slug: tastebrettenes-venner)
NM i AI 2026 — Astar Island Challenge
