# Plan: Universal Solver v2 — Smart Query Strategy

## Context
We're team **Paralov**, rank **#137** with weighted_score **43.0**. #1 has 158.12.
R12 is active (weight 1.7959, ~85 min left). We already burned 50 queries and submitted 5 seeds.
We can still **RESUBMIT** better predictions (overwrites last). No new queries needed.

## Immediate Action: Resubmit R12 with better fusion (~15 min)
We have 50 VP observations on disk (`data/viewport_795bfb1f.json`) + GT for R1-R11.
The autopilot's fusion was weak (CW=20, pw=15, temp=1.1). Fix these params and resubmit:

1. Load existing VP data + all GT/replay training data
2. Build model (same as autopilot — proven)
3. **Better fusion**: CW=5 (trust VP more), pw=3 (trust per-cell obs more), no temperature
4. **Cross-seed VP pooling**: pool all 50 VP obs into feature model (not just 10/seed)
5. Resubmit all 5 seeds

## Then: Rebuild universal_solver.js for future rounds

### Change 1: Smart VP Placement
**File**: `universal_solver.js` function `collectVP`

Replace dumb 3×3 grid with settlement-targeted viewports:
- Read initial_states → get all settlement (x,y) positions
- Simple clustering: divide map into quadrants, find settlement density
- Place 15×15 viewports centered on densest settlement areas
- Phase gates: `--phase 1` (5 queries recon), `--phase 2` (+15 coverage), `--phase 3` (+30 depth)
- **NEVER auto-fire all 50**. Each phase pauses and reports.

### Change 2: Cross-Seed VP Pooling
**File**: `universal_solver.js` function `fuseVP`

Current: pools VP by D0 key (already cross-seed). Good.
Fix: reduce CW from 20 to 5-10. VP is THIS round's data — it should dominate.

### Change 3: Better Per-Cell Weights
**File**: `universal_solver.js` function `applyPerCell`

Current: pw=15 for N=1 (barely moves prediction).
Fix: pw=3 always. 1 direct observation of a cell is worth more than a cross-round average.

### Change 4: Drop Temperature
**File**: `universal_solver.js` function `predict`

Remove TEMP=1.1. It flattens confident predictions for no reason.

### Change 5: Round-Type Detection
After phase 1 (5 queries), read settlement stats (pop, food, wealth, defense).
If avg food < 0.3 and many dead settlements → death round → include R3 in training.
If high pop and food → growth round → exclude R3.

## Files to Modify
- `universal_solver.js` — all changes above

## What NOT to Change
- Feature extraction D0-D4 keys — proven
- Model building from GT+replays — proven
- Dirichlet alpha=0.05 — proven
- Validation + submission mechanics — proven

## Verification
1. Run LOO on completed rounds with new fusion params → compare to old
2. Resubmit R12 immediately with improved params
3. Next round: use phase gates, never auto-fire
