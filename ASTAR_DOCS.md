# Astar Island — Full Documentation (Saved from user message)

## Simulation Mechanics

### The World
- 40x40 grid, 8 terrain types → 6 prediction classes
- Ocean(10), Plains(11), Empty(0) → Class 0
- Settlement(1) → Class 1, Port(2) → Class 2, Ruin(3) → Class 3
- Forest(4) → Class 4, Mountain(5) → Class 5
- Mountains STATIC (never change), Forests mostly static but can reclaim ruins

### Map Generation
- Ocean borders, fjords from random edges, mountain chains via random walks
- Forest patches clustered, initial settlements on land spaced apart

### Simulation Lifecycle (50 years, each year cycles through):

#### 1. Growth
- Settlements produce food based on adjacent terrain
- Prosperous settlements grow population, develop ports on coast, build longships
- Expansion: found new settlements on nearby land

#### 2. Conflict
- Settlements raid each other
- Longships extend raiding range
- Desperate (low food) settlements raid more aggressively
- Successful raids: loot resources, damage defender
- Conquered settlements may change faction (owner_id)

#### 3. Trade
- Ports trade if not at war, within range
- Generates wealth + food, technology diffuses

#### 4. Winter
- Varying severity each year
- All settlements lose food
- Collapse conditions: starvation, sustained raids, harsh winters → become Ruins
- Population disperses to nearby friendly settlements

#### 5. Environment
- Ruins reclaimed by nearby thriving settlements (new outposts, inherit resources)
- Coastal ruins restored as ports
- Otherwise: ruins → forest growth or → plains

### Settlement Properties
- position, population, food, wealth, defense, tech_level, port_status, longship, faction(owner_id)
- Initial states only show position + port status (internal stats hidden)

### Hidden Parameters
- Control world behavior, SAME for all 5 seeds in a round
- Different between rounds
- This is why cross-round prediction is hard!

## API Endpoints
- GET /rounds — list all
- GET /rounds/{id} — details + initial_states
- GET /budget — remaining queries
- POST /simulate — viewport observation (costs 1 query, 50/round)
- POST /submit — submit prediction (resubmission overwrites)
- GET /my-rounds — team scores/rank/budget
- GET /my-predictions/{round_id} — predictions with argmax/confidence
- GET /analysis/{round_id}/{seed_index} — GT after completion
- GET /leaderboard — public
- POST /replay — UNDOCUMENTED but works, unlimited, full 51 frames

## Scoring
- score = max(0, min(100, 100 × exp(-3 × weighted_kl)))
- weighted_kl = Σ entropy(cell) × KL(GT, pred) / Σ entropy(cell)
- Only dynamic cells (entropy > 0.01) count
- NEVER assign 0.0 probability — use floor
- Round score = average of 5 seeds
- Leaderboard = best round_score × round_weight ever

## Key Insight
- Each round has HIDDEN PARAMETERS that are the same across all 5 seeds
- Different rounds have different parameters → model must generalize
- Top teams likely have accurate forward simulator or parameter inference
