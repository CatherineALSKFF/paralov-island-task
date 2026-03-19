"""Analyze initial map state and plan optimal query viewports.

Strategy: Focus ALL queries on dynamic zones (near settlements).
Static cells (ocean, mountains, distant forests) are predicted from initial state.
"""
import numpy as np
from typing import List, Tuple, Dict
from client import TERRAIN_TO_CLASS, STATIC_TERRAINS, NUM_CLASSES


def parse_initial_state(state: dict) -> Tuple[np.ndarray, List[dict]]:
    """Parse initial state → grid array + settlement list."""
    grid = np.array(state["grid"])
    settlements = state.get("settlements", [])
    return grid, settlements


def terrain_to_class(grid: np.ndarray) -> np.ndarray:
    """Map terrain codes → prediction class indices."""
    class_grid = np.zeros_like(grid, dtype=int)
    for code, cls in TERRAIN_TO_CLASS.items():
        class_grid[grid == code] = cls
    return class_grid


def compute_dynamic_mask(grid: np.ndarray, settlements: List[dict],
                         radius: int = 6) -> np.ndarray:
    """Identify cells that could change during simulation.

    A cell is dynamic if:
    - It's within `radius` Manhattan distance of any settlement AND
    - It's NOT ocean (10) or mountain (5)

    These are the ONLY cells that affect scoring (entropy > 0).
    """
    h, w = grid.shape
    dynamic = np.zeros((h, w), dtype=bool)

    for s in settlements:
        sx, sy = s["x"], s["y"]
        y_lo = max(0, sy - radius)
        y_hi = min(h, sy + radius + 1)
        x_lo = max(0, sx - radius)
        x_hi = min(w, sx + radius + 1)

        for y in range(y_lo, y_hi):
            for x in range(x_lo, x_hi):
                if abs(y - sy) + abs(x - sx) <= radius:
                    code = grid[y, x]
                    if code not in STATIC_TERRAINS:
                        dynamic[y, x] = True

    return dynamic


def plan_query_allocation(initial_states: List[dict],
                          total_budget: int = 50) -> Dict[int, int]:
    """Allocate query budget across seeds based on dynamic zone complexity.

    Seeds with more dynamic cells get more queries.
    Minimum 8 queries per seed to ensure coverage.
    """
    n_seeds = len(initial_states)
    complexities = []

    for state in initial_states:
        grid, settlements = parse_initial_state(state)
        dynamic = compute_dynamic_mask(grid, settlements)
        complexities.append(np.sum(dynamic))

    total_complexity = sum(complexities) or 1
    allocation = {}

    for i, c in enumerate(complexities):
        # Proportional allocation with minimum floor
        share = max(8, int(total_budget * c / total_complexity))
        allocation[i] = share

    # Normalize to fit budget
    total_alloc = sum(allocation.values())
    if total_alloc > total_budget:
        # Scale down proportionally
        scale = total_budget / total_alloc
        for i in allocation:
            allocation[i] = max(6, int(allocation[i] * scale))

    # Distribute remaining budget to most complex seeds
    remaining = total_budget - sum(allocation.values())
    sorted_seeds = sorted(allocation.keys(),
                          key=lambda i: complexities[i], reverse=True)
    for i in sorted_seeds:
        if remaining <= 0:
            break
        allocation[i] += 1
        remaining -= 1

    return allocation


def plan_viewports(grid: np.ndarray, settlements: List[dict],
                   num_queries: int,
                   map_w: int = 40, map_h: int = 40
                   ) -> List[Tuple[int, int, int, int]]:
    """Plan viewport positions to maximize dynamic cell observation.

    Returns ordered list of (x, y, w, h) viewports to query.
    May repeat high-value viewports for more observation samples.
    """
    if not settlements:
        return _uniform_viewports(map_w, map_h, num_queries)

    dynamic = compute_dynamic_mask(grid, settlements)

    # Find bounding box of ALL dynamic cells
    dy, dx = np.where(dynamic)
    if len(dy) == 0:
        return _uniform_viewports(map_w, map_h, num_queries)

    # Generate candidate viewports centered on settlement clusters
    candidates = _generate_candidate_viewports(settlements, dynamic,
                                                map_w, map_h)

    # Greedy set-cover: pick viewports that maximize new dynamic cells covered
    selected = _greedy_viewport_selection(candidates, dynamic,
                                          map_w, map_h, num_queries)

    return selected


def _generate_candidate_viewports(settlements: List[dict],
                                   dynamic: np.ndarray,
                                   map_w: int, map_h: int,
                                   vp_size: int = 15
                                   ) -> List[Tuple[int, int, int, int]]:
    """Generate candidate viewport positions centered on/near settlements."""
    candidates = set()

    for s in settlements:
        sx, sy = s["x"], s["y"]
        # Center viewport on settlement
        cx = max(0, min(map_w - vp_size, sx - vp_size // 2))
        cy = max(0, min(map_h - vp_size, sy - vp_size // 2))
        candidates.add((cx, cy, vp_size, vp_size))

        # Also try offsets to capture different settlement groupings
        for offset_x in [-7, 0, 7]:
            for offset_y in [-7, 0, 7]:
                nx = max(0, min(map_w - vp_size, sx + offset_x - vp_size // 2))
                ny = max(0, min(map_h - vp_size, sy + offset_y - vp_size // 2))
                candidates.add((nx, ny, vp_size, vp_size))

    # Also add a grid of viewports across the map for coverage
    for y in range(0, map_h, vp_size):
        for x in range(0, map_w, vp_size):
            vx = min(x, map_w - vp_size)
            vy = min(y, map_h - vp_size)
            candidates.add((max(0, vx), max(0, vy), vp_size, vp_size))

    return list(candidates)


def _greedy_viewport_selection(candidates: List[Tuple[int, int, int, int]],
                                dynamic: np.ndarray,
                                map_w: int, map_h: int,
                                num_queries: int
                                ) -> List[Tuple[int, int, int, int]]:
    """Greedy select viewports maximizing dynamic cell coverage.

    After full coverage, repeat best viewports for more observation samples.
    """
    h, w = dynamic.shape
    covered = np.zeros((h, w), dtype=int)  # count of times each cell is observed
    selected = []

    # Score each candidate by dynamic cells it covers
    def score_viewport(vp):
        x, y, vw, vh = vp
        region = dynamic[y:y+vh, x:x+vw]
        observed = covered[y:y+vh, x:x+vw]
        # Prioritize uncovered dynamic cells, but repeated observations still valuable
        new_cells = np.sum(region & (observed == 0))
        total_dynamic = np.sum(region)
        return (new_cells * 10 + total_dynamic, total_dynamic)

    for _ in range(num_queries):
        # Score all candidates
        scored = [(score_viewport(vp), vp) for vp in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)

        best_vp = scored[0][1]
        selected.append(best_vp)

        # Update coverage
        x, y, vw, vh = best_vp
        covered[y:y+vh, x:x+vw] += 1

    return selected


def _uniform_viewports(map_w: int, map_h: int,
                       num_queries: int) -> List[Tuple[int, int, int, int]]:
    """Fallback: tile the map uniformly."""
    vp = 15
    viewports = []
    for y in range(0, map_h, 13):  # slight overlap
        for x in range(0, map_w, 13):
            vx = min(max(0, x), map_w - vp)
            vy = min(max(0, y), map_h - vp)
            viewports.append((vx, vy, vp, vp))

    viewports = list(dict.fromkeys(viewports))  # deduplicate, preserve order

    # Repeat if needed
    result = []
    while len(result) < num_queries and viewports:
        result.extend(viewports)
    return result[:num_queries]
