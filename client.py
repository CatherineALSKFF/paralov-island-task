"""API client for Astar Island challenge."""
import time
import requests
import numpy as np

BASE_URL = "https://api.ainm.no"

# Terrain codes → prediction class mapping
TERRAIN_TO_CLASS = {
    0: 0,   # Empty → class 0
    10: 0,  # Ocean → class 0
    11: 0,  # Plains → class 0
    1: 1,   # Settlement → class 1
    2: 2,   # Port → class 2
    3: 3,   # Ruin → class 3
    4: 4,   # Forest → class 4
    5: 5,   # Mountain → class 5
}

NUM_CLASSES = 6
CLASS_NAMES = ["Empty/Ocean/Plains", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

# Static terrain codes that never change
STATIC_TERRAINS = {10, 5}  # Ocean, Mountain
MOSTLY_STATIC = {4}         # Forest (changes only near ruins)


class AstarClient:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"
        self.rate_limit_delay = 0.22  # 5 req/sec max → 200ms between calls

    def _get(self, path: str):
        time.sleep(self.rate_limit_delay)
        resp = self.session.get(f"{BASE_URL}{path}")
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, data: dict):
        time.sleep(self.rate_limit_delay)
        resp = self.session.post(f"{BASE_URL}{path}", json=data)
        resp.raise_for_status()
        return resp.json()

    def get_rounds(self):
        return self._get("/astar-island/rounds")

    def get_active_round(self):
        rounds = self.get_rounds()
        for r in rounds:
            if r["status"] == "active":
                return r
        return None

    def get_round_detail(self, round_id: str):
        return self._get(f"/astar-island/rounds/{round_id}")

    def get_budget(self):
        return self._get("/astar-island/budget")

    def simulate(self, round_id: str, seed_index: int,
                 viewport_x: int, viewport_y: int,
                 viewport_w: int = 15, viewport_h: int = 15):
        return self._post("/astar-island/simulate", {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_w": viewport_w,
            "viewport_h": viewport_h,
        })

    def submit(self, round_id: str, seed_index: int, prediction: np.ndarray):
        """Submit prediction tensor [H, W, 6]."""
        return self._post("/astar-island/submit", {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction.tolist(),
        })

    def get_my_rounds(self):
        return self._get("/astar-island/my-rounds")

    def get_analysis(self, round_id: str, seed_index: int):
        return self._get(f"/astar-island/analysis/{round_id}/{seed_index}")

    def get_leaderboard(self):
        return self._get("/astar-island/leaderboard")
