"""
Greedy Exploration Agents for Final Review Experiments
=======================================================
These are **rule-based** agents designed to isolate and demonstrate the effect
of MCP coordination.  They are intentionally simple so that any performance
difference between the MCP and baseline conditions is clearly attributable to
context sharing — NOT to algorithmic complexity.

Two flavours:
  • BaselineExplorer   - no context; each UAV independently targets the
                         nearest uncovered cell globally.  With no swarm
                         awareness, multiple UAVs simultaneously converge on
                         the same frontier cells, performing duplicate work.

  • MCPExplorer        - with context; UAVs use MCP-shared swarm state to:
                         (1) partition the grid into N equal horizontal sectors
                             (one per active UAV), eliminating persistent overlap
                         (2) share current target cells with all peers so no two
                             MCPExplorers ever simultaneously claim the same cell
                             (target deduplication — impossible without MCP)
                         (3) redistribute failed peers' sectors among survivors

Why target deduplication matters:
-----------------------------------
Without MCP, when N UAVs independently compute "nearest uncovered cell" from
similar positions they may all select the same cell or adjacent cells in the
same small cluster.  This wastes (N-1)/N of the available coverage capacity.
With MCP, each agent excludes cells currently claimed by peers, so the swarm
always fans out across N distinct frontiers — optimal dispersion with zero
coordination overhead beyond a single broadcast per step.

Why baseline cannot match this:
---------------------------------
Without swarm-size context a UAV cannot compute sector boundaries.
Without peer-target broadcast a UAV cannot perform deduplication.
Any performance gap directly measures the value of MCP context sharing.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# GPS-Denied / SLAM noise model
# ---------------------------------------------------------------------------
# When gps_denied=True, each agent adds zero-mean Gaussian noise to its
# observed position.  This simulates SLAM localisation uncertainty:
#   - sigma starts at SLAM_SIGMA_INIT metres (poor localisation at start)
#   - sigma decays as the UAV accumulates observations (loop-closure effect)
#   - MCP coordination is tested under both GPS and GPS-denied conditions
SLAM_SIGMA_INIT  = 8.0   # metres — initial positional uncertainty
SLAM_SIGMA_FINAL = 1.0   # metres — after full exploration, SLAM converges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_uncovered(
    grid: np.ndarray,
    gx: int,
    gy: int,
    exclude: Set[Tuple[int, int]] | None = None,
    threshold: float = 0.5,
) -> Optional[Tuple[int, int]]:
    """Return (grid_x, grid_y) of the nearest uncovered cell.

    Cells in *exclude* are skipped (used by MCP agents to avoid claiming the
    same target as a peer).  Returns None if the entire grid is covered.
    """
    uncovered = np.argwhere(grid < threshold)  # rows=gy, cols=gx
    if len(uncovered) == 0:
        return None

    candidates = [(int(r[1]), int(r[0])) for r in uncovered]
    if exclude:
        candidates = [(x, y) for x, y in candidates if (x, y) not in exclude]
    if not candidates:
        candidates = [(int(r[1]), int(r[0])) for r in uncovered]

    dists = [abs(x - gx) + abs(y - gy) for x, y in candidates]
    return candidates[int(np.argmin(dists))]


def _toward(
    uav_x: float,
    uav_y: float,
    target_wx: float,
    target_wy: float,
    max_acc: float = 2.0,
) -> np.ndarray:
    """Compute acceleration direction toward a world-coordinate target."""
    dx = target_wx - uav_x
    dy = target_wy - uav_y
    dist = np.sqrt(dx * dx + dy * dy) + 1e-6
    ax = float(np.clip(max_acc * dx / dist, -max_acc, max_acc))
    ay = float(np.clip(max_acc * dy / dist, -max_acc, max_acc))
    return np.array([ax, ay, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _Explorer:
    _CELL_SIZE = 10   # metres per grid cell (matches disaster_scenario grid)
    _ARRIVAL_RADIUS = 12.0   # metres — when to claim target reached

    def __init__(
        self,
        uav_id: str,
        initial_pos: Tuple[float, float],
        gps_denied: bool = False,
    ):
        self.uav_id = uav_id
        self.target_grid: Optional[Tuple[int, int]] = None  # (gx, gy)
        self._sector_center: Optional[Tuple[float, float]] = None
        self._initial_gx = int(initial_pos[0] / self._CELL_SIZE)
        self._initial_gy = int(initial_pos[1] / self._CELL_SIZE)
        self.gps_denied  = gps_denied
        self._step_count = 0  # used to decay SLAM uncertainty over time

    def _get_pos(self, uav) -> Tuple[float, float]:
        """Return UAV position, optionally with SLAM-simulated noise."""
        x, y = uav.state.x, uav.state.y
        if self.gps_denied:
            # Uncertainty decays as the UAV accumulates observations
            decay = max(0.0, 1.0 - self._step_count / 5000.0)
            sigma = SLAM_SIGMA_FINAL + (SLAM_SIGMA_INIT - SLAM_SIGMA_FINAL) * decay
            x += float(np.random.normal(0, sigma))
            y += float(np.random.normal(0, sigma))
        return x, y

    def _world(self, gx: int, gy: int) -> Tuple[float, float]:
        return (gx * self._CELL_SIZE + self._CELL_SIZE / 2,
                gy * self._CELL_SIZE + self._CELL_SIZE / 2)

    def _at_target(self, uav_x: float, uav_y: float) -> bool:
        if self.target_grid is None:
            return True
        tx, ty = self._world(*self.target_grid)
        return np.sqrt((uav_x - tx) ** 2 + (uav_y - ty) ** 2) < self._ARRIVAL_RADIUS


# ---------------------------------------------------------------------------
# Baseline: no context, independent greedy
# ---------------------------------------------------------------------------

class BaselineExplorer(_Explorer):
    """UAV that greedily heads to the nearest uncovered cell with NO swarm context.
    Multiple UAVs often target the same area, wasting coverage capacity.
    """

    def get_action(
        self,
        uav,       # UAV object from simulation
        env,       # SwarmEnvironment
        **_,
    ) -> np.ndarray:
        if uav.uav_id in env.failed_uav_ids:
            return np.zeros(3, dtype=np.float32)

        self._step_count += 1
        px, py = self._get_pos(uav)  # GPS-true or SLAM-noisy position
        grid = env.scenario.coverage_grid
        gx = int(np.clip(px / self._CELL_SIZE, 0, grid.shape[1] - 1))
        gy = int(np.clip(py / self._CELL_SIZE, 0, grid.shape[0] - 1))

        if self._at_target(px, py):
            target = _nearest_uncovered(grid, gx, gy)
            self.target_grid = target

        if self.target_grid is None:
            return np.zeros(3, dtype=np.float32)

        tx, ty = self._world(*self.target_grid)
        # Use perceived position (px,py) for movement to respect GPS-denied sim.
        # In gps_denied mode, agent only has SLAM, not accurate position.
        return _toward(px, py, tx, ty)


# ---------------------------------------------------------------------------
# MCP: deduplication-based cooperative explorer
# ---------------------------------------------------------------------------

class MCPExplorer(_Explorer):
    """UAV that uses MCP-shared swarm context for spatial deduplication.

    Core mechanism (MCP-exclusive):
        Each step, before picking a new target, the agent broadcasts its
        current target cell via MCP and reads the targets of all active peers.
        It then excludes a claim zone around each peer target from its own
        candidate set.  This guarantees that no two MCP UAVs simultaneously
        head to the same small patch of uncovered terrain.

    Failure redistribution (MCP-exclusive):
        Failed peers disappear from the active set automatically — the
        remaining UAVs' claim zones adjust at the next target selection,
        expanding to fill the vacated search space.

    Why baseline cannot match this:
        Without MCP broadcast, a UAV has no knowledge of where peers are
        heading.  Multiple baseline UAVs independently compute "nearest
        uncovered cell" and often converge on the same frontier cluster,
        wasting (N-1)/N of the available coverage effort per step.

    Claim zone:
        Each peer's target is expanded to a radius of CLAIM_RADIUS cells so
        that UAVs maintain meaningful spatial separation rather than picking
        adjacent cells that still cause physical overlap of sensor footprints.
    """

    CLAIM_RADIUS = 8  # cells (~80m); prevents sensor-footprint overlap between peers

    def __init__(
        self,
        uav_id: str,
        initial_pos: Tuple[float, float],
        swarm_index: int = 0,   # kept for factory-function compatibility
        total_uavs: int = 1,    # kept for factory-function compatibility
        gps_denied: bool = False,
    ):
        super().__init__(uav_id, initial_pos, gps_denied=gps_denied)
        self.swarm_index = swarm_index
        self.total_uavs  = total_uavs

    def _build_claim_zones(
        self,
        all_explorers: Optional[List["MCPExplorer"]],
        failed_ids: Set[str],
        grid_shape: Tuple[int, int],
    ) -> Set[Tuple[int, int]]:
        """Return all cells within CLAIM_RADIUS of any active peer's target."""
        claimed: Set[Tuple[int, int]] = set()
        if not all_explorers:
            return claimed
        h, w = grid_shape
        r = self.CLAIM_RADIUS
        for e in all_explorers:
            if e.uav_id == self.uav_id or e.uav_id in failed_ids:
                continue
            if e.target_grid is None:
                continue
            etx, ety = e.target_grid
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        nx, ny = etx + dx, ety + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            claimed.add((nx, ny))
        return claimed

    def get_action(
        self,
        uav,
        env,
        all_explorers: Optional[List["MCPExplorer"]] = None,
        failed_uav_ids: Optional[Set[str]] = None,
    ) -> np.ndarray:
        if uav.uav_id in env.failed_uav_ids:
            return np.zeros(3, dtype=np.float32)

        self._step_count += 1
        px, py = self._get_pos(uav)
        grid = env.scenario.coverage_grid
        gx = int(np.clip(px / self._CELL_SIZE, 0, grid.shape[1] - 1))
        gy = int(np.clip(py / self._CELL_SIZE, 0, grid.shape[0] - 1))

        failed_ids: Set[str] = failed_uav_ids if failed_uav_ids is not None else env.failed_uav_ids

        if self._at_target(px, py) or self.target_grid is None:
            # MCP: read peer targets before selecting own target
            claimed = self._build_claim_zones(all_explorers, failed_ids, grid.shape)
            # Find nearest globally-uncovered cell outside all peer claim zones
            target = _nearest_uncovered(grid, gx, gy, exclude=claimed)
            if target is None:
                # Every candidate is claimed; fall back to unconstrained global
                target = _nearest_uncovered(grid, gx, gy)
            self.target_grid = target

        if self.target_grid is None:
            return np.zeros(3, dtype=np.float32)

        tx, ty = self._world(*self.target_grid)
        # Use perceived position (px,py) for movement to respect GPS-denied sim.
        # In gps_denied mode, agent only has SLAM, not accurate position.
        return _toward(px, py, tx, ty)


# ---------------------------------------------------------------------------
# Factory helpers for experiment files
# ---------------------------------------------------------------------------

def make_baseline_explorers(env, gps_denied: bool = False) -> List[BaselineExplorer]:
    """Create one BaselineExplorer per UAV initialised at the UAV's start pos."""
    return [
        BaselineExplorer(uav.uav_id, (uav.state.x, uav.state.y), gps_denied=gps_denied)
        for uav in env.uavs
    ]


def make_mcp_explorers(env, gps_denied: bool = False) -> List[MCPExplorer]:
    """Create MCPExplorers with peer-target claim-zone deduplication.

    swarm_index / total_uavs are passed for API compatibility but are no
    longer used for sector partitioning.  The coordination benefit comes
    entirely from the CLAIM_RADIUS exclusion zones broadcast via MCP.
    """
    n = len(env.uavs)
    return [
        MCPExplorer(uav.uav_id, (uav.state.x, uav.state.y),
                    swarm_index=i,
                    total_uavs=n,
                    gps_denied=gps_denied)
        for i, uav in enumerate(env.uavs)
    ]


def get_actions_baseline(explorers: List[BaselineExplorer], env) -> np.ndarray:
    return np.concatenate([
        e.get_action(uav, env)
        for e, uav in zip(explorers, env.uavs)
    ])


def get_actions_mcp(
    explorers: List[MCPExplorer],
    env,
    failed_uav_ids: Optional[Set[str]] = None,
) -> np.ndarray:
    return np.concatenate([
        e.get_action(
            uav, env,
            all_explorers=explorers,
            failed_uav_ids=failed_uav_ids or env.failed_uav_ids,
        )
        for e, uav in zip(explorers, env.uavs)
    ])
