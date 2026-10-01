"""Snapshot model of an accelerator pod.

The pod is a square grid of machines (default 32 x 32 = 1024 machines,
4 chips each = 4096 chips, the size of a TPUv4 pod). Machines are grouped
into square cubes (default 4 x 4 = 16 machines = 64 chips, one TPUv4 cube).

A machine is *blocked* if it is failed or occupied by another workload.
Two allocation policies are compared:

* static         -- the job needs one physically contiguous rectangle of
                    machines with nothing blocked inside it (TPUv2/v3 style).
* reconfigurable -- the job needs any ``k`` cubes that contain no blocked
                    machine, wherever they are (TPUv4 OCS style).

Functions that take a ``blocked`` array accept shape ``(trials, grid, grid)``
so that many Monte Carlo trials are evaluated at once.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class Pod:
    grid: int = 32
    cube_side: int = 4
    chips_per_machine: int = 4

    def __post_init__(self) -> None:
        if self.grid % self.cube_side:
            raise ValueError("grid must be a multiple of cube_side")

    @property
    def machines(self) -> int:
        return self.grid * self.grid

    @property
    def cubes_per_side(self) -> int:
        return self.grid // self.cube_side

    @property
    def cubes(self) -> int:
        return self.cubes_per_side ** 2

    @property
    def machines_per_cube(self) -> int:
        return self.cube_side ** 2

    @property
    def chips_per_cube(self) -> int:
        return self.machines_per_cube * self.chips_per_machine


# ---------------------------------------------------------------------------
# Sampling pod state
# ---------------------------------------------------------------------------

def sample_down(rng: np.random.Generator, pod: Pod, p_down: float, trials: int) -> np.ndarray:
    """Each machine is independently unavailable with probability ``p_down``."""
    return rng.random((trials, pod.grid, pod.grid)) < p_down


def expand_cubes(pod: Pod, cube_mask: np.ndarray) -> np.ndarray:
    """Turn a (trials, n, n) per-cube mask into a (trials, grid, grid) per-machine mask."""
    s = pod.cube_side
    return np.repeat(np.repeat(cube_mask, s, axis=1), s, axis=2)


def sample_occupied(rng: np.random.Generator, pod: Pod, fraction: float, trials: int,
                    layout: str = "random") -> np.ndarray:
    """Mark ``fraction`` of the cubes as occupied by other workloads.

    ``random`` scatters the occupied cubes (worst case for contiguity);
    ``packed`` fills cubes in row-major order, which is what an ideal
    defragmenting scheduler would achieve.
    """
    n_occ = int(round(fraction * pod.cubes))
    cube_mask = np.zeros((trials, pod.cubes), dtype=bool)
    if n_occ:
        if layout == "random":
            order = rng.random((trials, pod.cubes)).argsort(axis=1)
            np.put_along_axis(cube_mask, order[:, :n_occ], True, axis=1)
        elif layout == "packed":
            cube_mask[:, :n_occ] = True
        else:
            raise ValueError(f"unknown layout {layout!r}")
    n = pod.cubes_per_side
    return expand_cubes(pod, cube_mask.reshape(trials, n, n))


# ---------------------------------------------------------------------------
# Reconfigurable (any-cube) allocation
# ---------------------------------------------------------------------------

def usable_cubes(pod: Pod, blocked: np.ndarray) -> np.ndarray:
    """(trials, n, n) mask of cubes with no blocked machine."""
    t = blocked.shape[0]
    n, s = pod.cubes_per_side, pod.cube_side
    return ~blocked.reshape(t, n, s, n, s).any(axis=(2, 4))


def reconfigurable_fits(pod: Pod, blocked: np.ndarray, cubes_needed: int) -> np.ndarray:
    return usable_cubes(pod, blocked).sum(axis=(1, 2)) >= cubes_needed


def reconfigurable_find(pod: Pod, blocked: np.ndarray, cubes_needed: int) -> np.ndarray | None:
    """Single-trial version: return a (grid, grid) allocation mask or None."""
    cubes = usable_cubes(pod, blocked[None])[0]
    idx = np.flatnonzero(cubes)
    if idx.size < cubes_needed:
        return None
    chosen = np.zeros(pod.cubes, dtype=bool)
    chosen[idx[:cubes_needed]] = True
    n = pod.cubes_per_side
    return expand_cubes(pod, chosen.reshape(1, n, n))[0]


# ---------------------------------------------------------------------------
# Static (contiguous rectangle) allocation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def static_shapes(cubes_per_side: int, cubes_needed: int) -> tuple[tuple[int, int], ...]:
    """Most-square rectangle of ``cubes_needed`` cubes that fits the pod, in both orientations.

    Shapes are measured in cubes so that the static and reconfigurable
    policies are asked for exactly the same number of machines.
    """
    pairs = [(a, cubes_needed // a) for a in range(1, cubes_per_side + 1)
             if cubes_needed % a == 0 and cubes_needed // a <= cubes_per_side]
    if not pairs:
        return ()
    a, b = min(pairs, key=lambda ab: abs(ab[0] - ab[1]))
    return ((a, b),) if a == b else ((a, b), (b, a))


def valid_static_sizes(pod: Pod) -> list[int]:
    """Job sizes (in cubes) that can be expressed as a rectangle inside the pod."""
    return [c for c in range(1, pod.cubes + 1) if static_shapes(pod.cubes_per_side, c)]


def integral_image(blocked: np.ndarray) -> np.ndarray:
    t, g, _ = blocked.shape
    ii = np.zeros((t, g + 1, g + 1), dtype=np.int32)
    ii[:, 1:, 1:] = blocked.cumsum(axis=1, dtype=np.int32).cumsum(axis=2, dtype=np.int32)
    return ii


def _clear_windows(ii: np.ndarray, h: int, w: int) -> np.ndarray:
    """(trials, g-h+1, g-w+1) mask of h x w windows containing no blocked machine."""
    s = ii[:, h:, w:] - ii[:, :-h, w:] - ii[:, h:, :-w] + ii[:, :-h, :-w]
    return s == 0


def static_fits(pod: Pod, blocked: np.ndarray, cubes_needed: int,
                ii: np.ndarray | None = None) -> np.ndarray:
    """Whether a contiguous rectangle of ``cubes_needed`` cubes' worth of machines is free.

    The rectangle may start at any machine offset (not only cube-aligned),
    which is generous to the static design.
    """
    shapes = static_shapes(pod.cubes_per_side, cubes_needed)
    if not shapes:
        raise ValueError(f"{cubes_needed} cubes cannot form a rectangle in this pod")
    if ii is None:
        ii = integral_image(blocked)
    ok = np.zeros(blocked.shape[0], dtype=bool)
    for a, b in shapes:
        ok |= _clear_windows(ii, a * pod.cube_side, b * pod.cube_side).any(axis=(1, 2))
    return ok


def static_find(pod: Pod, blocked: np.ndarray, cubes_needed: int) -> np.ndarray | None:
    """Single-trial version: return a (grid, grid) allocation mask or None."""
    ii = integral_image(blocked[None])
    for a, b in static_shapes(pod.cubes_per_side, cubes_needed):
        h, w = a * pod.cube_side, b * pod.cube_side
        hits = np.argwhere(_clear_windows(ii, h, w)[0])
        if hits.size:
            i, j = hits[0]
            alloc = np.zeros((pod.grid, pod.grid), dtype=bool)
            alloc[i:i + h, j:j + w] = True
            return alloc
    return None


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def largest_job(pod: Pod, blocked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Largest job (in cubes) each policy could place, per trial."""
    ii = integral_image(blocked)
    static_best = np.zeros(blocked.shape[0], dtype=int)
    for c in valid_static_sizes(pod):
        fits = static_fits(pod, blocked, c, ii)
        static_best = np.where(fits, np.maximum(static_best, c), static_best)
    reconf_best = usable_cubes(pod, blocked).sum(axis=(1, 2))
    return static_best, reconf_best


def reconfigurable_analytic(pod: Pod, p_down: float, cubes_needed: int) -> float:
    """Exact P[at least ``cubes_needed`` of the cubes are fully healthy] (binomial tail)."""
    n = pod.cubes
    q = (1.0 - p_down) ** pod.machines_per_cube
    return sum(math.comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(cubes_needed, n + 1))


def success_rates(rng: np.random.Generator, pod: Pod, p_down: float, cubes_needed: int,
                  trials: int, occupancy: float = 0.0, layout: str = "random",
                  batch: int = 2000) -> tuple[float, float]:
    """Monte Carlo estimate of (static, reconfigurable) placement success."""
    s_hits = r_hits = 0
    done = 0
    while done < trials:
        t = min(batch, trials - done)
        blocked = sample_down(rng, pod, p_down, t)
        if occupancy:
            blocked |= sample_occupied(rng, pod, occupancy, t, layout)
        s_hits += int(static_fits(pod, blocked, cubes_needed).sum())
        r_hits += int(reconfigurable_fits(pod, blocked, cubes_needed).sum())
        done += t
    return s_hits / trials, r_hits / trials
