"""Event-driven simulation of one long training job while machines fail and get repaired.

Every machine in the pod fails as a Poisson process. A failed machine is
repaired after an exponentially distributed delay. If a failure hits a
machine the job is using, the whole job stops (gang scheduling) and must be
placed again under its allocation policy:

* ``static``   -- find a new contiguous healthy rectangle, restart from the
                  last checkpoint.
* ``reconfig`` -- take any healthy cubes, restart from the last checkpoint.
* ``hotspare`` -- take any healthy cubes and migrate live state to the
                  spare without losing work (the paper's Section 7 proposal).
                  Falls back to checkpoint restart if no spare is available.

While no placement exists the job waits for repairs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Pod, reconfigurable_find, static_find

POLICIES = ("static", "reconfig", "hotspare")


@dataclass(frozen=True)
class RunConfig:
    job_cubes: int
    policy: str = "reconfig"
    work_hours: float = 720.0
    fail_per_machine_day: float = 0.0008
    repair_hours: float = 72.0
    checkpoint_hours: float = 2.0
    restart_hours: float = 0.5
    migrate_hours: float = 0.1


@dataclass
class RunResult:
    useful_hours: float
    wall_hours: float
    interruptions: int
    lost_work_hours: float
    waiting_hours: float
    overhead_hours: float

    @property
    def goodput(self) -> float:
        return self.useful_hours / self.wall_hours


def simulate_job(pod: Pod, rc: RunConfig, rng: np.random.Generator) -> RunResult:
    if rc.policy not in POLICIES:
        raise ValueError(f"unknown policy {rc.policy!r}")
    find = static_find if rc.policy == "static" else reconfigurable_find

    pod_fail_rate = pod.machines * rc.fail_per_machine_day / 24.0
    down_until = np.zeros((pod.grid, pod.grid))

    t = 0.0
    progress = 0.0
    alloc = None
    ready_at = 0.0
    interruptions = 0
    lost = waiting = overhead = 0.0

    def fail_random_machine(now: float) -> tuple[int, int] | None:
        i, j = rng.integers(pod.grid, size=2)
        if down_until[i, j] > now:
            return None
        down_until[i, j] = now + rng.exponential(rc.repair_hours)
        return i, j

    while progress < rc.work_hours:
        if alloc is None:
            alloc = find(pod, down_until > t, rc.job_cubes)
            if alloc is None:
                pending = down_until[down_until > t]
                if pending.size == 0:
                    raise ValueError(f"a {rc.job_cubes}-cube job cannot be placed even on a healthy pod")
                t_repair = pending.min()
                t_fail = t + rng.exponential(1.0 / pod_fail_rate)
                t_next = min(t_repair, t_fail)
                waiting += t_next - t
                t = t_next
                if t_fail < t_repair:
                    fail_random_machine(t)
                continue
            cost = rc.restart_hours if t > 0 else 0.0
            ready_at = t + cost
            overhead += cost

        t_fail = t + rng.exponential(1.0 / pod_fail_rate)
        start = max(t, ready_at)
        t_finish = start + (rc.work_hours - progress)
        if t_finish <= t_fail:
            t = t_finish
            progress = rc.work_hours
            break

        progress += max(0.0, t_fail - start)
        t = t_fail
        hit = fail_random_machine(t)
        if hit is None or not alloc[hit]:
            continue

        interruptions += 1
        if rc.policy == "hotspare":
            replacement = find(pod, down_until > t, rc.job_cubes)
            if replacement is not None:
                alloc = replacement
                ready_at = t + rc.migrate_hours
                overhead += rc.migrate_hours
                continue
        kept = np.floor(progress / rc.checkpoint_hours) * rc.checkpoint_hours
        lost += progress - kept
        progress = kept
        alloc = None

    return RunResult(useful_hours=rc.work_hours, wall_hours=t, interruptions=interruptions,
                     lost_work_hours=lost, waiting_hours=waiting, overhead_hours=overhead)
