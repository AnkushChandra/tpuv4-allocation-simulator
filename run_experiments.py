"""Run all experiments and write figures, CSV tables and a JSON summary to ``results/``.

Usage:
    python3 run_experiments.py            # full run (about a minute)
    python3 run_experiments.py --quick    # fewer trials, for a smoke test
    python3 run_experiments.py --only e1 e3
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tpusim.model import (Pod, largest_job,  # noqa: E402
                          reconfigurable_analytic, reconfigurable_fits, sample_down,
                          sample_occupied, static_fits, static_shapes, valid_static_sizes)
from tpusim.runtime import POLICIES, RunConfig, simulate_job  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
POD = Pod()

# Figure 12 of the paper: 0.08% of machines fail per day. With an assumed
# 3-day mean repair time this gives a steady-state unavailability of 0.24%.
FAIL_PER_DAY = 0.0008
REPAIR_HOURS = 72.0
P_FAILURES_ONLY = FAIL_PER_DAY * REPAIR_HOURS / 24.0

# Figure 12(c): 0.04% of OCSes fail per day; a pod has 48 OCSes. Without
# fault-tolerant routing any OCS outage blocks every multi-cube job.
OCS_FAIL_PER_DAY = 0.0004
OCS_COUNT = 48
P_OCS_DOWN = OCS_FAIL_PER_DAY * REPAIR_HOURS / 24.0

# Figure 1 of the paper, digitised by eye (about +/-2 percentage points).
# Keys are job sizes in cubes (64 chips each) or, for TPUv3, in chips.
FIG1_V4_FT = {50: 0.99, 51: 0.99, 52: 0.97, 53: 0.93, 54: 0.87, 55: 0.78, 56: 0.66, 57: 0.51,
              58: 0.36, 59: 0.22, 60: 0.11, 61: 0.05, 62: 0.02, 63: 0.01}
FIG1_V4_NOFT = {1: 0.94, 10: 0.94, 20: 0.94, 30: 0.94, 40: 0.94, 50: 0.935, 52: 0.90, 54: 0.81,
                56: 0.60, 58: 0.32, 60: 0.10, 62: 0.01}
FIG1_V3_CHIPS = {64: 0.83, 128: 0.75, 256: 0.55, 512: 0.25, 1024: 0.0}
TPUV3_POD = Pod(grid=16, cube_side=4)

COLORS = {"static": "#d62728", "static-packed": "#ff9896", "reconfig": "#1f77b4",
          "hotspare": "#2ca02c", "analytic": "#000000"}
LABELS = {"static": "Static (contiguous)", "static-packed": "Static, packed neighbours",
          "reconfig": "Reconfigurable (any cubes)", "hotspare": "Reconfigurable + hot spare"}


def ci95(p: float, n: int) -> float:
    return 1.96 * np.sqrt(max(p * (1 - p), 1e-12) / n)


def write_csv(name: str, rows: list[dict]) -> None:
    with open(RESULTS / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(RESULTS / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# E1: success probability vs job size
# ---------------------------------------------------------------------------

def fit_p_to_fig1() -> tuple[float, float]:
    """Least-squares fit of machine unavailability to the digitised TPUv4 (FT routing) curve."""
    grid = np.linspace(0.001, 0.02, 1901)
    err = [np.mean([(reconfigurable_analytic(POD, p, c) - v) ** 2 for c, v in FIG1_V4_FT.items()])
           for p in grid]
    i = int(np.argmin(err))
    return float(grid[i]), float(np.sqrt(err[i]))


def _sweep(rng, pod, p, trials):
    static_sizes = set(valid_static_sizes(pod))
    rows = []
    for c in range(1, pod.cubes + 1):
        blocked = sample_down(rng, pod, p, trials)
        r = reconfigurable_fits(pod, blocked, c).mean()
        s = static_fits(pod, blocked, c).mean() if c in static_sizes else float("nan")
        rows.append({"pod_chips": pod.cubes * pod.chips_per_cube, "p_down": p, "job_cubes": c,
                     "job_chips": c * pod.chips_per_cube, "static": s, "static_ci95": ci95(s, trials),
                     "reconfig": r, "reconfig_ci95": ci95(r, trials),
                     "reconfig_analytic": reconfigurable_analytic(pod, p, c)})
    return rows


def e1_job_size(rng, trials, summary):
    p_fit, rmse = fit_p_to_fig1()
    ocs_ok = (1 - P_OCS_DOWN) ** OCS_COUNT
    summary.update({"p_fit": p_fit, "p_fit_rmse": rmse, "p_failures_only": P_FAILURES_ONLY,
                    "p_ocs_down": P_OCS_DOWN, "ocs_all_up": ocs_ok})

    v4_fit = _sweep(rng, POD, p_fit, trials)
    v4_fail = _sweep(rng, POD, P_FAILURES_ONLY, trials)
    v3_fit = _sweep(rng, TPUV3_POD, p_fit, trials)
    write_csv("e1_job_size.csv", v4_fit + v4_fail + v3_fit)

    def chips(rows):
        return [r["job_chips"] for r in rows]

    def static_pts(rows):
        return [r for r in rows if not np.isnan(r["static"])]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    ax = axes[0]
    ax.plot(chips(v4_fit), [r["reconfig"] for r in v4_fit], "-", color=COLORS["reconfig"],
            label="Model: reconfigurable, FT routing")
    ax.plot(chips(v4_fit), [r["reconfig"] * ocs_ok for r in v4_fit], "--", color="#9467bd",
            label="Model: reconfigurable, no FT routing")
    ax.plot(chips(static_pts(v3_fit)), [r["static"] for r in static_pts(v3_fit)], "-",
            color=COLORS["static"], label="Model: static 1024-chip pod")
    ax.plot([c * 64 for c in FIG1_V4_FT], list(FIG1_V4_FT.values()), "o", ms=4, mfc="none",
            color=COLORS["reconfig"], label="Paper: TPUv4 w/ FT routing")
    ax.plot([c * 64 for c in FIG1_V4_NOFT], list(FIG1_V4_NOFT.values()), "^", ms=4, mfc="none",
            color="#9467bd", label="Paper: TPUv4 w/o FT routing")
    ax.plot(list(FIG1_V3_CHIPS), list(FIG1_V3_CHIPS.values()), "x", ms=5,
            color=COLORS["static"], label="Paper: TPUv3 (static)")
    ax.set_title(f"(a) Model vs. paper Fig. 1 (fitted p = {p_fit:.2%})", fontsize=9)
    ax.set_ylabel("P(job can be placed)")

    ax = axes[1]
    for rows, ls, tag in ((v4_fit, "-", f"p={p_fit:.2%}"), (v4_fail, ":", f"p={P_FAILURES_ONLY:.2%}")):
        sp = static_pts(rows)
        ax.plot(chips(sp), [r["static"] for r in sp], ls, marker="o", ms=2.5, color=COLORS["static"],
                label=f"Static, {tag}")
        ax.plot(chips(rows), [r["reconfig"] for r in rows], ls, color=COLORS["reconfig"],
                label=f"Reconfigurable, {tag}")
    ax.set_title("(b) Same 4096-chip pod, same failures", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Job size (chips)")
        ax.set_xlim(0, 4200)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.set_size_inches(7.2, 3.9)
    save(fig, "e1_job_size")

    def at(rows, c):
        return next(r for r in rows if r["job_cubes"] == c)
    summary["e1"] = {
        "v4_fit": {c: {"static": at(v4_fit, c)["static"], "reconfig": at(v4_fit, c)["reconfig"]}
                   for c in (4, 9, 16, 25, 36, 48, 50, 56, 57, 60, 64)},
        "v4_fail": {c: {"static": at(v4_fail, c)["static"], "reconfig": at(v4_fail, c)["reconfig"]}
                    for c in (4, 9, 16, 25, 36, 48, 56, 64)},
        "v3_fit_static": {r["job_chips"]: r["static"] for r in static_pts(v3_fit)},
    }


# ---------------------------------------------------------------------------
# E2: per-machine availability needed for a 90%-available job
# ---------------------------------------------------------------------------

def _max_p(fits_at, lo=0.0, hi=0.2, target=0.9, iters=40):
    for _ in range(iters):
        mid = (lo + hi) / 2
        if fits_at(mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo


def e2_required_availability(rng, trials, summary, target=0.9):
    rows = []
    for c in (4, 16, 36, 48, 56, 64):
        u = rng.random((trials, POD.grid, POD.grid))  # common random numbers => monotone in p
        p_static = _max_p(lambda p: static_fits(POD, u < p, c).mean(), target=target)
        p_reconf = _max_p(lambda p: reconfigurable_analytic(POD, p, c), target=target)
        rows.append({"job_cubes": c, "job_chips": c * POD.chips_per_cube,
                     "job_machines": c * POD.machines_per_cube,
                     "static_required_availability": 1 - p_static,
                     "reconfig_required_availability": 1 - p_reconf,
                     "static_tolerable_unavailability": p_static,
                     "reconfig_tolerable_unavailability": p_reconf})
    write_csv("e2_required_availability.csv", rows)
    summary["e2"] = rows

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    chips = [r["job_chips"] for r in rows]
    ax.semilogy(chips, [r["static_tolerable_unavailability"] for r in rows], "o-",
                color=COLORS["static"], label=LABELS["static"])
    ax.semilogy(chips, [r["reconfig_tolerable_unavailability"] for r in rows], "s-",
                color=COLORS["reconfig"], label=LABELS["reconfig"])
    marks = [0.1, 0.01, 0.001, 0.0001]
    for y in marks:
        ax.axhline(y, color="grey", lw=0.6, ls=":")
    right = ax.secondary_yaxis("right")
    right.set_yticks(marks, labels=["90%", "99%", "99.9%", "99.99%"], fontsize=7)
    right.set_ylabel("Required machine availability", fontsize=8)
    ax.set_xlabel("Job size (chips)")
    ax.set_ylabel("Max machine unavailability\nfor 90% job availability", fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    save(fig, "e2_required_availability")


# ---------------------------------------------------------------------------
# E3: other workloads and fragmentation
# ---------------------------------------------------------------------------

def e3_occupancy(rng, trials, summary, job_cubes=16):
    fractions = np.round(np.arange(0.0, 0.76, 0.05), 2)
    rows = []
    for f in fractions:
        down = sample_down(rng, POD, P_FAILURES_ONLY, trials)
        row = {"occupied_fraction": f}
        for name, layout in (("static", "random"), ("static-packed", "packed"), ("reconfig", "random")):
            blocked = down | sample_occupied(rng, POD, f, trials, layout)
            fits = (reconfigurable_fits(POD, blocked, job_cubes) if name == "reconfig"
                    else static_fits(POD, blocked, job_cubes))
            s_best, r_best = largest_job(POD, blocked)
            best = r_best if name == "reconfig" else s_best
            free_machines = (~blocked).sum(axis=(1, 2))
            usable = best * POD.machines_per_cube
            stranded = 1 - usable / np.maximum(free_machines, 1)
            row[f"{name}_success"] = fits.mean()
            row[f"{name}_stranded"] = stranded.mean()
        rows.append(row)
    write_csv("e3_occupancy.csv", rows)
    summary["e3"] = {"job_cubes": job_cubes, "rows": rows}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for name, marker in (("static", "o"), ("static-packed", "^"), ("reconfig", "s")):
        axes[0].plot(fractions * 100, [r[f"{name}_success"] for r in rows], marker + "-", ms=3,
                     color=COLORS[name], label=LABELS[name])
        axes[1].plot(fractions * 100, [100 * r[f"{name}_stranded"] for r in rows], marker + "-", ms=3,
                     color=COLORS[name], label=LABELS[name])
    capacity_limit = 100 * (1 - job_cubes / POD.cubes)
    for ax in axes:
        ax.axvline(capacity_limit, color="grey", ls=":", lw=0.8)
        ax.set_xlabel("Pod occupied by other jobs (%)")
        ax.grid(alpha=0.3)
    axes[0].set_title(f"(a) Can a {job_cubes * POD.chips_per_cube}-chip job start?", fontsize=9)
    axes[0].set_ylabel("P(job can be placed)")
    axes[0].legend(fontsize=7, loc="lower left")
    axes[1].set_title("(b) Free capacity left after the largest job", fontsize=9)
    axes[1].set_ylabel("Stranded capacity (%)")
    save(fig, "e3_occupancy")


# ---------------------------------------------------------------------------
# E4: failures during a long run (not evaluated in the paper)
# ---------------------------------------------------------------------------

def _run_many(rng, n, **kw):
    res = [simulate_job(POD, RunConfig(**kw), rng) for _ in range(n)]
    g = np.array([r.goodput for r in res])
    return {"goodput": g.mean(), "goodput_ci95": 1.96 * g.std(ddof=1) / np.sqrt(n),
            "interruptions": np.mean([r.interruptions for r in res]),
            "lost_work_h": np.mean([r.lost_work_hours for r in res]),
            "waiting_h": np.mean([r.waiting_hours for r in res]),
            "overhead_h": np.mean([r.overhead_hours for r in res]),
            "wall_days": np.mean([r.wall_hours for r in res]) / 24}


def e4_midrun_failures(rng, runs, summary):
    sizes = [8, 16, 24, 32, 40, 48, 56, 60, 62, 63, 64]
    base = dict(fail_per_machine_day=FAIL_PER_DAY, repair_hours=REPAIR_HOURS)
    rows = []
    for c in sizes:
        for policy in POLICIES:
            if policy == "static" and not static_shapes(POD.cubes_per_side, c):
                continue
            stats = _run_many(rng, runs, job_cubes=c, policy=policy, **base)
            rows.append({"job_cubes": c, "job_chips": c * POD.chips_per_cube,
                         "spare_cubes": POD.cubes - c, "policy": policy, **stats})
    write_csv("e4_goodput.csv", rows)

    repair_times = [12, 24, 48, 72, 120, 168]
    sens = []
    for c in (56, 60, 63):
        for rh in repair_times:
            stats = _run_many(rng, max(runs // 2, 20), job_cubes=c, policy="reconfig",
                              fail_per_machine_day=FAIL_PER_DAY, repair_hours=rh)
            sens.append({"job_cubes": c, "repair_hours": rh, **stats})
    write_csv("e4_repair_sensitivity.csv", sens)
    summary["e4"] = {"rows": rows, "repair_sensitivity": sens}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for policy, marker in (("static", "o"), ("reconfig", "s"), ("hotspare", "^")):
        pts = [r for r in rows if r["policy"] == policy]
        axes[0].errorbar([r["job_chips"] for r in pts], [r["goodput"] for r in pts],
                         yerr=[r["goodput_ci95"] for r in pts], fmt=marker + "-", ms=3, capsize=2,
                         color=COLORS[policy], label=LABELS[policy])
    axes[0].set_title("(a) 30-day job, Fig. 12 failure rate", fontsize=9)
    axes[0].set_xlabel("Job size (chips)")
    axes[0].set_ylabel("Goodput (useful time / wall time)")
    axes[0].legend(fontsize=7, loc="lower left")
    for c, marker in ((56, "o"), (60, "s"), (63, "^")):
        pts = [r for r in sens if r["job_cubes"] == c]
        axes[1].plot([r["repair_hours"] for r in pts], [r["goodput"] for r in pts], marker + "-", ms=3,
                     label=f"{c} cubes ({POD.cubes - c} spare)")
    axes[1].set_title("(b) Reconfigurable: effect of repair time", fontsize=9)
    axes[1].set_xlabel("Mean repair time (hours)")
    axes[1].legend(fontsize=7, loc="lower left")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.02)
    save(fig, "e4_goodput")


EXPERIMENTS = {"e1": e1_job_size, "e2": e2_required_availability,
               "e3": e3_occupancy, "e4": e4_midrun_failures}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer trials for a fast smoke test")
    ap.add_argument("--seed", type=int, default=218)
    ap.add_argument("--only", nargs="+", choices=sorted(EXPERIMENTS), default=sorted(EXPERIMENTS))
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    trials = 500 if args.quick else 5000
    runs = 30 if args.quick else 300
    summary_path = RESULTS / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update({"seed": args.seed, "trials": trials, "runs": runs})

    for name in args.only:
        rng = np.random.default_rng([args.seed, int(name[1:])])
        t0 = time.time()
        fn = EXPERIMENTS[name]
        fn(rng, runs if name == "e4" else trials, summary)
        print(f"{name}: done in {time.time() - t0:.1f}s")

    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"results written to {RESULTS}")


if __name__ == "__main__":
    main()
