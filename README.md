# Static vs. reconfigurable allocation in an accelerator pod

A small Python simulator for CS218 Assignment 1, based on *Resiliency at Scale:
Managing Google's TPUv4 Machine Learning Supercomputer* (Zu et al., NSDI 2024).

It models a 4096-chip pod as a 32 x 32 grid of machines (4 chips each), grouped
into 64 cubes of 16 machines. A large training job needs every one of its
machines healthy at the same time. The simulator compares two ways of finding
those machines:

- **Static** (TPUv2/v3 style): the job needs one physically contiguous rectangle
  of machines with nothing failed or occupied inside it.
- **Reconfigurable** (TPUv4 style): optical circuit switches can wire together
  *any* cubes that are fully healthy and free, wherever they sit.

It does not need a cloud account, GPUs/TPUs or any ML framework.

## Running on Ubuntu

Tested with Python 3.10+ (Ubuntu 22.04 and 24.04 ship a suitable version).

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

git clone git@github.com:AnkushChandra/tpuv4-allocation-simulator.git
cd tpuv4-allocation-simulator

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 run_experiments.py          # all four experiments, about 30-60 seconds
python3 -m unittest discover -s tests -v
```

Useful options:

```bash
python3 run_experiments.py --quick         # fewer trials, a few seconds
python3 run_experiments.py --only e1 e4    # run selected experiments
python3 run_experiments.py --seed 7        # different random seed
```

Results are written to `results/`: one PDF and PNG figure per experiment, CSV
tables with the raw numbers, and `summary.json` with the headline values. The
default seed (218) reproduces the numbers in the report exactly.

## What each experiment does

| ID | Question | Output |
|----|----------|--------|
| e1 | How does the chance of placing a job change with job size? Fits one parameter (machine unavailability) to the paper's Figure 1. | `e1_job_size.*` |
| e2 | How available must each machine be for a job to be placeable 90% of the time? Tests the paper's "99.9% vs 99%" claim (Section 2.1). | `e2_required_availability.*` |
| e3 | What happens when other jobs occupy part of the pod (fragmentation, stranded capacity)? | `e3_occupancy.*` |
| e4 | What happens when machines fail *during* a 30-day run? Compares checkpoint restart with the paper's proposed hot-spare migration (Section 7). | `e4_goodput.*`, `e4_repair_sensitivity.csv` |

## Code layout

```
tpusim/model.py      pod layout, failure sampling, static and reconfigurable placement checks,
                     exact binomial formula for the reconfigurable case
tpusim/runtime.py    event-driven simulation of one long job with failures, repairs,
                     checkpoints and rescheduling
run_experiments.py   the four experiments and all plots
tests/               sanity checks, including simulation vs. exact formula
```

## Main assumptions

- Machines fail independently. Correlated failures (a whole rack, a power domain)
  are not modelled.
- The static pod is a 2D grid. A static job may start at any machine offset,
  which is generous to the static design. Shapes are the most square rectangle
  of whole cubes that fits, in either orientation.
- Other workloads occupy whole cubes.
- Failure rates come from the paper's Figure 12 (0.08% of machines and 0.04% of
  optical switches per day). Repair time, checkpoint interval and restart cost
  are not reported in the paper; the defaults (72 h, 2 h, 0.5 h) are assumptions
  and can be changed in `RunConfig`.
- Network topology, bandwidth, routing and switch-level detail are not modelled.
