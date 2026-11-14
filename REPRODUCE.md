# Reproducing the experiments

All commands assume the repository root unless stated otherwise.

## Environment

- Python 3.12
- NumPy (planner and experiments)
- Matplotlib (figures only)
- No GPU, no external solver

```bash
cd code
export PYTHONPATH=src            # PowerShell: $env:PYTHONPATH='src'
python -m unittest discover -s tests
```

## 1. Hybrid A* suite (18 instances)

```bash
cd code
python experiments/hybrid_suite.py --smoke
python experiments/hybrid_suite.py --families aisles  --targets 4 --seeds 0,1,2 --workers 1 --output results/parts/aisles_4.json
python experiments/hybrid_suite.py --families aisles  --targets 5 --seeds 0,1,2 --workers 1 --output results/parts/aisles_5.json
python experiments/hybrid_suite.py --families rooms   --targets 4 --seeds 0,1,2 --workers 1 --output results/parts/rooms_4.json
python experiments/hybrid_suite.py --families rooms   --targets 5 --seeds 0,1,2 --workers 1 --output results/parts/rooms_5.json
python experiments/hybrid_suite.py --families pillars --targets 4 --seeds 0,1,2 --workers 1 --output results/parts/pillars_4.json
python experiments/hybrid_suite.py --families pillars --targets 5 --seeds 0,1,2 --workers 1 --output results/parts/pillars_5.json
python experiments/merge_hybrid_suite.py
```

Six shards in parallel take about 8 to 10 minutes on a 20-core machine. Do not use a large ProcessPoolExecutor here; independent shard processes were stable, a single eight-worker pool was not.

## 2. Real-map suite (12 instances)

```bash
python experiments/real_suite.py --smoke
python experiments/real_suite.py --maps Denver_0_256 --cells 30 --seeds 0,1 --output results/real_parts/denver_30.json
# repeat for denver_40, boston_30, boston_40, paris_30, paris_40
python experiments/merge_real_suite.py
```

## 3. MST baseline

```bash
python experiments/mst_baseline.py --smoke --families pillars
python experiments/mst_baseline.py --families aisles  --output results/mst_parts/aisles.json
python experiments/mst_baseline.py --families rooms   --output results/mst_parts/rooms.json
python experiments/mst_baseline.py --families pillars --output results/mst_parts/pillars.json
python experiments/merge_mst_baseline.py
```

## 4. Repeated timing

Run on an idle machine.

```bash
python experiments/timing_repeat.py --repeats 3 --output results/timing_repeat.json
python experiments/timing_suite.py  --repeats 2 --output results/timing_suite.json
```

## 5. Expansion-budget check

```bash
python experiments/oracle_budget.py --workers 6 --output results/oracle_budget.json
```

The script evaluates every directed edge at 50,000 expansions, records the per-edge expansion count, and re-evaluates only the edges that reached the cap at 200,000 expansions.

## 6. Figures

```bash
export MPLCONFIGDIR=../.mplconfig
python figures/make_conf_figs.py
```

The figure script is also copied under `figures/` at the repository root. It writes PDF and PNG versions of both data figures.

## Result files used in the paper

- `code/results/scaling_proxy.json`, `scaling_lattice.json`
- `code/results/risk_sweep.json`
- `code/results/hybrid_suite.json`
- `code/results/real_suite.json`
- `code/results/mst_baseline.json`
- `code/results/timing_repeat.json`, `timing_suite.json`
- `code/results/oracle_budget.json`
