# Risk-Calibrated Service-Region Routing with Certified Lazy Motion Evaluation

Paper and supplementary material for the conference version of this work.

**Authors:** Rong Huang and Tingyu Liang (corresponding author), Chongqing University of Posts and Telecommunications, Chongqing 400065, China.

**Paper:** [`paper/main.pdf`](paper/main.pdf) · LaTeX source in [`paper/main.tex`](paper/main.tex)

## What this repository contains

| Path | Content |
|---|---|
| `paper/` | Conference paper (4 pages), LaTeX source and the response to reviewers |
| `code/src/psr_planner/` | Planner: risk calibration, grouped dynamic programming, lazy certificate solver, forced-edge screening, MST baseline, motion oracles |
| `code/experiments/` | All experiment scripts, including the 18-instance Hybrid A* suite, the 12-instance real-map suite, the MST baseline, repeated timing and the expansion-budget check |
| `code/results/` | Per-instance result files used in the paper |
| `code/data/maps/` | Occupancy grids from the Moving AI benchmark |
| `figures/` | The two data figures and the script that generates them |
| `supplementary/oracle_budget.json` | Edge-by-edge comparison of the 50,000 and 200,000 expansion caps |
| `REPRODUCE.md` | Commands to rerun every experiment and figure |

## Main results

- Zero-gap lazy evaluation matches the full-matrix objective on 135 proxy, 24 state-lattice, 18 Hybrid A* and 12 real-map instances.
- Exact-query reductions are 57.64% (proxy), 59.16% (lattice), 36.43% (Hybrid A*) and 48.53% (real maps).
- Safe screening removes another 13.26% and 23.21% of queries on the Hybrid A* and real-map suites.
- The one-shot lower-bound route is infeasible in 9 of 18 Hybrid A* instances and 8 of 12 real-map instances.
- In the expansion-budget check, none of the 463 failed queries at the 50,000 cap reached the cap; a 200,000 cap changed no cost, objective or route.

## Quick start

```bash
cd code
python -m unittest discover -s tests
```

Expected: 11 tests pass. Python 3.12 with NumPy is enough for the planner; the figures need Matplotlib.

## Data and attribution

The real-map suite uses street-network occupancy grids from the Moving AI benchmark (N. R. Sturtevant, "Benchmarks for grid-based pathfinding," IEEE TCIAIG, 2012), distributed under the Open Data Commons Attribution License. The warehouse maps in `code/data/maps/` were downloaded for a scale check but are not used in the paper.

## Contact

Rong Huang, 2028701576@qq.com · Tingyu Liang, 2650099818@qq.com

Code and data are provided for research use and review. Please contact the authors for other uses.
