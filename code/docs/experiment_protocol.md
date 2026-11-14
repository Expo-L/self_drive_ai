# Computational experiment protocol v0.1

Frozen on 2026-10-09, before running the scaling experiments described below.

## 1. Scope and research question

These experiments are algorithmic pre-experiments, not yet evidence for a physical-robot
performance claim. They test whether certificate-driven lazy evaluation can reduce expensive
directed motion-planning calls while preserving the optimum of a grouped route problem over
service-pose candidates.

Primary question: at zero requested tolerance, does the lazy solver return the same objective as
the full exact-cost matrix while evaluating materially fewer exact directed edges?

Secondary questions: how do candidate count, target count, and obstacle density affect query
savings; how much search effort is avoided; and how poor can a one-shot lower-bound route be when
it is not refined?

## 2. Methods fixed before execution

- `Full`: compute every start-to-candidate and inter-group directed exact edge, then run exact
  subset dynamic programming. This is the reference optimum for each finite instance.
- `One-shot`: optimize once with the admissible lower-bound matrix, then exactly evaluate only the
  chosen route. This minimizes queries but has no optimality guarantee.
- `Lazy-0`: iteratively evaluate edges on optimistic routes until the directly computed relative
  certificate `(UB - LB) / UB` is zero within numerical tolerance.
- `Lazy-5%` (proxy suite only): the same algorithm stopped when the certified relative gap is at
  most 0.05.

The proxy oracle adds non-negative turning, direction, and obstacle-crossing costs to Euclidean
distance. The lattice oracle runs discrete SE(2) A* with rotations and forward/reverse motion.
Euclidean distance is therefore an admissible lower bound for both suites.

## 3. Factor grid

### Broad proxy suite

- Targets: 4, 5, 6.
- Valid service candidates per target: 2, 3, 4.
- Obstacle density: low, medium, high.
- Random seeds: 0--4.
- Planned total: 135 instances.

### Lattice confirmation suite

- Targets: 4, 5.
- Valid service candidates per target: 2, 3.
- Obstacle density: low, medium, high.
- Random seeds: 0--1.
- Planned total: 24 instances.

Maps and candidate locations are generated deterministically from the declared density and seed.
Only candidates in free grid cells are retained. An instance that cannot supply the requested
number of valid candidates is reported as a generation failure rather than silently replaced.

## 4. Outcomes

The primary outcome is exact-query reduction of `Lazy-0` relative to `Full`. Secondary outcomes
are expanded A* states, wall-clock time, failed exact queries, certificate width, `Lazy-5%`
objective loss, and one-shot objective loss. Reports contain per-instance records; aggregation is
by oracle, target count, candidate count, and obstacle density.

## 5. Correctness gates

All of the following must hold before an efficiency result is treated as valid:

1. `Lazy-0` and `Full` objective values agree to absolute tolerance `1e-8` on every finite
   instance.
2. The reference optimum lies in every reported lazy interval `[LB, UB]` to numerical tolerance.
3. Every `Lazy-5%` result has a reported relative gap no greater than `0.05 + 1e-8`.
4. Automated unit tests for routing bounds and risk calibration pass.

The predeclared practical signal is a mean `Lazy-0` exact-query reduction of at least 30%, with a
positive reduction on at least 80% of valid instances. Failure to meet this threshold is reported,
not filtered.

## 6. Statistical summaries and exclusions

For each cell and overall, report count, mean, median, minimum, maximum, and interquartile range for
query reduction; report means for objective loss, search expansion reduction, and runtime. This
initial internal sweep is descriptive. A submission-quality study must add more maps and seeds,
bootstrap confidence intervals, paired nonparametric tests, and effect sizes.

Infeasible reference routes, timeouts, and generator failures are counted and retained with their
status. Instances are never removed because savings are small or a baseline performs better.

## 7. Risk calibration protocol

The service-pose threshold is success probability at least 0.85. Candidate-wise one-sided 95%
Clopper--Pearson bounds are reported for continuity with the pilot. A simultaneous family-wise
variant applies Bonferroni allocation within each generated candidate family: for `K` candidates,
each lower bound uses confidence `1 - 0.05/K`. Independent holdout simulation estimates false
acceptance and conservatism. The family-wise method controls the probability that any retained
candidate's bound fails in the family under the model assumptions; it does not prove real-world
collision or task-success probability.

## 8. Claim contract

Allowed after passing the gates: the implementation reduced exact oracle calls on the declared
synthetic suites while matching, or remaining within a declared certificate of, the finite-instance
reference optimum.

Forbidden without further evidence: superiority to IST*, state-of-the-art performance, real-time
operation, real-robot safety, generalization to arbitrary maps, or a formal guarantee under model
mismatch.
