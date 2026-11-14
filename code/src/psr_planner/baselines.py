"""IST*-style baseline for grouped directed routing.

The original IST* uses informed sampling and minimum-spanning-tree pruning
for fixed terminals in an undirected metric space.  This module adapts the
pruning idea to the grouped, directed problem:

* for every candidate pose v and every subset T of the remaining groups, it
  precomputes the MST cost of the contracted graph on {v} union T, where the
  weight between two groups is the cheapest lower bound over their poses;
* a route that contains edge e must pay the mixed cost of a prefix, the edge
  itself, and a continuation from the head of e through the remaining groups.
  The continuation is bounded below by that MST value;
* an unevaluated edge is pruned when this conditional bound reaches the
  incumbent, exactly as in the forced-edge rule.

Because the MST bound replaces the exact grouped dynamic program used by the
screened solver, the resulting bound is never tighter, so this baseline is
expected to prune fewer edges.  It is included as an adapted reference, not
as a reimplementation of IST*.
"""

from __future__ import annotations

import math

from .models import Pose, RouteSolution
from .routing import (
    _all_directed_edges,
    _route_edges,
    _validate_groups,
    solve_grouped_route,
)

Groups = dict[str, tuple[Pose, ...]]


def _group_minima(
    candidates: tuple[tuple[Pose, ...], ...], lower
) -> list[list[float]]:
    count = len(candidates)
    table = [[math.inf] * count for _ in range(count)]
    for first in range(count):
        for second in range(count):
            if first == second:
                continue
            table[first][second] = min(
                lower(pose_a, pose_b)
                for pose_a in candidates[first]
                for pose_b in candidates[second]
            )
    return table


def _mst_with_pose(
    pose: Pose,
    group_mask: int,
    candidates: tuple[tuple[Pose, ...], ...],
    minima: list[list[float]],
    lower,
) -> float:
    """MST of {pose} union the groups selected by group_mask (Prim)."""

    if group_mask == 0:
        return 0.0
    best: dict[int, float] = {}
    for group in range(len(candidates)):
        if group_mask & (1 << group):
            best[group] = min(lower(pose, candidate) for candidate in candidates[group])
    total = 0.0
    remaining = set(best)
    while remaining:
        group = min(remaining, key=best.__getitem__)
        total += best[group]
        remaining.remove(group)
        for other in remaining:
            if minima[group][other] < best[other]:
                best[other] = minima[group][other]
    return total


def mst_conditional_bounds(groups: Groups, start: Pose, edge_cost) -> dict[tuple[Pose, Pose], float]:
    """Lower bound on every route containing each directed edge."""

    names = _validate_groups(groups)
    if not names:
        return {}
    candidates = tuple(tuple(groups[name]) for name in names)
    group_count = len(names)
    full_mask = (1 << group_count) - 1
    minima = _group_minima(candidates, edge_cost)

    # mst_value[(group, index, remaining_mask)] = MST({pose} union remaining groups)
    mst_value: dict[tuple[int, int, int], float] = {}
    for group, poses in enumerate(candidates):
        for index, pose in enumerate(poses):
            for mask in range(full_mask + 1):
                if mask & (1 << group):
                    continue
                mst_value[(group, index, mask)] = _mst_with_pose(
                    pose, mask, candidates, minima, edge_cost
                )

    # forward mixed-cost DP, identical to the screened solver
    forward: dict[tuple[int, int, int], float] = {}
    for group, poses in enumerate(candidates):
        mask = 1 << group
        for index, pose in enumerate(poses):
            forward[(mask, group, index)] = edge_cost(start, pose)
    for mask in range(1, full_mask + 1):
        for last_group, poses in enumerate(candidates):
            if not mask & (1 << last_group):
                continue
            for last_index, last_pose in enumerate(poses):
                current = forward.get((mask, last_group, last_index))
                if current is None:
                    continue
                for next_group, next_poses in enumerate(candidates):
                    if mask & (1 << next_group):
                        continue
                    next_mask = mask | (1 << next_group)
                    for next_index, next_pose in enumerate(next_poses):
                        proposed = current + edge_cost(last_pose, next_pose)
                        key = (next_mask, next_group, next_index)
                        if proposed < forward.get(key, math.inf):
                            forward[key] = proposed

    bounds: dict[tuple[Pose, Pose], float] = {}
    for group, poses in enumerate(candidates):
        remaining = full_mask ^ (1 << group)
        for index, pose in enumerate(poses):
            direct = edge_cost(start, pose)
            bounds[(start, pose)] = direct + mst_value[(group, index, remaining)]

    for source_group, source_poses in enumerate(candidates):
        source_bit = 1 << source_group
        for destination_group, destination_poses in enumerate(candidates):
            if source_group == destination_group:
                continue
            destination_bit = 1 << destination_group
            for source_index, source_pose in enumerate(source_poses):
                for destination_index, destination_pose in enumerate(destination_poses):
                    best = math.inf
                    for prefix_mask in range(1, full_mask + 1):
                        if not prefix_mask & source_bit or prefix_mask & destination_bit:
                            continue
                        prefix = forward.get(
                            (prefix_mask, source_group, source_index), math.inf
                        )
                        remaining = full_mask & ~(prefix_mask | destination_bit)
                        tail = mst_value[(destination_group, destination_index, remaining)]
                        candidate = prefix + edge_cost(source_pose, destination_pose) + tail
                        if candidate < best:
                            best = candidate
                    bounds[(source_pose, destination_pose)] = best
    return bounds


def mst_screened_lazy_solve(
    groups: Groups,
    start: Pose,
    oracle,
    *,
    relative_tolerance: float = 0.0,
) -> RouteSolution:
    """Lazy evaluation with MST-based safe pruning (IST*-style baseline)."""

    if not 0.0 <= relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie in [0, 1)")
    _validate_groups(groups)
    all_edges = _all_directed_edges(groups, start)
    total_edges = len(all_edges)
    exact_cache: dict[tuple[Pose, Pose], float] = {}
    pruned: set[tuple[Pose, Pose]] = set()

    def mixed_cost(source: Pose, destination: Pose) -> float:
        edge = (source, destination)
        if edge in exact_cache:
            return exact_cache[edge]
        if edge in pruned:
            return math.inf
        return oracle.lower(source, destination)

    def evaluate_edge(edge: tuple[Pose, Pose]) -> float:
        if edge not in exact_cache:
            exact_cache[edge] = oracle.exact(*edge)
        return exact_cache[edge]

    def evaluate_route(route) -> float:
        return sum(evaluate_edge(edge) for edge in _route_edges(start, route))

    optimistic_seed = solve_grouped_route(groups, start, mixed_cost)
    incumbent_route = optimistic_seed
    upper_bound = evaluate_route(optimistic_seed)
    lower_bound = 0.0

    for _ in range(2 * total_edges + 2):
        conditional = mst_conditional_bounds(groups, start, mixed_cost)
        for edge in all_edges:
            if edge in exact_cache or edge in pruned:
                continue
            if conditional[edge] >= upper_bound - 1e-10:
                pruned.add(edge)

        relaxed_route = solve_grouped_route(groups, start, mixed_cost)
        lower_bound = relaxed_route.cost
        if math.isfinite(upper_bound) and (
            upper_bound - lower_bound <= relative_tolerance * upper_bound + 1e-10
        ):
            break

        relaxed_edges = _route_edges(start, relaxed_route)
        unknown = [edge for edge in relaxed_edges if edge not in exact_cache]
        if not unknown:
            exact_cost = sum(exact_cache[edge] for edge in relaxed_edges)
            if exact_cost < upper_bound:
                incumbent_route = relaxed_route
                upper_bound = exact_cost
            continue

        selected = max(unknown, key=lambda edge: oracle.lower(*edge))
        evaluate_edge(selected)
        if all(edge in exact_cache for edge in relaxed_edges):
            exact_cost = sum(exact_cache[edge] for edge in relaxed_edges)
            if exact_cost < upper_bound:
                incumbent_route = relaxed_route
                upper_bound = exact_cost
    else:
        raise RuntimeError("MST baseline exhausted the iteration guard")

    final_relaxed = solve_grouped_route(groups, start, mixed_cost)
    lower_bound = final_relaxed.cost
    relative_gap = (
        0.0 if upper_bound <= 1e-12 else max(0.0, (upper_bound - lower_bound) / upper_bound)
    )
    return RouteSolution(
        group_order=incumbent_route.group_order,
        poses=incumbent_route.poses,
        cost=upper_bound,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        relative_gap=relative_gap,
        exact_edge_evaluations=len(exact_cache),
        total_possible_edges=total_edges,
        safely_pruned_edges=len(pruned),
    )
