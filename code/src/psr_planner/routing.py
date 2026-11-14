from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .models import Pose, RouteSolution


EdgeCost = Callable[[Pose, Pose], float]
Groups = Mapping[str, Sequence[Pose]]


@dataclass(frozen=True, slots=True)
class _DiscreteRoute:
    group_order: tuple[str, ...]
    poses: tuple[Pose, ...]
    cost: float


def _validate_groups(groups: Groups) -> tuple[str, ...]:
    names = tuple(groups.keys())
    if not names:
        return names
    for name in names:
        if not groups[name]:
            raise ValueError(f"group {name!r} has no service candidates")
    return names


def solve_grouped_route(groups: Groups, start: Pose, edge_cost: EdgeCost) -> _DiscreteRoute:
    """Exactly solve an open grouped Hamiltonian path by subset dynamic programming."""

    names = _validate_groups(groups)
    if not names:
        return _DiscreteRoute((), (), 0.0)

    candidates = tuple(tuple(groups[name]) for name in names)
    state_cost: dict[tuple[int, int, int], float] = {}
    predecessor: dict[
        tuple[int, int, int], tuple[int, int, int] | None
    ] = {}

    for group_index, group_candidates in enumerate(candidates):
        mask = 1 << group_index
        for candidate_index, pose in enumerate(group_candidates):
            key = (mask, group_index, candidate_index)
            state_cost[key] = edge_cost(start, pose)
            predecessor[key] = None

    full_mask = (1 << len(names)) - 1
    for mask in range(1, full_mask + 1):
        for last_group in range(len(names)):
            if not mask & (1 << last_group):
                continue
            for last_candidate, last_pose in enumerate(candidates[last_group]):
                key = (mask, last_group, last_candidate)
                current = state_cost.get(key)
                if current is None:
                    continue
                for next_group in range(len(names)):
                    if mask & (1 << next_group):
                        continue
                    next_mask = mask | (1 << next_group)
                    for next_candidate, next_pose in enumerate(candidates[next_group]):
                        next_key = (next_mask, next_group, next_candidate)
                        proposed = current + edge_cost(last_pose, next_pose)
                        if proposed + 1e-12 < state_cost.get(next_key, math.inf):
                            state_cost[next_key] = proposed
                            predecessor[next_key] = key

    final_key = min(
        (key for key in state_cost if key[0] == full_mask),
        key=state_cost.__getitem__,
    )
    reverse_groups: list[str] = []
    reverse_poses: list[Pose] = []
    cursor: tuple[int, int, int] | None = final_key
    while cursor is not None:
        _, group_index, candidate_index = cursor
        reverse_groups.append(names[group_index])
        reverse_poses.append(candidates[group_index][candidate_index])
        cursor = predecessor[cursor]

    return _DiscreteRoute(
        group_order=tuple(reversed(reverse_groups)),
        poses=tuple(reversed(reverse_poses)),
        cost=state_cost[final_key],
    )


def _route_edges(start: Pose, route: _DiscreteRoute) -> tuple[tuple[Pose, Pose], ...]:
    if not route.poses:
        return ()
    nodes = (start,) + route.poses
    return tuple(zip(nodes[:-1], nodes[1:], strict=True))


def _total_possible_edges(groups: Groups) -> int:
    names = tuple(groups)
    start_edges = sum(len(groups[name]) for name in names)
    cross_edges = 0
    for source_name in names:
        for destination_name in names:
            if source_name != destination_name:
                cross_edges += len(groups[source_name]) * len(groups[destination_name])
    return start_edges + cross_edges


def _all_directed_edges(
    groups: Groups,
    start: Pose,
) -> tuple[tuple[Pose, Pose], ...]:
    names = tuple(groups)
    edges = [
        (start, pose)
        for name in names
        for pose in groups[name]
    ]
    edges.extend(
        (source, destination)
        for source_name in names
        for destination_name in names
        if source_name != destination_name
        for source in groups[source_name]
        for destination in groups[destination_name]
    )
    return tuple(edges)


def forced_edge_route_lower_bounds(
    groups: Groups,
    start: Pose,
    edge_cost: EdgeCost,
) -> dict[tuple[Pose, Pose], float]:
    """Return the best relaxed route cost conditional on using each directed edge.

    The calculation uses a forward subset DP and a suffix subset DP. If `edge_cost` is a
    pointwise lower bound on the exact directed costs, every returned value is a valid lower
    bound on any complete grouped route containing that edge.
    """

    names = _validate_groups(groups)
    if not names:
        return {}
    candidates = tuple(tuple(groups[name]) for name in names)
    full_mask = (1 << len(names)) - 1

    forward: dict[tuple[int, int, int], float] = {}
    for group_index, group_candidates in enumerate(candidates):
        mask = 1 << group_index
        for candidate_index, pose in enumerate(group_candidates):
            forward[(mask, group_index, candidate_index)] = edge_cost(start, pose)

    for mask in range(1, full_mask + 1):
        for last_group, group_candidates in enumerate(candidates):
            if not mask & (1 << last_group):
                continue
            for last_candidate, last_pose in enumerate(group_candidates):
                current = forward.get((mask, last_group, last_candidate))
                if current is None:
                    continue
                for next_group, next_candidates in enumerate(candidates):
                    if mask & (1 << next_group):
                        continue
                    next_mask = mask | (1 << next_group)
                    for next_candidate, next_pose in enumerate(next_candidates):
                        key = (next_mask, next_group, next_candidate)
                        proposed = current + edge_cost(last_pose, next_pose)
                        if proposed < forward.get(key, math.inf):
                            forward[key] = proposed

    suffix: dict[tuple[int, int, int], float] = {}
    for current_group, group_candidates in enumerate(candidates):
        for current_candidate in range(len(group_candidates)):
            suffix[(0, current_group, current_candidate)] = 0.0

    for mask in range(1, full_mask + 1):
        for current_group, group_candidates in enumerate(candidates):
            if mask & (1 << current_group):
                continue
            for current_candidate, current_pose in enumerate(group_candidates):
                best = math.inf
                for next_group, next_candidates in enumerate(candidates):
                    next_bit = 1 << next_group
                    if not mask & next_bit:
                        continue
                    remainder = mask ^ next_bit
                    for next_candidate, next_pose in enumerate(next_candidates):
                        proposed = edge_cost(current_pose, next_pose) + suffix[
                            (remainder, next_group, next_candidate)
                        ]
                        best = min(best, proposed)
                suffix[(mask, current_group, current_candidate)] = best

    bounds: dict[tuple[Pose, Pose], float] = {}
    for destination_group, destination_candidates in enumerate(candidates):
        remaining = full_mask ^ (1 << destination_group)
        for destination_candidate, destination in enumerate(destination_candidates):
            bounds[(start, destination)] = edge_cost(start, destination) + suffix[
                (remaining, destination_group, destination_candidate)
            ]

    for source_group, source_candidates in enumerate(candidates):
        source_bit = 1 << source_group
        for destination_group, destination_candidates in enumerate(candidates):
            if source_group == destination_group:
                continue
            destination_bit = 1 << destination_group
            for source_candidate, source in enumerate(source_candidates):
                for destination_candidate, destination in enumerate(destination_candidates):
                    best = math.inf
                    for prefix_mask in range(1, full_mask + 1):
                        if not prefix_mask & source_bit or prefix_mask & destination_bit:
                            continue
                        prefix = forward.get(
                            (prefix_mask, source_group, source_candidate), math.inf
                        )
                        remaining = full_mask & ~(prefix_mask | destination_bit)
                        tail = suffix[(remaining, destination_group, destination_candidate)]
                        best = min(
                            best,
                            prefix + edge_cost(source, destination) + tail,
                        )
                    bounds[(source, destination)] = best
    return bounds


def full_matrix_solve(groups: Groups, start: Pose, oracle: object) -> RouteSolution:
    """Baseline that evaluates every directed inter-group edge before optimization."""

    names = _validate_groups(groups)
    exact_cache: dict[tuple[Pose, Pose], float] = {}
    for name in names:
        for pose in groups[name]:
            exact_cache[(start, pose)] = oracle.exact(start, pose)
    for source_name in names:
        for destination_name in names:
            if source_name == destination_name:
                continue
            for source in groups[source_name]:
                for destination in groups[destination_name]:
                    exact_cache[(source, destination)] = oracle.exact(source, destination)

    route = solve_grouped_route(
        groups,
        start,
        lambda source, destination: exact_cache[(source, destination)],
    )
    return RouteSolution(
        group_order=route.group_order,
        poses=route.poses,
        cost=route.cost,
        lower_bound=route.cost,
        upper_bound=route.cost,
        relative_gap=0.0,
        exact_edge_evaluations=len(exact_cache),
        total_possible_edges=_total_possible_edges(groups),
    )


def one_shot_lower_solve(groups: Groups, start: Pose, oracle: object) -> RouteSolution:
    """Plan once on cheap lower bounds, then evaluate only the selected route.

    This baseline has minimal exact-query cost but does not refine a poor optimistic choice.
    """

    _validate_groups(groups)
    optimistic = solve_grouped_route(groups, start, oracle.lower)
    exact_cost = sum(oracle.exact(*edge) for edge in _route_edges(start, optimistic))
    if math.isinf(exact_cost):
        relative_gap = math.inf
    elif exact_cost <= 1e-12:
        relative_gap = 0.0
    else:
        relative_gap = max(0.0, (exact_cost - optimistic.cost) / exact_cost)
    return RouteSolution(
        group_order=optimistic.group_order,
        poses=optimistic.poses,
        cost=exact_cost,
        lower_bound=optimistic.cost,
        upper_bound=exact_cost,
        relative_gap=relative_gap,
        exact_edge_evaluations=len(optimistic.poses),
        total_possible_edges=_total_possible_edges(groups),
    )
def lazy_bounded_solve(
    groups: Groups,
    start: Pose,
    oracle: object,
    *,
    relative_tolerance: float = 0.0,
) -> RouteSolution:
    """Solve with admissible unevaluated-edge bounds and an anytime certificate.

    Unevaluated edges use `oracle.lower`. Evaluated edges use `oracle.exact`. Therefore the
    optimum of the mixed graph is a valid lower bound, while every fully evaluated route is
    a valid upper bound.
    """

    if not 0.0 <= relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie in [0, 1)")
    _validate_groups(groups)
    total_edges = _total_possible_edges(groups)
    exact_cache: dict[tuple[Pose, Pose], float] = {}

    def mixed_cost(source: Pose, destination: Pose) -> float:
        key = (source, destination)
        if key in exact_cache:
            return exact_cache[key]
        return oracle.lower(source, destination)

    def evaluate_edge(edge: tuple[Pose, Pose]) -> float:
        if edge not in exact_cache:
            exact_cache[edge] = oracle.exact(*edge)
        return exact_cache[edge]

    def evaluate_route(route: _DiscreteRoute) -> float:
        return sum(evaluate_edge(edge) for edge in _route_edges(start, route))

    optimistic_seed = solve_grouped_route(groups, start, mixed_cost)
    incumbent_route = optimistic_seed
    upper_bound = evaluate_route(optimistic_seed)

    lower_bound = 0.0
    for _ in range(2 * total_edges + 2):
        relaxed_route = solve_grouped_route(groups, start, mixed_cost)
        lower_bound = relaxed_route.cost
        if math.isfinite(upper_bound) and (
            upper_bound - lower_bound
            <= relative_tolerance * upper_bound + 1e-10
        ):
            break

        relaxed_edges = _route_edges(start, relaxed_route)
        unknown_edges = [edge for edge in relaxed_edges if edge not in exact_cache]
        if not unknown_edges:
            exact_cost = sum(exact_cache[edge] for edge in relaxed_edges)
            if exact_cost < upper_bound:
                incumbent_route = relaxed_route
                upper_bound = exact_cost
            continue

        selected_edge = max(unknown_edges, key=lambda edge: oracle.lower(*edge))
        evaluate_edge(selected_edge)
        if all(edge in exact_cache for edge in relaxed_edges):
            exact_cost = sum(exact_cache[edge] for edge in relaxed_edges)
            if exact_cost < upper_bound:
                incumbent_route = relaxed_route
                upper_bound = exact_cost
    else:
        raise RuntimeError("lazy solver exhausted the finite edge set without a certificate")

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
    )


def screened_lazy_bounded_solve(
    groups: Groups,
    start: Pose,
    oracle: object,
    *,
    relative_tolerance: float = 0.0,
) -> RouteSolution:
    """Lazy exact evaluation with group-aware safe directed-edge screening.

    For every unevaluated edge, the solver computes the minimum mixed-cost route constrained to
    contain that edge. If this conditional lower bound is no smaller than the incumbent feasible
    upper bound, the edge cannot improve the incumbent and is safely removed. Unlike IST*'s MST
    cycle rule, this test remains valid for directed costs and one-of-K group selection.
    """

    if not 0.0 <= relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie in [0, 1)")
    _validate_groups(groups)
    all_edges = _all_directed_edges(groups, start)
    total_edges = len(all_edges)
    exact_cache: dict[tuple[Pose, Pose], float] = {}
    pruned_edges: set[tuple[Pose, Pose]] = set()

    def mixed_cost(source: Pose, destination: Pose) -> float:
        edge = (source, destination)
        if edge in exact_cache:
            return exact_cache[edge]
        if edge in pruned_edges:
            return math.inf
        return oracle.lower(source, destination)

    def evaluate_edge(edge: tuple[Pose, Pose]) -> float:
        if edge not in exact_cache:
            exact_cache[edge] = oracle.exact(*edge)
        return exact_cache[edge]

    def evaluate_route(route: _DiscreteRoute) -> float:
        return sum(evaluate_edge(edge) for edge in _route_edges(start, route))

    optimistic_seed = solve_grouped_route(groups, start, mixed_cost)
    incumbent_route = optimistic_seed
    upper_bound = evaluate_route(optimistic_seed)
    lower_bound = 0.0

    for _ in range(2 * total_edges + 2):
        conditional_bounds = forced_edge_route_lower_bounds(groups, start, mixed_cost)
        for edge in all_edges:
            if edge in exact_cache or edge in pruned_edges:
                continue
            if conditional_bounds[edge] >= upper_bound - 1e-10:
                pruned_edges.add(edge)

        relaxed_route = solve_grouped_route(groups, start, mixed_cost)
        lower_bound = relaxed_route.cost
        if math.isfinite(upper_bound) and (
            upper_bound - lower_bound
            <= relative_tolerance * upper_bound + 1e-10
        ):
            break

        relaxed_edges = _route_edges(start, relaxed_route)
        unknown_edges = [edge for edge in relaxed_edges if edge not in exact_cache]
        if not unknown_edges:
            exact_cost = sum(exact_cache[edge] for edge in relaxed_edges)
            if exact_cost < upper_bound:
                incumbent_route = relaxed_route
                upper_bound = exact_cost
            continue

        def exclusion_lower_bound(edge: tuple[Pose, Pose]) -> float:
            pruned_edges.add(edge)
            try:
                return solve_grouped_route(groups, start, mixed_cost).cost
            finally:
                pruned_edges.remove(edge)

        # Query the optimistic-route edge whose exclusion most raises the global lower bound.
        # This is a directed group-aware analogue of prioritizing structurally influential edges.
        selected_edge = max(unknown_edges, key=exclusion_lower_bound)
        evaluate_edge(selected_edge)
        if all(edge in exact_cache for edge in relaxed_edges):
            exact_cost = sum(exact_cache[edge] for edge in relaxed_edges)
            if exact_cost < upper_bound:
                incumbent_route = relaxed_route
                upper_bound = exact_cost
    else:
        raise RuntimeError("screened lazy solver exhausted the finite edge set without a certificate")

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
        safely_pruned_edges=len(pruned_edges),
    )
