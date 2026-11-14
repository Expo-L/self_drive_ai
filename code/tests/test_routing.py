from __future__ import annotations

import math
import random
import unittest

from psr_planner.models import Pose
from psr_planner.oracles import (
    AxisAlignedRectangle,
    HybridAStarOracle,
    LatticeAStarOracle,
    OccupancyGrid,
    ProxyMotionOracle,
)
from psr_planner.routing import (
    forced_edge_route_lower_bounds,
    full_matrix_solve,
    lazy_bounded_solve,
    one_shot_lower_solve,
    screened_lazy_bounded_solve,
)


class RoutingCertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.start = Pose(0.0, 0.0, 0.0, "start")
        self.groups = {
            "A": (
                Pose(1.0, 0.2, 0.0, "A0"),
                Pose(1.1, 0.8, 0.5, "A1"),
            ),
            "B": (
                Pose(2.7, 0.3, 0.2, "B0"),
                Pose(2.3, 1.4, 1.0, "B1"),
            ),
            "C": (
                Pose(3.8, 2.3, 1.5, "C0"),
                Pose(2.8, 2.8, 2.6, "C1"),
            ),
            "D": (
                Pose(0.8, 2.7, -2.2, "D0"),
                Pose(1.5, 3.2, -1.8, "D1"),
            ),
        }
        self.obstacles = (AxisAlignedRectangle(1.6, 2.1, 0.4, 2.3),)

    def test_zero_gap_matches_full_matrix_with_fewer_queries(self) -> None:
        full = full_matrix_solve(
            self.groups,
            self.start,
            ProxyMotionOracle(self.obstacles),
        )
        lazy = lazy_bounded_solve(
            self.groups,
            self.start,
            ProxyMotionOracle(self.obstacles),
            relative_tolerance=0.0,
        )
        self.assertAlmostEqual(lazy.cost, full.cost, places=9)
        self.assertAlmostEqual(lazy.lower_bound, full.cost, places=9)
        self.assertAlmostEqual(lazy.upper_bound, full.cost, places=9)
        self.assertLess(lazy.exact_edge_evaluations, full.exact_edge_evaluations)

    def test_nonzero_tolerance_reports_valid_reference_interval(self) -> None:
        full = full_matrix_solve(
            self.groups,
            self.start,
            ProxyMotionOracle(self.obstacles),
        )
        lazy = lazy_bounded_solve(
            self.groups,
            self.start,
            ProxyMotionOracle(self.obstacles),
            relative_tolerance=0.10,
        )
        self.assertLessEqual(lazy.lower_bound, full.cost + 1e-9)
        self.assertGreaterEqual(lazy.upper_bound + 1e-9, full.cost)
        self.assertLessEqual(lazy.relative_gap, 0.10 + 1e-9)

    def test_forced_edge_bounds_are_admissible(self) -> None:
        lower_bounds = forced_edge_route_lower_bounds(
            self.groups,
            self.start,
            ProxyMotionOracle(self.obstacles).lower,
        )
        exact_conditional_costs = forced_edge_route_lower_bounds(
            self.groups,
            self.start,
            ProxyMotionOracle(self.obstacles).exact,
        )
        self.assertEqual(lower_bounds.keys(), exact_conditional_costs.keys())
        for edge, lower in lower_bounds.items():
            self.assertLessEqual(lower, exact_conditional_costs[edge] + 1e-9)

    def test_screened_lazy_matches_optimum_and_safely_prunes(self) -> None:
        full = full_matrix_solve(
            self.groups,
            self.start,
            ProxyMotionOracle(self.obstacles),
        )
        screened = screened_lazy_bounded_solve(
            self.groups,
            self.start,
            ProxyMotionOracle(self.obstacles),
            relative_tolerance=0.0,
        )
        self.assertAlmostEqual(screened.cost, full.cost, places=9)
        self.assertLessEqual(screened.lower_bound, full.cost + 1e-9)
        self.assertGreaterEqual(screened.upper_bound + 1e-9, full.cost)
        self.assertGreater(screened.safely_pruned_edges, 0)
        self.assertLess(screened.exact_edge_evaluations, full.exact_edge_evaluations)

    def test_one_shot_bounds_contain_full_optimum(self) -> None:
        full = full_matrix_solve(
            self.groups,
            self.start,
            ProxyMotionOracle(self.obstacles),
        )
        one_shot = one_shot_lower_solve(
            self.groups,
            self.start,
            ProxyMotionOracle(self.obstacles),
        )
        self.assertLessEqual(one_shot.lower_bound, full.cost + 1e-9)
        self.assertGreaterEqual(one_shot.upper_bound + 1e-9, full.cost)
        self.assertEqual(one_shot.exact_edge_evaluations, len(self.groups))

    def test_random_small_instances_match_full_optimum(self) -> None:
        for seed in range(8):
            rng = random.Random(seed)
            groups = {}
            for group_index in range(4):
                center_x = rng.uniform(0.5, 4.5)
                center_y = rng.uniform(0.5, 4.5)
                group = []
                for candidate_index in range(3):
                    angle = 2.0 * math.pi * candidate_index / 3.0
                    group.append(
                        Pose(
                            center_x + 0.35 * math.cos(angle),
                            center_y + 0.35 * math.sin(angle),
                            angle,
                            f"G{group_index}C{candidate_index}",
                        )
                    )
                groups[f"G{group_index}"] = tuple(group)

            obstacles = (
                AxisAlignedRectangle(1.7, 2.2, 1.1, 3.6),
                AxisAlignedRectangle(3.0, 3.5, 0.4, 2.1),
            )
            full = full_matrix_solve(
                groups,
                self.start,
                ProxyMotionOracle(obstacles),
            )
            lazy = lazy_bounded_solve(
                groups,
                self.start,
                ProxyMotionOracle(obstacles),
                relative_tolerance=0.0,
            )
            self.assertAlmostEqual(lazy.cost, full.cost, places=8, msg=f"seed={seed}")
            self.assertLessEqual(lazy.lower_bound, full.cost + 1e-8)
            self.assertGreaterEqual(lazy.upper_bound + 1e-8, full.cost)

            screened = screened_lazy_bounded_solve(
                groups,
                self.start,
                ProxyMotionOracle(obstacles),
                relative_tolerance=0.0,
            )
            self.assertAlmostEqual(screened.cost, full.cost, places=8, msg=f"seed={seed}")
            self.assertLessEqual(screened.lower_bound, full.cost + 1e-8)
            self.assertGreaterEqual(screened.upper_bound + 1e-8, full.cost)

    def test_lattice_exact_cost_dominates_euclidean_lower_bound(self) -> None:
        grid = OccupancyGrid.from_rectangles(
            width_metres=4.0,
            height_metres=4.0,
            resolution=0.25,
            rectangles=(AxisAlignedRectangle(1.5, 2.0, 0.0, 3.0),),
        )
        source = Pose(0.25, 0.25, 0.0, "source")
        destination = Pose(3.5, 3.5, math.pi / 2.0, "destination")
        oracle = LatticeAStarOracle(grid)
        exact = oracle.exact(source, destination)
        self.assertTrue(math.isfinite(exact))
        self.assertGreaterEqual(exact + 1e-12, oracle.lower(source, destination))

    def test_hybrid_astar_uses_curvature_primitives_and_valid_lower_bound(self) -> None:
        grid = OccupancyGrid.from_rectangles(
            width_metres=5.0,
            height_metres=5.0,
            resolution=0.20,
            rectangles=(AxisAlignedRectangle(2.0, 2.6, 0.7, 3.7),),
        )
        source = Pose(0.6, 0.6, 0.0, "source")
        destination = Pose(4.4, 4.2, math.pi / 2.0, "destination")
        oracle = HybridAStarOracle(
            grid,
            minimum_turning_radius=0.55,
            primitive_length=0.30,
            robot_radius=0.10,
        )
        exact = oracle.exact(source, destination)
        self.assertTrue(math.isfinite(exact))
        self.assertGreaterEqual(exact + 1e-12, oracle.lower(source, destination))
        self.assertGreater(oracle.expanded_states, 0)


if __name__ == "__main__":
    unittest.main()
