from __future__ import annotations

import math
from heapq import heappop, heappush
from dataclasses import dataclass

from .models import Pose
from .risk import angular_distance


@dataclass(frozen=True, slots=True)
class AxisAlignedRectangle:
    minimum_x: float
    maximum_x: float
    minimum_y: float
    maximum_y: float

    def __post_init__(self) -> None:
        if self.minimum_x >= self.maximum_x or self.minimum_y >= self.maximum_y:
            raise ValueError("rectangle bounds must have positive area")


def _segment_intersects_rectangle(
    source: Pose,
    destination: Pose,
    rectangle: AxisAlignedRectangle,
) -> bool:
    """Liang-Barsky line clipping test."""

    delta_x = destination.x - source.x
    delta_y = destination.y - source.y
    p = (-delta_x, delta_x, -delta_y, delta_y)
    q = (
        source.x - rectangle.minimum_x,
        rectangle.maximum_x - source.x,
        source.y - rectangle.minimum_y,
        rectangle.maximum_y - source.y,
    )
    lower = 0.0
    upper = 1.0
    for coefficient, distance in zip(p, q, strict=True):
        if abs(coefficient) < 1e-12:
            if distance < 0.0:
                return False
            continue
        ratio = distance / coefficient
        if coefficient < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return False
    return True


class ProxyMotionOracle:
    """Deterministic expensive-cost proxy for algorithmic tests.

    `lower` is admissible because every term added by `exact` is non-negative.
    The northbound term and different departure/arrival weights make the cost asymmetric.
    """

    def __init__(
        self,
        obstacles: tuple[AxisAlignedRectangle, ...] = (),
        *,
        departure_turn_weight: float = 0.30,
        arrival_turn_weight: float = 0.12,
        northbound_weight: float = 0.08,
        obstacle_weight: float = 1.25,
    ) -> None:
        self.obstacles = obstacles
        self.departure_turn_weight = departure_turn_weight
        self.arrival_turn_weight = arrival_turn_weight
        self.northbound_weight = northbound_weight
        self.obstacle_weight = obstacle_weight
        self.exact_calls = 0

    @staticmethod
    def lower(source: Pose, destination: Pose) -> float:
        return math.hypot(destination.x - source.x, destination.y - source.y)

    def exact(self, source: Pose, destination: Pose) -> float:
        self.exact_calls += 1
        delta_x = destination.x - source.x
        delta_y = destination.y - source.y
        distance = math.hypot(delta_x, delta_y)
        if distance <= 1e-12:
            travel_bearing = source.yaw
        else:
            travel_bearing = math.atan2(delta_y, delta_x)
        departure_turn = angular_distance(source.yaw, travel_bearing)
        arrival_turn = angular_distance(travel_bearing, destination.yaw)
        directional_cost = self.northbound_weight * max(0.0, delta_y)
        obstacle_cost = 0.0
        for rectangle in self.obstacles:
            if _segment_intersects_rectangle(source, destination, rectangle):
                diagonal = math.hypot(
                    rectangle.maximum_x - rectangle.minimum_x,
                    rectangle.maximum_y - rectangle.minimum_y,
                )
                obstacle_cost += self.obstacle_weight * diagonal
        return (
            distance
            + self.departure_turn_weight * departure_turn
            + self.arrival_turn_weight * arrival_turn
            + directional_cost
            + obstacle_cost
        )


@dataclass(frozen=True, slots=True)
class OccupancyGrid:
    width: int
    height: int
    resolution: float
    blocked: frozenset[tuple[int, int]]

    @classmethod
    def from_rectangles(
        cls,
        *,
        width_metres: float,
        height_metres: float,
        resolution: float,
        rectangles: tuple[AxisAlignedRectangle, ...],
    ) -> "OccupancyGrid":
        if width_metres <= 0.0 or height_metres <= 0.0 or resolution <= 0.0:
            raise ValueError("grid dimensions and resolution must be positive")
        width = int(math.floor(width_metres / resolution)) + 1
        height = int(math.floor(height_metres / resolution)) + 1
        blocked = set()
        for x_index in range(width):
            x = x_index * resolution
            for y_index in range(height):
                y = y_index * resolution
                if any(
                    rectangle.minimum_x <= x <= rectangle.maximum_x
                    and rectangle.minimum_y <= y <= rectangle.maximum_y
                    for rectangle in rectangles
                ):
                    blocked.add((x_index, y_index))
        return cls(width, height, resolution, frozenset(blocked))

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return round(x / self.resolution), round(y / self.resolution)

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def is_free_cell(self, cell: tuple[int, int]) -> bool:
        return self.in_bounds(cell) and cell not in self.blocked

    def is_free_world(self, x: float, y: float) -> bool:
        return self.is_free_cell(self.world_to_cell(x, y))


class LatticeAStarOracle:
    """A discrete SE(2) motion oracle with rotation and forward/reverse primitives."""

    _DIRECTIONS = (
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    )

    def __init__(
        self,
        grid: OccupancyGrid,
        *,
        rotation_cost: float = 0.14,
        reverse_multiplier: float = 1.20,
    ) -> None:
        if rotation_cost <= 0.0 or reverse_multiplier < 1.0:
            raise ValueError("invalid lattice motion costs")
        self.grid = grid
        self.rotation_cost = rotation_cost
        self.reverse_multiplier = reverse_multiplier
        self.exact_calls = 0
        self.expanded_states = 0

    @staticmethod
    def lower(source: Pose, destination: Pose) -> float:
        return math.hypot(destination.x - source.x, destination.y - source.y)

    @staticmethod
    def _heading_index(yaw: float) -> int:
        normalized = yaw % (2.0 * math.pi)
        return round(normalized / (2.0 * math.pi) * 8) % 8

    def _pose_state(self, pose: Pose) -> tuple[int, int, int]:
        x_index, y_index = self.grid.world_to_cell(pose.x, pose.y)
        return x_index, y_index, self._heading_index(pose.yaw)

    def _translation_is_valid(
        self,
        source_x: int,
        source_y: int,
        delta_x: int,
        delta_y: int,
    ) -> bool:
        destination = (source_x + delta_x, source_y + delta_y)
        if not self.grid.is_free_cell(destination):
            return False
        if delta_x != 0 and delta_y != 0:
            return self.grid.is_free_cell((source_x + delta_x, source_y)) and self.grid.is_free_cell(
                (source_x, source_y + delta_y)
            )
        return True

    def exact(self, source: Pose, destination: Pose) -> float:
        self.exact_calls += 1
        start = self._pose_state(source)
        goal = self._pose_state(destination)
        if not self.grid.is_free_cell(start[:2]) or not self.grid.is_free_cell(goal[:2]):
            return math.inf
        if start == goal:
            return 0.0

        def heuristic(state: tuple[int, int, int]) -> float:
            return self.grid.resolution * math.hypot(
                goal[0] - state[0], goal[1] - state[1]
            )

        queue: list[tuple[float, float, tuple[int, int, int]]] = []
        heappush(queue, (heuristic(start), 0.0, start))
        best_cost = {start: 0.0}

        while queue:
            _, cost, state = heappop(queue)
            if cost > best_cost.get(state, math.inf) + 1e-12:
                continue
            self.expanded_states += 1
            if state == goal:
                return cost

            x_index, y_index, heading = state
            for heading_delta in (-1, 1):
                successor = (x_index, y_index, (heading + heading_delta) % 8)
                proposed = cost + self.rotation_cost
                if proposed + 1e-12 < best_cost.get(successor, math.inf):
                    best_cost[successor] = proposed
                    heappush(queue, (proposed + heuristic(successor), proposed, successor))

            delta_x, delta_y = self._DIRECTIONS[heading]
            step = self.grid.resolution * math.hypot(delta_x, delta_y)
            for direction, multiplier in ((1, 1.0), (-1, self.reverse_multiplier)):
                move_x = direction * delta_x
                move_y = direction * delta_y
                if not self._translation_is_valid(x_index, y_index, move_x, move_y):
                    continue
                successor = (x_index + move_x, y_index + move_y, heading)
                proposed = cost + multiplier * step
                if proposed + 1e-12 < best_cost.get(successor, math.inf):
                    best_cost[successor] = proposed
                    heappush(queue, (proposed + heuristic(successor), proposed, successor))

        return math.inf


class HybridAStarOracle:
    """Continuous-curvature Hybrid A* research prototype.

    The search propagates bicycle-like arc primitives in continuous `(x, y, yaw)` and uses a
    discretized key only for duplicate detection. It is intentionally compact and is not yet a
    production replacement for Nav2 or OMPL.
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        *,
        minimum_turning_radius: float = 0.55,
        primitive_length: float = 0.30,
        heading_bins: int = 24,
        robot_radius: float = 0.16,
        reverse_multiplier: float = 1.25,
        steering_penalty: float = 0.06,
        collision_step: float = 0.05,
        max_expansions: int = 60000,
    ) -> None:
        if minimum_turning_radius <= 0.0 or primitive_length <= 0.0:
            raise ValueError("turning radius and primitive length must be positive")
        if heading_bins < 8:
            raise ValueError("heading_bins must be at least 8")
        if robot_radius < 0.0 or reverse_multiplier < 1.0:
            raise ValueError("invalid robot radius or reverse multiplier")
        self.grid = grid
        self.minimum_turning_radius = minimum_turning_radius
        self.primitive_length = primitive_length
        self.heading_bins = heading_bins
        self.robot_radius = robot_radius
        self.reverse_multiplier = reverse_multiplier
        self.steering_penalty = steering_penalty
        self.collision_step = collision_step
        self.max_expansions = max_expansions
        self.exact_calls = 0
        self.expanded_states = 0
        self.failed_queries = 0
        self._inflated_blocked = self._build_inflated_obstacles()

    def _build_inflated_obstacles(self) -> frozenset[tuple[int, int]]:
        padding = self.robot_radius / self.grid.resolution + math.sqrt(2.0) / 2.0
        cell_radius = math.ceil(padding)
        inflated = set()
        for blocked_x, blocked_y in self.grid.blocked:
            for delta_x in range(-cell_radius, cell_radius + 1):
                for delta_y in range(-cell_radius, cell_radius + 1):
                    if math.hypot(delta_x, delta_y) <= padding + 1e-12:
                        cell = (blocked_x + delta_x, blocked_y + delta_y)
                        if self.grid.in_bounds(cell):
                            inflated.add(cell)
        boundary_margin = math.ceil(self.robot_radius / self.grid.resolution)
        if boundary_margin:
            for x_index in range(self.grid.width):
                for y_index in range(self.grid.height):
                    if (
                        x_index < boundary_margin
                        or y_index < boundary_margin
                        or x_index >= self.grid.width - boundary_margin
                        or y_index >= self.grid.height - boundary_margin
                    ):
                        inflated.add((x_index, y_index))
        return frozenset(inflated)

    def _is_free_world(self, x: float, y: float) -> bool:
        cell = self.grid.world_to_cell(x, y)
        return self.grid.in_bounds(cell) and cell not in self._inflated_blocked

    def is_pose_valid(self, pose: Pose) -> bool:
        canonical = self._canonical_pose(pose)
        return self._is_free_world(canonical[0], canonical[1])

    def _heading_index(self, yaw: float) -> int:
        normalized = yaw % (2.0 * math.pi)
        return round(normalized / (2.0 * math.pi) * self.heading_bins) % self.heading_bins

    def _heading_value(self, index: int) -> float:
        return 2.0 * math.pi * index / self.heading_bins

    def _state_key(self, x: float, y: float, yaw: float) -> tuple[int, int, int]:
        cell_x, cell_y = self.grid.world_to_cell(x, y)
        return cell_x, cell_y, self._heading_index(yaw)

    def _canonical_pose(self, pose: Pose) -> tuple[float, float, float]:
        cell_x, cell_y = self.grid.world_to_cell(pose.x, pose.y)
        heading = self._heading_index(pose.yaw)
        return (
            cell_x * self.grid.resolution,
            cell_y * self.grid.resolution,
            self._heading_value(heading),
        )

    def lower(self, source: Pose, destination: Pose) -> float:
        source_cell = self.grid.world_to_cell(source.x, source.y)
        destination_cell = self.grid.world_to_cell(destination.x, destination.y)
        return self.grid.resolution * math.hypot(
            destination_cell[0] - source_cell[0],
            destination_cell[1] - source_cell[1],
        )

    @staticmethod
    def _propagate(
        x: float,
        y: float,
        yaw: float,
        signed_distance: float,
        curvature: float,
    ) -> tuple[float, float, float]:
        if abs(curvature) <= 1e-12:
            return (
                x + signed_distance * math.cos(yaw),
                y + signed_distance * math.sin(yaw),
                yaw,
            )
        delta_yaw = signed_distance * curvature
        next_yaw = yaw + delta_yaw
        return (
            x + (math.sin(next_yaw) - math.sin(yaw)) / curvature,
            y - (math.cos(next_yaw) - math.cos(yaw)) / curvature,
            next_yaw,
        )

    def _primitive_is_free(
        self,
        x: float,
        y: float,
        yaw: float,
        signed_distance: float,
        curvature: float,
    ) -> bool:
        samples = max(2, math.ceil(abs(signed_distance) / self.collision_step))
        for sample_index in range(1, samples + 1):
            fraction = sample_index / samples
            sample_x, sample_y, _ = self._propagate(
                x,
                y,
                yaw,
                signed_distance * fraction,
                curvature,
            )
            if not self._is_free_world(sample_x, sample_y):
                return False
        return True

    def exact(self, source: Pose, destination: Pose) -> float:
        self.exact_calls += 1
        start_pose = self._canonical_pose(source)
        goal_pose = self._canonical_pose(destination)
        start_key = self._state_key(*start_pose)
        goal_key = self._state_key(*goal_pose)
        if not self._is_free_world(start_pose[0], start_pose[1]) or not self._is_free_world(
            goal_pose[0], goal_pose[1]
        ):
            self.failed_queries += 1
            return math.inf
        if start_key == goal_key:
            return 0.0

        def heuristic(pose: tuple[float, float, float]) -> float:
            return math.hypot(goal_pose[0] - pose[0], goal_pose[1] - pose[1])

        queue: list[tuple[float, float, int, tuple[int, int, int]]] = []
        counter = 0
        heappush(queue, (heuristic(start_pose), 0.0, counter, start_key))
        best_cost = {start_key: 0.0}
        representative = {start_key: start_pose}
        local_expansions = 0
        curvatures = (
            -1.0 / self.minimum_turning_radius,
            0.0,
            1.0 / self.minimum_turning_radius,
        )

        while queue and local_expansions < self.max_expansions:
            _, cost, _, key = heappop(queue)
            if cost > best_cost.get(key, math.inf) + 1e-12:
                continue
            pose = representative[key]
            local_expansions += 1
            self.expanded_states += 1
            if key == goal_key:
                residual_distance = math.hypot(goal_pose[0] - pose[0], goal_pose[1] - pose[1])
                residual_heading = angular_distance(goal_pose[2], pose[2])
                return cost + residual_distance + self.minimum_turning_radius * residual_heading

            for direction in (1.0, -1.0):
                signed_distance = direction * self.primitive_length
                direction_multiplier = 1.0 if direction > 0.0 else self.reverse_multiplier
                for curvature in curvatures:
                    if not self._primitive_is_free(
                        pose[0], pose[1], pose[2], signed_distance, curvature
                    ):
                        continue
                    successor_pose = self._propagate(
                        pose[0], pose[1], pose[2], signed_distance, curvature
                    )
                    successor_key = self._state_key(*successor_pose)
                    if not self.grid.in_bounds(successor_key[:2]):
                        continue
                    action_cost = (
                        self.primitive_length * direction_multiplier
                        + self.steering_penalty
                        * abs(curvature)
                        * self.primitive_length
                    )
                    proposed = cost + action_cost
                    if proposed + 1e-12 < best_cost.get(successor_key, math.inf):
                        best_cost[successor_key] = proposed
                        representative[successor_key] = successor_pose
                        counter += 1
                        heappush(
                            queue,
                            (
                                proposed + heuristic(successor_pose),
                                proposed,
                                counter,
                                successor_key,
                            ),
                        )

        self.failed_queries += 1
        return math.inf
