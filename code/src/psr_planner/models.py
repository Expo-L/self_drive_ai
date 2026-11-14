from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Pose:
    """A planar robot pose used as a discrete service candidate."""

    x: float
    y: float
    yaw: float
    label: str


@dataclass(frozen=True, slots=True)
class TargetBelief:
    """Independent Gaussian position belief for one service target."""

    name: str
    mean_x: float
    mean_y: float
    sigma_x: float
    sigma_y: float


@dataclass(frozen=True, slots=True)
class CandidateCalibration:
    """Finite-sample risk calibration result for one candidate pose."""

    pose: Pose
    trials: int
    successes: int
    empirical_success: float
    lower_confidence: float
    risk_epsilon: float
    confidence: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class RouteSolution:
    """A route and its optimization certificate."""

    group_order: tuple[str, ...]
    poses: tuple[Pose, ...]
    cost: float
    lower_bound: float
    upper_bound: float
    relative_gap: float
    exact_edge_evaluations: int
    total_possible_edges: int
    safely_pruned_edges: int = 0
