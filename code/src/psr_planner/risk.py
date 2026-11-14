from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence

from .models import CandidateCalibration, Pose, TargetBelief


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angular_distance(a: float, b: float) -> float:
    return abs(wrap_angle(a - b))


def _binomial_tail_at_least(successes: int, trials: int, probability: float) -> float:
    """Return P[X >= successes] for X ~ Binomial(trials, probability)."""

    if successes <= 0:
        return 1.0
    if successes > trials:
        return 0.0
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0

    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    log_terms = []
    for count in range(successes, trials + 1):
        log_combination = (
            math.lgamma(trials + 1)
            - math.lgamma(count + 1)
            - math.lgamma(trials - count + 1)
        )
        log_terms.append(
            log_combination + count * log_p + (trials - count) * log_q
        )
    pivot = max(log_terms)
    return math.exp(pivot) * sum(math.exp(value - pivot) for value in log_terms)


def clopper_pearson_lower(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> float:
    """Exact one-sided Clopper-Pearson lower confidence bound.

    The returned value L satisfies P[Binomial(trials, L) >= successes] = 1-confidence.
    """

    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if successes == 0:
        return 0.0

    alpha = 1.0 - confidence
    lower = 0.0
    upper = 1.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if _binomial_tail_at_least(successes, trials, middle) < alpha:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def generate_ring_candidates(
    target: TargetBelief,
    radii: Sequence[float],
    angles: int,
) -> list[Pose]:
    """Generate base poses that face the target mean from concentric rings."""

    if angles <= 0:
        raise ValueError("angles must be positive")
    candidates: list[Pose] = []
    for radius_index, radius in enumerate(radii):
        if radius <= 0.0:
            raise ValueError("candidate radii must be positive")
        for angle_index in range(angles):
            bearing = 2.0 * math.pi * angle_index / angles
            candidates.append(
                Pose(
                    x=target.mean_x - radius * math.cos(bearing),
                    y=target.mean_y - radius * math.sin(bearing),
                    yaw=wrap_angle(bearing),
                    label=f"{target.name}:r{radius_index}:a{angle_index}",
                )
            )
    return candidates


def service_trial_succeeds(
    pose: Pose,
    target: TargetBelief,
    rng: random.Random,
    *,
    base_position_sigma: float = 0.04,
    base_yaw_sigma: float = math.radians(2.0),
    minimum_reach: float = 0.45,
    maximum_reach: float = 1.05,
    maximum_heading_error: float = math.radians(18.0),
    clearance_check: Callable[[Pose], bool] | None = None,
) -> bool:
    """Draw one execution disturbance and evaluate a fixed-tool service constraint."""

    actual_target_x = rng.gauss(target.mean_x, target.sigma_x)
    actual_target_y = rng.gauss(target.mean_y, target.sigma_y)
    actual_pose = Pose(
        x=rng.gauss(pose.x, base_position_sigma),
        y=rng.gauss(pose.y, base_position_sigma),
        yaw=wrap_angle(rng.gauss(pose.yaw, base_yaw_sigma)),
        label=pose.label,
    )
    delta_x = actual_target_x - actual_pose.x
    delta_y = actual_target_y - actual_pose.y
    distance = math.hypot(delta_x, delta_y)
    target_bearing = math.atan2(delta_y, delta_x)
    heading_ok = angular_distance(actual_pose.yaw, target_bearing) <= maximum_heading_error
    reach_ok = minimum_reach <= distance <= maximum_reach
    clearance_ok = clearance_check(actual_pose) if clearance_check is not None else True
    return reach_ok and heading_ok and clearance_ok


def calibrate_candidate(
    pose: Pose,
    target: TargetBelief,
    *,
    trials: int,
    risk_epsilon: float,
    confidence: float,
    seed: int,
    trial_function: Callable[..., bool] = service_trial_succeeds,
    **trial_kwargs: float,
) -> CandidateCalibration:
    """Calibrate one candidate on an independent Monte Carlo sample."""

    if not 0.0 < risk_epsilon < 1.0:
        raise ValueError("risk_epsilon must lie in (0, 1)")
    rng = random.Random(seed)
    successes = sum(
        trial_function(pose, target, rng, **trial_kwargs) for _ in range(trials)
    )
    lower = clopper_pearson_lower(successes, trials, confidence)
    return CandidateCalibration(
        pose=pose,
        trials=trials,
        successes=successes,
        empirical_success=successes / trials,
        lower_confidence=lower,
        risk_epsilon=risk_epsilon,
        confidence=confidence,
        accepted=lower >= 1.0 - risk_epsilon,
    )


def calibrate_candidates(
    candidates: Iterable[Pose],
    target: TargetBelief,
    *,
    trials: int,
    risk_epsilon: float,
    confidence: float,
    seed: int,
    **trial_kwargs: float,
) -> list[CandidateCalibration]:
    results = []
    for index, pose in enumerate(candidates):
        results.append(
            calibrate_candidate(
                pose,
                target,
                trials=trials,
                risk_epsilon=risk_epsilon,
                confidence=confidence,
                seed=seed + 104729 * index,
                **trial_kwargs,
            )
        )
    return results


def estimate_success_rate(
    pose: Pose,
    target: TargetBelief,
    *,
    trials: int,
    seed: int,
    **trial_kwargs: float,
) -> float:
    rng = random.Random(seed)
    successes = sum(
        service_trial_succeeds(pose, target, rng, **trial_kwargs)
        for _ in range(trials)
    )
    return successes / trials
