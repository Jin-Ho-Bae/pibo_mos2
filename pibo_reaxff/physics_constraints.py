"""
Physics-informed constraints for ReaxFF parameter optimization.

Two complementary mechanisms are implemented:

  1. ``PhysicsPenalty`` - additive loss term that grows quadratically once a
     parameter strays outside its physical bound. Differentiable, easy to
     wire into any optimizer (PSO, INDEEDopt, JAX-ReaxFF).

  2. ``ConstrainedAcquisition`` - multiplies the BO acquisition function by
     a feasibility probability, so PIBO never *proposes* infeasible points
     even before they get penalized. This is the mechanism that gives PIBO
     its sparse-data advantage over naive BO.

Physics rules encoded (all soft, all overridable in JSON config):
  - r_vdW must be positive and within (0.5, 4.0) A.
  - De_sigma must be positive (attractive sigma bond).
  - alpha (vdW exponent) bounded so the Morse curve does not blow up.
  - Equilibrium angles theta00 in [40, 170] deg.
  - Bond-order exponents p_bo1 < 0 (the ReaxFF physical convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

from .parameters import ParameterSpec, bounds_array


# A physics rule is a callable returning a non-negative violation magnitude.
PhysicsRule = Callable[[Dict[str, float]], float]


def rule_positive(name: str) -> PhysicsRule:
    def r(p: Dict[str, float]) -> float:
        return max(0.0, -p.get(name, 0.0))
    r.__name__ = f"positive({name})"
    return r


def rule_negative(name: str) -> PhysicsRule:
    def r(p: Dict[str, float]) -> float:
        return max(0.0, p.get(name, 0.0))
    r.__name__ = f"negative({name})"
    return r


def rule_range(name: str, lo: float, hi: float) -> PhysicsRule:
    def r(p: Dict[str, float]) -> float:
        v = p.get(name, 0.5 * (lo + hi))
        return max(0.0, lo - v) + max(0.0, v - hi)
    r.__name__ = f"range({name},{lo},{hi})"
    return r


def default_rules() -> List[PhysicsRule]:
    """Curated physics rules for the MoSH ReaxFF parameter set."""
    return [
        rule_positive("De_sigma_MoS"),
        rule_range("r_vdW_MoS", 0.5, 4.0),
        rule_range("alpha_MoS", 6.0, 16.0),
        rule_range("theta00_MoSMo", 40.0, 170.0),
        rule_range("theta00_SMoS", 40.0, 170.0),
        rule_negative("p_bo1_MoS"),
        rule_positive("p_val1_MoSMo"),
    ]


@dataclass
class PhysicsPenalty:
    """Quadratic-violation penalty added to the loss."""

    specs: List[ParameterSpec]
    rules: List[PhysicsRule] = field(default_factory=default_rules)
    lambda_: float = 1.0

    def __call__(self, params_vec: np.ndarray) -> float:
        pdict = {s.name: float(v) for s, v in zip(self.specs, params_vec)}
        violation = sum(r(pdict) ** 2 for r in self.rules)
        return self.lambda_ * violation

    def violations(self, params_vec: np.ndarray) -> Dict[str, float]:
        """Per-rule violation magnitudes (useful for plots / debugging)."""
        pdict = {s.name: float(v) for s, v in zip(self.specs, params_vec)}
        return {r.__name__: r(pdict) for r in self.rules}


class ConstrainedAcquisition:
    """Wrap any acquisition function with a feasibility multiplier.

    `feasibility(x)` is a smooth sigmoid of the negative penalty, so infeasible
    candidates receive vanishing acquisition value rather than infinite loss.
    """

    def __init__(self, base_acq: Callable[[np.ndarray], np.ndarray],
                 penalty: PhysicsPenalty,
                 tau: float = 1.0):
        self.base = base_acq
        self.penalty = penalty
        self.tau = tau

    def _feasibility(self, X: np.ndarray) -> np.ndarray:
        vals = np.array([self.penalty(x) for x in X])
        # Clip to avoid overflow when penalty saturates (sigmoid -> 0 anyway).
        return 1.0 / (1.0 + np.exp(np.clip(vals / self.tau, -50.0, 50.0)))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.base(X) * self._feasibility(X)


def stage_physics_weight(stage: int) -> float:
    """Physics weight scheduler (mirrors /CODE/main.py PHYSICS_WEIGHTS)."""
    schedule = {0: 0.3, 1: 0.7, 2: 0.5, 3: 0.2}
    return schedule.get(stage, 0.5)


def projected_into_bounds(x: np.ndarray, specs: List[ParameterSpec]) -> np.ndarray:
    """Clip a parameter vector back inside its physical bounds."""
    lo, hi = bounds_array(specs)
    return np.minimum(np.maximum(x, lo), hi)
