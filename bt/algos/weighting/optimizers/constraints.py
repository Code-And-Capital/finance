"""Constraint builders for optimization problems."""

from __future__ import annotations

from typing import Any

import cvxpy as cvx


def sum_to_one_constraint(weights: cvx.Variable):
    """Return CVXPY equality constraint enforcing ``sum(weights)=1``."""
    return cvx.sum(weights) == 1.0


def non_negative_constraint(weights: cvx.Variable):
    """Return CVXPY inequality constraint enforcing ``weights >= 0``."""
    return weights >= 0.0


def bound_constraints(
    weights: cvx.Variable,
    min_weights,
    max_weights,
) -> list[Any]:
    """Return per-asset lower/upper bound constraints."""
    return [weights >= min_weights, weights <= max_weights]
