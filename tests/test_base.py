"""Tests for base abstractions."""
import numpy as np
import pytest
import torch

from hecopt.core.base import OptSolution, CombinatorialOptModel, NonLinearOptModel


# ---------------------------------------------------------------------------
# Minimal concrete implementations for testing
# ---------------------------------------------------------------------------


class _TinyCombinatorialModel(CombinatorialOptModel):
    """Trivial 2-variable LP: min c^T x, x in [0,1]^2, sum(x)=1."""

    @property
    def n_vars(self) -> int:
        return 2

    def solve(self, theta):
        # Optimal: assign all flow to cheapest variable
        x = np.zeros(2)
        x[int(theta[1] < theta[0])] = 1.0
        return OptSolution(x_star=x, obj_val=float(min(theta)))


class _TinyNonLinearModel(NonLinearOptModel):
    """Trivial 1-variable NLP: min (x - theta)^2, x in [0, 1]."""

    @property
    def n_vars(self) -> int:
        return 1

    def solve(self, theta):
        x_star = np.clip(theta, 0.0, 1.0)
        return OptSolution(x_star=x_star, obj_val=float((x_star - theta) ** 2))

    def objective_torch(self, x, theta):
        return ((x - theta) ** 2).sum()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_opt_solution_defaults():
    sol = OptSolution(x_star=np.zeros(3), obj_val=0.0)
    assert sol.success
    assert sol.y_star is None
    assert sol.z_star is None


def test_combinatorial_model_is_combinatorial():
    m = _TinyCombinatorialModel()
    assert m.is_combinatorial is True


def test_nonlinear_model_is_combinatorial():
    m = _TinyNonLinearModel()
    assert m.is_combinatorial is False


def test_combinatorial_solve():
    m = _TinyCombinatorialModel()
    sol = m.solve(np.array([0.5, 0.2]))
    assert sol.success
    # Second variable is cheaper → x = [0, 1]
    np.testing.assert_array_equal(sol.x_star, [0.0, 1.0])
    assert sol.obj_val == pytest.approx(0.2)


def test_nonlinear_solve_clipped():
    m = _TinyNonLinearModel()
    sol = m.solve(np.array([1.5]))
    np.testing.assert_allclose(sol.x_star, [1.0], atol=1e-6)


def test_nonlinear_objective_torch():
    m = _TinyNonLinearModel()
    x = torch.tensor([0.5])
    theta = torch.tensor([0.3])
    obj = m.objective_torch(x, theta)
    assert obj.item() == pytest.approx((0.5 - 0.3) ** 2, abs=1e-6)


def test_nonlinear_default_eq_constraints():
    m = _TinyNonLinearModel()
    x = torch.zeros(1)
    theta = torch.zeros(1)
    eq = m.eq_constraints_torch(x, theta)
    assert eq.numel() == 0


def test_nonlinear_default_ineq_constraints():
    m = _TinyNonLinearModel()
    x = torch.zeros(1)
    theta = torch.zeros(1)
    ineq = m.ineq_constraints_torch(x, theta)
    assert ineq.numel() == 0
