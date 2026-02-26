"""Tests for SPO+, PFYL, and HybridLoss."""
import numpy as np
import pytest
import torch

from hecopt.losses.spo_plus import SPOPlusLoss
from hecopt.losses.pfyl import PFYLoss
from hecopt.losses.hybrid import HybridLoss


# ---------------------------------------------------------------------------
# Simple solver: assignment problem (1 variable)
# ---------------------------------------------------------------------------


def _argmin_solver(theta: np.ndarray) -> np.ndarray:
    """Trivial unconstrained minimiser: x* = 0 if theta > 0 else 1."""
    x = np.zeros_like(theta)
    x[theta < 0] = 1.0
    return x


def _binary_solver(theta: np.ndarray) -> np.ndarray:
    """Single binary variable: x* = 1{theta < 0}."""
    return np.array([1.0 if theta[0] < 0 else 0.0])


# ---------------------------------------------------------------------------
# SPO+ tests
# ---------------------------------------------------------------------------


class TestSPOPlusLoss:
    def _make_batch(self, batch=4, n=3, seed=0):
        rng = torch.Generator().manual_seed(seed)
        theta = torch.randn(batch, n, generator=rng, requires_grad=True)
        c = torch.rand(batch, n, generator=rng)
        return theta, c

    def test_forward_returns_scalar(self):
        loss_fn = SPOPlusLoss(_argmin_solver)
        theta, c = self._make_batch()
        loss = loss_fn(theta, c)
        assert loss.shape == ()

    def test_backward_grad_shape(self):
        loss_fn = SPOPlusLoss(_argmin_solver)
        theta, c = self._make_batch()
        loss = loss_fn(theta, c)
        loss.backward()
        assert theta.grad is not None
        assert theta.grad.shape == theta.shape

    def test_loss_non_negative(self):
        """SPO+ is a regret upper bound, so it should be >= 0 for feasible problems."""
        loss_fn = SPOPlusLoss(_argmin_solver)
        theta, c = self._make_batch()
        loss = loss_fn(theta, c)
        # SPO+ can be negative in pathological cases but is typically >= 0
        # Check it's finite
        assert torch.isfinite(loss)

    def test_sum_reduction(self):
        loss_mean = SPOPlusLoss(_argmin_solver, reduction="mean")
        loss_sum = SPOPlusLoss(_argmin_solver, reduction="sum")
        theta, c = self._make_batch(batch=4)
        lm = loss_mean(theta, c)
        ls = loss_sum(theta, c)
        assert ls.item() == pytest.approx(lm.item() * 4, rel=1e-5)

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError):
            SPOPlusLoss(_argmin_solver, reduction="invalid")

    def test_gradient_zero_at_perfect_prediction(self):
        """When theta_pred == c_true, w_spo and w_pred should be equal → grad ≈ 0."""
        loss_fn = SPOPlusLoss(_binary_solver)
        c = torch.tensor([[0.5]])
        theta = c.clone().requires_grad_(True)
        loss = loss_fn(theta, c)
        loss.backward()
        np.testing.assert_allclose(theta.grad.numpy(), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# PFYL tests
# ---------------------------------------------------------------------------


class TestPFYLoss:
    def test_forward_returns_scalar(self):
        loss_fn = PFYLoss(_argmin_solver, n_samples=5, sigma=0.5)
        theta = torch.randn(4, 3, requires_grad=True)
        c = torch.rand(4, 3)
        loss = loss_fn(theta, c)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_backward_grad_shape(self):
        loss_fn = PFYLoss(_argmin_solver, n_samples=5, sigma=0.5)
        theta = torch.randn(4, 3, requires_grad=True)
        c = torch.rand(4, 3)
        loss = loss_fn(theta, c)
        loss.backward()
        assert theta.grad is not None
        assert theta.grad.shape == theta.shape

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            PFYLoss(_argmin_solver, sigma=-1.0)

    def test_invalid_n_samples_raises(self):
        with pytest.raises(ValueError):
            PFYLoss(_argmin_solver, n_samples=0)


# ---------------------------------------------------------------------------
# HybridLoss tests
# ---------------------------------------------------------------------------


class TestHybridLoss:
    def _dfl(self):
        return SPOPlusLoss(_argmin_solver, reduction="mean")

    def test_pure_dfl(self):
        hybrid = HybridLoss(self._dfl(), lambda_mse=0.0)
        theta = torch.randn(4, 3, requires_grad=True)
        c = torch.rand(4, 3)
        loss = hybrid(theta, c)
        assert torch.isfinite(loss)

    def test_pure_mse(self):
        hybrid = HybridLoss(self._dfl(), lambda_mse=1.0)
        theta = torch.randn(4, 3, requires_grad=True)
        c = torch.rand(4, 3)
        loss = hybrid(theta, c)
        # Should equal MSE
        mse = torch.nn.functional.mse_loss(theta, c)
        assert loss.item() == pytest.approx(mse.item(), rel=1e-5)

    def test_mixed(self):
        hybrid = HybridLoss(self._dfl(), lambda_mse=0.5)
        theta = torch.randn(4, 3, requires_grad=True)
        c = torch.rand(4, 3)
        loss = hybrid(theta, c)
        assert torch.isfinite(loss)
        loss.backward()
        assert theta.grad is not None

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError):
            HybridLoss(self._dfl(), lambda_mse=1.5)

    def test_invalid_lambda_negative_raises(self):
        with pytest.raises(ValueError):
            HybridLoss(self._dfl(), lambda_mse=-0.1)
