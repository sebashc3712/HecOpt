"""Tests for NonLinearPtOLayer (end-to-end DFL with KKT differentiation)."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from hecopt import NonLinearPtOLayer
from hecopt.baselines.pricing import PricingModel, PricingDataset


@pytest.fixture(scope="module")
def model():
    return PricingModel(n_products=2, p_min=1.0, p_max=8.0, capacity=30.0)


@pytest.fixture(scope="module")
def layer(model):
    return NonLinearPtOLayer(model, tikhonov=1e-5)


class TestNonLinearPtOLayer:
    def test_invalid_model_type(self):
        from hecopt.core.base import CombinatorialOptModel, OptSolution

        class FakeCombo(CombinatorialOptModel):
            @property
            def n_vars(self):
                return 1

            def solve(self, theta):
                return OptSolution(x_star=theta, obj_val=0.0)

        with pytest.raises(TypeError):
            NonLinearPtOLayer(FakeCombo())

    def test_forward_shape(self, layer, model):
        batch = 6
        theta = torch.rand(batch, model.n_params) + 0.5
        x_star = layer(theta)
        assert x_star.shape == (batch, model.n_vars)

    def test_forward_prices_in_bounds(self, layer, model):
        theta = torch.rand(4, model.n_params) + 1.0
        x_star = layer(theta)
        assert torch.all(x_star >= model.p_min - 1e-3)
        assert torch.all(x_star <= model.p_max + 1e-3)

    def test_backward_gradient_flows(self, layer, model):
        # Create as leaf tensor (not via arithmetic) so .grad is populated
        theta = torch.rand(3, model.n_params) + 1.0
        theta = theta.detach().requires_grad_(True)          # leaf tensor
        x_star = layer(theta)
        x_star.sum().backward()
        assert theta.grad is not None
        assert theta.grad.shape == theta.shape
        assert torch.isfinite(theta.grad).all()

    def test_decision_loss_scalar(self, layer, model):
        theta_pred = (torch.rand(4, model.n_params) + 1.0).requires_grad_(True)
        theta_true = torch.rand(4, model.n_params) + 1.0
        loss = layer.decision_loss(theta_pred, theta_true)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_decision_loss_backward(self, layer, model):
        theta_pred = (torch.rand(4, model.n_params) + 1.0).requires_grad_(True)
        theta_true = torch.rand(4, model.n_params) + 1.0
        loss = layer.decision_loss(theta_pred, theta_true)
        loss.backward()
        assert theta_pred.grad is not None

    def test_regret_non_negative(self, layer, model):
        theta_pred = torch.rand(4, model.n_params) + 1.0
        theta_true = torch.rand(4, model.n_params) + 1.0
        regret = layer.regret(theta_pred, theta_true)
        # NLP solver may find marginally different optima; allow small tolerance
        assert regret.item() >= -0.1

    def test_training_loop_runs(self, model):
        """Mini training loop should complete without errors."""
        ds = PricingDataset(n_samples=40, n_products=2, n_features=4, seed=99)
        layer = NonLinearPtOLayer(model, tikhonov=1e-5)

        net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, model.n_params))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        for _ in range(3):
            feats, theta_true, _, _, _ = ds[:8]
            opt.zero_grad()
            theta_pred = net(feats)
            loss = layer.decision_loss(theta_pred, theta_true)
            loss.backward()
            opt.step()

        assert True  # No exceptions raised
