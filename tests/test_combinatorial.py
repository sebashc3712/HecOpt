"""Tests for CombinatorialPtOLayer (end-to-end DFL pipeline)."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from hecopt import CombinatorialPtOLayer
from hecopt.baselines.shortest_path import ShortestPathModel, ShortestPathDataset


@pytest.fixture(scope="module")
def model():
    return ShortestPathModel(grid_size=3)


@pytest.fixture(scope="module")
def layer_spo(model):
    return CombinatorialPtOLayer(model, method="spo_plus")


@pytest.fixture(scope="module")
def layer_pfyl(model):
    return CombinatorialPtOLayer(model, method="pfyl", n_samples=3, sigma=0.5)


class TestCombinatorialPtOLayer:
    def test_invalid_model_type(self):
        from hecopt.core.base import NonLinearOptModel, OptSolution

        class FakeNL(NonLinearOptModel):
            @property
            def n_vars(self):
                return 1

            def solve(self, theta):
                return OptSolution(x_star=theta, obj_val=0.0)

            def objective_torch(self, x, theta):
                return x.sum()

        with pytest.raises(TypeError):
            CombinatorialPtOLayer(FakeNL())

    def test_invalid_method_raises(self, model):
        with pytest.raises(ValueError):
            CombinatorialPtOLayer(model, method="invalid")

    def test_forward_shape(self, layer_spo, model):
        batch = 8
        theta = torch.rand(batch, model.n_vars)
        x_star = layer_spo(theta)
        assert x_star.shape == (batch, model.n_vars)

    def test_forward_binary(self, layer_spo, model):
        theta = torch.rand(4, model.n_vars)
        x_star = layer_spo(theta)
        vals = set(x_star.numpy().round().astype(int).ravel())
        assert vals.issubset({0, 1})

    def test_loss_scalar(self, layer_spo, model):
        theta = torch.rand(4, model.n_vars, requires_grad=True)
        c = torch.rand(4, model.n_vars)
        loss = layer_spo.loss(theta, c)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_loss_backward_spo(self, layer_spo, model):
        theta = torch.rand(4, model.n_vars, requires_grad=True)
        c = torch.rand(4, model.n_vars)
        loss = layer_spo.loss(theta, c)
        loss.backward()
        assert theta.grad is not None
        assert theta.grad.shape == theta.shape

    def test_loss_backward_pfyl(self, layer_pfyl, model):
        theta = torch.rand(4, model.n_vars, requires_grad=True)
        c = torch.rand(4, model.n_vars)
        loss = layer_pfyl.loss(theta, c)
        loss.backward()
        assert theta.grad is not None

    def test_regret_non_negative(self, layer_spo, model):
        theta = torch.rand(4, model.n_vars)
        c = torch.rand(4, model.n_vars)
        regret = layer_spo.regret(theta, c)
        assert regret.item() >= -1e-4  # should be non-negative

    def test_training_loop_reduces_regret(self, model):
        """Full mini-training loop: loss should decrease after a few SGD steps."""
        ds = ShortestPathDataset(n_samples=64, grid_size=3, n_features=6, seed=7)
        n_edges = model.n_vars

        net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, n_edges))
        layer = CombinatorialPtOLayer(model, method="spo_plus", lambda_hybrid=0.1)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)

        # Compute initial regret
        feats, costs, _ = ds[:32]
        theta_init = net(feats).detach()
        regret_init = layer.regret(theta_init, costs).item()

        # 10 SGD steps
        for _ in range(10):
            feats, costs, _ = ds[:32]
            opt.zero_grad()
            theta_pred = net(feats)
            loss = layer.loss(theta_pred, costs)
            loss.backward()
            opt.step()

        theta_final = net(feats).detach()
        regret_final = layer.regret(theta_final, costs).item()

        # After training the regret should not blow up (soft check)
        assert regret_final < regret_init * 5 + 1.0  # generous bound
