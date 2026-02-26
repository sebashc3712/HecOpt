"""Tests for the two baseline models and their datasets."""
import numpy as np
import pytest
import torch

from hecopt.baselines.shortest_path import ShortestPathModel, ShortestPathDataset
from hecopt.baselines.pricing import PricingModel, PricingDataset


# ---------------------------------------------------------------------------
# ShortestPathModel
# ---------------------------------------------------------------------------


class TestShortestPathModel:
    def test_n_vars_2x2(self):
        m = ShortestPathModel(grid_size=2)
        # 2x2 grid: 2 right edges (row 0: 0→1, row 1: 2→3) +
        #           2 down edges  (col 0: 0→2, col 1: 1→3) = 4 edges
        assert m.n_vars == 4

    def test_n_vars_5x5(self):
        m = ShortestPathModel(grid_size=5)
        # 2 * 5 * (5-1) = 40
        assert m.n_vars == 40

    def test_solve_returns_binary(self):
        m = ShortestPathModel(grid_size=3)
        theta = np.random.rand(m.n_vars)
        sol = m.solve(theta)
        assert sol.success
        assert set(np.unique(sol.x_star)).issubset({0.0, 1.0})

    def test_solve_valid_path_length(self):
        """A path from (0,0) to (m-1,m-1) on an m×m grid uses exactly 2(m-1) edges."""
        for g in [2, 3, 5]:
            m = ShortestPathModel(grid_size=g)
            theta = np.ones(m.n_vars)
            sol = m.solve(theta)
            assert int(sol.x_star.sum()) == 2 * (g - 1), f"grid={g}"

    def test_invalid_grid_size(self):
        with pytest.raises(ValueError):
            ShortestPathModel(grid_size=1)

    def test_path_cost(self):
        m = ShortestPathModel(grid_size=2)
        x = np.array([1.0, 0.0, 1.0, 0.0])
        c = np.array([2.0, 5.0, 3.0, 7.0])
        assert m.path_cost(x, c) == pytest.approx(5.0)

    def test_is_combinatorial(self):
        assert ShortestPathModel().is_combinatorial is True


class TestShortestPathDataset:
    def test_length(self):
        ds = ShortestPathDataset(n_samples=50, grid_size=3, seed=0)
        assert len(ds) == 50

    def test_item_shapes(self):
        ds = ShortestPathDataset(n_samples=10, grid_size=3, n_features=4, seed=1)
        feat, cost, sol = ds[0]
        m = ds.model
        assert feat.shape == (4,)
        assert cost.shape == (m.n_vars,)
        assert sol.shape == (m.n_vars,)

    def test_solutions_are_binary(self):
        ds = ShortestPathDataset(n_samples=20, grid_size=3, seed=2)
        for i in range(len(ds)):
            _, _, sol = ds[i]
            assert set(sol.numpy().round(0).astype(int)).issubset({0, 1})

    def test_nonlinear_degree(self):
        ds = ShortestPathDataset(n_samples=10, grid_size=3, degree=2, seed=3)
        feat, cost, _ = ds[0]
        assert feat.shape[0] == 4  # default n_features


# ---------------------------------------------------------------------------
# PricingModel
# ---------------------------------------------------------------------------


class TestPricingModel:
    def test_n_vars(self):
        m = PricingModel(n_products=4)
        assert m.n_vars == 4

    def test_n_params(self):
        m = PricingModel(n_products=4)
        assert m.n_params == 8

    def test_solve_prices_in_bounds(self):
        m = PricingModel(n_products=3, p_min=1.0, p_max=10.0)
        theta = np.array([2.0, 3.0, 1.5, 1.2, 1.8, 2.0])
        sol = m.solve(theta)
        assert sol.success or sol.x_star is not None
        assert np.all(sol.x_star >= m.p_min - 1e-4)
        assert np.all(sol.x_star <= m.p_max + 1e-4)

    def test_solve_obj_positive(self):
        m = PricingModel(n_products=2)
        theta = np.array([5.0, 4.0, 1.5, 2.0])
        sol = m.solve(theta)
        # Revenue should be positive for reasonable parameters
        assert sol.obj_val >= 0.0

    def test_is_combinatorial(self):
        assert PricingModel().is_combinatorial is False

    def test_n_ineq_constraints(self):
        assert PricingModel().n_ineq_constraints == 1

    def test_n_eq_constraints(self):
        assert PricingModel().n_eq_constraints == 0

    def test_objective_torch_differentiable(self):
        m = PricingModel(n_products=2)
        x = torch.tensor([3.0, 4.0], requires_grad=True)
        theta = torch.tensor([2.0, 3.0, 1.5, 2.0])
        obj = m.objective_torch(x, theta)
        assert obj.shape == ()
        obj.backward()
        assert x.grad is not None

    def test_ineq_constraints_torch_shape(self):
        m = PricingModel(n_products=2)
        x = torch.tensor([3.0, 4.0])
        theta = torch.tensor([2.0, 3.0, 1.5, 2.0])
        g = m.ineq_constraints_torch(x, theta)
        assert g.shape == (1,)


class TestPricingDataset:
    def test_length(self):
        ds = PricingDataset(n_samples=30, n_products=2, seed=0)
        assert len(ds) == 30

    def test_item_shapes(self):
        ds = PricingDataset(n_samples=10, n_products=2, n_features=4, seed=1)
        feat, theta_true, theta_noisy, sol, obj = ds[0]
        assert feat.shape == (4,)
        assert theta_true.shape == (4,)
        assert theta_noisy.shape == (4,)
        assert sol.shape == (2,)
        assert obj.shape == ()

    def test_obj_vals_positive(self):
        ds = PricingDataset(n_samples=20, n_products=2, seed=2)
        for i in range(len(ds)):
            *_, obj = ds[i]
            assert obj.item() >= -1e-4  # revenue ≥ 0
