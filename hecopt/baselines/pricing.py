"""
Baseline 2 (Non-Linear/Continuous): Multi-Product Pricing Optimisation.

This is the canonical benchmark for non-linear Decision-Focused Learning,
representing the revenue management problem common in retail and e-commerce.

Problem description
-------------------
A firm sells n products and must set prices p ∈ ℝⁿ to maximise profit
subject to a total demand capacity constraint:

    max_{p}  Σᵢ (pᵢ − cᵢ) · dᵢ(pᵢ; aᵢ, εᵢ)

    s.t.     p_min ≤ pᵢ ≤ p_max       ∀ i
             Σᵢ dᵢ(pᵢ; aᵢ, εᵢ) ≤ Q   (capacity)

where the demand model is a power-law (constant-elasticity) function:

    dᵢ(pᵢ; aᵢ, εᵢ) = aᵢ · pᵢ^(−εᵢ)

Parameters to predict:  θ = [a₁, …, aₙ, ε₁, …, εₙ]
Decision variables:     x = [p₁, …, pₙ]   (prices)

This is a **non-convex** NLP (the product of a power-law demand and a linear
margin makes the objective non-concave in general), so it is an ideal
stress test for the KKT implicit differentiation backend.

Synthetic dataset
-----------------
``PricingDataset`` generates (market features → demand params → optimal prices)
triples, simulating a setting where a neural network must estimate demand
elasticities from observable market signals before the optimiser computes
the best pricing strategy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.optimize import minimize

from hecopt.core.base import NonLinearOptModel, OptSolution


class PricingModel(NonLinearOptModel):
    """Multi-product pricing with constant-elasticity demand and capacity constraint.

    Parameters
    ----------
    n_products:
        Number of products.  Default: ``3``.
    p_min:
        Minimum price for all products.  Default: ``1.0``.
    p_max:
        Maximum price for all products.  Default: ``10.0``.
    capacity:
        Upper bound on total demand  Σᵢ dᵢ.  Default: ``50.0``.
    marginal_costs:
        Per-unit production costs ``[n_products]``.  Default: zeros.
    """

    def __init__(
        self,
        n_products: int = 3,
        p_min: float = 1.0,
        p_max: float = 10.0,
        capacity: float = 50.0,
        marginal_costs: Optional[np.ndarray] = None,
    ) -> None:
        self.n_products = n_products
        self.p_min = p_min
        self.p_max = p_max
        self.capacity = capacity
        self.marginal_costs = (
            marginal_costs.copy() if marginal_costs is not None
            else np.zeros(n_products, dtype=np.float64)
        )

    # ------------------------------------------------------------------
    # BaseOptModel interface
    # ------------------------------------------------------------------

    @property
    def n_vars(self) -> int:
        return self.n_products

    @property
    def n_params(self) -> int:
        """Length of the parameter vector θ = [a₁,…,aₙ, ε₁,…,εₙ]."""
        return 2 * self.n_products

    @property
    def n_eq_constraints(self) -> int:
        return 0

    @property
    def n_ineq_constraints(self) -> int:
        return 1  # capacity constraint: Σ dᵢ ≤ Q

    # ------------------------------------------------------------------
    # NumPy solver (used in forward pass)
    # ------------------------------------------------------------------

    def _demand_np(
        self, p: np.ndarray, a: np.ndarray, eps: np.ndarray
    ) -> np.ndarray:
        """Power-law demand: dᵢ = aᵢ · max(pᵢ, ε)^(−εᵢ)."""
        return a * np.power(np.maximum(p, 1e-6), -eps)

    def solve(self, theta: np.ndarray) -> OptSolution:
        """Solve the pricing NLP with SLSQP.

        Parameters
        ----------
        theta:
            Demand parameters ``[a₁,…,aₙ, ε₁,…,εₙ]``.

        Returns
        -------
        OptSolution
            Optimal prices, objective value (revenue), and capacity dual.
        """
        n = self.n_products
        # Ensure positive domain
        a = np.abs(theta[:n]) + 1e-3
        eps = np.abs(theta[n:]) + 0.5
        mc = self.marginal_costs

        def neg_revenue(p: np.ndarray) -> float:
            d = self._demand_np(p, a, eps)
            return -float(np.sum((p - mc) * d))

        def neg_revenue_grad(p: np.ndarray) -> np.ndarray:
            d = self._demand_np(p, a, eps)
            # ∂rev/∂pᵢ = dᵢ + (pᵢ − cᵢ)(−εᵢ/pᵢ)dᵢ
            return -(d + (p - mc) * (-eps / np.maximum(p, 1e-6)) * d)

        def capacity_con(p: np.ndarray) -> float:
            return float(self.capacity - self._demand_np(p, a, eps).sum())

        p0 = np.full(n, 0.5 * (self.p_min + self.p_max))
        bounds = [(self.p_min, self.p_max)] * n
        constraints = [{"type": "ineq", "fun": capacity_con}]

        result = minimize(
            neg_revenue,
            p0,
            jac=neg_revenue_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )

        if not result.success:
            # Fall back to mid-point if solver fails
            p_fb = p0.copy()
            return OptSolution(
                x_star=p_fb,
                obj_val=-neg_revenue(p_fb),
                z_star=np.zeros(1),
                active_ineq=np.array([False]),
                success=False,
                message=result.message,
            )

        p_star = result.x
        obj_val = -result.fun

        # Dual multiplier for the capacity constraint
        # SciPy SLSQP stores Lagrange multipliers in result.v (inequality)
        if hasattr(result, "v") and len(result.v) >= 1:
            z_star = np.array([abs(result.v[0])])
        else:
            z_star = np.zeros(1)

        # Active constraint: capacity is tight?
        demand_total = self._demand_np(p_star, a, eps).sum()
        active = np.array([demand_total >= self.capacity - 1e-4])

        return OptSolution(
            x_star=p_star,
            obj_val=obj_val,
            z_star=z_star,
            active_ineq=active,
            success=True,
        )

    # ------------------------------------------------------------------
    # PyTorch-differentiable counterparts (used in backward pass)
    # ------------------------------------------------------------------

    def objective_torch(
        self, x: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """Negative revenue − Σᵢ (pᵢ − cᵢ) dᵢ(pᵢ, θ)  (minimisation form)."""
        n = self.n_products
        a = torch.abs(theta[:n]) + 1e-3
        eps = torch.abs(theta[n:]) + 0.5
        mc = torch.as_tensor(self.marginal_costs, dtype=x.dtype, device=x.device)
        p = torch.clamp(x, min=1e-4)
        d = a * torch.pow(p, -eps)
        return -((p - mc) * d).sum()

    def ineq_constraints_torch(
        self, x: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """Capacity constraint:  Σᵢ dᵢ − Q ≤ 0."""
        n = self.n_products
        a = torch.abs(theta[:n]) + 1e-3
        eps = torch.abs(theta[n:]) + 0.5
        p = torch.clamp(x, min=1e-4)
        d = a * torch.pow(p, -eps)
        return (d.sum() - self.capacity).unsqueeze(0)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


class PricingDataset(torch.utils.data.Dataset):
    """Synthetic (market features → demand parameters → optimal prices) dataset.

    Parameters
    ----------
    n_samples:
        Number of instances.
    n_products:
        Number of products (same as ``PricingModel.n_products``).
    n_features:
        Dimension of the input market feature vector.
    noise_std:
        Gaussian noise added to the true demand parameters to simulate
        imperfect observations during neural network training.
    seed:
        Random seed.
    **pricing_kwargs:
        Additional keyword arguments forwarded to ``PricingModel``.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_products: int = 3,
        n_features: int = 6,
        noise_std: float = 0.05,
        seed: Optional[int] = 42,
        **pricing_kwargs,
    ) -> None:
        rng = np.random.default_rng(seed)
        n_params = 2 * n_products  # [a, ε]

        # Linear map: features → raw demand parameters
        B = rng.standard_normal((n_params, n_features))
        features = rng.uniform(-1.0, 1.0, (n_samples, n_features))
        theta_raw = features @ B.T  # [n_samples, n_params]

        # Rescale to meaningful ranges: a ∈ [1, 5], ε ∈ [1, 3]
        a_true = np.abs(theta_raw[:, :n_products]) * 2.0 + 1.0
        eps_true = np.abs(theta_raw[:, n_products:]) * 1.0 + 1.0
        theta_true = np.concatenate([a_true, eps_true], axis=1)

        # Noisy observations used as neural-network targets during training
        theta_noisy = theta_true + noise_std * rng.standard_normal(theta_true.shape)

        self.opt_model = PricingModel(n_products=n_products, **pricing_kwargs)

        # Pre-compute optimal prices and revenues for each true parameter
        solutions, obj_vals = [], []
        for i in range(n_samples):
            sol = self.opt_model.solve(theta_true[i])
            solutions.append(sol.x_star)
            obj_vals.append(sol.obj_val)

        self.features = torch.tensor(features, dtype=torch.float32)
        self.theta_true = torch.tensor(theta_true, dtype=torch.float32)
        self.theta_noisy = torch.tensor(theta_noisy, dtype=torch.float32)
        self.solutions = torch.tensor(np.array(solutions), dtype=torch.float32)
        self.obj_vals = torch.tensor(obj_vals, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        """Return (features, true_theta, noisy_theta, optimal_prices, optimal_revenue)."""
        return (
            self.features[idx],
            self.theta_true[idx],
            self.theta_noisy[idx],
            self.solutions[idx],
            self.obj_vals[idx],
        )
