"""
NonLinearPtOLayer: Differentiable layer for non-linear programs (NLPs).

Computes exact gradients through the NLP solver by solving the **KKT
adjoint system** (also called implicit differentiation via the IFT).

Mathematical flow
-----------------
Forward:  call the NLP solver to get x*(θ), λ*(θ), μ*(θ).
Backward: given upstream gradient v = ∂L_task/∂x*, build the KKT matrix K,
          solve  Kᵀu = [v; 0; …], and return  ∂L_task/∂θ = −uᵀ (∂F/∂θ).

Tikhonov regularisation (ridge on K's diagonal) prevents rank deficiency
from degenerate active sets or near-singular Hessians.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from hecopt.core.base import NonLinearOptModel
from hecopt.utils.kkt import build_kkt_matrix, solve_adjoint


class NonLinearPtOLayer(nn.Module):
    """Differentiable Predict-then-Optimize layer for non-linear programs.

    Wraps a ``NonLinearOptModel`` and differentiates through the NLP solver
    using KKT implicit differentiation (adjoint sensitivity method).

    Parameters
    ----------
    model:
        A ``NonLinearOptModel`` implementing both ``solve()`` (for the
        forward pass) and ``objective_torch`` / ``ineq_constraints_torch``
        (for the backward pass via KKT).
    tikhonov:
        Ridge regularisation for KKT matrix stability. Default: ``1e-6``.
    reduction:
        Loss reduction (``"mean"`` or ``"sum"``).

    Examples
    --------
    >>> from hecopt.baselines.pricing import PricingModel
    >>> from hecopt import NonLinearPtOLayer
    >>> model = PricingModel(n_products=3)
    >>> layer = NonLinearPtOLayer(model)
    >>> theta_pred = torch.randn(16, model.n_params, requires_grad=True)
    >>> x_star = layer(theta_pred)        # differentiable forward pass
    >>> x_star.sum().backward()           # backprop through KKT
    """

    def __init__(
        self,
        model: NonLinearOptModel,
        tikhonov: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if not isinstance(model, NonLinearOptModel):
            raise TypeError(
                f"model must be a NonLinearOptModel, got {type(model).__name__}"
            )
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")

        self.model = model
        self.tikhonov = tikhonov
        self.reduction = reduction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Solve the NLP for each instance and return x* with KKT gradients.

        Parameters
        ----------
        theta:
            Predicted parameters ``[batch_size, n_params]``.

        Returns
        -------
        torch.Tensor
            Optimal primal variables ``[batch_size, n_vars]``,
            differentiable with respect to ``theta`` via the KKT adjoint.
        """
        return _NonLinearFunction.apply(theta, self.model, self.tikhonov)

    def decision_loss(
        self,
        theta_pred: torch.Tensor,
        theta_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the mean objective regret loss (differentiable).

        Solves the NLP under both predicted and true parameters, then
        computes the mean gap between the two objective values.

        Parameters
        ----------
        theta_pred:
            Predicted parameters ``[batch_size, n_params]``.
        theta_true:
            True parameters ``[batch_size, n_params]``.

        Returns
        -------
        torch.Tensor
            Scalar regret loss (non-negative at optimum).
        """
        # Optimal decisions under predicted parameters (differentiable)
        x_pred = self.forward(theta_pred)

        # Objective values at predicted decisions under TRUE parameters
        batch_size = theta_pred.shape[0]
        obj_under_pred: list[torch.Tensor] = []
        obj_under_true: list[torch.Tensor] = []

        for i in range(batch_size):
            # Objective at predicted decision, true parameters
            obj_under_pred.append(
                self.model.objective_torch(x_pred[i], theta_true[i])
            )
            # Optimal objective under true parameters (non-diff, just a number)
            with torch.no_grad():
                sol_true = self.model.solve(theta_true[i].detach().cpu().numpy())
                x_true_i = torch.as_tensor(
                    sol_true.x_star, dtype=theta_pred.dtype, device=theta_pred.device
                )
            obj_under_true.append(
                self.model.objective_torch(x_true_i, theta_true[i]).detach()
            )

        obj_pred_t = torch.stack(obj_under_pred)   # [B]  (minimisation obj)
        obj_true_t = torch.stack(obj_under_true)   # [B]

        # Regret = obj(x_pred, c_true) − obj(x*, c_true)  ≥ 0
        regret = obj_pred_t - obj_true_t

        if self.reduction == "mean":
            return regret.mean()
        return regret.sum()

    @torch.no_grad()
    def regret(
        self, theta_pred: torch.Tensor, theta_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute the mean normalised objective regret (evaluation metric).

        Parameters
        ----------
        theta_pred:
            Predicted parameters ``[batch_size, n_params]``.
        theta_true:
            True parameters ``[batch_size, n_params]``.

        Returns
        -------
        torch.Tensor
            Scalar mean normalised regret.
        """
        regrets: list[float] = []
        for i in range(theta_pred.shape[0]):
            sol_pred = self.model.solve(theta_pred[i].cpu().numpy())
            sol_true = self.model.solve(theta_true[i].cpu().numpy())

            obj_pred = sol_pred.obj_val
            obj_true = sol_true.obj_val
            denom = abs(obj_true) if abs(obj_true) > 1e-10 else 1.0
            regrets.append((obj_pred - obj_true) / denom)

        return torch.tensor(np.mean(regrets), dtype=theta_pred.dtype)


class _NonLinearFunction(torch.autograd.Function):
    """Custom autograd: forward calls NLP solver; backward uses KKT adjoint."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        theta: torch.Tensor,
        model: NonLinearOptModel,
        tikhonov: float,
    ) -> torch.Tensor:
        batch_size = theta.shape[0]
        dtype = theta.dtype
        device = theta.device

        x_stars, y_stars, z_stars = [], [], []
        active_masks: list[torch.Tensor] = []

        n_eq = model.n_eq_constraints
        n_ineq = model.n_ineq_constraints

        for i in range(batch_size):
            sol = model.solve(theta[i].detach().cpu().numpy())

            x_stars.append(torch.as_tensor(sol.x_star, dtype=dtype, device=device))
            y_stars.append(
                torch.as_tensor(
                    sol.y_star if sol.y_star is not None else np.zeros(n_eq),
                    dtype=dtype,
                    device=device,
                )
            )
            z_i_np = sol.z_star if sol.z_star is not None else np.zeros(n_ineq)
            z_stars.append(torch.as_tensor(z_i_np, dtype=dtype, device=device))

            if sol.active_ineq is not None:
                active_masks.append(
                    torch.as_tensor(sol.active_ineq, dtype=torch.bool, device=device)
                )
            else:
                active_masks.append(z_stars[-1] > 1e-6)

        X = torch.stack(x_stars)
        Y = torch.stack(y_stars)
        Z = torch.stack(z_stars)

        ctx.save_for_backward(X, Y, Z, theta)
        ctx.active_masks = active_masks
        ctx.model = model
        ctx.tikhonov = tikhonov
        ctx.batch_size = batch_size

        return X

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_x: torch.Tensor,
    ):
        X, Y, Z, theta = ctx.saved_tensors
        model = ctx.model
        tikhonov = ctx.tikhonov
        batch_size = ctx.batch_size

        grad_theta = torch.zeros_like(theta)

        for i in range(batch_size):
            try:
                K, dF_dtheta = build_kkt_matrix(
                    objective_fn=model.objective_torch,
                    eq_fn=model.eq_constraints_torch,
                    ineq_fn=model.ineq_constraints_torch,
                    x_star=X[i],
                    theta=theta[i],
                    lambda_eq=Y[i],
                    mu_ineq=Z[i],
                    active_mask=ctx.active_masks[i],
                    tikhonov=tikhonov,
                )
                grad_theta[i] = solve_adjoint(K, dF_dtheta, grad_x[i], tikhonov)
            except Exception as exc:
                warnings.warn(
                    f"KKT backward failed for sample {i}: {exc}. "
                    "Gradient for this sample set to zero."
                )

        return grad_theta, None, None
