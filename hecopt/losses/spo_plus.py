"""
SPO+ (Smart Predict-and-Optimize Plus) loss function.

Reference
---------
Elmachtoub, A. N., & Grigas, P. (2022). Smart predict-then-optimize.
Operations Research, 70(1), 102–119. https://doi.org/10.1287/opre.2021.2190

Mathematical background
-----------------------
For a combinatorial problem with feasible set W and a predicted cost vector
θ̂, the optimal decision is  w*(θ̂) = argmin_{w ∈ W} θ̂ᵀ w.

The **decision regret** of predicting θ̂ when the true cost is c is:

    regret(θ̂, c) = cᵀ w*(θ̂) − cᵀ w*(c)    ≥ 0

The **SPO+ loss** is a convex surrogate upper bound:

    ℓ_SPO+(θ̂, c) = cᵀ w*(2θ̂ − c) − 2 θ̂ᵀ w*(θ̂) + cᵀ w*(θ̂)

Its subgradient w.r.t. θ̂ is:

    ∂ℓ_SPO+ / ∂θ̂  =  2 ( w*(2θ̂ − c) − w*(θ̂) )

This gradient is non-zero whenever the decision under the "SPO cost"
(2θ̂ − c) differs from the decision under the predicted cost, driving the
model to correct its predictions in decision-relevant directions.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn


class SPOPlusLoss(nn.Module):
    """SPO+ surrogate loss for combinatorial Predict-then-Optimize.

    Parameters
    ----------
    solver_fn:
        A callable ``(cost: np.ndarray) → solution: np.ndarray`` that returns
        the optimal solution ``w*(cost) = argmin_{w ∈ W} cost^T w``.
    reduction:
        ``"mean"`` (default) or ``"sum"`` over the batch.
    """

    def __init__(self, solver_fn: Callable, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")
        self.solver_fn = solver_fn
        self.reduction = reduction

    def forward(
        self, theta_pred: torch.Tensor, c_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute the SPO+ loss.

        Parameters
        ----------
        theta_pred:
            Predicted cost parameters ``[batch_size, n_vars]``.
            Must have ``requires_grad=True`` for backpropagation.
        c_true:
            True cost parameters ``[batch_size, n_vars]``.
            Does not need to track gradients.

        Returns
        -------
        torch.Tensor
            Scalar SPO+ loss with valid subgradients.
        """
        return _SPOPlusFunction.apply(
            theta_pred, c_true, self.solver_fn, self.reduction
        )


class _SPOPlusFunction(torch.autograd.Function):
    """Custom autograd function: calls solver in forward, returns SPO+ subgradient."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        theta_pred: torch.Tensor,
        c_true: torch.Tensor,
        solver_fn: Callable,
        reduction: str,
    ) -> torch.Tensor:
        batch_size = theta_pred.shape[0]
        dtype = theta_pred.dtype
        device = theta_pred.device

        w_pred_list: list[torch.Tensor] = []
        w_spo_list: list[torch.Tensor] = []

        for i in range(batch_size):
            theta_i = theta_pred[i].detach().cpu().numpy().astype(np.float64)
            c_i = c_true[i].detach().cpu().numpy().astype(np.float64)

            # Decision under predicted cost
            w_pred_i = solver_fn(theta_i)
            # Decision under SPO cost  2θ̂ − c
            w_spo_i = solver_fn(2.0 * theta_i - c_i)

            w_pred_list.append(torch.as_tensor(w_pred_i, dtype=dtype, device=device))
            w_spo_list.append(torch.as_tensor(w_spo_i, dtype=dtype, device=device))

        w_pred = torch.stack(w_pred_list)  # [B, n_vars]
        w_spo = torch.stack(w_spo_list)    # [B, n_vars]

        ctx.save_for_backward(w_pred, w_spo)
        ctx.reduction = reduction
        ctx.batch_size = batch_size

        # ℓ_SPO+(θ̂, c) = cᵀ w_spo − 2θ̂ᵀ w_pred + cᵀ w_pred
        loss_per_sample = (
            (c_true * w_spo).sum(dim=-1)
            - 2.0 * (theta_pred * w_pred).sum(dim=-1)
            + (c_true * w_pred).sum(dim=-1)
        )

        if reduction == "mean":
            return loss_per_sample.mean()
        return loss_per_sample.sum()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ):
        w_pred, w_spo = ctx.saved_tensors

        # Subgradient: 2(w_spo − w_pred)
        grad_theta = 2.0 * (w_spo - w_pred)

        if ctx.reduction == "mean":
            grad_theta = grad_theta / ctx.batch_size

        # Upstream scalar gradient broadcast to [B, n_vars]
        return grad_theta * grad_output, None, None, None
