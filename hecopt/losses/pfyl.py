"""
Perturbed Fenchel-Young Loss (PFYL) for combinatorial optimization.

Reference
---------
Berthet, Q., Blondel, M., Teboul, O., Cuturi, M., Vert, J.-P., & Bach, F.
(2020). Learning with differentiable perturbed optimizers.
Advances in Neural Information Processing Systems (NeurIPS), 33.

Mathematical background
-----------------------
The PFYL smooths the combinatorial map  θ → w*(θ)  via additive noise:

    ŵ(θ, σ) = 𝔼_{ε ~ N(0,I)}[ w*(θ + σε) ]

The loss is:

    ℓ_PFYL(θ̂, c) = cᵀ ŵ(θ̂, σ) − cᵀ w*(c)

Its gradient (approximated via Monte Carlo) is:

    ∂ℓ_PFYL / ∂θ̂  ≈  ŵ(θ̂, σ) − w*(c)

where the expectation is estimated from ``n_samples`` solver calls with
independent Gaussian noise realisations.  Higher ``n_samples`` gives a
lower-variance gradient estimate at the cost of extra solver calls.
The parameter ``sigma`` controls the smoothing level: larger σ gives
smoother (but more biased) gradients.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn


class PFYLoss(nn.Module):
    """Perturbed Fenchel-Young Loss for combinatorial Predict-then-Optimize.

    Parameters
    ----------
    solver_fn:
        A callable ``(cost: np.ndarray) → solution: np.ndarray``.
    n_samples:
        Number of Gaussian perturbation samples for the Monte Carlo estimate
        of ŵ(θ̂, σ). Default: ``10``.
    sigma:
        Standard deviation of the additive Gaussian noise. Default: ``1.0``.
    reduction:
        ``"mean"`` (default) or ``"sum"`` over the batch.
    """

    def __init__(
        self,
        solver_fn: Callable,
        n_samples: int = 10,
        sigma: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if n_samples < 1:
            raise ValueError("n_samples must be ≥ 1")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")

        self.solver_fn = solver_fn
        self.n_samples = n_samples
        self.sigma = sigma
        self.reduction = reduction

    def forward(
        self, theta_pred: torch.Tensor, c_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute the PFYL loss.

        Parameters
        ----------
        theta_pred:
            Predicted cost parameters ``[batch_size, n_vars]``.
        c_true:
            True cost parameters ``[batch_size, n_vars]``.

        Returns
        -------
        torch.Tensor
            Scalar PFYL loss with Monte Carlo gradient estimates.
        """
        return _PFYLFunction.apply(
            theta_pred,
            c_true,
            self.solver_fn,
            self.n_samples,
            self.sigma,
            self.reduction,
        )


class _PFYLFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        theta_pred: torch.Tensor,
        c_true: torch.Tensor,
        solver_fn: Callable,
        n_samples: int,
        sigma: float,
        reduction: str,
    ) -> torch.Tensor:
        batch_size, n_vars = theta_pred.shape
        dtype = theta_pred.dtype
        device = theta_pred.device

        w_mean_list: list[torch.Tensor] = []
        w_true_list: list[torch.Tensor] = []

        rng = np.random.default_rng()  # Thread-safe local RNG

        for i in range(batch_size):
            theta_i = theta_pred[i].detach().cpu().numpy().astype(np.float64)
            c_i = c_true[i].detach().cpu().numpy().astype(np.float64)

            # Monte Carlo estimate of  ŵ(θ̂, σ) = E[w*(θ̂ + σε)]
            perturbed_sols = np.stack(
                [solver_fn(theta_i + sigma * rng.standard_normal(n_vars))
                 for _ in range(n_samples)]
            )  # [n_samples, n_vars]
            w_mean_i = perturbed_sols.mean(axis=0)

            w_true_i = solver_fn(c_i)

            w_mean_list.append(torch.as_tensor(w_mean_i, dtype=dtype, device=device))
            w_true_list.append(torch.as_tensor(w_true_i, dtype=dtype, device=device))

        w_mean = torch.stack(w_mean_list)  # [B, n_vars]
        w_true = torch.stack(w_true_list)  # [B, n_vars]

        ctx.save_for_backward(w_mean, w_true)
        ctx.reduction = reduction
        ctx.batch_size = batch_size

        # ℓ_PFYL = cᵀ ŵ(θ̂) − cᵀ w*(c)
        loss_per_sample = (
            (c_true * w_mean).sum(dim=-1) - (c_true * w_true).sum(dim=-1)
        )

        if reduction == "mean":
            return loss_per_sample.mean()
        return loss_per_sample.sum()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ):
        w_mean, w_true = ctx.saved_tensors

        # Gradient: ŵ(θ̂, σ) − w*(c)
        grad_theta = w_mean - w_true

        if ctx.reduction == "mean":
            grad_theta = grad_theta / ctx.batch_size

        return grad_theta * grad_output, None, None, None, None, None
