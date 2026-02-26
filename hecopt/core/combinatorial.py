"""
CombinatorialPtOLayer: Differentiable layer for LP/MIP problems.

Wraps any ``CombinatorialOptModel`` and enables end-to-end training by
routing gradients through surrogate decision-focused losses (SPO+ or PFYL)
instead of differentiating through the combinatorial solver directly.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np
import torch
import torch.nn as nn

from hecopt.core.base import CombinatorialOptModel, OptSolution
from hecopt.losses.hybrid import HybridLoss
from hecopt.losses.pfyl import PFYLoss
from hecopt.losses.spo_plus import SPOPlusLoss


class CombinatorialPtOLayer(nn.Module):
    """Differentiable Predict-then-Optimize layer for combinatorial problems.

    This layer wraps a ``CombinatorialOptModel`` and provides:

    * A differentiable ``loss()`` method based on the chosen surrogate
      (SPO+ or PFYL) that can be used directly in a training loop.
    * A non-differentiable ``forward()`` method that returns the optimal
      solution for a batch of predicted cost vectors (useful for evaluation).
    * A ``regret()`` metric that quantifies how much worse the predicted
      decisions are compared to optimal.

    Parameters
    ----------
    model:
        A ``CombinatorialOptModel`` implementing ``solve(theta)``.
    method:
        Gradient estimation strategy.  ``"spo_plus"`` (default) uses the
        Smart Predict-and-Optimize+ surrogate; ``"pfyl"`` uses Perturbed
        Fenchel-Young loss.
    n_samples:
        Number of noise samples for PFYL (ignored for SPO+).
    sigma:
        Noise level for PFYL (ignored for SPO+).
    lambda_hybrid:
        MSE weight in the hybrid loss (0 = pure DFL, 1 = pure MSE).
        A small value (e.g. 0.1) can stabilise early training.
    reduction:
        ``"mean"`` or ``"sum"`` over the batch.

    Examples
    --------
    >>> from hecopt.baselines.shortest_path import ShortestPathModel
    >>> from hecopt import CombinatorialPtOLayer
    >>> model = ShortestPathModel(grid_size=5)
    >>> layer = CombinatorialPtOLayer(model, method="spo_plus")
    >>> theta_pred = torch.randn(32, model.n_vars, requires_grad=True)
    >>> c_true    = torch.rand(32, model.n_vars)
    >>> loss = layer.loss(theta_pred, c_true)
    >>> loss.backward()
    """

    METHODS = ("spo_plus", "pfyl")

    def __init__(
        self,
        model: CombinatorialOptModel,
        method: Literal["spo_plus", "pfyl"] = "spo_plus",
        n_samples: int = 10,
        sigma: float = 1.0,
        lambda_hybrid: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if not isinstance(model, CombinatorialOptModel):
            raise TypeError(
                f"model must be a CombinatorialOptModel, got {type(model).__name__}"
            )
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got '{method}'")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")

        self.model = model
        self.method = method
        self.reduction = reduction

        solver_fn = self._make_solver_fn()

        if method == "spo_plus":
            dfl = SPOPlusLoss(solver_fn, reduction=reduction)
        else:
            dfl = PFYLoss(solver_fn, n_samples=n_samples, sigma=sigma, reduction=reduction)

        self.loss_fn = HybridLoss(dfl_loss=dfl, lambda_mse=lambda_hybrid, reduction=reduction)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_solver_fn(self) -> Callable:
        """Return a plain Python function (np → np) wrapping the model's solver."""

        def _solve(theta: np.ndarray) -> np.ndarray:
            return self.model.solve(theta).x_star

        return _solve

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Solve the combinatorial problem for a batch of cost predictions.

        .. note::
            This call is **not** differentiable.  Gradients should be
            obtained via ``loss()``.

        Parameters
        ----------
        theta:
            Predicted cost parameters ``[batch_size, n_vars]``.

        Returns
        -------
        torch.Tensor
            Optimal solutions ``[batch_size, n_vars]``.
        """
        solutions = [
            torch.as_tensor(
                self.model.solve(theta[i].detach().cpu().numpy()).x_star,
                dtype=theta.dtype,
                device=theta.device,
            )
            for i in range(theta.shape[0])
        ]
        return torch.stack(solutions)

    def loss(
        self,
        theta_pred: torch.Tensor,
        c_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the surrogate decision-focused loss (differentiable).

        Parameters
        ----------
        theta_pred:
            Predicted cost parameters ``[batch_size, n_vars]``.
            Must have ``requires_grad=True`` for backpropagation.
        c_true:
            True cost parameters ``[batch_size, n_vars]``.

        Returns
        -------
        torch.Tensor
            Scalar loss with valid surrogate gradients.
        """
        return self.loss_fn(theta_pred, c_true)

    @torch.no_grad()
    def regret(
        self, theta_pred: torch.Tensor, c_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute the mean normalised decision regret (evaluation metric).

        regret = (cᵀ w(θ̂) − cᵀ w*(c)) / |cᵀ w*(c)|

        Parameters
        ----------
        theta_pred:
            Predicted cost parameters ``[batch_size, n_vars]``.
        c_true:
            True cost parameters ``[batch_size, n_vars]``.

        Returns
        -------
        torch.Tensor
            Scalar mean normalised regret.
        """
        regrets: list[float] = []
        for i in range(theta_pred.shape[0]):
            theta_i = theta_pred[i].cpu().numpy()
            c_i = c_true[i].cpu().numpy()

            w_pred = self.model.solve(theta_i).x_star
            w_true = self.model.solve(c_i).x_star

            obj_pred = float(c_i @ w_pred)
            obj_true = float(c_i @ w_true)
            denom = abs(obj_true) if abs(obj_true) > 1e-10 else 1.0
            regrets.append((obj_pred - obj_true) / denom)

        return torch.tensor(np.mean(regrets), dtype=theta_pred.dtype)
