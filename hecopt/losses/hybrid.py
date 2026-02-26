"""
Hybrid loss: interpolation between Decision-Focused Loss and MSE.

Motivation
----------
In early training epochs, the neural network outputs can be far from
feasible regions and the decision-focused gradient signal may vanish or
be extremely noisy.  A small MSE component provides a stable first-order
signal that guides the network towards reasonable predictions before the
DFL component takes over.

The hybrid loss is parameterised by a scalar λ ∈ [0, 1]:

    ℓ_hybrid = (1 − λ) · ℓ_DFL  +  λ · ℓ_MSE

Setting λ = 0 recovers pure decision-focused training; λ = 1 gives
standard predictive pre-training (e.g., useful for warm-starting).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    """Combine a decision-focused loss with standard MSE.

    Parameters
    ----------
    dfl_loss:
        A decision-focused loss module (e.g. ``SPOPlusLoss`` or ``PFYLoss``).
        Must accept ``(theta_pred, c_true)`` and return a scalar tensor.
    lambda_mse:
        Weight of the MSE term.  ``0.0`` → pure DFL; ``1.0`` → pure MSE.
        Default: ``0.0``.
    reduction:
        Loss reduction applied to the MSE component (``"mean"`` or ``"sum"``).
        The DFL loss uses its own internal reduction setting.
    """

    def __init__(
        self,
        dfl_loss: nn.Module,
        lambda_mse: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not 0.0 <= lambda_mse <= 1.0:
            raise ValueError(f"lambda_mse must be in [0, 1], got {lambda_mse}")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")

        self.dfl_loss = dfl_loss
        self.lambda_mse = lambda_mse
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(
        self, theta_pred: torch.Tensor, c_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute the hybrid loss.

        Parameters
        ----------
        theta_pred:
            Predicted parameters ``[batch_size, n_vars_or_params]``.
        c_true:
            True parameters ``[batch_size, n_vars_or_params]``.

        Returns
        -------
        torch.Tensor
            Scalar combined loss.
        """
        if self.lambda_mse == 1.0:
            return self.mse(theta_pred, c_true)

        loss_dfl = self.dfl_loss(theta_pred, c_true)

        if self.lambda_mse == 0.0:
            return loss_dfl

        loss_mse = self.mse(theta_pred, c_true)
        return (1.0 - self.lambda_mse) * loss_dfl + self.lambda_mse * loss_mse
