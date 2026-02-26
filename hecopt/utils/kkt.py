"""
KKT matrix construction and adjoint sensitivity method.

Given an optimal solution (x*, λ*, μ*) of a parameterised NLP, this module
builds the KKT block matrix and solves the adjoint system to propagate
upstream gradients back through the solver.

Mathematical background
-----------------------
For the NLP

    min_x  f(x, θ)
    s.t.   h(x, θ) = 0          [equality,   dual: λ ∈ ℝⁿᵉq]
           g(x, θ) ≤ 0          [inequality, dual: μ ∈ ℝⁿⁱⁿ, μ ≥ 0]

the KKT residual vector F(x, λ, μ, θ) = 0 at optimality:

    F = [ ∇_x L       ]   where L = f + λᵀh + μᵀg
        [ h           ]
        [ diag(μ) g_A ]   (active inequalities only)

By the Implicit Function Theorem (IFT):

    ∂x* / ∂θ = -(∂F/∂(x,λ,μ))⁻¹ · (∂F/∂θ)

The **adjoint sensitivity method** avoids forming this full Jacobian.
Given upstream gradient v = ∂L_task/∂x*, it solves

    Kᵀ u = [v; 0; …]
    ∂L_task/∂θ = -uᵀ (∂F/∂θ)

which is a single linear solve rather than an n×p matrix computation.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Tuple

import torch

# Default Tikhonov (ridge) regularisation added to K's diagonal for
# numerical stability when the KKT matrix is nearly singular.
TIKHONOV_DEFAULT: float = 1e-6


def build_kkt_matrix(
    objective_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eq_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ineq_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_star: torch.Tensor,
    theta: torch.Tensor,
    lambda_eq: torch.Tensor,
    mu_ineq: torch.Tensor,
    active_mask: Optional[torch.Tensor] = None,
    tikhonov: float = TIKHONOV_DEFAULT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the KKT block matrix K and cross-Jacobian dF/dθ.

    Parameters
    ----------
    objective_fn:
        f(x, θ) → scalar tensor.
    eq_fn:
        h(x, θ) → [n_eq] tensor (= 0 at feasibility).
    ineq_fn:
        g(x, θ) → [n_ineq] tensor (≤ 0 at feasibility).
    x_star:
        Optimal primal variables [n_vars].
    theta:
        Parameters [n_params].
    lambda_eq:
        Equality dual multipliers [n_eq].
    mu_ineq:
        Inequality dual multipliers [n_ineq].
    active_mask:
        Boolean mask [n_ineq] of active inequality constraints.
        If ``None``, inferred as ``mu_ineq > 1e-6``.
    tikhonov:
        Ridge regularisation on the (x,x) block of K.

    Returns
    -------
    K : torch.Tensor  [n_kkt, n_kkt]
        KKT block matrix.
    dF_dtheta : torch.Tensor  [n_kkt, n_params]
        Cross Jacobian ∂F/∂θ evaluated at the KKT point.
    """
    n_vars = x_star.shape[0]
    n_eq = lambda_eq.shape[0]

    if active_mask is None:
        active_mask = mu_ineq > 1e-6

    n_active = int(active_mask.sum().item())

    # ------------------------------------------------------------------ #
    # 1. Hessian of the Lagrangian  H = ∇²_xx L                          #
    # ------------------------------------------------------------------ #
    def lagrangian_x(x_: torch.Tensor) -> torch.Tensor:
        L = objective_fn(x_, theta)
        if n_eq > 0:
            h = eq_fn(x_, theta)
            L = L + (lambda_eq * h).sum()
        if n_active > 0:
            g = ineq_fn(x_, theta)
            L = L + (mu_ineq[active_mask] * g[active_mask]).sum()
        return L

    try:
        H = torch.autograd.functional.hessian(
            lagrangian_x, x_star, create_graph=False, strict=False
        )
        H = H.detach()
    except Exception as exc:
        warnings.warn(
            f"Hessian computation failed ({exc}); falling back to identity."
        )
        H = torch.eye(n_vars, dtype=x_star.dtype, device=x_star.device)

    # Tikhonov regularisation to prevent singularity
    H = H + tikhonov * torch.eye(n_vars, dtype=H.dtype, device=H.device)

    # ------------------------------------------------------------------ #
    # 2. Jacobians of constraints  J_h, J_g_active                       #
    # ------------------------------------------------------------------ #
    J_h = torch.zeros(n_eq, n_vars, dtype=H.dtype, device=H.device)
    if n_eq > 0:
        x_r = x_star.clone().requires_grad_(True)
        h_val = eq_fn(x_r, theta)
        for j in range(n_eq):
            (g_j,) = torch.autograd.grad(
                h_val[j], x_r, retain_graph=(j < n_eq - 1), create_graph=False
            )
            J_h[j] = g_j.detach()

    J_g_active = torch.zeros(n_active, n_vars, dtype=H.dtype, device=H.device)
    if n_active > 0:
        x_r2 = x_star.clone().requires_grad_(True)
        g_val = ineq_fn(x_r2, theta)
        g_a = g_val[active_mask]
        for j in range(n_active):
            (g_j,) = torch.autograd.grad(
                g_a[j], x_r2, retain_graph=(j < n_active - 1), create_graph=False
            )
            J_g_active[j] = g_j.detach()

    # ------------------------------------------------------------------ #
    # 3. Assemble KKT block matrix                                        #
    #                                                                      #
    #  K = [ H         J_hᵀ     J_g_Aᵀ ]                                 #
    #      [ J_h       0        0       ]                                 #
    #      [ J_g_A     0        0       ]                                 #
    # ------------------------------------------------------------------ #
    n_kkt = n_vars + n_eq + n_active
    K = torch.zeros(n_kkt, n_kkt, dtype=H.dtype, device=H.device)

    K[:n_vars, :n_vars] = H
    if n_eq > 0:
        K[:n_vars, n_vars : n_vars + n_eq] = J_h.T
        K[n_vars : n_vars + n_eq, :n_vars] = J_h
    if n_active > 0:
        K[:n_vars, n_vars + n_eq :] = J_g_active.T
        K[n_vars + n_eq :, :n_vars] = J_g_active

    # ------------------------------------------------------------------ #
    # 4. Cross Jacobian  dF/dθ                                            #
    # ------------------------------------------------------------------ #
    n_params = theta.shape[0]
    dF_dtheta = torch.zeros(n_kkt, n_params, dtype=H.dtype, device=H.device)

    # --- 4a. Mixed Hessian: ∂(∇_x L)/∂θ  [n_vars, n_params] ---
    def grad_lag_x_wrt_theta(th: torch.Tensor) -> torch.Tensor:
        """Returns ∇_x L(x*, th) as a [n_vars] vector."""
        x_r = x_star.clone().requires_grad_(True)
        L = objective_fn(x_r, th)
        if n_eq > 0:
            h = eq_fn(x_r, th)
            L = L + (lambda_eq * h).sum()
        if n_active > 0:
            g = ineq_fn(x_r, th)
            L = L + (mu_ineq[active_mask] * g[active_mask]).sum()
        (gx,) = torch.autograd.grad(L, x_r, create_graph=True)
        return gx  # [n_vars]

    try:
        mixed = torch.autograd.functional.jacobian(
            grad_lag_x_wrt_theta, theta, create_graph=False, strict=False
        )  # [n_vars, n_params]
        dF_dtheta[:n_vars] = mixed.detach()
    except Exception as exc:
        warnings.warn(f"Mixed Hessian ∂(∇_x L)/∂θ failed ({exc}); set to zero.")

    # --- 4b.  ∂h/∂θ  [n_eq, n_params] ---
    if n_eq > 0:
        try:

            def h_of_theta(th: torch.Tensor) -> torch.Tensor:
                return eq_fn(x_star, th)

            J_h_theta = torch.autograd.functional.jacobian(
                h_of_theta, theta, create_graph=False, strict=False
            )  # [n_eq, n_params]
            dF_dtheta[n_vars : n_vars + n_eq] = J_h_theta.detach()
        except Exception as exc:
            warnings.warn(f"∂h/∂θ failed ({exc}); set to zero.")

    # --- 4c.  ∂g_active/∂θ  [n_active, n_params] ---
    if n_active > 0:
        try:

            def g_active_of_theta(th: torch.Tensor) -> torch.Tensor:
                return ineq_fn(x_star, th)[active_mask]

            J_g_theta = torch.autograd.functional.jacobian(
                g_active_of_theta, theta, create_graph=False, strict=False
            )  # [n_active, n_params]
            dF_dtheta[n_vars + n_eq :] = J_g_theta.detach()
        except Exception as exc:
            warnings.warn(f"∂g_active/∂θ failed ({exc}); set to zero.")

    return K, dF_dtheta


def solve_adjoint(
    K: torch.Tensor,
    dF_dtheta: torch.Tensor,
    v: torch.Tensor,
    tikhonov: float = TIKHONOV_DEFAULT,
) -> torch.Tensor:
    """Solve the adjoint system and compute the parameter gradient.

    Solves  Kᵀ u = [v; 0; …]  and returns  dL/dθ = -uᵀ dF/dθ.

    Parameters
    ----------
    K:
        KKT matrix [n_kkt, n_kkt].
    dF_dtheta:
        Cross Jacobian [n_kkt, n_params].
    v:
        Upstream gradient w.r.t. primal variables [n_vars].
    tikhonov:
        Extra ridge regularisation if K is ill-conditioned.

    Returns
    -------
    torch.Tensor
        Gradient dL/dθ [n_params].
    """
    n_kkt = K.shape[0]
    n_vars = v.shape[0]

    rhs = torch.zeros(n_kkt, dtype=K.dtype, device=K.device)
    rhs[:n_vars] = v

    K_reg = K + tikhonov * torch.eye(n_kkt, dtype=K.dtype, device=K.device)

    try:
        u = torch.linalg.solve(K_reg.T, rhs)
    except torch.linalg.LinAlgError:
        warnings.warn(
            "KKT matrix is singular even with regularisation; using least-squares."
        )
        u = torch.linalg.lstsq(K_reg.T, rhs.unsqueeze(-1)).solution.squeeze(-1)

    return -(u @ dF_dtheta)
