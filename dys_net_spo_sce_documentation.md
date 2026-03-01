# DYS-Net + SPO+/SCE: Comprehensive Documentation

## Table of Contents
1. [Motivation: Why This Combination?](#1-motivation)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [Architecture Overview](#3-architecture-overview)
4. [Component 1: DYS-Net (The Differentiable Solver)](#4-dys-net)
5. [Component 2: SPO+ Loss](#5-spo-loss)
6. [Component 3: SCE Loss](#6-sce-loss)
7. [Complete Working Example](#7-complete-example)
8. [Hyperparameter Guide](#8-hyperparameters)
9. [Empirical Results & When to Use](#9-results)
10. [References](#10-references)

---

## 1. Motivation: Why This Combination?

### The Problem with Direct Regret Minimization

In Decision-Focused Learning (DFL), the natural approach is to minimize **regret** — the
gap between the decision cost using predicted parameters and the decision cost using
true parameters:

```
Regret(ĉ, c) = c^T x*(ĉ) - c^T x*(c)
```

To make this differentiable for LPs, researchers add quadratic regularization ("smoothing")
to convert the LP into a QP, then differentiate through it. However, **Mandi et al. (2025)**
showed a critical finding: **even after smoothing, the regret remains constant across large
regions of the parameter space**, yielding zero gradients that prevent learning.

### The Solution: DYS-Net + Surrogate Losses

The recommended architecture combines:

1. **DYS-Net** as the differentiable solver (fast, GPU-native, approximate QP solver)
2. **SPO+ or SCE** as the loss function (provides informative gradients everywhere)

This combination achieves **regret comparable to state-of-the-art methods** while
reducing training time by approximately **3x** versus CvxpyLayers.

```
                    ┌─────────────────────────────────────────────┐
   Features z ──►   │  Prediction Network  m_θ(z)                │
                    │  (standard neural net)                      │
                    └──────────────┬──────────────────────────────┘
                                   │  ĉ (predicted costs)
                                   ▼
                    ┌─────────────────────────────────────────────┐
                    │  DYS-Net Layer (differentiable QP solver)   │
                    │  Solves: min ĉ^T x + μ/2 ||x||²            │
                    │          s.t. Ax = b, x ≥ 0                │
                    │                                             │
                    │  Forward: Davis-Yin splitting iterations    │
                    │  Backward: Jacobian-Free Backprop (JFB)     │
                    └──────────────┬──────────────────────────────┘
                                   │  x*(ĉ) (approximate solution)
                                   ▼
                    ┌─────────────────────────────────────────────┐
                    │  Surrogate Loss (SPO+ or SCE)               │
                    │                                             │
                    │  SPO+: uses true optimal x*(c) + one extra  │
                    │        solve of x*(2ĉ - c)                  │
                    │                                             │
                    │  SCE:  (c^T x*(ĉ) - c^T x*(c))²            │
                    │        (squared cost error)                 │
                    └──────────────┬──────────────────────────────┘
                                   │  ∂L/∂ĉ  (informative gradient!)
                                   ▼
                    Backpropagate through DYS-Net → update θ
```

---

## 2. Mathematical Foundation

### 2.1 The Optimization Problem

Given a linear program:

```
x*(c) = argmin_{x ∈ S} c^T x
where S = {x : Ax = b, x ≥ 0}
```

DYS-Net solves the **quadratically regularized** (smoothed) version:

```
x*_μ(ĉ) = argmin_{x ∈ S} ĉ^T x + (μ/2) ||x||²
```

This converts the LP into a strictly convex QP with a unique, smooth solution.

### 2.2 Davis-Yin Splitting

The feasible set is decomposed: `S = F₁ ∩ F₂` where:
- `F₁ = {x : Ax = b}` (equality constraints — affine subspace)
- `F₂ = {x : x ≥ 0}` (non-negativity — the non-negative orthant)

The **Davis-Yin Splitting** iteration is a three-operator splitting scheme:

```
Given: z_k (current iterate), c (cost vector), μ (regularization), α (step size)

Step 1:  x_k = P_{F₂}(z_k)                         # Project onto x ≥ 0 (i.e., ReLU)
Step 2:  g_k = ∇h(x_k) = c + μ·x_k                 # Gradient of objective at x_k
Step 3:  y_k = P_{F₁}(2·x_k - z_k - α·g_k)         # Project onto Ax = b
Step 4:  z_{k+1} = z_k - x_k + y_k                  # Update z
```

**Key insight**: Each operation maps to a standard neural network layer:
- `P_{F₂}(·) = max(0, ·)` → **ReLU activation**
- `P_{F₁}(v) = v - A†(Av - b)` → **Matrix multiplication** (where A† is the pseudoinverse)
- `∇h(x) = c + μx` → **Linear layer**
- `z - x + y` → **Skip connection**

The pseudoinverse is computed via pre-computed SVD: `A† = V · diag(1/s) · U^T`
This SVD is computed **once** at initialization — not every forward pass.

### 2.3 Jacobian-Free Backpropagation (JFB)

Standard backpropagation through K iterations of DYS would require O(K) memory.
JFB is the critical efficiency trick:

```
Phase 1 (NO gradients):
    Run DYS iterations to convergence (K = 200-500 steps)
    with torch.no_grad():
        for k in range(K):
            z = dys_step(z, c)

Phase 2 (ONE step WITH gradients):
    z = dys_step(z.detach(), c)    # Only this step is tracked by autograd
    x = ReLU(z)                     # Final solution
```

The JFB approximation replaces the exact Jacobian of the fixed-point map with the
**identity matrix**. Fung et al. (2022) proved this still provides a descent direction
for the loss function, making it sufficient for gradient-based training.

**Result**: O(1) memory regardless of iteration count. Training is fast and stable.

### 2.4 SPO+ Loss

The **Smart Predict-then-Optimize Plus** loss (Elmachtoub & Grigas, 2022):

```
L_SPO+(ĉ, c) = max_{x ∈ S} (c - 2ĉ)^T x  +  2ĉ^T x*(c)  -  c^T x*(c)
```

Equivalently, if we define `z*(q) = min_{x ∈ S} q^T x`:

```
L_SPO+(ĉ, c) = -z*(2ĉ - c) + 2ĉ^T x*(c) - c^T x*(c)
```

**Subgradient** with respect to ĉ:

```
∂L_SPO+/∂ĉ = 2 · (x*(c) - x*(2ĉ - c))
```

This requires solving **one additional optimization**: `x*(2ĉ - c)`.

**Key properties**:
- Convex in ĉ (and in θ for linear prediction models)
- Statistically consistent (minimizing SPO+ asymptotically minimizes true regret)
- Non-negative: `L_SPO+ ≥ 0` and `L_SPO+ = 0` ⟺ `Regret = 0`
- Upper bounds twice the regret: `L_SPO+ ≥ Regret`

**With DYS-Net**: Instead of using an exact LP solver to compute `x*(2ĉ - c)`,
we use DYS-Net itself. This is an approximation but empirically produces
gradients of comparable quality at much lower cost.

### 2.5 SCE Loss (Squared Cost Error)

The **Self-Contrastive Estimation** loss (Mulamba et al., 2021; used with DYS-Net by Mandi et al., 2025):

```
L_SCE(x*(ĉ), c) = (c^T x*(ĉ) - c^T x*(c))²
```

This is simply the **squared regret**. With DYS-Net's smoothed solution:

```
L_SCE = (c^T x*_μ(ĉ) - c^T x*(c))²
```

**Gradient** with respect to ĉ (via chain rule through DYS-Net):

```
∂L_SCE/∂ĉ = 2 · (c^T x*_μ(ĉ) - c^T x*(c)) · c^T · (∂x*_μ/∂ĉ)
```

The term `∂x*_μ/∂ĉ` is computed automatically by JFB through the DYS layer.

**Key properties**:
- Zero iff regret is zero (Proposition 1 from Mandi et al., 2025)
- Provides informative (non-zero) gradients even in flat regions of the regret landscape
- After QP smoothing, L_SCE becomes convex and strictly decreasing toward the optimum
- Does NOT require an extra solver call (unlike SPO+) — uses DYS-Net output directly

### 2.6 SPO+ vs SCE: When to Use Which

| Property | SPO+ | SCE |
|----------|------|-----|
| Extra solver call per sample | Yes (solve `x*(2ĉ-c)`) | No |
| Theoretical guarantees | Convexity, Fisher consistency | Zero iff regret is zero |
| Performance with DYS-Net | State-of-the-art | State-of-the-art |
| Sensitivity to μ (smoothing) | Robust across μ values | Robust across μ values |
| Computational cost per step | Higher (extra solve) | Lower (only DYS forward) |
| Works with non-diff solver | Yes (only needs solver, not gradients) | Needs differentiable solver |

**Recommendation**: Start with **SCE** (simpler, no extra solver call). Switch to **SPO+** if
you need the stronger theoretical guarantees or if your problem has a non-differentiable solver.

---

## 3. Architecture Overview

### End-to-End Pipeline

```python
class DFLPipeline(nn.Module):
    """Complete DFL pipeline: Prediction → DYS-Net → Loss"""

    def __init__(self, n_features, n_costs, A, b, mu=1.0, alpha=0.1):
        super().__init__()
        # 1. Prediction network: features → predicted costs
        self.predictor = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, n_costs)
        )
        # 2. DYS-Net layer: predicted costs → approximate solution
        self.dys_layer = DYSLayer(A, b, mu=mu, alpha=alpha)

    def forward(self, features):
        c_hat = self.predictor(features)           # Predict costs
        x_hat = self.dys_layer(c_hat)              # Solve (approximately)
        return c_hat, x_hat

    def compute_spo_plus_loss(self, c_hat, c_true, x_true_optimal):
        """SPO+ loss using DYS-Net to solve the modified problem"""
        modified_cost = 2 * c_hat - c_true
        x_modified = self.dys_layer(modified_cost)  # Extra DYS-Net call
        loss = (
            torch.sum(modified_cost * x_modified, dim=-1)  # -z*(2ĉ-c) term
            + 2 * torch.sum(c_hat * x_true_optimal, dim=-1)
            - torch.sum(c_true * x_true_optimal, dim=-1)
        )
        return loss.mean()

    def compute_sce_loss(self, x_hat, c_true, optimal_obj):
        """SCE loss: squared difference in objective value"""
        predicted_obj = torch.sum(c_true * x_hat, dim=-1)
        loss = (predicted_obj - optimal_obj) ** 2
        return loss.mean()
```

---

## 4. DYS-Net: The Differentiable Solver

### Core Implementation

```python
import torch
import torch.nn as nn


class DYSLayer(nn.Module):
    """
    Davis-Yin Splitting layer for solving:
        min  c^T x + (mu/2)||x||^2
        s.t. Ax = b, x >= 0

    All operations are standard neural network layers (matrix multiply, ReLU).
    Uses Jacobian-Free Backpropagation (JFB) for memory efficiency.

    Args:
        A: Constraint matrix (m x n)
        b: Right-hand side vector (m,)
        mu: Quadratic regularization strength (default: 1.0)
        alpha: Step size, must satisfy alpha < 2/mu (default: 0.1)
        device: 'cuda', 'cpu', or 'mps'
    """

    def __init__(self, A, b, mu=1.0, alpha=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.mu = mu
        self.alpha = alpha
        self.m, self.n = A.shape

        # Store constraints (non-trainable)
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))
        self.register_buffer('b', torch.tensor(b, dtype=torch.float32))

        # Pre-compute SVD of A for efficient projection onto {x: Ax=b}
        # This is the ONE-TIME cost — not repeated per forward pass
        U, s, VT = torch.linalg.svd(self.A, full_matrices=False)
        s_inv = torch.where(s > 1e-6, 1.0 / s, torch.zeros_like(s))

        self.register_buffer('V', VT.T)        # (n x m)
        self.register_buffer('UT', U.T)         # (m x m)
        self.register_buffer('s_inv', s_inv)    # (m,)

    def project_nonneg(self, x):
        """Project onto F2 = {x >= 0}. This is just ReLU."""
        return torch.clamp(x, min=0)

    def project_equality(self, z):
        """
        Project onto F1 = {x : Ax = b}.
        Uses pre-computed SVD: P(z) = z - A†(Az - b)
        where A† = V diag(1/s) U^T
        """
        # Compute residual: Az - b
        # z shape: (batch, n), A shape: (m, n)
        residual = z @ self.A.T - self.b.unsqueeze(0)  # (batch, m)

        # Apply pseudoinverse: A†(residual) = V diag(1/s) U^T residual
        correction = residual @ self.UT.T               # (batch, m)
        correction = correction * self.s_inv.unsqueeze(0)  # element-wise
        correction = correction @ self.V.T              # (batch, n)

        return z - correction

    def grad_objective(self, x, c):
        """
        Gradient of h(x) = c^T x + (mu/2)||x||^2
        ∇h(x) = c + mu * x
        """
        return c + self.mu * x

    def dys_step(self, z, c):
        """
        One Davis-Yin Splitting iteration:
            x = P_{F2}(z)                              # ReLU
            g = ∇h(x) = c + μx                         # gradient
            y = P_{F1}(2x - z - α·g)                   # equality projection
            z_new = z - x + y                           # update
        """
        x = self.project_nonneg(z)
        g = self.grad_objective(x, c)
        y = self.project_equality(2.0 * x - z - self.alpha * g)
        return z - x + y

    def forward(self, c, max_iters=300, tol=1e-3):
        """
        Solve the smoothed QP using DYS with JFB.

        Args:
            c: Predicted cost vectors (batch_size, n)
            max_iters: Max iterations for convergence
            tol: Convergence tolerance

        Returns:
            x: Approximate solution (batch_size, n)
        """
        batch_size = c.shape[0]

        # ═══════════════════════════════════════════════
        # PHASE 1: Converge WITHOUT gradients (fast)
        # ═══════════════════════════════════════════════
        with torch.no_grad():
            z = torch.rand(batch_size, self.n, device=c.device) * 0.1

            for k in range(max_iters):
                z_prev = z.clone()
                z = self.dys_step(z, c)

                # Check convergence
                diff = torch.norm(z - z_prev, dim=-1).max()
                if diff < tol:
                    break

        # ═══════════════════════════════════════════════
        # PHASE 2: ONE step WITH gradients (JFB)
        # ═══════════════════════════════════════════════
        z = self.dys_step(z.detach(), c)  # ← only this step tracked by autograd
        x = self.project_nonneg(z)

        return x
```

### Why Each Operation Maps to a Neural Network Layer

| DYS Operation | Neural Network Equivalent | Complexity |
|---------------|--------------------------|------------|
| `P_{F2}(z) = max(0, z)` | ReLU activation | O(n) |
| `∇h(x) = c + μx` | Linear layer (bias=c, weight=μI) | O(n) |
| `2x - z - α·g` | Linear combination (skip connections) | O(n) |
| `P_{F1}(v) = v - V·diag(1/s)·U^T·(Av-b)` | Matrix multiplications | O(mn) |
| `z - x + y` | Residual connection | O(n) |

---

## 5. SPO+ Loss (Detailed)

### Implementation with DYS-Net

```python
class SPOPlusLoss(nn.Module):
    """
    SPO+ loss using DYS-Net as the solver.

    L_SPO+(ĉ, c) = max_{x ∈ S} (c - 2ĉ)^T x + 2ĉ^T x*(c) - c^T x*(c)

    In practice:
        1. Solve x_mod = x*(2ĉ - c) using DYS-Net
        2. L = (2ĉ - c)^T x_mod + 2ĉ^T x*(c) - c^T x*(c)

    Note: x*(c) is pre-computed offline using an exact solver.
    """

    def __init__(self, dys_layer):
        super().__init__()
        self.dys_layer = dys_layer

    def forward(self, c_hat, c_true, x_true_optimal, true_obj):
        """
        Args:
            c_hat: Predicted costs (batch, n) — output of prediction network
            c_true: True costs (batch, n)
            x_true_optimal: Pre-computed true optimal solutions x*(c) (batch, n)
            true_obj: Pre-computed c^T x*(c) (batch,)

        Returns:
            loss: Scalar loss value
        """
        # Compute modified cost
        c_modified = 2 * c_hat - c_true  # (batch, n)

        # Solve modified problem using DYS-Net
        x_modified = self.dys_layer(c_modified)  # (batch, n)

        # SPO+ loss computation
        # Term 1: (2ĉ - c)^T x*(2ĉ - c)  (value of modified problem)
        term1 = torch.sum(c_modified * x_modified, dim=-1)  # (batch,)

        # Term 2: 2ĉ^T x*(c)
        term2 = 2 * torch.sum(c_hat * x_true_optimal, dim=-1)  # (batch,)

        # Term 3: c^T x*(c) = true_obj
        term3 = true_obj  # (batch,)

        loss = term1 + term2 - term3  # (batch,)
        loss = torch.clamp(loss, min=0)  # Ensure non-negative

        return loss.mean()
```

### SPO+ Gradient Flow

```
∂L_SPO+/∂θ = ∂L_SPO+/∂ĉ · ∂ĉ/∂θ

where:
    ∂L_SPO+/∂ĉ = 2·(x*(c) - x*(2ĉ-c))
                   ↑ pre-computed   ↑ from DYS-Net

    ∂ĉ/∂θ = standard backprop through prediction network
```

---

## 6. SCE Loss (Detailed)

### Implementation with DYS-Net

```python
class SCELoss(nn.Module):
    """
    Squared Cost Error loss for use with DYS-Net.

    L_SCE = (c^T x*_μ(ĉ) - c^T x*(c))²

    No extra solver call needed — uses x*_μ(ĉ) directly from DYS-Net.
    """

    def forward(self, x_hat, c_true, true_obj):
        """
        Args:
            x_hat: DYS-Net solution x*_μ(ĉ) (batch, n)
            c_true: True costs (batch, n)
            true_obj: Pre-computed c^T x*(c) (batch,)

        Returns:
            loss: Scalar loss value
        """
        # Objective value under predicted solution
        predicted_obj = torch.sum(c_true * x_hat, dim=-1)  # (batch,)

        # Squared difference
        loss = (predicted_obj - true_obj) ** 2  # (batch,)

        return loss.mean()
```

### SCE Gradient Flow

```
∂L_SCE/∂θ = ∂L_SCE/∂x_hat · ∂x_hat/∂ĉ · ∂ĉ/∂θ

where:
    ∂L_SCE/∂x_hat = 2·(c^T x_hat - c^T x*(c)) · c
                      ↑ scalar (regret)            ↑ cost vector

    ∂x_hat/∂ĉ = JFB approximation through DYS-Net

    ∂ĉ/∂θ = standard backprop through prediction network
```

The key advantage: **even when regret is zero but the smoothed solution differs
from the LP solution, SCE still provides non-zero gradients** guiding ĉ toward
the region where the correct vertex is optimal.

---

## 7. Complete Working Example

### Shortest Path on a 5×5 Grid

```python
"""
Complete DYS-Net + SPO+/SCE example: Shortest Path on Grid Graph

Problem: Find shortest path from southwest to northeast corner of a 5×5 grid.
Edge costs are unknown and must be predicted from features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import linprog


# ═══════════════════════════════════════════════════════════════
# 1. BUILD THE SHORTEST PATH LP FORMULATION
# ═══════════════════════════════════════════════════════════════

def build_shortest_path_lp(grid_size=5):
    """
    Build the LP formulation for shortest path on a grid graph.

    Returns:
        A: Node-edge incidence matrix (n_nodes × n_edges)
        b: Flow conservation vector (source=+1, sink=-1, others=0)
        n_edges: Number of edges (decision variables)
    """
    n = grid_size
    nodes = [(i, j) for i in range(n) for j in range(n)]
    node_idx = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)

    # Build edges: right and up directions
    edges = []
    for i in range(n):
        for j in range(n):
            if j + 1 < n:  # right edge
                edges.append(((i, j), (i, j + 1)))
            if i + 1 < n:  # up edge
                edges.append(((i, j), (i + 1, j)))
    n_edges = len(edges)

    # Node-edge incidence matrix
    A = np.zeros((n_nodes, n_edges))
    for e_idx, (u, v) in enumerate(edges):
        A[node_idx[u], e_idx] = 1   # flow out
        A[node_idx[v], e_idx] = -1  # flow in

    # Flow conservation: +1 at source (0,0), -1 at sink (n-1,n-1)
    b = np.zeros(n_nodes)
    b[node_idx[(0, 0)]] = 1
    b[node_idx[(n - 1, n - 1)]] = -1

    return A, b, n_edges, edges


# ═══════════════════════════════════════════════════════════════
# 2. DYS-NET LAYER
# ═══════════════════════════════════════════════════════════════

class DYSLayer(nn.Module):
    """Davis-Yin Splitting layer for LP with quadratic regularization."""

    def __init__(self, A, b, mu=1.0, alpha=0.1):
        super().__init__()
        self.mu = mu
        self.alpha = alpha
        self.m, self.n = A.shape

        A_t = torch.tensor(A, dtype=torch.float32)
        b_t = torch.tensor(b, dtype=torch.float32)

        self.register_buffer('A_mat', A_t)
        self.register_buffer('b_vec', b_t)

        # Pre-compute SVD for projection onto {x: Ax=b}
        U, s, VT = torch.linalg.svd(A_t, full_matrices=False)
        s_inv = torch.where(s > 1e-6, 1.0 / s, torch.zeros_like(s))

        self.register_buffer('V', VT.T)
        self.register_buffer('UT', U.T)
        self.register_buffer('s_inv', s_inv)

    def project_nonneg(self, x):
        return torch.clamp(x, min=0)

    def project_equality(self, z):
        residual = z @ self.A_mat.T - self.b_vec.unsqueeze(0)
        correction = residual @ self.UT.T
        correction = correction * self.s_inv.unsqueeze(0)
        correction = correction @ self.V.T
        return z - correction

    def dys_step(self, z, c):
        x = self.project_nonneg(z)
        grad = c + self.mu * x
        y = self.project_equality(2.0 * x - z - self.alpha * grad)
        return z - x + y

    def forward(self, c, max_iters=300, tol=1e-3):
        batch_size = c.shape[0]

        # Phase 1: Converge without gradients
        with torch.no_grad():
            z = torch.rand(batch_size, self.n, device=c.device) * 0.1
            for _ in range(max_iters):
                z_prev = z
                z = self.dys_step(z, c)
                if torch.norm(z - z_prev).max() < tol:
                    break

        # Phase 2: One step with gradients (JFB)
        z = self.dys_step(z.detach(), c)
        return self.project_nonneg(z)


# ═══════════════════════════════════════════════════════════════
# 3. PREDICTION NETWORK + DFL MODEL
# ═══════════════════════════════════════════════════════════════

class ShortestPathDFL(nn.Module):
    """End-to-end DFL model for shortest path."""

    def __init__(self, n_features, n_edges, A, b, mu=1.0, alpha=0.1):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_edges),
        )
        self.dys = DYSLayer(A, b, mu=mu, alpha=alpha)

    def forward(self, features):
        c_hat = self.predictor(features)
        x_hat = self.dys(c_hat)
        return c_hat, x_hat


# ═══════════════════════════════════════════════════════════════
# 4. LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def spo_plus_loss(c_hat, c_true, x_true_opt, true_obj, dys_layer):
    """SPO+ loss using DYS-Net for the modified problem."""
    c_mod = 2 * c_hat - c_true
    x_mod = dys_layer(c_mod)

    term1 = torch.sum(c_mod * x_mod, dim=-1)
    term2 = 2 * torch.sum(c_hat * x_true_opt, dim=-1)
    term3 = true_obj

    loss = torch.clamp(term1 + term2 - term3, min=0)
    return loss.mean()


def sce_loss(x_hat, c_true, true_obj):
    """SCE loss: squared difference in objective value."""
    pred_obj = torch.sum(c_true * x_hat, dim=-1)
    return ((pred_obj - true_obj) ** 2).mean()


def mse_loss(c_hat, c_true):
    """Standard two-stage MSE loss (baseline)."""
    return ((c_hat - c_true) ** 2).mean()


# ═══════════════════════════════════════════════════════════════
# 5. DATA GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_data(n_samples, n_features, n_edges, seed=42):
    """
    Generate synthetic predict-then-optimize data.

    True costs: c = softplus(W @ features + noise)
    This creates a nonlinear, misspecified setting where DFL shines.
    """
    rng = np.random.RandomState(seed)

    W_true = rng.randn(n_edges, n_features) * 0.5
    features = rng.randn(n_samples, n_features)
    noise = rng.randn(n_samples, n_edges) * 0.1

    # Nonlinear cost generation (model is misspecified)
    costs = np.log1p(np.exp(features @ W_true.T + noise))  # softplus
    costs = np.maximum(costs, 0.01)  # ensure positive

    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(costs, dtype=torch.float32),
    )


def solve_shortest_path_exact(costs_np, A, b):
    """Solve shortest path LP exactly using scipy."""
    n_edges = costs_np.shape[1]
    bounds = [(0, None)] * n_edges

    solutions = []
    objectives = []
    for i in range(costs_np.shape[0]):
        res = linprog(costs_np[i], A_eq=A, b_eq=b, bounds=bounds, method='highs')
        solutions.append(res.x)
        objectives.append(res.fun)

    return (
        torch.tensor(np.array(solutions), dtype=torch.float32),
        torch.tensor(np.array(objectives), dtype=torch.float32),
    )


# ═══════════════════════════════════════════════════════════════
# 6. EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_regret(model, features, costs_np, A, b):
    """Compute normalized regret on test set."""
    model.eval()
    with torch.no_grad():
        c_hat, _ = model(features)
        c_hat_np = c_hat.numpy()

    n_edges = c_hat_np.shape[1]
    bounds = [(0, None)] * n_edges
    total_regret = 0.0
    total_optimal = 0.0

    for i in range(len(features)):
        # Decision with predicted costs
        res_pred = linprog(c_hat_np[i], A_eq=A, b_eq=b, bounds=bounds, method='highs')
        # True optimal decision
        res_true = linprog(costs_np[i], A_eq=A, b_eq=b, bounds=bounds, method='highs')

        # Regret: true cost of predicted decision - true optimal cost
        pred_true_cost = costs_np[i] @ res_pred.x
        true_opt_cost = res_true.fun

        total_regret += (pred_true_cost - true_opt_cost)
        total_optimal += abs(true_opt_cost)

    normalized_regret = total_regret / (total_optimal + 1e-8)
    return normalized_regret


# ═══════════════════════════════════════════════════════════════
# 7. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train_model(loss_type='sce', n_epochs=50, lr=1e-3, grid_size=5,
                n_train=200, n_test=100, n_features=5, seed=42):
    """
    Train a DFL model with specified loss function.

    Args:
        loss_type: 'spo+', 'sce', or 'mse' (baseline)
    """
    # Build problem
    A, b, n_edges, edges = build_shortest_path_lp(grid_size)
    print(f"Grid: {grid_size}x{grid_size}, Edges: {n_edges}, "
          f"Nodes: {grid_size**2}")

    # Generate data
    train_features, train_costs = generate_data(n_train, n_features, n_edges, seed)
    test_features, test_costs = generate_data(n_test, n_features, n_edges, seed + 1)

    # Pre-compute true optimal solutions (needed for SPO+ and SCE)
    print("Pre-computing optimal solutions...")
    train_solutions, train_objectives = solve_shortest_path_exact(
        train_costs.numpy(), A, b
    )
    print(f"  Done. Average optimal cost: {train_objectives.mean():.4f}")

    # Build model
    mu = 2.0
    alpha = 0.05  # Must satisfy alpha < 2/mu
    model = ShortestPathDFL(n_features, n_edges, A, b, mu=mu, alpha=alpha)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining with {loss_type.upper()} loss")
    print(f"  mu={mu}, alpha={alpha}, lr={lr}")
    print("-" * 60)

    # Training loop
    batch_size = 32
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        indices = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = indices[start:start + batch_size]
            feat_batch = train_features[idx]
            cost_batch = train_costs[idx]
            sol_batch = train_solutions[idx]
            obj_batch = train_objectives[idx]

            optimizer.zero_grad()
            c_hat, x_hat = model(feat_batch)

            # Compute loss based on chosen method
            if loss_type == 'sce':
                loss = sce_loss(x_hat, cost_batch, obj_batch)
            elif loss_type == 'spo+':
                loss = spo_plus_loss(
                    c_hat, cost_batch, sol_batch, obj_batch, model.dys
                )
            elif loss_type == 'mse':
                loss = mse_loss(c_hat, cost_batch)
            else:
                raise ValueError(f"Unknown loss: {loss_type}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / n_batches
            test_regret = evaluate_regret(
                model, test_features, test_costs.numpy(), A, b
            )
            print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.6f} | "
                  f"Test Regret: {test_regret:.4%}")

    # Final evaluation
    final_regret = evaluate_regret(model, test_features, test_costs.numpy(), A, b)
    print(f"\nFinal normalized regret ({loss_type.upper()}): {final_regret:.4%}")
    return final_regret


# ═══════════════════════════════════════════════════════════════
# 8. RUN COMPARISON
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("DYS-Net + SPO+/SCE vs Two-Stage MSE Comparison")
    print("Shortest Path on 5x5 Grid")
    print("=" * 60)

    results = {}
    for loss_type in ['mse', 'sce', 'spo+']:
        print(f"\n{'='*60}")
        regret = train_model(
            loss_type=loss_type,
            n_epochs=50,
            lr=1e-3,
            grid_size=5,
            n_train=200,
            n_test=100,
            n_features=5,
        )
        results[loss_type] = regret

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    for method, regret in results.items():
        print(f"  {method.upper():6s}: {regret:.4%} normalized regret")
    print()
    print("Expected: DFL methods (SCE, SPO+) should have LOWER regret")
    print("          than the two-stage baseline (MSE), especially")
    print("          under model misspecification (nonlinear costs).")
```

---

## 8. Hyperparameter Guide

### DYS-Net Hyperparameters

| Parameter | Symbol | Range | Effect | Recommended |
|-----------|--------|-------|--------|-------------|
| Regularization | μ | 0.5 - 5.0 | Higher = smoother but less accurate | 1.0 - 2.0 |
| Step size | α | 0.01 - 0.5 | Must satisfy α < 2/μ | 0.05 - 0.1 |
| Max iterations (train) | K | 200 - 500 | More = more accurate, slower | 300 |
| Max iterations (eval) | K_eval | 500 - 2000 | More precise at test time | 1000 |
| Convergence tolerance | ε | 1e-4 - 1e-2 | Stricter = more iterations | 1e-3 |

**Critical constraint**: `α < 2/μ`. If violated, DYS diverges. With μ=2.0, use α ≤ 0.9.
In practice, use α much smaller than the theoretical bound for stability.

### Loss Function Hyperparameters

| Parameter | Applies to | Range | Notes |
|-----------|-----------|-------|-------|
| Learning rate | All | 1e-4 - 1e-2 | SPO+ may need lower LR |
| Gradient clipping | All | 0.5 - 5.0 | Recommended to prevent instability |
| Batch size | All | 16 - 128 | Larger batches stabilize DYS convergence |

### Tuning Strategy

1. **Start with SCE** (simpler, fewer hyperparameters)
2. Set μ=1.0, α=0.1, max_iters=300
3. If regret is high, try μ=0.5 (closer to LP but noisier gradients)
4. If training is unstable, increase μ to 2.0-5.0
5. If SCE plateaus, try SPO+ (stronger theoretical guarantees)
6. At test time, either increase DYS iterations for better LP approximation,
   or swap in an exact LP/ILP solver for the final decision

---

## 9. Empirical Results & When to Use

### Results from Mandi et al. (2025)

**Benchmark**: Shortest Path (grid), Knapsack (ILP), Capacitated Facility Location (MILP)

Key findings:

1. **DYS-Net + SPO+ or SCE matches state-of-the-art regret** across all three problems
2. **DYS-Net + direct regret minimization fails** (zero gradients in most of parameter space)
3. **Training time reduced by ~3x** compared to CvxpyLayers with surrogate losses
4. **SPO+ yields similar regret regardless of solver type** (exact ILP, LP relaxation,
   CvxpyLayers, or DYS-Net) — confirming SPO+'s robustness
5. SCE provides comparable results to SPO+ when using differentiable solvers

### When to Use DYS-Net + Surrogate Losses

**Use when**:
- Problem is formulated as LP, ILP, or MILP with linear constraints
- Problem has hundreds to thousands of variables (DYS-Net scales well)
- You need GPU-native training without external solver dependencies
- Training speed is a priority

**Don't use when**:
- Problem has nonlinear constraints (DYS-Net requires Ax=b, x≥0 structure)
- Problem is very small (<50 variables) — CvxpyLayers overhead is negligible
- You need exact integer solutions during training (DYS-Net gives continuous relaxation)
- Your constraint matrix A changes between instances (SVD must be re-computed)

### Comparison Matrix

| Configuration | Regret Quality | Training Speed | Memory | Implementation Effort |
|---------------|---------------|----------------|--------|----------------------|
| CvxpyLayers + SPO+ | Excellent | Slow | High | Low (CVXPY API) |
| CvxpyLayers + Regret | Poor (zero grads) | Slow | High | Low |
| DYS-Net + SPO+ | Excellent | ~3x faster | O(1) via JFB | Medium |
| DYS-Net + SCE | Excellent | ~3x faster | O(1) via JFB | Medium |
| DYS-Net + Regret | Poor (zero grads) | Fast | O(1) | Medium |
| SPO+ (no diff solver) | Excellent | Medium | Low | Low |
| Two-Stage MSE | Baseline | Fastest | Lowest | Trivial |

---

## 10. References

1. **McKenzie, D., Heaton, H., & Fung, S. W.** (2024). Differentiating Through Integer
   Linear Programs with Quadratic Regularization and Davis-Yin Splitting. *TMLR*.
   [arXiv:2301.13395](https://arxiv.org/abs/2301.13395) |
   [Code: mines-opt-ml/fpo-dys](https://github.com/mines-opt-ml/fpo-dys)

2. **Mandi, J., et al.** (2025). Minimizing Surrogate Losses for Decision-Focused Learning
   using Differentiable Optimization. [arXiv:2508.11365](https://arxiv.org/abs/2508.11365) |
   [Code: JayMan91/DYS-NET-SCE](https://github.com/JayMan91/DYS-NET-SCE)

3. **Elmachtoub, A. N., & Grigas, P.** (2022). Smart "Predict, then Optimize".
   *Management Science*, 68(1), 9-26.

4. **Fung, S. W., Heaton, H., et al.** (2022). JFB: Jacobian-Free Backpropagation for
   Implicit Networks. *AAAI*.

5. **Mulamba, M., et al.** (2021). Contrastive Losses and Solution Caching for
   Predict-and-Optimize. *IJCAI*.

6. **Mandi, J., et al.** (2024). Decision-Focused Learning: Foundations, State of the Art,
   Benchmark and Future Opportunities. *JAIR*, 80, 1623-1701.

7. **Tang, B., & Khalil, E. B.** (2023). PyEPO: A PyTorch-based End-to-End
   Predict-then-Optimize Library. [github.com/khalil-research/PyEPO](https://github.com/khalil-research/PyEPO)
