# HecOpt

**Unified End-to-End Predict-Then-Optimize Library for Combinatorial and Non-Linear Decision-Focused Learning**

> A PyTorch library that bridges machine learning and operations research by enabling neural networks to be trained directly against decision quality — not just predictive accuracy.

---

## Table of Contents

1. [The Problem HecOpt Solves](#1-the-problem-hecopt-solves)
2. [Why Existing Libraries Fall Short](#2-why-existing-libraries-fall-short)
3. [HecOpt's Unified Approach](#3-hecopts-unified-approach)
4. [Installation](#4-installation)
5. [Quickstart](#5-quickstart)
6. [Architecture Overview](#6-architecture-overview)
7. [Mathematical Background](#7-mathematical-background)
8. [API Reference](#8-api-reference)
9. [Built-in Baselines](#9-built-in-baselines)
10. [Implementing Custom Problems](#10-implementing-custom-problems)
11. [Full Training Examples](#11-full-training-examples)
12. [Comparison with Existing Libraries](#12-comparison-with-existing-libraries)
13. [Design Decisions and Trade-offs](#13-design-decisions-and-trade-offs)
14. [References](#14-references)

---

## 1. The Problem HecOpt Solves

Modern industrial decision-making pipelines follow a two-stage pattern called **Predict-Then-Optimize (PtO)**:

```
Stage 1 (ML):   raw features  →  predict unknown parameters  θ̂
Stage 2 (OR):   θ̂            →  solve optimization problem  →  decision x*
```

**The critical flaw:** the machine learning model in Stage 1 is trained to minimise a statistical error metric (MSE, MAE, cross-entropy), completely ignoring the downstream optimization problem. This creates a fundamental **prediction-decision misalignment**:

- A small prediction error in a *non-critical* region of the parameter space has zero impact on the final decision.
- A small prediction error near an *active constraint* or *decision boundary* can cascade into a completely wrong decision.
- The ML model, unaware of the downstream problem, penalises all errors equally — wasting model capacity on irrelevant predictions.

**Decision-Focused Learning (DFL)** resolves this by integrating the optimization problem directly into the neural network's training loop. Instead of minimising prediction error, the network minimises **decision regret** — the gap between the quality of the decision made with predicted parameters and the quality of the decision that would have been made with the true parameters.

The central mathematical challenge: optimization problems have **zero or undefined gradients** almost everywhere with respect to their parameters (for combinatorial problems the solution jumps discretely; for non-linear problems the gradient requires differentiating through the KKT conditions). HecOpt implements the state-of-the-art solutions to both cases within a single, unified library.

---

## 2. Why Existing Libraries Fall Short

| Library | Handles | Mechanism | Key Limitation |
|---|---|---|---|
| **PyEPO** | LP / MIP only | SPO+, PFYL, black-box perturbations | Cannot handle non-linear objectives or quadratic constraints |
| **CVXPYLayers** | Disciplined Convex Programs only | KKT implicit differentiation via conic form | Fails on non-convex problems; degenerate on LPs |
| **HecOpt** | **Both** combinatorial and non-linear | SPO+/PFYL for combinatorial; KKT adjoint for NLPs | — |

In practice, an industrial engineer faces both domains simultaneously. A logistics optimiser needs surrogate losses for TSP-like routing problems; a pricing engine needs KKT differentiation for revenue maximisation. Today, these require entirely different codebases, frameworks, and theoretical setups. HecOpt unifies both under a single, consistent API.

---

## 3. HecOpt's Unified Approach

HecOpt automatically selects the correct differentiation strategy based on the nature of the downstream problem:

```
                          ┌─────────────────────────────────────────────┐
                          │               Neural Network                │
                          │         features → predicted θ̂             │
                          └──────────────────┬──────────────────────────┘
                                             │
                          ┌──────────────────▼──────────────────────────┐
                          │          HecOpt Dispatch Layer              │
                          └──────┬───────────────────────┬──────────────┘
                                 │                       │
               ┌─────────────────▼──────┐   ┌───────────▼───────────────┐
               │  CombinatorialPtOLayer  │   │   NonLinearPtOLayer        │
               │  (LP / MIP problems)    │   │   (NLP problems)           │
               │                         │   │                            │
               │  Forward: call solver   │   │  Forward: call NLP solver  │
               │  Backward: SPO+ / PFYL  │   │  Backward: KKT adjoint     │
               │  surrogate loss         │   │  (Implicit Function Thm)   │
               └────────────────────────┘   └────────────────────────────┘
```

**Both paths are fully differentiable end-to-end.** Gradients flow back through the optimizer layer to the neural network parameters, training the model to make predictions that yield better *decisions*, not just better *predictions*.

---

## 4. Installation

### Standard Install (pip)

```bash
pip install hecopt
```

### From Source (Development)

```bash
git clone https://github.com/hecopt/hecopt
cd hecopt
pip install -e ".[dev]"
```

### With Example Dependencies

```bash
pip install "hecopt[examples]"   # adds matplotlib, networkx, tqdm
pip install "hecopt[all]"        # dev + examples
```

### Requirements

| Package | Minimum Version | Purpose |
|---|---|---|
| Python | 3.9 | — |
| PyTorch | 2.0.0 | Neural networks, autograd |
| NumPy | 1.24.0 | Numerical arrays |
| SciPy | 1.10.0 | LP and NLP solvers |

No GPU required. All solvers run on CPU via SciPy. PyTorch GPU support works for the neural network components.

---

## 5. Quickstart

### Combinatorial: Shortest Path

```python
import torch
import torch.nn as nn
from hecopt import CombinatorialPtOLayer
from hecopt.baselines.shortest_path import ShortestPathModel, ShortestPathDataset

# 1. Define the combinatorial problem
opt_model = ShortestPathModel(grid_size=5)   # 5×5 grid, 40 edges

# 2. Wrap it in the differentiable layer (SPO+ loss by default)
layer = CombinatorialPtOLayer(opt_model, method="spo_plus", lambda_hybrid=0.1)

# 3. Build a neural network: features → predicted edge costs
net = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, opt_model.n_vars))

# 4. Load synthetic data
dataset = ShortestPathDataset(n_samples=1000, grid_size=5, n_features=8)
loader  = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 5. Standard PyTorch training loop
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

for features, true_costs, _ in loader:
    optimizer.zero_grad()
    theta_pred = net(features)                      # predicted edge costs
    loss = layer.loss(theta_pred, true_costs)       # SPO+ surrogate loss
    loss.backward()                                 # gradients flow back
    optimizer.step()

# 6. Evaluation: decision regret (not prediction error)
regret = layer.regret(theta_pred.detach(), true_costs)
print(f"Normalised decision regret: {regret:.4f}")
```

### Non-Linear: Pricing Optimisation

```python
import torch
import torch.nn as nn
from hecopt import NonLinearPtOLayer
from hecopt.baselines.pricing import PricingModel, PricingDataset

# 1. Define the non-linear pricing problem (3 products)
opt_model = PricingModel(n_products=3, p_min=1.0, p_max=10.0, capacity=50.0)

# 2. Wrap it in the KKT-differentiable layer
layer = NonLinearPtOLayer(opt_model, tikhonov=1e-6)

# 3. Neural network: market features → demand parameters [a₁,a₂,a₃, ε₁,ε₂,ε₃]
net = nn.Sequential(
    nn.Linear(8, 64), nn.ReLU(),
    nn.Linear(64, opt_model.n_params),
    nn.Softplus()     # ensure positive demand parameters
)

# 4. Synthetic dataset
dataset = PricingDataset(n_samples=1000, n_products=3, n_features=8)
loader  = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# 5. Training loop with KKT-based gradients
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

for features, theta_true, theta_noisy, prices_true, revenue_true in loader:
    optimizer.zero_grad()
    theta_pred = net(features)                              # predicted demand params
    loss = layer.decision_loss(theta_pred, theta_true)      # regret loss (KKT diff)
    loss.backward()                                         # KKT adjoint backward
    optimizer.step()

# 6. Evaluate regret
regret = layer.regret(theta_pred.detach(), theta_true)
print(f"Normalised objective regret: {regret:.4f}")
```

---

## 6. Architecture Overview

```
hecopt/
├── core/
│   ├── base.py            # Abstract base classes and OptSolution dataclass
│   ├── combinatorial.py   # CombinatorialPtOLayer (SPO+ / PFYL)
│   └── nonlinear.py       # NonLinearPtOLayer (KKT adjoint)
├── losses/
│   ├── spo_plus.py        # SPO+ loss (custom autograd.Function)
│   ├── pfyl.py            # Perturbed Fenchel-Young loss
│   └── hybrid.py          # λ·MSE + (1−λ)·DFL hybrid wrapper
├── baselines/
│   ├── shortest_path.py   # Linear baseline: grid shortest path + dataset
│   └── pricing.py         # Non-linear baseline: pricing NLP + dataset
└── utils/
    └── kkt.py             # KKT matrix builder + adjoint system solver
```

### Class Hierarchy

```
BaseOptModel (ABC)
├── CombinatorialOptModel          ← subclass this for LP/MIP problems
│   └── ShortestPathModel          (built-in baseline)
└── NonLinearOptModel              ← subclass this for NLP problems
    └── PricingModel               (built-in baseline)

nn.Module
├── CombinatorialPtOLayer          ← wraps CombinatorialOptModel
│   └── uses: SPOPlusLoss | PFYLoss | HybridLoss
└── NonLinearPtOLayer              ← wraps NonLinearOptModel
    └── uses: build_kkt_matrix, solve_adjoint

torch.autograd.Function
├── _SPOPlusFunction               (internal)
├── _PFYLFunction                  (internal)
└── _NonLinearFunction             (internal)
```

---

## 7. Mathematical Background

### 7.1 The Decision Regret

For any optimization problem with feasible set W and true parameter c, the **decision regret** of predicting θ̂ instead of c is:

```
regret(θ̂, c)  =  obj(x*(θ̂), c)  −  obj(x*(c), c)  ≥  0
```

where `x*(θ) = argmin_{x ∈ W} obj(x, θ)`. The regret is always non-negative and zero only when the predicted decision is optimal under true parameters.

The key insight of DFL is to train the neural network to minimise regret, not prediction error.

---

### 7.2 Combinatorial Path: SPO+ Loss

**Problem:** for linear programs `min θᵀx s.t. x ∈ W`, the solution `x*(θ)` jumps discontinuously between vertices of the polytope W as θ changes, so `∂x*/∂θ = 0` almost everywhere. Standard backpropagation is impossible.

**SPO+ (Smart Predict-and-Optimize Plus)** constructs a **convex surrogate upper bound** on the regret that has non-zero, informative subgradients everywhere.

**Definition:**

```
ℓ_SPO+(θ̂, c)  =  cᵀ w*(2θ̂ − c)  −  2θ̂ᵀ w*(θ̂)  +  cᵀ w*(θ̂)
```

where:
- `w*(θ) = argmin_{w ∈ W} θᵀw` is the solution under cost vector θ
- `w*(2θ̂ − c)` is the solution under the "SPO cost" `2θ̂ − c`

**Subgradient** (used in backpropagation):

```
∂ℓ_SPO+ / ∂θ̂  =  2 · ( w*(2θ̂ − c) − w*(θ̂) )
```

**Interpretation:** the gradient is zero when `w*(2θ̂ − c) = w*(θ̂)` (the predictions are already decision-aligned). It pushes θ̂ in the direction that makes the "SPO solution" converge towards the "predicted solution", which in turn converges towards `w*(c)`.

**Forward pass:** calls the combinatorial solver twice (once with θ̂, once with `2θ̂ − c`).
**Backward pass:** returns the subgradient above — no solver call needed.

---

### 7.3 Combinatorial Path: PFYL Loss

**Perturbed Fenchel-Young Loss (PFYL)** smooths the combinatorial map by injecting Gaussian noise:

```
ŵ(θ̂, σ)  =  𝔼_{ε ~ N(0,I)}[ w*(θ̂ + σε) ]
```

The loss and its gradient are:

```
ℓ_PFYL(θ̂, c)  =  cᵀ ŵ(θ̂, σ) − cᵀ w*(c)

∂ℓ_PFYL / ∂θ̂  =  ŵ(θ̂, σ) − w*(c)
```

**Forward pass:** calls the solver `n_samples` times with perturbed costs `θ̂ + σεₖ`.
**Backward pass:** gradient is the mean perturbed solution minus the optimal solution under true costs.

**Trade-off vs SPO+:** PFYL requires more solver calls but provides smoother gradient estimates. Useful when the combinatorial landscape is highly discontinuous and SPO+ gradients are too sparse.

---

### 7.4 Non-Linear Path: KKT Implicit Differentiation

**Problem:** for a parameterised NLP

```
min_x  f(x, θ)
s.t.   h(x, θ) = 0        [equality,   dual: λ ∈ ℝⁿᵉq]
       g(x, θ) ≤ 0        [inequality, dual: μ ∈ ℝⁿⁱⁿ, μ ≥ 0]
```

the optimal solution `x*(θ)` is differentiable with respect to θ (under regularity conditions). HecOpt computes these derivatives using the **Implicit Function Theorem (IFT)** applied to the KKT conditions.

**KKT Conditions** at optimality `(x*, λ*, μ*)`:

```
F(x*, λ*, μ*, θ) = 0  where:

F = [ ∇_x L(x*, λ*, μ*)     ]    L = f + λᵀh + μᵀg  (Lagrangian)
    [ h(x*, θ)               ]
    [ diag(μ*) g(x*, θ)_A    ]    (active inequalities only)
```

**IFT gives:**

```
∂x* / ∂θ  =  −(∂F/∂(x,λ,μ))⁻¹ · (∂F/∂θ)
```

#### The KKT Block Matrix

```
K  =  ∂F / ∂(x, λ, μ)  =  [ H        J_hᵀ     J_g_Aᵀ ]
                            [ J_h      0         0      ]
                            [ J_g_A    0         0      ]
```

where:
- `H = ∇²_xx L` — Hessian of the Lagrangian w.r.t. primal variables
- `J_h = ∂h/∂x` — Jacobian of equality constraints
- `J_g_A = ∂g_A/∂x` — Jacobian of **active** inequality constraints (where `g_j(x*) ≈ 0`)

**Active-set tracking:** only active inequalities enter the KKT matrix. Including inactive constraints (where `g_j(x*) < 0` strictly) causes rank deficiency. HecOpt automatically tracks the active set from the dual multipliers returned by the solver.

**Tikhonov Regularisation:** `K_reg = K + εI` with `ε = 1e-6` (configurable) prevents singular matrix failures from near-degenerate active sets or non-convex Hessians.

#### The Adjoint Sensitivity Method

Computing the full Jacobian `∂x*/∂θ` (shape `n_vars × n_params`) is expensive. For backpropagation, we only need the vector-Jacobian product (VJP) with the upstream gradient `v = ∂L_task/∂x*`.

The adjoint method computes this with a single linear solve:

```
Step 1: Solve  Kᵀ u  =  [v; 0; 0; …]      (adjoint system, shape n_kkt)
Step 2: Compute  ∂L_task/∂θ  =  −uᵀ (∂F/∂θ)
```

Cost: **one linear solve** (O(n³) or faster with factorisation) instead of forming an `n×p` Jacobian.

All Hessians, Jacobians, and mixed partial derivatives (`∂F/∂θ`) are computed automatically via `torch.autograd.functional.hessian` and `torch.autograd.functional.jacobian`.

---

### 7.5 Hybrid Loss

To stabilise training in early epochs (before the network has learned reasonable predictions), HecOpt offers a **hybrid loss**:

```
ℓ_hybrid  =  (1 − λ) · ℓ_DFL  +  λ · ℓ_MSE
```

- `λ = 0`: pure decision-focused loss (best asymptotic performance)
- `λ = 1`: pure MSE (standard supervised pre-training)
- `λ ∈ (0, 1)`: interpolation — recommended `λ ∈ [0.05, 0.2]` for early training

---

## 8. API Reference

### `OptSolution`

Dataclass returned by every call to `model.solve()`.

```python
@dataclass
class OptSolution:
    x_star:     np.ndarray           # Optimal primal variables [n_vars]
    obj_val:    float                # Optimal objective value
    y_star:     Optional[np.ndarray] # Equality dual multipliers [n_eq]
    z_star:     Optional[np.ndarray] # Inequality dual multipliers [n_ineq]
    active_ineq: Optional[np.ndarray] # Active constraint mask [n_ineq]
    success:    bool                 # True if solver converged
    message:    str                  # Solver status string
```

---

### `CombinatorialPtOLayer`

```python
CombinatorialPtOLayer(
    model,                  # CombinatorialOptModel — the problem to solve
    method="spo_plus",      # "spo_plus" | "pfyl"
    n_samples=10,           # PFYL only: number of noise perturbations
    sigma=1.0,              # PFYL only: noise standard deviation
    lambda_hybrid=0.0,      # MSE weight in hybrid loss [0, 1]
    reduction="mean",       # "mean" | "sum"
)
```

| Method | Signature | Description |
|---|---|---|
| `forward(theta)` | `[B, n_vars] → [B, n_vars]` | Solve for each instance. **Not differentiable.** |
| `loss(theta_pred, c_true)` | `[B, n_vars], [B, n_vars] → scalar` | Surrogate loss with valid gradients. |
| `regret(theta_pred, c_true)` | `[B, n_vars], [B, n_vars] → scalar` | Mean normalised decision regret (no grad). |

---

### `NonLinearPtOLayer`

```python
NonLinearPtOLayer(
    model,              # NonLinearOptModel — the problem to solve
    tikhonov=1e-6,      # Ridge regularisation for KKT matrix
    reduction="mean",   # "mean" | "sum"
)
```

| Method | Signature | Description |
|---|---|---|
| `forward(theta)` | `[B, n_params] → [B, n_vars]` | Solve NLP; **differentiable** via KKT adjoint. |
| `decision_loss(theta_pred, theta_true)` | `[B, n_params], [B, n_params] → scalar` | Objective regret loss with KKT gradients. |
| `regret(theta_pred, theta_true)` | `[B, n_params], [B, n_params] → scalar` | Mean normalised objective regret (no grad). |

---

### `SPOPlusLoss`

```python
SPOPlusLoss(solver_fn, reduction="mean")
```

Standalone SPO+ loss. `solver_fn` is any callable `(cost: np.ndarray) → solution: np.ndarray`.

```python
loss = loss_fn(theta_pred, c_true)   # [B, n_vars], [B, n_vars] → scalar
loss.backward()                       # gradient: 2 * (w*(2θ̂−c) − w*(θ̂))
```

---

### `PFYLoss`

```python
PFYLoss(solver_fn, n_samples=10, sigma=1.0, reduction="mean")
```

Standalone PFYL loss. Same interface as `SPOPlusLoss`.

```python
loss = loss_fn(theta_pred, c_true)
loss.backward()                       # gradient: ŵ(θ̂, σ) − w*(c)
```

---

### `HybridLoss`

```python
HybridLoss(dfl_loss, lambda_mse=0.0, reduction="mean")
```

Wraps any DFL loss with an MSE component.

```python
hybrid = HybridLoss(SPOPlusLoss(solver_fn), lambda_mse=0.1)
loss = hybrid(theta_pred, c_true)
loss.backward()
```

---

### `build_kkt_matrix` / `solve_adjoint`

Low-level utilities for custom KKT differentiation workflows.

```python
from hecopt.utils.kkt import build_kkt_matrix, solve_adjoint

K, dF_dtheta = build_kkt_matrix(
    objective_fn,   # f(x, θ) → scalar tensor
    eq_fn,          # h(x, θ) → [n_eq] tensor
    ineq_fn,        # g(x, θ) → [n_ineq] tensor
    x_star,         # [n_vars]
    theta,          # [n_params]
    lambda_eq,      # [n_eq]
    mu_ineq,        # [n_ineq]
    active_mask,    # [n_ineq] bool — None to auto-infer
    tikhonov,       # float, default 1e-6
)

grad_theta = solve_adjoint(K, dF_dtheta, v=upstream_grad, tikhonov=1e-6)
```

---

## 9. Built-in Baselines

HecOpt ships with two reference problems — one for each domain — that are standard benchmarks in the DFL literature.

---

### Baseline 1: Shortest Path (Linear / Combinatorial)

**File:** `hecopt/baselines/shortest_path.py`

**Problem:** minimum-cost path on an `m × m` directed grid graph from the top-left to the bottom-right node.

```
Nodes:           m² nodes arranged on a grid
Edges:           right (→) and down (↓), 2·m·(m−1) total
Decision vars:   xₑ ∈ {0, 1}  for each edge e
Objective:       min  θᵀ x   (θ = predicted edge costs)
Constraints:     flow conservation: Ax = b
                 source sends 1 unit of flow; sink receives 1 unit
```

Because the constraint matrix is **Totally Dual Integral (TDI)**, the LP relaxation is solved exactly with SciPy HiGHS. The result is always integral.

| Grid size | Nodes | Edges (n_vars) | Path length |
|---|---|---|---|
| 2×2 | 4 | 4 | 2 edges |
| 5×5 | 25 | 40 | 8 edges |
| 8×8 | 64 | 112 | 14 edges |
| 10×10 | 100 | 180 | 18 edges |

```python
from hecopt.baselines.shortest_path import ShortestPathModel, ShortestPathDataset

model = ShortestPathModel(grid_size=5)
print(model.n_vars)      # 40 edges

sol = model.solve(theta)               # theta: np.ndarray [40]
print(sol.x_star)                      # binary [40] — selected path edges
print(sol.obj_val)                     # path cost under theta
print(model.path_edges(sol.x_star))   # list of (u, v) node pairs

# Synthetic dataset: features → true costs → optimal path
dataset = ShortestPathDataset(
    n_samples=1000,
    grid_size=5,
    n_features=4,     # input feature dimension
    degree=2,         # 1=linear mapping, 2=quadratic (harder)
    noise_std=0.1,    # additive noise on true costs
    seed=42,
)
features, costs, solutions = dataset[0]
# features:  [4]   — observable market/route signals
# costs:     [40]  — true edge costs (targets for supervised pre-training)
# solutions: [40]  — binary optimal path (for evaluation only)
```

---

### Baseline 2: Multi-Product Pricing (Non-Linear / Continuous)

**File:** `hecopt/baselines/pricing.py`

**Problem:** set prices for n products to maximise revenue under constant-elasticity demand, subject to price bounds and a capacity constraint.

```
Decision vars:   p = [p₁, …, pₙ]  (prices)
Parameters:      θ = [a₁,…,aₙ, ε₁,…,εₙ]  (demand scale + elasticities)

max_{p}   Σᵢ (pᵢ − cᵢ) · aᵢ · pᵢ^(−εᵢ)       (revenue)

s.t.      p_min ≤ pᵢ ≤ p_max    ∀ i             (price bounds)
          Σᵢ aᵢ · pᵢ^(−εᵢ) ≤ Q                 (total demand capacity)
```

This is a **non-convex NLP** (the product `pᵢ^(1−εᵢ)` is non-concave when `εᵢ ≠ 1`), making it a strong test for the KKT backend. Solved with SciPy SLSQP with analytic gradients.

```python
from hecopt.baselines.pricing import PricingModel, PricingDataset

model = PricingModel(
    n_products=3,
    p_min=1.0,
    p_max=10.0,
    capacity=50.0,
    marginal_costs=np.array([0.5, 0.3, 0.7]),  # optional per-unit costs
)

# theta = [a₁, a₂, a₃, ε₁, ε₂, ε₃]
theta = np.array([3.0, 2.5, 4.0, 1.5, 2.0, 1.8])
sol = model.solve(theta)
print(sol.x_star)    # optimal prices [p₁*, p₂*, p₃*]
print(sol.obj_val)   # maximum revenue

# Synthetic dataset: market features → demand params → optimal prices
dataset = PricingDataset(
    n_samples=1000,
    n_products=3,
    n_features=6,    # observable market signals
    noise_std=0.05,  # noise on demand parameters
    seed=42,
)
features, theta_true, theta_noisy, prices, revenue = dataset[0]
# features:     [6]  — observable market signals
# theta_true:   [6]  — true demand parameters [a, ε]
# theta_noisy:  [6]  — noisy observations (for pre-training)
# prices:       [3]  — optimal prices under true params
# revenue:      []   — optimal revenue (scalar)
```

---

## 10. Implementing Custom Problems

### Custom Combinatorial Problem

Subclass `CombinatorialOptModel` and implement `n_vars` and `solve()`. The solver must accept a NumPy cost vector and return an `OptSolution`.

```python
import numpy as np
from scipy.optimize import linprog
from hecopt import CombinatorialOptModel, OptSolution, CombinatorialPtOLayer

class KnapsackModel(CombinatorialOptModel):
    """0/1 Knapsack: max vᵀx s.t. wᵀx ≤ W, x ∈ {0,1}ⁿ."""

    def __init__(self, weights: np.ndarray, capacity: float):
        self.weights = weights
        self.capacity = capacity
        self._n = len(weights)

    @property
    def n_vars(self) -> int:
        return self._n

    def solve(self, theta: np.ndarray) -> OptSolution:
        # LP relaxation with upper bounds (TDI for some knapsack variants)
        # For general knapsack: use branch-and-bound or external MIP solver
        result = linprog(
            c=-theta,            # minimise negative value (= maximise value)
            A_ub=self.weights.reshape(1, -1),
            b_ub=np.array([self.capacity]),
            bounds=[(0.0, 1.0)] * self._n,
            method="highs",
        )
        x = np.round(result.x)
        return OptSolution(x_star=x, obj_val=-result.fun, success=result.success)

# Use exactly like the built-in baselines
model = KnapsackModel(weights=np.array([2.0, 3.0, 1.5, 4.0]), capacity=5.0)
layer = CombinatorialPtOLayer(model, method="spo_plus")

theta_pred = torch.randn(32, model.n_vars, requires_grad=True)
c_true     = torch.rand(32, model.n_vars)
loss = layer.loss(theta_pred, c_true)
loss.backward()
```

---

### Custom Non-Linear Problem

Subclass `NonLinearOptModel` and implement:
1. `n_vars` — number of decision variables
2. `solve(theta)` — NumPy-based solver returning an `OptSolution` **with dual multipliers**
3. `objective_torch(x, theta)` — PyTorch-differentiable objective
4. `ineq_constraints_torch(x, theta)` *(optional)* — PyTorch-differentiable inequality constraints
5. `eq_constraints_torch(x, theta)` *(optional)* — PyTorch-differentiable equality constraints

**Critical:** the PyTorch methods must compute the *same* function as the NumPy solver. They are used only in the backward pass for KKT matrix construction.

```python
import numpy as np
import torch
from scipy.optimize import minimize
from hecopt import NonLinearOptModel, OptSolution, NonLinearPtOLayer

class PortfolioModel(NonLinearOptModel):
    """Markowitz portfolio: min variance s.t. return ≥ r_target, weights sum to 1."""

    def __init__(self, n_assets: int, r_target: float):
        self.n_assets = n_assets
        self.r_target = r_target

    @property
    def n_vars(self) -> int:
        return self.n_assets

    @property
    def n_eq_constraints(self) -> int:
        return 1   # Σ wᵢ = 1

    @property
    def n_ineq_constraints(self) -> int:
        return 1   # μᵀw ≥ r_target  →  r_target - μᵀw ≤ 0

    def solve(self, theta: np.ndarray) -> OptSolution:
        """theta = [μ₁,…,μₙ, Σ₁₁,Σ₁₂,…,Σₙₙ] (expected returns + covariance)."""
        n = self.n_assets
        mu = theta[:n]
        Sigma = theta[n:].reshape(n, n)

        def variance(w):
            return float(w @ Sigma @ w)

        def variance_grad(w):
            return 2 * Sigma @ w

        result = minimize(
            variance, np.ones(n) / n,
            jac=variance_grad,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=[
                {"type": "eq",  "fun": lambda w: w.sum() - 1.0},
                {"type": "ineq","fun": lambda w: mu @ w - self.r_target},
            ],
            options={"ftol": 1e-10},
        )

        w_star = result.x
        return OptSolution(
            x_star=w_star,
            obj_val=float(w_star @ Sigma @ w_star),
            y_star=np.array([result.v[0]] if hasattr(result, 'v') else [0.0]),
            z_star=np.array([result.v[1]] if hasattr(result, 'v') and len(result.v)>1 else [0.0]),
            active_ineq=np.array([mu @ w_star - self.r_target < 1e-4]),
            success=result.success,
        )

    def objective_torch(self, x, theta):
        n = self.n_assets
        Sigma = theta[n:].reshape(n, n)
        return x @ Sigma @ x    # portfolio variance

    def eq_constraints_torch(self, x, theta):
        return (x.sum() - 1.0).unsqueeze(0)   # Σ wᵢ = 1

    def ineq_constraints_torch(self, x, theta):
        mu = theta[:self.n_assets]
        return (self.r_target - mu @ x).unsqueeze(0)   # r_target − μᵀw ≤ 0

# Use just like the built-in pricing model
model = PortfolioModel(n_assets=5, r_target=0.08)
layer = NonLinearPtOLayer(model, tikhonov=1e-5)

theta_pred = torch.randn(8, model.n_vars * (model.n_vars + 1), requires_grad=True)
x_star = layer(theta_pred)          # differentiable via KKT
x_star.sum().backward()
```

---

## 11. Full Training Examples

Two complete end-to-end training scripts are provided in the `examples/` directory.

### `examples/shortest_path_example.py`

Trains a neural network with SPO+ loss on the grid shortest path problem, with a final comparison against an MSE baseline.

```bash
python examples/shortest_path_example.py
```

Output format:
```
Epoch  Train Loss   Test Regret
    1      0.2341        0.1823
    5      0.1102        0.0921
   20      0.0453        0.0312
MSE baseline test regret: 0.0891
```

Key hyperparameters you can tune in the script:

| Variable | Default | Effect |
|---|---|---|
| `GRID_SIZE` | 5 | Problem complexity |
| `METHOD` | `"spo_plus"` | `"spo_plus"` or `"pfyl"` |
| `LAMBDA_HYBRID` | 0.1 | MSE weight during training |
| `N_EPOCHS` | 20 | Training duration |
| `LR` | 5e-3 | Learning rate |

### `examples/pricing_example.py`

Trains a neural network with KKT-based decision loss on the 3-product pricing problem.

```bash
python examples/pricing_example.py
```

---

## 12. Comparison with Existing Libraries

| Feature | HecOpt | PyEPO | CVXPYLayers |
|---|---|---|---|
| **Combinatorial (LP/MIP)** | SPO+, PFYL | SPO+, PFYL, NCE | Partial |
| **Non-linear NLP** | KKT adjoint | — | Convex NLP only |
| **Non-convex problems** | Supported (with regularisation) | — | — |
| **Active-set tracking** | Automatic | — | Partial |
| **Hybrid MSE+DFL loss** | Built-in | Separate | — |
| **Custom solver plug-in** | Any solver via ABC | Limited | CVXPY only |
| **Dependencies** | torch, numpy, scipy | torch, numpy, scipy, pyepo | torch, cvxpy |
| **Regret evaluation** | Built-in | Built-in | Manual |
| **GPU neural network** | Yes (solver on CPU) | Yes (solver on CPU) | Yes |

**When to use HecOpt vs alternatives:**
- **HecOpt**: you need both combinatorial and non-linear support; or your problem is non-convex; or you want a single, lightweight dependency.
- **PyEPO**: you need more advanced combinatorial loss variants (NCE, blackbox perturbations) or direct Gurobi integration.
- **CVXPYLayers**: your NLP is provably convex and you want exact KKT differentiability without worrying about active-set degeneracy.

---

## 13. Design Decisions and Trade-offs

### Why SciPy instead of Gurobi / IPOPT?

SciPy ships with every Python installation — no license, no extra install step. The HiGHS LP solver (used for shortest path via `linprog`) and SLSQP (used for pricing via `minimize`) are state-of-the-art for small-to-medium problems. For large-scale industrial use, HecOpt's architecture makes it straightforward to swap in any external solver: you only need to override `solve()` in your model subclass.

### Why the adjoint method instead of full Jacobian?

The full Jacobian `∂x*/∂θ` has shape `[n_vars, n_params]`. In neural network backprop we only ever need the VJP `vᵀ(∂x*/∂θ)` for a scalar upstream gradient. The adjoint method computes exactly this at the cost of one `n_kkt × n_kkt` linear solve — significantly cheaper for large problems and numerically more stable.

### Why Tikhonov regularisation?

Real-world NLPs frequently produce degenerate KKT matrices from:
- **Near-active constraints** (where `g_j(x*) ≈ 0` but `μⱼ ≈ 0` too)
- **Non-convex Hessians** (indefinite `∇²L`)
- **Poorly-conditioned problems** at certain parameter values

Ridge regularisation `K_reg = K + εI` guarantees invertibility at the cost of a slightly approximate gradient. The default `ε = 1e-6` is virtually invisible in practice; increase it if you see NaN gradients or use `torch.linalg.lstsq` (the automatic fallback when `solve` fails).

### Why `lambda_hybrid > 0` in early training?

In the first few epochs of DFL training, the network output is essentially random. The SPO+ gradient signal exists only where the SPO solution and the predicted solution differ — which in early training may be inconsistent across batches. Adding even 5–10% MSE provides a smooth, consistent gradient that "warms up" the network before the DFL signal takes over. Anneal `lambda_hybrid` towards 0 as training progresses for best results.

### Batch processing

Both layers process instances **independently** in a Python loop. This is intentional: combinatorial solvers are not vectorised, and KKT systems differ per instance. For large batches, the dominant cost is the solver calls (O(batch_size) LP/NLP solutions). If solver throughput is a bottleneck, consider using `multiprocessing` to parallelise the forward pass.

---

## 14. References

**Decision-Focused Learning / Predict-Then-Optimize:**

- Elmachtoub, A. N., & Grigas, P. (2022). *Smart Predict-then-Optimize.* Operations Research, 70(1), 102–119.
- Wilder, B., Dilkina, B., & Tambe, M. (2019). *Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization.* AAAI.
- Mandi, J., Stuckey, P. J., Guns, T., et al. (2020). *Smart Predict-and-Optimize for Hard Combinatorial Optimization Problems.* AAAI.

**Perturbed Optimizers / Fenchel-Young Losses:**

- Berthet, Q., Blondel, M., Teboul, O., Cuturi, M., Vert, J.-P., & Bach, F. (2020). *Learning with Differentiable Perturbed Optimizers.* NeurIPS.
- Blondel, M., Martins, A. F. T., & Niculae, V. (2020). *Learning with Fenchel-Young Losses.* JMLR, 21(35), 1–69.

**KKT Implicit Differentiation:**

- Amos, B., & Kolter, J. Z. (2017). *OptNet: Differentiable Optimization as a Layer in Neural Networks.* ICML.
- Agrawal, A., Amos, B., Barratt, S., Boyd, S., Diamond, S., & Kolter, J. Z. (2019). *Differentiable Convex Optimization Layers.* NeurIPS.
- Blondel, M., Berthet, Q., Cuturi, M., Frostig, R., Hoyer, S., Llinares-López, F., Pedregosa, F., & Vert, J.-P. (2022). *Efficient and Modular Implicit Differentiation.* NeurIPS.

**Adjoint Sensitivity Methods:**

- Pontryagin, L. S., et al. (1962). *The Mathematical Theory of Optimal Processes.* Interscience.
- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). *Neural Ordinary Differential Equations.* NeurIPS.

**Benchmark Problems:**

- Tang, B., & Khalil, E. B. (2022). *PyEPO: A PyTorch-based End-to-End Predict-then-Optimize Library for Linear and Integer Programming.* arXiv:2206.14234.

---

## License

MIT License. See `LICENSE` for details.

---

## Contributing

Contributions are welcome. To add a new optimization problem, subclass `CombinatorialOptModel` or `NonLinearOptModel` and open a pull request. To add a new surrogate loss, implement `forward()` and `backward()` as a `torch.autograd.Function` and wrap it in an `nn.Module`.

Please run the full test suite before submitting:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
