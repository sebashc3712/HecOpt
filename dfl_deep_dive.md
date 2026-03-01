# Decision-Focused Learning: A Comprehensive Technical Deep Dive

## 1. Foundational Concepts

### 1.1 The Predict-Then-Optimize (PtO) Paradigm

Most real-world decision-making under uncertainty follows a two-stage process:

1. **Prediction Stage**: A machine learning model `m_θ(z)` maps features `z` to predicted parameters `ĉ`
2. **Optimization Stage**: A solver finds the optimal decision `x*(ĉ) = argmin_{x ∈ S} ĉᵀx`

The **standard (two-stage) approach** trains these independently: the ML model minimizes prediction error (e.g., MSE between `ĉ` and true `c`), then the optimizer takes `ĉ` as input. This is also called **Prediction-Focused Learning (PFL)**.

**The core problem**: Minimizing prediction error does *not* necessarily minimize decision error. A prediction that is slightly wrong in a "decision-critical" dimension can lead to terrible decisions, while a prediction that is very wrong in a "decision-irrelevant" dimension may produce the same optimal decision.

### 1.2 Decision-Focused Learning (DFL) — The Paradigm Shift

DFL trains the ML model end-to-end, directly optimizing the quality of the resulting decisions:

```
Features z → ML Model m_θ(z) → Predicted params ĉ → Optimizer x*(ĉ) → Task Loss ℒ_task
                ↑                                                            |
                └────────────── Backpropagate gradients ─────────────────────┘
```

**Key insight**: The accuracy of `ĉ` is *not* the primary focus. The focus is on the error incurred *after optimization*. DFL trains `θ` to minimize a **task loss** (or **decision loss**) that measures how good the resulting decisions are.

### 1.3 Formal Problem Definition

Given:
- Feature vector: `z ∈ ℝᵖ`
- Unknown cost/parameter vector: `c ∈ ℝᵈ` (to be predicted)
- ML model: `m_θ: ℝᵖ → ℝᵈ` parameterized by `θ`, producing `ĉ = m_θ(z)`
- Constrained optimization problem: `x*(ĉ) = argmin_{x ∈ S} ĉᵀx` where `S = {x | Ax = b, x ≥ 0}` (or more generally any feasible set)
- Training data: `{(zᵢ, cᵢ)}ᵢ₌₁ᴺ`

**Prediction-Focused objective** (two-stage):
```
min_θ  Σᵢ ℒ_pred(m_θ(zᵢ), cᵢ)     e.g., ℒ_pred = ||ĉ - c||²
```

**Decision-Focused objective**:
```
min_θ  Σᵢ ℒ_task(x*(m_θ(zᵢ)), cᵢ)
```

The most natural task loss is **regret**:
```
Regret(ĉ, c) = cᵀx*(ĉ) - cᵀx*(c)
```
This measures how much worse the decision based on `ĉ` is compared to the decision that would have been made with perfect information `c`. Regret is always ≥ 0.

---

## 2. The Core Technical Challenge: Differentiating Through the Optimizer

### 2.1 Why This Is Hard

To train end-to-end via gradient descent, we need:
```
∂ℒ_task/∂θ = (∂ℒ_task/∂x*) · (∂x*/∂ĉ) · (∂ĉ/∂θ)
```

The critical bottleneck is `∂x*/∂ĉ` — the Jacobian of the optimization mapping. For:

- **Linear Programs (LPs)**: `x*(ĉ)` is piecewise constant in `ĉ`. The gradient is **zero almost everywhere** and **undefined at discontinuities** (where the optimal vertex changes). This is the fundamental challenge — not that gradients don't exist, but that they are *uninformative*.

- **Integer Linear Programs (ILPs)**: The discrete feasible space makes `x*(ĉ)` a step function — even worse than LPs.

- **Convex QPs**: Smoother behavior, but KKT-based differentiation can be expensive.

### 2.2 Taxonomy of DFL Methods

The field has developed four major categories of gradient-based approaches, plus gradient-free methods:

```
DFL Methods
├── Gradient-Based
│   ├── (1) Analytical Differentiation of Optimization Mappings
│   │       (OptNet, CvxpyLayers, KKT-based)
│   ├── (2) Analytical Smoothing of Optimization Mappings
│   │       (QPTL, Interior Point methods, regularization)
│   ├── (3) Smoothing by Random Perturbations
│   │       (DBB, DPO/Berthet, Perturb-and-MAP, IMLE)
│   └── (4) Surrogate Loss Functions
│           (SPO+, Contrastive losses, Learning-to-Rank, LODL, LANCER)
└── Gradient-Free
        (LODL, Contrastive, Caching methods)
```

---

## 3. Method Category 1: Analytical Differentiation (KKT-Based)

### 3.1 OptNet (Amos & Kolter, 2017)

**Key idea**: Embed QP layers directly into neural networks and differentiate via KKT conditions.

For a QP: `x* = argmin_{x} ½xᵀQx + qᵀx  s.t.  Ax = b, Gx ≤ h`

The KKT conditions define the optimality:
```
Qx* + q + Aᵀν* + Gᵀλ* = 0
Ax* - b = 0
diag(λ*)(Gx* - h) = 0
```

Implicit differentiation of the KKT system gives:
```
[Q   Aᵀ   Gᵀ ] [dx*]     [dq + dQx*        ]
[A   0    0  ] [dν*]  = - [dAx* - db          ]
[Λ*G 0  D(Gx*-h)] [dλ*]  [diag(λ*)(dGx* - dh)]
```

This linear system can be solved to get `∂x*/∂q`, etc.

**Strengths**: Exact gradients for strictly convex QPs.
**Limitations**: Only QPs; computational cost O(d³) per sample; requires Q ≻ 0.

### 3.2 CvxpyLayers (Agrawal et al., 2019)

Generalizes OptNet to *any* disciplined convex program by:
1. Converting any CVXPY problem to a cone program
2. Differentiating the cone program via its KKT conditions
3. Mapping gradients back to original problem parameters

```python
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

x = cp.Variable(n)
c_param = cp.Parameter(n)
constraints = [A @ x <= b, x >= 0]
problem = cp.Problem(cp.Minimize(c_param @ x), constraints)
layer = CvxpyLayer(problem, parameters=[c_param], variables=[x])

# In training loop:
x_star, = layer(c_hat)  # differentiable forward pass
loss = true_cost @ x_star
loss.backward()  # gradients flow through the solver
```

**Strengths**: Very general (any convex program); clean API.
**Limitations**: Cannot handle integer variables; computational overhead.

### 3.3 Task-Based End-to-End Learning (Donti, Amos & Kolter, 2017)

The pioneering work that demonstrated DFL for stochastic optimization. Used OptNet-style differentiation for energy scheduling and portfolio optimization, showing that task-aware training improves decision quality even when prediction accuracy decreases.

---

## 4. Method Category 2: Analytical Smoothing

### 4.1 The Smoothing Idea

Since LP solutions are piecewise constant, add a regularizer to make the mapping smooth:

**Original LP**: `x*(ĉ) = argmin_{x ∈ S} ĉᵀx`  (piecewise constant, zero gradients)

**Smoothed QP**: `x*_μ(ĉ) = argmin_{x ∈ S} ĉᵀx + μ/2 ||x||²`  (now differentiable)

As `μ → 0`, the smoothed solution approaches the LP solution, but the trade-off is: larger `μ` gives better gradients but less accurate solutions.

### 4.2 QPTL (Wilder, Dilkina & Tambe, 2019)

"Melding the Data-Decisions Pipeline" — a seminal DFL paper.

- Adds quadratic regularization `μ/2 ||x||²` to the LP objective
- For combinatorial problems with integer variables, uses continuous (multilinear) relaxation
- Differentiates the resulting QP using OptNet
- Shows significant improvements on combinatorial problems like bipartite matching, diverse matching

**Key contribution**: First general framework for DFL with combinatorial optimization.

### 4.3 Interior Point Methods (Mandi & Guns, 2020)

- Uses interior point method steps to smooth the LP
- Adds a logarithmic barrier: `x*(ĉ) = argmin_{x ∈ S} ĉᵀx - μ Σᵢ log(xᵢ)`
- Differentiates through the barrier-smoothed problem

### 4.4 Important Caveat: Smoothing ≠ Useful Gradients

Recent work (Mandi et al., 2025) shows that even after smoothing an LP into a QP, the **regret** remains constant across large regions of the parameter space. Thus, minimizing regret directly through a smoothed solver can still yield zero gradients. The recommendation is to minimize **surrogate losses** (like SPO+ or SCE) even when using differentiable optimization layers.

---

## 5. Method Category 3: Smoothing by Random Perturbations

### 5.1 Differentiable Black-Box (DBB) Solver (Pogančić et al., 2020)

**Key insight**: The fundamental problem is not that gradients don't exist — they're zero almost everywhere. DBB replaces the piecewise-constant mapping with a piecewise-linear interpolation.

**Forward pass**: Solve `x*(ĉ)` using any black-box solver.

**Backward pass**: Instead of the true (zero) gradient, compute:
```
∂x*/∂ĉ ≈ (x*(ĉ) - x*(ĉ + λ · ∂ℒ/∂x*)) / λ
```
This requires solving the optimization problem on a *perturbed* input `ĉ + λ · ∂ℒ/∂x*` and using the finite difference as an approximate gradient.

**Strengths**: Works with any solver (truly black-box); handles combinatorial problems.
**Limitations**: Requires an additional solver call per backward pass; hyperparameter λ.

### 5.2 Differentiable Perturbed Optimizers (DPO) (Berthet et al., 2020)

**Key idea**: Replace the deterministic optimizer with its stochastic perturbation:
```
f_ε(ĉ) = 𝔼[x*(ĉ + ε·η)]    where η ~ N(0, I) or Gumbel
```

This expected solution is smooth in `ĉ` (by the smoothing effect of the expectation), and its gradient can be estimated via:
```
∂f_ε/∂ĉ ≈ (1/Mε) Σₘ x*(ĉ + ε·ηₘ) · ηₘᵀ
```

**Key properties**:
- Works with *any* optimization problem (LP, ILP, combinatorial)
- Temperature `ε` controls the smoothing-accuracy trade-off
- Associated **Fenchel-Young loss** provides a principled loss function
- Gradient estimation via Monte Carlo sampling of `M` perturbations

**Strengths**: Very general; principled theoretical foundation.
**Limitations**: Requires `M` solver calls for gradient estimation; variance increases as `ε → 0`.

### 5.3 IMLE (Niepert et al., 2021)

Implicit Maximum Likelihood Estimation — improves perturbation sampling by targeting perturbations that are most informative for learning.

### 5.4 Identity with Projection (I+P) (Sahoo et al., 2022)

Simplifies the backward pass by using the identity function as a Jacobian surrogate, followed by projection onto the feasible set.

---

## 6. Method Category 4: Surrogate Loss Functions

### 6.1 SPO+ Loss (Elmachtoub & Grigas, 2017/2022)

The **most influential** surrogate loss in DFL. Derived from duality theory.

**SPO Loss** (the ideal but intractable loss):
```
ℒ_SPO(ĉ, c) = cᵀx*(ĉ) - cᵀx*(c)    (= regret)
```
This is non-convex and non-continuous in `ĉ`.

**SPO+ Loss** (the convex surrogate):
```
ℒ_SPO+(ĉ, c) = max_{x ∈ S} {(c - 2ĉ)ᵀx} + 2ĉᵀx*(c) - cᵀx*(c)
            = -z*(2ĉ - c) + 2ĉᵀx*(c) - cᵀx*(c)
```

where `z*(q) = min_{x ∈ S} qᵀx`.

**Key properties**:
- **Convex** in `ĉ` (and therefore in `θ` for linear models)
- **Statistically consistent**: minimizing SPO+ asymptotically minimizes SPO loss
- **Fisher consistent**: under mild conditions
- Handles LPs, convex programs, and even MILPs with linear objectives
- **Subgradient**: `∂ℒ_SPO+/∂ĉ = 2(x*(2ĉ - c) - x*(c))`
- Requires solving one optimization per training sample: `x*(2ĉ - c)`

**Implementation in PyEPO**:
```python
import pyepo

# Define optimization model
optmodel = pyepo.model.grb.shortestPathModel(grid=(5,5))

# SPO+ loss function
spo_loss = pyepo.func.SPOPlus(optmodel, n_jobs=4)

# In training loop:
loss = spo_loss(pred_cost, true_cost, optimal_sol, optimal_obj)
loss.backward()
```

### 6.2 Contrastive Losses (Mulamba et al., 2021)

Uses **Noise Contrastive Estimation (NCE)** over a set of feasible solutions:
```
ℒ_NCE = -log[ exp(-cᵀx*(c)/τ) / Σ_{x̃ ∈ Pool} exp(-cᵀx̃/τ) ]
```

This encourages the optimizer to rank the true optimal solution higher than other feasible solutions. Paired with **solution caching** to maintain a pool of previously encountered solutions.

### 6.3 Learning-to-Rank Perspective (Mandi et al., 2022)

**Key observation**: DFL can be viewed as a **learning-to-rank** problem — the goal is to learn an objective function that ranks feasible solutions correctly (optimal first).

Three families of ranking losses:
- **Pointwise**: Predict the cost of each solution independently
- **Pairwise**: Ensure the optimal solution is ranked above suboptimal ones
- **Listwise**: Optimize the ranking over the entire list of solutions

**MAP (Maximum A Posteriori)** and **NCE** losses from this framework are differentiated in closed form given a subset of solutions, making them solver-free during backpropagation.

### 6.4 LODL — Locally Optimized Decision Losses (Shah et al., 2022)

**Key idea**: Instead of differentiating through the optimizer, *learn* a surrogate loss function.

Three steps:
1. **Generate inputs**: Sample predictions `ĉ` near true `c` via Gaussian perturbations
2. **Compute targets**: For each sample, solve the optimization and compute true regret
3. **Fit a loss model**: Learn a parametric function (e.g., quadratic, neural net) that approximates `Regret(ĉ, c)` locally around each training instance

**Properties**:
- Black-box: only needs an oracle to solve the optimization
- Can be convex by construction
- Replaces the optimizer entirely during training
- Three key phenomena it captures: (a) asymmetric costs of over/under-prediction, (b) irrelevance of some prediction dimensions, (c) interaction effects between dimensions

### 6.5 LANCER — Landscape Surrogate (Zharmagambetov et al., 2024)

Uses alternating optimization to jointly learn:
- A predictive model `c_θ(z)`
- A global surrogate loss model `M` that approximates the compound function `f ∘ g`

Employs a single neural network as a global surrogate (vs. LODL's per-instance surrogates).

### 6.6 SCE — Squared Cost Error (Mandi et al., 2025)

A recently proposed surrogate for use with differentiable optimization layers:
```
ℒ_SCE(x*(ĉ), c) = ||cᵀx*(ĉ) - cᵀx*(c)||²
```
Shown to provide informative gradients even in regions where regret has zero gradients.

---

## 7. Gradient-Free Approaches

### 7.1 Contrastive Predict-and-Optimize

Solution caching + contrastive loss avoids differentiating through the solver entirely. A cache of previously computed solutions is maintained and used to form contrastive training signals.

### 7.2 Score Function Gradient Estimation (SFGE)

Uses REINFORCE-style gradient estimation:
```
∇_θ 𝔼[ℒ] ≈ (1/M) Σₘ ℒ(x*(ĉₘ), c) · ∇_θ log p(ĉₘ|θ)
```
High variance but fully black-box.

---

## 8. Existing Libraries and Software

### 8.1 PyEPO (Tang & Khalil, 2023)

**The primary Python library** for DFL. PyTorch-based.

**Implemented methods**:
| Method | Type | Key Reference |
|--------|------|--------------|
| SPO+ | Surrogate loss | Elmachtoub & Grigas, 2022 |
| DBB | Black-box perturbation | Pogančić et al., 2020 |
| DPO (Perturbation) | Perturbation-based | Berthet et al., 2020 |
| NCE / Contrastive | Ranking-based loss | Mulamba et al., 2021 |
| LTR (Pointwise/Pairwise/Listwise) | Ranking losses | Mandi et al., 2022 |
| I+P | Identity + Projection | Sahoo et al., 2022 |
| PFYL (Perturbed Fenchel-Young) | Fenchel-Young loss | Berthet et al., 2020 |
| Two-Stage (MSE baseline) | Prediction loss | Standard |
| Robust losses | Robust regret | Schutte et al., 2023 |
| Directional Gradients | Directional DFL | Gupta & Huang, 2024 |

**Solver backends**: GurobiPy, Pyomo, Google OR-Tools, COPT, MPAX (JAX-based GPU solver)

**Architecture**:
```python
# 1. Define optimization model
class MyModel(optGrbModel):
    def _getModel(self):
        m = gp.Model()
        x = m.addVars(n, vtype=GRB.BINARY)
        m.setObjective(...)
        m.addConstrs(...)
        return m, x

# 2. Create dataset
dataset = pyepo.data.dataset.optDataset(optmodel, features, costs)

# 3. Choose DFL method
loss_fn = pyepo.func.SPOPlus(optmodel)
# or: pyepo.func.blackboxOpt(optmodel, lambd=10)
# or: pyepo.func.perturbedOpt(optmodel, n_samples=10, sigma=1.0)
# or: pyepo.func.NCE(optmodel, sol_pool_size=100)

# 4. Train
for x_batch, c_batch, w_batch, z_batch in dataloader:
    c_hat = pred_model(x_batch)
    loss = loss_fn(c_hat, c_batch, w_batch, z_batch)
    loss.backward()
    optimizer.step()
```

### 8.2 PredOpt Benchmarks (Mandi et al., 2024)

Reference benchmark implementation for the DFL survey paper. Implements 11 methods across 7 problem domains.

**GitHub**: `https://github.com/PredOpt/predopt-benchmarks`

**Benchmark problems**:
1. Shortest path (grid graphs)
2. Multi-dimensional knapsack
3. Traveling salesperson
4. Combinatorial portfolio optimization
5. Diverse bipartite matching
6. Energy-cost aware scheduling
7. Capacitated facility location

### 8.3 CvxpyLayers

```python
from cvxpylayers.torch import CvxpyLayer
# Any CVXPY problem → differentiable PyTorch layer
```

### 8.4 qpth (OptNet)

```python
from qpth.qp import QPFunction
# Batched differentiable QP solver on GPU
```

---

## 9. Benchmark Problems and Evaluation

### 9.1 Standard Metrics

- **Normalized Regret**: `(cᵀx*(ĉ) - cᵀx*(c)) / |cᵀx*(c)|` — decision quality
- **MSE/MAE**: Standard prediction accuracy (secondary metric)
- **Runtime**: Wall-clock training time (critical for practical adoption)

### 9.2 Canonical Problem Domains

#### Shortest Path
- Grid graph, southwest to northeast
- Edge costs are unknown, predicted from features
- LP formulation (totally unimodular → integer solutions from LP relaxation)
- Grid sizes: 5×5 (small) to 12×12 (large)

#### Multi-Dimensional Knapsack
- Binary variables, multiple capacity constraints
- NP-hard → requires ILP solver or heuristic
- Tests scalability of DFL methods to hard combinatorial problems

#### Portfolio Optimization
- Predict asset returns, then optimize portfolio allocation
- Both continuous (Markowitz) and combinatorial variants
- Direct financial interpretation of regret

#### Energy Scheduling
- Schedule tasks across machines to minimize energy cost
- Electricity prices are uncertain and must be predicted
- Strong practical relevance for renewable energy integration

### 9.3 Key Empirical Findings

From the comprehensive survey (Mandi et al., 2024):

1. **DFL methods consistently outperform two-stage** when models are misspecified (which is the common case in practice)
2. **SPO+** is the most robust and reliable method overall — works well across problem types with minimal tuning
3. **DBB** can underperform, especially on problems with non-linear objectives or MILP structure
4. **Perturbation methods (DPO)** offer good generality but require tuning temperature `ε` and number of samples `M`
5. **Ranking-based methods** are competitive and allow controlling runtime by limiting the solution pool size
6. **When models are well-specified** (i.e., the true data-generating process is captured by the model class), the gap between PFL and DFL narrows — two-stage can sometimes match or beat DFL
7. **Training time** is a major practical concern: DFL requires solving optimization problems in the training loop

---

## 10. Mathematical Details for Implementation

### 10.1 SPO+ Loss — Complete Derivation

**Setup**: `min_{x ∈ S} cᵀx` with `S` polyhedral.

**SPO Loss**:
```
ℓ_SPO(ĉ, c) = cᵀx*(ĉ) - z*(c)
```
where `z*(c) = min_{x ∈ S} cᵀx`.

**Deriving SPO+ via Duality**:

Starting from:
```
ℓ_SPO(ĉ, c) = max_{x ∈ W*(ĉ)} cᵀx - z*(c)
```
where `W*(ĉ) = argmin_{x ∈ S} ĉᵀx`.

Using Lagrangian duality on the constraint `x ∈ W*(ĉ)`:
```
= max_{x ∈ S} {cᵀx + α(z*(ĉ) - ĉᵀx)} - z*(c)    for some α ≥ 0
```

Setting `α = 2` and upper bounding:
```
ℓ_SPO+(ĉ, c) = max_{x ∈ S} {(c - 2ĉ)ᵀx} + 2z*(ĉ) - z*(c)
              = -z*(2ĉ - c) + 2ĉᵀx*(c) - z*(c)
```

Wait — more precisely:
```
ℓ_SPO+(ĉ, c) = max_{x ∈ S} (c - 2ĉ)ᵀx + 2ĉᵀx*(c) - cᵀx*(c)
```

**Subgradient**:
```
∂ℓ_SPO+/∂ĉ = 2(x*(c) - x*(2ĉ - c))
```

To compute this, you need to solve one additional optimization: `x*(2ĉ - c)`.

### 10.2 DBB Backward Pass

**Forward**: `x̂ = x*(ĉ)` (solve normally)

**Backward**: Given upstream gradient `∂ℒ/∂x̂`, compute:
```
x_λ = x*(ĉ + λ · ∂ℒ/∂x̂)    # solve perturbed problem
∂ℒ/∂ĉ = -(x̂ - x_λ) / λ       # finite difference approximation
```

Hyperparameter `λ > 0` controls interpolation smoothness.

### 10.3 DPO (Berthet) Gradient Estimation

**Forward**: `f_ε(ĉ) = 𝔼_η[x*(ĉ + ε·η)]` estimated as `(1/M) Σₘ x*(ĉ + ε·ηₘ)`

**Gradient**:
```
∂f_ε/∂ĉ ≈ (1/Mε) Σₘ x*(ĉ + ε·ηₘ) · ηₘᵀ
```
where `ηₘ ~ N(0, I)` or Gumbel distribution.

### 10.4 Fenchel-Young Loss (Blondel et al., 2020)

Associated with DPO:
```
ℒ_FY(ĉ, c) = Ω*(ĉ) + Ω(x*(c)) - ĉᵀx*(c)
```
where `Ω*` is the conjugate of the regularization function.

For perturbation-based smoothing:
```
Ω*(ĉ) = ε · 𝔼_η[max_{x ∈ S} (ĉ + ε·η)ᵀx]
```

### 10.5 Regret Normalization

For comparing across problems:
```
Normalized Regret = (cᵀx*(ĉ) - cᵀx*(c)) / |cᵀx*(c)|
```

---

## 11. Key Design Decisions for Your Library

### 11.1 Architecture Decisions

1. **Solver abstraction**: Support multiple solver backends (Gurobi, SCIP, OR-Tools, custom)
2. **PyTorch autograd integration**: Each DFL method should implement `torch.autograd.Function` with custom `forward()` and `backward()` methods
3. **Batched operations**: Parallel solver calls across training batch (critical for performance)
4. **Solution caching**: Cache previously computed solutions to avoid redundant solver calls
5. **GPU acceleration**: Consider MPAX or similar for GPU-native LP solving

### 11.2 Loss Functions to Implement (Priority Order)

| Priority | Method | Complexity | Performance |
|----------|--------|-----------|-------------|
| 1 | SPO+ | Low | Excellent, most robust |
| 2 | Two-Stage (MSE) | Trivial | Baseline |
| 3 | DBB | Low | Good, general |
| 4 | DPO/PFYL | Medium | Good, general |
| 5 | NCE/Contrastive | Medium | Good, flexible |
| 6 | LTR (Pairwise/Listwise) | Medium | Competitive |
| 7 | LODL | High | Can beat others |

### 11.3 Common Pitfalls in Implementation

1. **Zero gradients**: Even after smoothing, regret can have zero gradients. Use surrogate losses (SPO+, SCE) rather than direct regret minimization
2. **Solution stability**: Perturbation intensity during training can cause instability. Consider cost regularization (recent work, Jan 2025)
3. **Numerical precision**: LP solvers may return slightly different vertex solutions for near-degenerate problems — use tolerances carefully
4. **Warm-starting**: Reuse previous solutions as starting points for the solver to speed up training
5. **LP relaxation**: For ILPs, the LP relaxation often suffices for gradient computation even if the actual decision uses integer solutions
6. **Hyperparameter sensitivity**: DBB's `λ` and DPO's `ε` require careful tuning; SPO+ is more robust

### 11.4 Scalability Considerations

- **Solver calls dominate training time**: Each epoch requires O(N) solver calls (or O(N·M) for perturbation methods with M samples)
- **Parallel solving**: Essential for practical use. PyEPO uses `pathos` for multiprocessing
- **DYS-Net**: A recent fully-neural optimization layer that replaces the QP solver with feedforward neural operations — dramatically faster but approximate
- **Solution caching**: For contrastive/ranking methods, maintaining and reusing a solution cache avoids redundant solver calls
- **Surrogate models**: LODL and LANCER replace solver calls with cheap function evaluations during training

---

## 12. Recent Advances and Open Problems (2024-2025)

### 12.1 Recent Advances

- **Directional Gradients** (Gupta & Huang, 2024): Alternative gradient computation using directional derivatives
- **DYS-Net** (McKenzie et al., 2024): Fully-neural differentiable optimization, O(100×) faster training
- **Online DFL** (2025): Extension to non-stationary settings with dynamic regret bounds
- **Prediction Loss Guided DFL** (2025): Combining prediction and decision loss gradients via multi-objective optimization
- **Locally Convex Global Loss Networks** (LCGL, 2024): Learning convex surrogate losses globally
- **Robust Losses** (Schutte et al., 2023): Addressing the gap between empirical and expected regret
- **Solution Stability** (Jan 2025): Cost regularization to manage perturbation-induced instability

### 12.2 Real-World Applications

- **Energy storage optimization**: Battery scheduling with uncertain prices (SPO+ shown effective)
- **Portfolio optimization**: Asset allocation with predicted returns
- **Vehicle routing**: Routing under demand uncertainty
- **Resource allocation**: Public health, disaster response
- **Supply chain**: Inventory management, scheduling
- **Power systems**: Decision-dependent uncertainty, reserve market participation

### 12.3 Open Research Problems

1. **Scalability to large combinatorial problems**: Most benchmarks are small-scale; real-world problems have thousands of variables
2. **Non-linear objectives**: Most DFL theory assumes linear objectives; extending to non-linear is challenging
3. **Uncertain constraints**: Most work focuses on uncertain objectives; uncertain feasible sets are less explored
4. **Stochastic/robust DFL**: Handling distributional uncertainty in the prediction
5. **Multi-stage decisions**: Sequential decision-making under uncertainty (beyond single-shot optimization)
6. **When does DFL help?**: The "Price of Correlation" (Cameron et al., 2022) shows DFL helps most when prediction errors are correlated with decision sensitivity
7. **Feature-dependent constraints**: Constraints that depend on the same features as the objective

---

## 13. Key Papers — Reading List

| Year | Paper | Key Contribution |
|------|-------|-----------------|
| 2017 | Amos & Kolter, "OptNet" | Differentiable QP layers |
| 2017 | Donti, Amos & Kolter, "Task-based E2E learning" | First DFL for stochastic optimization |
| 2017 | Elmachtoub & Grigas, "Smart Predict then Optimize" | SPO+ loss function |
| 2019 | Wilder, Dilkina & Tambe, "Melding Data-Decisions" | QPTL, DFL for combinatorial problems |
| 2019 | Agrawal et al., "Differentiable Convex Optimization Layers" | CvxpyLayers |
| 2020 | Pogančić et al., "Differentiation of BB Combinatorial Solvers" | DBB method |
| 2020 | Berthet et al., "Learning with Differentiable Perturbed Optimizers" | DPO / perturbation methods |
| 2020 | Mandi & Guns, "Interior Point Solving for LP-based PtO" | Interior point DFL |
| 2021 | Mulamba et al., "Contrastive Losses and Solution Caching" | NCE + caching |
| 2022 | Mandi et al., "DFL: Through the Lens of Learning to Rank" | Ranking perspective |
| 2022 | Shah et al., "LODL" | Learned loss functions |
| 2022 | Cameron et al., "Perils of Learning Before Optimizing" | Price of Correlation analysis |
| 2023 | Tang & Khalil, "PyEPO" | Reference library |
| 2024 | Mandi et al., "DFL: Foundations, State of Art, Benchmark" | Comprehensive survey |
| 2024 | Gupta & Huang, "Directional Gradients" | New gradient approach |
| 2025 | Mandi et al., "Minimizing Surrogate Losses" | Surrogate > regret with diff. solvers |

---

## 14. Summary: The Decision-Focused Learning Landscape

```
                    ┌─────────────────────────────────┐
                    │     Problem: Predict-Then-Opt    │
                    │  min_{x∈S} c^T x  (c unknown)   │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────┴────────────────────┐
                    │                                  │
           ┌───────┴───────┐               ┌──────────┴──────────┐
           │   Two-Stage   │               │  Decision-Focused   │
           │  (PFL / MSE)  │               │      Learning       │
           └───────────────┘               └──────────┬──────────┘
                                                      │
                              ┌───────────┬───────────┼───────────┬──────────┐
                              │           │           │           │          │
                         ┌────┴────┐ ┌────┴────┐ ┌───┴────┐ ┌───┴───┐ ┌───┴────┐
                         │ KKT /   │ │Smoothing│ │Perturb.│ │Surrog.│ │Grad-   │
                         │Implicit │ │  + QP   │ │ DBB /  │ │ Loss  │ │ Free   │
                         │  Diff   │ │         │ │  DPO   │ │SPO+/  │ │LODL /  │
                         │OptNet / │ │ QPTL /  │ │  IMLE  │ │NCE/LTR│ │Caching │
                         │CvxpyLyr│ │ IntPt   │ │        │ │       │ │        │
                         └─────────┘ └─────────┘ └────────┘ └───────┘ └────────┘
                         Convex only  LP→QP      Any solver  Any prob  Black-box
                         Exact grad   Approx     M samples   1 solve   No diff
```

**Bottom line for your library**: Start with SPO+ (most robust, well-understood), add DBB and DPO for black-box solver support, implement two-stage MSE as baseline, and consider ranking-based methods for flexibility. Ensure batched parallel solver calls, solution caching, and clean PyTorch autograd integration.
