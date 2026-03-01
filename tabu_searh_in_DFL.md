# Differentiable Tabu Search: Local Search MCMC Layers

Integrating Tabu Search directly into a Decision-Focused Learning (DFL) pipeline represents a major leap in handling NP-hard combinatorial problems where exact solvers or continuous relaxations (like those used in DYS-Net) fall short. 

In standard DFL, predicting parameters like $\hat{c}$ and differentiating through the exact optimizer $x^*(\hat{c})$ often requires dealing with piecewise-constant functions and uninformative gradients. While methods like Differentiable Perturbed Optimizers (DPO) and Differentiable Black-Box (DBB) smoothing exist, they still require calling an expensive exact solver multiple times per backward pass. 

Recent literature (specifically the *Local Search MCMC Layers* framework) bypasses this by converting heuristic solvers into rigorous, differentiable Markov Chain Monte Carlo (MCMC) samplers. Here is the comprehensive documentation on how Tabu Search is mathematically modeled as a Markov Chain to compute end-to-end gradients.

---

## 1. The Core Concept: Local Search as MCMC

At its heart, local search explores a discrete space $Y$ of feasible solutions by iteratively moving from a current solution $y \in Y$ to a neighboring solution $y' \in N(y)$. 

To make this differentiable, researchers map local search to **Metropolis-Hastings MCMC**. Instead of strictly moving to the best neighbor (hill-climbing), the neighborhood system $N(y)$ is converted into a **proposal distribution** $q(y'|y)$. The network's predicted costs $\hat{c}$ parameterize a target Boltzmann distribution $p_\theta(y) \propto \exp(-\hat{c}^\top y)$. 



By doing this, the local search acts as a stochastic sampler, generating solutions that follow the probability distribution defined by the neural network's predictions.

---

## 2. Modeling Tabu Search as a Markov Chain

A standard Markov Chain satisfies the **Markov Property**: the next state depends *only* on the current state, not the sequence of events that preceded it. 

Standard Tabu Search intentionally violates this by maintaining a memory (the "Tabu list") of the last $T$ visited states to prevent cycling. To model Tabu Search as a rigorous Markov Chain, the **state space itself must be expanded via a Cartesian product**.

### 2.1 The Expanded State Space
Let $Y$ be the set of feasible combinatorial solutions. Instead of the Markov Chain operating on $Y$, we define a new expanded state space $\mathcal{S} \subseteq Y^T$, which represents sequences of solutions of length $T$ (the size of the Tabu list).

A single state in this new Markov Chain at step $k$ is a history vector:
$$\mathbf{s}_k = (y_{k-T+1}, y_{k-T+2}, \dots, y_k)$$

### 2.2 The Neighborhood Operator for Sequences
The Tabu list dynamically restricts the neighborhood. For an expanded state $\mathbf{s}_k$, the neighborhood $\mathcal{N}(\mathbf{s}_k)$ is defined as the set of all valid sequences resulting from appending a new, non-tabu neighbor and dropping the oldest memory:

$$\mathcal{N}(\mathbf{s}_k) = \left\{ (y_{k-T+2}, \dots, y_k, y') \mid y' \in N(y_k) \setminus \{y_{k-T+1}, \dots, y_k\} \right\}$$



This formulation elegantly restores the Markov Property: the transition to the next state sequence $\mathbf{s}_{k+1}$ is strictly dependent *only* on the current state sequence $\mathbf{s}_k$, encapsulating the entire Tabu logic within the expanded state definition.

---

## 3. Transition Probabilities and Metropolis-Hastings

With the state space formally defined as $Y^T$, Tabu Search can now be executed as an MCMC layer. 

### 3.1 The Proposal Distribution
The proposal distribution $q(\mathbf{s}_{k+1} | \mathbf{s}_k)$ defines the probability of suggesting a new Tabu state sequence. Typically, this is defined uniformly over the valid, non-tabu neighbors:

$$q(\mathbf{s}_{k+1} | \mathbf{s}_k) = \frac{1}{|\mathcal{N}(\mathbf{s}_k)|}$$

### 3.2 The Target Distribution
The neural network outputs continuous predicted costs $\hat{c}$. These costs define an Energy-Based Model (EBM) over the individual solutions $y$. We extend this to the sequence state space by defining the energy of a sequence simply as the energy of its most recent solution $y_k$:

$$E_\theta(\mathbf{s}_k) = \hat{c}^\top y_k$$
$$p_\theta(\mathbf{s}_k) = \frac{\exp(-E_\theta(\mathbf{s}_k))}{Z_\theta}$$

### 3.3 The Acceptance Ratio
When the Tabu-MCMC layer proposes a move to a new sequence $\mathbf{s}_{k+1}$, it accepts or rejects the move based on the Metropolis-Hastings acceptance probability $\alpha$:

$$\alpha(\mathbf{s}_k \to \mathbf{s}_{k+1}) = \min \left( 1, \frac{\exp(-\hat{c}^\top y_{k+1}) \cdot q(\mathbf{s}_k | \mathbf{s}_{k+1})}{\exp(-\hat{c}^\top y_k) \cdot q(\mathbf{s}_{k+1} | \mathbf{s}_k)} \right)$$

This stochastic acceptance allows the Tabu Search to explore the landscape defined by the network while avoiding local minima.

---

## 4. Differentiating the Tabu-MCMC Layer

The ultimate goal is to pass gradients back to the neural network parameters $\theta$. Because the Tabu layer is an MCMC sampler targeting the Boltzmann distribution $p_\theta$, it inherently minimizes a **Fenchel-Young Loss** or a Negative Log-Likelihood.

The gradient of the expected energy with respect to the network's predictions $\hat{c}$ relies on the difference between the actual observed optimal solutions (or best cached solutions) and the solutions sampled by the Tabu-MCMC layer:

$$\nabla_{\hat{c}} \mathcal{L} \approx \mathbb{E}_{\mathbf{s} \sim \text{Tabu-MCMC}} [y] - y_{true}$$

### Why this is powerful:
1. **No Exact Solvers:** You never run an ILP solver during training. You only run a fast, truncated Tabu Search.
2. **Informative Gradients:** Because the Tabu search avoids cycling and samples broadly, the expectation $\mathbb{E}[y]$ provides a smooth, dense gradient signal, avoiding the zero-gradient problem of piecewise-constant LP mappings.
3. **Control over Variance:** Increasing the number of MCMC chains or iterations provides finer gradient estimates without the exponential time scaling of exact solvers.

---

## 5. Architectural Summary

If you were to implement this in a pipeline, it replaces the `x_hat = solver(c_hat)` step entirely.

```text
                    ┌─────────────────────────────────────────────┐
   Features z ──►   │  Prediction Network  m_θ(z)                 │
                    │  (Standard Neural Network)                  │
                    └──────────────┬──────────────────────────────┘
                                   │  ĉ (predicted costs/energy)
                                   ▼
                    ┌─────────────────────────────────────────────┐
                    │  Tabu-MCMC Layer (Differentiable)           │
                    │  1. Init H_0 = (y_0, ..., y_T)              │
                    │  2. Propose neighbor y' NOT in H_t          │
                    │  3. Accept/Reject via exp(-ĉ^T y')          │
                    │  4. Update H_{t+1} (shift window)           │
                    └──────────────┬──────────────────────────────┘
                                   │  Samples y ~ p_θ
                                   ▼
                    ┌─────────────────────────────────────────────┐
                    │  Gradient Computation                       │
                    │  ∇_ĉ L ≈ (Average Sampled y) - y_true       │
                    └──────────────┬──────────────────────────────┘
                                   │  ∂L/∂ĉ 
                                   ▼
                    Backpropagate to update θ