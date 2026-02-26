"""
Baseline 1 (Linear/Combinatorial): Shortest Path on a Grid Graph.

This is the canonical benchmark for combinatorial Decision-Focused Learning,
popularised by Elmachtoub & Grigas (2022) and the PyEPO library.

Problem description
-------------------
Given an m×m directed grid graph where edges go *right* (→) and *down* (↓),
find the minimum-cost path from the top-left node (source) to the bottom-right
node (sink).

    Decision variables:  x_e ∈ {0, 1}  for each edge e
    Objective:           min  θᵀ x   (θ = predicted edge costs)
    Constraints:         flow conservation at each node

Because the constraint matrix is totally dual integral (TDI), the LP
relaxation always has an integral optimum, so we solve a simple LP with
SciPy and round the result.

Synthetic dataset
-----------------
``ShortestPathDataset`` generates (feature, cost, solution) triples where
the true edge cost is a polynomial function of the features plus noise.
This simulates the practical setting where a neural network must learn
θ̂(features) ≈ θ_true before passing the prediction to the solver.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.optimize import linprog

from hecopt.core.base import CombinatorialOptModel, OptSolution


class ShortestPathModel(CombinatorialOptModel):
    """Shortest path on an (m × m) directed grid graph.

    Parameters
    ----------
    grid_size:
        Side length of the square grid.  The graph has ``grid_size²`` nodes
        and ``2 * grid_size * (grid_size - 1)`` edges.  Default: ``5``.
    """

    def __init__(self, grid_size: int = 5) -> None:
        if grid_size < 2:
            raise ValueError("grid_size must be ≥ 2")
        self.grid_size = grid_size
        m = grid_size

        # Build edge list: (from_node, to_node) — right edges then down edges
        self._edges: list[tuple[int, int]] = []
        for r in range(m):
            for c in range(m):
                node = r * m + c
                if c + 1 < m:                    # right →
                    self._edges.append((node, node + 1))
                if r + 1 < m:                    # down ↓
                    self._edges.append((node, node + m))

        self._n_edges = len(self._edges)
        self._n_nodes = m * m
        self._source = 0
        self._sink = self._n_nodes - 1

        # Precompute flow-conservation equality constraint matrix
        self._A_eq, self._b_eq = self._build_flow_constraints()

    # ------------------------------------------------------------------
    # Flow constraint builder
    # ------------------------------------------------------------------

    def _build_flow_constraints(self) -> tuple[np.ndarray, np.ndarray]:
        """Build Ax = b encoding flow conservation at every node.

        Source sends 1 unit of flow; sink receives 1 unit.
        """
        n_nodes = self._n_nodes
        n_edges = self._n_edges
        source = self._source
        sink = self._sink

        A = np.zeros((n_nodes, n_edges), dtype=np.float64)
        b = np.zeros(n_nodes, dtype=np.float64)

        for j, (u, v) in enumerate(self._edges):
            A[u, j] += 1.0   # outflow from u
            A[v, j] -= 1.0   # inflow to v

        # Source: net outflow = 1
        b[source] = 1.0
        # Sink: net outflow = -1 (i.e., inflow = 1)
        b[sink] = -1.0
        # Internal nodes: b[i] = 0 (already zero)

        return A, b

    # ------------------------------------------------------------------
    # BaseOptModel interface
    # ------------------------------------------------------------------

    @property
    def n_vars(self) -> int:
        return self._n_edges

    @property
    def n_edges(self) -> int:
        return self._n_edges

    def solve(self, theta: np.ndarray) -> OptSolution:
        """Find the shortest path via LP relaxation.

        Parameters
        ----------
        theta:
            Edge cost vector ``[n_edges]``.

        Returns
        -------
        OptSolution
            ``x_star`` is a binary vector selecting the shortest-path edges.
        """
        result = linprog(
            c=theta.astype(np.float64),
            A_eq=self._A_eq,
            b_eq=self._b_eq,
            bounds=[(0.0, 1.0)] * self._n_edges,
            method="highs",
        )

        if not result.success:
            return OptSolution(
                x_star=np.zeros(self._n_edges),
                obj_val=float("inf"),
                success=False,
                message=result.message,
            )

        x = np.round(result.x).astype(np.float64)
        return OptSolution(x_star=x, obj_val=float(theta @ x), success=True)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def path_edges(self, x: np.ndarray) -> list[tuple[int, int]]:
        """Return the list of (from, to) node pairs on the selected path."""
        return [self._edges[j] for j in range(self._n_edges) if x[j] > 0.5]

    def path_cost(self, x: np.ndarray, c: np.ndarray) -> float:
        """Evaluate the actual cost of path ``x`` under cost vector ``c``."""
        return float(c @ x)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


class ShortestPathDataset(torch.utils.data.Dataset):
    """Synthetic (feature → edge costs → shortest path) dataset.

    The true edge cost is generated as a polynomial function of a feature
    vector plus i.i.d. Gaussian noise, mimicking a realistic supervised
    learning / predict-then-optimize pipeline.

    Parameters
    ----------
    n_samples:
        Number of instances.
    grid_size:
        Grid side length (same as ``ShortestPathModel``).
    n_features:
        Dimension of the input feature vector.
    degree:
        Polynomial degree of the cost–feature mapping.
        ``1`` = linear, ``2`` = quadratic (harder to predict).
    noise_std:
        Standard deviation of additive Gaussian noise on true costs.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        grid_size: int = 5,
        n_features: int = 4,
        degree: int = 1,
        noise_std: float = 0.1,
        seed: Optional[int] = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.model = ShortestPathModel(grid_size)
        n_edges = self.model.n_edges

        # Random projection matrix: features → edge costs
        B = rng.standard_normal((n_edges, n_features))

        features = rng.uniform(-1.0, 1.0, (n_samples, n_features))

        if degree == 1:
            raw_costs = features @ B.T
        else:
            feat_aug = np.concatenate([features, features ** 2], axis=1)
            B_aug = rng.standard_normal((n_edges, feat_aug.shape[1]))
            raw_costs = feat_aug @ B_aug.T

        raw_costs += noise_std * rng.standard_normal((n_samples, n_edges))

        # Shift costs to be strictly positive (path costs must be well-defined)
        costs = raw_costs - raw_costs.min(axis=1, keepdims=True) + 0.1

        # Compute optimal solutions for every instance
        solutions = np.array([self.model.solve(costs[i]).x_star for i in range(n_samples)])

        self.features = torch.tensor(features, dtype=torch.float32)
        self.costs = torch.tensor(costs, dtype=torch.float32)
        self.solutions = torch.tensor(solutions, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        """Return (features, true_costs, optimal_solution)."""
        return self.features[idx], self.costs[idx], self.solutions[idx]
