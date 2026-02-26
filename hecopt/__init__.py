"""
HecOpt — Unified End-to-End Predict-Then-Optimize Library
==========================================================

A PyTorch library for **Decision-Focused Learning (DFL)** that seamlessly
handles both:

* **Combinatorial / LP problems** (e.g. Shortest Path, TSP, Knapsack) via
  surrogate losses (SPO+, PFYL).
* **Non-linear continuous problems** (e.g. Pricing, Revenue Management) via
  KKT implicit differentiation (adjoint sensitivity method).

Quick start
-----------
Combinatorial (Shortest Path):

    >>> from hecopt import CombinatorialPtOLayer
    >>> from hecopt.baselines.shortest_path import ShortestPathModel, ShortestPathDataset
    >>> model = ShortestPathModel(grid_size=5)
    >>> layer = CombinatorialPtOLayer(model, method="spo_plus")

Non-linear (Pricing):

    >>> from hecopt import NonLinearPtOLayer
    >>> from hecopt.baselines.pricing import PricingModel, PricingDataset
    >>> model = PricingModel(n_products=3)
    >>> layer = NonLinearPtOLayer(model)
"""

from hecopt.core.base import (
    BaseOptModel,
    CombinatorialOptModel,
    NonLinearOptModel,
    OptSolution,
)
from hecopt.core.combinatorial import CombinatorialPtOLayer
from hecopt.core.nonlinear import NonLinearPtOLayer
from hecopt.losses.hybrid import HybridLoss
from hecopt.losses.pfyl import PFYLoss
from hecopt.losses.spo_plus import SPOPlusLoss

__version__ = "0.1.0"
__all__ = [
    # Core abstractions
    "BaseOptModel",
    "CombinatorialOptModel",
    "NonLinearOptModel",
    "OptSolution",
    # Differentiable layers
    "CombinatorialPtOLayer",
    "NonLinearPtOLayer",
    # Loss functions
    "SPOPlusLoss",
    "PFYLoss",
    "HybridLoss",
]
