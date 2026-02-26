"""
Base abstractions for HecOpt optimization models.

Every optimization problem in HecOpt inherits from BaseOptModel and
specialises into either CombinatorialOptModel (LP/MIP) or NonLinearOptModel (NLP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class OptSolution:
    """Container returned by every solver call.

    Attributes
    ----------
    x_star:
        Optimal primal decision variables as a NumPy array ``[n_vars]``.
    obj_val:
        Optimal objective function value (scalar float).
    y_star:
        Dual multipliers for *equality* constraints ``[n_eq]``.
        ``None`` when there are no equality constraints.
    z_star:
        Dual multipliers for *inequality* constraints ``[n_ineq]``.
        ``None`` when there are no inequality constraints.
    active_ineq:
        Boolean mask ``[n_ineq]`` marking active inequality constraints
        (i.e., constraints where ``g_j(x*) ≈ 0``).
        ``None`` when there are no inequality constraints.
    success:
        ``True`` if the solver found a feasible optimal solution.
    message:
        Solver status message (useful for debugging).
    """

    x_star: np.ndarray
    obj_val: float
    y_star: Optional[np.ndarray] = None
    z_star: Optional[np.ndarray] = None
    active_ineq: Optional[np.ndarray] = None
    success: bool = True
    message: str = ""


class BaseOptModel(ABC):
    """Abstract base class for all optimization models in HecOpt.

    Defines the minimal interface that both combinatorial and non-linear
    optimization problems must implement so that the differentiable layers
    (``CombinatorialPtOLayer`` and ``NonLinearPtOLayer``) can call them
    uniformly.
    """

    @property
    @abstractmethod
    def is_combinatorial(self) -> bool:
        """``True`` for combinatorial/discrete problems, ``False`` for NLPs."""
        ...

    @property
    @abstractmethod
    def n_vars(self) -> int:
        """Number of primal decision variables."""
        ...

    @abstractmethod
    def solve(self, theta: np.ndarray) -> OptSolution:
        """Solve the optimization problem for the given parameter vector.

        Parameters
        ----------
        theta:
            Parameter vector predicted by the upstream neural network.
            For combinatorial problems this is typically a *cost vector*.
            For NLPs it can be any differentiable parameter (e.g. demand
            model coefficients).

        Returns
        -------
        OptSolution
            Full solution container including primal variables and, for
            continuous problems, dual multipliers.
        """
        ...


class CombinatorialOptModel(BaseOptModel):
    """Base class for combinatorial (LP / MIP) optimization problems.

    The solver must treat ``theta`` as a cost vector and return

        x* = argmin_{x ∈ W}  theta^T x

    where ``W`` is the feasible set (polytope or integer lattice).
    """

    @property
    def is_combinatorial(self) -> bool:
        return True


class NonLinearOptModel(BaseOptModel):
    """Base class for non-linear continuous optimization problems (NLPs).

    Beyond the base interface, subclasses must implement
    ``objective_torch``, ``eq_constraints_torch``, and
    ``ineq_constraints_torch`` — PyTorch-differentiable counterparts of the
    solver functions — so that ``NonLinearPtOLayer`` can build the KKT
    matrix automatically via ``torch.autograd``.
    """

    @property
    def is_combinatorial(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # PyTorch-differentiable problem functions (required for KKT backward)
    # ------------------------------------------------------------------

    @abstractmethod
    def objective_torch(
        self, x: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable objective f(x, theta) → scalar tensor.

        The layer *minimises* this objective, so for revenue maximisation
        return ``-revenue``.

        Parameters
        ----------
        x:
            Decision variable vector ``[n_vars]`` (requires_grad may be True).
        theta:
            Parameter vector ``[n_params]`` (requires_grad may be True).

        Returns
        -------
        torch.Tensor
            Scalar tensor.
        """
        ...

    def eq_constraints_torch(
        self, x: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable equality constraints h(x, theta) = 0.

        Returns a vector ``[n_eq]``. Override in subclasses that have
        equality constraints. Default: no equality constraints.
        """
        return torch.zeros(0, dtype=x.dtype, device=x.device)

    def ineq_constraints_torch(
        self, x: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable inequality constraints g(x, theta) ≤ 0.

        Returns a vector ``[n_ineq]``. Override in subclasses that have
        inequality constraints. Default: no inequality constraints.
        """
        return torch.zeros(0, dtype=x.dtype, device=x.device)

    @property
    def n_eq_constraints(self) -> int:
        """Number of equality constraints."""
        return 0

    @property
    def n_ineq_constraints(self) -> int:
        """Number of inequality constraints."""
        return 0
