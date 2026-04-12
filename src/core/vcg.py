"""Generic Vickrey-Clarke-Groves (VCG) mechanism engine.

The VCG mechanism is the unique (up to additive constants) efficient and
strategyproof mechanism for quasi-linear environments.  Given:

  - A set of agents  N = {1, ..., n}
  - A set of feasible allocations  A
  - Valuation functions  v_i(a, theta_i)  for each agent i

the mechanism:

  1. Chooses the welfare-maximising allocation:
         a*(theta) = argmax_{a in A}  sum_i v_i(a, theta_i)

  2. Charges each agent the Clarke pivot payment:
         p_i = max_{a in A} sum_{j != i} v_j(a, theta_j)
              - sum_{j != i} v_j(a*(theta), theta_j)

This module provides a generic VCGMechanism class that can be subclassed
or composed with custom allocation rules.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MechanismResult:
    """Container for VCG mechanism outputs."""

    allocation: Any
    payments: Dict[int, float]
    utilities: Dict[int, float]
    social_welfare: float
    budget_surplus: float  # sum of payments (>= 0 for Clarke pivot)

    def __repr__(self) -> str:
        lines = [
            "MechanismResult(",
            f"  allocation  = {self.allocation}",
            f"  payments    = {self.payments}",
            f"  utilities   = {self.utilities}",
            f"  welfare     = {self.social_welfare:.4f}",
            f"  budget      = {self.budget_surplus:.4f}",
            ")",
        ]
        return "\n".join(lines)


class VCGMechanism:
    """Generic VCG mechanism.

    Parameters
    ----------
    num_agents : int
        Number of agents.
    allocations : list
        Exhaustive list of feasible allocations (brute-force mode)
        or ``None`` if ``allocation_rule`` is supplied.
    valuation_fn : callable
        ``valuation_fn(agent_id, allocation, type_report)`` -> float
    allocation_rule : callable or None
        Optional custom allocation optimiser.  If ``None``, the engine
        enumerates ``allocations`` to find the welfare maximiser.
    h_functions : dict or None
        Optional agent-specific functions h_i(theta_{-i}) that depend
        only on the reports of others.  If ``None``, Clarke pivot
        payments are used (h_i = max_{a} sum_{j!=i} v_j(a, theta_j)).
    """

    def __init__(
        self,
        num_agents: int,
        allocations: Optional[List[Any]] = None,
        valuation_fn: Optional[Callable] = None,
        allocation_rule: Optional[Callable] = None,
        h_functions: Optional[Dict[int, Callable]] = None,
    ):
        self.num_agents = num_agents
        self.allocations = allocations
        self.valuation_fn = valuation_fn
        self.allocation_rule = allocation_rule
        self.h_functions = h_functions

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def solve(self, type_reports: Dict[int, Any]) -> MechanismResult:
        """Run the mechanism on reported types.

        Parameters
        ----------
        type_reports : dict
            Mapping from agent index (0-based) to reported type.

        Returns
        -------
        MechanismResult
        """
        agents = list(range(self.num_agents))

        # Step 1: welfare-maximising allocation
        allocation, welfare = self._find_optimal_allocation(agents, type_reports)

        # Step 2: VCG payments
        payments: Dict[int, float] = {}
        utilities: Dict[int, float] = {}

        for i in agents:
            others = [j for j in agents if j != i]

            # h_i(theta_{-i})
            if self.h_functions is not None and i in self.h_functions:
                h_i = self.h_functions[i](
                    {j: type_reports[j] for j in others}
                )
            else:
                # Clarke pivot: max over allocations of sum_{j!=i} v_j
                h_i = self._clarke_pivot_h(i, others, type_reports)

            # sum_{j!=i} v_j(a*(theta), theta_j)
            others_welfare = sum(
                self.valuation_fn(j, allocation, type_reports[j])
                for j in others
            )

            payments[i] = h_i - others_welfare
            vi = self.valuation_fn(i, allocation, type_reports[i])
            utilities[i] = vi - payments[i]

        budget_surplus = sum(payments.values())

        return MechanismResult(
            allocation=allocation,
            payments=payments,
            utilities=utilities,
            social_welfare=welfare,
            budget_surplus=budget_surplus,
        )

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def _find_optimal_allocation(
        self,
        agents: List[int],
        type_reports: Dict[int, Any],
    ) -> Tuple[Any, float]:
        """Find the allocation maximising total welfare."""
        if self.allocation_rule is not None:
            return self.allocation_rule(agents, type_reports)

        if self.allocations is None:
            raise ValueError(
                "Either 'allocations' or 'allocation_rule' must be provided."
            )

        best_alloc = None
        best_welfare = -np.inf

        for alloc in self.allocations:
            welfare = sum(
                self.valuation_fn(i, alloc, type_reports[i]) for i in agents
            )
            if welfare > best_welfare:
                best_welfare = welfare
                best_alloc = alloc

        return best_alloc, best_welfare

    def _find_optimal_allocation_without(
        self,
        agents: List[int],
        type_reports: Dict[int, Any],
    ) -> Tuple[Any, float]:
        """Find the allocation maximising welfare for a subset of agents."""
        if self.allocations is None:
            raise ValueError("Brute-force search requires 'allocations' list.")

        best_alloc = None
        best_welfare = -np.inf

        for alloc in self.allocations:
            welfare = sum(
                self.valuation_fn(j, alloc, type_reports[j]) for j in agents
            )
            if welfare > best_welfare:
                best_welfare = welfare
                best_alloc = alloc

        return best_alloc, best_welfare

    # ------------------------------------------------------------------
    # Clarke pivot
    # ------------------------------------------------------------------

    def _clarke_pivot_h(
        self,
        i: int,
        others: List[int],
        type_reports: Dict[int, Any],
    ) -> float:
        """Compute Clarke pivot h_i = max_a sum_{j!=i} v_j(a, theta_j)."""
        _, best_welfare_without_i = self._find_optimal_allocation_without(
            others, type_reports
        )
        return best_welfare_without_i

    # ------------------------------------------------------------------
    # Truthfulness verification
    # ------------------------------------------------------------------

    def verify_truthfulness(
        self,
        true_types: Dict[int, Any],
        type_space: Optional[Dict[int, List[Any]]] = None,
        agent: Optional[int] = None,
    ) -> Dict[int, bool]:
        """Verify that truthful reporting is a dominant strategy.

        For each agent (or a specific ``agent``), iterate over alternative
        reports and confirm that no deviation improves utility.

        Parameters
        ----------
        true_types : dict
            True type profile.
        type_space : dict or None
            For each agent, a list of possible types to try.
            If None, only the truthful profile is checked.
        agent : int or None
            If given, only check this agent.

        Returns
        -------
        dict mapping agent index to bool (True = truthful is dominant).
        """
        agents_to_check = [agent] if agent is not None else list(range(self.num_agents))

        truthful_result = self.solve(true_types)
        results: Dict[int, bool] = {}

        for i in agents_to_check:
            if type_space is None or i not in type_space:
                results[i] = True
                continue

            truthful_utility = truthful_result.utilities[i]
            is_truthful = True

            for alt_type in type_space[i]:
                deviated_reports = dict(true_types)
                deviated_reports[i] = alt_type
                deviated_result = self.solve(deviated_reports)
                # Utility under deviation: use TRUE type for valuation
                vi_true = self.valuation_fn(
                    i, deviated_result.allocation, true_types[i]
                )
                deviated_utility = vi_true - deviated_result.payments[i]

                if deviated_utility > truthful_utility + 1e-10:
                    is_truthful = False
                    break

            results[i] = is_truthful

        return results
