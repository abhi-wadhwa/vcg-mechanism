"""Facility location mechanism on a line.

The facility location problem: place a public facility (e.g., a library)
on the real line given agents' reported ideal locations.  Each agent's
cost is the distance from the facility to their ideal point.

Strategyproof mechanisms
------------------------
The MEDIAN mechanism is the unique anonymous, Pareto-optimal,
strategyproof mechanism for single-facility location on a line.

  - Location chosen: median of reported peaks
  - No monetary transfers needed
  - Agents cannot benefit from misreporting (the median is
    non-responsive to deviations that don't cross it)

Generalised median mechanisms (Moulin, 1980):
  - Add (n-1) "phantom" points, then take median of all 2n-1 values
  - These are the ONLY strategyproof mechanisms on a line

VCG for facility location
-------------------------
VCG can also be applied, choosing the location that minimises total
distance and using Clarke pivot payments.  However, VCG is overkill
here since the median mechanism achieves strategyproofness without
money.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.core.vcg import MechanismResult, VCGMechanism


@dataclass
class FacilityResult:
    """Result of facility location mechanism."""

    location: float
    agent_costs: Dict[int, float]  # distance from facility
    payments: Dict[int, float]
    total_cost: float  # sum of distances
    is_pareto_optimal: bool


class FacilityLocationMechanism:
    """Strategyproof facility location mechanism.

    Parameters
    ----------
    num_agents : int
        Number of agents.
    method : str
        ``"median"`` for the classic median mechanism (no payments),
        ``"vcg"`` for VCG with Clarke pivot payments.
    phantoms : list of float or None
        Phantom voters for generalised median mechanisms.
        If None and method="median", uses standard median.
    grid_resolution : float
        For VCG mode, the spacing of candidate locations.
    grid_range : tuple
        For VCG mode, (min, max) of candidate locations.
    """

    def __init__(
        self,
        num_agents: int,
        method: str = "median",
        phantoms: Optional[List[float]] = None,
        grid_resolution: float = 0.1,
        grid_range: Optional[tuple] = None,
    ):
        self.num_agents = num_agents
        self.method = method
        self.phantoms = phantoms or []
        self.grid_resolution = grid_resolution
        self.grid_range = grid_range

    def run(self, reports: Dict[int, float]) -> FacilityResult:
        """Run the facility location mechanism.

        Parameters
        ----------
        reports : dict
            Mapping from agent index to reported ideal location.
        """
        if len(reports) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} reports, got {len(reports)}"
            )

        if self.method == "median":
            return self._run_median(reports)
        elif self.method == "vcg":
            return self._run_vcg(reports)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _run_median(self, reports: Dict[int, float]) -> FacilityResult:
        """Median mechanism (strategyproof, no payments)."""
        all_points = sorted(
            [reports[i] for i in range(self.num_agents)] + list(self.phantoms)
        )
        n = len(all_points)
        # Median: middle element (lower median for even count)
        location = float(np.median(all_points))

        agent_costs = {i: abs(reports[i] - location) for i in range(self.num_agents)}
        payments = {i: 0.0 for i in range(self.num_agents)}
        total_cost = sum(agent_costs.values())

        return FacilityResult(
            location=location,
            agent_costs=agent_costs,
            payments=payments,
            total_cost=total_cost,
            is_pareto_optimal=True,
        )

    def _run_vcg(self, reports: Dict[int, float]) -> FacilityResult:
        """VCG mechanism for facility location with Clarke pivot payments.

        Minimise total distance (negative valuation = cost).
        """
        # Determine grid range from reports if not specified
        values = list(reports.values())
        if self.grid_range is not None:
            lo, hi = self.grid_range
        else:
            lo = min(values) - 1.0
            hi = max(values) + 1.0

        # Candidate locations
        candidates = list(
            np.arange(lo, hi + self.grid_resolution / 2, self.grid_resolution)
        )
        # Also include all reported points as candidates for exactness
        for v in values:
            if v not in candidates:
                candidates.append(v)
        candidates.sort()

        # Valuation: negative distance (we MAXIMISE welfare = MINIMISE total distance)
        def total_neg_distance(loc: float, agents: List[int]) -> float:
            return -sum(abs(reports[j] - loc) for j in agents)

        # Optimal allocation with all agents
        best_loc = min(candidates, key=lambda loc: sum(abs(reports[j] - loc) for j in range(self.num_agents)))
        welfare_all = total_neg_distance(best_loc, list(range(self.num_agents)))

        # Clarke pivot payments
        payments: Dict[int, float] = {}
        for i in range(self.num_agents):
            others = [j for j in range(self.num_agents) if j != i]
            # Optimal for others without i
            best_loc_without_i = min(
                candidates,
                key=lambda loc: sum(abs(reports[j] - loc) for j in others),
            )
            welfare_others_without_i = total_neg_distance(best_loc_without_i, others)
            welfare_others_with_i = total_neg_distance(best_loc, others)
            payments[i] = welfare_others_without_i - welfare_others_with_i

        agent_costs = {i: abs(reports[i] - best_loc) for i in range(self.num_agents)}
        total_cost = sum(agent_costs.values())

        return FacilityResult(
            location=best_loc,
            agent_costs=agent_costs,
            payments=payments,
            total_cost=total_cost,
            is_pareto_optimal=True,
        )

    def verify_strategyproofness(
        self,
        true_reports: Dict[int, float],
        deviation_range: tuple = (-10.0, 10.0),
        num_deviations: int = 100,
    ) -> Dict[int, bool]:
        """Verify strategyproofness by checking deviations.

        For the median mechanism, no agent can reduce their cost by
        misreporting.

        Returns
        -------
        dict mapping agent index to bool (True = no beneficial deviation found).
        """
        truthful_result = self.run(true_reports)
        results: Dict[int, bool] = {}

        deviations = np.linspace(
            deviation_range[0], deviation_range[1], num_deviations
        )

        for i in range(self.num_agents):
            truthful_cost = truthful_result.agent_costs[i]
            is_sp = True

            for dev_loc in deviations:
                deviated_reports = dict(true_reports)
                deviated_reports[i] = dev_loc
                dev_result = self.run(deviated_reports)
                # True cost under deviation (use TRUE location)
                dev_cost = abs(true_reports[i] - dev_result.location)
                dev_payment = dev_result.payments[i]
                # Total disutility: cost + payment
                if (dev_cost + dev_payment) < (
                    truthful_cost + truthful_result.payments[i] - 1e-10
                ):
                    is_sp = False
                    break

            results[i] = is_sp

        return results
