"""Public project provision with VCG payments.

Binary public good
------------------
A public project (e.g., a bridge) costs C to build.  Each of n agents
has a private value theta_i for the project being built.  The
efficient decision is to build if  sum(theta_i) >= C.

VCG (Clarke pivot) payments:

  If the project is built (a* = 1):
    p_i = max(0,  C - sum_{j != i} theta_j)

  If the project is not built (a* = 0):
    p_i = max(0,  sum_{j != i} theta_j - C)   [paid by those who prevented it]

Key insight: VCG payments for public goods typically result in a
BUDGET DEFICIT (payments collected < cost).  This is a manifestation
of the Green-Laffont impossibility theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.vcg import MechanismResult, VCGMechanism


@dataclass
class PublicGoodResult:
    """Result of public goods provision mechanism."""

    build: bool
    payments: Dict[int, float]
    utilities: Dict[int, float]
    social_welfare: float
    total_payments: float
    cost: float
    budget_deficit: float  # cost - total_payments (positive = deficit)


class PublicProjectMechanism:
    """Binary public project mechanism with VCG/Clarke pivot payments.

    Parameters
    ----------
    num_agents : int
        Number of agents.
    cost : float
        Cost of the public project.
    """

    def __init__(self, num_agents: int, cost: float):
        if cost < 0:
            raise ValueError("Cost must be non-negative.")
        self.num_agents = num_agents
        self.cost = cost

    def run(self, valuations: Dict[int, float]) -> PublicGoodResult:
        """Run the public project mechanism.

        Parameters
        ----------
        valuations : dict
            Mapping from agent index to private value for the project.

        Returns
        -------
        PublicGoodResult
        """
        if len(valuations) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} valuations, got {len(valuations)}"
            )

        total_value = sum(valuations.values())
        build = total_value >= self.cost

        payments: Dict[int, float] = {}
        utilities: Dict[int, float] = {}

        for i in range(self.num_agents):
            others_value = total_value - valuations[i]

            if build:
                # Project is built.
                # h_i = max welfare of others without i
                # Without i, build if others_value >= cost:
                #   welfare_without_i_build = others_value - cost
                #   welfare_without_i_nobuild = 0
                # h_i = max(others_value - cost, 0)
                #
                # Others' welfare with i present (project built):
                #   others_value - cost ... but we split: others get
                #   sum_{j!=i} v_j(build) = others_value
                #   (cost is NOT subtracted from individual valuations in
                #    the VCG formulation; it appears in social welfare)
                #
                # Correct VCG formulation:
                #   Social welfare of allocation a for all agents:
                #     W(a=build) = sum(v_i) - C
                #     W(a=nobuild) = 0
                #
                # For Clarke pivot:
                #   h_i = max_a sum_{j!=i} w_j(a)
                #   where w_j(build) = v_j - C/n  (sharing cost equally?)
                #
                # Standard formulation: treat cost as part of welfare.
                #   v_i(build, theta_i) = theta_i
                #   v_i(nobuild, theta_i) = 0
                #   Social cost of building = C
                #   W(build) = sum(theta_i) - C
                #   W(nobuild) = 0
                #
                # VCG with cost: We need a "seller" or cost agent.
                # Standard approach: add a dummy agent 'seller' with
                #   v_seller(build) = -C, v_seller(nobuild) = 0
                #
                # Clarke pivot for agent i:
                #   h_i = max_a [sum_{j!=i} v_j(a) + v_seller(a)]
                #       = max(others_value - C, 0)
                #   Welfare of others + seller at a*=build:
                #       = others_value - C
                #   p_i = max(others_value - C, 0) - (others_value - C)
                #
                # If others_value >= C (project would be built even without i):
                #   p_i = (others_value - C) - (others_value - C) = 0
                #
                # If others_value < C (i is pivotal):
                #   p_i = 0 - (others_value - C) = C - others_value

                if others_value >= self.cost:
                    # Not pivotal: project built even without i
                    payments[i] = 0.0
                else:
                    # Pivotal: i's report swung the decision
                    payments[i] = self.cost - others_value

                utilities[i] = valuations[i] - payments[i]
            else:
                # Project is NOT built.
                # Clarke pivot for agent i:
                #   h_i = max_a [sum_{j!=i} v_j(a) + v_seller(a)]
                #       = max(others_value - C, 0) = 0
                #     (since total_value < C implies others_value < C too,
                #      unless agent i has negative value)
                #
                # Others' welfare at a*=nobuild = 0
                # p_i = 0 - 0 = 0
                #
                # Edge case: if others_value >= C but total < C (i has
                # negative value and is pivotal for NOT building):
                if others_value >= self.cost:
                    # i prevented the project from being built
                    payments[i] = others_value - self.cost
                else:
                    payments[i] = 0.0

                utilities[i] = -payments[i]  # gets 0 value from no project

        total_payments = sum(payments.values())
        social_welfare = (total_value - self.cost) if build else 0.0
        budget_deficit = (self.cost - total_payments) if build else -total_payments

        return PublicGoodResult(
            build=build,
            payments=payments,
            utilities=utilities,
            social_welfare=social_welfare,
            total_payments=total_payments,
            cost=self.cost,
            budget_deficit=budget_deficit,
        )

    def as_vcg(self) -> VCGMechanism:
        """Return an equivalent generic VCG mechanism.

        Includes a dummy 'seller' agent (index = num_agents) to
        represent the project cost.
        """
        n = self.num_agents
        cost = self.cost
        allocations = [True, False]  # build or not

        def valuation_fn(agent: int, allocation: bool, report: float) -> float:
            if agent == n:
                # Seller / cost agent
                return -cost if allocation else 0.0
            return report if allocation else 0.0

        # Create mechanism with n+1 agents (agents + seller)
        vcg = VCGMechanism(
            num_agents=n + 1,
            allocations=allocations,
            valuation_fn=valuation_fn,
        )
        return vcg

    def demonstrate_budget_deficit(
        self, valuations: Dict[int, float]
    ) -> Dict[str, Any]:
        """Show the budget deficit inherent in VCG for public goods.

        Returns a summary dict with build decision, payments, deficit,
        and an explanation.
        """
        result = self.run(valuations)

        explanation_lines = []
        if result.build:
            explanation_lines.append(
                f"Project BUILT: total value {sum(valuations.values()):.2f} >= "
                f"cost {self.cost:.2f}"
            )
            explanation_lines.append(
                f"Total VCG payments collected: {result.total_payments:.2f}"
            )
            explanation_lines.append(f"Project cost: {self.cost:.2f}")
            if result.budget_deficit > 0:
                explanation_lines.append(
                    f"BUDGET DEFICIT: {result.budget_deficit:.2f}"
                )
                explanation_lines.append(
                    "This demonstrates the Green-Laffont impossibility: "
                    "no mechanism can simultaneously be efficient, "
                    "strategyproof, individually rational, and budget-balanced "
                    "for public goods."
                )
            else:
                explanation_lines.append("No budget deficit in this case.")
        else:
            explanation_lines.append(
                f"Project NOT BUILT: total value {sum(valuations.values()):.2f} < "
                f"cost {self.cost:.2f}"
            )

        return {
            "result": result,
            "explanation": "\n".join(explanation_lines),
        }
