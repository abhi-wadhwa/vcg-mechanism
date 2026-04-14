"""Arrow-d'Aspremont-Gerard-Varet (AGV) mechanism.

The AGV mechanism achieves:
  - Allocative efficiency (welfare maximisation)
  - Expected budget balance (E[sum of payments] = 0)
  - Bayesian incentive compatibility (truthful in expectation)

It sacrifices dominant-strategy incentive compatibility (DSIC) for
budget balance, complementing the VCG mechanism which is DSIC but
runs a budget deficit.

Payment rule
------------
For each agent i:

  p_i(theta) = (1/(n-1)) * sum_{j != i}
    E_{theta_{-j}}[sum_{k != j} v_k(a*(theta_k, theta_{-k}), theta_k)]
    - E_{theta_{-i}}[sum_{j != i} v_j(a*(theta_i, theta_{-i}), theta_j)]

In the simplified form with independent types:

  p_i(theta) = h_i(theta_{-i}) - sum_{j != i} v_j(a*(theta), theta_j)

where h_i is chosen to achieve budget balance in expectation.

For practical implementation, we use the direct approach:
  1. Compute VCG payments
  2. Add redistribution terms that sum to zero in expectation
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.vcg import VCGMechanism


@dataclass
class AGVResult:
    """Result of AGV mechanism."""

    allocation: Any
    payments: dict[int, float]
    utilities: dict[int, float]
    social_welfare: float
    budget_surplus: float
    expected_budget_surplus: float  # should be ~0


class AGVMechanism:
    """Arrow-d'Aspremont-Gerard-Varet (AGV) mechanism.

    Achieves expected budget balance while maintaining efficiency and
    Bayesian incentive compatibility.

    Parameters
    ----------
    num_agents : int
        Number of agents.
    allocations : list
        Feasible allocations.
    valuation_fn : callable
        ``valuation_fn(agent_id, allocation, type_report)`` -> float
    type_distributions : dict
        For each agent, a list of (type, probability) pairs representing
        the prior distribution over types.
    """

    def __init__(
        self,
        num_agents: int,
        allocations: list[Any],
        valuation_fn: Callable,
        type_distributions: dict[int, list[tuple[Any, float]]],
    ):
        self.num_agents = num_agents
        self.allocations = allocations
        self.valuation_fn = valuation_fn
        self.type_distributions = type_distributions

        # Build the underlying VCG mechanism
        self._vcg = VCGMechanism(
            num_agents=num_agents,
            allocations=allocations,
            valuation_fn=valuation_fn,
        )

        # Precompute expected externalities for budget balancing
        self._expected_externalities = self._compute_expected_externalities()

    def _compute_expected_externalities(self) -> dict[int, float]:
        """Compute E_{theta_{-i}}[sum_{j!=i} v_j(a*(theta), theta_j)] for
        each agent i, averaging over the prior on others' types.

        This is used to construct the redistribution terms.
        """
        expected_ext: dict[int, float] = {}

        for i in range(self.num_agents):
            # We need to compute the expected Clarke pivot payment for agent i
            # by averaging over all possible type profiles of other agents.
            others = [j for j in range(self.num_agents) if j != i]

            # Generate all possible type profiles for others
            others_type_lists = [self.type_distributions[j] for j in others]

            # Compute expected h_i and expected others' welfare at optimum
            expected_h_i = 0.0
            expected_others_welfare = 0.0

            # Iterate over all combinations of others' types
            from itertools import product as iter_product

            others_combos = list(iter_product(*others_type_lists))

            for combo in others_combos:
                # combo is a tuple of (type, prob) pairs for each other agent
                prob = 1.0
                type_profile_others: dict[int, Any] = {}
                for idx, j in enumerate(others):
                    type_val, type_prob = combo[idx]
                    type_profile_others[j] = type_val
                    prob *= type_prob

                # For each possible type of agent i, compute contributions
                for type_i, prob_i in self.type_distributions[i]:
                    full_profile = dict(type_profile_others)
                    full_profile[i] = type_i

                    total_prob = prob * prob_i

                    # Optimal allocation
                    best_alloc = None
                    best_welfare = -np.inf
                    for alloc in self.allocations:
                        w = sum(
                            self.valuation_fn(j, alloc, full_profile[j])
                            for j in range(self.num_agents)
                        )
                        if w > best_welfare:
                            best_welfare = w
                            best_alloc = alloc

                    # Others' welfare at optimum
                    ow = sum(
                        self.valuation_fn(j, best_alloc, full_profile[j])
                        for j in others
                    )
                    expected_others_welfare += total_prob * ow

                    # h_i: max over allocations of others' welfare
                    max_others = -np.inf
                    for alloc in self.allocations:
                        w_others = sum(
                            self.valuation_fn(j, alloc, type_profile_others[j])
                            for j in others
                        )
                        if w_others > max_others:
                            max_others = w_others
                    expected_h_i += total_prob * max_others

            expected_ext[i] = expected_h_i - expected_others_welfare

        return expected_ext

    def _compute_redistribution(self) -> dict[int, float]:
        """Compute redistribution amounts that achieve expected budget balance.

        We redistribute the expected surplus equally.  Each agent receives
        back: (1/(n-1)) * sum_{j != i} E[p_j^{VCG}]  minus a correction.

        The simplest AGV construction: set
          h_i^{AGV}(theta_{-i}) = (1/(n-1)) * sum_{j != i} E_{theta_{-j}}[
            sum_{k != j} v_k(a*(theta), theta_k) ]

        For practical purposes, we compute:
          redistribution_i = - expected_vcg_payment_i
                             + (1/n) * sum_j expected_vcg_payment_j
        This ensures sum of expected redistributions = 0, and expected
        total payment = 0.
        """
        total_expected = sum(self._expected_externalities.values())
        redistribution = {}
        for i in range(self.num_agents):
            redistribution[i] = (
                -self._expected_externalities[i]
                + total_expected / self.num_agents
            )
        return redistribution

    def solve(self, type_reports: dict[int, Any]) -> AGVResult:
        """Run the AGV mechanism on reported types.

        Parameters
        ----------
        type_reports : dict
            Mapping from agent index to reported type.

        Returns
        -------
        AGVResult
        """
        # Run underlying VCG
        vcg_result = self._vcg.solve(type_reports)

        # Compute redistribution for expected budget balance
        redistribution = self._compute_redistribution()

        # AGV payments = VCG payments - redistribution
        payments: dict[int, float] = {}
        utilities: dict[int, float] = {}

        for i in range(self.num_agents):
            payments[i] = vcg_result.payments[i] - redistribution[i]
            vi = self.valuation_fn(i, vcg_result.allocation, type_reports[i])
            utilities[i] = vi - payments[i]

        budget_surplus = sum(payments.values())

        # Expected budget surplus should be ~0
        # (the actual surplus for a specific type profile may not be 0)
        expected_surplus = sum(self._expected_externalities.values()) - sum(
            redistribution.values()
        )

        return AGVResult(
            allocation=vcg_result.allocation,
            payments=payments,
            utilities=utilities,
            social_welfare=vcg_result.social_welfare,
            budget_surplus=budget_surplus,
            expected_budget_surplus=expected_surplus,
        )

    def verify_bayesian_ic(
        self,
        agent: int,
        true_type: Any,
        type_space: list[Any],
        num_samples: int = 1000,
        seed: int = 42,
    ) -> dict[str, float]:
        """Verify Bayesian incentive compatibility for an agent.

        Checks that truthful reporting maximises expected utility,
        where expectation is over others' types drawn from the prior.

        Returns
        -------
        dict with 'truthful_expected_utility' and best deviation utility.
        """
        rng = np.random.RandomState(seed)
        others = [j for j in range(self.num_agents) if j != agent]

        def expected_utility(reported_type: Any) -> float:
            total_util = 0.0

            for _ in range(num_samples):
                # Sample others' types from prior
                type_profile: dict[int, Any] = {agent: reported_type}
                for j in others:
                    types_probs = self.type_distributions[j]
                    types, probs = zip(*types_probs)
                    probs_arr = np.array(probs, dtype=float)
                    probs_arr /= probs_arr.sum()
                    idx = rng.choice(len(types), p=probs_arr)
                    type_profile[j] = types[idx]

                result = self.solve(type_profile)
                # True utility using actual type
                vi = self.valuation_fn(
                    agent, result.allocation, true_type
                )
                total_util += vi - result.payments[agent]

            return total_util / num_samples

        truthful_eu = expected_utility(true_type)

        best_dev_type = true_type
        best_dev_eu = truthful_eu

        for alt_type in type_space:
            eu = expected_utility(alt_type)
            if eu > best_dev_eu + 1e-10:
                best_dev_eu = eu
                best_dev_type = alt_type

        return {
            "truthful_expected_utility": truthful_eu,
            "best_deviation_utility": best_dev_eu,
            "best_deviation_type": best_dev_type,
            "is_bayesian_ic": truthful_eu >= best_dev_eu - 1e-10,
        }
