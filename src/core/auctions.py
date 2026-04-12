"""Auction mechanisms as VCG special cases.

Single-item Vickrey auction
---------------------------
The Vickrey (second-price sealed-bid) auction is the VCG mechanism for
the allocation problem of assigning a single indivisible item to one
of n bidders.

  - Allocation set  A = {0, 1, ..., n-1}  (index of winner)
  - v_i(a, theta_i) = theta_i  if  a == i,  else 0
  - Clarke pivot:  p_i = max_{j != i} theta_j   (second-highest bid)

Multi-unit auction
------------------
k identical items among n bidders, each wanting at most one unit.
  - Allocation: subset S of {0,..,n-1} with |S| <= k
  - VCG payment for winner i: (k+1)-th highest bid among others
    (the externality imposed on others).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.vcg import MechanismResult, VCGMechanism


class VickreyAuction:
    """Single-item second-price (Vickrey) auction.

    This is implemented directly for efficiency, but the results
    match the generic VCG engine exactly.
    """

    def __init__(self, num_bidders: int):
        self.num_bidders = num_bidders

    def run(self, bids: Dict[int, float]) -> MechanismResult:
        """Run the auction.

        Parameters
        ----------
        bids : dict
            Mapping from bidder index to bid value.

        Returns
        -------
        MechanismResult
        """
        if len(bids) != self.num_bidders:
            raise ValueError(
                f"Expected {self.num_bidders} bids, got {len(bids)}"
            )

        # Find winner (highest bid; tie-break by lowest index)
        sorted_bidders = sorted(bids.keys(), key=lambda i: (-bids[i], i))
        winner = sorted_bidders[0]
        second_price = bids[sorted_bidders[1]] if len(sorted_bidders) > 1 else 0.0

        # Allocation: which bidder gets the item
        allocation = winner

        # Payments (Clarke pivot)
        payments = {}
        utilities = {}
        for i in range(self.num_bidders):
            if i == winner:
                payments[i] = second_price
                utilities[i] = bids[i] - second_price
            else:
                payments[i] = 0.0
                utilities[i] = 0.0

        social_welfare = bids[winner]
        budget_surplus = sum(payments.values())

        return MechanismResult(
            allocation=allocation,
            payments=payments,
            utilities=utilities,
            social_welfare=social_welfare,
            budget_surplus=budget_surplus,
        )

    def as_vcg(self) -> VCGMechanism:
        """Return an equivalent generic VCG mechanism instance."""
        allocations = list(range(self.num_bidders))

        def valuation_fn(agent: int, allocation: int, bid: float) -> float:
            return bid if allocation == agent else 0.0

        return VCGMechanism(
            num_agents=self.num_bidders,
            allocations=allocations,
            valuation_fn=valuation_fn,
        )


class MultiUnitAuction:
    """Multi-unit VCG auction for k identical items.

    Each bidder wants at most one unit and has a private value for
    receiving that unit.
    """

    def __init__(self, num_bidders: int, num_items: int):
        if num_items < 1:
            raise ValueError("num_items must be >= 1")
        if num_items > num_bidders:
            raise ValueError("num_items must be <= num_bidders")
        self.num_bidders = num_bidders
        self.num_items = num_items

    def run(self, bids: Dict[int, float]) -> MechanismResult:
        """Run the multi-unit auction.

        Parameters
        ----------
        bids : dict
            Mapping from bidder index to bid value.

        Returns
        -------
        MechanismResult
            allocation is the set of winning bidder indices.
        """
        if len(bids) != self.num_bidders:
            raise ValueError(
                f"Expected {self.num_bidders} bids, got {len(bids)}"
            )

        # Sort bidders by value descending (tie-break by index)
        sorted_bidders = sorted(bids.keys(), key=lambda i: (-bids[i], i))
        winners = set(sorted_bidders[: self.num_items])

        allocation = frozenset(winners)
        social_welfare = sum(bids[i] for i in winners)

        # VCG payments for each bidder
        payments: Dict[int, float] = {}
        utilities: Dict[int, float] = {}

        for i in range(self.num_bidders):
            if i in winners:
                # Clarke pivot: welfare of others without i minus welfare
                # of others with i.
                #
                # Without agent i, the top-k from the remaining agents win.
                others_sorted = sorted(
                    [j for j in range(self.num_bidders) if j != i],
                    key=lambda j: (-bids[j], j),
                )
                # Optimal welfare for others without i:
                others_winners_without_i = set(others_sorted[: self.num_items])
                welfare_others_without_i = sum(
                    bids[j] for j in others_winners_without_i
                )
                # Welfare of others in the actual allocation:
                welfare_others_with_i = sum(
                    bids[j] for j in winners if j != i
                )
                payments[i] = welfare_others_without_i - welfare_others_with_i
                utilities[i] = bids[i] - payments[i]
            else:
                payments[i] = 0.0
                utilities[i] = 0.0

        budget_surplus = sum(payments.values())

        return MechanismResult(
            allocation=allocation,
            payments=payments,
            utilities=utilities,
            social_welfare=social_welfare,
            budget_surplus=budget_surplus,
        )

    def as_vcg(self) -> VCGMechanism:
        """Return an equivalent generic VCG mechanism instance.

        Warning: exponential in the number of bidders for large inputs.
        """
        # Enumerate all subsets of size <= num_items
        agents = list(range(self.num_bidders))
        allocations = []
        for k in range(1, self.num_items + 1):
            for combo in itertools.combinations(agents, k):
                allocations.append(frozenset(combo))
        # Also include the empty allocation
        allocations.append(frozenset())

        def valuation_fn(
            agent: int, allocation: frozenset, bid: float
        ) -> float:
            return bid if agent in allocation else 0.0

        return VCGMechanism(
            num_agents=self.num_bidders,
            allocations=allocations,
            valuation_fn=valuation_fn,
        )
