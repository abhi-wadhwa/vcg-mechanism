"""Tests for the generic VCG mechanism engine."""

from __future__ import annotations

from src.core.vcg import MechanismResult, VCGMechanism


class TestVCGMechanism:
    """Test suite for the generic VCG engine."""

    def _make_single_item_vcg(self, n: int) -> VCGMechanism:
        """Create a VCG mechanism equivalent to a single-item auction."""
        allocations = list(range(n))

        def valuation_fn(agent: int, allocation: int, bid: float) -> float:
            return bid if allocation == agent else 0.0

        return VCGMechanism(
            num_agents=n,
            allocations=allocations,
            valuation_fn=valuation_fn,
        )

    def test_single_item_allocation(self):
        """Highest bidder wins."""
        vcg = self._make_single_item_vcg(3)
        result = vcg.solve({0: 100, 1: 80, 2: 60})
        assert result.allocation == 0

    def test_single_item_payments(self):
        """Winner pays second-highest bid (Clarke pivot)."""
        vcg = self._make_single_item_vcg(3)
        result = vcg.solve({0: 100, 1: 80, 2: 60})
        # Winner (agent 0): pays second-highest = 80
        assert abs(result.payments[0] - 80.0) < 1e-10
        # Losers pay 0
        assert abs(result.payments[1]) < 1e-10
        assert abs(result.payments[2]) < 1e-10

    def test_single_item_utilities(self):
        """Winner utility = value - payment; loser utility = 0."""
        vcg = self._make_single_item_vcg(3)
        result = vcg.solve({0: 100, 1: 80, 2: 60})
        assert abs(result.utilities[0] - 20.0) < 1e-10
        assert abs(result.utilities[1]) < 1e-10
        assert abs(result.utilities[2]) < 1e-10

    def test_social_welfare(self):
        """Social welfare = highest bid."""
        vcg = self._make_single_item_vcg(3)
        result = vcg.solve({0: 100, 1: 80, 2: 60})
        assert abs(result.social_welfare - 100.0) < 1e-10

    def test_budget_surplus_nonnegative(self):
        """Clarke pivot always yields non-negative surplus for auctions."""
        vcg = self._make_single_item_vcg(4)
        result = vcg.solve({0: 50, 1: 30, 2: 20, 3: 10})
        assert result.budget_surplus >= -1e-10

    def test_vcg_matches_vickrey(self):
        """Generic VCG produces same results as direct Vickrey implementation."""
        from src.core.auctions import VickreyAuction

        bids = {0: 90, 1: 70, 2: 50, 3: 30}
        n = len(bids)

        vcg = self._make_single_item_vcg(n)
        vcg_result = vcg.solve(bids)

        auction = VickreyAuction(n)
        vickrey_result = auction.run(bids)

        assert vcg_result.allocation == vickrey_result.allocation
        for i in range(n):
            assert abs(vcg_result.payments[i] - vickrey_result.payments[i]) < 1e-10
            assert abs(vcg_result.utilities[i] - vickrey_result.utilities[i]) < 1e-10

    def test_two_bidders(self):
        """With two bidders, VCG = second-price auction."""
        vcg = self._make_single_item_vcg(2)
        result = vcg.solve({0: 50, 1: 30})
        assert result.allocation == 0
        assert abs(result.payments[0] - 30.0) < 1e-10
        assert abs(result.payments[1]) < 1e-10

    def test_tied_bids(self):
        """Tied bids: lower-indexed agent wins (tie-breaking rule)."""
        vcg = self._make_single_item_vcg(3)
        result = vcg.solve({0: 50, 1: 50, 2: 30})
        # Agent 0 wins (first in enumeration)
        assert result.allocation == 0
        assert abs(result.payments[0] - 50.0) < 1e-10

    def test_custom_h_function(self):
        """Test VCG with custom h functions (not Clarke pivot)."""
        allocations = list(range(3))

        def valuation_fn(agent: int, allocation: int, bid: float) -> float:
            return bid if allocation == agent else 0.0

        # Custom h: always 0 (Groves mechanism variant)
        h_fns = {i: lambda reports: 0.0 for i in range(3)}

        vcg = VCGMechanism(
            num_agents=3,
            allocations=allocations,
            valuation_fn=valuation_fn,
            h_functions=h_fns,
        )
        result = vcg.solve({0: 100, 1: 80, 2: 60})

        # With h_i = 0, payment = -others_welfare
        # p_0 = 0 - 0 = 0 (others get 0 since only agent 0 wins)
        assert abs(result.payments[0]) < 1e-10

    def test_verify_truthfulness(self):
        """Verify that truthful reporting is dominant."""
        vcg = self._make_single_item_vcg(3)
        true_types = {0: 100, 1: 80, 2: 60}
        type_space = {
            0: [0, 50, 80, 100, 120, 200],
            1: [0, 50, 80, 100, 120, 200],
            2: [0, 50, 60, 80, 100, 200],
        }
        results = vcg.verify_truthfulness(true_types, type_space)
        for i in range(3):
            assert results[i] is True, f"Agent {i} can manipulate"

    def test_mechanism_result_repr(self):
        """MechanismResult has a readable string representation."""
        result = MechanismResult(
            allocation=0,
            payments={0: 80.0, 1: 0.0},
            utilities={0: 20.0, 1: 0.0},
            social_welfare=100.0,
            budget_surplus=80.0,
        )
        s = repr(result)
        assert "allocation" in s
        assert "payments" in s
