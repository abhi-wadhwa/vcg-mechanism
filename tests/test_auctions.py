"""Tests for Vickrey and multi-unit auction mechanisms."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.auctions import MultiUnitAuction, VickreyAuction


class TestVickreyAuction:
    """Tests for the single-item Vickrey (second-price) auction."""

    def test_highest_bidder_wins(self):
        auction = VickreyAuction(4)
        result = auction.run({0: 100, 1: 80, 2: 60, 3: 40})
        assert result.allocation == 0

    def test_second_price_payment(self):
        auction = VickreyAuction(4)
        result = auction.run({0: 100, 1: 80, 2: 60, 3: 40})
        assert abs(result.payments[0] - 80.0) < 1e-10

    def test_losers_pay_zero(self):
        auction = VickreyAuction(3)
        result = auction.run({0: 100, 1: 80, 2: 60})
        assert abs(result.payments[1]) < 1e-10
        assert abs(result.payments[2]) < 1e-10

    def test_two_bidders(self):
        auction = VickreyAuction(2)
        result = auction.run({0: 30, 1: 50})
        assert result.allocation == 1
        assert abs(result.payments[1] - 30.0) < 1e-10

    def test_winner_utility(self):
        auction = VickreyAuction(3)
        result = auction.run({0: 100, 1: 60, 2: 40})
        assert abs(result.utilities[0] - 40.0) < 1e-10  # 100 - 60

    def test_social_welfare_equals_winner_bid(self):
        auction = VickreyAuction(3)
        result = auction.run({0: 50, 1: 90, 2: 70})
        assert abs(result.social_welfare - 90.0) < 1e-10

    def test_budget_surplus_nonnegative(self):
        auction = VickreyAuction(5)
        rng = np.random.RandomState(42)
        for _ in range(50):
            bids = {i: rng.uniform(0, 100) for i in range(5)}
            result = auction.run(bids)
            assert result.budget_surplus >= -1e-10

    def test_tie_breaking(self):
        """Tied bids: lower index wins."""
        auction = VickreyAuction(3)
        result = auction.run({0: 50, 1: 50, 2: 30})
        assert result.allocation == 0

    def test_as_vcg_equivalence(self):
        """VCG representation matches direct implementation."""
        auction = VickreyAuction(3)
        bids = {0: 70, 1: 90, 2: 50}

        direct = auction.run(bids)
        vcg = auction.as_vcg()
        vcg_result = vcg.solve(bids)

        assert direct.allocation == vcg_result.allocation
        for i in range(3):
            assert abs(direct.payments[i] - vcg_result.payments[i]) < 1e-10

    def test_wrong_number_of_bids(self):
        auction = VickreyAuction(3)
        with pytest.raises(ValueError, match="Expected 3 bids"):
            auction.run({0: 10, 1: 20})


class TestMultiUnitAuction:
    """Tests for the multi-unit VCG auction."""

    def test_top_k_win(self):
        auction = MultiUnitAuction(5, 2)
        result = auction.run({0: 90, 1: 70, 2: 50, 3: 30, 4: 10})
        assert result.allocation == frozenset({0, 1})

    def test_three_items(self):
        auction = MultiUnitAuction(5, 3)
        result = auction.run({0: 100, 1: 80, 2: 60, 3: 40, 4: 20})
        assert result.allocation == frozenset({0, 1, 2})

    def test_vcg_payments(self):
        """VCG payment = value of the agent displaced by winner."""
        auction = MultiUnitAuction(4, 2)
        # Bids: 100, 80, 60, 40; winners: {0, 1}
        result = auction.run({0: 100, 1: 80, 2: 60, 3: 40})

        # Without agent 0: winners would be {1, 2}
        # Welfare others without 0: 80 + 60 = 140
        # Welfare others with 0:    80 = 80  (only agent 1 among winners)
        # p_0 = 140 - 80 = 60
        assert abs(result.payments[0] - 60.0) < 1e-10

        # Without agent 1: winners would be {0, 2}
        # Welfare others without 1: 100 + 60 = 160
        # Welfare others with 1:    100 = 100
        # p_1 = 160 - 100 = 60
        assert abs(result.payments[1] - 60.0) < 1e-10

    def test_losers_pay_zero(self):
        auction = MultiUnitAuction(4, 2)
        result = auction.run({0: 100, 1: 80, 2: 60, 3: 40})
        assert abs(result.payments[2]) < 1e-10
        assert abs(result.payments[3]) < 1e-10

    def test_single_item_matches_vickrey(self):
        """Multi-unit with k=1 should match Vickrey auction."""
        multi = MultiUnitAuction(4, 1)
        vickrey = VickreyAuction(4)
        bids = {0: 100, 1: 80, 2: 60, 3: 40}

        multi_result = multi.run(bids)
        vickrey_result = vickrey.run(bids)

        assert vickrey_result.allocation in multi_result.allocation
        assert abs(
            multi_result.payments[vickrey_result.allocation]
            - vickrey_result.payments[vickrey_result.allocation]
        ) < 1e-10

    def test_budget_surplus_nonnegative(self):
        """Revenue is always non-negative."""
        auction = MultiUnitAuction(6, 3)
        rng = np.random.RandomState(123)
        for _ in range(50):
            bids = {i: rng.uniform(0, 100) for i in range(6)}
            result = auction.run(bids)
            assert result.budget_surplus >= -1e-10

    def test_social_welfare(self):
        auction = MultiUnitAuction(4, 2)
        bids = {0: 100, 1: 80, 2: 60, 3: 40}
        result = auction.run(bids)
        assert abs(result.social_welfare - 180.0) < 1e-10  # 100 + 80

    def test_as_vcg_equivalence(self):
        """VCG representation matches direct implementation."""
        auction = MultiUnitAuction(4, 2)
        bids = {0: 90, 1: 70, 2: 50, 3: 30}

        direct = auction.run(bids)
        vcg = auction.as_vcg()
        vcg_result = vcg.solve(bids)

        assert direct.allocation == vcg_result.allocation
        for i in range(4):
            assert abs(direct.payments[i] - vcg_result.payments[i]) < 1e-10

    def test_invalid_num_items(self):
        with pytest.raises(ValueError):
            MultiUnitAuction(3, 0)

        with pytest.raises(ValueError):
            MultiUnitAuction(3, 5)

    def test_wrong_number_of_bids(self):
        auction = MultiUnitAuction(4, 2)
        with pytest.raises(ValueError, match="Expected 4 bids"):
            auction.run({0: 10})
