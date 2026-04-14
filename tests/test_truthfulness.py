"""Tests for truthfulness (incentive compatibility) properties."""

from __future__ import annotations

import numpy as np

from src.core.auctions import MultiUnitAuction, VickreyAuction
from src.core.manipulation import ManipulationDetector


class TestTruthfulnessVickrey:
    """Verify DSIC for single-item Vickrey auction."""

    def test_no_beneficial_deviation_small(self):
        """Brute-force: no agent gains by misreporting (3 bidders)."""
        n = 3
        auction = VickreyAuction(n)
        vcg = auction.as_vcg()

        type_space = list(range(0, 101, 10))  # 0, 10, 20, ..., 100
        type_spaces = {i: type_space for i in range(n)}

        detector = ManipulationDetector(vcg, type_spaces)

        # Check for several true type profiles
        rng = np.random.RandomState(42)
        for _ in range(10):
            true_types = {i: float(rng.choice(type_space)) for i in range(n)}
            analysis = detector.analyse(true_types)
            assert analysis.is_strategyproof, (
                f"VCG is not strategyproof for profile {true_types}: "
                f"{analysis.summary()}"
            )

    def test_dsic_brute_force(self):
        """Full DSIC check: for EVERY profile, no agent benefits."""
        n = 2
        auction = VickreyAuction(n)
        vcg = auction.as_vcg()

        type_space = [10.0, 20.0, 30.0, 40.0, 50.0]
        type_spaces = {i: type_space for i in range(n)}

        detector = ManipulationDetector(vcg, type_spaces)

        for i in range(n):
            result = detector.brute_force_dsic_check(i)
            assert result["is_dsic"], (
                f"Agent {i} can manipulate! Violations: {result['violations']}"
            )

    def test_individual_rationality(self):
        """All agents have non-negative utility."""
        n = 4
        auction = VickreyAuction(n)
        vcg = auction.as_vcg()

        type_space = [10.0, 30.0, 50.0, 70.0, 90.0]
        type_spaces = {i: type_space for i in range(n)}
        detector = ManipulationDetector(vcg, type_spaces)

        for _ in range(20):
            rng = np.random.RandomState(_)
            true_types = {i: float(rng.choice(type_space)) for i in range(n)}
            analysis = detector.analyse(true_types)
            assert analysis.is_individually_rational


class TestTruthfulnessMultiUnit:
    """Verify DSIC for multi-unit VCG auction."""

    def test_no_beneficial_deviation(self):
        """No agent benefits from misreporting in multi-unit auction."""
        n = 4
        k = 2
        auction = MultiUnitAuction(n, k)
        vcg = auction.as_vcg()

        type_space = [10.0, 30.0, 50.0, 70.0, 90.0]
        type_spaces = {i: type_space for i in range(n)}

        detector = ManipulationDetector(vcg, type_spaces)

        rng = np.random.RandomState(123)
        for _ in range(10):
            true_types = {i: float(rng.choice(type_space)) for i in range(n)}
            analysis = detector.analyse(true_types)
            assert analysis.is_strategyproof, (
                f"Multi-unit VCG is not strategyproof for {true_types}"
            )

    def test_dsic_brute_force_small(self):
        """Full DSIC check for small multi-unit auction."""
        n = 3
        k = 1
        auction = MultiUnitAuction(n, k)
        vcg = auction.as_vcg()

        type_space = [10.0, 25.0, 40.0, 55.0]
        type_spaces = {i: type_space for i in range(n)}

        detector = ManipulationDetector(vcg, type_spaces)

        for i in range(n):
            result = detector.brute_force_dsic_check(i)
            assert result["is_dsic"], (
                f"Agent {i} can manipulate multi-unit auction"
            )


class TestBudgetProperties:
    """Test budget properties of VCG mechanisms."""

    def test_clarke_pivot_nonneg_single_item(self):
        """Clarke pivot budget surplus is non-negative for single item."""
        n = 5
        rng = np.random.RandomState(42)
        auction = VickreyAuction(n)

        for _ in range(100):
            bids = {i: rng.uniform(0, 100) for i in range(n)}
            result = auction.run(bids)
            assert result.budget_surplus >= -1e-10

    def test_clarke_pivot_nonneg_multi_unit(self):
        """Clarke pivot budget surplus is non-negative for multi-unit."""
        n = 6
        k = 3
        rng = np.random.RandomState(99)
        auction = MultiUnitAuction(n, k)

        for _ in range(100):
            bids = {i: rng.uniform(0, 100) for i in range(n)}
            result = auction.run(bids)
            assert result.budget_surplus >= -1e-10


class TestManipulationDetector:
    """Test the manipulation detection utilities."""

    def test_compare_truthful_vs_deviation(self):
        """Side-by-side comparison works correctly."""
        n = 3
        auction = VickreyAuction(n)
        vcg = auction.as_vcg()

        type_spaces = {i: [10.0, 50.0, 90.0] for i in range(n)}
        detector = ManipulationDetector(vcg, type_spaces)

        comparison = detector.compare_truthful_vs_deviation(
            true_types={0: 90.0, 1: 50.0, 2: 10.0},
            agent=0,
            deviation_type=30.0,  # underbid
        )

        assert "truthful" in comparison
        assert "deviated" in comparison
        assert "utility_change" in comparison
        # Underbidding to 30 when true value is 90: still wins, same payment
        assert not comparison["manipulation_profitable"]

    def test_summary_string(self):
        """ManipulationAnalysis produces readable summary."""
        n = 2
        auction = VickreyAuction(n)
        vcg = auction.as_vcg()

        type_spaces = {0: [10.0, 50.0], 1: [10.0, 50.0]}
        detector = ManipulationDetector(vcg, type_spaces)

        analysis = detector.analyse({0: 50.0, 1: 10.0})
        summary = analysis.summary()
        assert "Manipulation Analysis" in summary
        assert "Strategyproof" in summary
