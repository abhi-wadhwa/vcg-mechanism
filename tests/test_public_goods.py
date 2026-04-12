"""Tests for public goods provision mechanism."""

from __future__ import annotations

import pytest
import numpy as np

from src.core.public_goods import PublicProjectMechanism
from src.core.facility import FacilityLocationMechanism


class TestPublicProjectMechanism:
    """Tests for public project VCG mechanism."""

    def test_build_when_value_exceeds_cost(self):
        mech = PublicProjectMechanism(3, cost=30.0)
        result = mech.run({0: 15, 1: 12, 2: 8})
        # total = 35 >= 30
        assert result.build is True

    def test_no_build_when_value_below_cost(self):
        mech = PublicProjectMechanism(3, cost=50.0)
        result = mech.run({0: 15, 1: 12, 2: 8})
        # total = 35 < 50
        assert result.build is False

    def test_pivotal_agent_pays(self):
        """Agent whose presence changes the outcome pays."""
        mech = PublicProjectMechanism(3, cost=30.0)
        # Valuations: 20, 8, 7.  Total = 35 >= 30 -> build
        result = mech.run({0: 20, 1: 8, 2: 7})

        # Agent 0: others_value = 15 < 30 -> pivotal
        # p_0 = 30 - 15 = 15
        assert abs(result.payments[0] - 15.0) < 1e-10

        # Agent 1: others_value = 27 < 30 -> pivotal
        # p_1 = 30 - 27 = 3
        assert abs(result.payments[1] - 3.0) < 1e-10

        # Agent 2: others_value = 28 < 30 -> pivotal
        # p_2 = 30 - 28 = 2
        assert abs(result.payments[2] - 2.0) < 1e-10

    def test_non_pivotal_agent_pays_zero(self):
        """Agent who doesn't change the outcome pays nothing."""
        mech = PublicProjectMechanism(3, cost=20.0)
        # Valuations: 15, 12, 8.  Total = 35 >= 20
        result = mech.run({0: 15, 1: 12, 2: 8})

        # Agent 2: others_value = 27 >= 20, not pivotal -> p_2 = 0
        assert abs(result.payments[2]) < 1e-10

    def test_budget_deficit_for_public_goods(self):
        """Demonstrate that VCG payments don't cover the cost."""
        mech = PublicProjectMechanism(4, cost=80.0)
        # Valuations: 30, 25, 20, 15.  Total = 90 >= 80 -> build
        result = mech.run({0: 30, 1: 25, 2: 20, 3: 15})

        assert result.build is True
        # Each agent is pivotal (removing any drops total below 80)
        # p_0 = 80 - 60 = 20
        # p_1 = 80 - 65 = 15
        # p_2 = 80 - 70 = 10
        # p_3 = 80 - 75 = 5
        # Total payments = 50 < 80 = cost
        assert abs(result.payments[0] - 20.0) < 1e-10
        assert abs(result.payments[1] - 15.0) < 1e-10
        assert abs(result.payments[2] - 10.0) < 1e-10
        assert abs(result.payments[3] - 5.0) < 1e-10
        assert abs(result.total_payments - 50.0) < 1e-10
        assert result.budget_deficit > 0  # 80 - 50 = 30

    def test_individual_rationality(self):
        """Each agent's utility is non-negative when project is built."""
        mech = PublicProjectMechanism(3, cost=30.0)
        result = mech.run({0: 20, 1: 8, 2: 7})

        if result.build:
            for i in range(3):
                assert result.utilities[i] >= -1e-10, (
                    f"Agent {i} has negative utility: {result.utilities[i]}"
                )

    def test_no_build_zero_payments(self):
        """When project isn't built and no one is pivotal, payments are zero."""
        mech = PublicProjectMechanism(3, cost=100.0)
        result = mech.run({0: 10, 1: 10, 2: 10})
        # total = 30 < 100, no one is pivotal for NOT building
        assert result.build is False
        for i in range(3):
            assert abs(result.payments[i]) < 1e-10

    def test_marginal_build_decision(self):
        """Exactly at the cost threshold: build."""
        mech = PublicProjectMechanism(2, cost=50.0)
        result = mech.run({0: 25, 1: 25})
        assert result.build is True  # total = 50 >= 50

    def test_demonstrate_budget_deficit(self):
        """The demonstration method returns correct explanation."""
        mech = PublicProjectMechanism(3, cost=30.0)
        info = mech.demonstrate_budget_deficit({0: 20, 1: 8, 2: 7})
        assert "result" in info
        assert "explanation" in info
        assert "BUILT" in info["explanation"]

    def test_invalid_cost(self):
        with pytest.raises(ValueError, match="non-negative"):
            PublicProjectMechanism(3, cost=-10.0)


class TestFacilityLocation:
    """Tests for facility location mechanism."""

    def test_median_with_odd_agents(self):
        mech = FacilityLocationMechanism(5, method="median")
        result = mech.run({0: 1, 1: 3, 2: 5, 3: 7, 4: 9})
        assert abs(result.location - 5.0) < 1e-10

    def test_median_with_even_agents(self):
        mech = FacilityLocationMechanism(4, method="median")
        result = mech.run({0: 1, 1: 3, 2: 5, 3: 7})
        # numpy median of [1,3,5,7] = 4.0
        assert abs(result.location - 4.0) < 1e-10

    def test_median_no_payments(self):
        """Median mechanism doesn't use payments."""
        mech = FacilityLocationMechanism(3, method="median")
        result = mech.run({0: 1, 1: 5, 2: 9})
        for i in range(3):
            assert abs(result.payments[i]) < 1e-10

    def test_median_strategyproofness(self):
        """No agent can reduce cost by misreporting."""
        mech = FacilityLocationMechanism(5, method="median")
        reports = {0: 2, 1: 4, 2: 6, 3: 8, 4: 10}
        sp = mech.verify_strategyproofness(reports, deviation_range=(0, 12))
        for i in range(5):
            assert sp[i] is True, f"Agent {i} can manipulate the median"

    def test_vcg_facility_location(self):
        """VCG version also places facility optimally."""
        mech = FacilityLocationMechanism(
            3, method="vcg", grid_resolution=0.5
        )
        result = mech.run({0: 1, 1: 5, 2: 9})
        # Optimal location minimising total distance = median = 5
        assert abs(result.location - 5.0) < 1e-10

    def test_facility_costs(self):
        """Agent costs are correct distances."""
        mech = FacilityLocationMechanism(3, method="median")
        result = mech.run({0: 0, 1: 4, 2: 10})
        assert abs(result.location - 4.0) < 1e-10
        assert abs(result.agent_costs[0] - 4.0) < 1e-10
        assert abs(result.agent_costs[1] - 0.0) < 1e-10
        assert abs(result.agent_costs[2] - 6.0) < 1e-10

    def test_all_same_location(self):
        """All agents at same spot: facility placed there."""
        mech = FacilityLocationMechanism(3, method="median")
        result = mech.run({0: 5, 1: 5, 2: 5})
        assert abs(result.location - 5.0) < 1e-10
        assert abs(result.total_cost) < 1e-10

    def test_phantom_voters(self):
        """Generalised median with phantom voters."""
        mech = FacilityLocationMechanism(
            2, method="median", phantoms=[0.0]
        )
        # Reports: [3, 7], phantom: [0]
        # All points: [0, 3, 7], median = 3
        result = mech.run({0: 3, 1: 7})
        assert abs(result.location - 3.0) < 1e-10
