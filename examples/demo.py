"""Demonstration of VCG mechanism design framework.

Run:  python -m examples.demo
"""

from __future__ import annotations

import numpy as np

from src.core.auctions import MultiUnitAuction, VickreyAuction
from src.core.public_goods import PublicProjectMechanism
from src.core.facility import FacilityLocationMechanism
from src.core.agv import AGVMechanism
from src.core.manipulation import ManipulationDetector
from src.core.vcg import VCGMechanism


def demo_vickrey_auction() -> None:
    """Single-item Vickrey (second-price) auction."""
    print("\n" + "=" * 60)
    print("1. VICKREY (SECOND-PRICE) AUCTION")
    print("=" * 60)
    print(
        "\nThe Vickrey auction is the VCG mechanism for single-item\n"
        "allocation: the highest bidder wins and pays the second-\n"
        "highest bid.\n"
    )

    bids = {0: 100, 1: 80, 2: 60, 3: 40}
    auction = VickreyAuction(4)
    result = auction.run(bids)

    print(f"Bids: {bids}")
    print(f"Winner: Bidder {result.allocation}")
    print(f"Payment: {result.payments[result.allocation]:.2f} (second-highest bid)")
    print(f"Winner's utility: {result.utilities[result.allocation]:.2f}")
    print(f"Social welfare: {result.social_welfare:.2f}")

    # Verify it matches generic VCG
    vcg = auction.as_vcg()
    vcg_result = vcg.solve(bids)
    print(f"\nGeneric VCG produces same result: "
          f"allocation={vcg_result.allocation}, "
          f"payment={vcg_result.payments[result.allocation]:.2f}")


def demo_multi_unit_auction() -> None:
    """Multi-unit VCG auction."""
    print("\n" + "=" * 60)
    print("2. MULTI-UNIT VCG AUCTION")
    print("=" * 60)
    print(
        "\nAuctioning k=2 identical items among 5 bidders.\n"
        "VCG payment: the externality each winner imposes on others.\n"
    )

    bids = {0: 90, 1: 75, 2: 60, 3: 45, 4: 30}
    auction = MultiUnitAuction(5, 2)
    result = auction.run(bids)

    print(f"Bids: {bids}")
    print(f"Winners: {sorted(result.allocation)}")
    print(f"Social welfare: {result.social_welfare:.2f}")
    print(f"\nPayments and utilities:")
    for i in range(5):
        won = "WIN " if i in result.allocation else "LOSE"
        print(
            f"  Bidder {i}: bid={bids[i]:5.0f}  {won}  "
            f"payment={result.payments[i]:6.2f}  "
            f"utility={result.utilities[i]:6.2f}"
        )
    print(f"\nTotal revenue: {result.budget_surplus:.2f}")


def demo_public_goods() -> None:
    """Public project provision with budget deficit."""
    print("\n" + "=" * 60)
    print("3. PUBLIC PROJECT VCG MECHANISM")
    print("=" * 60)
    print(
        "\nA public project costs C=80 to build. Four agents have\n"
        "private values. VCG decides efficiently but cannot cover\n"
        "the cost (Green-Laffont impossibility).\n"
    )

    valuations = {0: 30, 1: 25, 2: 20, 3: 15}
    mech = PublicProjectMechanism(4, cost=80.0)
    info = mech.demonstrate_budget_deficit(valuations)
    result = info["result"]

    print(f"Valuations: {valuations}")
    print(f"Total value: {sum(valuations.values()):.0f}")
    print(f"Project cost: {mech.cost:.0f}")
    print(f"Build decision: {'YES' if result.build else 'NO'}")

    if result.build:
        print(f"\nPayments (Clarke pivot):")
        for i in range(4):
            pivotal = "pivotal" if result.payments[i] > 0 else "not pivotal"
            print(
                f"  Agent {i}: valuation={valuations[i]:5.0f}  "
                f"payment={result.payments[i]:6.2f}  "
                f"utility={result.utilities[i]:6.2f}  ({pivotal})"
            )
        print(f"\nTotal payments: {result.total_payments:.2f}")
        print(f"Budget deficit: {result.budget_deficit:.2f}")
        print(
            "\n>>> This demonstrates the Green-Laffont impossibility:\n"
            ">>> VCG is efficient and strategyproof, but cannot raise\n"
            ">>> enough revenue to cover the cost of public goods."
        )


def demo_facility_location() -> None:
    """Facility location on a line."""
    print("\n" + "=" * 60)
    print("4. FACILITY LOCATION (MEDIAN MECHANISM)")
    print("=" * 60)
    print(
        "\nPlacing a public facility on a line. The median mechanism\n"
        "is strategyproof without any payments.\n"
    )

    reports = {0: 1, 1: 3, 2: 7, 3: 8, 4: 12}
    mech = FacilityLocationMechanism(5, method="median")
    result = mech.run(reports)

    print(f"Agent locations: {reports}")
    print(f"Facility placed at: {result.location:.1f} (median)")
    print(f"Total cost (sum of distances): {result.total_cost:.1f}")
    print(f"\nAgent distances:")
    for i in range(5):
        print(f"  Agent {i}: location={reports[i]:5.1f}  distance={result.agent_costs[i]:5.1f}")

    # Verify strategyproofness
    sp = mech.verify_strategyproofness(reports)
    print(f"\nStrategyproof: {all(sp.values())}")


def demo_truthfulness_verification() -> None:
    """Demonstrate manipulation detection."""
    print("\n" + "=" * 60)
    print("5. TRUTHFULNESS VERIFICATION")
    print("=" * 60)
    print(
        "\nVerify that no agent gains by misreporting in a\n"
        "Vickrey auction (brute-force over type space).\n"
    )

    n = 3
    auction = VickreyAuction(n)
    vcg = auction.as_vcg()

    type_space = [10.0, 30.0, 50.0, 70.0, 90.0]
    type_spaces = {i: type_space for i in range(n)}

    detector = ManipulationDetector(vcg, type_spaces)

    # Check a specific profile
    true_types = {0: 70.0, 1: 50.0, 2: 30.0}
    analysis = detector.analyse(true_types)

    print(f"True types: {true_types}")
    print(f"Type space checked: {type_space}")
    print(f"Strategyproof: {analysis.is_strategyproof}")
    print(f"Individually rational: {analysis.is_individually_rational}")

    # Show side-by-side comparison
    print("\n--- Side-by-side: Agent 1 overbids ---")
    comparison = detector.compare_truthful_vs_deviation(
        true_types=true_types,
        agent=1,
        deviation_type=90.0,  # overbid from 50 to 90
    )

    print(f"Agent 1 true value: {comparison['true_type']:.0f}")
    print(f"Agent 1 deviation:  {comparison['deviation_type']:.0f}")
    print(f"\nTruthful:  allocation={comparison['truthful']['allocation']}, "
          f"payment={comparison['truthful']['payment']:.0f}, "
          f"utility={comparison['truthful']['utility']:.0f}")
    print(f"Deviated:  allocation={comparison['deviated']['allocation']}, "
          f"payment={comparison['deviated']['payment']:.0f}, "
          f"true_utility={comparison['deviated']['true_utility']:.0f}")
    print(f"Utility change: {comparison['utility_change']:.2f}")
    print(f"Profitable? {comparison['manipulation_profitable']}")


def demo_agv_mechanism() -> None:
    """AGV mechanism with expected budget balance."""
    print("\n" + "=" * 60)
    print("6. AGV MECHANISM (EXPECTED BUDGET BALANCE)")
    print("=" * 60)
    print(
        "\nThe AGV mechanism sacrifices dominant-strategy IC for\n"
        "expected budget balance. It is Bayesian IC instead.\n"
    )

    # Single-item allocation with 2 agents
    n = 2
    allocations = [0, 1]  # who gets the item

    def valuation_fn(agent: int, allocation: int, bid: float) -> float:
        return bid if allocation == agent else 0.0

    # Prior: each agent's type is uniform over {20, 40, 60, 80}
    type_dist = [(20.0, 0.25), (40.0, 0.25), (60.0, 0.25), (80.0, 0.25)]
    type_distributions = {0: type_dist, 1: type_dist}

    agv = AGVMechanism(
        num_agents=n,
        allocations=allocations,
        valuation_fn=valuation_fn,
        type_distributions=type_distributions,
    )

    # Run for a specific profile
    reports = {0: 60.0, 1: 40.0}
    result = agv.solve(reports)

    print(f"Reports: {reports}")
    print(f"Allocation: agent {result.allocation} wins")
    print(f"AGV payments: {result.payments}")
    print(f"AGV utilities: {result.utilities}")
    print(f"Budget surplus: {result.budget_surplus:.2f}")
    print(f"Expected budget surplus: {result.expected_budget_surplus:.4f}")

    # Compare with VCG
    vcg = VCGMechanism(
        num_agents=n, allocations=allocations, valuation_fn=valuation_fn
    )
    vcg_result = vcg.solve(reports)
    print(f"\nVCG payments for comparison: {vcg_result.payments}")
    print(f"VCG budget surplus: {vcg_result.budget_surplus:.2f}")


def main() -> None:
    print("=" * 60)
    print("VCG MECHANISM DESIGN FRAMEWORK - FULL DEMONSTRATION")
    print("=" * 60)

    demo_vickrey_auction()
    demo_multi_unit_auction()
    demo_public_goods()
    demo_facility_location()
    demo_truthfulness_verification()
    demo_agv_mechanism()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print(
        "\nFor interactive exploration, run:\n"
        "  streamlit run src/viz/app.py\n"
    )


if __name__ == "__main__":
    main()
