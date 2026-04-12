"""Command-line interface for VCG mechanism framework.

Usage examples:
    python -m src.cli vickrey --bids 100 80 60 40
    python -m src.cli multi-unit --bids 100 80 60 40 --items 2
    python -m src.cli public-good --valuations 30 25 20 15 --cost 50
    python -m src.cli facility --locations 1 3 5 7 9
    python -m src.cli demo
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List

from src.core.auctions import MultiUnitAuction, VickreyAuction
from src.core.facility import FacilityLocationMechanism
from src.core.public_goods import PublicProjectMechanism


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="VCG Mechanism Design Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Mechanism to run")

    # Vickrey auction
    p_vickrey = subparsers.add_parser("vickrey", help="Single-item Vickrey auction")
    p_vickrey.add_argument(
        "--bids", nargs="+", type=float, required=True, help="Bid values"
    )

    # Multi-unit auction
    p_multi = subparsers.add_parser("multi-unit", help="Multi-unit VCG auction")
    p_multi.add_argument(
        "--bids", nargs="+", type=float, required=True, help="Bid values"
    )
    p_multi.add_argument(
        "--items", type=int, required=True, help="Number of items"
    )

    # Public project
    p_pub = subparsers.add_parser("public-good", help="Public project VCG mechanism")
    p_pub.add_argument(
        "--valuations", nargs="+", type=float, required=True, help="Agent valuations"
    )
    p_pub.add_argument(
        "--cost", type=float, required=True, help="Project cost"
    )

    # Facility location
    p_fac = subparsers.add_parser("facility", help="Facility location mechanism")
    p_fac.add_argument(
        "--locations", nargs="+", type=float, required=True, help="Agent ideal locations"
    )
    p_fac.add_argument(
        "--method",
        choices=["median", "vcg"],
        default="median",
        help="Mechanism type (default: median)",
    )

    # Demo
    subparsers.add_parser("demo", help="Run demonstration examples")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    if args.command == "vickrey":
        _run_vickrey(args.bids)
    elif args.command == "multi-unit":
        _run_multi_unit(args.bids, args.items)
    elif args.command == "public-good":
        _run_public_good(args.valuations, args.cost)
    elif args.command == "facility":
        _run_facility(args.locations, args.method)
    elif args.command == "demo":
        _run_demo()


def _run_vickrey(bids: List[float]) -> None:
    n = len(bids)
    bids_dict = {i: b for i, b in enumerate(bids)}

    print(f"\n{'='*50}")
    print("SINGLE-ITEM VICKREY AUCTION")
    print(f"{'='*50}")
    print(f"Bidders: {n}")
    print(f"Bids: {bids}")

    auction = VickreyAuction(n)
    result = auction.run(bids_dict)

    print(f"\nWinner: Bidder {result.allocation}")
    print(f"Winning bid: {bids_dict[result.allocation]:.2f}")
    print(f"Payment (2nd price): {result.payments[result.allocation]:.2f}")
    print(f"Social welfare: {result.social_welfare:.2f}")

    print("\nAll outcomes:")
    print(f"  {'Bidder':<10}{'Bid':<12}{'Payment':<12}{'Utility':<12}")
    print(f"  {'-'*46}")
    for i in range(n):
        print(
            f"  {i:<10}{bids_dict[i]:<12.2f}"
            f"{result.payments[i]:<12.2f}{result.utilities[i]:<12.2f}"
        )


def _run_multi_unit(bids: List[float], num_items: int) -> None:
    n = len(bids)
    bids_dict = {i: b for i, b in enumerate(bids)}

    print(f"\n{'='*50}")
    print("MULTI-UNIT VCG AUCTION")
    print(f"{'='*50}")
    print(f"Bidders: {n}, Items: {num_items}")
    print(f"Bids: {bids}")

    auction = MultiUnitAuction(n, num_items)
    result = auction.run(bids_dict)

    print(f"\nWinners: {sorted(result.allocation)}")
    print(f"Social welfare: {result.social_welfare:.2f}")
    print(f"Total revenue: {result.budget_surplus:.2f}")

    print("\nAll outcomes:")
    print(f"  {'Bidder':<10}{'Bid':<12}{'Won':<8}{'Payment':<12}{'Utility':<12}")
    print(f"  {'-'*54}")
    for i in range(n):
        won = "Yes" if i in result.allocation else "No"
        print(
            f"  {i:<10}{bids_dict[i]:<12.2f}{won:<8}"
            f"{result.payments[i]:<12.2f}{result.utilities[i]:<12.2f}"
        )


def _run_public_good(valuations: List[float], cost: float) -> None:
    n = len(valuations)
    vals_dict = {i: v for i, v in enumerate(valuations)}

    print(f"\n{'='*50}")
    print("PUBLIC PROJECT VCG MECHANISM")
    print(f"{'='*50}")
    print(f"Agents: {n}, Cost: {cost}")
    print(f"Valuations: {valuations}")

    mech = PublicProjectMechanism(n, cost)
    info = mech.demonstrate_budget_deficit(vals_dict)
    result = info["result"]

    print(f"\nBuild project: {'YES' if result.build else 'NO'}")
    print(f"Total value: {sum(valuations):.2f}")
    print(f"Total payments: {result.total_payments:.2f}")

    if result.build:
        print(f"Budget deficit: {result.budget_deficit:.2f}")

    print(f"\n{info['explanation']}")

    print("\nAll outcomes:")
    print(f"  {'Agent':<10}{'Valuation':<12}{'Payment':<12}{'Utility':<12}")
    print(f"  {'-'*46}")
    for i in range(n):
        print(
            f"  {i:<10}{vals_dict[i]:<12.2f}"
            f"{result.payments[i]:<12.2f}{result.utilities[i]:<12.2f}"
        )


def _run_facility(locations: List[float], method: str) -> None:
    n = len(locations)
    locs_dict = {i: loc for i, loc in enumerate(locations)}

    print(f"\n{'='*50}")
    print(f"FACILITY LOCATION ({method.upper()} MECHANISM)")
    print(f"{'='*50}")
    print(f"Agents: {n}")
    print(f"Ideal locations: {locations}")

    mech = FacilityLocationMechanism(n, method=method)
    result = mech.run(locs_dict)

    print(f"\nFacility location: {result.location:.2f}")
    print(f"Total cost (sum of distances): {result.total_cost:.2f}")

    print("\nAll outcomes:")
    print(f"  {'Agent':<10}{'Location':<12}{'Distance':<12}{'Payment':<12}")
    print(f"  {'-'*46}")
    for i in range(n):
        print(
            f"  {i:<10}{locs_dict[i]:<12.2f}"
            f"{result.agent_costs[i]:<12.2f}{result.payments[i]:<12.2f}"
        )


def _run_demo() -> None:
    print("\n" + "=" * 60)
    print("VCG MECHANISM DESIGN FRAMEWORK - DEMONSTRATION")
    print("=" * 60)

    # Demo 1: Vickrey auction
    _run_vickrey([100, 80, 60, 40])

    # Demo 2: Multi-unit auction
    _run_multi_unit([90, 75, 60, 45, 30], 2)

    # Demo 3: Public good
    _run_public_good([30, 25, 20, 15], 50)

    # Demo 4: Facility location
    _run_facility([1, 3, 5, 7, 9], "median")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
