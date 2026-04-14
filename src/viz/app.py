"""Streamlit interactive mechanism design simulator.

Run with:  streamlit run src/viz/app.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from src.core.auctions import MultiUnitAuction, VickreyAuction
from src.core.manipulation import ManipulationDetector
from src.core.public_goods import PublicProjectMechanism


def main() -> None:
    st.set_page_config(
        page_title="VCG Mechanism Design Simulator",
        page_icon="",
        layout="wide",
    )

    st.title("VCG Mechanism Design Simulator")
    st.markdown(
        "Interactive exploration of Vickrey-Clarke-Groves mechanisms "
        "for auctions, public goods, and facility location."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Mechanism Simulator",
            "Truthfulness Proof",
            "Budget Analysis",
            "Impossibility Explorer",
        ]
    )

    with tab1:
        _mechanism_simulator()

    with tab2:
        _truthfulness_proof()

    with tab3:
        _budget_analysis()

    with tab4:
        _impossibility_explorer()


# =====================================================================
# Tab 1: Mechanism Simulator
# =====================================================================


def _mechanism_simulator() -> None:
    st.header("Mechanism Simulator")
    st.markdown(
        "Define agents, their types (valuations), and see the "
        "welfare-maximising allocation and VCG payments."
    )

    mechanism_type = st.selectbox(
        "Mechanism type",
        ["Single-item Vickrey auction", "Multi-unit auction", "Public project"],
        key="sim_mech_type",
    )

    if mechanism_type == "Single-item Vickrey auction":
        _sim_vickrey()
    elif mechanism_type == "Multi-unit auction":
        _sim_multi_unit()
    elif mechanism_type == "Public project":
        _sim_public_project()


def _sim_vickrey() -> None:
    n = st.slider("Number of bidders", 2, 10, 3, key="sim_vickrey_n")
    st.markdown("Enter bids:")
    bids: dict[int, float] = {}
    cols = st.columns(min(n, 5))
    for i in range(n):
        col = cols[i % len(cols)]
        bids[i] = col.number_input(
            f"Bidder {i}", value=float((i + 1) * 10), key=f"sim_vickrey_bid_{i}"
        )

    if st.button("Run Vickrey Auction", key="sim_vickrey_run"):
        auction = VickreyAuction(n)
        result = auction.run(bids)

        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Winner", f"Bidder {result.allocation}")
            st.metric("Winning bid", f"{bids[result.allocation]:.2f}")
        with col2:
            winner_payment = result.payments[result.allocation]
            st.metric("Payment (2nd price)", f"{winner_payment:.2f}")
            st.metric("Social welfare", f"{result.social_welfare:.2f}")

        st.subheader("All payments and utilities")
        data = {
            "Bidder": list(range(n)),
            "Bid": [bids[i] for i in range(n)],
            "Payment": [result.payments[i] for i in range(n)],
            "Utility": [result.utilities[i] for i in range(n)],
        }
        st.table(data)


def _sim_multi_unit() -> None:
    n = st.slider("Number of bidders", 2, 10, 5, key="sim_multi_n")
    k = st.slider("Number of items", 1, n - 1, 2, key="sim_multi_k")

    st.markdown("Enter bids:")
    bids: dict[int, float] = {}
    cols = st.columns(min(n, 5))
    for i in range(n):
        col = cols[i % len(cols)]
        bids[i] = col.number_input(
            f"Bidder {i}",
            value=float(np.random.RandomState(i).randint(10, 100)),
            key=f"sim_multi_bid_{i}",
        )

    if st.button("Run Multi-Unit Auction", key="sim_multi_run"):
        auction = MultiUnitAuction(n, k)
        result = auction.run(bids)

        st.subheader("Results")
        st.metric("Winners", str(sorted(result.allocation)))
        st.metric("Social welfare", f"{result.social_welfare:.2f}")

        st.subheader("All payments and utilities")
        data = {
            "Bidder": list(range(n)),
            "Bid": [bids[i] for i in range(n)],
            "Won": ["Yes" if i in result.allocation else "No" for i in range(n)],
            "Payment": [result.payments[i] for i in range(n)],
            "Utility": [result.utilities[i] for i in range(n)],
        }
        st.table(data)


def _sim_public_project() -> None:
    n = st.slider("Number of agents", 2, 10, 4, key="sim_pub_n")
    cost = st.number_input("Project cost", value=50.0, key="sim_pub_cost")

    st.markdown("Enter valuations for the project:")
    vals: dict[int, float] = {}
    cols = st.columns(min(n, 5))
    for i in range(n):
        col = cols[i % len(cols)]
        vals[i] = col.number_input(
            f"Agent {i}", value=float(15 + i * 5), key=f"sim_pub_val_{i}"
        )

    if st.button("Run Public Project Mechanism", key="sim_pub_run"):
        mech = PublicProjectMechanism(n, cost)
        result = mech.run(vals)

        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Build project?", "YES" if result.build else "NO")
            st.metric("Total value", f"{sum(vals.values()):.2f}")
        with col2:
            st.metric("Project cost", f"{cost:.2f}")
            st.metric("Total payments", f"{result.total_payments:.2f}")

        if result.build and result.budget_deficit > 0:
            st.warning(
                f"Budget DEFICIT: {result.budget_deficit:.2f}. "
                "VCG does not collect enough to cover the cost!"
            )

        st.subheader("Individual outcomes")
        data = {
            "Agent": list(range(n)),
            "Valuation": [vals[i] for i in range(n)],
            "Payment": [result.payments[i] for i in range(n)],
            "Utility": [result.utilities[i] for i in range(n)],
        }
        st.table(data)


# =====================================================================
# Tab 2: Truthfulness Proof
# =====================================================================


def _truthfulness_proof() -> None:
    st.header("Truthfulness Verification")
    st.markdown(
        "Side-by-side comparison: what happens if an agent reports "
        "truthfully vs. deviates?"
    )

    mech_type = st.selectbox(
        "Mechanism",
        ["Vickrey auction", "Multi-unit auction"],
        key="truth_mech_type",
    )

    if mech_type == "Vickrey auction":
        n = st.slider("Number of bidders", 2, 6, 3, key="truth_vickrey_n")

        st.markdown("**True valuations:**")
        true_vals: dict[int, float] = {}
        cols = st.columns(min(n, 5))
        for i in range(n):
            col = cols[i % len(cols)]
            true_vals[i] = col.number_input(
                f"Bidder {i} true value",
                value=float((n - i) * 20),
                key=f"truth_val_{i}",
            )

        deviator = st.selectbox(
            "Agent considering deviation",
            list(range(n)),
            key="truth_deviator",
        )
        dev_value = st.number_input(
            "Deviated report",
            value=true_vals[deviator] * 1.5,
            key="truth_dev_val",
        )

        if st.button("Compare", key="truth_compare"):
            auction = VickreyAuction(n)
            vcg = auction.as_vcg()

            detector = ManipulationDetector(
                mechanism=vcg,
                type_spaces={i: [true_vals[i]] for i in range(n)},
            )

            comparison = detector.compare_truthful_vs_deviation(
                true_types=true_vals,
                agent=deviator,
                deviation_type=dev_value,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Truthful reporting")
                st.metric("Allocation", str(comparison["truthful"]["allocation"]))
                st.metric("Payment", f"{comparison['truthful']['payment']:.2f}")
                st.metric("Utility", f"{comparison['truthful']['utility']:.2f}")

            with col2:
                st.subheader(f"Deviation to {dev_value:.2f}")
                st.metric("Allocation", str(comparison["deviated"]["allocation"]))
                st.metric("Payment", f"{comparison['deviated']['payment']:.2f}")
                dev_util = comparison["deviated"]["true_utility"]
                st.metric("True utility", f"{dev_util:.2f}")

            change = comparison["utility_change"]
            if comparison["manipulation_profitable"]:
                st.error(
                    f"Deviation is profitable! Utility gain: {change:.4f}"
                )
            else:
                st.success(
                    f"Deviation is NOT profitable. Utility change: {change:.4f}"
                )

    elif mech_type == "Multi-unit auction":
        n = st.slider("Number of bidders", 3, 8, 5, key="truth_multi_n")
        k = st.slider("Number of items", 1, n - 1, 2, key="truth_multi_k")

        st.markdown("**True valuations:**")
        true_vals_m: dict[int, float] = {}
        cols = st.columns(min(n, 5))
        for i in range(n):
            col = cols[i % len(cols)]
            true_vals_m[i] = col.number_input(
                f"Bidder {i} true value",
                value=float(50 - i * 8),
                key=f"truth_multi_val_{i}",
            )

        deviator_m = st.selectbox(
            "Agent considering deviation",
            list(range(n)),
            key="truth_multi_deviator",
        )
        dev_value_m = st.number_input(
            "Deviated report",
            value=true_vals_m[deviator_m] * 2,
            key="truth_multi_dev_val",
        )

        if st.button("Compare", key="truth_multi_compare"):
            auction_m = MultiUnitAuction(n, k)

            # Truthful
            truth_res = auction_m.run(true_vals_m)

            # Deviated
            dev_vals = dict(true_vals_m)
            dev_vals[deviator_m] = dev_value_m
            dev_res = auction_m.run(dev_vals)

            # True utility under deviation
            won_dev = deviator_m in dev_res.allocation
            true_util_dev = (
                true_vals_m[deviator_m] - dev_res.payments[deviator_m]
                if won_dev
                else 0.0
            )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Truthful reporting")
                st.metric("Winners", str(sorted(truth_res.allocation)))
                st.metric("Payment", f"{truth_res.payments[deviator_m]:.2f}")
                st.metric("Utility", f"{truth_res.utilities[deviator_m]:.2f}")

            with col2:
                st.subheader(f"Deviation to {dev_value_m:.2f}")
                st.metric("Winners", str(sorted(dev_res.allocation)))
                st.metric("Payment", f"{dev_res.payments[deviator_m]:.2f}")
                st.metric("True utility", f"{true_util_dev:.2f}")

            change = true_util_dev - truth_res.utilities[deviator_m]
            if change > 1e-10:
                st.error(f"Deviation is profitable! Utility gain: {change:.4f}")
            else:
                st.success(
                    f"Deviation is NOT profitable. Utility change: {change:.4f}"
                )


# =====================================================================
# Tab 3: Budget Analysis
# =====================================================================


def _budget_analysis() -> None:
    st.header("Budget Analysis Dashboard")
    st.markdown(
        "Explore budget properties of different mechanisms. "
        "VCG (Clarke pivot) always runs a non-negative budget surplus "
        "for private goods, but a deficit for public goods."
    )

    analysis_type = st.selectbox(
        "Analysis type",
        [
            "Single auction budget",
            "Public goods deficit demo",
            "Monte Carlo budget distribution",
        ],
        key="budget_type",
    )

    if analysis_type == "Single auction budget":
        n = st.slider("Number of bidders", 2, 10, 5, key="budget_auction_n")
        bids_budget: dict[int, float] = {}
        cols = st.columns(min(n, 5))
        for i in range(n):
            col = cols[i % len(cols)]
            bids_budget[i] = col.number_input(
                f"Bid {i}",
                value=float(np.random.RandomState(42 + i).randint(10, 100)),
                key=f"budget_bid_{i}",
            )

        if st.button("Analyse", key="budget_auction_run"):
            auction = VickreyAuction(n)
            result = auction.run(bids_budget)
            st.metric("Budget surplus (revenue)", f"{result.budget_surplus:.2f}")
            st.info(
                "For single-item Vickrey auction, budget surplus = "
                f"second-price = {result.budget_surplus:.2f} >= 0."
            )

    elif analysis_type == "Public goods deficit demo":
        n = st.slider("Number of agents", 2, 8, 4, key="budget_pub_n")
        cost = st.number_input(
            "Project cost", value=40.0, key="budget_pub_cost"
        )
        vals_budget: dict[int, float] = {}
        cols = st.columns(min(n, 5))
        for i in range(n):
            col = cols[i % len(cols)]
            vals_budget[i] = col.number_input(
                f"Value {i}",
                value=float(12 + i * 3),
                key=f"budget_pub_val_{i}",
            )

        if st.button("Analyse", key="budget_pub_run"):
            mech = PublicProjectMechanism(n, cost)
            info = mech.demonstrate_budget_deficit(vals_budget)
            result = info["result"]
            st.text(info["explanation"])

            if result.build:
                st.metric("Total payments", f"{result.total_payments:.2f}")
                st.metric("Project cost", f"{cost:.2f}")
                deficit = result.budget_deficit
                if deficit > 0:
                    st.error(f"Budget DEFICIT: {deficit:.2f}")
                else:
                    st.success(f"Budget surplus: {-deficit:.2f}")

    elif analysis_type == "Monte Carlo budget distribution":
        n = st.slider("Number of bidders", 2, 8, 4, key="budget_mc_n")
        max_val = st.number_input(
            "Max valuation", value=100.0, key="budget_mc_max"
        )
        num_sims = st.slider(
            "Number of simulations", 100, 5000, 1000, key="budget_mc_sims"
        )

        if st.button("Run Monte Carlo", key="budget_mc_run"):
            rng = np.random.RandomState(42)
            surpluses = []
            for _ in range(num_sims):
                bids_mc = {i: rng.uniform(0, max_val) for i in range(n)}
                auction = VickreyAuction(n)
                result = auction.run(bids_mc)
                surpluses.append(result.budget_surplus)

            surpluses_arr = np.array(surpluses)
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean surplus", f"{surpluses_arr.mean():.2f}")
            col2.metric("Min surplus", f"{surpluses_arr.min():.2f}")
            col3.metric("Max surplus", f"{surpluses_arr.max():.2f}")

            st.bar_chart(
                np.histogram(surpluses_arr, bins=30)[0],
            )
            st.caption("Distribution of budget surplus across simulations")


# =====================================================================
# Tab 4: Impossibility Explorer
# =====================================================================


def _impossibility_explorer() -> None:
    st.header("Impossibility Theorem Explorer")
    st.markdown(
        "In mechanism design, certain combinations of desirable "
        "properties are impossible to achieve simultaneously."
    )

    st.subheader("Select desired properties")

    col1, col2 = st.columns(2)

    with col1:
        efficiency = st.checkbox("Allocative efficiency", value=True, key="imp_eff")
        dsic = st.checkbox(
            "Dominant-strategy incentive compatible (DSIC)", value=True, key="imp_dsic"
        )
        bic = st.checkbox(
            "Bayesian incentive compatible (BIC)", value=False, key="imp_bic"
        )

    with col2:
        ir = st.checkbox("Individual rationality (IR)", value=True, key="imp_ir")
        bb = st.checkbox("Budget balance", value=True, key="imp_bb")
        no_deficit = st.checkbox(
            "No budget deficit (weak BB)", value=False, key="imp_nodef"
        )

    st.markdown("---")
    st.subheader("Feasibility analysis")

    # Analyse which combinations are achievable
    results = _analyse_impossibility(efficiency, dsic, bic, ir, bb, no_deficit)

    for item in results:
        if item["achievable"]:
            st.success(f"ACHIEVABLE: {item['description']}")
            st.markdown(f"  *Mechanism:* {item['mechanism']}")
        else:
            st.error(f"IMPOSSIBLE: {item['description']}")
            st.markdown(f"  *Theorem:* {item['theorem']}")


def _analyse_impossibility(
    efficiency: bool,
    dsic: bool,
    bic: bool,
    ir: bool,
    bb: bool,
    no_deficit: bool,
) -> list[dict[str, Any]]:
    """Analyse feasibility of desired property combinations."""
    results = []

    # Check known impossibility results
    if efficiency and dsic and bb:
        results.append(
            {
                "description": "Efficiency + DSIC + Budget Balance",
                "achievable": False,
                "theorem": (
                    "Green-Laffont (1979): No mechanism for public goods "
                    "can be efficient, DSIC, and budget-balanced. "
                    "VCG is efficient and DSIC but runs a deficit."
                ),
                "mechanism": None,
            }
        )

    if efficiency and dsic and not bb:
        results.append(
            {
                "description": "Efficiency + DSIC (without budget balance)",
                "achievable": True,
                "theorem": None,
                "mechanism": (
                    "VCG mechanism (Clarke pivot): maximises welfare, "
                    "truthful in dominant strategies, but may run a deficit."
                ),
            }
        )

    if efficiency and bic and bb:
        results.append(
            {
                "description": "Efficiency + BIC + Budget Balance",
                "achievable": True,
                "theorem": None,
                "mechanism": (
                    "AGV mechanism (d'Aspremont-Gerard-Varet): efficient, "
                    "Bayesian IC, and expected budget-balanced."
                ),
            }
        )

    if dsic and ir and efficiency:
        results.append(
            {
                "description": "DSIC + IR + Efficiency (private goods)",
                "achievable": True,
                "theorem": None,
                "mechanism": (
                    "VCG mechanism for private goods (e.g., auctions): "
                    "all three properties are satisfied."
                ),
            }
        )

    if dsic and not efficiency:
        results.append(
            {
                "description": "DSIC without efficiency",
                "achievable": True,
                "theorem": None,
                "mechanism": (
                    "Dictatorial mechanisms, posted-price mechanisms, or "
                    "serial dictatorships are DSIC but not necessarily efficient."
                ),
            }
        )

    if efficiency and dsic and ir and bb:
        results.append(
            {
                "description": "Efficiency + DSIC + IR + Budget Balance",
                "achievable": False,
                "theorem": (
                    "Myerson-Satterthwaite (1983): For bilateral trade, "
                    "no mechanism can be simultaneously efficient, DSIC, "
                    "IR, and budget-balanced."
                ),
                "mechanism": None,
            }
        )

    if no_deficit and efficiency and dsic:
        results.append(
            {
                "description": "No deficit + Efficiency + DSIC",
                "achievable": True,
                "theorem": None,
                "mechanism": (
                    "VCG with Clarke pivot for auctions: the sum of "
                    "payments is always non-negative (revenue >= 0)."
                ),
            }
        )

    if not results:
        results.append(
            {
                "description": "Selected combination",
                "achievable": True,
                "theorem": None,
                "mechanism": (
                    "The selected properties can generally be achieved. "
                    "Select more restrictive combinations to see "
                    "impossibility results."
                ),
            }
        )

    return results


if __name__ == "__main__":
    main()
