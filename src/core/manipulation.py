"""Manipulation detection for mechanism design.

Compares truthful reporting against all possible misreports to detect
whether any agent can gain by deviating from truthful behaviour.

Key concepts:
  - Dominant-strategy incentive compatibility (DSIC): no agent can gain
    by misreporting, regardless of others' reports.
  - Bayesian incentive compatibility (BIC): no agent can gain in
    expectation by misreporting, given the prior.
  - Individual rationality (IR): no agent is worse off by participating.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core.vcg import MechanismResult, VCGMechanism


@dataclass
class ManipulationReport:
    """Report of manipulation analysis for a single agent."""

    agent: int
    truthful_type: Any
    truthful_utility: float
    best_deviation_type: Any
    best_deviation_utility: float
    is_manipulable: bool  # True if deviation improves utility
    utility_gain: float  # best_deviation_utility - truthful_utility
    all_deviations: List[Tuple[Any, float]]  # (type, utility) pairs


@dataclass
class ManipulationAnalysis:
    """Complete manipulation analysis across all agents."""

    agent_reports: Dict[int, ManipulationReport]
    is_strategyproof: bool
    is_individually_rational: bool
    participation_utilities: Dict[int, float]  # utility from participating vs. 0

    def summary(self) -> str:
        lines = ["=== Manipulation Analysis ==="]
        lines.append(f"Strategyproof: {self.is_strategyproof}")
        lines.append(
            f"Individually rational: {self.is_individually_rational}"
        )
        lines.append("")

        for i, report in sorted(self.agent_reports.items()):
            lines.append(f"Agent {i}:")
            lines.append(f"  Truthful type:    {report.truthful_type}")
            lines.append(f"  Truthful utility: {report.truthful_utility:.4f}")
            if report.is_manipulable:
                lines.append(
                    f"  MANIPULABLE! Best deviation: {report.best_deviation_type}"
                )
                lines.append(
                    f"  Deviation utility: {report.best_deviation_utility:.4f}"
                )
                lines.append(f"  Utility gain:      {report.utility_gain:.4f}")
            else:
                lines.append("  No beneficial deviation found.")
            lines.append("")

        return "\n".join(lines)


class ManipulationDetector:
    """Detect manipulation opportunities in mechanisms.

    Parameters
    ----------
    mechanism : VCGMechanism
        The mechanism to analyse.
    type_spaces : dict
        For each agent, a list of possible types (the type space to
        search over for deviations).
    """

    def __init__(
        self,
        mechanism: VCGMechanism,
        type_spaces: Dict[int, List[Any]],
    ):
        self.mechanism = mechanism
        self.type_spaces = type_spaces

    def analyse(
        self,
        true_types: Dict[int, Any],
        agents: Optional[List[int]] = None,
    ) -> ManipulationAnalysis:
        """Run complete manipulation analysis.

        Parameters
        ----------
        true_types : dict
            The true type profile.
        agents : list or None
            Agents to check; if None, check all.

        Returns
        -------
        ManipulationAnalysis
        """
        if agents is None:
            agents = list(range(self.mechanism.num_agents))

        # Truthful outcome
        truthful_result = self.mechanism.solve(true_types)

        agent_reports: Dict[int, ManipulationReport] = {}
        is_strategyproof = True
        is_ir = True
        participation_utilities: Dict[int, float] = {}

        for i in agents:
            truthful_utility = truthful_result.utilities[i]
            participation_utilities[i] = truthful_utility

            if truthful_utility < -1e-10:
                is_ir = False

            type_space = self.type_spaces.get(i, [true_types[i]])
            all_deviations: List[Tuple[Any, float]] = []

            best_dev_type = true_types[i]
            best_dev_utility = truthful_utility

            for alt_type in type_space:
                deviated_reports = dict(true_types)
                deviated_reports[i] = alt_type

                dev_result = self.mechanism.solve(deviated_reports)

                # Agent's TRUE utility under deviated allocation
                # (they get the deviated allocation but their true
                # valuation determines real satisfaction)
                true_vi = self.mechanism.valuation_fn(
                    i, dev_result.allocation, true_types[i]
                )
                dev_utility = true_vi - dev_result.payments[i]

                all_deviations.append((alt_type, dev_utility))

                if dev_utility > best_dev_utility + 1e-10:
                    best_dev_utility = dev_utility
                    best_dev_type = alt_type

            is_manipulable = best_dev_utility > truthful_utility + 1e-10
            if is_manipulable:
                is_strategyproof = False

            agent_reports[i] = ManipulationReport(
                agent=i,
                truthful_type=true_types[i],
                truthful_utility=truthful_utility,
                best_deviation_type=best_dev_type,
                best_deviation_utility=best_dev_utility,
                is_manipulable=is_manipulable,
                utility_gain=max(0.0, best_dev_utility - truthful_utility),
                all_deviations=all_deviations,
            )

        return ManipulationAnalysis(
            agent_reports=agent_reports,
            is_strategyproof=is_strategyproof,
            is_individually_rational=is_ir,
            participation_utilities=participation_utilities,
        )

    def brute_force_dsic_check(
        self,
        agent: int,
    ) -> Dict[str, Any]:
        """Brute-force DSIC check: for EVERY possible type profile of
        others, check that agent i cannot gain by deviating.

        This checks dominant strategy incentive compatibility (not just
        for a specific profile of others).

        Returns
        -------
        dict with 'is_dsic', 'violations' (list of violation details).
        """
        others = [j for j in range(self.mechanism.num_agents) if j != agent]
        others_type_lists = [self.type_spaces[j] for j in others]

        violations = []

        # Iterate over all profiles of others
        for combo in itertools.product(*others_type_lists):
            type_profile_others = {j: combo[idx] for idx, j in enumerate(others)}

            # For each true type of agent i
            for true_type_i in self.type_spaces[agent]:
                full_true = dict(type_profile_others)
                full_true[agent] = true_type_i

                truthful_result = self.mechanism.solve(full_true)
                truthful_utility = truthful_result.utilities[agent]

                # Check all deviations
                for dev_type_i in self.type_spaces[agent]:
                    full_dev = dict(type_profile_others)
                    full_dev[agent] = dev_type_i

                    dev_result = self.mechanism.solve(full_dev)
                    true_vi = self.mechanism.valuation_fn(
                        agent, dev_result.allocation, true_type_i
                    )
                    dev_utility = true_vi - dev_result.payments[agent]

                    if dev_utility > truthful_utility + 1e-10:
                        violations.append(
                            {
                                "true_type": true_type_i,
                                "deviation_type": dev_type_i,
                                "others_types": dict(type_profile_others),
                                "truthful_utility": truthful_utility,
                                "deviation_utility": dev_utility,
                                "gain": dev_utility - truthful_utility,
                            }
                        )

        return {
            "agent": agent,
            "is_dsic": len(violations) == 0,
            "num_profiles_checked": (
                len(list(itertools.product(*others_type_lists)))
                * len(self.type_spaces[agent])
                * len(self.type_spaces[agent])
            ),
            "violations": violations,
        }

    def compare_truthful_vs_deviation(
        self,
        true_types: Dict[int, Any],
        agent: int,
        deviation_type: Any,
    ) -> Dict[str, Any]:
        """Side-by-side comparison of truthful vs. deviated outcome.

        Parameters
        ----------
        true_types : dict
            True type profile.
        agent : int
            Agent considering deviation.
        deviation_type : any
            The alternative type to report.

        Returns
        -------
        dict with both outcomes and comparison.
        """
        # Truthful outcome
        truthful_result = self.mechanism.solve(true_types)

        # Deviated outcome
        deviated_reports = dict(true_types)
        deviated_reports[agent] = deviation_type
        deviated_result = self.mechanism.solve(deviated_reports)

        # True utility under deviation
        true_vi_dev = self.mechanism.valuation_fn(
            agent, deviated_result.allocation, true_types[agent]
        )
        dev_utility = true_vi_dev - deviated_result.payments[agent]

        return {
            "agent": agent,
            "true_type": true_types[agent],
            "deviation_type": deviation_type,
            "truthful": {
                "allocation": truthful_result.allocation,
                "payment": truthful_result.payments[agent],
                "utility": truthful_result.utilities[agent],
                "social_welfare": truthful_result.social_welfare,
            },
            "deviated": {
                "allocation": deviated_result.allocation,
                "payment": deviated_result.payments[agent],
                "true_utility": dev_utility,
                "social_welfare": deviated_result.social_welfare,
            },
            "utility_change": dev_utility - truthful_result.utilities[agent],
            "manipulation_profitable": (
                dev_utility > truthful_result.utilities[agent] + 1e-10
            ),
        }
