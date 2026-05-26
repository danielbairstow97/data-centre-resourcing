"""
Data Centre Network Solver
==========================

Wraps PyPSA's optimisation with:
  - Rooftop solar capacity constraint (area-limited p_nom_max)
  - Rooftop solar TMY/clear-sky p_max_pu profile injection
  - Post-solve metric computation (PUE, WUE, CER, CUE, REF, ERE)
  - Gurobi (or any PyPSA-supported solver) configuration

Usage::

    from dc.network.builder import Builder
    from dc.network.solver import Solver
    from dc.config.models import DataCentreConfig

    cfg = DataCentreConfig.from_yaml("config.yaml")
    network = Builder(cfg).build()

    solver = Solver(network, cfg)
    result = solver.solve()

    print(result.metrics.summary())
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import pandas as pd
import pypsa

from dc.network.metrics import DCMetrics, MetricsCalculator
from dc.network.models import SolverConfig

# Set the license file path before importing gurobipy
os.environ["GRB_LICENSE_FILE"] = "/workspaces/Uni/gurobi.lic"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SolveResult:
    """
    Outcome of a Solver.solve() call.

    Attributes
    ----------
    status : str
        PyPSA optimisation status ("ok", "warning", etc.)
    termination_condition : str
        Solver termination condition ("optimal", "infeasible", etc.)
    objective : float
        Optimal objective value (total system cost, $)
    metrics : DCMetrics
        Computed data centre efficiency metrics.
    n : pypsa.Network
        The solved network (same object as Solver.n, post-solve).
    """

    status: str
    termination_condition: str
    objective: float
    metrics: DCMetrics
    n: pypsa.Network

    @property
    def optimal(self) -> bool:
        return self.termination_condition == "optimal"


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@dataclass
class Solver:
    """
    Solves a built PyPSA data centre network.

    Parameters
    ----------
    n : pypsa.Network
        Network from Builder.build() — not yet solved.
    cfg : DataCentreConfig
        Full facility configuration (used for roof constraint, PR, etc.)
    solver_name : str
        PyPSA solver name: "highs" (default, free), "gurobi", "glpk", etc.
    gurobi_license : str | None
        Path to Gurobi licence file.  If None, uses GRB_LICENSE_FILE env var
        or Gurobi's default search path.
    use_tmy : bool
        If True, attempt to fetch PVGIS TMY data for the rooftop solar profile.
        Falls back to clear-sky on failure.  Default True.
    solver_options : dict
        Extra keyword arguments passed to n.optimize() solver_options.
    """

    n: pypsa.Network
    cfg: SolverConfig

    def solve(self) -> SolveResult:
        """
        Prepare the network and run the optimisation.

        Steps:
          1. Set Gurobi licence if applicable
          2. Apply rooftop solar p_max_pu profile (TMY or clear-sky)
          3. Apply rooftop solar p_nom_max from roof area geometry
          4. Run n.optimize() with extra_functionality hook
          5. Compute DCMetrics from solved network
        """

        logger.info(
            "Solving (snapshots=%d, components: %dG %dL %dS)",
            len(self.n.snapshots),
            len(self.n.generators),
            len(self.n.links),
            len(self.n.stores),
        )

        status, condition = self.n.optimize(
            solver_name=self.cfg.name,
            log_to_console=False,
            solver_options={
                "NumericFocus": 3,       # max numerical care
                "ObjScale": -1,          # auto-scale objective (Gurobi picks)
                "ScaleFlag": 2,          # aggressive scaling
            },
            extra_functionality=self._extra_functionality,
        )

        logger.info("Solve complete: status=%s, condition=%s", status, condition)
        if condition != "optimal":
            logger.warning("Non-optimal termination: %s", condition)
            self.n.model.print_infeasibilities()
            raise ValueError("Not optimal")

        objective = self.n.objective

        metrics = MetricsCalculator(
            self.n,
            grid_co2_intensity=1.0,
        ).compute()

        logger.info("\n%s", metrics.summary())

        return SolveResult(
            status=status,
            termination_condition=condition,
            objective=objective,
            metrics=metrics,
            n=self.n,
        )

    # ------------------------------------------------------------------
    # Extra functionality hook (called by PyPSA before solving)
    # ------------------------------------------------------------------
    def _extra_functionality(self, n: pypsa.Network, snapshots: pd.Index) -> None:

        pass
