"""
Data Centre Metrics
===================

Computes industry-standard data centre efficiency metrics from a solved
PyPSA network.  All metrics are computed as both time-series (per snapshot)
and annual averages.

Metrics
-------
PUE  Power Usage Effectiveness        total facility power / IT power
WUE  Water Usage Effectiveness        total water (L) / IT energy (MWh)
CER  Cooling Efficiency Ratio         heat removed (MW_th) / cooling elec (MW_e)
ERE  Energy Reuse Effectiveness       reused energy / (IT energy - reused energy)
CUE  Carbon Usage Effectiveness       total CO₂e (t) / IT energy (MWh)
REF  Renewable Energy Factor          renewable MWh / total facility MWh
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DCMetrics:
    """
    Computed data centre metrics, both time-series and annual summaries.

    Time-series attributes are pd.Series indexed to n.snapshots.
    Scalar attributes are annual averages (or totals where noted).
    """

    snapshots: pd.DatetimeIndex

    # ── Time series ────────────────────────────────────────────────────────
    pue_t:  pd.Series = field(default_factory=pd.Series)   # MW_total / MW_it
    wue_t:  pd.Series = field(default_factory=pd.Series)   # L / MWh_it
    cer_t:  pd.Series = field(default_factory=pd.Series)   # MW_th / MW_cooling_elec
    cue_t:  pd.Series = field(default_factory=pd.Series)   # tCO2e / MWh_it
    ref_t:  pd.Series = field(default_factory=pd.Series)   # fraction

    # ── Annual scalars ─────────────────────────────────────────────────────
    pue:    float = float("nan")   # load-weighted average
    wue:    float = float("nan")   # total L / total MWh_it
    cer:    float = float("nan")   # energy-weighted average
    ere:    float = 0.0            # placeholder — zero unless heat recovery modelled
    cue:    float = float("nan")   # total tCO2e / total MWh_it
    ref:    float = float("nan")   # total renewable MWh / total MWh

    # ── Annual totals (MWh unless noted) ──────────────────────────────────
    total_it_MWh:           float = float("nan")
    total_facility_MWh:     float = float("nan")
    total_cooling_elec_MWh: float = float("nan")
    total_water_L:          float = float("nan")
    total_scope1_co2_t:     float = float("nan")
    total_scope2_co2_t:     float = float("nan")
    total_renewable_MWh:    float = float("nan")

    def summary(self) -> str:
        lines = [
            "══ Data Centre Metrics ══════════════════════════════════════════",
            f"  PUE   Power Usage Effectiveness      : {self.pue:.4f}",
            f"  WUE   Water Usage Effectiveness      : {self.wue:.2f} L/MWh_it",
            f"  CER   Cooling Efficiency Ratio        : {self.cer:.4f}",
            f"  ERE   Energy Reuse Effectiveness      : {self.ere:.4f}  (0 = no reuse)",
            f"  CUE   Carbon Usage Effectiveness      : {self.cue:.4f} tCO₂e/MWh_it",
            f"  REF   Renewable Energy Factor         : {self.ref:.1%}",
            "──────────────────────────────────────────────────────────────────",
            f"  IT load             : {self.total_it_MWh:>12,.1f} MWh/yr",
            f"  Total facility      : {self.total_facility_MWh:>12,.1f} MWh/yr",
            f"  Cooling electricity : {self.total_cooling_elec_MWh:>12,.1f} MWh/yr",
            f"  Water consumed      : {self.total_water_L:>12,.0f} L/yr",
            f"  Scope 1 CO₂         : {self.total_scope1_co2_t:>12,.1f} t/yr",
            f"  Scope 2 CO₂         : {self.total_scope2_co2_t:>12,.1f} t/yr",
            f"  Renewable energy    : {self.total_renewable_MWh:>12,.1f} MWh/yr",
            "═════════════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class MetricsCalculator:
    """
    Computes DCMetrics from a solved PyPSA network.

    The network must have been optimised (n.optimize() called) before
    this is instantiated — it reads dispatch results from n.generators_t.p,
    n.links_t.p0, n.stores_t.e, etc.

    Parameters
    ----------
    n : pypsa.Network
        Solved network from Builder.build() + Solver.solve().
    grid_co2_intensity : float
        tCO₂/MWh for grid electricity imports. Falls back to
        n.carriers.loc["grid_electricity", "co2_emissions"] if available.
    """

    def __init__(self, n: pypsa.Network, grid_co2_intensity: float | None = None):
        self.n = n
        self._dt = n.snapshot_weightings.generators  # hours per snapshot

        # Grid CO₂ intensity
        if grid_co2_intensity is not None:
            self._grid_co2 = grid_co2_intensity
        elif "grid_electricity" in n.carriers.index:
            self._grid_co2 = float(n.carriers.loc["grid_electricity", "co2_emissions"])
        else:
            logger.warning("grid_co2_intensity not set and no grid_electricity carrier found; using 0")
            self._grid_co2 = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> DCMetrics:
        """Run all metric calculations and return a DCMetrics instance."""
        it_mw = self._it_load_mw()
        facility_mw = self._total_facility_mw()
        cooling_elec_mw = self._cooling_elec_mw()
        water_l_per_h = self._water_flow_l_per_h()
        scope1_co2_t_per_h = self._scope1_co2_t_per_h()
        scope2_co2_t_per_h = self._scope2_co2_t_per_h()
        renewable_mw = self._renewable_mw()

        # ── PUE ──────────────────────────────────────────────────────────
        # Avoid division by zero at snapshots with zero IT load
        pue_t = (facility_mw / it_mw.replace(0, np.nan)).fillna(1.0)

        # ── WUE ──────────────────────────────────────────────────────────
        # Water (L/h) / IT energy (MWh/h) = L/MWh_it
        wue_t = (water_l_per_h / it_mw.replace(0, np.nan)).fillna(0.0)

        # ── CER ──────────────────────────────────────────────────────────
        # Heat removed = IT load (all IT electricity becomes heat)
        # Cooling electricity = sum of cooling link draws on facility bus
        cer_t = (it_mw / cooling_elec_mw.replace(0, np.nan)).fillna(np.nan)

        # ── CUE ──────────────────────────────────────────────────────────
        total_co2_t_per_h = scope1_co2_t_per_h + scope2_co2_t_per_h
        cue_t = (total_co2_t_per_h / it_mw.replace(0, np.nan)).fillna(0.0)

        # ── REF ──────────────────────────────────────────────────────────
        ref_t = (renewable_mw / facility_mw.replace(0, np.nan)).clip(0, 1).fillna(0.0)

        # ── Annual totals ─────────────────────────────────────────────────
        dt = self._dt
        total_it        = float((it_mw * dt).sum())
        total_facility  = float((facility_mw * dt).sum())
        total_cool_elec = float((cooling_elec_mw * dt).sum())
        total_water     = float((water_l_per_h * dt).sum())
        total_s1_co2    = float((scope1_co2_t_per_h * dt).sum())
        total_s2_co2    = float((scope2_co2_t_per_h * dt).sum())
        total_renewable = float((renewable_mw * dt).sum())

        # Load-weighted annual averages
        weights = it_mw * dt
        pue_annual = float((pue_t * weights).sum() / weights.sum()) if weights.sum() > 0 else float("nan")
        wue_annual = total_water / total_it if total_it > 0 else float("nan")
        cer_annual = total_it / total_cool_elec if total_cool_elec > 0 else float("nan")
        cue_annual = (total_s1_co2 + total_s2_co2) / total_it if total_it > 0 else float("nan")
        ref_annual = total_renewable / total_facility if total_facility > 0 else float("nan")

        return DCMetrics(
            snapshots=self.n.snapshots,
            pue_t=pue_t, wue_t=wue_t, cer_t=cer_t, cue_t=cue_t, ref_t=ref_t,
            pue=pue_annual, wue=wue_annual, cer=cer_annual, ere=0.0,
            cue=cue_annual, ref=ref_annual,
            total_it_MWh=total_it,
            total_facility_MWh=total_facility,
            total_cooling_elec_MWh=total_cool_elec,
            total_water_L=total_water,
            total_scope1_co2_t=total_s1_co2,
            total_scope2_co2_t=total_s2_co2,
            total_renewable_MWh=total_renewable,
        )

    # ------------------------------------------------------------------
    # Component extractors
    # ------------------------------------------------------------------

    def _it_load_mw(self) -> pd.Series:
        """
        IT electrical load (MW) — the p1 output of the compute_to_facility Link.
        This is the electricity consumed by servers only, excluding cooling.
        """
        n = self.n
        if "compute_to_facility" in n.links.index:
            # p1 = flow into facility bus = IT load
            return n.links_t.p1["compute_to_facility"].abs()

        # Fallback: sum all compute-carrier loads
        compute_loads = [
            l for l in n.loads.index
            if n.loads.loc[l, "carrier"] == "compute"
        ]
        if compute_loads:
            p = n.loads_t.p_set[compute_loads].sum(axis=1)
            # Scale by elec_MW_per_request if stored in meta
            scale = n.meta.get("elec_MW_per_request", 1.0)
            return p * scale

        logger.warning("Cannot identify IT load — returning zeros")
        return pd.Series(0.0, index=n.snapshots)

    def _total_facility_mw(self) -> pd.Series:
        """
        Total electrical power drawn by the facility (MW).
        = sum of all generation dispatched to the facility bus.
        """
        n = self.n
        total = pd.Series(0.0, index=n.snapshots)

        # Generators on facility bus
        fac_gens = n.generators[n.generators.bus == "facility"].index
        if len(fac_gens) > 0 and len(n.generators_t.p) > 0:
            cols = [g for g in fac_gens if g in n.generators_t.p.columns]
            if cols:
                total += n.generators_t.p[cols].sum(axis=1)

        # Links that feed into facility bus (p1 > 0 = generation)
        # e.g. battery_discharge, CCGT/OCGT (their p1 is output)
        for link in n.links.index:
            if n.links.loc[link, "bus1"] == "facility":
                if link in n.links_t.p1.columns:
                    flow = n.links_t.p1[link]
                    total += flow.clip(lower=0)  # only positive flows (generation)

        return total.clip(lower=0)

    def _cooling_elec_mw(self) -> pd.Series:
        """
        Electrical power consumed by cooling systems (MW).
        Cooling Links have bus1=facility with negative efficiency,
        so their p1 values are negative (they draw from the bus).
        """
        n = self.n
        total = pd.Series(0.0, index=n.snapshots)

        cooling_links = [l for l in n.links.index if l.startswith("cooling:")]
        for link in cooling_links:
            if link in n.links_t.p1.columns:
                # p1 is negative (draw from facility) — take abs for consumption
                total += n.links_t.p1[link].abs()

        return total

    def _water_flow_l_per_h(self) -> pd.Series:
        """
        Water consumed by the facility (L/h).
        Sourced from the water_supply generator dispatch on water_bus.
        """
        n = self.n
        if "water_supply" in n.generators.index and "water_supply" in n.generators_t.p.columns:
            return n.generators_t.p["water_supply"].clip(lower=0)
        return pd.Series(0.0, index=n.snapshots)

    def _scope1_co2_t_per_h(self) -> pd.Series:
        """
        Scope 1 CO₂ emissions (t/h) from on-site gas combustion.
        Read from the rate of change of the scope1_co2_store state of charge.
        """
        n = self.n
        if "scope1_co2_store" in n.stores.index and "scope1_co2_store" in n.stores_t.e.columns:
            e = n.stores_t.e["scope1_co2_store"]
            # Differentiate: CO₂ rate = Δe / Δt
            # e is cumulative; diff gives per-snapshot increment
            rate = e.diff().fillna(e.iloc[0] if len(e) > 0 else 0.0)
            return rate.clip(lower=0) / self._dt

        # Fallback: compute from gas generator dispatches directly
        total = pd.Series(0.0, index=n.snapshots)
        for link in ["OCGT", "CCGT"]:
            if link in n.links.index and link in n.links_t.p0.columns:
                p_gas = n.links_t.p0[link].abs()   # MWh_th/h
                eff3 = n.links.loc[link, "efficiency3"] if "efficiency3" in n.links.columns else 0.202
                total += p_gas * eff3
        return total

    def _scope2_co2_t_per_h(self) -> pd.Series:
        """
        Scope 2 CO₂ emissions (t/h) from grid electricity imports.
        """
        n = self.n
        if "grid_import" in n.generators.index and "grid_import" in n.generators_t.p.columns:
            p_grid = n.generators_t.p["grid_import"].clip(lower=0)
            return p_grid * self._grid_co2
        return pd.Series(0.0, index=n.snapshots)

    def _renewable_mw(self) -> pd.Series:
        """
        Renewable electricity generation dispatched to the facility bus (MW).
        Includes rooftop solar and all PPA generators.
        """
        n = self.n
        total = pd.Series(0.0, index=n.snapshots)

        renewable_carriers = {c for c in n.generators.carrier.unique()
                              if "solar" in c or "wind" in c or c.startswith("ppa_")}

        for gen in n.generators.index:
            carrier = n.generators.loc[gen, "carrier"]
            if carrier in renewable_carriers and gen in n.generators_t.p.columns:
                total += n.generators_t.p[gen].clip(lower=0)

        return total
