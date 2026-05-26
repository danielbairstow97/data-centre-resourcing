"""
Data Centre Metrics
===================

Computes industry-standard data centre efficiency metrics from a solved
PyPSA network, plus operational statistics for load shifting.

Metrics reported
----------------
PUE   Power Usage Effectiveness    total facility MW / IT MW
WUE   Water Usage Effectiveness    total water (L) / IT energy (MWh)
        - WUE_it   : water from compute_to_facility link only
        - WUE_ccgt : water from CCGT combustion
        - WUE_total: combined
CUE   Carbon Usage Effectiveness   scope1 CO₂e (t) / IT energy (MWh)
REF   Renewable Energy Factor      renewable MWh / total facility MWh

Load shift statistics
---------------------
shift_utilisation   mean |Δe| as fraction of store capacity
shift_mwh_deferred  total MWh deferred across year
shift_mwh_advanced  total MWh advanced across year
shift_peak_reduction peak facility MW reduction vs no-shift baseline
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import ClassVar

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class LoadShiftStats:
    enabled: bool = False
    store_capacity_mwh: float = float("nan")
    shift_mwh_deferred: float = 0.0    # MWh where store charged (demand pushed forward)
    shift_mwh_advanced: float = 0.0    # MWh where store discharged (demand pulled back)
    net_shift_mwh: float = 0.0         # should be ~0 with e_cyclic
    shift_utilisation: float = 0.0     # mean |delta_soc| / e_nom
    peak_facility_mw: float = float("nan")
    soc_t: pd.Series = field(default_factory=pd.Series)

    def summary(self) -> str:
        if not self.enabled:
            return "  Load shifting        : disabled"
        lines = [
            f"  Store capacity       : {self.store_capacity_mwh:>10,.1f} MWh",
            f"  Deferred             : {self.shift_mwh_deferred:>10,.1f} MWh/yr",
            f"  Advanced             : {self.shift_mwh_advanced:>10,.1f} MWh/yr",
            f"  Net shift            : {self.net_shift_mwh:>10,.1f} MWh/yr  (≈0 if cyclic)",
            f"  Utilisation          : {self.shift_utilisation:>10.1%}  of store capacity",
            f"  Peak facility draw   : {self.peak_facility_mw:>10,.1f} MW",
        ]
        return "\n".join(lines)


@dataclass
class GenerationMix:
    """Annual energy by source, all in MWh."""
    grid_mwh: float = 0.0
    ccgt_mwh: float = 0.0
    solar_ppa_mwh: float = 0.0
    wind_ppa_mwh: float = 0.0
    rooftop_solar_mwh: float = 0.0
    battery_net_mwh: float = 0.0       # net discharge (positive = net supply)

    # Capacities (MW or MWh for battery)
    grid_capacity_mw: float = 0.0
    ccgt_capacity_mw: float = 0.0
    solar_ppa_capacity_mw: float = 0.0
    wind_ppa_capacity_mw: float = 0.0
    battery_capacity_mwh: float = 0.0

    def summary(self) -> str:
        total = (self.grid_mwh + self.ccgt_mwh + self.solar_ppa_mwh
                 + self.wind_ppa_mwh + self.rooftop_solar_mwh)
        def pct(v):
            return f"{v/total:.1%}" if total > 0 else "n/a"
        lines = [
            f"  Grid import          : {self.grid_mwh:>12,.1f} MWh  {pct(self.grid_mwh)}"
            f"  [{self.grid_capacity_mw:.1f} MW]",
            f"  CCGT                 : {self.ccgt_mwh:>12,.1f} MWh  {pct(self.ccgt_mwh)}"
            f"  [{self.ccgt_capacity_mw:.1f} MW]",
            f"  Solar PPA            : {self.solar_ppa_mwh:>12,.1f} MWh  {pct(self.solar_ppa_mwh)}"
            f"  [{self.solar_ppa_capacity_mw:.1f} MW]",
            f"  Wind PPA             : {self.wind_ppa_mwh:>12,.1f} MWh  {pct(self.wind_ppa_mwh)}"
            f"  [{self.wind_ppa_capacity_mw:.1f} MW]",
            f"  Rooftop solar        : {self.rooftop_solar_mwh:>12,.1f} MWh  {pct(self.rooftop_solar_mwh)}",
            f"  Battery (net)        : {self.battery_net_mwh:>12,.1f} MWh"
            f"  [{self.battery_capacity_mwh:.1f} MWh]",
        ]
        return "\n".join(lines)


@dataclass
class DCMetrics:
    snapshots: pd.DatetimeIndex

    # ── Time series ───────────────────────────────────────────────────
    pue_t: pd.Series = field(default_factory=pd.Series)
    wue_total_t: pd.Series = field(default_factory=pd.Series)
    cue_t: pd.Series = field(default_factory=pd.Series)
    ref_t: pd.Series = field(default_factory=pd.Series)
    facility_mw_t: pd.Series = field(default_factory=pd.Series)

    # ── Annual scalars ────────────────────────────────────────────────
    pue: float = float("nan")
    wue_it: float = float("nan")
    wue_ccgt: float = float("nan")
    wue_total: float = float("nan")
    cue: float = float("nan")
    ref: float = float("nan")

    # ── Annual totals ─────────────────────────────────────────────────
    total_it_mwh: float = float("nan")
    total_facility_mwh: float = float("nan")
    total_water_it_l: float = float("nan")
    total_water_ccgt_l: float = float("nan")
    total_water_l: float = float("nan")
    total_scope1_co2_t: float = float("nan")
    total_renewable_mwh: float = float("nan")

    # ── Sub-reports ───────────────────────────────────────────────────
    generation: GenerationMix = field(default_factory=GenerationMix)
    load_shift: LoadShiftStats = field(default_factory=LoadShiftStats)

    # ── Comparison index ─────────────────────────────────────────────
    # Human-readable labels aligned to to_series() key order
    _METRIC_LABELS: ClassVar[dict[str, str]] = {
        # Efficiency metrics
        "pue":                    "PUE",
        "wue_it":                 "WUE — IT cooling (kL/MWh)",
        "wue_ccgt":               "WUE — CCGT water (kL/MWh)",
        "wue_total":              "WUE — total (kL/MWh)",
        "cue":                    "CUE (tCO₂/MWh_it)",
        "ref":                    "REF — renewable fraction",
        # Annual totals
        "total_it_mwh":           "IT energy (MWh/yr)",
        "total_facility_mwh":     "Facility energy (MWh/yr)",
        "total_water_l":          "Water — total (kL/yr)",
        "total_water_it_l":       "Water — IT cooling (kL/yr)",
        "total_water_ccgt_l":     "Water — CCGT (kL/yr)",
        "total_scope1_co2_t":     "Scope 1 CO₂ (t/yr)",
        "total_renewable_mwh":    "Renewable energy (MWh/yr)",
        # Generation mix — energy
        "grid_mwh":               "Grid import (MWh/yr)",
        "ccgt_mwh":               "CCGT generation (MWh/yr)",
        "solar_ppa_mwh":          "Solar PPA (MWh/yr)",
        "wind_ppa_mwh":           "Wind PPA (MWh/yr)",
        "rooftop_solar_mwh":      "Rooftop solar (MWh/yr)",
        "battery_net_mwh":        "Battery net dispatch (MWh/yr)",
        # Generation mix — capacity
        "grid_capacity_mw":       "Grid connection (MW)",
        "ccgt_capacity_mw":       "CCGT capacity (MW)",
        "solar_ppa_capacity_mw":  "Solar PPA capacity (MW)",
        "wind_ppa_capacity_mw":   "Wind PPA capacity (MW)",
        "battery_capacity_mwh":   "Battery capacity (MWh)",
        # Load shifting
        "shift_enabled":          "Load shifting enabled",
        "shift_store_mwh":        "Shift store capacity (MWh)",
        "shift_deferred_mwh":     "Demand deferred (MWh/yr)",
        "shift_advanced_mwh":     "Demand advanced (MWh/yr)",
        "shift_utilisation":      "Shift utilisation",
        "peak_facility_mw":       "Peak facility draw (MW)",
    }

    def to_series(self) -> pd.Series:
        """
        All aggregate metrics as a labelled pd.Series, suitable for
        pd.concat()-ing across scenarios to produce a comparison table.

        Usage
        -----
        results = {
            "Tier 1 / no shift": metrics_t1.to_series(),
            "Tier 2 / 10% shift": metrics_t2.to_series(),
        }
        comparison = pd.DataFrame(results)   # metrics as rows, scenarios as columns
        """
        data = self.to_dict()
        # Add generation sub-fields not already in to_dict
        data["rooftop_solar_mwh"] = self.generation.rooftop_solar_mwh
        data["battery_net_mwh"] = self.generation.battery_net_mwh

        return pd.Series(
            {self._METRIC_LABELS[k]: v for k, v in data.items() if k in self._METRIC_LABELS}
        )

    def to_dict(self) -> dict:
        """Flat dict for appending to the sweep results CSV."""
        return {
            "pue": self.pue,
            "wue_it": self.wue_it,
            "wue_ccgt": self.wue_ccgt,
            "wue_total": self.wue_total,
            "cue": self.cue,
            "ref": self.ref,
            "total_it_mwh": self.total_it_mwh,
            "total_facility_mwh": self.total_facility_mwh,
            "total_water_l": self.total_water_l,
            "total_water_it_l": self.total_water_it_l,
            "total_water_ccgt_l": self.total_water_ccgt_l,
            "total_scope1_co2_t": self.total_scope1_co2_t,
            "total_renewable_mwh": self.total_renewable_mwh,
            "grid_mwh": self.generation.grid_mwh,
            "ccgt_mwh": self.generation.ccgt_mwh,
            "solar_ppa_mwh": self.generation.solar_ppa_mwh,
            "wind_ppa_mwh": self.generation.wind_ppa_mwh,
            "rooftop_solar_mwh": self.generation.rooftop_solar_mwh,
            "battery_net_mwh": self.generation.battery_net_mwh,
            "grid_capacity_mw": self.generation.grid_capacity_mw,
            "ccgt_capacity_mw": self.generation.ccgt_capacity_mw,
            "solar_ppa_capacity_mw": self.generation.solar_ppa_capacity_mw,
            "wind_ppa_capacity_mw": self.generation.wind_ppa_capacity_mw,
            "battery_capacity_mwh": self.generation.battery_capacity_mwh,
            "shift_enabled": self.load_shift.enabled,
            "shift_store_mwh": self.load_shift.store_capacity_mwh,
            "shift_deferred_mwh": self.load_shift.shift_mwh_deferred,
            "shift_advanced_mwh": self.load_shift.shift_mwh_advanced,
            "shift_utilisation": self.load_shift.shift_utilisation,
            "peak_facility_mw": self.load_shift.peak_facility_mw,
        }

    def summary(self) -> str:
        lines = [
            "══ Data Centre Metrics ══════════════════════════════════════════",
            f"  PUE   Power Usage Effectiveness  : {self.pue:.4f}",
            f"  WUE   (IT cooling)               : {self.wue_it:.4f} kL/MWh_it",
            f"  WUE   (CCGT water)               : {self.wue_ccgt:.4f} kL/MWh_it",
            f"  WUE   (total)                    : {self.wue_total:.4f} kL/MWh_it",
            f"  CUE   Carbon Usage Effectiveness : {self.cue:.4f} tCO₂/MWh_it",
            f"  REF   Renewable Energy Factor    : {self.ref:.1%}",
            "── Totals ────────────────────────────────────────────────────────",
            f"  IT load             : {self.total_it_mwh:>12,.1f} MWh/yr",
            f"  Total facility      : {self.total_facility_mwh:>12,.1f} MWh/yr",
            f"  Water (IT)          : {self.total_water_it_l:>12,.0f} kL/yr",
            f"  Water (CCGT)        : {self.total_water_ccgt_l:>12,.0f} kL/yr",
            f"  Water (total)       : {self.total_water_l:>12,.0f} kL/yr",
            f"  Scope 1 CO₂         : {self.total_scope1_co2_t:>12,.1f} t/yr",
            f"  Renewable energy    : {self.total_renewable_mwh:>12,.1f} MWh/yr",
            "── Generation mix ───────────────────────────────────────────────",
            self.generation.summary(),
            "── Load shifting ─────────────────────────────────────────────────",
            self.load_shift.summary(),
            "═════════════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class MetricsCalculator:
    """
    Computes DCMetrics from a solved PyPSA network.

    Parameters
    ----------
    n : pypsa.Network
        Solved and optimised network.
    grid_co2_intensity : float, optional
        tCO₂/MWh for grid imports. Used for scope 2 if needed later.
    """

    # Link and component name constants — update if builder names change
    LINK_FACILITY_TO_COMPUTE = "facility_to_compute"
    LINK_CCGT = "CCGT"
    LINK_GRID = "grid_connection"
    GEN_WATER_SUPPLY = "water_supply"
    GEN_NEM = "nem"
    STORE_CO2 = "co2_store"
    STORE_BATTERY = "facility_battery"
    STORE_SHIFT = "compute_shift"

    def __init__(self, n: pypsa.Network, grid_co2_intensity: float = 0.0):
        self.n = n
        self._dt = n.snapshot_weightings.generators

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> DCMetrics:
        n = self.n
        dt = self._dt

        it_mw = self._it_load_mw()
        facility_mw = self._facility_draw_mw()
        water_it = self._water_it_l_per_h()
        water_ccgt = self._water_ccgt_l_per_h()
        water_total = water_it + water_ccgt
        scope1_co2 = self._scope1_co2_t_per_h()
        renewable_mw = self._renewable_mw()

        safe_it = it_mw.replace(0, np.nan)
        safe_fac = facility_mw.replace(0, np.nan)

        # ── Time series metrics ───────────────────────────────────────
        pue_t = (facility_mw / safe_it).fillna(1.0)
        wue_total_t = (water_total / safe_it).fillna(0.0)
        cue_t = (scope1_co2 / safe_it).fillna(0.0)
        ref_t = (renewable_mw / safe_fac).clip(0, 1).fillna(0.0)

        # ── Annual totals ─────────────────────────────────────────────
        total_it = float((it_mw * dt).sum())
        total_fac = float((facility_mw * dt).sum())
        total_water_it = float((water_it * dt).sum())
        total_water_ccgt = float((water_ccgt * dt).sum())
        total_water = total_water_it + total_water_ccgt
        total_co2 = float((scope1_co2 * dt).sum())
        total_renewable = float((renewable_mw * dt).sum())

        # ── Annual scalar metrics ─────────────────────────────────────
        weights = it_mw * dt
        weight_sum = float(weights.sum())

        pue = float((pue_t * weights).sum() / weight_sum) if weight_sum > 0 else float("nan")
        wue_it = total_water_it / total_it if total_it > 0 else float("nan")
        wue_ccgt = total_water_ccgt / total_it if total_it > 0 else float("nan")
        wue_total = total_water / total_it if total_it > 0 else float("nan")
        cue = total_co2 / total_it if total_it > 0 else float("nan")
        ref = total_renewable / total_fac if total_fac > 0 else float("nan")

        return DCMetrics(
            snapshots=n.snapshots,
            pue_t=pue_t,
            wue_total_t=wue_total_t,
            cue_t=cue_t,
            ref_t=ref_t,
            facility_mw_t=facility_mw,
            pue=pue,
            wue_it=wue_it,
            wue_ccgt=wue_ccgt,
            wue_total=wue_total,
            cue=cue,
            ref=ref,
            total_it_mwh=total_it,
            total_facility_mwh=total_fac,
            total_water_it_l=total_water_it,
            total_water_ccgt_l=total_water_ccgt,
            total_water_l=total_water,
            total_scope1_co2_t=total_co2,
            total_renewable_mwh=total_renewable,
            generation=self._generation_mix(),
            load_shift=self._load_shift_stats(facility_mw),
        )

    # ------------------------------------------------------------------
    # IT and facility load
    # ------------------------------------------------------------------

    def _it_load_mw(self) -> pd.Series:
        """
        IT electrical load (MW).

        facility_to_compute draws p0 from the facility bus.
        IT load = p0 × (1/PUE) is the compute delivered, but for PUE
        calculation we want the gross facility draw attributable to IT,
        which is p0 itself — the link efficiency already encodes PUE.

        We return p0 / PUE to get MW_it (server-only consumption),
        consistent with PUE = facility_draw / it_load.
        """
        n = self.n
        link = self.LINK_FACILITY_TO_COMPUTE
        if link not in n.links_t.p0.columns:
            logger.warning("facility_to_compute not found in links_t.p0")
            return pd.Series(0.0, index=n.snapshots)

        p0 = n.links_t.p0[link].abs()  # gross facility draw for IT+overhead

        # efficiency = 1/PUE stored on the link
        efficiency = float(n.links.loc[link, "efficiency"])
        # p0 is facility MW, efficiency × p0 = compute units delivered
        # IT MW = p0 × efficiency (since efficiency = compute_MW / facility_MW = 1/PUE)
        # Therefore PUE = p0 / (p0 × efficiency) = 1/efficiency ✓
        it_mw = p0 * efficiency
        return it_mw

    def _facility_draw_mw(self) -> pd.Series:
        """
        Gross electrical draw of the facility (MW).
        = p0 of facility_to_compute (what the facility bus must supply for IT)
        + any other loads on the facility bus (none currently modelled).
        """
        n = self.n
        link = self.LINK_FACILITY_TO_COMPUTE
        if link not in n.links_t.p0.columns:
            return pd.Series(0.0, index=n.snapshots)
        return n.links_t.p0[link].abs()

    # ------------------------------------------------------------------
    # Water
    # ------------------------------------------------------------------

    def _water_it_l_per_h(self) -> pd.Series:
        """
        Water consumed by IT cooling (kL/h).
        Sourced from the p2 port of facility_to_compute (WUE × IT load).
        p2 is negative in PyPSA convention for consumption — take abs.
        """
        n = self.n
        link = self.LINK_FACILITY_TO_COMPUTE
        if "p2" in n.links_t and link in n.links_t["p2"].columns:
            return n.links_t["p2"][link].abs()

        # Fallback: reconstruct from IT load × WUE efficiency
        if link in n.links_t.p0.columns:
            eff2 = abs(float(n.links.loc[link, "efficiency2"]))
            it_mw = self._it_load_mw()
            return it_mw * eff2
        return pd.Series(0.0, index=n.snapshots)

    def _water_ccgt_l_per_h(self) -> pd.Series:
        """
        Water consumed by CCGT cooling (kL/h).
        CCGT p2 port draws from water bus with negative efficiency2.
        """
        n = self.n
        link = self.LINK_CCGT
        if link not in n.links.index:
            return pd.Series(0.0, index=n.snapshots)

        if "p2" in n.links_t and link in n.links_t["p2"].columns:
            return n.links_t["p2"][link].abs()

        # Fallback: reconstruct from gas throughput
        if link in n.links_t.p0.columns:
            eff2 = abs(float(n.links.loc[link, "efficiency2"]))
            p0 = n.links_t.p0[link].abs()
            return p0 * eff2
        return pd.Series(0.0, index=n.snapshots)

    # ------------------------------------------------------------------
    # Emissions
    # ------------------------------------------------------------------

    def _scope1_co2_t_per_h(self) -> pd.Series:
        """
        Scope 1 CO₂ rate (t/h) from CCGT combustion.
        Primary: differentiate co2_store state of charge.
        Fallback: efficiency3 × p0[CCGT].
        """
        n = self.n

        if (self.STORE_CO2 in n.stores.index
                and self.STORE_CO2 in n.stores_t.e.columns):
            e = n.stores_t.e[self.STORE_CO2]
            # Forward difference; first snapshot uses its own value
            rate = e.diff().fillna(e.iloc[0] if len(e) > 0 else 0.0)
            return rate.clip(lower=0) / self._dt

        # Fallback
        if (self.LINK_CCGT in n.links.index
                and self.LINK_CCGT in n.links_t.p0.columns):
            p0 = n.links_t.p0[self.LINK_CCGT].abs()
            eff3 = float(n.links.loc[self.LINK_CCGT, "efficiency3"]) \
                if "efficiency3" in n.links.columns else 0.0
            return p0 * eff3
        return pd.Series(0.0, index=n.snapshots)

    # ------------------------------------------------------------------
    # Renewables
    # ------------------------------------------------------------------

    def _renewable_mw(self) -> pd.Series:
        """
        Renewable generation dispatched (MW).
        Includes all PPA generators and rooftop solar.
        NEM/grid import is excluded regardless of mix.
        """
        n = self.n
        total = pd.Series(0.0, index=n.snapshots)

        renewable_carriers = {
            c for c in n.generators.carrier.unique()
            if any(tag in c for tag in ("solar", "wind", "ppa_"))
        }

        # Exclude the NEM slack generator even if it somehow gets a solar carrier
        exclude = {self.GEN_NEM, "nem"}

        for gen in n.generators.index:
            if gen in exclude:
                continue
            if (n.generators.loc[gen, "carrier"] in renewable_carriers
                    and gen in n.generators_t.p.columns):
                total += n.generators_t.p[gen].clip(lower=0)

        return total

    # ------------------------------------------------------------------
    # Generation mix
    # ------------------------------------------------------------------

    def _generation_mix(self) -> GenerationMix:
        n = self.n
        dt = self._dt

        def link_p1_mwh(name: str) -> float:
            if name in n.links_t.p1.columns:
                return float((n.links_t.p1[name].abs() * dt).sum())
            return 0.0

        def gen_mwh(name: str) -> float:
            if name in n.generators_t.p.columns:
                return float((n.generators_t.p[name].abs() * dt).sum())
            return 0.0

        def opt_cap(component_df, name: str, col: str = "p_nom_opt") -> float:
            if name in component_df.index and col in component_df.columns:
                return float(component_df.loc[name, col])
            return 0.0

        # Grid: p1 of grid_connection link (delivered to facility bus)
        grid_mwh = link_p1_mwh(self.LINK_GRID)

        # CCGT: p1 of CCGT link
        ccgt_mwh = link_p1_mwh(self.LINK_CCGT)

        # PPAs: generators prefixed with ppa:
        solar_ppa_mwh = sum(
            gen_mwh(g) for g in n.generators.index
            if g.startswith("ppa:solar_farm") or "ppa_solar" in n.generators.loc[g, "carrier"]
        )
        wind_ppa_mwh = sum(
            gen_mwh(g) for g in n.generators.index
            if g.startswith("ppa:wind_farm") or "ppa_wind" in n.generators.loc[g, "carrier"]
        )
        rooftop_mwh = gen_mwh("rooftop_solar")

        # Battery: net discharge = total dispatch - total store
        bat_dispatch = 0.0
        bat_store = 0.0
        if self.STORE_BATTERY in n.storage_units.index:
            su = n.storage_units_t
            if self.STORE_BATTERY in su.p.columns:
                p = su.p[self.STORE_BATTERY]
                bat_dispatch = float((p.clip(lower=0) * dt).sum())
                bat_store = float((p.clip(upper=0).abs() * dt).sum())

        return GenerationMix(
            grid_mwh=grid_mwh,
            ccgt_mwh=ccgt_mwh,
            solar_ppa_mwh=solar_ppa_mwh,
            wind_ppa_mwh=wind_ppa_mwh,
            rooftop_solar_mwh=rooftop_mwh,
            battery_net_mwh=bat_dispatch - bat_store,
            grid_capacity_mw=opt_cap(n.links, self.LINK_GRID),
            ccgt_capacity_mw=opt_cap(n.links, self.LINK_CCGT),
            solar_ppa_capacity_mw=sum(
                opt_cap(n.generators, g) for g in n.generators.index
                if g.startswith("ppa:solar")
            ),
            wind_ppa_capacity_mw=sum(
                opt_cap(n.generators, g) for g in n.generators.index
                if g.startswith("ppa:wind")
            ),
            battery_capacity_mwh=opt_cap(n.storage_units, self.STORE_BATTERY, "p_nom_opt"),
        )

    # ------------------------------------------------------------------
    # Load shifting
    # ------------------------------------------------------------------

    def _load_shift_stats(self, facility_mw: pd.Series) -> LoadShiftStats:
        n = self.n

        if self.STORE_SHIFT not in n.stores.index:
            return LoadShiftStats(
                enabled=False,
                peak_facility_mw=float(facility_mw.max()),
            )

        e_nom = float(n.stores.loc[self.STORE_SHIFT, "e_nom"])
        dt = self._dt

        if self.STORE_SHIFT not in n.stores_t.e.columns:
            return LoadShiftStats(
                enabled=True,
                store_capacity_mwh=e_nom,
                peak_facility_mw=float(facility_mw.max()),
            )

        soc = n.stores_t.e[self.STORE_SHIFT]
        delta = soc.diff().fillna(0.0)

        # Charging the store = deferring compute demand
        deferred = float((delta.clip(lower=0) * dt).sum())
        # Discharging = advancing demand
        advanced = float((delta.clip(upper=0).abs() * dt).sum())
        net = deferred - advanced

        # Utilisation: mean absolute SOC change as fraction of capacity
        utilisation = float(delta.abs().mean() / e_nom) if e_nom > 0 else 0.0

        return LoadShiftStats(
            enabled=True,
            store_capacity_mwh=e_nom,
            shift_mwh_deferred=deferred,
            shift_mwh_advanced=advanced,
            net_shift_mwh=net,
            shift_utilisation=utilisation,
            peak_facility_mw=float(facility_mw.max()),
            soc_t=soc,
        )
