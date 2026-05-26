"""
Configuration models for the data centre network builder.

Mirrors the YAML structure and adds validation / defaults.
Load with:

    cfg = DataCentreConfig.from_yaml("config.yaml")
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from pydantic_extra_types.coordinate import Coordinate
from timezonefinder import TimezoneFinder
import yaml

from dc.network.carriers import Carriers
from dc.network.ppa.profiles import FlatProfile, LocationSolarProfile, MonthlyProfile, PPAProfiler

tzFinder = TimezoneFinder()


# ---------------------------------------------------------------------------
# Leaf-level models
# ---------------------------------------------------------------------------
class ComputeConfig(BaseModel):
    """
    Physical resource consumption of the IT (server) load only.

    Water consumption is no longer specified here — it is fully determined by
    the active cooling technologies in ``CoolingConfig``.  The heat rejected
    by the servers (= IT electrical load) drives the cooling load via each
    technology's ``pue_contribution`` parameter.
    """

    capacity_mw: float = Field(
        ge=0,
        description="MW of IT electrical draw per compute request (server power only, excluding cooling)",
    )
    power_use_effectiveness: float
    water_use_effectiveness: float
    load_realisation_factor: float
    load_shift: LoadShiftConfig

    def build_compute_ts(self, sns: pd.DatetimeIndex) -> pd.Series | float:
        return self.capacity_mw * self.load_realisation_factor


class LoadShiftConfig(BaseModel):
    """
    Temporal flexibility of the compute load.

    Models a 'buffer' Store on the compute bus that lets requests be
    absorbed earlier or later within a rolling window.
    """

    enabled: bool = False
    max_shift_hours: float = Field(
        ge=0,
        description="Maximum hours a request can be deferred or advanced",
    )
    # Maximum energy stored in the shift buffer expressed as a multiple of the
    # average-snapshot compute load (dimensionless — scaled at build time).
    max_store_energy_factor: float = Field(
        ge=0,
        description=(
            "Buffer capacity as a multiple of average-snapshot compute load. "
            "E.g. 1.0 → can absorb one snapshot-worth of requests."
        ),
    )
    marginal_cost_shift: float = Field(
        ge=0,
        description="Extra cost per unit of shifted compute (to penalise excessive shifting)",
    )


class WaterSupplyConfig(BaseModel):
    cost_per_L: float = Field(ge=0, description="$/L of mains water")


class GasSupplyConfig(BaseModel):
    cost_per_GJ: float = Field(ge=0, description="$/GJ of natural gas")


class SuppliesConfig(BaseModel):
    water: WaterSupplyConfig
    gas: GasSupplyConfig


class CCGTConfig(BaseModel):
    """Combined-Cycle Gas Turbine — baseload, higher efficiency."""

    enabled: bool = True
    capex_per_MW: float = Field(ge=0, description="$/MW overnight CAPEX")
    opex_per_MW: float = Field(ge=0, description="$/MW/year OPEX")
    water_kL_per_MWh: float = Field(
        ge=0,
        description="Litres of cooling water per MWh of electrical output (wet-cooled)",
    )
    electrical_efficiency: float = Field(
        ge=0, le=1.0, description="MWe produced per MW thermal input"
    )
    lifetime_years: float
    block_size_MW: float = Field(default=400.0, gt=0)
    p_nom_extendable: bool = True


class RooftopSolarConfig(BaseModel):
    enabled: bool = True
    capex_per_MW: float = Field(ge=0, description="$/MW")
    opex_per_MW: float = Field(ge=0, description="$/MW/year fixed O&M")
    ground_cover_ratio: float
    panel_efficiency: float
    lifetime_years: float
    p_nom_extendable: bool = True

    @property
    def m2_footprint_per_MW(self):
        return 1e6 / (self.panel_efficiency * 1e3 * self.ground_cover_ratio)


# ---------------------------------------------------------------------------
# PPA (Power Purchase Agreement) models
# ---------------------------------------------------------------------------
class PPAProfileFlat(BaseModel):
    """
    Simplest profile: the generator delivers a fixed fraction of contracted
    capacity every hour.  Appropriate when no seasonal or diurnal shaping
    data is available from the counterparty.

    capacity_factor : float
        Fraction of p_nom available every hour (0–1).
        e.g. 0.35 → wind farm delivers 35% of contracted MW continuously.
    """

    type: str = Field(default="flat", description="Profile type discriminator")
    capacity_factor: float = Field(
        gt=0,
        le=1.0,
        description="Flat capacity factor applied to every snapshot",
    )

    def build_profile(self) -> FlatProfile:
        return FlatProfile(self.capacity_factor)


class PPAProfileMonthlyFactors(BaseModel):
    """
    Twelve monthly capacity factors defining seasonal variation.
    Each factor is the fraction of contracted capacity available during that
    month (average — no intra-month variation).  The builder interpolates
    linearly to hourly resolution.

    Appropriate for wind PPAs where the counterparty provides a monthly
    P50 generation schedule.
    """

    type: str = Field(default="monthly_factors", description="Profile type discriminator")
    # Jan–Dec capacity factors, must all be in (0, 1]
    monthly_factors: list[float] = Field(
        min_length=12,
        max_length=12,
        description="Capacity factor for each calendar month (Jan=0 … Dec=11)",
    )

    @model_validator(mode="after")
    def check_factors(self) -> PPAProfileMonthlyFactors:
        bad = [f for f in self.monthly_factors if not (0 < f <= 1.0)]
        if bad:
            raise ValueError(f"All monthly_factors must be in (0, 1]; got: {bad}")
        return self

    def build_profile(self):
        return MonthlyProfile(np.array(self.monthly_factors))


class PPAProfileLocationSolar(BaseModel):
    """
    Solar PPA profile derived from the farm's physical location using PVLib.

    The profile is computed entirely from location and array geometry — no
    external weather files or user-specified yield are required.
    """

    type: str = Field(default="location_solar", description="Profile type discriminator")

    location: Coordinate

    # ── Array geometry ─────────────────────────────────────────────────────
    orientation_from_location: bool = Field(
        default=True,
        description=(
            "If true, surface_tilt is set to |latitude| automatically. "
            "This is the rule-of-thumb optimum for annual energy yield on a "
            "fixed-tilt array.  Set false to specify surface_tilt manually."
        ),
    )

    # ── Performance ratio ──────────────────────────────────────────────────
    performance_ratio: float = Field(
        gt=0,
        le=1.0,
        description=(
            "System performance ratio (PR) — the ratio of actual annual energy "
            "output to the ideal output under clear-sky conditions. "
            "Captures cloud cover, temperature losses, soiling, inverter losses, "
            "and availability. Sourced from the counterparty's P50 energy report. "
            "Typical values: UK/Germany 0.78–0.84 | Spain 0.72–0.80 | "
            "Australia/MENA 0.68–0.76."
        ),
    )

    @property
    def farm_tz(self) -> str:
        tz = tzFinder.timezone_at(lng=self.location.longitude, lat=self.location.latitude)
        if tz is None:
            raise ValueError("Farm location timezone not found")
        return tz

    def build_profile(self) -> PPAProfiler:
        return LocationSolarProfile(
            latitude=self.location.latitude,
            longitude=self.location.longitude,
            performance_ratio=self.performance_ratio,
            farm_tz=self.farm_tz,
        )


# Union type for profile config — Hydra selects via 'type' field
PPAProfile = PPAProfileFlat | PPAProfileMonthlyFactors | PPAProfileLocationSolar


class PPAConfig(BaseModel):
    """
    Power Purchase Agreement for grid-connected renewable generation.

    Models the *contract* rather than the physical plant.  The data centre
    does not own the generator — it purchases output at a fixed strike price
    according to a pre-agreed generation profile.

    PyPSA representation
    --------------------
    Pay-as-produced (default):
        A single Generator with marginal_cost = strike_price_per_MWh and
        p_max_pu = contract profile.  The optimiser dispatches up to the
        available profile at the PPA price; any unused generation is curtailed
        at zero cost.

    Take-or-pay:
        Same Generator, plus a must-take Load equal to p_nom × p_max_pu.
        A spill Generator (marginal_cost = 0, p_nom = large) allows surplus
        to be absorbed without grid export.  The strike price is still paid
        on all scheduled MWh even if curtailed.

    Sizing via optimisation
    -----------------------
    p_nom_extendable=True lets the optimiser choose the contracted MW.
    The capital_cost parameter encodes any upfront contract fee or capacity
    charge ($/MW of contracted capacity), annualised by the builder.
    For pure pay-as-produced contracts with no capacity charge, set
    contract_capacity_fee_per_MW=0 and rely solely on strike_price_per_MWh.
    """

    name: str
    technology: Carriers
    enabled: bool = True

    # ── Contract financial terms ───────────────────────────────────────────
    lcoe: float = Field(
        gt=0,
        description="Fixed price paid per MWh of contracted generation ($/MWh)",
    )
    contract_margin: float
    contract_capacity_fee_per_MW: float = Field(
        default=1.0,
        ge=0,
        description=("Upfront or annual capacity charge per contracted MW ($/MW). "),
    )

    # ── Generation profile ─────────────────────────────────────────────────
    profile: PPAProfile = Field(
        description="Contract generation profile — defines p_max_pu time-series",
    )

    @property
    def marginal_cost_per_MWh(self) -> float:
        return self.lcoe * (1 + self.contract_margin)


class BatteryConfig(BaseModel):
    """
    Utility-scale battery storage for short-duration energy shifting.
    Modelled as a PyPSA Store + two Links (charge / discharge).
    """

    enabled: bool = False
    capex_per_MWh: float = Field(ge=0, description="$/MWh storage CAPEX")
    opex_per_MWh: float = Field(ge=0, description="$/MW inverter CAPEX")
    efficiency_charge: float = Field(gt=0, le=1.0)
    efficiency_discharge: float = Field(gt=0, le=1.0)
    lifetime_years: float
    max_hours: float = Field(gt=0, description="E/P ratio (hours at full power)")
    e_nom_extendable: bool = True


class GridConnectionConfig(BaseModel):
    """
    Connection to the external grid — allows import/export.
    Grid import has a carbon intensity and a price time-series (or flat rate).
    """

    capex_per_MW: float
    cost_per_MWh: float
    max_capacity_MW: float
    transmission_loss_factor: float


class OnsiteGenerationConfig(BaseModel):
    ccgt: CCGTConfig
    battery: BatteryConfig


class GenerationConfig(BaseModel):
    onsite: OnsiteGenerationConfig = Field(
        description="Onsite generation options",
    )
    ppa: list[PPAConfig] = Field(
        description="Portfolio of PPA contracts (replaces grid_solar / grid_wind)",
    )


# ---------------------------------------------------------------------------
# Solver / optimisation settings
# ---------------------------------------------------------------------------
class SolverConfig(BaseModel):
    name: str = Field(description="Solver name passed to PyPSA (highs, glpk, gurobi…)")
    options: dict = Field(default_factory=dict, description="Solver-specific keyword options")


class PerformanceTargetConfig(BaseModel):
    power_usage_effectiveness: float
    renewable_energy_factor: float
    water_usage_effectiveness: float


class SimulationConfig(BaseModel):
    snapshots_start: date = Field(description="ISO date string")
    snapshots_end: date = Field(description="ISO date string (inclusive)")
    snapshot_freq: str = Field(default="h", description="Pandas frequency string (e.g. 'h', '3h')")
    location: Coordinate

    @property
    def facility_tz(self) -> str | None:
        tz = tzFinder.timezone_at(lng=self.location.longitude, lat=self.location.latitude)
        if tz is None:
            raise ValueError("No timezone found for location")
        return tz


class FinancialConfig(BaseModel):
    inflation_rate: float
    wacc: float
    project_lifetime: float

    @property
    def real_discount_rate(self) -> float:
        return (1 + self.wacc) / (1 + self.inflation_rate) - 1


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
class DataCentreConfig(BaseModel):
    """Root configuration for the data centre network model."""

    name: str = Field(default="DataCentre", description="Facility identifier")

    # Simulation horizon
    simulation: SimulationConfig
    financial: FinancialConfig
    compute: ComputeConfig
    supplies: SuppliesConfig
    grid_connection: GridConnectionConfig

    generation: GenerationConfig
    solver: SolverConfig

    # ---------------------------------------------------------------------------
    # Convenience constructors
    # ---------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataCentreConfig:
        """Load and validate config from a plain YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        # The YAML root key is optional 'builder:' wrapper — unwrap if present
        data = raw.get("builder", raw)
        return cls.model_validate(data)

    @classmethod
    def from_hydra(cls, cfg) -> DataCentreConfig:
        """
        Construct from a Hydra ``DictConfig`` object (the ``cfg`` argument
        received by a ``@hydra.main``-decorated function).

        Hydra composes the root config with the selected config group overrides
        (e.g. ``cooling=dry_air_cooled``) before this is called, so the
        ``cfg.cooling`` subtree already reflects the chosen scenario.

        Usage::

            @hydra.main(config_path="../conf", config_name="config", version_base=None)
            def main(cfg: DictConfig) -> None:
                dc_cfg = DataCentreConfig.from_hydra(cfg)
                network = Builder(dc_cfg).build()
        """
        from omegaconf import OmegaConf

        raw = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        """Serialise config back to YAML (useful for saving resolved defaults)."""
        import yaml as _yaml

        with open(path, "w") as f:
            _yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
