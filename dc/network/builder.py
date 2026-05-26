"""
Data Centre Network Builder
===========================

Constructs a PyPSA network representing the data centre facility.

Network topology
----------------

  [compute_bus] ──(compute_to_facility)──► [facility bus (electricity)]  ← IT load only
                                       ──► [cooling_bus]                 ← heat load = IT_MW × fraction

  [cooling_bus] is balanced by one Link per enabled cooling technology:

    [cooling_bus] ──(cooling:<tech> Link)──► [facility bus]   bus1: elec draw (positive)
                                         ──► [water_bus]      bus2: water use  (negative)

  [load_shift store]  (optional, on compute_bus)

  [facility bus] supply side:
        ├── CCGT Link ◄── [gas supply bus]
        ├── Rooftop Solar Generator
        ├── Battery Store (charge/discharge Links)
        └── Grid import
            ├── Solar Farm PPA
            ├── Wind Farm PPA
            └── Grid wholesale market

  [water_bus]
        └── water_supply Generator (marginal_cost = cost_per_L)


Carrier conventions
-------------------
- Electricity: MW
- Gas:         MWh(th)  — heat content so efficiency = MWh_e / MWh_th = 1/heat_rate
- Water:       Litres   — treated as a flow carrier; costs in $/L
- Compute:     requests — dimensionless; scaled to electrical load via Link efficiency
- Cooling:     MW_th    — thermal heat load to be removed by cooling technologies
"""

from __future__ import annotations

from enum import StrEnum, auto
import logging
from typing import Optional

import pandas as pd
import pypsa

from dc.network.carriers import _CARRIER_META, Carriers
from dc.network.models import DataCentreConfig, PPAConfig

logger = logging.getLogger(__name__)


# Constants
GAS_GJ_TO_MWth = 1 / 3.6
GAS_t_CO2_per_GJ = 0.05153
GAS_t_CO2_per_MWth = GAS_t_CO2_per_GJ * GAS_GJ_TO_MWth


# Bus names
class BusName(StrEnum):
    FACILITY = auto()
    COMPUTE = auto()
    COOLING = auto()
    FACILITY_EMISSIONS = auto()
    BATTERY_BUS = auto()

    GRID = auto()
    WATER_SUPPLY = auto()
    GAS_SUPPLY = auto()


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class Builder:
    """
    Builds a PyPSA network from a :class:`DataCentreConfig`.

    Usage::

        cfg  = DataCentreConfig.from_yaml("config.yaml")
        net  = Builder(cfg).build()
    """

    def __init__(self, cfg: DataCentreConfig):
        self.cfg = cfg
        self.discount_rate = cfg.financial.real_discount_rate
        self.project_lifetime = cfg.financial.project_lifetime
        self.n = pypsa.Network()

        self._snapshots: Optional[pd.DatetimeIndex] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def build(self) -> pypsa.Network:
        """Construct and return the fully populated PyPSA network."""

        self._setup_snapshots()

        # Add carriers
        self._add_carriers()

        # Core infrastructure (order matters — buses before links)
        self._setup_facility_infrastructure()

        # Compute load + optional shifting
        self._add_compute()

        # Generation & storage
        self._add_onsite_generation()
        for contract in self.cfg.generation.ppa:
            self._add_ppa_contract(contract)

        logger.info(
            "Network built: %d buses, %d generators, %d links, %d stores",
            len(self.n.buses),
            len(self.n.generators),
            len(self.n.links),
            len(self.n.stores),
        )
        return self.n

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------
    def _setup_snapshots(self) -> None:
        cfg = self.cfg.simulation
        idx = pd.date_range(
            start=cfg.snapshots_start,
            end=cfg.snapshots_end,
            freq=cfg.snapshot_freq,
            inclusive="left",
        )
        self.n.set_snapshots(idx)
        self._snapshots = idx
        logger.debug("Snapshots: %d periods (%s → %s)", len(idx), idx[0], idx[-1])

    # ------------------------------------------------------------------
    # Carriers
    # ------------------------------------------------------------------
    def _add_carriers(self) -> None:
        for carrier in Carriers:
            meta = _CARRIER_META[carrier.value]
            self.n.add(
                "Carrier",
                carrier.value,
                nice_name=meta.nice_name,
                color=meta.color,
                co2_emissions=meta.co2_tonnes_per_mwh,
            )

    def _setup_facility_infrastructure(self):
        # Setup electricity supply
        self.n.add("Bus", BusName.FACILITY.value, carrier=Carriers.ELECTRICITY.value)

        # Add CO2 emissions
        self.n.add("Bus", BusName.FACILITY_EMISSIONS.value, carrier=Carriers.CARBON.value)
        self.n.add(
            "Store",
            "co2_store",
            bus=BusName.FACILITY_EMISSIONS.value,
            carrier=Carriers.CARBON.value,
            e_nom=1e9,
            e_nom_extendable=False,
            marginal_cost=0.0,  # swap for a carbon price to penalise in objective
            e_cyclic=False,  # emissions accumulate, don't need to return to zero
        )

        # Add grid connection
        gccfg = self.cfg.grid_connection
        self.n.add("Bus", BusName.GRID.value, carrier=Carriers.GRID.value)
        self.n.add(
            "Generator",
            "nem",
            bus=BusName.GRID.value,
            p_nom=1e3,
            control="Slack",
            marginal_cost=gccfg.cost_per_MWh,
        )
        self.n.add(
            "Link",
            name="grid_connection",
            bus0=BusName.GRID.value,
            bus1=BusName.FACILITY.value,
            efficiency=gccfg.transmission_loss_factor,
            p_nom_extendable=True,
            p_min_pu=0.0,
            p_nom_max=gccfg.max_capacity_MW,
            discount_rate=self.discount_rate,
            lifetime=self.project_lifetime,
            overnight_cost=gccfg.capex_per_MW,
            fom_cost=gccfg.capex_per_MW * 0.025,
        )

        # Add water supply
        wcfg = self.cfg.supplies.water
        self.n.add("Bus", BusName.WATER_SUPPLY.value, carrier=Carriers.WATER.value)
        self.n.add(
            "Generator",
            "water_supply",
            bus=BusName.WATER_SUPPLY.value,
            carrier=Carriers.WATER.value,
            marginal_cost=wcfg.cost_per_L,
            p_nom=0.0,
            p_nom_max=1e3,  # effectively unlimited capacity
            p_nom_extendable=True,
            control="Slack",
        )

        # Add gas supply
        gcfg = self.cfg.supplies.gas
        cost_per_MWh_th = gcfg.cost_per_GJ * GAS_GJ_TO_MWth
        self.n.add("Bus", BusName.GAS_SUPPLY.value, carrier=Carriers.GAS.value)
        self.n.add(
            "Generator",
            "gas_supply",
            bus=BusName.GAS_SUPPLY.value,
            carrier=Carriers.GAS.value,
            marginal_cost=cost_per_MWh_th,
            p_nom=0.0,
            p_nom_max=1e3,
            p_nom_extendable=True,
        )

    # ------------------------------------------------------------------
    # Compute bus, load, and optional load shifting
    # ------------------------------------------------------------------
    def _add_compute(self) -> None:
        """
        Compute layer.

        Creates:
          - compute_bus      : carrier=compute, holds the raw request load
          - cooling_bus      : carrier=cooling (MW_th), receives IT heat to be removed
          - compute_to_facility Link:
              bus0 = compute_bus  (requests consumed)
              bus1 = facility     (IT electricity drawn,  efficiency  = elec_MW_per_request)
              bus2 = cooling_bus  (heat injected,         efficiency2 = elec_MW_per_request
                                   because 1 MW_it ≡ 1 MW_th rejected)
          - compute_shift Store (optional, on compute_bus)

        Cooling technologies (added in _add_cooling) then draw from cooling_bus:
          bus0 = cooling_bus → bus1 = facility (electricity overhead)
                             → bus2 = water_bus (water consumption, negative)
        """
        ccfg = self.cfg.compute

        # ── Buses ──────────────────────────────────────────────────────────
        self.n.add("Bus", BusName.COMPUTE.value, carrier=Carriers.COMPUTE.value)

        # ── Demand on compute bus ──────────────────────────────────────────
        load_profile = ccfg.build_compute_ts(self._snapshots)
        self.n.add(
            "Load",
            "compute_load",
            bus=BusName.COMPUTE.value,
            carrier=Carriers.COMPUTE.value,
            p_set=load_profile,
        )

        # ── Optional load-shift buffer ─────────────────────────────────────
        lscfg = ccfg.load_shift
        if lscfg.enabled and lscfg.max_store_energy_factor:
            e_nom = load_profile * lscfg.max_store_energy_factor * lscfg.max_shift_hours

            self.n.add(
                "Store",
                "compute_shift",
                bus=BusName.COMPUTE.value,
                carrier=Carriers.COMPUTE_SHIFT.value,
                e_nom=e_nom,
                e_nom_extendable=False,
                e_cyclic=True,
                marginal_cost=lscfg.marginal_cost_shift,
                e_min_pu=0.0,
                e_initial=e_nom / 2,
            )
            logger.debug(
                "Load shift store: e_nom=%.1f requests, max_shift=%.1f h",
                e_nom,
                lscfg.max_shift_hours,
            )

        # ── compute_to_facility Link ───────────────────────────────────────
        self.n.add(
            "Link",
            "facility_to_compute",
            bus0=BusName.FACILITY.value,  # draws from facility — appears as load
            bus1=BusName.COMPUTE.value,  # satisfies compute bus demand
            bus2=BusName.WATER_SUPPLY.value,  # water consumed per IT-MW
            efficiency=1.0 / ccfg.power_use_effectiveness,  # compute MW per facility MW
            efficiency2=-ccfg.water_use_effectiveness / ccfg.power_use_effectiveness,
            p_nom=load_profile * 2,
            p_nom_extendable=False,
        )

    def _add_onsite_generation(self):
        gen = self.cfg.generation.onsite
        if gen.ccgt.enabled:
            self._add_ccgt()
        if gen.battery.enabled:
            self._add_battery()

    # ------------------------------------------------------------------
    # Gas generation
    # ------------------------------------------------------------------
    def _add_ccgt(self) -> None:
        ccfg = self.cfg.generation.onsite.ccgt
        water_kL_per_MWh_th = ccfg.water_kL_per_MWh
        self.n.add(
            "Link",
            "CCGT",
            bus0=BusName.GAS_SUPPLY.value,
            bus1=BusName.FACILITY.value,
            bus2=BusName.WATER_SUPPLY.value,
            bus3=BusName.FACILITY_EMISSIONS.value,
            efficiency=ccfg.electrical_efficiency,
            efficiency2=-water_kL_per_MWh_th,
            efficiency3=GAS_t_CO2_per_MWth,
            p_nom_extendable=True,
            p_nom=0.0,
            overnight_cost=ccfg.capex_per_MW,
            fom_cost=ccfg.opex_per_MW,
            lifetime=ccfg.lifetime_years,
            discount_rate=self.discount_rate,
            carrier="ccgt",
        )

    # ------------------------------------------------------------------
    # Renewables
    # ------------------------------------------------------------------

    def _add_ppa_contract(self, contract: PPAConfig) -> None:
        """
        Add a PPA contract as a Generator (and optional must-take Load).

        Pay-as-produced (take_or_pay=False)
        ------------------------------------
        A single Generator with:
          - marginal_cost = strike_price_per_MWh
          - p_max_pu      = contract generation profile (built from profile terms)
          - p_nom         = contracted capacity (optimised if p_nom_extendable)

        The optimiser dispatches up to the available profile at the PPA price.
        Unused generation is curtailed at zero additional cost — the facility
        only pays for what it consumes.

        Take-or-pay (take_or_pay=True)
        --------------------------------
        Same Generator, but the strike price is paid on ALL scheduled MWh.
        This is implemented by setting marginal_cost=0 on the Generator
        (dispatch is free once committed) and adding:

          1. A committed payment Load = p_nom × p_max_pu (the must-take schedule).
             This Load has a fixed p_set equal to the full available profile,
             so the facility pays strike_price × profile_MWh regardless of use.

          2. A spill Generator (marginal_cost=0, large p_nom) that absorbs
             surplus PPA generation that cannot be used or stored, preventing
             infeasibility without artificially benefiting the objective.

        The committed payment cost is recovered by setting the Load's
        carrier to a dedicated 'ppa_commitment' carrier — this is a
        modelling artefact to track the cost separately from dispatch.

        Note: for the screening phase, pay-as-produced is recommended.
        Take-or-pay adds complexity that is only worth modelling once a
        candidate contract structure has been identified.
        """
        gen_name = f"ppa:{contract.name}"
        carrier = f"ppa_{contract.technology.value}"

        # Register a carrier for this PPA technology type (idempotent)
        if carrier not in self.n.carriers.index:
            self.n.add(
                "Carrier",
                carrier,
                nice_name=f"PPA {contract.technology.title()}",
                color="#27ae60" if contract.technology == "solar" else "#2980b9",
                co2_emissions=0.0,
            )

        # ── Build p_max_pu profile from contract terms ─────────────────────
        profiler = contract.profile.build_profile()
        p_max_pu = profiler.profile(self._snapshots)

        # ── Pay-as-produced ────────────────────────────────────────────
        self.n.add(
            "Generator",
            gen_name,
            bus="grid",
            carrier=carrier,
            marginal_cost=contract.marginal_cost_per_MWh,
            p_nom_extendable=True,
            p_nom=0.0,
            overnight_cost=contract.contract_capacity_fee_per_MW,
            discount_rate=self.discount_rate,
            lifetime=self.project_lifetime,
        )
        self.n.generators_t.p_max_pu[gen_name] = p_max_pu

    # ------------------------------------------------------------------
    # Battery storage
    # ------------------------------------------------------------------
    def _add_battery(self) -> None:
        """
        Battery modelled as a StorageUnit
        """
        bcfg = self.cfg.generation.onsite.battery

        capex_per_MW = bcfg.capex_per_MWh / bcfg.max_hours
        opex_per_MW = bcfg.opex_per_MWh / bcfg.max_hours
        self.n.add(
            "StorageUnit",
            "facility_battery",
            bus=BusName.FACILITY.value,
            carrier=Carriers.BESS.value,
            p_nom=0.0,
            p_nom_extendable=True,
            cyclic_state_of_charge=True,
            state_of_charge_initial=0.5,
            marginal_cost=0.0,
            efficiency_store=bcfg.efficiency_charge,
            efficiency_dispatch=bcfg.efficiency_discharge,
            p_max_pu=1.0,
            p_min_pu=-1.0,
            max_hours=bcfg.max_hours,
            overnight_cost=capex_per_MW,
            fom_cost=opex_per_MW,
            lifetime=bcfg.lifetime_years,
            discount_rate=self.discount_rate,
        )
