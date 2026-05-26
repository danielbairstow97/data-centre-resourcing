"""Carriers used across the data centre network."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class CarrierMeta:
    name: str
    co2_tonnes_per_mwh: float = 0.0
    nice_name: str = ""
    color: str = "#888888"


class Carriers(str, Enum):
    """Canonical carrier names used as PyPSA carrier strings."""

    ELECTRICITY = "electricity"
    GRID = "grid"
    GAS = "gas"
    CCGT = "ccgt"
    WATER = "water"
    COMPUTE = "compute"
    COMPUTE_SHIFT = "compute_shift"
    CARBON = "co2"
    BESS = "bess"
    SOLAR = "solar"
    WIND = "wind"

    def meta(self) -> CarrierMeta:
        return _CARRIER_META[self.value]


_CARRIER_META: dict[str, CarrierMeta] = {
    "grid": CarrierMeta(name="grid", co2_tonnes_per_mwh=0.0, nice_name="Grid Electricity"),
    "solar": CarrierMeta(name="solar", nice_name="Solar PV", color="#e2f10f"),
    "ccgt": CarrierMeta(name="ccgt", nice_name="Closed Cycle Gas Turbine", color="#a30a4a"),
    "wind": CarrierMeta(name="wind", nice_name="Wind", color="#9a0ff1"),
    "bess": CarrierMeta(name="bess", nice_name="Battery Energy Storage System", color="#0ff1de"),
    "compute_shift": CarrierMeta(name="compute_shift", nice_name="Flexible Compute", color="#00218b"),
    "electricity": CarrierMeta(
        name="electricity",
        co2_tonnes_per_mwh=0.0,
        nice_name="Electricity",
        color="#f10f1a",
    ),
    "gas": CarrierMeta(
        name="gas",
        co2_tonnes_per_mwh=0.0,
        nice_name="Natural Gas",
        color="#e67e22",
    ),
    "water": CarrierMeta(
        name="water",
        co2_tonnes_per_mwh=0.0,
        nice_name="Water",
        color="#3498db",
    ),
    "compute": CarrierMeta(
        name="compute",
        co2_tonnes_per_mwh=0.0,
        nice_name="Compute",
        color="#9b59b6",
    ),
    "co2": CarrierMeta(
        name="co2",
        co2_tonnes_per_mwh=0.0,
        nice_name="CO2",
        color="#2c3e50",
    ),
}
