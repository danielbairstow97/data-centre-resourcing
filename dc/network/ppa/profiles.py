from dataclasses import dataclass, field
import logging
from typing import Protocol

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

logger = logging.getLogger(__name__)

HOURS_PER_YEAR = 8760.0


class PPAProfiler(Protocol):
    def profile(self, snapshots: pd.DatetimeIndex) -> pd.Series: ...


@dataclass
class FlatProfile:
    capacity_factor: float

    def profile(self, snapshots: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(self.capacity_factor, index=snapshots)


@dataclass
class MonthlyProfile:
    monthly_factors: np.ndarray

    def __post_init__(self):
        if len(self.monthly_factors) != 12:
            raise ValueError(
                f"monthly_factors must be of length 12 instead of {len(self.monthly_factors)}"
            )
        if self.monthly_factors.max() > 1 | self.monthly_factors.min() < 0.0:
            raise ValueError("monthly_factors must be in the range [0,1]")

    def profile(self, snapshots: pd.DatetimeIndex) -> pd.Series:
        factors = self.monthly_factors  # list[float], length 12
        month_map = {m: f for m, f in enumerate(factors, start=1)}  # 1-indexed

        values = snapshots.month.map(month_map)
        return pd.Series(values.values, index=snapshots, dtype=float)


@dataclass
class LocationSolarProfile:
    latitude: float
    longitude: float
    performance_ratio: float
    farm_tz: str

    location: Location = field(init=False)

    def __post_init__(self):
        self.location = Location(
            latitude=self.latitude,
            longitude=self.longitude,
            tz=self.farm_tz,
            name=f"PPA farm ({self.latitude:.2f}°, {self.longitude:.2f}°)",
        )

    @property
    def surface_tilt(self) -> float:
        return np.abs(self.latitude)

    @property
    def surface_azimuth(self) -> float:
        if self.latitude > 0:
            return 0.0
        else:
            return 180.0

    def profile(self, snapshots: pd.DatetimeIndex) -> pd.Series:
        if snapshots.tz is None:
            times_local = snapshots.tz_localize("UTC").tz_convert(self.farm_tz)
        else:
            times_local = snapshots.tz_convert(self.farm_tz)

        clearsky = self.location.get_clearsky(times_local)
        solar_position = self.location.get_solarposition(times_local)

        airmass_relative = pvlib.atmosphere.get_relative_airmass(solar_position["apparent_zenith"])
        airmass_absolute = pvlib.atmosphere.get_absolute_airmass(airmass_relative)
        dni_extra = pvlib.irradiance.get_extra_radiation(times_local)

        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=self.surface_tilt,
            surface_azimuth=self.surface_azimuth,
            dni=clearsky["dni"],
            ghi=clearsky["ghi"],
            dhi=clearsky["dhi"],
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
            model="perez",
            airmass=airmass_absolute,
            dni_extra=dni_extra,
        )

        poa_global = poa["poa_global"].fillna(0.0).clip(lower=0.0)

        POA_STC = 1_000.0  # W/m²

        p_max_pu = (poa_global / POA_STC) * self.performance_ratio
        p_max_pu.index = snapshots

        # ── Diagnostic: log computed annual yield for term-sheet cross-check ───
        computed_yield = float(p_max_pu.mean() * 8760)
        clear_sky_yield = float((poa_global / POA_STC).mean() * 8760)
        logger.info(
            "PVLib location_solar %s: tilt=%.1f° az=%.1f° PR=%.2f | "
            "clear-sky yield=%.0f MWh/MWp/yr → P50 yield=%.0f MWh/MWp/yr | "
            "mean CF=%.4f  max CF=%.4f  zero-hours=%d",
            self.location.name,
            self.surface_tilt,
            self.surface_azimuth,
            self.performance_ratio,
            clear_sky_yield,
            computed_yield,
            float(p_max_pu.mean()),
            float(p_max_pu.max()),
            int((p_max_pu < 1e-6).sum()),
        )

        if p_max_pu.max() > 1.0:
            over = int((p_max_pu > 1.0).sum())
            logger.warning(
                "%d snapshots exceed p_max_pu=1.0 (PR=%.2f may be too high for "
                "this location, or clearsky model is overestimating). "
                "Values will be clipped to 1.0.",
                over,
                self.performance_ratio,
            )

        return p_max_pu
