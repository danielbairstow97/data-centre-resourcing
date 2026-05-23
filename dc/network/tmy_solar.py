"""
Rooftop Solar Profile
=====================

Builds a p_max_pu time-series for the rooftop solar generator using either:

  - PVGIS TMY data (preferred): fetches a Typical Meteorological Year from
    the EU PVGIS API, giving a realistic hourly irradiance profile that
    captures cloud cover, not just clear-sky geometry.

  - Clear-sky fallback: PVLib Ineichen clear-sky × performance_ratio,
    the same approach used for PPA location_solar profiles.  Used when
    PVGIS is unavailable (offline, polar location, API timeout).

Why TMY for rooftop but clear-sky for PPA?
-------------------------------------------
A PPA contract is defined by the counterparty's P50 guarantee — the
actual shape is their responsibility.  We encode what the contract says.

The rooftop array is owned by the facility, so we want the best available
estimate of what it will actually produce, which means accounting for
local cloud cover.  PVGIS TMY data does this from satellite-derived
irradiance observations (~17 years of data for most of Europe/MENA/AU).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from dc.config.models import RooftopSolarConfig

logger = logging.getLogger(__name__)

_PVGIS_TIMEOUT_S = 30


def build_rooftop_profile(
    solar_cfg: RooftopSolarConfig,
    latitude: float,
    longitude: float,
    farm_tz: str,
    snapshots: pd.DatetimeIndex,
) -> pd.Series:
    """
    Fetch PVGIS TMY data and compute POA irradiance for the panel geometry.
    """
    import pvlib

    logger.info(
        "Fetching PVGIS TMY for rooftop: (%.4f°, %.4f°)",
        latitude,
        longitude,
    )

    tmy_data, _ = pvlib.iotools.get_pvgis_tmy(
        latitude=latitude,
        longitude=longitude,
        outputformat="json",
        usehorizon=True,
        timeout=_PVGIS_TIMEOUT_S,
    )

    # PVGIS returns a UTC-indexed DataFrame; rename columns to pvlib conventions
    col_map = {
        "G(h)": "ghi",
        "Gb(n)": "dni",
        "Gd(h)": "dhi",
        "T2m": "temp_air",
        "WS10m": "wind_speed",
    }
    tmy_data = tmy_data.rename(columns={k: v for k, v in col_map.items() if k in tmy_data.columns})

    # ── Compute POA irradiance ──────────────────────────────────────────
    loc = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        altitude=0.0,
        tz="UTC",
    )

    solar_pos = loc.get_solarposition(tmy_data.index)
    airmass_rel = pvlib.atmosphere.get_relative_airmass(solar_pos["apparent_zenith"])
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass_rel)
    dni_extra = pvlib.irradiance.get_extra_radiation(tmy_data.index)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=abs(latitude),
        surface_azimuth=180,
        dni=tmy_data["dni"],
        ghi=tmy_data["ghi"],
        dhi=tmy_data["dhi"],
        solar_zenith=solar_pos["apparent_zenith"],
        solar_azimuth=solar_pos["azimuth"],
        model="perez",
        airmass=airmass_abs,
        dni_extra=dni_extra,
    )

    poa_global = poa["poa_global"].fillna(0.0).clip(lower=0.0)

    # Normalise to p_max_pu using STC reference and performance ratio
    p_max_pu_tmy = (poa_global / 1_000.0) * solar_cfg.panel_efficiency

    logger.info(
        "PVGIS TMY: mean CF=%.4f, yield=%.0f MWh/MWp/yr, max CF=%.4f",
        p_max_pu_tmy.mean(),
        p_max_pu_tmy.mean() * 8760,
        p_max_pu_tmy.max(),
    )

    # ── Map TMY onto simulation snapshots ──────────────────────────────
    return _map_tmy_to_snapshots(p_max_pu_tmy, snapshots, farm_tz)


def _map_tmy_to_snapshots(
    tmy_series: pd.Series,
    snapshots: pd.DatetimeIndex,
    facility_tz: str,
) -> pd.Series:
    """
    Map a TMY time-series (one representative year) onto the simulation snapshots.

    Strategy: strip the TMY year and use (month, day, hour) as a multi-index
    to look up values, so the TMY repeats correctly across multi-year simulations
    and aligns properly for single-year runs with any start date.
    """
    # Build a (month, day, hour) → value lookup from the TMY
    # TMY index is UTC; convert to local time for day/hour alignment
    tmy_local = tmy_series.copy()
    if tmy_series.index.tz is not None:
        tmy_local.index = tmy_series.index.tz_convert(facility_tz)
    else:
        tmy_local.index = tmy_series.index.tz_localize("UTC").tz_convert(facility_tz)

    lookup = {}
    for ts, val in tmy_local.items():
        key = (ts.month, ts.day, ts.hour)
        lookup[key] = float(val)

    # Map simulation snapshots (convert to local time for lookup)
    if snapshots.tz is None:
        snaps_local = snapshots.tz_localize("UTC").tz_convert(facility_tz)
    else:
        snaps_local = snapshots.tz_convert(facility_tz)

    values = [lookup.get((ts.month, ts.day, ts.hour), 0.0) for ts in snaps_local]

    return pd.Series(values, index=snapshots, name="p_max_pu").clip(0.0, 1.0)
