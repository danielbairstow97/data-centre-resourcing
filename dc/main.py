"""
Data Centre Network — Hydra entry point
=======================================

Run examples
------------

# Default cooling scenario (cooling tower):
python main.py

# Swap cooling scenario:
python main.py cooling=dry_air_cooled
python main.py cooling=liquid_immersion
python main.py cooling=hybrid_tower_economiser
python main.py cooling=airside_economiser

# Override a parameter inline:
python main.py cooling=dry_air_cooled \\
    cooling.technologies.0.elec_MW_per_MW_heat=0.25

# Multi-run sweep over all cooling scenarios:
python main.py --multirun \\
    cooling=cooling_tower,dry_air_cooled,liquid_immersion,adiabatic,airside_economiser,hybrid_tower_economiser,hybrid_economiser_dry_chiller

# Combine scenario sweep with water constraint:
python main.py --multirun \\
    cooling=cooling_tower,dry_air_cooled \\
    environmental.max_water_L_per_year=5e9,null

# Disable gas generation and sweep cooling:
python main.py --multirun \\
    cooling=cooling_tower,dry_air_cooled \\
    generation.ocgt.enabled=false \\
    generation.ccgt.enabled=false
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from dc.config import CONFIG_DIR

log = logging.getLogger(__name__)


@hydra.main(config_path=str(CONFIG_DIR), config_name="brisbane_facility", version_base=None)
def main(cfg: DictConfig) -> None:
    # Import here so Hydra's logging config is applied first
    from dc.network.builder import Builder
    from dc.network.models import DataCentreConfig

    dc_cfg = DataCentreConfig.from_hydra(cfg)

    log.info("=== Data Centre Network Builder ===")
    log.info("Facility  : %s", dc_cfg.name)
    log.info("Cooling   : %s", " | ".join([opt.name for opt in dc_cfg.cooling]))

    network = Builder(dc_cfg).build()

    log.info(
        "Network   : %d buses | %d generators | %d links | %d stores",
        len(network.buses),
        len(network.generators),
        len(network.links),
        len(network.stores),
    )


if __name__ == "__main__":
    main()
