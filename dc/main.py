from genericpath import exists
import logging
from pathlib import Path

from dc.network.solver import Solver

import hydra
from omegaconf import DictConfig
from dc.config import CONFIG_DIR, OUTPUT_DIR
import pandas as pd
import pypsa

log = logging.getLogger(__name__)


def extract_results(network: pypsa.Network, cfg: DictConfig) -> dict:
    """Pull scalar results from an optimised network."""
    n = network

    # --- Objective ---
    total_cost = n.objective  # annualised £/$/yr from PyPSA

    # --- Optimal capacities ---
    # Generators (rooftop solar, PPA)
    gen_caps = (
        n.generators[["p_nom_opt"]].rename(columns={"p_nom_opt": "capacity_mw"}).add_prefix("gen_")
        if not n.generators.empty
        else pd.DataFrame()
    )
    # Links (grid connection, CCGT)
    link_caps = (
        n.links[["p_nom_opt"]].rename(columns={"p_nom_opt": "capacity_mw"}).add_prefix("link_")
        if not n.links.empty
        else pd.DataFrame()
    )
    # Stores (battery, load shift)
    store_caps = (
        n.stores[["e_nom_opt"]].rename(columns={"e_nom_opt": "capacity_mwh"}).add_prefix("store_")
        if not n.stores.empty
        else pd.DataFrame()
    )

    caps = {}
    for df in (gen_caps, link_caps, store_caps):
        for col in df.columns:
            # flatten: "gen_ppa:solar_farm_capacity_mw" → manageable key
            key = col.replace(":", "_").replace(" ", "_").lower()
            caps[key] = float(df[col].iloc[0]) if len(df) else None

    # --- Dispatch summary (annual totals, MWh) ---
    dt = n.snapshot_weightings.generators  # hours per snapshot

    dispatch = {}
    if not n.generators_t.p.empty:
        for col in n.generators_t.p.columns:
            key = ("dispatch_mwh_gen_" + col).replace(":", "_").replace(" ", "_").lower()
            dispatch[key] = float((n.generators_t.p[col] * dt).sum())

    if not n.links_t.p0.empty:
        for col in n.links_t.p0.columns:
            key = ("dispatch_mwh_link_" + col).replace(":", "_").replace(" ", "_").lower()
            dispatch[key] = float((n.links_t.p0[col] * dt).sum())

    # --- Sweep identity (from Hydra cfg) ---
    identity = {
        "scenario_name": cfg.name,
        "grid_capacity": int(cfg.grid_connection.max_capacity_MW),
        "load_shift_fraction": float(cfg.compute.load_shift.max_store_energy_factor),
        "pue": float(cfg.compute.power_use_effectiveness),
        "wue": float(cfg.compute.water_use_effectiveness),
    }

    return {**identity, "total_cost_aud_pa": total_cost, **caps, **dispatch}


@hydra.main(config_path=str(CONFIG_DIR), config_name="base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    from dc.network.builder import Builder
    from dc.network.models import DataCentreConfig

    dc_cfg = DataCentreConfig.from_hydra(cfg)

    log.info(
        f"=== Run: grid_capacity={cfg.grid_connection.max_capacity_MW} MW  shift={cfg.compute.load_shift.max_store_energy_factor*100:.1f}% ===",
    )

    # --- Build and optimise ---
    network = Builder(dc_cfg).build()
    solver = Solver(network, dc_cfg.solver)
    res = solver.solve()

    log.info(
        "Solved | obj=%.2f | buses=%d generators=%d links=%d stores=%d",
        network.objective,
        len(network.buses),
        len(network.generators),
        len(network.links),
        len(network.stores),
    )

    # --- Save NetCDF (one per run, path is Hydra's run dir) ---
    out_dir = OUTPUT_DIR / f"sweeps/tier={dc_cfg.grid_connection.max_capacity_MW}_shift={dc_cfg.compute.load_shift.max_store_energy_factor}"
    out_dir.mkdir(parents=True, exist_ok=True)
    nc_path = out_dir / "network.nc"
    network.export_to_netcdf(str(nc_path))
    log.info("Network saved → %s", nc_path)

    # --- Save per-run results row as JSON (aggregated later by summarise.py) ---
    results = extract_results(network, cfg)
    results_path = out_dir / "results.json"

    import json

    results_path.write_text(json.dumps(results, indent=2))
    log.info("Results saved → %s", results_path)


if __name__ == "__main__":
    main()
