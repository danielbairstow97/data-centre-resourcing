"""
Post-sweep aggregator.

Usage:
    python summarise.py outputs/sweeps/<sweep_dir>

Walks the sweep output directory, collects every results.json,
and writes a single aggregated CSV alongside a summary parquet.
"""

import json
from pathlib import Path
from dc.network.metrics import MetricsCalculator
import pypsa
import pandas as pd

from dc.config import OUTPUT_DIR


def collect_results(sweep_dir: Path) -> pd.DataFrame:
    records = []
    for results_file in sorted(sweep_dir.rglob("results.json")):
        try:
            records.append(json.loads(results_file.read_text()))
        except Exception as exc:
            print(f"[WARN] skipping {results_file}: {exc}")

    if not records:
        raise RuntimeError(f"No results.json found under {sweep_dir}")

    df = pd.DataFrame(records)

    # Sort by the primary sweep axes for readability
    sort_cols = [c for c in ("tier", "load_shift_fraction") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df

def collect_metrics(sweep_dir: Path):
    all_metrics = []
    for run_dir in [p for p in sweep_dir.glob("tier=*_shift=*") if p.is_dir()]:
        record = json.loads((run_dir / "results.json").read_text())
        n = pypsa.Network()
        n.import_from_netcdf(run_dir / "network.nc")
        srs = MetricsCalculator(n).compute().to_series()
        all_metrics.append(pd.concat([
            pd.Series([record["grid_capacity"], record["load_shift_fraction"]], ["grid_capacity", "load_shift_fraction"]),
            srs
        ]

        ))
    comparison = pd.DataFrame(all_metrics)

    # # Slice to just efficiency metrics for a report table
    # efficiency_rows = ["PUE", "WUE — total (kL/MWh)", "CUE (tCO₂/MWh_it)", "REF — renewable fraction"]
    # comparison.loc[efficiency_rows]

    return comparison




def main():
    sweep_dir = OUTPUT_DIR / "sweeps"

    if not sweep_dir.exists():
        raise FileNotFoundError(sweep_dir)

    df = collect_results(sweep_dir)
    metrics = collect_metrics(sweep_dir)

    out_csv = OUTPUT_DIR / "results_aggregated.csv"
    df.to_csv(out_csv, index=False)

    metrics.to_csv(OUTPUT_DIR / "metrics.csv")

    print(f"Aggregated {len(df)} runs → {out_csv}")


if __name__ == "__main__":
    main()
