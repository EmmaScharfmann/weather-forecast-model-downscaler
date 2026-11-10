"""Measure how far the training target itself sits from the station observations.

Usage::

    python scripts/check_target_floor.py --config config_spatial.yaml --model runs/crps/model.pt \
        --start 2020-07-01 --end 2020-08-31

Four fields are read at the same stations over the same period: the coarse
analysis, the downscaled field, the target reanalysis, and the observations.
Because the model is trained to reproduce the target, the target's own distance
from the observations is the floor the model cannot pass. The gap between the
model and that floor is what better training could still recover; the floor
itself can only be lowered by changing the target.

The target normally ends years before the forecast period, so this is run on
the validation period rather than on the forecasts, and the floor is carried
across on the assumption that it changes slowly.

Outputs in the model's directory: ``floor.csv``, ``floor.png`` and ``floor.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from downscale.config import load_config
from downscale.constants import elevation_edges
from downscale.downscaler import Downscaler
from downscale.diagnostics import floor_figure, score_against_stations
from downscale.station_interp import curvilinear_for
from downscale.stations import MeteoFranceHourly, QCConfig, load_stations

logger = logging.getLogger("floor")
FIELDS = ["coarse", "downscaled", "target"]
LABELS = {
    "coarse": "coarse analysis",
    "downscaled": "coarse analysis + downscaler",
    "target": "target reanalysis (the floor)",
}


def load_period(pairs_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> xr.Dataset:
    """Open the training pairs restricted to a period.

    :param pairs_dir: Directory written by the pairing stage.
    :type pairs_dir: pathlib.Path
    :param start: First time to keep.
    :type start: pandas.Timestamp
    :param end: Last time to keep.
    :type end: pandas.Timestamp
    :returns: Dataset with ``coarse_t2m`` and ``fine_t2m`` over the period.
    :rtype: xarray.Dataset
    :raises ValueError: If the period contains no time.
    """
    store = pairs_dir / "pairs.zarr"
    ds = xr.open_zarr(store) if store.exists() else xr.open_mfdataset(
        [str(p) for p in sorted((pairs_dir / "months").glob("pairs_*.zarr"))],
        engine="zarr", combine="nested", concat_dim="time",
    )
    ds = ds.sel(time=slice(start, end))
    if ds.sizes["time"] == 0:
        raise ValueError(f"the training pairs hold no time between {start} and {end}")
    return ds.load()


def main(argv: Sequence[str] | None = None, provider=None) -> int:
    """Command-line entry point.

    :param argv: Argument list; ``None`` reads ``sys.argv``.
    :type argv: collections.abc.Sequence[str] | None
    :param provider: Station provider; ``None`` builds the configured one.
    :type provider: downscale.stations.StationProvider | None
    :returns: Exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pairs", default=None, help="overrides paths.pairs")
    parser.add_argument("--cache", default=None, help="overrides paths.stations_cache")
    parser.add_argument("--out", default=None, help="defaults to the model's directory")
    parser.add_argument("--start", required=True, help="first time, inside the target's period")
    parser.add_argument("--end", required=True)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("wetterdienst").setLevel(logging.WARNING)

    cfg = load_config(args.config)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    out = Path(args.out) if args.out else Path(args.model).parent
    out.mkdir(parents=True, exist_ok=True)

    pairs = load_period(Path(args.pairs or cfg.paths.pairs), start, end)
    times = pd.DatetimeIndex(pairs.time.values)
    model = Downscaler.load(args.model, args.device)
    predicted = model.predict(
        pairs.coarse_t2m.values.astype("f4"), times.to_numpy(), members=args.members
    ).mean(1)
    logger.info("%d time(s) between %s and %s", len(times), times.min(), times.max())

    stations, report = load_stations(
        provider or MeteoFranceHourly(), cfg.region_box, times.min(), times.max(),
        variables=cfg.stations.variables, cache_dir=args.cache or cfg.paths.stations_cache,
        qc=QCConfig(min_completeness=cfg.stations.min_completeness, primary=cfg.stations.variables[0]),
        holdout_frac=cfg.stations.holdout_frac,
    )
    logger.info("%d station(s) kept", stations.sizes["station"])

    interpolator = curvilinear_for(model.lat, model.lon, stations, k=4)
    fields = {
        "coarse": interpolator.apply(
            xr.DataArray(pairs.coarse_t2m.values, dims=("time", "y", "x"))
        ).values,
        "downscaled": interpolator.apply(xr.DataArray(predicted, dims=("time", "y", "x"))).values,
        "target": interpolator.apply(
            xr.DataArray(pairs.fine_t2m.values, dims=("time", "y", "x"))
        ).values,
    }
    observed = stations[cfg.stations.variables[0]].reindex(
        time=times.to_numpy().astype("datetime64[ns]")
    ).values

    edges = elevation_edges(cfg.evaluation.elevation_edges)
    scores = score_against_stations(
        fields, observed, stations.elevation.values, times.hour.values, edges
    )
    scores.to_csv(out / "floor.csv", index=False)
    floor_figure(scores, out / "floor.png", fields=FIELDS, labels=LABELS)

    overall = scores[scores.grouping == "all"].set_index("field")
    summary = {
        "period": [str(times.min()), str(times.max())],
        "n_times": len(times),
        "n_stations": int(stations.sizes["station"]),
        "rmse_K": {name: float(overall.loc[name, "rmse_K"]) for name in FIELDS},
        "bias_K": {name: float(overall.loc[name, "bias_K"]) for name in FIELDS},
    }
    floor = summary["rmse_K"]["target"]
    gap = summary["rmse_K"]["downscaled"] - floor
    recoverable = summary["rmse_K"]["coarse"] - floor
    summary["remaining_gap_to_floor_K"] = round(gap, 3)
    summary["share_of_recoverable_error_removed"] = (
        round(1 - gap / recoverable, 3) if recoverable > 0 else None
    )
    (out / "floor.json").write_text(json.dumps(summary, indent=2))

    print("\nRMSE against stations over the target's own period (K)")
    for name in FIELDS:
        print(f"  {LABELS[name]:32s} {summary['rmse_K'][name]:.3f}  (bias {summary['bias_K'][name]:+.3f})")
    print(
        f"\nthe target itself is {floor:.2f} K from the stations, so that is the floor; "
        f"the downscaled field sits {gap:.2f} K above it"
    )
    if summary["share_of_recoverable_error_removed"] is not None:
        print(
            f"the model removes {100 * summary['share_of_recoverable_error_removed']:.0f}% of the error "
            "that separating the coarse field from the target could remove"
        )
    print("\nby elevation band")
    print(
        scores[scores.grouping == "elevation"]
        .pivot(index="band", columns="field", values="rmse_K")[FIELDS]
        .to_string()
    )
    print(f"\nwrote {out}/floor.csv, floor.png, floor.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())