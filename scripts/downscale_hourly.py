"""Downscale 6-hourly forecasts to every hour, ready for ``eval_downscaled.py``.

Usage::

    python scripts/downscale_hourly.py --hourly runs/hourly/model.pt --spatial runs/spatial/model.pt \\
        --forecasts outputs/forecasts/*.zarr --out outputs/hourly

For each forecast store three stores of the same name are written, all at every hour from the first
lead time to the hour before the last:

- ``forecasts/``: the coarse forecast interpolated in time, the reference ``eval_downscaled.py`` reads;
- ``hourly/``: the hourly model, each sample keeping its noise from hour to hour;
- ``spatial/``: the spatial model applied to the interpolated coarse forecast, the baseline.

Score both against the stations with::

    python scripts/eval_downscaled.py --forecasts outputs/hourly/forecasts/*.zarr \\
        --downscaled outputs/hourly/hourly/*.zarr --out outputs/hourly/eval_hourly
    python scripts/eval_downscaled.py --forecasts outputs/hourly/forecasts/*.zarr \\
        --downscaled outputs/hourly/spatial/*.zarr --out outputs/hourly/eval_spatial
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

import xarray as xr

from downscale import hourly as hr
from downscale.config import load_config
from downscale.inference import downscale_forecast

logger = logging.getLogger("downscale_hourly")


def write(ds: xr.Dataset, target: Path, done: bool = False) -> Path:
    """Write a store through a temporary name, optionally with a completion marker.

    :param ds: Dataset to write.
    :type ds: xarray.Dataset
    :param target: Destination store.
    :type target: pathlib.Path
    :param done: Also write the ``.done`` marker forecast stores carry.
    :type done: bool
    :returns: The destination.
    :rtype: pathlib.Path
    """
    for variable in ds.variables.values():
        variable.encoding.clear()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    ds.to_zarr(tmp, mode="w")
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    if done:
        Path(str(target) + ".done").touch()
    return target


def run_one(forecast_path: Path, hourly, spatial, out: Path, variable: str, samples: int, batch: int,
            noise_hours: float | None = None) -> None:
    """Write the three hourly stores for one forecast.

    :param forecast_path: 6-hourly forecast store.
    :type forecast_path: pathlib.Path
    :param hourly: Hourly model.
    :type hourly: downscale.hourly.HourlyDownscaler
    :param spatial: Spatial model.
    :type spatial: downscale.downscaler.Downscaler
    :param out: Output directory.
    :type out: pathlib.Path
    :param variable: Variable to downscale.
    :type variable: str
    :param samples: Fine maps per coarse member.
    :type samples: int
    :param batch: Hours processed at once.
    :type batch: int
    :param noise_hours: Correlation time [h] of the noise; ``None`` keeps it fixed.
    :type noise_hours: float | None
    :returns: ``None``
    :rtype: None
    """
    started = time.time()
    forecast = xr.open_zarr(forecast_path)[variable]
    if "ensemble" not in forecast.dims:
        forecast = forecast.expand_dims(ensemble=[0])
    forecast = forecast.transpose("time", "lead_time", "ensemble", "lat", "lon").load()
    coarse_hourly = hr.hourly_forecast(forecast)
    name = Path(forecast_path).name
    write(coarse_hourly.to_dataset(name=variable), out / "forecasts" / name, done=True)
    leads = coarse_hourly["lead_time"].values
    write(hr.downscale_hours(hourly, forecast, leads, samples, batch, noise_hours=noise_hours), out / "hourly" / name)
    write(downscale_forecast(spatial, coarse_hourly, samples, batch), out / "spatial" / name)
    logger.info("%s: %d hourly lead times in %.0fs", name, len(leads), time.time() - started)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    :param argv: Argument list; ``None`` reads ``sys.argv``.
    :type argv: collections.abc.Sequence[str] | None
    :returns: Exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--forecasts", nargs="+", default=None, help="overrides paths.forecasts")
    parser.add_argument("--hourly", required=True, help="checkpoint of the hourly model")
    parser.add_argument("--spatial", required=True, help="checkpoint of the spatial model")
    parser.add_argument("--out", default="outputs/hourly")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--noise-hours", type=float, default=None,
                        help="correlation time [h] of the noise; overrides temporal.noise_hours")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    forecasts = sorted(args.forecasts) if args.forecasts else sorted(
        str(p) for p in Path(cfg.paths.forecasts).glob("*.zarr")
    )
    if not forecasts:
        parser.error("no forecasts found; pass --forecasts or set paths.forecasts")
    clash = [p for p in forecasts if Path(p).resolve().parent == (Path(args.out) / "forecasts").resolve()]
    if clash:
        parser.error(f"{clash[0]} would be overwritten by its hourly version; pick another --out")
    hourly = hr.HourlyDownscaler.load(args.hourly, args.device)
    spatial = hr.load_spatial(args.spatial, args.device)
    samples = args.samples or cfg.inference.samples
    batch = args.batch_size or cfg.inference.batch_size
    for path in forecasts:
        run_one(Path(path), hourly, spatial, Path(args.out), cfg.coarse.variable, samples, batch,
                args.noise_hours if args.noise_hours is not None else cfg.temporal.noise_hours)
    print(f"\n{len(forecasts)} forecast(s) written to {args.out}/{{forecasts,hourly,spatial}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
