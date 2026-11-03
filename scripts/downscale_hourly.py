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

import numpy as np
import xarray as xr

from downscale import hourly as hr
from downscale.config import load_config
from downscale.inference import downscale_forecast, to_fine_grid

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


def hourly_forecast(forecast: xr.DataArray) -> xr.DataArray:
    """The coarse forecast at every hour, from its first lead time to the hour before its last.

    The last lead time has no window after it, so the hours stop just before it.

    :param forecast: Forecast over ``(time, lead_time, ensemble, lat, lon)``.
    :type forecast: xarray.DataArray
    :returns: The interpolated forecast.
    :rtype: xarray.DataArray
    """
    return hr.hourly_leads(forecast).isel(lead_time=slice(None, -1))


def downscale_hours(
    model: hr.HourlyDownscaler, forecast: xr.DataArray, leads: np.ndarray, samples: int, batch: int, seed: int = 0
) -> xr.Dataset:
    """Run the hourly model over every member of a 6-hourly forecast.

    :param model: Hourly model.
    :type model: downscale.hourly.HourlyDownscaler
    :param forecast: 6-hourly forecast over ``(time, lead_time, ensemble, lat, lon)`` in kelvin.
    :type forecast: xarray.DataArray
    :param leads: Hourly lead times to produce.
    :type leads: numpy.ndarray
    :param samples: Fine maps per coarse member.
    :type samples: int
    :param batch: Hours processed at once.
    :type batch: int
    :param seed: Seed of the noise; member ``m`` uses ``seed + m`` at every hour.
    :type seed: int
    :returns: ``t2m_interp`` and ``t2m_ds`` laid out as :func:`downscale.inference.downscale_forecast` does.
    :rtype: xarray.Dataset
    :raises ValueError: If a lead time's window does not start on a forecast step.
    """
    coarse = to_fine_grid(forecast, model.lat, model.lon).astype("f4")
    steps = forecast["lead_time"].values
    init = forecast["time"].values
    n_init, _, n_ensemble = coarse.shape[:3]
    n_samples = samples if model.net.cfg.stochastic else 1
    valid = init[:, None] + leads[None, :]
    offsets = hr.offsets_of(valid[0])
    if not all(hr.offsets_of(v).tolist() == offsets.tolist() for v in valid):
        raise ValueError("initialisations at different hours of the 6-hour cycle")
    start_leads = leads - offsets.astype("timedelta64[h]").astype(leads.dtype)
    position = {lead: i for i, lead in enumerate(steps.tolist())}
    missing = [s for s in start_leads.tolist() if s not in position]
    if missing:
        raise ValueError(f"windows start between forecast steps, e.g. at lead {missing[0]}")
    window = np.timedelta64(hr.WINDOW_HOURS, "h").astype(leads.dtype)
    starts = np.array([position[s] for s in start_leads.tolist()])
    ends = np.array([position[s] for s in (start_leads + window).tolist()])

    line = np.empty((n_init, len(leads), n_ensemble, *model.lat.shape), "f4")
    fine = np.empty((n_init, len(leads), n_ensemble, n_samples, *model.lat.shape), "f4")
    for i in range(n_init):
        for member in range(n_ensemble):
            c0, c1 = coarse[i, starts, member], coarse[i, ends, member]
            line[i, :, member] = hr.temporal_interpolation(c0, c1, offsets)
            fine[i, :, member] = model.predict_hours(
                c0, c1, offsets, valid[i], members=samples, batch=batch, seed=seed + member
            )
    return xr.Dataset(
        {
            "t2m_interp": (("time", "lead_time", "ensemble", "y", "x"), line, {"units": "K"}),
            "t2m_ds": (("time", "lead_time", "ensemble", "sample", "y", "x"), fine, {"units": "K"}),
            "land_sea_mask": (("y", "x"), model.static["land_sea_mask"].astype("f4")),
            "orography": (("y", "x"), model.static["cerra_orography"].astype("f4"), {"units": "m"}),
        },
        coords={
            "time": init,
            "lead_time": leads,
            "ensemble": forecast["ensemble"].values,
            "sample": np.arange(n_samples),
            "lat": (("y", "x"), model.lat),
            "lon": (("y", "x"), model.lon),
            "valid_time": (("time", "lead_time"), valid),
        },
    )


def run_one(forecast_path: Path, hourly, spatial, out: Path, variable: str, samples: int, batch: int) -> None:
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
    :returns: ``None``
    :rtype: None
    """
    started = time.time()
    forecast = xr.open_zarr(forecast_path)[variable]
    if "ensemble" not in forecast.dims:
        forecast = forecast.expand_dims(ensemble=[0])
    forecast = forecast.transpose("time", "lead_time", "ensemble", "lat", "lon").load()
    coarse_hourly = hourly_forecast(forecast)
    name = Path(forecast_path).name
    write(coarse_hourly.to_dataset(name=variable), out / "forecasts" / name, done=True)
    leads = coarse_hourly["lead_time"].values
    write(downscale_hours(hourly, forecast, leads, samples, batch), out / "hourly" / name)
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
        run_one(Path(path), hourly, spatial, Path(args.out), cfg.coarse.variable, samples, batch)
    print(f"\n{len(forecasts)} forecast(s) written to {args.out}/{{forecasts,hourly,spatial}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
