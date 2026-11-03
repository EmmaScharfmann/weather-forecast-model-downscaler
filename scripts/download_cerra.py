"""Download the high-resolution reanalysis used as the training target.

Requires a Copernicus Climate Data Store account, credentials in
``~/.cdsapirc`` and acceptance of the dataset licence on its web page.

Usage::

    python scripts/download_cerra.py --config config.yaml
    python scripts/download_cerra.py --config config.yaml --consolidate
    python scripts/download_cerra.py --hourly --start 2020-07-01 --end 2020-08-31 --out data/cerra_hourly

One request is issued per month and each month is written to its own store, so
rerunning the command resumes an interrupted download.

``--hourly`` fetches every hour instead of ``target.times``: CERRA analyses exist
every three hours, so the two hours after each come from its +1 h and +2 h forecasts.
``--start`` and ``--end`` restrict the download to a range of days.
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from downscale.config import Config, load_config
from downscale.constants import G, T2M_RANGE_K

logger = logging.getLogger("cerra")

#: Times of the three-hourly analyses, and the forecast lead times that fill the hours between them.
ANALYSIS_TIMES = [f"{h:02d}:00" for h in range(0, 24, 3)]
FORECAST_LEADS = ["1", "2"]
HOURLY_TIMES = [f"{h:02d}:00" for h in range(24)]


def build_request(
    dataset_variables: Sequence[str], year: int, month: int, times: Sequence[str],
    days: Sequence[int] | None = None, leadtimes: Sequence[str] | None = None,
) -> dict:
    """Build one Climate Data Store request.

    :param dataset_variables: Dataset variable names.
    :type dataset_variables: collections.abc.Sequence[str]
    :param year: Calendar year.
    :type year: int
    :param month: Calendar month.
    :type month: int
    :param times: Times of day as ``"HH:MM"``.
    :type times: collections.abc.Sequence[str]
    :param days: Days of month; ``None`` requests the whole month.
    :type days: collections.abc.Sequence[int] | None
    :param leadtimes: Forecast lead times in hours; ``None`` requests analyses.
    :type leadtimes: collections.abc.Sequence[str] | None
    :returns: Request body.
    :rtype: dict
    """
    n_days = calendar.monthrange(year, month)[1]
    extra = {"leadtime_hour": list(leadtimes)} if leadtimes else {}
    return {
        "level_type": "surface_or_atmosphere",
        "data_type": "reanalysis",
        "product_type": "forecast" if leadtimes else "analysis",
        **extra,
        "variable": list(dataset_variables),
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in (days or range(1, n_days + 1))],
        "time": list(times),
        "data_format": "grib",
    }


def retrieve(dataset: str, request: dict, target: Path, client=None) -> Path:
    """Run one request, writing to a temporary name until it completes.

    :param dataset: Dataset identifier.
    :type dataset: str
    :param request: Request body.
    :type request: dict
    :param target: Destination file.
    :type target: pathlib.Path
    :param client: Climate Data Store client; ``None`` builds one.
    :type client: typing.Any
    :returns: The destination file.
    :rtype: pathlib.Path
    """
    if client is None:
        import cdsapi

        client = cdsapi.Client()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    client.retrieve(dataset, request, str(partial))
    partial.rename(target)
    return target


def open_grib(path: Path) -> xr.Dataset:
    """Open a GRIB file.

    :param path: File to open.
    :type path: pathlib.Path
    :returns: Dataset as decoded by ``cfgrib``.
    :rtype: xarray.Dataset
    """
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def split_steps(ds: xr.Dataset) -> list[xr.Dataset]:
    """Split a forecast decoded over ``(time, step)`` into one dataset per lead time.

    :param ds: Dataset as decoded from GRIB.
    :type ds: xarray.Dataset
    :returns: The dataset itself when it has no ``step`` dimension, otherwise one dataset per step,
        each with a one-dimensional ``valid_time`` that :func:`standardise` turns into its time.
    :rtype: list[xarray.Dataset]
    """
    if "step" not in ds.dims:
        return [ds]
    return [ds.isel(step=i) for i in range(ds.sizes["step"])]


def standardise(ds: xr.Dataset) -> xr.Dataset:
    """Normalise coordinate names and longitude convention.

    :param ds: Dataset as decoded from GRIB.
    :type ds: xarray.Dataset
    :returns: Dataset with ``lat`` and ``lon`` two-dimensional coordinates and signed longitudes.
    :rtype: xarray.Dataset
    :raises ValueError: If the coordinates are not two dimensional.
    """
    if "valid_time" in ds.coords and "time" in ds.dims and ds["valid_time"].ndim == 1:
        ds = ds.assign_coords(time=ds["valid_time"].values)
    for coord in ("step", "heightAboveGround", "surface", "valid_time", "number"):
        if coord in ds.coords and coord not in ds.dims:
            ds = ds.drop_vars(coord)
    renames = {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in ds.coords}
    ds = ds.rename(renames)
    if ds["lat"].ndim != 2:
        raise ValueError(f"expected two-dimensional lat/lon coordinates, got dims {ds['lat'].dims}")
    return ds.assign_coords(lon=((ds["lon"] + 180.0) % 360.0) - 180.0)


def crop_box(lat2d: np.ndarray, lon2d: np.ndarray, box: tuple[float, float, float, float]) -> dict:
    """Find the index rectangle containing a bounding box.

    A rectangle in index space keeps the array regular for a convolutional
    network; its corners reach slightly outside the requested box.

    :param lat2d: Grid latitudes with shape ``(y, x)``.
    :type lat2d: numpy.ndarray
    :param lon2d: Grid longitudes with shape ``(y, x)``, signed.
    :type lon2d: numpy.ndarray
    :param box: ``(lat_min, lat_max, lon_min, lon_max)``.
    :type box: tuple[float, float, float, float]
    :returns: Slice bounds, the source grid shape and the box.
    :rtype: dict
    :raises ValueError: If no grid point falls inside the box.
    """
    lat_min, lat_max, lon_min, lon_max = box
    inside = (lat2d >= lat_min) & (lat2d <= lat_max) & (lon2d >= lon_min) & (lon2d <= lon_max)
    if not inside.any():
        raise ValueError(f"no grid point inside {box}")
    ys, xs = np.nonzero(inside)
    return {
        "y0": int(ys.min()), "y1": int(ys.max()) + 1,
        "x0": int(xs.min()), "x1": int(xs.max()) + 1,
        "grid_shape": [int(s) for s in lat2d.shape], "box": list(box),
    }


def apply_crop(ds: xr.Dataset, crop: dict) -> xr.Dataset:
    """Apply a saved index rectangle.

    :param ds: Dataset on the full source grid.
    :type ds: xarray.Dataset
    :param crop: Rectangle from :func:`crop_box`.
    :type crop: dict
    :returns: Cropped dataset with dimensions named ``y`` and ``x``.
    :rtype: xarray.Dataset
    :raises ValueError: If the dataset's grid differs from the one the rectangle was built for.
    """
    y_dim, x_dim = ds["lat"].dims
    shape = [int(ds.sizes[y_dim]), int(ds.sizes[x_dim])]
    if shape != crop["grid_shape"]:
        raise ValueError(f"grid {shape} differs from the saved crop's grid {crop['grid_shape']}")
    out = ds.isel({y_dim: slice(crop["y0"], crop["y1"]), x_dim: slice(crop["x0"], crop["x1"])})
    return out.rename({y_dim: "y", x_dim: "x"}) if (y_dim, x_dim) != ("y", "x") else out


def _clean(ds: xr.Dataset) -> xr.Dataset:
    """Strip encodings and GRIB attributes before writing.

    :param ds: Dataset to clean.
    :type ds: xarray.Dataset
    :returns: The same dataset, modified in place.
    :rtype: xarray.Dataset
    """
    for variable in ds.variables.values():
        variable.encoding.clear()
    ds.attrs = {k: v for k, v in ds.attrs.items() if not k.startswith("GRIB")}
    return ds


def expected_times(
    year: int, month: int, times: Sequence[str], days: Sequence[int] | None = None
) -> pd.DatetimeIndex:
    """List every timestamp a complete month should contain.

    :param year: Calendar year.
    :type year: int
    :param month: Calendar month.
    :type month: int
    :param times: Times of day as ``"HH:MM"``.
    :type times: collections.abc.Sequence[str]
    :param days: Days of month; ``None`` expects the whole month.
    :type days: collections.abc.Sequence[int] | None
    :returns: Expected timestamps.
    :rtype: pandas.DatetimeIndex
    """
    days = days or range(1, calendar.monthrange(year, month)[1] + 1)
    dates = [pd.Timestamp(year, month, d) for d in days]
    return pd.DatetimeIndex([d + pd.Timedelta(t + ":00") for d in dates for t in times])


def month_days(year: int, month: int, start: str | None, end: str | None) -> list[int] | None:
    """The days of a month that fall inside a date range.

    :param year: Calendar year.
    :type year: int
    :param month: Calendar month.
    :type month: int
    :param start: First day, ``None`` for no lower bound.
    :type start: str | None
    :param end: Last day, ``None`` for no upper bound.
    :type end: str | None
    :returns: The days, or ``None`` when the whole month is inside the range.
    :rtype: list[int] | None
    """
    n_days = calendar.monthrange(year, month)[1]
    first = pd.Timestamp(start).normalize() if start else pd.Timestamp.min
    last = pd.Timestamp(end).normalize() if end else pd.Timestamp.max
    days = [d for d in range(1, n_days + 1) if first <= pd.Timestamp(year, month, d) <= last]
    return None if len(days) == n_days else days


def check_month(
    ds: xr.Dataset, year: int, month: int, times: Sequence[str], days: Sequence[int] | None = None
) -> list[str]:
    """Check one month for missing, duplicate or implausible values.

    :param ds: Cropped month with a ``t2m`` variable.
    :type ds: xarray.Dataset
    :param year: Calendar year.
    :type year: int
    :param month: Calendar month.
    :type month: int
    :param times: Expected times of day.
    :type times: collections.abc.Sequence[str]
    :param days: Expected days; ``None`` expects the whole month.
    :type days: collections.abc.Sequence[int] | None
    :returns: Human-readable problems; empty when the month is sound.
    :rtype: list[str]
    """
    problems = []
    found = pd.DatetimeIndex(ds["time"].values)
    missing = expected_times(year, month, times, days).difference(found)
    if len(missing):
        problems.append(f"{len(missing)} missing time(s), e.g. {list(missing[:3])}")
    if found.has_duplicates:
        problems.append("duplicate times")
    values = ds["t2m"].values
    if np.isnan(values).any():
        problems.append(f"{int(np.isnan(values).sum())} NaN value(s)")
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if not (T2M_RANGE_K[0] < low and high < T2M_RANGE_K[1]):
        problems.append(f"t2m range {low:.1f}..{high:.1f} K outside {T2M_RANGE_K}")
    return problems


def load_or_make_crop(out: Path, standardised: xr.Dataset, box) -> dict:
    """Read the saved index rectangle, computing it on first use.

    :param out: Output directory.
    :type out: pathlib.Path
    :param standardised: Dataset carrying the full source grid.
    :type standardised: xarray.Dataset
    :param box: Bounding box.
    :type box: tuple[float, float, float, float]
    :returns: The rectangle, shared by every month.
    :rtype: dict
    """
    path = out / "crop.json"
    if path.exists():
        return json.loads(path.read_text())
    crop = crop_box(standardised["lat"].values, standardised["lon"].values, box)
    path.write_text(json.dumps(crop, indent=2))
    logger.info(
        "crop fixed: y %d:%d, x %d:%d of grid %s, giving %d x %d points",
        crop["y0"], crop["y1"], crop["x0"], crop["x1"], crop["grid_shape"],
        crop["y1"] - crop["y0"], crop["x1"] - crop["x0"],
    )
    return crop


def adopt_crop(out: Path, like: str) -> dict:
    """Use the crop of a published dataset, so a new download lands on the same grid.

    :param out: Output directory.
    :type out: pathlib.Path
    :param like: Local folder or Hub dataset whose ``manifest.json`` records its crop.
    :type like: str
    :returns: The crop, written to ``crop.json``.
    :rtype: dict
    :raises ValueError: If the dataset records no crop, or this download already has a different one.
    """
    from downscale.fetch import _manifest

    manifest = _manifest(like)
    if not manifest or "crop" not in manifest:
        raise ValueError(f"{like} records no crop in its manifest")
    crop = manifest["crop"]
    path = out / "crop.json"
    if path.exists() and json.loads(path.read_text()) != crop:
        raise ValueError(f"{path} holds a different crop than {like}: use a fresh output directory")
    path.write_text(json.dumps(crop, indent=2))
    logger.info("using the crop of %s: y %d:%d, x %d:%d", like, crop["y0"], crop["y1"], crop["x0"], crop["x1"])
    return crop


def do_static(cfg: Config, out: Path, client=None, opener: Callable = open_grib) -> Path:
    """Download and crop the time-invariant fields.

    :param cfg: Pipeline configuration.
    :type cfg: downscale.config.Config
    :param out: Output directory.
    :type out: pathlib.Path
    :param client: Climate Data Store client; ``None`` builds one.
    :type client: typing.Any
    :param opener: Callable opening a GRIB file, replaceable for testing.
    :type opener: typing.Callable
    :returns: Path to the static store.
    :rtype: pathlib.Path
    :raises KeyError: If the orography or land-sea mask is absent from the response.
    """
    target = out / "static.zarr"
    if target.exists():
        logger.info("static fields already downloaded")
        return target
    grib = out / "grib" / "static.grib"
    if not grib.exists():
        request = build_request(cfg.target.static_variables, cfg.target.years[0], 1, ["00:00"], days=[1])
        retrieve(cfg.target.dataset, request, grib, client)
    ds = standardise(opener(grib))
    crop = load_or_make_crop(out, ds, cfg.region_box)
    ds = apply_crop(ds, crop)
    if "time" in ds.dims:
        ds = ds.isel(time=0, drop=True)
    orography_name = next((v for v in ds.data_vars if v in ("orog", "z", "orography", "h")), None)
    mask_name = next((v for v in ds.data_vars if v in ("lsm", "land_sea_mask")), None)
    if orography_name is None or mask_name is None:
        raise KeyError(f"orography or land-sea mask not found; variables: {list(ds.data_vars)}")
    height = ds[orography_name].astype("f4")
    units = str(height.attrs.get("units", "")).replace(" ", "")
    if units in ("m**2s**-2", "m2s-2") or float(height.max()) > 20000:
        height = height / G
    static = xr.Dataset(
        {
            "orography": height.assign_attrs(units="m"),
            "land_sea_mask": ds[mask_name].astype("f4").assign_attrs(units="1"),
        },
        coords={"lat": ds["lat"], "lon": ds["lon"]},
    )
    static.attrs = {"source": f"{cfg.target.dataset}, cropped", "crop": json.dumps(crop)}
    _clean(static).to_zarr(target, mode="w")
    logger.info(
        "static fields: orography %.0f..%.0f m, land fraction %.2f",
        float(static.orography.min()), float(static.orography.max()), float(static.land_sea_mask.mean()),
    )
    return target


def do_month(
    cfg: Config, out: Path, year: int, month: int, keep_grib: bool = False,
    client=None, opener: Callable = open_grib, hourly: bool = False, days: Sequence[int] | None = None,
) -> tuple[str, list[str]]:
    """Download, crop and check one month.

    :param cfg: Pipeline configuration.
    :type cfg: downscale.config.Config
    :param out: Output directory.
    :type out: pathlib.Path
    :param year: Calendar year.
    :type year: int
    :param month: Calendar month.
    :type month: int
    :param keep_grib: Keep the downloaded GRIB file after cropping.
    :type keep_grib: bool
    :param client: Climate Data Store client; ``None`` builds one.
    :type client: typing.Any
    :param opener: Callable opening a GRIB file, replaceable for testing.
    :type opener: typing.Callable
    :param hourly: Fetch every hour, from the analyses and their +1 h and +2 h forecasts,
        instead of ``target.times``.
    :type hourly: bool
    :param days: Days of the month to fetch; ``None`` fetches all of them.
    :type days: collections.abc.Sequence[int] | None
    :returns: The month tag and any problems found; nothing is written when problems exist.
    :rtype: tuple[str, list[str]]
    :raises KeyError: If the target variable is absent from the response.
    """
    tag = f"{year}{month:02d}"
    target = out / "months" / f"t2m_{tag}.zarr"
    if target.exists():
        return tag, []
    variable = [cfg.target.variable]
    if hourly:
        requests = {
            out / "grib" / f"t2m_{tag}_an.grib": build_request(variable, year, month, ANALYSIS_TIMES, days),
            out / "grib" / f"t2m_{tag}_fc.grib": build_request(
                variable, year, month, ANALYSIS_TIMES, days, leadtimes=FORECAST_LEADS
            ),
        }
    else:
        requests = {out / "grib" / f"t2m_{tag}.grib": build_request(variable, year, month, cfg.target.times, days)}
    parts = []
    for grib, request in requests.items():
        if not grib.exists():
            logger.info("requesting %s %s", tag, request["product_type"])
            retrieve(cfg.target.dataset, request, grib, client)
        parts += [standardise(part) for part in split_steps(opener(grib))]
    ds = xr.concat(parts, "time") if len(parts) > 1 else parts[0]
    ds = apply_crop(ds, load_or_make_crop(out, ds, cfg.region_box))
    name = next((v for v in ds.data_vars if v in ("t2m", "2t")), None)
    if name is None:
        raise KeyError(f"2 m temperature not found; variables: {list(ds.data_vars)}")
    ds = ds[[name]].rename({name: "t2m"}).sortby("time")
    ds["t2m"] = ds["t2m"].astype("f4").assign_attrs(units="K")
    problems = check_month(ds, year, month, HOURLY_TIMES if hourly else cfg.target.times, days)
    if problems:
        return tag, problems
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    _clean(ds.load()).to_zarr(tmp, mode="w")
    tmp.rename(target)
    if not keep_grib:
        for grib in requests:
            grib.unlink()
    return tag, []


def consolidate(cfg: Config, out: Path, months: Sequence[tuple[int, int]] | None = None) -> Path:
    """Merge the downloaded months into a single store.

    :param cfg: Pipeline configuration.
    :type cfg: downscale.config.Config
    :param out: Output directory.
    :type out: pathlib.Path
    :param months: ``(year, month)`` pairs to merge; ``None`` merges every month of ``target.years``.
    :type months: collections.abc.Sequence[tuple[int, int]] | None
    :returns: Path to the consolidated store.
    :rtype: pathlib.Path
    :raises FileNotFoundError: If any month is still missing.
    :raises ValueError: If the static grid and the monthly grid differ.
    """
    first, last = cfg.target.years
    months = months or [(y, m) for y in range(int(first), int(last) + 1) for m in range(1, 13)]
    tags = [f"{y}{m:02d}" for y, m in months]
    paths = [out / "months" / f"t2m_{t}.zarr" for t in tags]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} month(s) not downloaded yet, e.g. {missing[:3]}")
    ds = xr.open_mfdataset([str(p) for p in paths], engine="zarr", combine="nested", concat_dim="time")
    ds = ds.chunk({"time": 124, "y": -1, "x": -1})
    times = pd.DatetimeIndex(ds.time.values)
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("consolidated times are not strictly increasing")
    static = xr.open_zarr(out / "static.zarr")
    if not (np.array_equal(static.lat.values, ds.lat.values) and np.array_equal(static.lon.values, ds.lon.values)):
        raise ValueError("the static grid and the monthly grid differ")
    ds.attrs = {
        "source": f"{cfg.target.dataset}, cropped",
        "years": f"{tags[0][:4]}-{tags[-1][:4]}",
        "times_utc": ",".join(sorted({f"{t.hour:02d}:00" for t in times})),
        "license": "Copernicus licence; contains modified Copernicus Climate Change Service information",
    }
    target = out / "target.zarr"
    _clean(ds).to_zarr(target, mode="w")
    logger.info(
        "consolidated %d time steps on a %d x %d grid into %s",
        ds.sizes["time"], ds.sizes["y"], ds.sizes["x"], target,
    )
    return target


def sanity_plot(out: Path, tag: str, path: Path) -> None:
    """Plot the warmest step of a month next to the orography.

    :param out: Output directory.
    :type out: pathlib.Path
    :param tag: Month tag such as ``"201407"``.
    :type tag: str
    :param path: Destination image.
    :type path: pathlib.Path
    :returns: ``None``
    :rtype: None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = xr.open_zarr(out / "months" / f"t2m_{tag}.zarr")
    static = xr.open_zarr(out / "static.zarr")
    index = int(np.argmax(ds.t2m.mean(("y", "x")).values))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    temperature = ax[0].pcolormesh(ds.lon, ds.lat, ds.t2m.isel(time=index) - 273.15, cmap="RdYlBu_r", shading="auto")
    fig.colorbar(temperature, ax=ax[0], label="degC")
    ax[0].set_title(f"2 m temperature, {pd.Timestamp(ds.time.values[index]):%Y-%m-%d %H UTC}")
    orography = ax[1].pcolormesh(static.lon, static.lat, static.orography, cmap="terrain", shading="auto")
    fig.colorbar(orography, ax=ax[1], label="m")
    ax[1].set_title("orography")
    mid_lat = float(np.mean(static.lat.values))
    for axis in ax:
        axis.set_aspect(1 / np.cos(np.deg2rad(mid_lat)))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logger.info("sanity plot written to %s", path)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    :param argv: Argument list; ``None`` reads ``sys.argv``.
    :type argv: collections.abc.Sequence[str] | None
    :returns: Exit status, non-zero when a month reported problems.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None, help="overrides paths.target")
    parser.add_argument("--workers", type=int, default=None, help="overrides target.cds_workers")
    parser.add_argument("--keep-grib", action="store_true")
    parser.add_argument("--consolidate", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the requests and exit")
    parser.add_argument("--hourly", action="store_true", help="fetch every hour instead of target.times")
    parser.add_argument("--start", default=None, help="first day, overriding target.years")
    parser.add_argument("--end", default=None, help="last day, overriding target.years")
    parser.add_argument(
        "--crop-like", default=None, metavar="DATASET",
        help="take the crop from a published dataset's manifest, e.g. EmmaScharfmann/cerra-france-t2m-2016-2020, "
        "so the grid matches models trained on it",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    out = Path(args.out or cfg.paths.target)
    out.mkdir(parents=True, exist_ok=True)
    if args.crop_like:
        adopt_crop(out, args.crop_like)
    workers = args.workers or cfg.target.cds_workers
    first, last = int(cfg.target.years[0]), int(cfg.target.years[1])
    months = [(y, m) for y in range(first, last + 1) for m in range(1, 13)]
    if args.start or args.end:
        start = pd.Timestamp(args.start or f"{first}-01-01")
        end = pd.Timestamp(args.end or f"{last}-12-31")
        months = [(d.year, d.month) for d in pd.date_range(start.replace(day=1), end, freq="MS")]
    days = {(y, m): month_days(y, m, args.start, args.end) for y, m in months}

    if args.consolidate:
        consolidate(cfg, out, months)
        return 0
    if args.dry_run:
        times = ANALYSIS_TIMES if args.hourly else cfg.target.times
        print(json.dumps(build_request(cfg.target.static_variables, first, 1, ["00:00"], days=[1]), indent=1))
        print(json.dumps(build_request([cfg.target.variable], *months[0], times, days[months[0]]), indent=1))
        if args.hourly:
            print(json.dumps(build_request(
                [cfg.target.variable], *months[0], times, days[months[0]], leadtimes=FORECAST_LEADS
            ), indent=1))
        print(f"... {len(months)} month(s) in total")
        return 0

    do_static(cfg, out)
    failed: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(do_month, cfg, out, y, m, args.keep_grib, hourly=args.hourly, days=days[(y, m)]): (y, m)
            for y, m in months
        }
        for n, future in enumerate(as_completed(futures), 1):
            year, month = futures[future]
            try:
                tag, problems = future.result()
                if problems:
                    failed[tag] = problems
                logger.info(
                    "[%d/%d] %d-%02d %s", n, len(months), year, month,
                    "problems: " + "; ".join(problems) if problems else "ok",
                )
            except Exception as exc:  # noqa: BLE001
                failed[f"{year}{month:02d}"] = [repr(exc)]
                logger.error("[%d/%d] %d-%02d failed: %r", n, len(months), year, month, exc)

    done = sorted(p.stem.split("_")[1] for p in (out / "months").glob("t2m_*.zarr"))
    if done:
        sanity_plot(out, done[len(done) // 2], out / "sanity.png")
    print(f"\n{len(done)}/{len(months)} months downloaded; {len(failed)} with problems")
    for tag, problems in sorted(failed.items()):
        print(f"  {tag}: {'; '.join(problems)}")
    print(
        "rerun the same command to retry, then add --consolidate"
        if failed or len(done) < len(months)
        else "all months downloaded: rerun with --consolidate"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())