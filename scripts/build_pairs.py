"""Build coarse/fine training pairs for the downscaler.

The coarse source is interpolated onto the target grid, so the model sees input
and target pixel by pixel. The interpolated field is both the model's main
input and the baseline it has to beat.

Usage::

    python scripts/build_pairs.py --config config.yaml
    python scripts/build_pairs.py --config config.yaml --consolidate

Months whose target file is not downloaded yet are reported as pending, so this
can run alongside the download.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from downscale.config import Config, load_config
from downscale.constants import COARSE_CROP_MARGIN_DEG, G
from downscale.diagnostics import land_mask
from downscale.regions import crop_indices
from downscale.station_interp import GridInterpolator

logger = logging.getLogger("pairs")


class CoarseOnTargetGrid:
    """Fetches a coarse field and interpolates it onto the target grid."""

    def __init__(self, source, target_lat: np.ndarray, target_lon: np.ndarray, variable: str, batch: int = 28):
        """
        :param source: Callable taking ``(times, variables)`` and returning ``(time, variable, lat, lon)``.
        :type source: typing.Callable
        :param target_lat: Target grid latitudes with shape ``(y, x)``.
        :type target_lat: numpy.ndarray
        :param target_lon: Target grid longitudes with shape ``(y, x)``.
        :type target_lon: numpy.ndarray
        :param variable: Variable name to request.
        :type variable: str
        :param batch: Time steps per request.
        :type batch: int
        """
        self.source, self.variable, self.batch = source, variable, batch
        self.shape = target_lat.shape
        lat, lon = target_lat.ravel(), target_lon.ravel()
        self.box = (
            float(lat.min()) - COARSE_CROP_MARGIN_DEG, float(lat.max()) + COARSE_CROP_MARGIN_DEG,
            float(lon.min()) - COARSE_CROP_MARGIN_DEG, float(lon.max()) + COARSE_CROP_MARGIN_DEG,
        )
        self.points = (lat, lon)
        self._interp: GridInterpolator | None = None
        self._crop_idx: tuple[np.ndarray, np.ndarray] | None = None

    def _crop(self, da: xr.DataArray) -> xr.DataArray:
        """Cut a global field down to the target box plus a margin.

        :param da: Field with ``lat`` and ``lon`` coordinates.
        :type da: xarray.DataArray
        :returns: Cropped field.
        :rtype: xarray.DataArray
        """
        if self._crop_idx is None:
            self._crop_idx = crop_indices(self.box, da["lat"].values, da["lon"].values)
        return da.isel(lat=self._crop_idx[0], lon=self._crop_idx[1])

    def _to_target(self, da: xr.DataArray) -> np.ndarray:
        """Interpolate a cropped field onto the target grid.

        :param da: Cropped field.
        :type da: xarray.DataArray
        :returns: Values reshaped onto ``(..., y, x)``.
        :rtype: numpy.ndarray
        :raises ValueError: If any target point falls outside the cropped field.
        """
        if self._interp is None:
            self._interp = GridInterpolator.build(da["lat"].values, da["lon"].values, *self.points)
            if not self._interp.inside.all():
                raise ValueError("some target points fall outside the coarse crop; increase the margin")
        out = self._interp.apply(da)
        return out.values.reshape(*out.shape[:-1], *self.shape)

    def field(self, times) -> np.ndarray:
        """Fetch and interpolate the configured variable.

        :param times: Times to fetch.
        :type times: collections.abc.Sequence
        :returns: Values of shape ``(time, y, x)``.
        :rtype: numpy.ndarray
        """
        times = pd.DatetimeIndex(times)
        parts = []
        for i in range(0, times.size, self.batch):
            da = self.source(list(times[i:i + self.batch]), [self.variable]).sel(variable=self.variable, drop=True)
            parts.append(self._to_target(self._crop(da)))
        return np.concatenate(parts, axis=0).astype("f4")

    def surface_height(self, geopotential: xr.DataArray) -> np.ndarray:
        """Interpolate the coarse surface height onto the target grid.

        :param geopotential: Surface geopotential [m^2 s^-2].
        :type geopotential: xarray.DataArray
        :returns: Surface height [m] of shape ``(y, x)``.
        :rtype: numpy.ndarray
        """
        return self._to_target(self._crop(geopotential / G)).astype("f4")


def _clean(ds: xr.Dataset) -> xr.Dataset:
    """Strip encodings before writing.

    :param ds: Dataset to clean.
    :type ds: xarray.Dataset
    :returns: The same dataset, modified in place.
    :rtype: xarray.Dataset
    """
    for variable in ds.variables.values():
        variable.encoding.clear()
    return ds


def month_checks(coarse: np.ndarray, fine: np.ndarray) -> dict:
    """Summarise the agreement between a coarse and a fine field.

    :param coarse: Interpolated coarse field [K].
    :type coarse: numpy.ndarray
    :param fine: Target field [K].
    :type fine: numpy.ndarray
    :returns: Counts of missing values, correlation, mean difference and RMSE.
    :rtype: dict
    """
    valid = np.isfinite(coarse) & np.isfinite(fine)
    difference = (fine - coarse)[valid]
    correlation = float(np.corrcoef(coarse[valid], fine[valid])[0, 1]) if valid.sum() > 2 else float("nan")
    return {
        "nan_coarse": int((~np.isfinite(coarse)).sum()),
        "nan_fine": int((~np.isfinite(fine)).sum()),
        "corr": round(correlation, 4),
        "mean_diff_K": round(float(difference.mean()), 3),
        "rmse_K": round(float(np.sqrt((difference ** 2).mean())), 3),
    }


def do_month(
    target_dir: Path, out: Path, tag: str, coarse: CoarseOnTargetGrid,
    lat: np.ndarray, lon: np.ndarray,
) -> dict | None:
    """Pair one month of target data with the coarse source.

    :param target_dir: Directory holding the downloaded target months.
    :type target_dir: pathlib.Path
    :param out: Output directory.
    :type out: pathlib.Path
    :param tag: Month tag such as ``"201407"``.
    :type tag: str
    :param coarse: Coarse source bound to the target grid.
    :type coarse: CoarseOnTargetGrid
    :param lat: Target grid latitudes with shape ``(y, x)``.
    :type lat: numpy.ndarray
    :param lon: Target grid longitudes with shape ``(y, x)``.
    :type lon: numpy.ndarray
    :returns: A report, or ``None`` when the month was already built; nothing is
        written when the report lists problems.
    :rtype: dict | None
    :raises ValueError: If the month's grid differs from the static grid.
    """
    target = out / "months" / f"pairs_{tag}.zarr"
    if target.exists():
        return None
    fine = xr.open_zarr(target_dir / "months" / f"t2m_{tag}.zarr").load()
    if not (np.array_equal(fine.lat.values, lat) and np.array_equal(fine.lon.values, lon)):
        raise ValueError(f"{tag}: the month's grid differs from the static grid")
    started = time.time()
    coarse_values = coarse.field(fine.time.values)
    report = month_checks(coarse_values, fine.t2m.values)
    report.update(tag=tag, steps=int(fine.sizes["time"]), fetch_seconds=round(time.time() - started, 1))
    problems = []
    if report["nan_coarse"] or report["nan_fine"]:
        problems.append("missing values")
    if not report["corr"] > 0.9:
        problems.append(f"low correlation {report['corr']}")
    if abs(report["mean_diff_K"]) > 3:
        problems.append(f"large mean difference {report['mean_diff_K']} K")
    report["problems"] = problems
    if problems:
        return report

    ds = xr.Dataset(
        {
            "coarse_t2m": (("time", "y", "x"), coarse_values),
            "fine_t2m": (("time", "y", "x"), fine.t2m.values.astype("f4")),
        },
        coords={"time": fine.time.values, "lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
    )
    for name in ("coarse_t2m", "fine_t2m"):
        ds[name].attrs["units"] = "K"
    ds["coarse_t2m"].attrs["long_name"] = "coarse 2 m temperature, bilinear onto the target grid"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    _clean(ds).to_zarr(tmp, mode="w")
    tmp.rename(target)
    return report


def do_static(target_dir: Path, out: Path, coarse: CoarseOnTargetGrid, geopotential: xr.DataArray) -> Path:
    """Write the static fields both grids contribute.

    :param target_dir: Directory holding the downloaded target data.
    :type target_dir: pathlib.Path
    :param out: Output directory.
    :type out: pathlib.Path
    :param coarse: Coarse source bound to the target grid.
    :type coarse: CoarseOnTargetGrid
    :param geopotential: Coarse surface geopotential [m^2 s^-2].
    :type geopotential: xarray.DataArray
    :returns: Path to the static store.
    :rtype: pathlib.Path
    """
    target = out / "static.zarr"
    if target.exists():
        return target
    fine = xr.open_zarr(target_dir / "static.zarr").load()
    ds = xr.Dataset(
        {
            "cerra_orography": fine.orography.astype("f4"),
            "land_sea_mask": fine.land_sea_mask.astype("f4"),
            "era5_orography": (("y", "x"), coarse.surface_height(geopotential)),
        },
        coords={"lat": fine.lat, "lon": fine.lon},
    )
    for name in ("cerra_orography", "era5_orography"):
        ds[name].attrs["units"] = "m"
    out.mkdir(parents=True, exist_ok=True)
    _clean(ds).to_zarr(target, mode="w")
    difference = ds.cerra_orography - ds.era5_orography
    logger.info(
        "static fields: fine orography %.0f..%.0f m, coarse %.0f..%.0f m, difference %.0f..%.0f m",
        float(ds.cerra_orography.min()), float(ds.cerra_orography.max()),
        float(ds.era5_orography.min()), float(ds.era5_orography.max()),
        float(difference.min()), float(difference.max()),
    )
    return target


def consolidate(cfg: Config, out: Path) -> dict:
    """Merge the months, compute normalisation statistics and draw a check figure.

    Statistics come from the training years only, so the validation years stay
    untouched.

    :param cfg: Pipeline configuration.
    :type cfg: downscale.config.Config
    :param out: Output directory.
    :type out: pathlib.Path
    :returns: Normalisation statistics and sanity checks, also written to ``stats.json``.
    :rtype: dict
    :raises FileNotFoundError: If any month is still missing.
    :raises ValueError: If the consolidated times are not strictly increasing.
    """
    first, last = int(cfg.target.years[0]), int(cfg.target.years[1])
    tags = [f"{y}{m:02d}" for y in range(first, last + 1) for m in range(1, 13)]
    paths = [out / "months" / f"pairs_{t}.zarr" for t in tags]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} month(s) not built yet, e.g. {missing[:3]}")
    ds = xr.open_mfdataset([str(p) for p in paths], engine="zarr", combine="nested", concat_dim="time")
    ds = ds.chunk({"time": 124, "y": -1, "x": -1})
    times = pd.DatetimeIndex(ds.time.values)
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("consolidated times are not strictly increasing")
    static = xr.open_zarr(out / "static.zarr").load()

    train_first, train_last = int(cfg.target.train_years[0]), int(cfg.target.train_years[1])
    train = ds.sel(time=slice(f"{train_first}-01-01", f"{train_last}-12-31T23:59"))
    residual = train.fine_t2m - train.coarse_t2m
    height_diff = static.cerra_orography - static.era5_orography
    land = land_mask(static)
    if cfg.training.land_only and land is not None:
        keep = xr.DataArray(land, dims=("y", "x"))
        train = train.where(keep)
        residual = residual.where(keep)
        logger.info("statistics computed over land only (%.0f%% of the grid)", 100 * land.mean())
    stats = {
        "train_years": [train_first, train_last],
        "val_years": cfg.val_years,
        "n_train_steps": int(train.sizes["time"]),
        "era5_t2m": {"mean": float(train.coarse_t2m.mean().compute()), "std": float(train.coarse_t2m.std().compute())},
        "residual": {"mean": float(residual.mean().compute()), "std": float(residual.std().compute())},
        "cerra_orography": {"mean": float(static.cerra_orography.mean()), "std": float(static.cerra_orography.std())},
        "orography_diff": {"mean": float(height_diff.mean()), "std": float(height_diff.std())},
    }
    mean_residual = residual.mean("time").compute().values.ravel()
    heights = height_diff.values.ravel()
    valid = np.isfinite(mean_residual) & np.isfinite(heights)
    slope = float(np.polyfit(heights[valid], mean_residual[valid], 1)[0])
    stats["land_only"] = bool(cfg.training.land_only and land is not None)
    stats["checks"] = {
        "residual_vs_height_slope_K_per_km": round(slope * 1000, 2),
        "baseline_rmse_train_K": round(float(np.sqrt((residual ** 2).mean().compute())), 3),
    }
    ds.attrs = {
        "source": "coarse source interpolated onto the target grid, with the target",
        "years": f"{first}-{last}",
        "train_years": f"{train_first}-{train_last}",
    }
    _clean(ds).to_zarr(out / "pairs.zarr", mode="w")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(
        "consolidated %d steps; residual mean %.2f K, std %.2f K; slope against height %.2f K/km",
        ds.sizes["time"], stats["residual"]["mean"], stats["residual"]["std"], slope * 1000,
    )
    check_figure(out)
    return stats


def check_figure(out: Path) -> None:
    """Draw the coarse field, the target, their difference and its dependence on height.

    :param out: Output directory holding ``pairs.zarr`` and ``static.zarr``.
    :type out: pathlib.Path
    :returns: ``None``
    :rtype: None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds, static = xr.open_zarr(out / "pairs.zarr"), xr.open_zarr(out / "static.zarr")
    midday = ds.time.dt.hour == 12
    index = int(ds.fine_t2m.where(midday).mean(("y", "x")).argmax().compute())
    coarse = ds.coarse_t2m.isel(time=index).load() - 273.15
    fine = ds.fine_t2m.isel(time=index).load() - 273.15
    low, high = float(np.nanpercentile(fine, 1)), float(np.nanpercentile(fine, 99))
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)
    for axis, field, title in (
        (ax[0], coarse, "coarse, interpolated onto the target grid"),
        (ax[1], fine, "target"),
    ):
        mesh = axis.pcolormesh(static.lon, static.lat, field, cmap="RdYlBu_r", vmin=low, vmax=high, shading="auto")
        axis.set_title(title)
        fig.colorbar(mesh, ax=axis, label="degC")
    difference = fine - coarse
    limit = float(np.nanpercentile(np.abs(difference), 99))
    mesh = ax[2].pcolormesh(static.lon, static.lat, difference, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
    ax[2].set_title("target minus coarse: what the downscaler must learn")
    fig.colorbar(mesh, ax=ax[2], label="K")
    heights = (static.cerra_orography - static.era5_orography).values.ravel()
    ax[3].scatter(heights, difference.values.ravel(), s=1, alpha=0.2)
    ax[3].plot(
        [heights.min(), heights.max()], [-0.0065 * heights.min(), -0.0065 * heights.max()],
        "k--", lw=1, label="-6.5 K/km",
    )
    ax[3].set_xlabel("target minus coarse surface height (m)")
    ax[3].set_ylabel("target minus coarse temperature (K)")
    ax[3].set_title("difference against height difference")
    ax[3].legend()
    mid_lat = float(np.mean(static.lat.values))
    for axis in ax[:3]:
        axis.set_aspect(1 / np.cos(np.deg2rad(mid_lat)))
    fig.suptitle(f"{pd.Timestamp(ds.time.values[index]):%Y-%m-%d %H UTC}")
    fig.savefig(out / "check.png", dpi=100)
    plt.close(fig)


def main(argv: Sequence[str] | None = None, source=None, geopotential=None) -> int:
    """Command-line entry point.

    :param argv: Argument list; ``None`` reads ``sys.argv``.
    :type argv: collections.abc.Sequence[str] | None
    :param source: Coarse data source; ``None`` builds the configured one.
    :type source: typing.Callable | None
    :param geopotential: Coarse surface geopotential; ``None`` reads it from the source.
    :type geopotential: xarray.DataArray | None
    :returns: Exit status, non-zero when a month reported problems.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--target", default=None, help="overrides paths.target")
    parser.add_argument("--out", default=None, help="overrides paths.pairs")
    parser.add_argument("--batch", type=int, default=None, help="overrides coarse.fetch_batch")
    parser.add_argument("--consolidate", action="store_true")
    parser.add_argument("--coarse-source", choices=["era5", "era5-arco"], default=None,
                        help="overrides coarse.source; era5-arco reads Google's public copy, no account needed")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    if args.coarse_source:
        cfg.coarse.source = args.coarse_source
    target_dir = Path(args.target or cfg.paths.target)
    out = Path(args.out or cfg.paths.pairs)

    if args.consolidate:
        print(json.dumps(consolidate(cfg, out), indent=2))
        return 0

    static = xr.open_zarr(target_dir / "static.zarr")
    if source is None or geopotential is None:
        if cfg.coarse.source not in ("era5", "era5-arco"):
            raise ValueError(f"coarse.source {cfg.coarse.source!r} is not supported; use 'era5' or 'era5-arco'")
        from downscale.sources import ERA5, open_arco_era5

        era5 = ERA5(dataset=open_arco_era5()) if cfg.coarse.source == "era5-arco" else ERA5()
        source = source or era5
        geopotential = geopotential if geopotential is not None else era5.surface_geopotential()
    coarse = CoarseOnTargetGrid(
        source, static.lat.values, static.lon.values, cfg.coarse.variable,
        batch=args.batch or cfg.coarse.fetch_batch,
    )
    do_static(target_dir, out, coarse, geopotential)

    first, last = int(cfg.target.years[0]), int(cfg.target.years[1])
    tags = [f"{y}{m:02d}" for y in range(first, last + 1) for m in range(1, 13)]
    pending, reports = [], []
    for tag in tags:
        if not (target_dir / "months" / f"t2m_{tag}.zarr").exists():
            pending.append(tag)
            continue
        report = do_month(target_dir, out, tag, coarse, static.lat.values, static.lon.values)
        if report is None:
            continue
        reports.append(report)
        logger.info(
            "%s: %d steps, correlation %.3f, mean difference %+.2f K, rmse %.2f K (%ss) %s",
            tag, report["steps"], report["corr"], report["mean_diff_K"], report["rmse_K"],
            report["fetch_seconds"], "problems: " + "; ".join(report["problems"]) if report["problems"] else "ok",
        )
    with open(out / "month_reports.jsonl", "a") as handle:
        for report in reports:
            handle.write(json.dumps(report) + "\n")

    done = len(list((out / "months").glob("pairs_*.zarr")))
    bad = [r["tag"] for r in reports if r["problems"]]
    print(f"\n{done}/{len(tags)} months built; {len(pending)} waiting for the target; {len(bad)} with problems {bad or ''}")
    print(
        "all months built: rerun with --consolidate" if done == len(tags)
        else "rerun later to pick up newly downloaded months"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())