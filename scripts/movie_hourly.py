"""Animate an hourly downscaled forecast over the region.

Usage::

    python scripts/movie_hourly.py --run outputs/hourly --name aifs-ens_20250810T0000.zarr \\
        --out outputs/hourly/movie.mp4 --stations

Reads the three stores ``downscale_hourly.py`` wrote for one forecast and draws, at every hour, the
coarse forecast, the spatial model on the coarse forecast interpolated in time, and the hourly model,
for one member and sample. ``--stations`` adds a panel of the observed station values. Under the maps
a strip follows one point through the whole forecast.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib import animation  # noqa: E402

from downscale.config import load_config  # noqa: E402

logger = logging.getLogger("movie")
PARIS = (48.85, 2.35)


def nearest(lat: np.ndarray, lon: np.ndarray, point: tuple[float, float]) -> tuple[int, int]:
    """Index of the grid point closest to a location.

    :param lat: Latitudes over ``(y, x)``.
    :type lat: numpy.ndarray
    :param lon: Longitudes over ``(y, x)``.
    :type lon: numpy.ndarray
    :param point: ``(lat, lon)``.
    :type point: tuple[float, float]
    :returns: ``(y, x)``.
    :rtype: tuple[int, int]
    """
    distance = (lat - point[0]) ** 2 + ((lon - point[1]) * np.cos(np.radians(point[0]))) ** 2
    return np.unravel_index(int(np.argmin(distance)), lat.shape)


def frames(run: Path, name: str, variable: str, member: int, sample: int) -> dict:
    """Load the fields of one forecast, in degrees Celsius.

    :param run: Output directory of ``downscale_hourly.py``.
    :type run: pathlib.Path
    :param name: Store name.
    :type name: str
    :param variable: Variable of the coarse store.
    :type variable: str
    :param member: Ensemble member.
    :type member: int
    :param sample: Sample of that member.
    :type sample: int
    :returns: ``init``, ``valid``, ``coarse`` with its ``coarse_lat`` and ``coarse_lon``, ``spatial``, ``hourly``,
        ``lat``, ``lon`` and ``land``.
    :rtype: dict
    """
    coarse = xr.open_zarr(run / "forecasts" / name)[variable].isel(time=0, ensemble=member)
    spatial = xr.open_zarr(run / "spatial" / name)
    hourly = xr.open_zarr(run / "hourly" / name)
    pick = dict(time=0, ensemble=member, sample=sample)
    return {
        "init": pd.Timestamp(hourly["time"].values[0]),
        "valid": pd.DatetimeIndex(hourly["valid_time"].isel(time=0).values),
        "coarse": coarse.values - 273.15,
        "coarse_lat": coarse["lat"].values,
        "coarse_lon": coarse["lon"].values,
        "spatial": spatial["t2m_ds"].isel(pick).values - 273.15,
        "hourly": hourly["t2m_ds"].isel(pick).values - 273.15,
        "lat": hourly["lat"].values,
        "lon": hourly["lon"].values,
        "land": hourly["land_sea_mask"].values > 0.5,
    }


def observed(cfg, valid: pd.DatetimeIndex) -> xr.Dataset:
    """Station observations at every hour of the forecast.

    :param cfg: Configuration.
    :type cfg: downscale.config.Config
    :param valid: Valid times.
    :type valid: pandas.DatetimeIndex
    :returns: Stations with ``t2m`` in degrees Celsius over ``(station, time)``.
    :rtype: xarray.Dataset
    """
    from downscale.stations import MeteoFranceHourly, QCConfig, at_times, load_stations

    stations, _ = load_stations(
        MeteoFranceHourly(), cfg.region_box, valid[0], valid[-1], variables=("t2m",),
        cache_dir=cfg.paths.stations_cache, qc=QCConfig(min_completeness=cfg.stations.min_completeness),
        active=cfg.stations.active,
    )
    stations = at_times(stations, valid)
    return stations.assign(t2m=stations["t2m"] - 273.15)


def coarse_land(coarse_lat: np.ndarray, coarse_lon: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                land: np.ndarray) -> np.ndarray:
    """Land on the coarse grid, from the nearest point of the fine grid; points off the fine grid are not land.

    :param coarse_lat: Coarse latitudes.
    :type coarse_lat: numpy.ndarray
    :param coarse_lon: Coarse longitudes.
    :type coarse_lon: numpy.ndarray
    :param lat: Fine latitudes over ``(y, x)``.
    :type lat: numpy.ndarray
    :param lon: Fine longitudes over ``(y, x)``.
    :type lon: numpy.ndarray
    :param land: Fine land mask.
    :type land: numpy.ndarray
    :returns: Coarse land mask over ``(lat, lon)``.
    :rtype: numpy.ndarray
    """
    from scipy.spatial import cKDTree

    scale = np.cos(np.radians(lat.mean()))
    tree = cKDTree(np.column_stack([lat.ravel(), lon.ravel() * scale]))
    grid_lat, grid_lon = np.meshgrid(coarse_lat, coarse_lon, indexing="ij")
    distance, index = tree.query(np.column_stack([grid_lat.ravel(), grid_lon.ravel() * scale]))
    # Fine points are ~0.05 degrees apart: anything farther than that is off the fine grid.
    return (land.ravel()[index] & (distance < 0.1)).reshape(grid_lat.shape)


def render(
    data: dict, stations: xr.Dataset | None, out: Path, fps: int, point=PARIS, sea: bool = False
) -> Path:
    """Write the animation.

    :param data: Fields from :func:`frames`.
    :type data: dict
    :param stations: Observations from :func:`observed`, or ``None``.
    :type stations: xarray.Dataset | None
    :param out: Output file, ``.mp4`` or ``.gif``.
    :type out: pathlib.Path
    :param fps: Frames per second.
    :type fps: int
    :param point: ``(lat, lon)`` followed in the strip under the maps.
    :type point: tuple[float, float]
    :param sea: Also draw the fields over the sea, where the models are not trained.
    :type sea: bool
    :returns: The output file.
    :rtype: pathlib.Path
    """
    valid, lat, lon, land = data["valid"], data["lat"], data["lon"], data["land"]
    vmin, vmax = np.percentile(data["hourly"][:, land], [1, 99.5])
    # The loss only covers land, so the downscaled fields mean nothing over the sea; hide it everywhere alike.
    shown = (lambda f, mask=land: f) if sea else (lambda f, mask=land: np.where(mask, f, np.nan))
    coarse_mask = coarse_land(data["coarse_lat"], data["coarse_lon"], lat, lon, land)
    panels = [
        ("AIFS-ENS 0.25°, linear in time", data["coarse_lon"], data["coarse_lat"], shown(data["coarse"], coarse_mask)),
        ("spatial model, linear in time", lon, lat, shown(data["spatial"])),
        ("hourly model", lon, lat, shown(data["hourly"])),
    ]
    n_maps = len(panels) + (stations is not None)
    fig = plt.figure(figsize=(4.6 * n_maps, 6.6))
    grid = fig.add_gridspec(2, n_maps, height_ratios=[3.2, 1], hspace=0.28, wspace=0.05)
    extent = (lon.min(), lon.max(), lat.min(), lat.max())
    aspect = 1 / np.cos(np.radians(lat.mean()))
    meshes, axes = [], []
    for k, (title, x, y, field) in enumerate(panels):
        ax = fig.add_subplot(grid[0, k], facecolor="0.85")
        meshes.append(ax.pcolormesh(x, y, field[0], cmap="RdYlBu_r", vmin=vmin, vmax=vmax, shading="auto"))
        ax.set_title(title, fontsize=10)
        axes.append(ax)
    if stations is not None:
        ax = fig.add_subplot(grid[0, len(panels)])
        dots = ax.scatter(
            stations["longitude"], stations["latitude"], c=stations["t2m"].isel(time=0), s=5,
            cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
        )
        ax.set_title(f"{stations.sizes['station']} stations, observed", fontsize=10)
        axes.append(ax)
    for ax in axes:
        ax.contour(lon, lat, land.astype(float), levels=[0.5], colors="k", linewidths=0.4)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_aspect(aspect)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(meshes[-1], ax=axes, shrink=0.8, pad=0.01, label="2 m temperature (°C)")

    y, x = nearest(lat, lon, point)
    cy = int(np.argmin(np.abs(data["coarse_lat"] - point[0])))
    cx = int(np.argmin(np.abs(data["coarse_lon"] - point[1])))
    for ax in axes:
        ax.plot(lon[y, x], lat[y, x], "k+", ms=8)
    strip = fig.add_subplot(grid[1, :])
    strip.plot(valid, data["coarse"][:, cy, cx], color="0.5", label="AIFS-ENS, linear in time")
    strip.plot(valid, data["spatial"][:, y, x], color="tab:blue", label="spatial model")
    strip.plot(valid, data["hourly"][:, y, x], color="tab:red", label="hourly model")
    if stations is not None:
        distance = (stations["latitude"] - point[0]) ** 2 + (stations["longitude"] - point[1]) ** 2
        station = stations.isel(station=int(np.argmin(distance.values)))
        strip.plot(valid, station["t2m"], "k.", ms=3, label=f"station {station['station'].item()}")
    cursor = strip.axvline(valid[0], color="k", lw=0.8)
    strip.set_ylabel("°C")
    strip.set_title(f"at {point[0]:.2f}°N {point[1]:.2f}°E (+)", fontsize=9)
    strip.legend(fontsize=8, ncol=4, loc="upper left")
    strip.grid(alpha=0.3)
    title = fig.suptitle("")

    def update(t: int):
        for mesh, (_, _, _, field) in zip(meshes, panels):
            mesh.set_array(field[t].ravel())
        if stations is not None:
            dots.set_array(stations["t2m"].isel(time=t).values)
        cursor.set_xdata([valid[t], valid[t]])
        lead = int((valid[t] - data["init"]) / pd.Timedelta(hours=1))
        title.set_text(f"{valid[t]:%a %d %b %Y %H:00} UTC   (+{lead}h)")
        return [*meshes, cursor, title]

    out.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(fig, update, frames=len(valid), blit=False)
    writer = animation.PillowWriter(fps=fps) if out.suffix == ".gif" else animation.FFMpegWriter(fps=fps, bitrate=4000)
    movie.save(out, writer=writer, dpi=110)
    plt.close(fig)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    :param argv: Argument list; ``None`` reads ``sys.argv``.
    :type argv: collections.abc.Sequence[str] | None
    :returns: Exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run", default="outputs/hourly", help="output directory of downscale_hourly.py")
    parser.add_argument("--name", default=None, help="store to animate; the first one if not given")
    parser.add_argument("--out", default=None, help="mp4 or gif; <run>/<name>.mp4 if not given")
    parser.add_argument("--member", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--hours", type=int, default=None, help="only the first so many hours")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--stations", action="store_true", help="add the observed station values")
    parser.add_argument("--show-sea", action="store_true", help="draw the fields over the sea too")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    run = Path(args.run)
    name = args.name or sorted(p.name for p in (run / "hourly").glob("*.zarr"))[0]
    data = frames(run, name, cfg.coarse.variable, args.member, args.sample)
    if args.hours:
        data = {k: v[: args.hours] if k in ("valid", "coarse", "spatial", "hourly") else v for k, v in data.items()}
    stations = observed(cfg, data["valid"]) if args.stations else None
    out = Path(args.out or run / (Path(name).stem + ".mp4"))
    render(data, stations, out, args.fps, sea=args.show_sea)
    print(f"{len(data['valid'])} frames written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
