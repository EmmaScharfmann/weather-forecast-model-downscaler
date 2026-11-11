"""Measure how much of the correction is learnable from the model's current inputs.

Usage::

    python scripts/check_predictability.py --config config_spatial.yaml

The downscaler sees the coarse field, static terrain and the calendar. Anything
it can learn about a given pixel is therefore a function of the time of year and
the time of day: a climatology of the correction. This script computes that
climatology from the training years and reports the share of the correction's
variance it accounts for, by elevation and by distance from the coast.

A high share means a well-posed problem the model should be able to solve. A low
share means the correction there is driven by something the model cannot see,
such as the day's wind direction, and no amount of training on these inputs will
recover it.

Outputs in the pairs directory: ``predictability.csv``, ``predictability.png``
and ``predictability.json``.
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
from downscale.diagnostics import distance_to_coast_km, grid_spacing_km, land_mask

logger = logging.getLogger("predictability")
COAST_EDGES = [0, 10, 25, 50, 100, 200, 1e6]


class ClimatologyAccumulator:
    """Accumulates per-pixel statistics of the correction, grouped by month and hour."""

    def __init__(self, shape: tuple[int, int], hours: Sequence[int]):
        """
        :param shape: Grid shape ``(y, x)``.
        :type shape: tuple[int, int]
        :param hours: Distinct UTC hours present in the data.
        :type hours: collections.abc.Sequence[int]
        """
        self.hours = {hour: index for index, hour in enumerate(sorted(hours))}
        self.n_groups = 12 * len(self.hours)
        self.group_sum = np.zeros((self.n_groups, *shape))
        self.group_count = np.zeros(self.n_groups)
        self.total_sum = np.zeros(shape)
        self.total_squared = np.zeros(shape)
        self.n = 0

    def add(self, residual: np.ndarray, times: pd.DatetimeIndex) -> None:
        """Accumulate a block of times.

        :param residual: Correction of shape ``(time, y, x)``.
        :type residual: numpy.ndarray
        :param times: Matching times.
        :type times: pandas.DatetimeIndex
        :returns: ``None``
        :rtype: None
        """
        group = (times.month.values - 1) * len(self.hours) + np.array(
            [self.hours[h] for h in times.hour.values]
        )
        for index in np.unique(group):
            chosen = group == index
            self.group_sum[index] += residual[chosen].sum(axis=0)
            self.group_count[index] += chosen.sum()
        self.total_sum += residual.sum(axis=0)
        self.total_squared += (residual ** 2).sum(axis=0)
        self.n += len(times)

    def explained_fraction(self) -> tuple[np.ndarray, np.ndarray]:
        """Share of the correction's variance accounted for by the climatology.

        :returns: Per-pixel explained fraction and the per-pixel variance of the correction.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        grand_mean = self.total_sum / self.n
        total = self.total_squared - self.n * grand_mean ** 2
        populated = self.group_count > 0
        group_mean = np.zeros_like(self.group_sum)
        group_mean[populated] = self.group_sum[populated] / self.group_count[populated, None, None]
        explained = (
            (self.group_count[:, None, None] * (group_mean - grand_mean) ** 2)[populated].sum(axis=0)
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = np.where(total > 0, explained / total, np.nan)
        return fraction, total / max(self.n - 1, 1)


def band_table(
    fraction: np.ndarray, variance: np.ndarray, values: np.ndarray, edges: Sequence[float],
    mask: np.ndarray, name: str, unit: str,
) -> pd.DataFrame:
    """Summarise the explained fraction within bands of a field.

    :param fraction: Per-pixel explained fraction.
    :type fraction: numpy.ndarray
    :param variance: Per-pixel variance of the correction [K^2].
    :type variance: numpy.ndarray
    :param values: Field to band, such as elevation or distance from the coast.
    :type values: numpy.ndarray
    :param edges: Band edges.
    :type edges: collections.abc.Sequence[float]
    :param mask: Points to include, normally land.
    :type mask: numpy.ndarray
    :param name: Name of the banded field, used in the output.
    :type name: str
    :param unit: Unit shown in the band labels.
    :type unit: str
    :returns: One row per band with the point count, the correction's standard deviation
        and the share of its variance the climatology explains.
    :rtype: pandas.DataFrame
    """
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        chosen = mask & (values >= low) & (values < high) & np.isfinite(fraction)
        if not chosen.any():
            continue
        label = f"{low:.0f}-{high:.0f} {unit}" if high < 1e5 else f"above {low:.0f} {unit}"
        rows.append({
            "grouping": name,
            "band": label,
            "n_points": int(chosen.sum()),
            "correction_sd_K": round(float(np.sqrt(variance[chosen].mean())), 3),
            "explained_fraction": round(float(fraction[chosen].mean()), 3),
        })
    return pd.DataFrame(rows)


def predictability_figure(table: pd.DataFrame, fraction: np.ndarray, lat2d, lon2d, path) -> None:
    """Plot the explained fraction as a map and by band.

    :param table: Output of :func:`band_table` for both groupings.
    :type table: pandas.DataFrame
    :param fraction: Per-pixel explained fraction.
    :type fraction: numpy.ndarray
    :param lat2d: Fine-grid latitudes.
    :type lat2d: numpy.ndarray
    :param lon2d: Fine-grid longitudes.
    :type lon2d: numpy.ndarray
    :param path: Destination image.
    :type path: str | pathlib.Path
    :returns: ``None``
    :rtype: None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5), constrained_layout=True)
    mesh = ax[0].pcolormesh(lon2d, lat2d, fraction, cmap="viridis", vmin=0, vmax=1, shading="auto")
    ax[0].set_aspect(1 / np.cos(np.deg2rad(float(np.mean(lat2d)))))
    ax[0].set_title("share of the correction explained by\nterrain and calendar alone")
    fig.colorbar(mesh, ax=ax[0], label="explained fraction")

    for axis, grouping, title in (
        (ax[1], "elevation", "by elevation"),
        (ax[2], "distance from the coast", "by distance from the coast"),
    ):
        subset = table[table.grouping == grouping]
        positions = np.arange(len(subset))
        axis.bar(positions, subset.explained_fraction, color="tab:blue", alpha=0.85)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [f"{band}\nsd {sd:.1f} K" for band, sd in zip(subset.band, subset.correction_sd_K)],
            fontsize=7, rotation=15,
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("explained fraction")
        axis.set_title(title)
        axis.grid(alpha=0.3, axis="y")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    :param argv: Argument list; ``None`` reads ``sys.argv``.
    :type argv: collections.abc.Sequence[str] | None
    :returns: Exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--pairs", default=None, help="overrides paths.pairs")
    parser.add_argument("--out", default=None, help="defaults to the pairs directory")
    parser.add_argument("--block", type=int, default=248, help="time steps read at once")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    pairs_dir = Path(args.pairs or cfg.paths.pairs)
    out = Path(args.out or pairs_dir)
    out.mkdir(parents=True, exist_ok=True)

    store = pairs_dir / "pairs.zarr"
    ds = xr.open_zarr(store) if store.exists() else xr.open_mfdataset(
        [str(p) for p in sorted((pairs_dir / "months").glob("pairs_*.zarr"))],
        engine="zarr", combine="nested", concat_dim="time",
    )
    first, last = int(cfg.target.train_years[0]), int(cfg.target.train_years[1])
    ds = ds.sel(time=slice(f"{first}-01-01", f"{last}-12-31T23:59"))
    times = pd.DatetimeIndex(ds.time.values)
    static = xr.open_zarr(pairs_dir / "static.zarr").load()
    shape = (ds.sizes["y"], ds.sizes["x"])
    logger.info("%d training time(s) on a %dx%d grid", len(times), *shape)

    accumulator = ClimatologyAccumulator(shape, sorted(set(times.hour)))
    for start in range(0, len(times), args.block):
        block = ds.isel(time=slice(start, start + args.block))
        residual = (block.fine_t2m.values - block.coarse_t2m.values).astype("f8")
        accumulator.add(residual, pd.DatetimeIndex(block.time.values))
        logger.info("accumulated %d/%d", min(start + args.block, len(times)), len(times))
    fraction, variance = accumulator.explained_fraction()

    mask = land_mask(static)
    if mask is None:
        mask = np.ones(shape, bool)
    spacing = grid_spacing_km(static.lat.values, static.lon.values)
    distance = distance_to_coast_km(mask, spacing)
    table = pd.concat([
        band_table(
            fraction, variance, static.cerra_orography.values,
            list(elevation_edges(cfg.evaluation.elevation_edges)), mask, "elevation", "m",
        ),
        band_table(fraction, variance, distance, COAST_EDGES, mask, "distance from the coast", "km"),
    ], ignore_index=True)
    table.to_csv(out / "predictability.csv", index=False)
    predictability_figure(table, np.where(mask, fraction, np.nan), static.lat.values, static.lon.values,
                          out / "predictability.png")

    coastal = mask & (distance < 25)
    inland = mask & (distance >= 25)
    high = mask & (static.cerra_orography.values >= 1000)
    summary = {
        "train_years": [first, last],
        "n_times": len(times),
        "explained_fraction_land": round(float(np.nanmean(fraction[mask])), 3),
        "explained_fraction_within_25km_of_coast": round(float(np.nanmean(fraction[coastal])), 3),
        "explained_fraction_inland": round(float(np.nanmean(fraction[inland])), 3),
        "explained_fraction_above_1000m": round(float(np.nanmean(fraction[high])), 3),
        "correction_sd_within_25km_of_coast_K": round(float(np.sqrt(variance[coastal].mean())), 3),
        "correction_sd_above_1000m_K": round(float(np.sqrt(variance[high].mean())), 3),
    }
    (out / "predictability.json").write_text(json.dumps(summary, indent=2))

    print("\nhow much of the correction is learnable from terrain and calendar alone")
    print(table.to_string(index=False))
    print("\nsummary")
    for key, value in summary.items():
        print(f"  {key:44s} {value}")
    coastal_share = summary["explained_fraction_within_25km_of_coast"]
    high_share = summary["explained_fraction_above_1000m"]
    if coastal_share < 0.5 * high_share:
        print(
            f"\nnear the coast only {100 * coastal_share:.0f}% of the correction is predictable from these "
            f"inputs, against {100 * high_share:.0f}% on high ground: the coastal correction depends on "
            "something the model cannot see, most likely the day's wind"
        )
    print(f"\nwrote {out}/predictability.csv, predictability.png, predictability.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())