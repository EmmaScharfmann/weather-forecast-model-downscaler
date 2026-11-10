"""Train the spatial or the temporal (hourly) downscaler and validate it against plain interpolation.

Usage::

    python scripts/train_downscaler.py --config config_spatial.yaml --out runs/crps
    python scripts/train_downscaler.py --config config_spatial.yaml --out runs/mse --loss mse
    python scripts/train_downscaler.py --config config_temporal.yaml --temporal --out runs/hourly

Validation scores are in kelvin against the target reanalysis: the root mean
squared error of the ensemble mean, the fair continuous ranked probability
score, and the spread-skill ratio, which is one when the ensemble spread
matches its error.

``--temporal`` trains the model that fills the hours between 6-hourly coarse steps. Its pairs must
be hourly: build them with ``build_pairs.py`` from a target downloaded with ``config_temporal.yaml``.
Every hour is given the coarse fields at the two ends of its 6-hour window (00, 06, 12 and 18 UTC),
so only the coarse steps a forecast actually has are used. The network starts from the spatial
model's weights (``temporal.spatial_model``) with its two new channels switched off, so before
training it is the spatial model applied to the straight line between the two coarse steps.
Validation compares three ways of filling an hour, for each hour of the window:

- ``line``: the coarse field interpolated in time, the baseline the hourly model corrects;
- ``spatial``: the spatial model applied to that line;
- ``hourly``: this model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr

from downscale import downscaler as dsc
from downscale import hourly as hr
from downscale.config import Config, load_config
from downscale.constants import elevation_edges
from downscale.constants import COAST_CHANNEL
from downscale.diagnostics import (
    distance_to_coast_km,
    grid_spacing_km,
    land_mask,
    maritime_influence,
)

logger = logging.getLogger("train")
METHODS = ("line", "spatial", "hourly")
SHOWN = {"line": "coarse, linear in time", "spatial": "spatial, linear in time"}


def open_pairs(pairs_dir: Path) -> xr.Dataset:
    """Open the training pairs, falling back to the months built so far.

    :param pairs_dir: Directory written by ``build_pairs.py``.
    :type pairs_dir: pathlib.Path
    :returns: Dataset with ``coarse_t2m`` and ``fine_t2m`` over ``(time, y, x)``.
    :rtype: xarray.Dataset
    :raises FileNotFoundError: If no pairs exist at all.
    """
    consolidated = pairs_dir / "pairs.zarr"
    if consolidated.exists():
        return xr.open_zarr(consolidated)
    months = sorted((pairs_dir / "months").glob("pairs_*.zarr"))
    if not months:
        raise FileNotFoundError(f"no training pairs in {pairs_dir}")
    logger.warning("pairs not consolidated: using the %d month(s) built so far", len(months))
    return xr.open_mfdataset([str(m) for m in months], engine="zarr", combine="nested", concat_dim="time")


def split_times(
    times: pd.DatetimeIndex, val_years: Sequence[int], val_start: str | None, val_end: str | None
) -> tuple[np.ndarray, np.ndarray]:
    """Split times into training and validation periods.

    :param times: All available times.
    :type times: pandas.DatetimeIndex
    :param val_years: Validation years from the configuration.
    :type val_years: collections.abc.Sequence[int]
    :param val_start: Explicit validation start, overriding the years.
    :type val_start: str | None
    :param val_end: Explicit validation end.
    :type val_end: str | None
    :returns: Boolean masks for the training and validation periods.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    :raises ValueError: If either period is empty.
    """
    if val_start:
        start = pd.Timestamp(val_start)
        end = pd.Timestamp(val_end) if val_end else times.max()
    elif val_years:
        start, end = pd.Timestamp(f"{min(val_years)}-01-01"), pd.Timestamp(f"{max(val_years)}-12-31T23:59")
    else:
        start, end = times[int(len(times) * 0.8)], times.max()
        logger.warning("no validation period configured: using the last 20%% of times from %s", start)
    validation = (times >= start) & (times <= end)
    training = times < start
    if not training.any() or not validation.any():
        raise ValueError(f"empty split: {training.sum()} training and {validation.sum()} validation steps")
    return training, validation


def get_stats(stats_file: Path, coarse_train: np.ndarray, fine_train: np.ndarray, static: dict) -> dict:
    """Read the normalisation statistics, computing them if absent.

    :param stats_file: ``stats.json`` written by ``build_pairs.py``.
    :type stats_file: pathlib.Path
    :param coarse_train: Coarse training values [K].
    :type coarse_train: numpy.ndarray
    :param fine_train: Target training values [K].
    :type fine_train: numpy.ndarray
    :param static: Static fields on the target grid.
    :type static: dict
    :returns: Statistics for every normalised channel.
    :rtype: dict
    """
    if stats_file.exists():
        stats = json.loads(stats_file.read_text())
    else:
        logger.warning("stats.json missing: computing statistics from the training slice")
        residual = fine_train - coarse_train
        height_diff = static["cerra_orography"] - static["era5_orography"]
        stats = {
            "era5_t2m": {"mean": float(coarse_train.mean()), "std": float(coarse_train.std())},
            "residual": {"mean": float(residual.mean()), "std": float(residual.std())},
            "cerra_orography": {
                "mean": float(static["cerra_orography"].mean()), "std": float(static["cerra_orography"].std()),
            },
            "orography_diff": {"mean": float(height_diff.mean()), "std": float(height_diff.std())},
        }
    for key in ("lat", "lon", COAST_CHANNEL):
        if key in static:
            stats[key] = {"mean": float(static[key].mean()), "std": float(static[key].std())}
    return stats


class Scores:
    """Accumulates squared error, score and variance sums per group."""

    def __init__(self):
        self.sums: dict[str, np.ndarray] = {}

    def add(self, key: str, squared_error: float, crps: float, variance: float, n: float) -> None:
        """Add one batch's contribution to a group.

        :param key: Group name.
        :type key: str
        :param squared_error: Sum of squared errors of the ensemble mean.
        :type squared_error: float
        :param crps: Sum of per-pixel scores.
        :type crps: float
        :param variance: Sum of ensemble variances.
        :type variance: float
        :param n: Number of values contributing.
        :type n: float
        :returns: ``None``
        :rtype: None
        """
        self.sums.setdefault(key, np.zeros(4))[:] += [squared_error, crps, variance, n]

    def table(self) -> dict:
        """Turn the sums into scores.

        :returns: Group names mapped to ``rmse``, ``crps``, ``spread``, ``ssr`` and ``n``.
        :rtype: dict
        """
        out = {}
        for key, (squared_error, crps, variance, n) in self.sums.items():
            rmse, spread = np.sqrt(squared_error / n), np.sqrt(variance / n)
            out[key] = {
                "rmse": round(float(rmse), 4),
                "crps": round(float(crps / n), 4),
                "spread": round(float(spread), 4),
                "ssr": round(float(spread / rmse), 3) if rmse > 0 else None,
                "n": int(n),
            }
        return out


def crps_per_pixel(ensemble: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Fair continuous ranked probability score for every pixel.

    :param ensemble: Ensemble of shape ``(batch, members, height, width)``.
    :type ensemble: numpy.ndarray
    :param truth: Truth of shape ``(batch, height, width)``.
    :type truth: numpy.ndarray
    :returns: Score of shape ``(batch, height, width)``.
    :rtype: numpy.ndarray
    """
    m = ensemble.shape[1]
    skill = np.abs(ensemble - truth[:, None]).mean(1)
    if m == 1:
        return skill
    spread = np.abs(ensemble[:, :, None] - ensemble[:, None]).sum((1, 2)) / (2 * m * (m - 1))
    return skill - spread


@torch.no_grad()
def validate(
    net, coarse: np.ndarray, fine: np.ndarray, times, static_norm: np.ndarray, stats: dict,
    members: int, device: str, groups: dict | None = None, batch: int = 8,
) -> tuple[dict, dict]:
    """Score the model and plain interpolation over the validation period.

    :param net: Network to evaluate.
    :type net: downscale.downscaler.DownscalerNet
    :param coarse: Coarse validation fields [K].
    :type coarse: numpy.ndarray
    :param fine: Target validation fields [K].
    :type fine: numpy.ndarray
    :param times: Validation times.
    :type times: numpy.ndarray
    :param static_norm: Normalised static channels.
    :type static_norm: numpy.ndarray
    :param stats: Normalisation statistics.
    :type stats: dict
    :param members: Ensemble members to draw.
    :type members: int
    :param device: Torch device.
    :type device: str
    :param groups: Named boolean masks over the grid; ``None`` scores all pixels together.
    :type groups: dict | None
    :param batch: Time steps per forward pass.
    :type batch: int
    :returns: Model scores and interpolation scores, keyed by group.
    :rtype: tuple[dict, dict]
    """
    net.eval()
    model_scores, baseline_scores = Scores(), Scores()
    hours = pd.DatetimeIndex(times).hour.values
    for i in range(0, len(coarse), batch):
        coarse_batch, truth = coarse[i:i + batch], fine[i:i + batch]
        x = torch.from_numpy(dsc.make_inputs(coarse_batch, times[i:i + batch], static_norm, stats)).to(device)
        ensemble = dsc.to_kelvin(coarse_batch, net(x, members).cpu().numpy(), stats)
        mean = ensemble.mean(1)
        squared_error, crps = (mean - truth) ** 2, crps_per_pixel(ensemble, truth)
        variance = ensemble.var(1, ddof=1) if ensemble.shape[1] > 1 else np.zeros_like(mean)
        base_squared, base_absolute = (coarse_batch - truth) ** 2, np.abs(coarse_batch - truth)
        selection = {"all": np.ones(truth.shape[1:], bool)}
        if groups is not None:
            selection.update(groups)
        for name, mask in selection.items():
            n = mask.sum() * len(truth)
            model_scores.add(name, squared_error[:, mask].sum(), crps[:, mask].sum(), variance[:, mask].sum(), n)
            baseline_scores.add(name, base_squared[:, mask].sum(), base_absolute[:, mask].sum(), 0.0, n)
            if groups is not None and name == "land":
                for hour in np.unique(hours[i:i + batch]):
                    chosen = hours[i:i + batch] == hour
                    n_hour = mask.sum() * chosen.sum()
                    key = f"land_{hour:02d}UTC"
                    model_scores.add(
                        key, squared_error[chosen][:, mask].sum(), crps[chosen][:, mask].sum(),
                        variance[chosen][:, mask].sum(), n_hour,
                    )
                    baseline_scores.add(
                        key, base_squared[chosen][:, mask].sum(), base_absolute[chosen][:, mask].sum(), 0.0, n_hour,
                    )
    net.train()
    return model_scores.table(), baseline_scores.table()


def figures(out: Path, history: list[dict], model: dsc.Downscaler, coarse, fine, times, members: int) -> None:
    """Draw the training curves and a validation map.

    :param out: Run directory.
    :type out: pathlib.Path
    :param history: Per-epoch records.
    :type history: list[dict]
    :param model: Best model, used for the map.
    :type model: downscale.downscaler.Downscaler
    :param coarse: Coarse validation fields [K].
    :type coarse: numpy.ndarray
    :param fine: Target validation fields [K].
    :type fine: numpy.ndarray
    :param times: Validation times.
    :type times: numpy.ndarray
    :param members: Ensemble members to draw for the map.
    :type members: int
    :returns: ``None``
    :rtype: None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax[0].plot(epochs, [h["train_loss"] for h in history], marker="o")
    ax[0].set_title("training loss")
    ax[0].set_xlabel("epoch")
    ax[1].plot(epochs, [h["val"]["rmse"] for h in history], marker="o", label="model rmse")
    ax[1].plot(epochs, [h["val"]["crps"] for h in history], marker="s", label="model crps")
    ax[1].axhline(history[0]["baseline"]["rmse"], color="k", ls="--", label="interpolation rmse")
    ax[1].axhline(history[0]["baseline"]["crps"], color="gray", ls=":", label="interpolation mae")
    ax[1].set_title("validation against the target (K)")
    ax[1].set_xlabel("epoch")
    ax[1].legend(fontsize=8)
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)

    hours = pd.DatetimeIndex(times).hour
    candidates = np.flatnonzero(hours == 12) if (hours == 12).any() else np.arange(len(times))
    index = int(candidates[np.argmax(fine[candidates].mean((1, 2)))])
    ensemble = model.predict(coarse[index:index + 1], times[index:index + 1], members=max(members, 1))[0]
    panels = [
        ("interpolated coarse field", coarse[index]),
        ("target", fine[index]),
        ("model, member 1", ensemble[0]),
        ("model, ensemble mean", ensemble.mean(0)),
    ]
    low, high = np.percentile(fine[index], [1, 99]) - 273.15
    fig, ax = plt.subplots(1, 5, figsize=(24, 4.6), constrained_layout=True)
    for axis, (title, field) in zip(ax, panels):
        mesh = axis.pcolormesh(
            model.lon, model.lat, field - 273.15, cmap="RdYlBu_r", vmin=low, vmax=high, shading="auto"
        )
        axis.set_title(title)
        fig.colorbar(mesh, ax=axis, label="degC")
    spread = ensemble.std(0, ddof=1) if ensemble.shape[0] > 1 else np.zeros_like(ensemble[0])
    mesh = ax[4].pcolormesh(model.lon, model.lat, spread, cmap="magma", shading="auto")
    ax[4].set_title("ensemble spread")
    fig.colorbar(mesh, ax=ax[4], label="K")
    mid_lat = float(np.mean(model.lat))
    for axis in ax:
        axis.set_aspect(1 / np.cos(np.deg2rad(mid_lat)))
    fig.suptitle(f"validation, {pd.Timestamp(times[index]):%Y-%m-%d %H UTC}")
    fig.savefig(out / "val_maps.png", dpi=95)
    plt.close(fig)


def tendency_stats(coarse: np.ndarray, starts: np.ndarray, ends: np.ndarray, land: np.ndarray) -> dict:
    """Mean and standard deviation of the change over a window, over land.

    :param coarse: Coarse fields [K], ``(time, y, x)``.
    :type coarse: numpy.ndarray
    :param starts: Index of the start of each training window.
    :type starts: numpy.ndarray
    :param ends: Index of the end of each training window.
    :type ends: numpy.ndarray
    :param land: Land mask over the grid.
    :type land: numpy.ndarray
    :returns: ``{"mean": ..., "std": ...}``.
    :rtype: dict
    """
    windows = np.unique(np.stack([starts, ends], 1), axis=0)
    change = (coarse[windows[:, 1]] - coarse[windows[:, 0]])[:, land]
    return {"mean": float(change.mean()), "std": float(change.std())}


class Validator:
    """Scores the straight line, the spatial model and the hourly model on the same hours."""

    def __init__(self, coarse, fine, times, hours, starts, ends, spatial: dsc.Downscaler, land: np.ndarray,
                 members: int, device: str, batch: int = 8):
        """
        :param coarse: Coarse fields [K].
        :type coarse: numpy.ndarray
        :param fine: Target fields [K].
        :type fine: numpy.ndarray
        :param times: Times of the fields.
        :type times: numpy.ndarray
        :param hours: Index of each validation hour.
        :type hours: numpy.ndarray
        :param starts: Index of the start of its window.
        :type starts: numpy.ndarray
        :param ends: Index of the end of its window.
        :type ends: numpy.ndarray
        :param spatial: The spatial model.
        :type spatial: downscale.downscaler.Downscaler
        :param land: Land mask over the grid.
        :type land: numpy.ndarray
        :param members: Ensemble members to draw.
        :type members: int
        :param device: Torch device.
        :type device: str
        :param batch: Hours per forward pass.
        :type batch: int
        """
        self.coarse, self.fine, self.times = coarse, fine, times
        self.hours, self.starts, self.ends = hours, starts, ends
        self.spatial, self.land, self.members, self.device, self.batch = spatial, land, members, device, batch

    @torch.no_grad()
    def __call__(self, net: hr.HourlyNet, stats: dict, methods: Sequence[str] = METHODS, seed: int = 0) -> dict:
        """Score some methods over the validation hours.

        :param net: Hourly network.
        :type net: downscale.hourly.HourlyNet
        :param stats: Normalisation statistics of the hourly model.
        :type stats: dict
        :param methods: Methods to score, among :data:`METHODS`.
        :type methods: collections.abc.Sequence[str]
        :param seed: Seed of the noise, so that every epoch is scored on the same draws.
        :type seed: int
        :returns: For each method, scores over land and over land at each hour of the window.
        :rtype: dict
        """
        net.eval()
        torch.manual_seed(seed)
        scores = {name: Scores() for name in methods}
        for i in range(0, len(self.hours), self.batch):
            h, s, e = (a[i:i + self.batch] for a in (self.hours, self.starts, self.ends))
            times = self.times[h]
            offsets = hr.offsets_of(times)
            truth = self.fine[h]
            x, line = hr.make_hourly_inputs(
                self.coarse[s], self.coarse[e], offsets, times, self.spatial.static_norm, stats
            )
            ensembles = {"line": line[:, None]}
            if "hourly" in methods:
                pred = net(torch.from_numpy(x).to(self.device), self.members).cpu().numpy()
                ensembles["hourly"] = dsc.to_kelvin(line, pred, stats)
            if "spatial" in methods:
                ensembles["spatial"] = self.spatial.predict(line, times, members=self.members, batch=self.batch)
            for name in methods:
                ensemble = ensembles[name]
                mean = ensemble.mean(1)
                squared, crps = (mean - truth) ** 2, crps_per_pixel(ensemble, truth)
                variance = ensemble.var(1, ddof=1) if ensemble.shape[1] > 1 else np.zeros_like(mean)
                for key, chosen in [("land", np.ones(len(h), bool))] + [
                    (f"land_+{k}h", offsets == k) for k in np.unique(offsets)
                ]:
                    n = self.land.sum() * chosen.sum()
                    scores[name].add(
                        key, squared[chosen][:, self.land].sum(), crps[chosen][:, self.land].sum(),
                        variance[chosen][:, self.land].sum(), n,
                    )
        net.train()
        return {name: table.table() for name, table in scores.items()}


def temporal_figures(out: Path, report: dict, history: list[dict], series: dict) -> None:
    """Draw the scores by hour of the window, the training curves and a station-like time series.

    :param out: Run directory.
    :type out: pathlib.Path
    :param report: Final scores, per method.
    :type report: dict
    :param history: Per-epoch records.
    :type history: list[dict]
    :param series: Times and values at one land point for the time-series panel.
    :type series: dict
    :returns: ``None``
    :rtype: None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(22, 4.2), constrained_layout=True)
    offsets = sorted(int(k[len("land_+"):-1]) for k in report["hourly"] if k.startswith("land_+"))
    for name, style in zip(METHODS, ("k--", "C0-o", "C3-s")):
        if name not in report:
            continue
        for axis, metric in zip(ax[:2], ("rmse", "crps")):
            axis.plot(offsets, [report[name][f"land_+{k}h"][metric] for k in offsets], style, label=SHOWN.get(name, name))
    for axis, metric in zip(ax[:2], ("RMSE", "CRPS")):
        axis.set_title(f"{metric} over land against CERRA (K)")
        axis.set_xlabel("hours after the coarse step")
        axis.legend(fontsize=8)
    epochs = [h["epoch"] for h in history]
    ax[2].plot(epochs, [h["train_loss"] for h in history], marker="o")
    ax[2].set_title("training loss")
    ax[2].set_xlabel("epoch")
    t = pd.DatetimeIndex(series["times"])
    ax[3].plot(t, series["truth"] - 273.15, "k-", label="CERRA")
    ax[3].plot(t, series["line"] - 273.15, "k--", label=SHOWN["line"])
    ax[3].plot(t, series["hourly"] - 273.15, "C3-", label="hourly, mean")
    ax[3].plot(t[series["steps"]], series["truth"][series["steps"]] - 273.15, "ko", ms=4)
    ax[3].set_title(f"one land point, {series['where']}")
    ax[3].set_ylabel("degC")
    ax[3].legend(fontsize=8)
    ax[3].tick_params(axis="x", labelrotation=30)
    fig.savefig(out / "hourly_scores.png", dpi=100)
    plt.close(fig)


def point_series(model: hr.HourlyDownscaler, coarse, fine, times, hours, starts, ends, land, days: int = 3) -> dict:
    """The truth, the line and the hourly model's mean at a land point with a large but typical daily range.

    The point at the 90th percentile of the range is used rather than the largest, which tends to be a
    jump in the target rather than weather.

    :returns: Times and values over the first ``days`` days of the validation hours.
    :rtype: dict
    """
    h, s, e = hours[: 24 * days], starts[: 24 * days], ends[: 24 * days]
    truth = fine[h]
    daily_range = np.where(land, truth.max(0) - truth.min(0), np.nan)
    target = np.nanpercentile(daily_range, 90)
    y, x = np.unravel_index(int(np.nanargmin(np.abs(daily_range - target))), daily_range.shape)
    offsets = hr.offsets_of(times[h])
    pred = model.predict_hours(coarse[s], coarse[e], offsets, times[h], members=4)
    line = hr.temporal_interpolation(coarse[s], coarse[e], offsets)
    return {
        "times": times[h], "truth": truth[:, y, x], "line": line[:, y, x], "hourly": pred.mean(1)[:, y, x],
        "steps": offsets == 0,
        "where": f"{float(model.lat[y, x]):.2f}N {float(model.lon[y, x]):.2f}E",
    }


def train_spatial(cfg: Config, args: argparse.Namespace) -> int:
    """Train the spatial downscaler.

    :param cfg: Configuration, with the command-line values already applied.
    :type cfg: downscale.config.Config
    :param args: Parsed arguments, for the output, pairs, validation period and device.
    :type args: argparse.Namespace
    :returns: Exit status.
    :rtype: int
    """
    training = cfg.training
    torch.manual_seed(training.seed)
    rng = np.random.default_rng(training.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs_dir = Path(args.pairs or cfg.paths.pairs)

    ds = open_pairs(pairs_dir)
    static_ds = xr.open_zarr(pairs_dir / "static.zarr").load()
    static = {k: static_ds[k].values.astype("f4") for k in ("cerra_orography", "era5_orography", "land_sea_mask")}
    static["lat"] = static_ds.lat.values.astype("f4")
    static["lon"] = static_ds.lon.values.astype("f4")
    land = land_mask(static)
    if land is None:
        land = np.ones(static["lat"].shape, bool)
    if training.coast_distance:
        spacing = grid_spacing_km(static["lat"], static["lon"])
        distance = distance_to_coast_km(land, spacing)
        static[COAST_CHANNEL] = maritime_influence(distance, training.coast_decay_km)
        logger.info(
            "maritime influence added as an input channel: decay scale %.0f km, "
            "distances 0 to %.0f km, grid spacing %.2f km",
            training.coast_decay_km, float(distance.max()), spacing,
        )
    times = pd.DatetimeIndex(ds.time.values)
    train_mask, val_mask = split_times(times, cfg.val_years, args.val_start, args.val_end)
    started = time.time()
    coarse, fine = ds.coarse_t2m.values.astype("f4"), ds.fine_t2m.values.astype("f4")
    logger.info(
        "loaded %d steps (%d training, %d validation) on a %dx%d grid in %.0fs",
        len(times), train_mask.sum(), val_mask.sum(), coarse.shape[1], coarse.shape[2], time.time() - started,
    )
    stats = get_stats(pairs_dir / "stats.json", coarse[train_mask], fine[train_mask], static)
    static_norm = dsc.static_stack(static, stats, training.coast_distance)
    train_idx = np.flatnonzero(train_mask)
    val_coarse, val_fine = coarse[val_mask], fine[val_mask]
    val_times = times[val_mask].to_numpy()

    groups = {"land": land}
    edges = elevation_edges(cfg.evaluation.elevation_edges)
    for low, high in zip(edges[:-1], edges[1:]):
        groups[f"land_{int(low)}-{int(high)}m"] = (
            land & (static["cerra_orography"] >= low) & (static["cerra_orography"] < high)
        )
    groups = {k: v for k, v in groups.items() if v.any()}

    model_cfg = dsc.Config(
        loss=training.loss, base=training.base_channels, depth=training.depth,
        noise_channels=training.noise_channels if training.loss == "crps" else 0,
        upsample=training.upsample, noise_fine_fraction=training.noise_fine_fraction,
        noise_coarse_factor=training.noise_coarse_factor, coast_distance=training.coast_distance,
    )
    net = dsc.DownscalerNet(model_cfg).to(device)
    steps_per_epoch = len(train_idx) // training.batch_size
    if training.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, training.max_steps_per_epoch)
    optimiser = torch.optim.AdamW(net.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=training.learning_rate, total_steps=max(1, training.epochs * steps_per_epoch)
    )
    loss_fn = dsc.fair_crps if model_cfg.stochastic else dsc.mse
    train_members = training.members if model_cfg.stochastic else 1
    eval_members = training.eval_members if model_cfg.stochastic else 1
    loss_mask = torch.from_numpy(land).to(device) if training.land_only else None
    headline = "land" if training.land_only else "all"
    logger.info(
        "model: %s loss over %s, %s upsampling, %.0f%% full-resolution noise, %.2fM parameters, "
        "%d steps per epoch, device %s",
        training.loss, "land only" if training.land_only else "every grid point", training.upsample,
        100 * training.noise_fine_fraction, sum(p.numel() for p in net.parameters()) / 1e6,
        steps_per_epoch, device,
    )

    best, history = None, []
    for epoch in range(1, training.epochs + 1):
        started = time.time()
        order = rng.permutation(train_idx)
        losses = []
        for step in range(steps_per_epoch):
            idx = np.sort(order[step * training.batch_size:(step + 1) * training.batch_size])
            coarse_batch, fine_batch = coarse[idx], fine[idx]
            x = torch.from_numpy(
                dsc.make_inputs(coarse_batch, times[idx].to_numpy(), static_norm, stats)
            ).to(device)
            target = torch.from_numpy(dsc.residual_norm(coarse_batch, fine_batch, stats)).to(device)
            loss = loss_fn(net(x, train_members), target, loss_mask)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            losses.append(float(loss))
        model_scores, baseline_scores = validate(
            net, val_coarse, val_fine, val_times, static_norm, stats, eval_members, device,
            groups={"land": land},
        )
        record = {
            "epoch": epoch, "train_loss": float(np.mean(losses)), "scored_over": headline,
            "val": model_scores[headline], "baseline": baseline_scores[headline],
            "seconds": round(time.time() - started, 1),
        }
        history.append(record)
        logger.info(
            "epoch %d/%d  loss %.4f  val rmse %.3f K (interpolation %.3f)  crps %.3f K (interpolation %.3f)  "
            "ssr %s  over %s  [%.0fs]",
            epoch, training.epochs, record["train_loss"], model_scores[headline]["rmse"],
            baseline_scores[headline]["rmse"], model_scores[headline]["crps"],
            baseline_scores[headline]["crps"], model_scores[headline]["ssr"], headline, record["seconds"],
        )
        score = model_scores[headline]["crps"] if model_cfg.stochastic else model_scores[headline]["rmse"]
        if best is None or score < best:
            best = score
            dsc.Downscaler(net, stats, static, device).save(
                out / "model.pt", extra={"epoch": epoch, "land_only": bool(training.land_only)}
            )
        (out / "history.json").write_text(json.dumps(history, indent=2))

    best_model = dsc.Downscaler.load(out / "model.pt", device)
    model_scores, baseline_scores = validate(
        best_model.net, val_coarse, val_fine, val_times, static_norm, stats, eval_members, device, groups=groups
    )
    report = {
        "loss": training.loss,
        "land_only": bool(training.land_only),
        "scored_over": headline,
        "upsample": training.upsample,
        "noise_fine_fraction": training.noise_fine_fraction,
        "coast_distance": bool(training.coast_distance),
        "coast_decay_km": training.coast_decay_km,
        "val_period": [str(times[val_mask].min()), str(times[val_mask].max())],
        "best_epoch": torch.load(out / "model.pt", weights_only=False)["extra"]["epoch"],
        "model": model_scores,
        "interpolation": baseline_scores,
    }
    (out / "val_report.json").write_text(json.dumps(report, indent=2))
    figures(out, history, best_model, val_coarse, val_fine, val_times, eval_members)

    print("\nvalidation (K)        interp rmse   model rmse   model crps   ssr")
    for key in model_scores:
        print(
            f"  {key:20s} {baseline_scores[key]['rmse']:10.3f}  {model_scores[key]['rmse']:11.3f}  "
            f"{model_scores[key]['crps']:11.3f}   {model_scores[key]['ssr']}"
        )
    improved = model_scores[headline]["rmse"] < baseline_scores[headline]["rmse"]
    print(f"\n{'model beats interpolation' if improved else 'model does not beat interpolation'}; outputs in {out}/")
    return 0


def train_temporal(cfg: Config, out: Path, pairs_dir: Path, device: str | None = None) -> int:
    """Train the hourly downscaler, which fills the hours between 6-hourly coarse steps.

    :param cfg: Configuration, with the command-line values already applied; ``temporal`` holds the
        spatial model to start from, the validation period and the loss's block sizes.
    :type cfg: downscale.config.Config
    :param out: Run directory.
    :type out: pathlib.Path
    :param pairs_dir: Hourly pairs from ``build_pairs.py``.
    :type pairs_dir: pathlib.Path
    :param device: Torch device; ``None`` picks one.
    :type device: str | None
    :returns: Exit status.
    :rtype: int
    """
    training, temporal = cfg.training, cfg.temporal
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(training.seed)
    rng = np.random.default_rng(training.seed)
    device = device or hr.default_device()
    out.mkdir(parents=True, exist_ok=True)

    spatial = hr.load_spatial(temporal.spatial_model, device)
    ds = open_pairs(pairs_dir)
    times = ds.time.values
    coarse, fine = ds.coarse_t2m.values.astype("f4"), ds.fine_t2m.values.astype("f4")
    if coarse.shape[1:] != spatial.lat.shape:
        raise ValueError(f"pairs grid {coarse.shape[1:]} differs from the spatial model's {spatial.lat.shape}")
    static_path = pairs_dir / "static.zarr"
    if static_path.exists():
        pairs_static = xr.open_zarr(static_path)
        offset = max(float(np.abs(pairs_static.lat.values - spatial.lat).max()),
                     float(np.abs(pairs_static.lon.values - spatial.lon).max()))
        if offset > 1e-3:
            raise ValueError(f"pairs grid is {offset:.3f} degrees away from the spatial model's: crop the target alike")
    land = land_mask(spatial.static)
    land = np.ones(spatial.lat.shape, bool) if land is None else land

    hours, starts, ends = hr.window_samples(times)
    if not len(hours):
        raise ValueError("no hour has both ends of its window: are the pairs hourly?")
    train_mask, val_mask = split_times(pd.DatetimeIndex(times[hours]), cfg.val_years, temporal.val_start, temporal.val_end)
    train = tuple(a[train_mask] for a in (hours, starts, ends))
    val = tuple(a[val_mask] for a in (hours, starts, ends))
    logger.info(
        "%d hourly steps: %d training and %d validation hours with both ends of their window",
        len(times), len(train[0]), len(val[0]),
    )

    if temporal.init_hourly:
        start = hr.HourlyDownscaler.load(temporal.init_hourly, device)
        stats, net, model_cfg = start.stats, start.net, start.net.cfg
    else:
        stats = {**spatial.stats, "tendency": tendency_stats(coarse, train[1], train[2], land)}
        model_cfg = hr.HourlyConfig(**asdict(spatial.net.cfg))
        net = hr.HourlyNet(model_cfg)
        if not temporal.from_scratch:
            hr.init_from_spatial(net, temporal.spatial_model)
    net = net.to(device)
    static_norm = spatial.static_norm
    validator = Validator(coarse, fine, times, *val, spatial, land, training.eval_members, device)

    steps_per_epoch = max(1, len(train[0]) // training.batch_size)
    if training.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, training.max_steps_per_epoch)
    optimiser = torch.optim.AdamW(net.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=training.learning_rate, total_steps=max(1, training.epochs * steps_per_epoch)
    )
    loss_fn = dsc.fair_crps if model_cfg.stochastic else dsc.mse
    if temporal.multiscale and model_cfg.stochastic:
        def loss_fn(pred, target, mask):
            return dsc.multiscale_fair_crps(pred, target, mask, temporal.multiscale)
    members = training.members if model_cfg.stochastic else 1
    loss_mask = torch.from_numpy(land).to(device) if training.land_only else None
    logger.info(
        "model: %s loss, %.2fM parameters, %s, %d steps per epoch, device %s; tendency std %.2f K",
        training.loss, sum(p.numel() for p in net.parameters()) / 1e6,
        "from scratch" if temporal.from_scratch else f"started from {temporal.init_hourly or temporal.spatial_model}", steps_per_epoch, device,
        stats["tendency"]["std"],
    )

    start_scores = validator(net, stats)
    logger.info(
        "before training: crps line %.3f, spatial %.3f, hourly %.3f K",
        *(start_scores[m]["land"]["crps"] for m in METHODS),
    )
    best, history = None, []
    for epoch in range(1, training.epochs + 1):
        started = time.time()
        order = rng.permutation(len(train[0]))
        losses = []
        for step in range(steps_per_epoch):
            chosen = np.sort(order[step * training.batch_size:(step + 1) * training.batch_size])
            h, s, e = (a[chosen] for a in train)
            x, line = hr.make_hourly_inputs(coarse[s], coarse[e], hr.offsets_of(times[h]), times[h], static_norm, stats)
            target = torch.from_numpy(hr.hourly_residual(line, fine[h], stats)).to(device)
            loss = loss_fn(net(torch.from_numpy(x).to(device), members), target, loss_mask)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            losses.append(loss.item())
        scores = validator(net, stats, methods=("hourly",))["hourly"]["land"]
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": scores,
                  "seconds": round(time.time() - started, 1)}
        history.append(record)
        logger.info(
            "epoch %d/%d  loss %.4f  val rmse %.3f K  crps %.3f K  ssr %s  [%.0fs]",
            epoch, training.epochs, record["train_loss"], scores["rmse"], scores["crps"], scores["ssr"],
            record["seconds"],
        )
        score = scores["crps"] if model_cfg.stochastic else scores["rmse"]
        if best is None or score < best:
            best = score
            hr.HourlyDownscaler(net, stats, spatial.static, device).save(
                out / "model.pt", extra={"epoch": epoch, "spatial": str(temporal.spatial_model), "multiscale": list(temporal.multiscale)}
            )
        (out / "history.json").write_text(json.dumps(history, indent=2))

    model = hr.HourlyDownscaler.load(out / "model.pt", device)
    report = validator(model.net, stats)
    summary = {
        "val_period": [str(pd.Timestamp(times[val[0]].min())), str(pd.Timestamp(times[val[0]].max()))],
        "n_train_hours": int(len(train[0])),
        "n_val_hours": int(len(val[0])),
        "best_epoch": hr.safe_load(out / "model.pt")["extra"]["epoch"],
        "before_training": start_scores["hourly"]["land"],
        **report,
    }
    (out / "val_report.json").write_text(json.dumps(summary, indent=2))
    temporal_figures(out, report, history, point_series(model, coarse, fine, times, *val, land))

    keys = ["land"] + sorted(k for k in report["hourly"] if k.startswith("land_+"))
    print("\nvalidation over land (K)   line rmse  spatial rmse  hourly rmse   spatial crps  hourly crps")
    for key in keys:
        print(
            f"  {key:24s} {report['line'][key]['rmse']:9.3f} {report['spatial'][key]['rmse']:13.3f} "
            f"{report['hourly'][key]['rmse']:12.3f} {report['spatial'][key]['crps']:14.3f} "
            f"{report['hourly'][key]['crps']:12.3f}"
        )
    print(f"\noutputs in {out}/")
    return 0


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
    parser.add_argument("--out", required=True)
    parser.add_argument("--loss", choices=["crps", "mse"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--members", type=int, default=None)
    parser.add_argument("--eval-members", type=int, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--upsample", choices=["transpose", "bilinear"], default=None)
    parser.add_argument("--noise-fine-fraction", type=float, default=None)
    parser.add_argument("--coast-decay-km", type=float, default=None)
    parser.add_argument(
        "--no-coast-distance", dest="coast_distance", action="store_false", default=None,
        help="leave out the distance-to-coast input channel",
    )
    parser.add_argument(
        "--all-points", dest="land_only", action="store_false", default=None,
        help="train on every grid point instead of land only",
    )
    parser.add_argument("--val-start", default=None, help="overrides temporal.val_start with --temporal")
    parser.add_argument("--val-end", default=None, help="overrides temporal.val_end with --temporal")
    temporal = parser.add_argument_group("temporal", "the hourly model, with --temporal; defaults from temporal")
    temporal.add_argument("--temporal", action="store_true", help="train the hourly model")
    temporal.add_argument("--spatial", default=None, help="overrides temporal.spatial_model")
    temporal.add_argument("--from-scratch", action="store_true", default=None, help="do not start from the spatial weights")
    temporal.add_argument(
        "--init-hourly", default=None, help="start from this hourly checkpoint, keeping its statistics",
    )
    temporal.add_argument(
        "--multiscale", type=int, nargs="*", default=None, metavar="FACTOR",
        help="also score block averages of these sizes in pixels, e.g. 2 4 8 16, against noise at the wrong scale",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    training = cfg.training
    for name in (
        "loss", "epochs", "batch_size", "learning_rate", "members", "eval_members",
        "max_steps_per_epoch", "land_only", "upsample", "noise_fine_fraction", "coast_distance", "coast_decay_km",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(training, name, value)
    if args.temporal:
        for name, value in [("spatial_model", args.spatial), ("init_hourly", args.init_hourly),
                            ("from_scratch", args.from_scratch), ("multiscale", args.multiscale),
                            ("val_start", args.val_start), ("val_end", args.val_end)]:
            if value is not None:
                setattr(cfg.temporal, name, value)
        return train_temporal(cfg, Path(args.out), Path(args.pairs or cfg.paths.pairs), args.device)
    return train_spatial(cfg, args)


if __name__ == "__main__":
    sys.exit(main())