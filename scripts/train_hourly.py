"""Train the hourly downscaler, which fills the hours between 6-hourly coarse steps.

Usage::

    python scripts/train_hourly.py --pairs data/pairs_hourly --spatial runs/spatial/model.pt --out runs/hourly

The pairs must be hourly: build them with ``build_pairs.py`` from a target downloaded with
``download_cerra.py --hourly``. Every hour is given the coarse fields at the two ends of its 6-hour
window (00, 06, 12 and 18 UTC), so only the coarse steps a forecast actually has are used.

The network starts from the spatial model's weights with its two new channels switched off, so before
training it is the spatial model applied to the straight line between the two coarse steps. Validation
compares three ways of filling an hour, for each hour of the window:

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
from downscale.config import load_config
from downscale.diagnostics import land_mask

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_downscaler import Scores, crps_per_pixel, open_pairs, split_times  # noqa: E402

logger = logging.getLogger("train_hourly")
METHODS = ("line", "spatial", "hourly")


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


def figures(out: Path, report: dict, history: list[dict], series: dict) -> None:
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
            axis.plot(offsets, [report[name][f"land_+{k}h"][metric] for k in offsets], style, label=name)
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
    ax[3].plot(t, series["line"] - 273.15, "k--", label="line")
    ax[3].plot(t, series["hourly"] - 273.15, "C3-", label="hourly, mean")
    ax[3].plot(t[series["steps"]], series["truth"][series["steps"]] - 273.15, "ko", ms=4)
    ax[3].set_title(f"one land point, {series['where']}")
    ax[3].set_ylabel("degC")
    ax[3].legend(fontsize=8)
    ax[3].tick_params(axis="x", labelrotation=30)
    fig.savefig(out / "hourly_scores.png", dpi=100)
    plt.close(fig)


def point_series(model: hr.HourlyDownscaler, coarse, fine, times, hours, starts, ends, land, days: int = 3) -> dict:
    """The truth, the line and the hourly model's mean at the land point of highest daily range.

    :returns: Times and values over the first ``days`` days of the validation hours.
    :rtype: dict
    """
    h, s, e = hours[: 24 * days], starts[: 24 * days], ends[: 24 * days]
    truth = fine[h]
    daily_range = np.where(land, truth.max(0) - truth.min(0), -np.inf)
    y, x = np.unravel_index(int(np.argmax(daily_range)), daily_range.shape)
    offsets = hr.offsets_of(times[h])
    pred = model.predict_hours(coarse[s], coarse[e], offsets, times[h], members=4)
    line = hr.temporal_interpolation(coarse[s], coarse[e], offsets)
    return {
        "times": times[h], "truth": truth[:, y, x], "line": line[:, y, x], "hourly": pred.mean(1)[:, y, x],
        "steps": offsets == 0,
        "where": f"{float(model.lat[y, x]):.2f}N {float(model.lon[y, x]):.2f}E",
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    :param argv: Argument list; ``None`` reads ``sys.argv``.
    :type argv: collections.abc.Sequence[str] | None
    :returns: Exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--pairs", required=True, help="hourly pairs from build_pairs.py")
    parser.add_argument("--spatial", required=True, help="checkpoint of the spatial model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--from-scratch", action="store_true", help="do not start from the spatial weights")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--members", type=int, default=None)
    parser.add_argument("--eval-members", type=int, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--val-start", default=None)
    parser.add_argument("--val-end", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    training = cfg.training
    for name in ("epochs", "batch_size", "learning_rate", "members", "eval_members", "max_steps_per_epoch"):
        if getattr(args, name) is not None:
            setattr(training, name, getattr(args, name))
    torch.manual_seed(training.seed)
    rng = np.random.default_rng(training.seed)
    device = args.device or hr.default_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spatial = hr.load_spatial(args.spatial, device)
    ds = open_pairs(Path(args.pairs))
    times = ds.time.values
    coarse, fine = ds.coarse_t2m.values.astype("f4"), ds.fine_t2m.values.astype("f4")
    if coarse.shape[1:] != spatial.lat.shape:
        raise ValueError(f"pairs grid {coarse.shape[1:]} differs from the spatial model's {spatial.lat.shape}")
    static_path = Path(args.pairs) / "static.zarr"
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
    train_mask, val_mask = split_times(pd.DatetimeIndex(times[hours]), cfg.val_years, args.val_start, args.val_end)
    train = tuple(a[train_mask] for a in (hours, starts, ends))
    val = tuple(a[val_mask] for a in (hours, starts, ends))
    logger.info(
        "%d hourly steps: %d training and %d validation hours with both ends of their window",
        len(times), len(train[0]), len(val[0]),
    )

    stats = {**spatial.stats, "tendency": tendency_stats(coarse, train[1], train[2], land)}
    model_cfg = hr.HourlyConfig(**asdict(spatial.net.cfg))
    net = hr.HourlyNet(model_cfg)
    if not args.from_scratch:
        hr.init_from_spatial(net, args.spatial)
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
    members = training.members if model_cfg.stochastic else 1
    loss_mask = torch.from_numpy(land).to(device) if training.land_only else None
    logger.info(
        "model: %s loss, %.2fM parameters, %s, %d steps per epoch, device %s; tendency std %.2f K",
        training.loss, sum(p.numel() for p in net.parameters()) / 1e6,
        "from scratch" if args.from_scratch else f"started from {args.spatial}", steps_per_epoch, device,
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
                out / "model.pt", extra={"epoch": epoch, "spatial": str(args.spatial)}
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
    figures(out, report, history, point_series(model, coarse, fine, times, *val, land))

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


if __name__ == "__main__":
    sys.exit(main())
