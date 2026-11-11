"""Apply a trained downscaler to coarse ensemble forecasts.

Usage::

    python scripts/downscale_forecasts.py --config config_spatial.yaml --model runs/crps/model.pt

Each coarse member yields several plausible fine-scale maps, so a ten-member
forecast becomes a larger downscaled ensemble. Initialisations already
processed are skipped unless the forecast has gained members since.
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
import pandas as pd
import xarray as xr

from downscale.config import load_config
from downscale.diagnostics import land_mask, mask_sea, sea_report
from downscale.downscaler import Downscaler
from downscale.inference import downscale_forecast

logger = logging.getLogger("downscale")


def downscale_init(
    forecast_path: Path, model: Downscaler, out_dir: Path, variable: str, samples: int, batch: int
) -> Path:
    """Downscale one initialisation.

    :param forecast_path: Forecast store for a single initialisation.
    :type forecast_path: pathlib.Path
    :param model: Trained downscaler.
    :type model: downscale.downscaler.Downscaler
    :param out_dir: Output directory.
    :type out_dir: pathlib.Path
    :param variable: Variable to downscale.
    :type variable: str
    :param samples: Fine-scale maps per coarse member.
    :type samples: int
    :param batch: Lead times processed at once.
    :type batch: int
    :returns: Path to the downscaled store.
    :rtype: pathlib.Path
    :raises RuntimeError: If the forecast has no completion marker.
    """
    target = out_dir / Path(forecast_path).name
    if not Path(str(forecast_path).rstrip("/") + ".done").exists():
        raise RuntimeError(
            f"{Path(forecast_path).name} has no .done marker: the forecast run is unfinished, and a "
            "partial store can hold fewer members than requested"
        )
    ds = xr.open_zarr(forecast_path)
    n_members = ds.sizes.get("ensemble", 1)
    if target.exists():
        existing = xr.open_zarr(target).sizes["ensemble"]
        if existing == n_members:
            logger.info("skipping %s, already downscaled with %d members", target.name, existing)
            return target
        logger.warning(
            "%s holds %d members but the forecast has %d: redoing it", target.name, existing, n_members
        )
        shutil.rmtree(target)

    started = time.time()
    result = downscale_forecast(model, ds[variable], samples, batch)
    result.attrs["forecast"] = str(forecast_path)
    coarse, fine = result["t2m_interp"].values, result["t2m_ds"].values
    n_lead, n_ensemble, n_samples = coarse.shape[1], coarse.shape[2], fine.shape[3]
    result = result.chunk({"lead_time": 10, "ensemble": 1, "sample": -1, "y": -1, "x": -1})
    for name in ("land_sea_mask", "orography"):
        result[name] = result[name].chunk({"y": -1, "x": -1})
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    result.to_zarr(tmp, mode="w")
    tmp.rename(target)
    report = sea_report(result)
    if report and report["correction_ratio_sea_over_land"] > 2:
        logger.warning(
            "the correction over sea is %.1f times the correction over land (%.2f against %.2f K): "
            "the target reanalysis treats water differently, so sea values are not trustworthy",
            report["correction_ratio_sea_over_land"], report["sea_correction_K"], report["land_correction_K"],
        )
    correction = fine.mean(3) - coarse
    logger.info(
        "%s: %d lead times x %d members x %d samples in %.0fs; mean correction %+.2f K, "
        "sample spread %.2f K, range %.1f..%.1f K",
        target.name, n_lead, n_ensemble, n_samples, time.time() - started, correction.mean(),
        fine.std(3, ddof=1).mean() if n_samples > 1 else 0.0, fine.min(), fine.max(),
    )
    return target


def map_figure(path: Path, png: Path) -> None:
    """Draw the coarse and downscaled fields for the warmest lead time.

    :param path: Downscaled store.
    :type path: pathlib.Path
    :param png: Destination image.
    :type png: pathlib.Path
    :returns: ``None``
    :rtype: None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = xr.open_zarr(path)
    mean_field = ds["t2m_interp"].isel(time=0).mean(("ensemble", "y", "x"))
    hours = pd.to_datetime(ds["valid_time"].isel(time=0).values).hour
    candidates = np.flatnonzero(hours == 12) if (hours == 12).any() else np.arange(len(hours))
    index = int(candidates[np.argmax(mean_field.values[candidates])])
    mask = land_mask(ds)
    coarse = mask_sea(ds["t2m_interp"].isel(time=0, lead_time=index, ensemble=0).values - 273.15, mask)
    fine = np.stack([
        mask_sea(f, mask) for f in ds["t2m_ds"].isel(time=0, lead_time=index, ensemble=0).values - 273.15
    ])
    low, high = np.nanpercentile(fine, [1, 99])
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6), constrained_layout=True)
    panels = [
        ("coarse member, interpolated", coarse),
        ("downscaled, sample 1", fine[0]),
        ("downscaled, mean of samples", fine.mean(0)),
    ]
    for axis, (title, field) in zip(ax, panels):
        mesh = axis.pcolormesh(ds.lon, ds.lat, field, cmap="RdYlBu_r", vmin=low, vmax=high, shading="auto")
        axis.set_title(title)
        fig.colorbar(mesh, ax=axis, label="degC")
    spread = fine.std(0, ddof=1) if fine.shape[0] > 1 else np.zeros_like(fine[0])
    mesh = ax[3].pcolormesh(ds.lon, ds.lat, spread, cmap="magma", shading="auto")
    ax[3].set_title("spread of samples")
    fig.colorbar(mesh, ax=ax[3], label="K")
    mid_lat = float(np.mean(ds.lat.values))
    for axis in ax:
        axis.set_aspect(1 / np.cos(np.deg2rad(mid_lat)))
    valid = pd.Timestamp(ds["valid_time"].isel(time=0, lead_time=index).values)
    lead_hours = int(ds.lead_time.values[index] / np.timedelta64(1, "h"))
    fig.suptitle(
        f"initialised {pd.Timestamp(ds.time.values[0]):%Y-%m-%d %H} UTC, "
        f"valid {valid:%Y-%m-%d %H} UTC (+{lead_hours} h)"
    )
    fig.savefig(png, dpi=95)
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
    parser.add_argument("--forecasts", nargs="+", default=None, help="overrides paths.forecasts")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default=None, help="overrides paths.downscaled")
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
    out = Path(args.out or cfg.paths.downscaled)
    samples = args.samples or cfg.inference.samples
    batch = args.batch_size or cfg.inference.batch_size

    model = Downscaler.load(args.model, args.device)
    logger.info("model %s on %s, fine grid %dx%d", model.net.cfg, model.device, *model.lat.shape)
    done = [
        downscale_init(Path(f), model, out, cfg.coarse.variable, samples, batch) for f in forecasts
    ]
    map_figure(done[-1], out / "map_check.png")
    print(f"\n{len(done)} initialisation(s) downscaled into {out}/ (map: {out / 'map_check.png'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())