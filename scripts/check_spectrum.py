"""Compare the power spectra of the coarse field, the model and the target.

Usage::

    python scripts/check_spectrum.py --config config_spatial.yaml --model runs/crps/model.pt
    python scripts/check_spectrum.py --config config_temporal.yaml --model runs/spatial/model.pt \\
        --hourly runs/hourly/model.pt

Answers whether the model adds detail at the scales the target actually has.
Power well below the target's at short wavelengths means the field is too
smooth; power well above means the model is inventing texture the target does
not contain. The ensemble mean is expected to sit below the target, since
averaging removes unpredictable detail; an individual member is the one that
should match.

Outputs in the model's directory: ``spectrum.csv``, ``spectrum.png`` and a
printed summary.

With ``--hourly``, the hourly model is checked in space and in time instead, over the
longest run of consecutive validation hours from ``temporal.val_start``, and four ways
of filling an hour are compared with the target:

- ``line``: the coarse field interpolated linearly in time, shown as "coarse, linear in time";
- ``spatial``: the spatial model (``--model``) on that line, one member, drawing new noise every hour;
- ``hourly``: the hourly model, one member, keeping its noise from hour to hour;
- ``hourly mean``: the mean of the hourly members.

The spatial spectrum is the usual radially averaged one, over every validation hour. The temporal
spectrum is that of each land point's hourly series, averaged over land: a member that redraws its
detail every hour shows up there as too much power at periods of a few hours. Sea points are set to
the target in every field, as the models are not trained there. The target's own series carries a
small jump every three hours, where CERRA's analyses replace its +1 h and +2 h forecasts, so its
power at a 3-hour period is partly an artefact. Outputs go to the hourly model's directory as
``hourly_spectrum.{png,json}`` and ``hourly_spectrum_{space,time}.csv``.
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

from downscale import hourly as hr
from downscale.config import load_config
from downscale.diagnostics import (
    effective_resolution_km,
    grid_spacing_km,
    land_mask,
    longest_run,
    space_time_spectrum_figure,
    spectrum_comparison,
    spectrum_figure,
    temporal_spectrum,
)
from downscale.downscaler import Downscaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_downscaler import open_pairs  # noqa: E402

logger = logging.getLogger("spectrum")
LABELS = ["target", "coarse", "model member", "model mean"]
HOURLY_LABELS = ["target", "line", "spatial", "hourly", "hourly mean"]
HOURLY_COLOURS = {"target": "k", "line": "0.55", "spatial": "tab:blue", "hourly": "tab:red", "hourly mean": "tab:orange"}
SHOWN = {"line": "coarse, linear in time", "spatial": "spatial, linear in time"}


def validation_slice(pairs_dir: Path, val_years: Sequence[int], n_times: int, hour: int | None, seed: int):
    """Select a sample of validation times from the training pairs.

    :param pairs_dir: Directory written by the pairing stage.
    :type pairs_dir: pathlib.Path
    :param val_years: Years held out of training.
    :type val_years: collections.abc.Sequence[int]
    :param n_times: Number of times to draw.
    :type n_times: int
    :param hour: Restrict to this UTC hour; ``None`` uses every hour.
    :type hour: int | None
    :param seed: Random seed for the draw.
    :type seed: int
    :returns: Coarse fields, target fields and their times.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :raises ValueError: If no time matches even after dropping the year filter.
    """
    store = pairs_dir / "pairs.zarr"
    ds = xr.open_zarr(store) if store.exists() else xr.open_mfdataset(
        [str(p) for p in sorted((pairs_dir / "months").glob("pairs_*.zarr"))],
        engine="zarr", combine="nested", concat_dim="time",
    )
    times = pd.DatetimeIndex(ds.time.values)
    chosen = times.year.isin(list(val_years)) if val_years else np.ones(len(times), bool)
    if not chosen.any():
        logger.warning(
            "no time falls in the validation years %s: sampling every year instead, so the spectrum "
            "includes data the model was trained on",
            list(val_years),
        )
        chosen = np.ones(len(times), bool)
    if hour is not None:
        chosen = chosen & (times.hour == hour)
    index = np.flatnonzero(chosen)
    if index.size == 0:
        raise ValueError(f"no time matches hour {hour}")
    picked = np.sort(np.random.default_rng(seed).choice(index, min(n_times, index.size), replace=False))
    subset = ds.isel(time=picked)
    return (
        subset.coarse_t2m.values.astype("f4"),
        subset.fine_t2m.values.astype("f4"),
        times[picked].to_numpy(),
    )


def fair_crps(ensemble: np.ndarray, observed: np.ndarray) -> float:
    """Mean fair CRPS of an ensemble.

    :param ensemble: Members on axis 1, ``(n, members, ...)``.
    :type ensemble: numpy.ndarray
    :param observed: The truth, ``(n, ...)``.
    :type observed: numpy.ndarray
    :returns: The CRPS averaged over every point.
    :rtype: float
    """
    m = ensemble.shape[1]
    skill = np.abs(ensemble - observed[:, None]).mean(1)
    spread = sum(np.abs(ensemble[:, i] - ensemble[:, j]) for i in range(m) for j in range(i + 1, m))
    return float((skill - spread / (m * (m - 1))).mean())


def hour_to_hour(target: np.ndarray, line: np.ndarray, spatial: np.ndarray, hourly: np.ndarray,
                 land: np.ndarray) -> dict:
    """How each way of filling the hours changes from one hour to the next, over land.

    :param target: The target, ``(hours, y, x)``, at consecutive hours.
    :type target: numpy.ndarray
    :param line: The straight line, ``(hours, y, x)``.
    :type line: numpy.ndarray
    :param spatial: Spatial model members, ``(hours, members, y, x)``.
    :type spatial: numpy.ndarray
    :param hourly: Hourly model members, ``(hours, members, y, x)``.
    :type hourly: numpy.ndarray
    :param land: Land mask, ``(y, x)``.
    :type land: numpy.ndarray
    :returns: Mean absolute change per hour of one member [K/h], and the CRPS of the members' fields and of
        their one-hour changes against the target's [K].
    :rtype: dict
    """
    change = {name: np.diff(field[..., land], axis=0) for name, field in
              {"target": target, "line": line, "spatial": spatial, "hourly": hourly}.items()}
    one = {name: c if c.ndim == 2 else c[:, 0] for name, c in change.items()}
    return {
        "abs_change_K_per_h": {name: round(float(np.abs(c).mean()), 3) for name, c in one.items()},
        "crps_K": {name: round(fair_crps(f[..., land], target[..., land]), 3)
                   for name, f in {"spatial": spatial, "hourly": hourly}.items()},
        "crps_1h_change_K": {name: round(fair_crps(change[name], change["target"]), 3)
                             for name in ("spatial", "hourly")},
    }


def hourly_spectra(
    pairs_dir: Path, hourly_path: str, spatial_path: str, out: Path, val_start: str, val_end: str | None = None,
    members: int = 4, device: str | None = None, noise_hours: float | None = None, seed: int = 0,
) -> int:
    """Power spectra of the hourly model, in space and in time, against the target.

    :param pairs_dir: Hourly pairs from ``build_pairs.py``.
    :type pairs_dir: pathlib.Path
    :param hourly_path: Checkpoint of the hourly model.
    :type hourly_path: str
    :param spatial_path: Checkpoint of the spatial model.
    :type spatial_path: str
    :param out: Output directory.
    :type out: pathlib.Path
    :param val_start: First validation time.
    :type val_start: str
    :param val_end: Last validation time, ``None`` for the end of the pairs.
    :type val_end: str | None
    :param members: Ensemble members to draw.
    :type members: int
    :param device: Torch device; ``None`` picks one.
    :type device: str | None
    :param noise_hours: Correlation time [h] of the hourly model's noise; ``None`` keeps it fixed.
    :type noise_hours: float | None
    :param seed: Seed of the hourly model's noise.
    :type seed: int
    :returns: Exit status.
    :rtype: int
    :raises ValueError: If fewer than 24 consecutive validation hours are available.
    """
    out.mkdir(parents=True, exist_ok=True)
    ds = open_pairs(pairs_dir)
    times = ds.time.values
    hours, starts, ends = hr.window_samples(times)
    chosen = times[hours] >= np.datetime64(pd.Timestamp(val_start))
    if val_end:
        chosen &= times[hours] <= np.datetime64(pd.Timestamp(val_end))
    hours, starts, ends = hours[chosen], starts[chosen], ends[chosen]
    run = longest_run(times[hours])
    hours, starts, ends = hours[run], starts[run], ends[run]
    if len(hours) < 24:
        raise ValueError(f"only {len(hours)} consecutive validation hours; the temporal spectrum needs a day or more")

    hourly = hr.HourlyDownscaler.load(hourly_path, device)
    spatial = hr.load_spatial(spatial_path, device)
    land = land_mask(spatial.static)
    land = np.ones(spatial.lat.shape, bool) if land is None else land
    valid = times[hours]
    offsets = hr.offsets_of(valid)
    c0 = ds.coarse_t2m.values[starts].astype("f4")
    c1 = ds.coarse_t2m.values[ends].astype("f4")
    target = ds.fine_t2m.values[hours].astype("f4")
    logger.info("%d consecutive hours from %s, %d members", len(hours), pd.Timestamp(valid[0]), members)

    line = hr.temporal_interpolation(c0, c1, offsets)
    by_spatial = spatial.predict(line, valid, members=members)
    by_hourly = hourly.predict_hours(c0, c1, offsets, valid, members=members, seed=seed, noise_hours=noise_hours)
    sea = ~land

    def on_land(stack: np.ndarray) -> np.ndarray:
        stack = stack.copy()
        stack[..., sea] = target[(slice(None),) + (None,) * (stack.ndim - 3)][..., sea]
        return stack

    fields = {
        "target": target,
        "line": on_land(line),
        "spatial": on_land(by_spatial[:, 0]),
        "hourly": on_land(by_hourly[:, 0]),
        "hourly mean": on_land(by_hourly.mean(1)),
    }
    spacing = grid_spacing_km(spatial.lat, spatial.lon)
    in_space = spectrum_comparison(fields, spacing_km=spacing)
    in_time = temporal_spectrum(
        {**fields, "spatial": on_land(by_spatial), "hourly": on_land(by_hourly)}, land
    )
    in_space.to_csv(out / "hourly_spectrum_space.csv", index=False)
    in_time.to_csv(out / "hourly_spectrum_time.csv", index=False)
    space_time_spectrum_figure(in_space, in_time, HOURLY_LABELS, out / "hourly_spectrum.png", HOURLY_COLOURS, SHOWN)

    short = in_time.period_h <= 4
    fine_scale = in_space.wavelength_km < 10 * spacing
    summary = {
        "hours": len(hours),
        "start": str(pd.Timestamp(valid[0])),
        "members": members,
        "noise_hours": noise_hours,
        **hour_to_hour(target, line, by_spatial, by_hourly, land),
        "power_ratio_periods_up_to_4h": {
            name: round(float((in_time[name][short] / in_time["target"][short]).mean()), 3) for name in HOURLY_LABELS[1:]
        },
        f"power_ratio_below_{int(10 * spacing)}km": {
            name: round(float((in_space[name][fine_scale] / in_space["target"][fine_scale]).mean()), 3)
            for name in HOURLY_LABELS[1:]
        },
    }
    (out / "hourly_spectrum.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}/hourly_spectrum.png, hourly_spectrum_space.csv, hourly_spectrum_time.csv, hourly_spectrum.json")
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
    parser.add_argument("--model", "--spatial", dest="model", required=True,
                        help="the spatial model; with --hourly, the baseline the hourly model is compared with")
    parser.add_argument("--hourly", default=None, help="check this hourly model in space and time instead")
    parser.add_argument("--val-start", default=None, help="with --hourly, overrides temporal.val_start")
    parser.add_argument("--val-end", default=None, help="with --hourly, overrides temporal.val_end")
    parser.add_argument("--pairs", default=None, help="overrides paths.pairs")
    parser.add_argument("--out", default=None, help="defaults to the model's directory")
    parser.add_argument("--n-times", type=int, default=24)
    parser.add_argument("--hour", type=int, default=12, help="UTC hour to sample; -1 uses every hour")
    parser.add_argument("--members", type=int, default=None, help="8, or 4 with --hourly")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-hours", type=float, default=None,
                        help="with --hourly, correlation time [h] of the noise; overrides temporal.noise_hours")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    pairs_dir = Path(args.pairs or cfg.paths.pairs)
    if args.hourly:
        val_start = args.val_start or cfg.temporal.val_start
        if not val_start:
            parser.error("--hourly needs --val-start or temporal.val_start")
        out = Path(args.out) if args.out else Path(args.hourly).parent
        return hourly_spectra(pairs_dir, args.hourly, args.model, out, val_start,
                              args.val_end or cfg.temporal.val_end, args.members or 4, args.device,
                              args.noise_hours if args.noise_hours is not None else cfg.temporal.noise_hours, args.seed)
    args.members = args.members or 8
    out = Path(args.out) if args.out else Path(args.model).parent
    out.mkdir(parents=True, exist_ok=True)

    coarse, target, times = validation_slice(
        pairs_dir, cfg.val_years, args.n_times, None if args.hour < 0 else args.hour, args.seed
    )
    model = Downscaler.load(args.model, args.device)
    predicted = model.predict(coarse, times, members=args.members)
    spacing = grid_spacing_km(model.lat, model.lon)
    logger.info(
        "%d time(s), grid spacing %.2f km, Nyquist wavelength %.1f km", len(times), spacing, 2 * spacing
    )

    comparison = spectrum_comparison(
        {
            "target": target,
            "coarse": coarse,
            "model member": predicted[:, 0],
            "model mean": predicted.mean(1),
        },
        spacing_km=spacing,
    )
    comparison.to_csv(out / "spectrum.csv", index=False)
    spectrum_figure(comparison, LABELS, out / "spectrum.png")

    summary = {
        "n_times": len(times),
        "grid_spacing_km": round(spacing, 3),
        "effective_resolution_km": {
            name: round(effective_resolution_km(comparison, name, "target"), 1)
            for name in ("coarse", "model member", "model mean")
        },
    }
    short = comparison[comparison.wavelength_km < 10 * spacing]
    summary["power_ratio_below_%dkm" % int(10 * spacing)] = {
        name: round(float((short[name] / short["target"]).mean()), 3)
        for name in ("coarse", "model member", "model mean")
    }
    (out / "spectrum.json").write_text(json.dumps(summary, indent=2))

    bands = [(400, np.inf), (200, 400), (100, 200), (50, 100), (25, 50), (2 * spacing, 25)]
    rows = []
    for low, high in bands:
        chosen = comparison[(comparison.wavelength_km >= low) & (comparison.wavelength_km < high)]
        if not len(chosen):
            continue
        label = f"{low:.0f}-{high:.0f} km" if np.isfinite(high) else f"above {low:.0f} km"
        rows.append({
            "wavelengths": label,
            **{name: round(float((chosen[name] / chosen["target"]).mean()), 3)
               for name in ("coarse", "model member", "model mean")},
        })
    print("\npower relative to the target (one means a perfect match)")
    print(pd.DataFrame(rows).set_index("wavelengths").to_string())
    print("\nsmallest wavelength still resolved, at half the target's power:")
    for name, value in summary["effective_resolution_km"].items():
        print(f"  {name:14s} {value} km")
    member_ratio = summary[f"power_ratio_below_{int(10 * spacing)}km"]["model member"]
    if member_ratio > 1.5:
        print(
            f"\nthe member carries {member_ratio:.1f} times the target's power below "
            f"{int(10 * spacing)} km: the model is adding detail the target does not have"
        )
    elif member_ratio < 0.5:
        print(f"\nthe member is too smooth below {int(10 * spacing)} km ({member_ratio:.2f} of the target)")
    else:
        print(f"\nthe member matches the target below {int(10 * spacing)} km ({member_ratio:.2f})")
    print(f"wrote {out}/spectrum.csv, spectrum.png, spectrum.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())