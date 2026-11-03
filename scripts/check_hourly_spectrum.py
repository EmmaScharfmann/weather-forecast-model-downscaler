"""Power spectra of the hourly model, in space and in time, against the target.

Usage::

    python scripts/check_hourly_spectrum.py --pairs data/pairs_hourly --hourly runs/hourly/model.pt \\
        --spatial runs/spatial/model.pt --val-start 2020-08-01

Over the validation hours, four ways of filling an hour are compared with the target:

- ``line``: the coarse field interpolated in time;
- ``spatial``: the spatial model on that line, one member, drawing new noise every hour;
- ``hourly``: the hourly model, one member, keeping its noise from hour to hour;
- ``hourly mean``: the mean of the hourly members.

The spatial spectrum is the usual radially averaged one, over every validation hour. The temporal
spectrum is that of each land point's hourly series, averaged over land: a member that redraws its
detail every hour shows up there as too much power at periods of a few hours. Sea points are set to
the target in every field, as the models are not trained there.

The target's own series carries a small jump every three hours, where CERRA's analyses replace its
+1 h and +2 h forecasts, so its power at a 3-hour period is partly an artefact.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from downscale import hourly as hr  # noqa: E402
from downscale.diagnostics import grid_spacing_km, land_mask, spectrum_comparison  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_downscaler import open_pairs  # noqa: E402

logger = logging.getLogger("hourly_spectrum")
LABELS = ["target", "line", "spatial", "hourly", "hourly mean"]


def longest_run(times: np.ndarray) -> slice:
    """The longest stretch of consecutive hours.

    :param times: Sorted times.
    :type times: numpy.ndarray
    :returns: Slice selecting it.
    :rtype: slice
    """
    t = np.asarray(times, "datetime64[h]")
    breaks = np.flatnonzero(np.diff(t) != np.timedelta64(1, "h")) + 1
    edges = np.concatenate([[0], breaks, [len(t)]])
    k = int(np.argmax(np.diff(edges)))
    return slice(int(edges[k]), int(edges[k + 1]))


def temporal_spectrum(fields: dict[str, np.ndarray], land: np.ndarray) -> pd.DataFrame:
    """Power of each land point's series, averaged over land.

    :param fields: Labels mapped to hourly stacks ``(time, y, x)`` or ``(time, member, y, x)``.
    :type fields: dict[str, numpy.ndarray]
    :param land: Land mask over ``(y, x)``.
    :type land: numpy.ndarray
    :returns: Power per period in hours, one column per label.
    :rtype: pandas.DataFrame
    """
    n = next(iter(fields.values())).shape[0]
    window = np.hanning(n)
    frequency = np.fft.rfftfreq(n, d=1.0)[1:]
    out = pd.DataFrame({"period_h": 1.0 / frequency})
    for name, stack in fields.items():
        series = stack[..., land]  # (time, [member,] points)
        series = series - series.mean(0, keepdims=True)
        shape = (n,) + (1,) * (series.ndim - 1)
        power = np.abs(np.fft.rfft(series * window.reshape(shape), axis=0)) ** 2
        out[name] = power[1:].reshape(len(frequency), -1).mean(1)
    return out


def figure(spatial: pd.DataFrame, temporal: pd.DataFrame, path: Path) -> None:
    """Plot both spectra and their ratio to the target.

    :param spatial: Output of :func:`downscale.diagnostics.spectrum_comparison`.
    :type spatial: pandas.DataFrame
    :param temporal: Output of :func:`temporal_spectrum`.
    :type temporal: pandas.DataFrame
    :param path: Destination image.
    :type path: pathlib.Path
    :returns: ``None``
    :rtype: None
    """
    colours = {"target": "k", "line": "0.55", "spatial": "tab:blue", "hourly": "tab:red", "hourly mean": "tab:orange"}
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, (frame, x, unit, what) in enumerate([
        (spatial, "wavelength_km", "wavelength (km)", "spatial"),
        (temporal, "period_h", "period (hours)", "temporal"),
    ]):
        for name in LABELS:
            ax[row, 0].loglog(frame[x], frame[name], color=colours[name], marker="o", ms=2, label=name)
            if name != "target":
                ax[row, 1].semilogx(frame[x], frame[name] / frame["target"], color=colours[name], marker="o",
                                    ms=2, label=name)
        ax[row, 0].set_title(f"{what} power spectrum over land")
        ax[row, 0].set_ylabel("power")
        ax[row, 1].axhline(1.0, color="k", lw=0.8, ls="--")
        ax[row, 1].set_title("relative to the target: too smooth below one, too noisy above")
        ax[row, 1].set_ylabel("power / target")
        for a in ax[row]:
            a.set_xlabel(unit)
            a.invert_xaxis()
            a.grid(alpha=0.3, which="both")
            a.legend(fontsize=8)
    for period in (3, 6, 12, 24):
        ax[1, 0].axvline(period, color="0.8", lw=0.6, zorder=0)
        ax[1, 1].axvline(period, color="0.8", lw=0.6, zorder=0)
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
    parser.add_argument("--pairs", required=True, help="hourly pairs from build_pairs.py")
    parser.add_argument("--hourly", required=True, help="checkpoint of the hourly model")
    parser.add_argument("--spatial", required=True, help="checkpoint of the spatial model")
    parser.add_argument("--out", default=None, help="defaults to the hourly model's directory")
    parser.add_argument("--val-start", required=True)
    parser.add_argument("--val-end", default=None)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    out = Path(args.out) if args.out else Path(args.hourly).parent
    out.mkdir(parents=True, exist_ok=True)
    ds = open_pairs(Path(args.pairs))
    times = ds.time.values
    hours, starts, ends = hr.window_samples(times)
    chosen = times[hours] >= np.datetime64(pd.Timestamp(args.val_start))
    if args.val_end:
        chosen &= times[hours] <= np.datetime64(pd.Timestamp(args.val_end))
    hours, starts, ends = hours[chosen], starts[chosen], ends[chosen]
    run = longest_run(times[hours])
    hours, starts, ends = hours[run], starts[run], ends[run]
    if len(hours) < 24:
        parser.error(f"only {len(hours)} consecutive validation hours; the temporal spectrum needs a day or more")

    hourly = hr.HourlyDownscaler.load(args.hourly, args.device)
    spatial = hr.load_spatial(args.spatial, args.device)
    land = land_mask(spatial.static)
    land = np.ones(spatial.lat.shape, bool) if land is None else land
    valid = times[hours]
    offsets = hr.offsets_of(valid)
    c0 = ds.coarse_t2m.values[starts].astype("f4")
    c1 = ds.coarse_t2m.values[ends].astype("f4")
    target = ds.fine_t2m.values[hours].astype("f4")
    logger.info("%d consecutive hours from %s, %d members", len(hours), pd.Timestamp(valid[0]), args.members)

    line = hr.temporal_interpolation(c0, c1, offsets)
    by_spatial = spatial.predict(line, valid, members=args.members)
    by_hourly = hourly.predict_hours(c0, c1, offsets, valid, members=args.members, seed=0)
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
    figure(in_space, in_time, out / "hourly_spectrum.png")

    short = in_time.period_h <= 4
    fine_scale = in_space.wavelength_km < 10 * spacing
    summary = {
        "hours": len(hours),
        "start": str(pd.Timestamp(valid[0])),
        "members": args.members,
        "power_ratio_periods_up_to_4h": {
            name: round(float((in_time[name][short] / in_time["target"][short]).mean()), 3) for name in LABELS[1:]
        },
        f"power_ratio_below_{int(10 * spacing)}km": {
            name: round(float((in_space[name][fine_scale] / in_space["target"][fine_scale]).mean()), 3)
            for name in LABELS[1:]
        },
    }
    (out / "hourly_spectrum.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}/hourly_spectrum.png, hourly_spectrum_space.csv, hourly_spectrum_time.csv, hourly_spectrum.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
