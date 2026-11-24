"""Score the rainfall downscaler against independent observations.

Takes trained weights directly (a local ``model.pt`` or ``hf://repo/model.pt``)
and a test pack (ERA5 forcing + PARDIG target), then compares five forecasts:

    ERA5 interval mean | ERA5 smooth (PCHIP) | PARDIG | downscaler members | downscaler ensemble mean

against three references: PARDIG (what the model was trained to imitate),
EURADCLIM (gauge-adjusted radar maps) and Météo-France hourly gauges (points).
Gridded references use the same cells: observed by EURADCLIM, outside the
model's context ring.

    export KNMI_API_KEY=...   # free key from developer.dataplatform.knmi.nl, for EURADCLIM
    python scripts/precipitation_obs_eval.py --pairs data/precipitation/obs_test/test.zarr \
        --model hf://EmmaScharfmann/france-precipitation-downscaler/model.pt --out outputs/precipitation_obs_eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import xarray as xr

from downscaler.data.precipitation_observations import (
    at_stations,
    gauge_lag_check,
    gauge_matrix,
    load_euradclim,
    load_gauges,
)
from downscaler.evaluation import precipitation_metrics as metrics


def resolve_model(path):
    if str(path).startswith("hf://"):
        from huggingface_hub import hf_hub_download

        parts = str(path)[5:].split("/")
        return hf_hub_download("/".join(parts[:2]), "/".join(parts[2:]) or "model.pt")
    return path


def forecasts(pairs, model=None, predictions=None, members=8, seed=2):
    """All compared forecasts as (time, member, lat, lon) arrays in mm."""
    from downscaler.processing.precipitation_temporal import hourly_baseline

    if predictions is None:
        from downscaler.models.precipitation import predict

        predictions = predict(resolve_model(model), pairs, members=members, noise_seed=seed)
    rain = predictions.precipitation
    for name in ("time", "lat", "lon"):
        if not np.array_equal(rain[name].values, pairs[name].values):
            raise ValueError(f"predictions and pack {name} differ")
    model = rain.values.astype("float32")
    return {
        "ERA5 interval mean": pairs.predictors.sel(variable="tp").values[:, None].astype("float32"),
        "ERA5 smooth (PCHIP)": hourly_baseline(pairs)[:, None],
        "PARDIG": pairs.target.values[:, None].astype("float32"),
        "Downscaler (members)": model,
        "Downscaler (ensemble mean)": model.mean(1, keepdims=True),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", required=True, help="test pack (zarr) with ERA5 predictors and PARDIG target")
    parser.add_argument(
        "--model",
        default="hf://EmmaScharfmann/france-precipitation-downscaler/model.pt",
        help="model.pt path or hf://user/repo/model.pt",
    )
    parser.add_argument("--predictions", help="existing predictions.zarr, instead of running the model")
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2, help="noise seed; matches evaluation_noise_seed")
    parser.add_argument("--cache", default="data/observations")
    parser.add_argument("--out", default="outputs/precipitation_obs_eval")
    parser.add_argument("--margin", type=int, default=8, help="context ring excluded from every score")
    parser.add_argument("--no-radar", action="store_true", help="skip EURADCLIM (e.g. without a KNMI key)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = xr.open_zarr(args.pairs).load()
    lat, lon, times = pairs.lat.values, pairs.lon.values, pd.DatetimeIndex(pairs.time.values)
    predictions = xr.open_zarr(args.predictions).load() if args.predictions else None
    models = forecasts(pairs, args.model, predictions, args.members, args.seed)
    cell_km = float(np.mean([111.2 * (lat[1] - lat[0]), 111.2 * (lon[1] - lon[0]) * np.cos(np.radians(lat.mean()))]))

    # References on the rainfall grid, restricted to the same cells.
    m = args.margin
    ring = np.zeros((len(lat), len(lon)), dtype=bool)
    ring[m : -m or None, m : -m or None] = True
    radar = None if args.no_radar else load_euradclim(times, lat, lon, args.cache)
    observed = np.broadcast_to(ring, (len(times), *ring.shape)).copy() if radar is None else np.isfinite(radar) & ring
    if radar is not None:
        radar[~observed] = np.nan
    pardig = np.where(observed, pairs.target.values, np.nan).astype("float32")
    gauges = load_gauges(times, args.cache, bbox=(lat[m], lat[-m - 1], lon[m], lon[-m - 1]))
    gauge_values, stations = gauge_matrix(gauges, times, lat, lon)

    report = {
        "pairs": str(args.pairs),
        "model": str(args.model or args.predictions),
        "members": args.members,
        "hours": len(times),
        "days": sorted({str(t.date()) for t in times - pd.Timedelta(hours=1)}),
        "cell_km": cell_km,
        "gridded_cells": int(observed.any(0).sum()),
        "stations": len(stations),
        "gauge_lag_correlation": None if radar is None else gauge_lag_check(gauge_values, stations, radar),
        "scores": {},
    }
    references = {"PARDIG": (pardig, True), "EURADCLIM": (radar, True), "Gauges": (gauge_values, False)}
    references = {k: v for k, v in references.items() if v[0] is not None}
    for ref_name, (reference, gridded) in references.items():
        report["scores"][ref_name] = {}
        for name, forecast in models.items():
            if name == ref_name:
                continue
            field = forecast if gridded else at_stations(forecast, stations)
            report["scores"][ref_name][name] = metrics.all_scores(field, reference, gridded, cell_km)
            report["scores"][ref_name][name]["qq"] = metrics.qq(field, reference)

    # Spectra on the largest square observed at every hour, same box for all.
    box = metrics.complete_box(pardig)
    if box is not None:
        r, c, s, hours = box

        def crop(a):
            return a[hours][..., r : r + s, c : c + s]

        report["spectrum_box"] = {
            "lat": [float(lat[r]), float(lat[r + s - 1])],
            "lon": [float(lon[c]), float(lon[c + s - 1])],
            "hours": int(hours.sum()),
        }
        report["spectra"] = {} if radar is None else {"EURADCLIM": metrics.spectrum(crop(radar), cell_km)}
        for name, forecast in models.items():
            report["spectra"][name] = metrics.spectrum(crop(forecast).reshape(-1, s, s), cell_km)

    (out / "results.json").write_text(json.dumps(report, indent=1))
    (out / "results.md").write_text(tables(report))
    figures(report, models, pardig if radar is None else radar, pairs, out)
    print((out / "results.md").read_text())


def tables(report):
    rows = [
        ("Bias (mm)", "amount", "bias_mm"),
        ("MAE (mm)", "amount", "mae_mm"),
        ("RMSE (mm)", "amount", "rmse_mm"),
        ("CRPS (mm)", "probability", "ensemble_crps_mm"),
        ("CSI ≥1 mm", "occurrence", "csi_1mm"),
        ("CSI ≥5 mm", "occurrence", "csi_5mm"),
        ("Frequency bias ≥1 mm", "occurrence", "frequency_bias_1mm"),
        ("Useful scale ≥1 mm (km)", "structure", "useful_scale_km_1mm"),
        ("Useful scale ≥5 mm (km)", "structure", "useful_scale_km_5mm"),
        ("99th percentile (mm)", "intensity", "q99_mm"),
        ("99.9th percentile (mm)", "intensity", "q99.9_mm"),
    ]
    text = [
        f"Hours: {report['hours']} ({', '.join(report['days'])}); stations: {report['stations']}; "
        f"gridded cells: {report['gridded_cells']}\n"
    ]
    for ref, scores in report["scores"].items():
        names = list(scores)
        first = scores[names[0]]
        text.append(f"\n### Against {ref}\n")
        text.append("| Metric | " + " | ".join(names) + " | Reference |")
        text.append("|---|" + "---:|" * (len(names) + 1))
        for label, group, key in rows:
            if group not in first or key not in first[group]:
                continue
            values = [scores[n][group][key] for n in names]
            ref_value = first[group].get(f"reference_{key}", "")

            def fmt(v):
                return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

            text.append(
                f"| {label} | "
                + " | ".join(fmt(v) for v in values)
                + f" | {fmt(ref_value) if ref_value != '' else ''} |"
            )
    return "\n".join(text) + "\n"


def figures(report, models, radar, pairs, out):
    """FSS vs scale, spectra, wet-tail Q-Q and maps of the wettest hour."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "ERA5 interval mean": "#999999",
        "ERA5 smooth (PCHIP)": "#555555",
        "PARDIG": "#1f77b4",
        "Downscaler (members)": "#d62728",
        "Downscaler (ensemble mean)": "#ff9896",
        "EURADCLIM": "black",
    }

    grid_ref = "EURADCLIM" if "EURADCLIM" in report["scores"] else "PARDIG"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, t in zip(axes, ("1mm", "5mm")):
        for name, s in report["scores"][grid_ref].items():
            ax.plot(s["structure"]["sizes_km"], s["structure"][f"fss_{t}"], "o-", label=name, color=colors.get(name))
        ax.axhline(next(iter(report["scores"][grid_ref].values()))["structure"][f"useful_fss_{t}"], ls=":", c="k")
        ax.set(
            xscale="log", xlabel="neighbourhood size (km)", ylabel="FSS", title=f"FSS vs {grid_ref}, ≥{t[:-2]} mm/h"
        )
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fss_vs_scale.png", dpi=150)

    if "spectra" in report:
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, s in report["spectra"].items():
            ax.loglog(s["wavelength_km"], s["power"], label=name, color=colors.get(name))
        ax.invert_xaxis()
        ax.set(xlabel="wavelength (km)", ylabel="power (mm²)", title="Spatial power spectrum")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "spectrum.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ref in zip(axes, (grid_ref, "Gauges")):
        top = 0
        for name, s in report["scores"][ref].items():
            ax.plot(s["qq"]["reference"], s["qq"]["forecast"], label=name, color=colors.get(name))
            top = max(top, max(s["qq"]["reference"]), max(s["qq"]["forecast"]))
        ax.plot([0, top], [0, top], "k:")
        ax.set(
            xlabel=f"{ref} quantile (mm/h)", ylabel="forecast quantile (mm/h)", title=f"Wet tail (90-99.99%) vs {ref}"
        )
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "qq.png", dpi=150)

    wettest = int(np.nanargmax(np.nanmean(radar, (1, 2))))
    panels = {
        "ERA5 interval mean": models["ERA5 interval mean"][wettest, 0],
        "Downscaler (one member)": models["Downscaler (members)"][wettest, 0],
        "Downscaler (ensemble mean)": models["Downscaler (ensemble mean)"][wettest, 0],
        "PARDIG": models["PARDIG"][wettest, 0],
        grid_ref: radar[wettest],
    }
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), constrained_layout=True)
    extent = [pairs.lon.values[0], pairs.lon.values[-1], pairs.lat.values[0], pairs.lat.values[-1]]
    vmax = max(1.0, float(np.nanquantile(radar[wettest], 0.995)))
    for ax, (name, field) in zip(axes, panels.items()):
        image = ax.imshow(field, origin="lower", extent=extent, vmin=0, vmax=vmax, cmap="Blues")
        ax.set_title(name, fontsize=9)
    fig.colorbar(image, ax=axes, shrink=0.8, label="mm/h")
    fig.suptitle(f"Hour ending {pd.Timestamp(pairs.time.values[wettest]):%Y-%m-%d %H:%M} UTC")
    fig.savefig(out / "maps.png", dpi=130)


if __name__ == "__main__":
    main()
