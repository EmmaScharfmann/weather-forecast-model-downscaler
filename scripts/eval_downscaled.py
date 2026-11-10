"""Compare the coarse ensemble with its downscaled version at station locations.

Usage::

    python scripts/eval_downscaled.py --config config_spatial.yaml

Three series are read at every station and forecast hour: the raw coarse
ensemble, the downscaled ensemble, and the raw field after interpolation onto
the fine grid. The third exists only as a check that the two reading paths
agree, since it should match the first closely.
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

from downscale import station_interp as si
from downscale import station_scoring as ss
from downscale.config import load_config
from downscale.constants import SCORING_REGIONS, elevation_edges
from downscale.diagnostics import (
    DistributionSamples,
    StationErrors,
    edge_figure,
    edge_profile,
    coast_figure,
    coast_profile,
    gain_profile,
    grid_spacing_km,
    improvement_figure,
    land_mask,
    station_distance_to_coast,
    mask_sea,
    quantile_figure,
    sea_report,
    station_error_figure,
    suggested_margin,
)
from downscale.stations import MeteoFranceHourly, QCConfig, load_stations

logger = logging.getLogger("eval")

METHODS = ["coarse", "downscaled", "coarse_on_fine_grid"]
LABELS = {"coarse": "coarse ensemble", "downscaled": "coarse ensemble + downscaler"}


def at_stations(forecast_path: Path, downscaled_path: Path, stations: xr.Dataset, variable: str) -> xr.Dataset:
    """Read both forecasts at the station locations.

    :param forecast_path: Coarse forecast store.
    :type forecast_path: pathlib.Path
    :param downscaled_path: Matching downscaled store.
    :type downscaled_path: pathlib.Path
    :param stations: Station dataset.
    :type stations: xarray.Dataset
    :param variable: Variable to read from the coarse forecast.
    :type variable: str
    :returns: Dataset with one variable per method over ``(init_time, lead_time, ensemble*, station)``.
    :rtype: xarray.Dataset
    :raises ValueError: If the two stores disagree on times or member count.
    """
    forecast = xr.open_zarr(forecast_path)[variable]
    if "ensemble" not in forecast.dims:
        forecast = forecast.expand_dims(ensemble=[0])
    forecast = forecast.transpose("time", "lead_time", "ensemble", "lat", "lon").load().rename({"time": "init_time"})
    raw = si.interpolator_for(forecast, stations).apply(forecast).drop_vars("ensemble")

    downscaled = xr.open_zarr(downscaled_path)
    same_times = np.array_equal(downscaled["time"].values, forecast["init_time"].values) and np.array_equal(
        downscaled["lead_time"].values, forecast["lead_time"].values
    )
    if not same_times:
        raise ValueError(f"{downscaled_path.name}: times differ from {forecast_path.name}")
    if downscaled.sizes["ensemble"] != forecast.sizes["ensemble"]:
        raise ValueError(
            f"{downscaled_path.name} holds {downscaled.sizes['ensemble']} members but "
            f"{forecast_path.name} has {forecast.sizes['ensemble']}: the downscaled store is stale, "
            "delete it and rerun downscale_forecasts.py"
        )
    if not Path(str(forecast_path).rstrip("/") + ".done").exists():
        logger.warning("%s has no .done marker: the forecast run may be incomplete", forecast_path.name)

    interp = si.curvilinear_for(downscaled["lat"].values, downscaled["lon"].values, stations, k=4)
    if not interp.inside.all():
        logger.warning("%d station(s) lie beyond the fine grid and become NaN", int((~interp.inside).sum()))
    fine = interp.apply(downscaled["t2m_ds"].load().rename({"time": "init_time"}))
    fine = fine.stack(member=("ensemble", "sample")).transpose("init_time", "lead_time", "member", "station")
    fine = fine.drop_vars(["member", "ensemble", "sample"]).rename({"member": "ensemble_ds"})
    on_fine_grid = interp.apply(downscaled["t2m_interp"].load().rename({"time": "init_time"})).drop_vars("ensemble")

    out = xr.Dataset({"coarse": raw, "downscaled": fine, "coarse_on_fine_grid": on_fine_grid})
    out = out.assign_coords(station=stations["station"].values)
    for coord in ("latitude", "longitude", "elevation"):
        out = out.assign_coords({coord: ("station", stations[coord].values)})
    return out.assign_coords(
        valid_time=(
            ("init_time", "lead_time"),
            out["init_time"].values[:, None] + out["lead_time"].values[None, :],
        )
    )


class EventScores:
    """Brier scores and hit rates for exceedance of fixed thresholds."""

    def __init__(self, thresholds_k: Sequence[float]):
        """
        :param thresholds_k: Thresholds [K].
        :type thresholds_k: collections.abc.Sequence[float]
        """
        self.thresholds = list(thresholds_k)
        self.sums: dict[tuple[str, float, int], np.ndarray] = {}

    def add(self, forecast_at_stations: xr.Dataset, obs: xr.DataArray, methods: Sequence[str]) -> None:
        """Accumulate one initialisation.

        :param forecast_at_stations: Forecasts read at stations.
        :type forecast_at_stations: xarray.Dataset
        :param obs: Observations over ``(init_time, lead_time, station)``.
        :type obs: xarray.DataArray
        :param methods: Method names to score.
        :type methods: collections.abc.Sequence[str]
        :returns: ``None``
        :rtype: None
        """
        lead_hours = (forecast_at_stations["lead_time"].values / np.timedelta64(1, "h")).astype(int)
        truth = obs.values
        for method in methods:
            values = forecast_at_stations[method].values
            for threshold in self.thresholds:
                probability = (values >= threshold).mean(2)
                observed = truth >= threshold
                for j, hour in enumerate(lead_hours):
                    valid = np.isfinite(truth[:, j]) & np.isfinite(probability[:, j])
                    p, o = probability[:, j][valid], observed[:, j][valid]
                    entry = self.sums.setdefault((method, threshold, int(hour)), np.zeros(6))
                    entry += [
                        ((p - o) ** 2).sum(), valid.sum(), ((p >= 0.5) & o).sum(),
                        ((p < 0.5) & o).sum(), ((p >= 0.5) & ~o).sum(), o.sum(),
                    ]

    def table(self) -> pd.DataFrame:
        """Turn the sums into a table.

        :returns: One row per method, threshold and lead time.
        :rtype: pandas.DataFrame
        """
        rows = []
        for (method, threshold, hour), (brier, n, hits, misses, false_alarms, events) in self.sums.items():
            rows.append({
                "method": method, "threshold_C": round(threshold - 273.15, 1), "lead_h": hour,
                "n": int(n), "obs_events": int(events),
                "brier": brier / n if n else np.nan,
                "hit_rate": hits / (hits + misses) if hits + misses else np.nan,
                "false_alarm_ratio": false_alarms / (hits + false_alarms) if hits + false_alarms else np.nan,
            })
        return pd.DataFrame(rows).sort_values(["threshold_C", "method", "lead_h"])


def scorecard(results, binned, intervals, events, stations, edges, by_lead, path: Path) -> None:
    """Draw the six-panel scorecard.

    :param results: Metrics per method over lead time.
    :type results: dict[str, xarray.Dataset]
    :param binned: Metrics per method, region and elevation band.
    :type binned: dict[str, xarray.Dataset]
    :param intervals: Bootstrap bounds per method, or ``None``.
    :type intervals: dict | None
    :param events: Event score table.
    :type events: pandas.DataFrame
    :param stations: Station dataset, for the per-band counts.
    :type stations: xarray.Dataset
    :param edges: Elevation band edges [m].
    :type edges: numpy.ndarray
    :param by_lead: Long table of metrics by lead time, carrying ``valid_hour``.
    :type by_lead: pandas.DataFrame
    :param path: Destination image.
    :type path: pathlib.Path
    :returns: ``None``
    :rtype: None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hours = lambda ds: ds["lead_time"].values / np.timedelta64(1, "h")
    colour = {"coarse": "tab:blue", "downscaled": "tab:red"}
    fig, ax = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)

    for metric, axis, title in (
        ("rmse", ax[0, 0], "RMSE of the ensemble mean against stations (K)"),
        ("crps", ax[0, 1], "fair CRPS against stations (K)"),
        ("bias", ax[1, 0], "bias, forecast minus observation (K)"),
    ):
        for method in ("coarse", "downscaled"):
            axis.plot(
                hours(results[method]), results[method][metric].values,
                color=colour[method], marker="o", ms=3, label=LABELS[method],
            )
            if intervals:
                lower, upper = intervals[method]
                axis.fill_between(
                    hours(results[method]), lower[metric].values, upper[metric].values,
                    color=colour[method], alpha=0.15,
                )
        axis.set_title(title)
        axis.set_xlabel("lead time (h)")
        axis.grid(alpha=0.3)
    ax[1, 0].axhline(0, color="k", lw=0.8)
    ax[0, 0].legend(fontsize=8)

    hourly = by_lead.dropna(subset=["valid_hour"])
    if len(hourly):
        for metric, marker, style in (("bias", "o", "-"), ("rmse", "s", "--")):
            for method in ("coarse", "downscaled"):
                grouped = (
                    hourly[hourly.method == method].groupby("valid_hour")[metric].mean().sort_index()
                )
                ax[1, 1].plot(
                    grouped.index, grouped.values, color=colour[method], marker=marker, ls=style,
                    label=f"{LABELS[method]}, {metric}",
                )
        ax[1, 1].axhline(0, color="k", lw=0.8)
        ax[1, 1].set_xticks(sorted(hourly.valid_hour.unique()))
        ax[1, 1].set_xlabel("valid hour (UTC)")
        ax[1, 1].set_title("bias and RMSE by time of day, averaged over lead times (K)")
        ax[1, 1].grid(alpha=0.3)
        ax[1, 1].legend(fontsize=7)
    else:
        ax[1, 1].axis("off")

    (lat_min, lat_max), (lon_min, lon_max) = SCORING_REGIONS["france"]
    in_box = (
        (stations.latitude >= lat_min) & (stations.latitude <= lat_max)
        & (stations.longitude >= lon_min) & (stations.longitude <= lon_max)
    ).values
    counts = [
        int((in_box & (stations.elevation.values >= low) & (stations.elevation.values < high)).sum())
        for low, high in zip(edges[:-1], edges[1:])
    ]
    for method in ("coarse", "downscaled"):
        values = binned[method].rmse.sel(region="france").mean("lead_time")
        ax[2, 0].plot(range(len(values)), values.values, marker="o", color=colour[method], label=LABELS[method])
    ax[2, 0].set_xticks(range(len(counts)))
    ax[2, 0].set_xticklabels(
        [f"{int(low)}-{int(high)} m\nn={n}" for low, high, n in zip(edges[:-1], edges[1:], counts)], fontsize=7
    )
    ax[2, 0].set_title("RMSE by station elevation, averaged over lead times")
    ax[2, 0].grid(alpha=0.3)
    ax[2, 0].legend(fontsize=8)

    hottest = events[events.threshold_C == events.threshold_C.max()]
    active = hottest[hottest.obs_events > 0]
    plotted = active if len(active) else hottest
    for method in ("coarse", "downscaled"):
        subset = plotted[plotted.method == method].sort_values("lead_h")
        ax[2, 1].plot(
            subset.lead_h, subset.brier, color=colour[method], marker="o", ms=4, ls="none", label=LABELS[method]
        )
    ax[2, 1].set_title(
        f"Brier score, station temperature at or above {hottest.threshold_C.max():.0f} degC\n"
        "(lead times where the event was observed)"
    )
    ax[2, 1].set_xlabel("lead time (h)")
    ax[2, 1].grid(alpha=0.3)
    ax[2, 1].legend(fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def heatwave_map(peak, stations: xr.Dataset, variable: str, path: Path) -> None:
    """Draw both forecasts at the hottest observed hour, with the observations.

    :param peak: Tuple of the observed mean, the downscaled store and the lead index.
    :type peak: tuple
    :param stations: Station dataset.
    :type stations: xarray.Dataset
    :param variable: Observed variable name.
    :type variable: str
    :param path: Destination image.
    :type path: pathlib.Path
    :returns: ``None``
    :rtype: None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, downscaled_path, index = peak
    ds = xr.open_zarr(downscaled_path)
    valid = pd.Timestamp(ds["valid_time"].isel(time=0, lead_time=index).values)
    mask = land_mask(ds)
    coarse = mask_sea(ds["t2m_interp"].isel(time=0, lead_time=index).mean("ensemble").values - 273.15, mask)
    fine = mask_sea(ds["t2m_ds"].isel(time=0, lead_time=index).mean(("ensemble", "sample")).values - 273.15, mask)
    observed = stations[variable].sel(time=valid, method="nearest").values - 273.15
    finite_fine = fine[np.isfinite(fine)]
    low, high = np.nanpercentile(
        np.concatenate([finite_fine, observed[np.isfinite(observed)]]), [2, 98]
    )
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, (title, field) in zip(
        ax, [("coarse ensemble mean, interpolated", coarse), ("downscaled ensemble mean", fine)]
    ):
        mesh = axis.pcolormesh(ds.lon, ds.lat, field, cmap="RdYlBu_r", vmin=low, vmax=high, shading="auto")
        axis.scatter(
            stations.longitude, stations.latitude, c=observed, cmap="RdYlBu_r",
            vmin=low, vmax=high, s=9, edgecolors="k", linewidths=0.2,
        )
        axis.set_title(f"{title}\n(dots: station observations)")
        axis.set_aspect(1 / np.cos(np.deg2rad(float(np.mean(ds.lat.values)))))
        fig.colorbar(mesh, ax=axis, label="degC")
    lead_hours = int(ds.lead_time.values[index] / np.timedelta64(1, "h"))
    fig.suptitle(
        f"valid {valid:%Y-%m-%d %H} UTC, initialised "
        f"{pd.Timestamp(ds.time.values[0]):%Y-%m-%d} (+{lead_hours} h)"
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(cfg, forecast_paths, downscaled_paths, out_dir, provider=None) -> dict:
    """Score both forecasts and write the tables and figures.

    :param cfg: Pipeline configuration.
    :type cfg: downscale.config.Config
    :param forecast_paths: Coarse forecast stores.
    :type forecast_paths: collections.abc.Sequence[str]
    :param downscaled_paths: Matching downscaled stores.
    :type downscaled_paths: collections.abc.Sequence[str]
    :param out_dir: Output directory.
    :type out_dir: str | pathlib.Path
    :param provider: Station provider; ``None`` builds the configured one.
    :type provider: downscale.stations.StationProvider | None
    :returns: Summary of the comparison, also written to ``summary.json``.
    :rtype: dict
    :raises FileNotFoundError: If a forecast has no matching downscaled store.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    variable = cfg.coarse.variable
    pairs = []
    for forecast in sorted(forecast_paths):
        matches = [d for d in downscaled_paths if Path(d).name == Path(forecast).name]
        if not matches:
            raise FileNotFoundError(f"no downscaled store named {Path(forecast).name}")
        pairs.append((Path(forecast), Path(matches[0])))

    reference = xr.open_zarr(pairs[0][0])
    inits = pd.DatetimeIndex([xr.open_zarr(f).time.values[0] for f, _ in pairs])
    start = inits.min().floor("h")
    end = (inits.max() + pd.Timedelta(reference.lead_time.values[-1])).floor("h")
    if cfg.stations.window:
        start, end = (pd.Timestamp(x) for x in cfg.stations.window)
    stations, _ = load_stations(
        provider or MeteoFranceHourly(), cfg.region_box, start, end,
        variables=cfg.stations.variables, cache_dir=cfg.paths.stations_cache,
        qc=QCConfig(min_completeness=cfg.stations.min_completeness, primary=cfg.stations.variables[0]),
        holdout_frac=cfg.stations.holdout_frac, active=cfg.stations.active,
        include_unknown_start=cfg.stations.include_unknown_start,
    )
    logger.info(
        "%d stations, %d initialisations, window %s .. %s", stations.sizes["station"], len(pairs), start, end
    )

    edges = elevation_edges(cfg.evaluation.elevation_edges)
    bins = ss.default_bins(edges=cfg.evaluation.elevation_edges)
    thresholds = [t + 273.15 for t in cfg.evaluation.heat_thresholds_c]
    cases_all, cases_binned, events = [], [], EventScores(thresholds)
    station_errors = StationErrors(["coarse", "downscaled"])
    distributions = DistributionSamples(["coarse", "downscaled"])
    peak = None
    for forecast_path, downscaled_path in pairs:
        at = at_stations(forecast_path, downscaled_path, stations, variable)
        obs = si.observations_at_valid_times(at, stations, cfg.stations.variables[0])
        if peak is None:
            difference = (at["coarse_on_fine_grid"] - at["coarse"]).values
            logger.info(
                "reading-path check: mean absolute difference %.2f K, 95th percentile %.2f K",
                np.nanmean(np.abs(difference)), np.nanpercentile(np.abs(difference), 95),
            )
        cases_all.append(
            ss.score_station_case(
                at, obs, variable, methods=METHODS, bin_by=None,
                density_alpha_deg=cfg.evaluation.density_alpha_deg,
            )
        )
        cases_binned.append(
            ss.score_station_case(
                at, obs, variable, methods=METHODS, bin_by=bins,
                density_alpha_deg=cfg.evaluation.density_alpha_deg,
            )
        )
        events.add(at, obs, ["coarse", "downscaled"])
        station_errors.add(at, obs)
        distributions.add(at, obs)
        observed_mean = np.nanmean(obs.values, axis=2)
        hottest = np.unravel_index(np.nanargmax(observed_mean), observed_mean.shape)
        if peak is None or observed_mean[hottest] > peak[0]:
            peak = (observed_mean[hottest], downscaled_path, int(hottest[1]))
        logger.info("scored %s", forecast_path.name)

    results, binned = ss.combine_cases(cases_all), ss.combine_cases(cases_binned)
    intervals = (
        {
            m: ss.bootstrap_ci(
                cases_all, m, n_boot=cfg.evaluation.n_bootstrap, block=cfg.evaluation.bootstrap_block
            )
            for m in ("coarse", "downscaled")
        }
        if len(cases_all) >= 3 else None
    )

    by_lead = pd.concat({m: results[m].to_dataframe() for m in METHODS}, names=["method"]).reset_index()
    by_lead["lead_h"] = (pd.to_timedelta(by_lead.pop("lead_time")).dt.total_seconds() / 3600).astype(int)
    by_lead["valid_hour"] = (by_lead["lead_h"] + inits[0].hour) % 24 if inits.hour.nunique() == 1 else np.nan
    by_lead.to_csv(out / "by_lead.csv", index=False)
    pd.concat({m: binned[m].to_dataframe() for m in METHODS}, names=["method"]).reset_index().to_csv(
        out / "by_region_elevation.csv", index=False
    )
    event_table = events.table()
    event_table.to_csv(out / "heat_events.csv", index=False)

    coarse_scores, fine_scores = results["coarse"], results["downscaled"]
    first_downscaled = xr.open_zarr(pairs[0][1])
    summary = {
        "n_inits": len(pairs),
        "n_stations": int(stations.sizes["station"]),
        "members": {
            "coarse": int(xr.open_zarr(pairs[0][0]).sizes.get("ensemble", 1)),
            "downscaled": int(first_downscaled.sizes["ensemble"] * first_downscaled.sizes["sample"]),
        },
        "rmse_K": {"coarse": float(coarse_scores.rmse.mean()), "downscaled": float(fine_scores.rmse.mean())},
        "crps_K": {"coarse": float(coarse_scores.crps.mean()), "downscaled": float(fine_scores.crps.mean())},
        "bias_K": {"coarse": float(coarse_scores.bias.mean()), "downscaled": float(fine_scores.bias.mean())},
        "ssr": {"coarse": float(coarse_scores.ssr.mean()), "downscaled": float(fine_scores.ssr.mean())},
    }
    summary["rmse_change_pct"] = 100 * (summary["rmse_K"]["downscaled"] / summary["rmse_K"]["coarse"] - 1)
    summary["crps_change_pct"] = 100 * (summary["crps_K"]["downscaled"] / summary["crps_K"]["coarse"] - 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 160)
    shown = [h for h in cfg.evaluation.lead_hours_shown if h in by_lead.lead_h.values]
    print("\ncoarse ensemble against coarse ensemble plus downscaler, all stations (K)")
    for metric in ("rmse", "crps", "ssr"):
        table = by_lead.pivot(index="lead_h", columns="method", values=metric)
        print(f"\n{metric.upper()}\n{table.loc[shown, ['coarse', 'downscaled']].round(3).to_string()}")
    print(
        f"\noverall: RMSE {summary['rmse_K']['coarse']:.2f} -> {summary['rmse_K']['downscaled']:.2f} K "
        f"({summary['rmse_change_pct']:+.0f}%), CRPS {summary['crps_K']['coarse']:.2f} -> "
        f"{summary['crps_K']['downscaled']:.2f} K ({summary['crps_change_pct']:+.0f}%), "
        f"SSR {summary['ssr']['coarse']:.2f} -> {summary['ssr']['downscaled']:.2f}"
    )

    first_downscaled_store = xr.open_zarr(pairs[0][1])
    diagnostics = {"sea": sea_report(first_downscaled_store)}
    try:
        profile = edge_profile(first_downscaled_store)
        margin = suggested_margin(profile)
        profile.to_csv(out / "edge_profile.csv", index=False)
        edge_figure(profile, margin, out / "edge_check.png")
        diagnostics["edge"] = {
            "suggested_margin_cells": margin,
            "interior_spread_K": profile.attrs["interior_spread_K"],
            "interior_correction_K": profile.attrs["interior_correction_K"],
        }
        logger.info("edge effect reaches %d grid cell(s) into the domain", margin)
    except ValueError as exc:
        logger.warning("edge profile skipped: %s", exc)
    if diagnostics["sea"]:
        logger.info(
            "correction over sea is %.1f times that over land",
            diagnostics["sea"]["correction_ratio_sea_over_land"],
        )
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    per_station = station_errors.table()
    per_station.to_csv(out / "station_errors.csv", index=False)
    station_error_figure(per_station, ["coarse", "downscaled"], LABELS, out / "station_errors.png")
    first_store = xr.open_zarr(pairs[0][1])
    diagnostics["improvement"] = improvement_figure(
        per_station, "coarse", "downscaled",
        first_store["orography"].values if "orography" in first_store else None,
        first_store["lat"].values if "orography" in first_store else None,
        first_store["lon"].values if "orography" in first_store else None,
        out / "improvement.png",
    )
    coast_mask = land_mask(first_store)
    if coast_mask is not None:
        spacing = grid_spacing_km(first_store["lat"].values, first_store["lon"].values)
        distance = station_distance_to_coast(
            per_station, coast_mask, first_store["lat"].values, first_store["lon"].values, spacing
        )
        per_station["distance_to_coast_km"] = distance
        per_station.to_csv(out / "station_errors.csv", index=False)
        profile = coast_profile(per_station, distance, "coarse", "downscaled")
        profile.to_csv(out / "coast_profile.csv", index=False)
        diagnostics["coast"] = coast_figure(per_station, distance, profile, out / "coast.png")
        print("\ngain from downscaling by distance from the coast")
        print(profile.to_string(index=False))

    elevation_bands = gain_profile(per_station, "elevation", list(edges[:-1]) + [1e6])
    elevation_bands.to_csv(out / "gain_by_elevation.csv", index=False)
    print("\ngain from downscaling by station elevation")
    print(elevation_bands.to_string(index=False))

    quantiles = distributions.quantiles()
    quantiles.to_csv(out / "quantiles.csv", index=False)
    quantile_figure(quantiles, ["coarse", "downscaled"], LABELS, out / "quantiles.png")
    diagnostics["extremes"] = {
        "observed_99.9th_degC": round(float(quantiles.loc[quantiles.level == 0.999, "observed"].iloc[0] - 273.15), 2),
        **{
            f"{method}_{kind}_99.9th_degC": round(
                float(quantiles.loc[quantiles.level == 0.999, f"{method} {kind}"].iloc[0] - 273.15), 2
            )
            for method in ("coarse", "downscaled") for kind in ("member", "mean")
        },
    }
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    print("\nwhere downscaling helps")
    for key, value in diagnostics["improvement"].items():
        print(f"  {key:34s} {value}")
    print("\nhighest temperatures, 99.9th percentile over all stations and times (degC)")
    for key, value in diagnostics["extremes"].items():
        print(f"  {key:34s} {value}")
    afternoon = event_table[(event_table.obs_events > 0)]
    if len(afternoon):
        summary_events = afternoon.groupby(["threshold_C", "method"]).agg(
            events=("obs_events", "sum"), hit_rate=("hit_rate", "mean"),
            false_alarm_ratio=("false_alarm_ratio", "mean"), brier=("brier", "mean"),
        ).round(3)
        print("\nheat events, averaged over the lead times where they occurred")
        print(summary_events.to_string())

    scorecard(results, binned, intervals, event_table, stations, edges, by_lead, out / "scorecard.png")
    heatwave_map(peak, stations, cfg.stations.variables[0], out / "heatwave_map.png")
    print(f"\nwrote {out}/")
    return summary


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
    parser.add_argument("--downscaled", nargs="+", default=None, help="overrides paths.downscaled")
    parser.add_argument("--out", default=None, help="overrides paths.evaluation")
    parser.add_argument("--station-window", default=None, metavar="START,END")
    parser.add_argument("--n-bootstrap", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("wetterdienst").setLevel(logging.WARNING)

    cfg = load_config(args.config)
    if args.station_window:
        cfg.stations.window = args.station_window.split(",")
    if args.n_bootstrap is not None:
        cfg.evaluation.n_bootstrap = args.n_bootstrap
    forecasts = args.forecasts or sorted(str(p) for p in Path(cfg.paths.forecasts).glob("*.zarr"))
    downscaled = args.downscaled or sorted(str(p) for p in Path(cfg.paths.downscaled).glob("*.zarr"))
    if not forecasts or not downscaled:
        parser.error("no forecasts or downscaled stores found; pass the paths or set them in the configuration")
    run(cfg, forecasts, downscaled, args.out or cfg.paths.evaluation)
    return 0


if __name__ == "__main__":
    sys.exit(main())