"""Scoring of forecasts against station observations."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from downscaler.constants import ENSEMBLE_DIM, ENSEMBLE_METRICS, SCORING_REGIONS, elevation_edges


logger = logging.getLogger(__name__)


def build_metrics(ensemble: bool) -> dict:
    """Build the metric set.

    :param ensemble: Include ensemble metrics, which need more than one member.
    :type ensemble: bool
    :returns: Metric names mapped to WeatherBench-X metric objects.
    :rtype: dict
    """
    from weatherbenchX.metrics import deterministic as det
    from weatherbenchX.metrics import probabilistic as prob

    metrics = {"rmse": det.RMSE(), "mae": det.MAE(), "bias": det.Bias()}
    if ensemble:
        metrics["crps"] = prob.CRPSEnsemble(ensemble_dim=ENSEMBLE_DIM, fair=True)
        metrics["ens_mean_rmse"] = prob.UnbiasedEnsembleMeanRMSE(ensemble_dim=ENSEMBLE_DIM)
        metrics["spread"] = prob.EnsembleRootMeanVariance(ensemble_dim=ENSEMBLE_DIM)
        metrics["ssr"] = prob.UnbiasedSpreadSkillRatio(ensemble_dim=ENSEMBLE_DIM)
    return metrics


def default_bins(regions: Mapping | None = None, edges: Sequence[float] | None = None) -> list:
    """Build the binning used to stratify scores.

    :param regions: Named sub-regions; ``None`` uses
        :data:`~downscaler.constants.SCORING_REGIONS`, a falsy value drops region binning.
    :type regions: collections.abc.Mapping | None
    :param edges: Elevation band edges [m]; ``None`` uses the defaults.
    :type edges: collections.abc.Sequence[float] | None
    :returns: WeatherBench-X binning objects, multiplied together when both are present.
    :rtype: list
    """
    from weatherbenchX import binning

    bins = []
    regions = SCORING_REGIONS if regions is None else regions
    if regions:
        bins.append(binning.Regions(dict(regions)))
    bins.append(binning.ByCoordBins("elevation", elevation_edges(list(edges) if edges is not None else None)))
    return bins


def select_stations(*arrays: xr.Dataset | xr.DataArray, which: str = "all", holdout=None):
    """Restrict arrays to a subset of stations.

    :param arrays: Objects carrying a ``station`` dimension.
    :type arrays: xarray.Dataset | xarray.DataArray
    :param which: ``"all"``, ``"holdout"`` or ``"train"``.
    :type which: str
    :param holdout: Boolean mask flagging held-out stations.
    :type holdout: xarray.DataArray | numpy.ndarray | None
    :returns: The subset arrays, unwrapped when a single array was given.
    :rtype: tuple | xarray.Dataset | xarray.DataArray
    :raises ValueError: If a subset is requested without a mask, or ``which`` is unknown.
    """
    if which == "all":
        return arrays if len(arrays) > 1 else arrays[0]
    if holdout is None:
        raise ValueError("a holdout mask is required for which='holdout' or 'train'")
    mask = np.asarray(holdout, dtype=bool)
    if which == "train":
        mask = ~mask
    elif which != "holdout":
        raise ValueError("which must be 'all', 'holdout' or 'train'")
    out = tuple(a.isel(station=np.flatnonzero(mask)) for a in arrays)
    return out if len(out) > 1 else out[0]


@dataclass
class StationScores:
    """Accumulated statistics for one initialisation, one entry per method."""

    states: dict
    ensemble: dict
    n_obs: dict = field(default_factory=dict)


def _scoring_inputs(da: xr.DataArray, like: xr.Dataset) -> xr.DataArray:
    """Prepare an array for WeatherBench-X.

    Drops the two-dimensional ``valid_time`` coordinate, which cannot survive
    reduction over initialisations, and restores the station coordinates used
    for binning and weighting.

    :param da: Array over ``(init_time, lead_time[, ensemble], station)``.
    :type da: xarray.DataArray
    :param like: Object carrying the station coordinates.
    :type like: xarray.Dataset
    :returns: Array ready for scoring.
    :rtype: xarray.DataArray
    """
    da = da.drop_vars([c for c in ("valid_time",) if c in da.coords])
    for coord in ("latitude", "longitude", "elevation"):
        if coord not in da.coords:
            da = da.assign_coords({coord: ("station", like[coord].values)})
    return da


def score_station_case(
    forecast_at_stations: xr.Dataset,
    obs: xr.DataArray,
    variable: str = "t2m",
    analysis: xr.Dataset | None = None,
    methods: Sequence[str] | None = None,
    bin_by: Sequence | str | None = "default",
    density_alpha_deg: float | None = 0.75,
) -> StationScores:
    """Score every method of one initialisation against observations.

    :param forecast_at_stations: Forecasts read at stations, one variable per method.
    :type forecast_at_stations: xarray.Dataset
    :param obs: Observations laid out over ``(init_time, lead_time, station)``.
    :type obs: xarray.DataArray
    :param variable: Name under which the variable is registered with the metrics.
    :type variable: str
    :param analysis: Analysis baseline, added as ``analysis_<method>`` entries.
    :type analysis: xarray.Dataset | None
    :param methods: Method names to score; ``None`` scores the baseline ladder present.
    :type methods: collections.abc.Sequence[str] | None
    :param bin_by: ``"default"`` for region and elevation bins, a list of binning objects, or ``None``.
    :type bin_by: collections.abc.Sequence | str | None
    :param density_alpha_deg: Station density weighting scale [degrees]; ``None`` weights equally.
    :type density_alpha_deg: float | None
    :returns: Accumulated statistics for this initialisation.
    :rtype: StationScores
    """
    from weatherbenchX import aggregation, weighting
    from weatherbenchX.metrics import base as metrics_base

    bins = default_bins() if isinstance(bin_by, str) else bin_by
    methods = list(methods or [m for m in ("nearest", "bilinear", "bilinear_lapse") if m in forecast_at_stations])
    obs = _scoring_inputs(obs, forecast_at_stations)
    weights = [weighting.StationDensityWeighting(alpha_0_degrees=density_alpha_deg)] if density_alpha_deg else None
    aggregator = aggregation.Aggregator(
        reduce_dims=["init_time", "station"], bin_by=bins, weigh_by=weights, skipna=True
    )

    jobs = {m: forecast_at_stations[m] for m in methods}
    if analysis is not None:
        jobs.update({f"analysis_{m}": analysis[m] for m in ("nearest", "bilinear", "bilinear_lapse") if m in analysis})

    states, ensemble_flag, n_obs = {}, {}, {}
    for name, da in jobs.items():
        da = _scoring_inputs(da, forecast_at_stations)
        ensemble_dims = [d for d in da.dims if str(d).startswith(ENSEMBLE_DIM)]
        if ensemble_dims and ensemble_dims[0] != ENSEMBLE_DIM:
            da = da.rename({ensemble_dims[0]: ENSEMBLE_DIM})
        if ENSEMBLE_DIM in da.coords:
            da = da.drop_vars(ENSEMBLE_DIM)
        is_ensemble = ENSEMBLE_DIM in da.dims
        metrics = build_metrics(is_ensemble)
        deterministic = {k: v for k, v in metrics.items() if k not in ENSEMBLE_METRICS}
        stats = dict(
            metrics_base.compute_unique_statistics_for_all_metrics(
                deterministic,
                {variable: da.mean(ENSEMBLE_DIM) if is_ensemble else da},
                {variable: obs},
            )
        )
        if is_ensemble:
            probabilistic = {k: v for k, v in metrics.items() if k in ENSEMBLE_METRICS}
            stats.update(
                metrics_base.compute_unique_statistics_for_all_metrics(probabilistic, {variable: da}, {variable: obs})
            )
        states[name] = aggregator.aggregate_statistics(stats)
        ensemble_flag[name] = is_ensemble
        n_obs[name] = int(np.isfinite(obs.values).sum())
    return StationScores(states, ensemble_flag, n_obs)


def _values(state, ensemble: bool) -> xr.Dataset:
    """Turn accumulated statistics into metric values.

    :param state: WeatherBench-X aggregation state.
    :type state: weatherbenchX.aggregation.AggregationState
    :param ensemble: Whether ensemble metrics were accumulated.
    :type ensemble: bool
    :returns: Metrics named without the variable suffix.
    :rtype: xarray.Dataset
    """
    ds = state.metric_values(build_metrics(ensemble))
    return ds.rename({k: k.split(".")[0] for k in ds.data_vars})


def combine_cases(cases: Sequence[StationScores]) -> dict[str, xr.Dataset]:
    """Combine several initialisations into final scores.

    :param cases: Per-initialisation statistics.
    :type cases: collections.abc.Sequence[StationScores]
    :returns: Method names mapped to their metrics.
    :rtype: dict[str, xarray.Dataset]
    """
    from weatherbenchX import aggregation

    out = {}
    for name in cases[0].states:
        total = aggregation.AggregationState.sum([c.states[name] for c in cases])
        out[name] = _values(total, cases[0].ensemble[name])
    return out


def bootstrap_ci(
    cases: Sequence[StationScores],
    method: str,
    n_boot: int = 300,
    block: int = 2,
    seed: int = 0,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> tuple[xr.Dataset, xr.Dataset]:
    """Estimate confidence intervals by resampling initialisations in blocks.

    Neighbouring initialisations share weather, so blocks of consecutive cases
    are resampled together. With few cases the intervals describe sampling noise
    within the period studied, not uncertainty about the method in general.

    :param cases: Per-initialisation statistics.
    :type cases: collections.abc.Sequence[StationScores]
    :param method: Method to resample.
    :type method: str
    :param n_boot: Number of bootstrap draws.
    :type n_boot: int
    :param block: Consecutive cases resampled together.
    :type block: int
    :param seed: Random seed.
    :type seed: int
    :param quantiles: Lower and upper quantiles of the interval.
    :type quantiles: tuple[float, float]
    :returns: Lower and upper bounds, with the same variables as the metrics.
    :rtype: tuple[xarray.Dataset, xarray.Dataset]
    :raises ValueError: If fewer than three cases are given.
    """
    from weatherbenchX import aggregation

    n = len(cases)
    if n < 3:
        raise ValueError("the bootstrap needs at least 3 cases")
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    n_blocks = int(np.ceil(n / block))
    ensemble = cases[0].ensemble[method]
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, n_blocks)])[:n]
        total = aggregation.AggregationState.sum([cases[i].states[method] for i in idx])
        draws.append(_values(total, ensemble))
    stacked = xr.concat(draws, dim="boot")
    lower, upper = (stacked.quantile(q, dim="boot", skipna=True).drop_vars("quantile") for q in quantiles)
    return lower, upper


def to_frame(results: Mapping[str, xr.Dataset], metric: str = "rmse", **selection) -> pd.DataFrame:
    """Lay one metric out with lead time as rows and methods as columns.

    :param results: Method names mapped to metrics, from :func:`combine_cases`.
    :type results: collections.abc.Mapping[str, xarray.Dataset]
    :param metric: Metric to extract.
    :type metric: str
    :param selection: Bin coordinates to select, for example ``region="alps"``.
    :type selection: typing.Any
    :returns: Frame indexed by lead time in hours.
    :rtype: pandas.DataFrame
    :raises ValueError: If the selection leaves more than one dimension.
    """
    columns = {}
    for name, ds in results.items():
        da = ds[metric]
        if selection:
            da = da.sel(selection)
        da = da.squeeze(drop=True)
        if da.ndim != 1:
            raise ValueError(f"{name}: {metric} still has dimensions {da.dims}; select more bins")
        columns[name] = pd.Series(da.values, index=(da["lead_time"].values / np.timedelta64(1, "h")).astype(int))
    return pd.DataFrame(columns).rename_axis("lead_h")
