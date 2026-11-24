"""Score an ensemble forecast, raw or downscaled, against a reference, per lead time.

The reference is either a gridded reanalysis on the fine grid or station observations;
both are handled the same way once the forecast has been brought to the same points.
Scores are kept as sums so that initialisations, points and lead times can be
combined exactly before taking square roots.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr


def fair_crps(ensemble: np.ndarray, truth: np.ndarray, axis: int = 0) -> np.ndarray:
    """Fair continuous ranked probability score of each forecast.

    Uses the sorted-member identity for the mean absolute difference between members,
    so the cost grows with ``m log m`` rather than ``m^2``.

    :param ensemble: Members along ``axis``.
    :type ensemble: numpy.ndarray
    :param truth: Reference with the ensemble's shape minus ``axis``.
    :type truth: numpy.ndarray
    :param axis: Member axis.
    :type axis: int
    :returns: Score with the member axis removed, NaN where the reference is missing.
    :rtype: numpy.ndarray
    :raises ValueError: If fewer than two members are given.
    """
    ens = np.moveaxis(np.asarray(ensemble, "f8"), axis, 0)
    m = ens.shape[0]
    if m < 2:
        raise ValueError("fair CRPS needs at least 2 members")
    skill = np.abs(ens - np.asarray(truth, "f8")[None]).mean(0)
    ranks = (2 * np.arange(1, m + 1) - m - 1).reshape(-1, *[1] * (ens.ndim - 1))
    pairwise = 2 * (ranks * np.sort(ens, axis=0)).sum(0)
    return skill - pairwise / (2 * m * (m - 1))


def lead_scores(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    member_dim: str = "member",
    lead_dim: str = "lead_time",
) -> pd.DataFrame:
    """Ensemble scores per lead time, pooled over every other dimension.

    :param forecast: Ensemble with ``member_dim``, ``lead_dim`` and any other dimensions
        (initialisation, points), in kelvin. Without ``member_dim`` it is a single field, such as
        a reanalysis: its CRPS is then its absolute error and its spread zero.
    :type forecast: xarray.DataArray
    :param truth: Reference with the forecast's dimensions minus ``member_dim``; NaN marks
        points left out (sea, missing observations).
    :type truth: xarray.DataArray
    :param member_dim: Name of the member dimension.
    :type member_dim: str
    :param lead_dim: Name of the lead time dimension.
    :type lead_dim: str
    :returns: One row per lead time with ``lead_hours``, ``n``, ``rmse_K`` and ``bias_K`` of the
        ensemble mean, ``crps_K``, ``spread_K`` and ``spread_skill`` (the ratio of spread
        to RMSE, near one for a reliable ensemble).
    :rtype: pandas.DataFrame
    """
    single = member_dim not in forecast.dims
    if single:
        forecast = forecast.expand_dims({member_dim: 1})
    truth = truth.broadcast_like(forecast.isel({member_dim: 0}, drop=True))
    others = [d for d in truth.dims if d != lead_dim]
    ens = forecast.transpose(member_dim, lead_dim, *others).values.reshape(
        forecast.sizes[member_dim], forecast.sizes[lead_dim], -1
    )
    ref = truth.transpose(lead_dim, *others).values.reshape(truth.sizes[lead_dim], -1)
    valid = np.isfinite(ref) & np.isfinite(ens).all(0)
    error = np.where(valid, ens.mean(0) - ref, 0.0)
    if single:
        crps, variance = np.abs(error), np.zeros_like(error)
    else:
        crps = np.where(valid, fair_crps(np.where(valid, ens, 0.0), np.where(valid, ref, 0.0)), 0.0)
        variance = np.where(valid, ens.var(0, ddof=1), 0.0)
    n = valid.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = np.sqrt((error**2).sum(1) / n)
        spread = np.sqrt(variance.sum(1) / n)
        table = pd.DataFrame(
            {
                "lead_hours": (truth[lead_dim].values / np.timedelta64(1, "h")).astype(int),
                "n": n,
                "rmse_K": rmse,
                "bias_K": error.sum(1) / n,
                "crps_K": crps.sum(1) / n,
                "spread_K": spread,
                "spread_skill": spread / rmse,
            }
        )
    return table[table.n > 0].reset_index(drop=True)


def pooled(table: pd.DataFrame) -> pd.Series:
    """Combine the rows of :func:`lead_scores` into one score over every lead time.

    :param table: Output of :func:`lead_scores`.
    :type table: pandas.DataFrame
    :returns: ``rmse_K``, ``bias_K``, ``crps_K`` and ``spread_K`` over all leads, weighted by ``n``.
    :rtype: pandas.Series
    """
    n = table.n.sum()
    return pd.Series(
        {
            "rmse_K": float(np.sqrt((table.n * table.rmse_K**2).sum() / n)),
            "bias_K": float((table.n * table.bias_K).sum() / n),
            "crps_K": float((table.n * table.crps_K).sum() / n),
            "spread_K": float(np.sqrt((table.n * table.spread_K**2).sum() / n)),
        }
    )


def combine_leads(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Pool the outputs of :func:`lead_scores` for several initialisations, lead by lead.

    :param tables: Outputs of :func:`lead_scores`.
    :type tables: collections.abc.Sequence[pandas.DataFrame]
    :returns: The same columns, weighted by the number of points behind each row.
    :rtype: pandas.DataFrame
    """
    rows = pd.concat(tables)
    rows = rows.assign(
        se=rows.n * rows.rmse_K**2, e=rows.n * rows.bias_K, c=rows.n * rows.crps_K, v=rows.n * rows.spread_K**2
    )
    sums = rows.groupby("lead_hours")[["n", "se", "e", "c", "v"]].sum().reset_index()
    out = pd.DataFrame(
        {
            "lead_hours": sums.lead_hours,
            "n": sums.n,
            "rmse_K": np.sqrt(sums.se / sums.n),
            "bias_K": sums.e / sums.n,
            "crps_K": sums.c / sums.n,
            "spread_K": np.sqrt(sums.v / sums.n),
        }
    )
    out["spread_skill"] = out.spread_K / out.rmse_K
    return out


def stack_members(downscaled: xr.Dataset) -> xr.DataArray:
    """Flatten the downscaled ensemble's coarse members and samples into one member dimension.

    :param downscaled: Output of :func:`downscale.inference.downscale_forecast`.
    :type downscaled: xarray.Dataset
    :returns: ``t2m_ds`` over ``(time, lead_time, member, y, x)``.
    :rtype: xarray.DataArray
    """
    da = downscaled["t2m_ds"].stack(member=("ensemble", "sample")).transpose("time", "lead_time", "member", "y", "x")
    return da.drop_vars(["member", "ensemble", "sample"]).assign_coords(member=np.arange(da.sizes["member"]))


def at_valid_times(reference: xr.DataArray, valid_time: xr.DataArray) -> xr.DataArray:
    """Pick a time series at each forecast's valid times, NaN where it has none.

    :param reference: Values over ``time`` and any other dimensions.
    :type reference: xarray.DataArray
    :param valid_time: Valid times over ``(time, lead_time)``, ``time`` being the initialisation.
    :type valid_time: xarray.DataArray
    :returns: The reference over ``(time, lead_time, ...)``.
    :rtype: xarray.DataArray
    """
    flat = pd.DatetimeIndex(valid_time.values.ravel())
    picked = reference.reindex(time=flat.values.astype(reference.time.dtype))
    others = [d for d in picked.dims if d != "time"]
    values = picked.transpose("time", *others).values.reshape(*valid_time.shape, *[picked.sizes[d] for d in others])
    return xr.DataArray(
        values,
        dims=["time", "lead_time", *others],
        coords={"time": valid_time.time.values, "lead_time": valid_time.lead_time.values},
    )


def station_rmse(field: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """RMSE of a deterministic field at each station, over every case where both are known.

    :param field: Values at stations, the stations on the last axis.
    :type field: numpy.ndarray
    :param observed: Observations of the same shape.
    :type observed: numpy.ndarray
    :returns: One value per station, NaN for a station with no case.
    :rtype: numpy.ndarray
    """
    difference = (np.asarray(field) - np.asarray(observed)).reshape(-1, np.shape(observed)[-1])
    ok = np.isfinite(difference)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.where(ok, difference**2, 0.0).sum(0) / ok.sum(0))


def elevation_scores(
    fields: dict[str, np.ndarray],
    observed: np.ndarray,
    elevation: np.ndarray,
    edges: Sequence[float],
) -> pd.DataFrame:
    """RMSE of several deterministic fields against stations, per elevation band.

    :param fields: Names mapped to values at stations, the stations on the last axis.
    :type fields: dict[str, numpy.ndarray]
    :param observed: Observations of the same shape.
    :type observed: numpy.ndarray
    :param elevation: Station elevations [m].
    :type elevation: numpy.ndarray
    :param edges: Band edges [m].
    :type edges: collections.abc.Sequence[float]
    :returns: One row per field and band with ``n`` and ``rmse_K``, plus an ``all`` band.
    :rtype: pandas.DataFrame
    """
    rows = []
    bands = [("all", np.ones_like(elevation, bool))] + [
        (f"{int(lo)}-{int(hi)} m", (elevation >= lo) & (elevation < hi)) for lo, hi in zip(edges[:-1], edges[1:])
    ]
    for name, values in fields.items():
        for band, chosen in bands:
            difference = (values - observed)[..., chosen]
            difference = difference[np.isfinite(difference)]
            if difference.size:
                rows.append(
                    {
                        "field": name,
                        "band": band,
                        "n": int(difference.size),
                        "rmse_K": float(np.sqrt((difference**2).mean())),
                    }
                )
    return pd.DataFrame(rows)
