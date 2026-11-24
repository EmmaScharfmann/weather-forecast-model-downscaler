"""Apply a trained downscaler to a coarse ensemble forecast."""

from __future__ import annotations

import dataclasses
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from downscaler.constants import MAX_FRACTION_OUTSIDE
from downscaler.models.temperature import Downscaler
from downscaler.processing.station_interpolation import GridInterpolator


logger = logging.getLogger(__name__)


def to_fine_grid(forecast: xr.DataArray, lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray:
    """Interpolate a forecast onto the fine grid.

    A rotated fine grid can have corners slightly outside the forecast domain;
    a small fraction is allowed and takes edge values.

    :param forecast: Forecast whose last two dimensions are ``lat`` and ``lon``.
    :type forecast: xarray.DataArray
    :param lat2d: Fine grid latitudes with shape ``(y, x)``.
    :type lat2d: numpy.ndarray
    :param lon2d: Fine grid longitudes with shape ``(y, x)``.
    :type lon2d: numpy.ndarray
    :returns: Values over the forecast's leading dimensions followed by ``(y, x)``.
    :rtype: numpy.ndarray
    :raises ValueError: If too much of the fine grid lies outside the forecast domain.
    """
    interp = GridInterpolator.build(forecast["lat"].values, forecast["lon"].values, lat2d.ravel(), lon2d.ravel())
    fraction_outside = 1 - interp.inside.mean()
    if fraction_outside > MAX_FRACTION_OUTSIDE:
        raise ValueError(
            f"{fraction_outside:.1%} of the fine grid lies outside the forecast domain: "
            "the forecast region is too small"
        )
    if fraction_outside > 0:
        logger.warning(
            "%.2f%% of fine grid points lie outside the forecast domain and take edge values",
            100 * fraction_outside,
        )
        interp = dataclasses.replace(interp, inside=np.ones_like(interp.inside))
    out = interp.apply(forecast)
    return out.values.reshape(*out.shape[:-1], *lat2d.shape)


def downscale_forecast(model: Downscaler, forecast: xr.DataArray, samples: int = 4, batch: int = 8) -> xr.Dataset:
    """Downscale every member of an ensemble forecast.

    Each coarse member is interpolated onto the fine grid, then passed through the
    model ``samples`` times with fresh noise, so a forecast with ``n`` members becomes
    ``n * samples`` fine-scale maps.

    :param model: Trained downscaler.
    :type model: downscaler.models.temperature.Downscaler
    :param forecast: Forecast over ``(time, lead_time, ensemble, lat, lon)`` in kelvin;
        ``ensemble`` may be absent for a single forecast.
    :type forecast: xarray.DataArray
    :param samples: Fine-scale maps per coarse member; a deterministic model gives one.
    :type samples: int
    :param batch: Lead times processed at once.
    :type batch: int
    :returns: ``t2m_interp`` (the coarse forecast on the fine grid) over
        ``(time, lead_time, ensemble, y, x)`` and ``t2m_ds`` (the downscaled maps) over
        ``(time, lead_time, ensemble, sample, y, x)``, both in kelvin, with the fine grid's
        ``land_sea_mask`` and ``orography``.
    :rtype: xarray.Dataset
    """
    if "ensemble" not in forecast.dims:
        forecast = forecast.expand_dims(ensemble=[0])
    forecast = forecast.transpose("time", "lead_time", "ensemble", "lat", "lon").load()
    coarse = to_fine_grid(forecast, model.lat, model.lon).astype("f4")
    valid = forecast["time"].values[:, None] + forecast["lead_time"].values[None, :]
    n_init, n_lead, n_ensemble = coarse.shape[:3]
    n_samples = samples if model.net.cfg.stochastic else 1
    fine = np.empty((n_init, n_lead, n_ensemble, n_samples, *model.lat.shape), "f4")
    for i in range(n_init):
        for member in range(n_ensemble):
            fine[i, :, member] = model.predict(coarse[i, :, member], valid[i], members=samples, batch=batch)

    result = xr.Dataset(
        {
            "t2m_interp": (("time", "lead_time", "ensemble", "y", "x"), coarse, {"units": "K"}),
            "t2m_ds": (("time", "lead_time", "ensemble", "sample", "y", "x"), fine, {"units": "K"}),
            "land_sea_mask": (
                ("y", "x"),
                model.static["land_sea_mask"].astype("f4"),
                {"long_name": "land fraction of the fine grid cell"},
            ),
            "orography": (("y", "x"), model.static["cerra_orography"].astype("f4"), {"units": "m"}),
        },
        coords={
            "time": forecast["time"].values,
            "lead_time": forecast["lead_time"].values,
            "ensemble": forecast["ensemble"].values,
            "sample": np.arange(n_samples),
            "lat": (("y", "x"), model.lat),
            "lon": (("y", "x"), model.lon),
            "valid_time": (("time", "lead_time"), valid),
        },
    )
    result.attrs = {
        "model": str(model.net.cfg),
        "init_time": str(pd.Timestamp(forecast["time"].values[0])),
        "note": "t2m_ds holds, for each coarse member, several plausible fine-scale maps",
    }
    return result


def save_downscaled(downscaled: xr.Dataset, path) -> Path:
    """Write a downscaled forecast, replacing any earlier one only once the new one is complete.

    The store is written under a temporary name and renamed at the end, so an interrupted
    write never leaves a store that looks finished.

    :param downscaled: Output of :func:`downscale_forecast`.
    :type downscaled: xarray.Dataset
    :param path: Destination store.
    :type path: str | pathlib.Path
    :returns: The destination.
    :rtype: pathlib.Path
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    downscaled.to_zarr(tmp, mode="w")
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)
    return path
