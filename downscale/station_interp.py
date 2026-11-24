"""Interpolation of gridded fields to station points.

Two interpolators are provided:
:class:`GridInterpolator` for regular
latitude/longitude grids such as a global forecast, and
:class:`CurvilinearInterpolator` for grids whose coordinates are two
dimensional, such as the Lambert grid of a regional reanalysis.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .constants import EARTH_RADIUS_KM, MAX_STATION_DISTANCE_KM, STD_LAPSE, G
from .regions import to_signed_lon


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridInterpolator:
    """Weights from a regular latitude/longitude grid to a set of points."""

    lat_order: np.ndarray
    lon_order: np.ndarray
    i0: np.ndarray
    j0: np.ndarray
    wy: np.ndarray
    wx: np.ndarray
    inside: np.ndarray
    method: str
    n_lat: int
    n_lon: int

    @classmethod
    def build(
        cls,
        grid_lat: np.ndarray,
        grid_lon: np.ndarray,
        point_lat: Sequence[float],
        point_lon: Sequence[float],
        method: str = "bilinear",
    ) -> GridInterpolator:
        """Compute interpolation weights for a set of points.

        :param grid_lat: Grid latitudes, ascending or descending.
        :type grid_lat: numpy.ndarray
        :param grid_lon: Grid longitudes, 0..360 or signed, possibly non-monotonic.
        :type grid_lon: numpy.ndarray
        :param point_lat: Target latitudes.
        :type point_lat: collections.abc.Sequence[float]
        :param point_lon: Target longitudes, signed.
        :type point_lon: collections.abc.Sequence[float]
        :param method: ``"bilinear"`` or ``"nearest"``.
        :type method: str
        :returns: Interpolator bound to this grid and these points.
        :rtype: GridInterpolator
        :raises ValueError: If the method is unknown or the grid is smaller than 2x2.
        """
        if method not in ("bilinear", "nearest"):
            raise ValueError("method must be 'bilinear' or 'nearest'")
        grid_lat = np.asarray(grid_lat, dtype="f8")
        signed_lon = to_signed_lon(grid_lon)
        lat_order = np.argsort(grid_lat, kind="stable")
        lon_order = np.argsort(signed_lon, kind="stable")
        sorted_lat, sorted_lon = grid_lat[lat_order], signed_lon[lon_order]
        if sorted_lat.size < 2 or sorted_lon.size < 2:
            raise ValueError("need at least a 2x2 grid to interpolate")

        y = np.asarray(point_lat, dtype="f8")
        x = to_signed_lon(point_lon)

        def locate(axis: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Find the bracketing cell and the weight of its upper edge.

            :param axis: Ascending coordinate values.
            :type axis: numpy.ndarray
            :param points: Target coordinate values.
            :type points: numpy.ndarray
            :returns: Lower indices and weights in 0..1.
            :rtype: tuple[numpy.ndarray, numpy.ndarray]
            """
            lower = np.clip(np.searchsorted(axis, points, side="right") - 1, 0, axis.size - 2)
            weight = np.clip((points - axis[lower]) / (axis[lower + 1] - axis[lower]), 0.0, 1.0)
            return lower, weight

        i0, wy = locate(sorted_lat, y)
        j0, wx = locate(sorted_lon, x)
        inside = (y >= sorted_lat[0]) & (y <= sorted_lat[-1]) & (x >= sorted_lon[0]) & (x <= sorted_lon[-1])
        return cls(lat_order, lon_order, i0, j0, wy, wx, inside, method, sorted_lat.size, sorted_lon.size)

    def apply(self, da: xr.DataArray, lat_dim: str = "lat", lon_dim: str = "lon") -> xr.DataArray:
        """Interpolate a field to the points.

        :param da: Field with the grid on its latitude and longitude dimensions.
        :type da: xarray.DataArray
        :param lat_dim: Name of the latitude dimension.
        :type lat_dim: str
        :param lon_dim: Name of the longitude dimension.
        :type lon_dim: str
        :returns: Field with the two grid dimensions replaced by a trailing ``station`` dimension.
        :rtype: xarray.DataArray
        :raises ValueError: If the field's grid differs from the one the interpolator was built for.
        """
        if da.sizes[lat_dim] != self.n_lat or da.sizes[lon_dim] != self.n_lon:
            raise ValueError(
                f"array grid ({da.sizes[lat_dim]}x{da.sizes[lon_dim]}) is not the grid this interpolator "
                f"was built for ({self.n_lat}x{self.n_lon})"
            )
        lead = [d for d in da.dims if d not in (lat_dim, lon_dim)]
        values = da.transpose(*lead, lat_dim, lon_dim).values.astype("f8")
        values = values[..., self.lat_order, :][..., :, self.lon_order]

        i0, j0 = self.i0, self.j0
        if self.method == "nearest":
            out = values[..., i0 + (self.wy >= 0.5), j0 + (self.wx >= 0.5)]
        else:
            wy, wx = self.wy, self.wx
            out = (
                (1 - wy) * (1 - wx) * values[..., i0, j0]
                + (1 - wy) * wx * values[..., i0, j0 + 1]
                + wy * (1 - wx) * values[..., i0 + 1, j0]
                + wy * wx * values[..., i0 + 1, j0 + 1]
            )
        out = np.where(self.inside, out, np.nan).astype("f4")
        coords = {d: da.coords[d] for d in lead if d in da.coords}
        return xr.DataArray(out, dims=[*lead, "station"], coords=coords, name=da.name)


@dataclass(frozen=True)
class CurvilinearInterpolator:
    """Weights from a grid with two-dimensional coordinates to a set of points."""

    flat_idx: np.ndarray
    weights: np.ndarray
    inside: np.ndarray
    shape: tuple[int, int]
    k: int

    @staticmethod
    def _unit_vectors(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Convert spherical coordinates to Cartesian unit vectors.

        :param lat: Latitudes in degrees.
        :type lat: numpy.ndarray
        :param lon: Longitudes in degrees.
        :type lon: numpy.ndarray
        :returns: Array of unit vectors with a trailing axis of length three.
        :rtype: numpy.ndarray
        """
        phi, lam = np.deg2rad(np.asarray(lat, "f8")), np.deg2rad(np.asarray(lon, "f8"))
        return np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], -1)

    @classmethod
    def build(
        cls,
        grid_lat2d: np.ndarray,
        grid_lon2d: np.ndarray,
        point_lat: Sequence[float],
        point_lon: Sequence[float],
        k: int = 4,
        max_km: float = MAX_STATION_DISTANCE_KM,
    ) -> CurvilinearInterpolator:
        """Compute inverse-distance weights from the nearest grid cells.

        :param grid_lat2d: Grid latitudes with shape ``(y, x)``.
        :type grid_lat2d: numpy.ndarray
        :param grid_lon2d: Grid longitudes with shape ``(y, x)``.
        :type grid_lon2d: numpy.ndarray
        :param point_lat: Target latitudes.
        :type point_lat: collections.abc.Sequence[float]
        :param point_lon: Target longitudes.
        :type point_lon: collections.abc.Sequence[float]
        :param k: Number of neighbouring cells; ``1`` gives nearest-neighbour.
        :type k: int
        :param max_km: Points further than this from any cell are flagged as outside.
        :type max_km: float
        :returns: Interpolator bound to this grid and these points.
        :rtype: CurvilinearInterpolator
        """
        from scipy.spatial import cKDTree

        shape = np.asarray(grid_lat2d).shape
        tree = cKDTree(cls._unit_vectors(np.ravel(grid_lat2d), np.ravel(grid_lon2d)))
        n_points = len(np.atleast_1d(point_lat))
        chord, idx = tree.query(cls._unit_vectors(point_lat, point_lon), k=k)
        chord = np.atleast_2d(chord).reshape(n_points, k)
        idx = np.atleast_2d(idx).reshape(n_points, k)
        km = EARTH_RADIUS_KM * 2 * np.arcsin(np.clip(chord / 2, 0, 1))
        inside = km[:, 0] <= max_km
        if k == 1:
            weights = np.ones_like(km)
        else:
            weights = 1.0 / np.maximum(km, 0.05) ** 2
            weights /= weights.sum(1, keepdims=True)
        return cls(idx, weights, inside, (int(shape[0]), int(shape[1])), k)

    def apply(self, da: xr.DataArray, y_dim: str = "y", x_dim: str = "x") -> xr.DataArray:
        """Interpolate a field to the points.

        :param da: Field with the grid on its ``y`` and ``x`` dimensions.
        :type da: xarray.DataArray
        :param y_dim: Name of the first grid dimension.
        :type y_dim: str
        :param x_dim: Name of the second grid dimension.
        :type x_dim: str
        :returns: Field with the two grid dimensions replaced by a trailing ``station`` dimension.
        :rtype: xarray.DataArray
        :raises ValueError: If the field's grid differs from the one the interpolator was built for.
        """
        if (da.sizes[y_dim], da.sizes[x_dim]) != self.shape:
            raise ValueError(
                f"grid {(da.sizes[y_dim], da.sizes[x_dim])} is not the grid this interpolator "
                f"was built for {self.shape}"
            )
        lead = [d for d in da.dims if d not in (y_dim, x_dim)]
        values = da.transpose(*lead, y_dim, x_dim).values.reshape(*[da.sizes[d] for d in lead], -1)
        out = (values[..., self.flat_idx] * self.weights).sum(-1)
        out = np.where(self.inside, out, np.nan).astype("f4")
        coords = {d: da.coords[d] for d in lead if d in da.coords}
        return xr.DataArray(out, dims=[*lead, "station"], coords=coords, name=da.name)


def interpolator_for(
    grid: xr.DataArray | xr.Dataset, stations: xr.Dataset, method: str = "bilinear"
) -> GridInterpolator:
    """Build a regular-grid interpolator for a station set.

    :param grid: Object carrying ``lat`` and ``lon`` coordinates.
    :type grid: xarray.DataArray | xarray.Dataset
    :param stations: Station dataset with ``latitude`` and ``longitude`` coordinates.
    :type stations: xarray.Dataset
    :param method: ``"bilinear"`` or ``"nearest"``.
    :type method: str
    :returns: Interpolator for those stations.
    :rtype: GridInterpolator
    """
    return GridInterpolator.build(
        grid["lat"].values, grid["lon"].values, stations["latitude"].values, stations["longitude"].values, method
    )


def curvilinear_for(lat2d: np.ndarray, lon2d: np.ndarray, stations: xr.Dataset, k: int = 4) -> CurvilinearInterpolator:
    """Build a curvilinear-grid interpolator for a station set.

    :param lat2d: Grid latitudes with shape ``(y, x)``.
    :type lat2d: numpy.ndarray
    :param lon2d: Grid longitudes with shape ``(y, x)``.
    :type lon2d: numpy.ndarray
    :param stations: Station dataset with ``latitude`` and ``longitude`` coordinates.
    :type stations: xarray.Dataset
    :param k: Number of neighbouring cells to average.
    :type k: int
    :returns: Interpolator for those stations.
    :rtype: CurvilinearInterpolator
    """
    return CurvilinearInterpolator.build(lat2d, lon2d, stations["latitude"].values, stations["longitude"].values, k=k)


def lapse_correct(
    temperature: xr.DataArray | np.ndarray,
    height_target: np.ndarray,
    height_model: np.ndarray,
    gamma: float = STD_LAPSE,
) -> xr.DataArray | np.ndarray:
    """Adjust a temperature from the model surface height to a target height.

    The correction uses a single lapse rate and so cannot represent nocturnal
    inversions, where valley air is colder than this relationship predicts.

    :param temperature: Temperature at the model surface [K].
    :type temperature: xarray.DataArray | numpy.ndarray
    :param height_target: Height of the target point [m].
    :type height_target: numpy.ndarray
    :param height_model: Height of the model surface at that point [m].
    :type height_model: numpy.ndarray
    :param gamma: Lapse rate [K m^-1].
    :type gamma: float
    :returns: Temperature at the target height [K].
    :rtype: xarray.DataArray | numpy.ndarray
    """
    return temperature - gamma * (height_target - height_model)


def height_from_geopotential(z: xr.DataArray) -> xr.DataArray:
    """Convert surface geopotential to height.

    :param z: Surface geopotential [m^2 s^-2].
    :type z: xarray.DataArray
    :returns: Surface height [m], named ``h_model``.
    :rtype: xarray.DataArray
    """
    return (z / G).rename("h_model")


def orography_on_grid(orography: xr.DataArray, like: xr.DataArray | xr.Dataset) -> xr.DataArray:
    """Select an orography field onto another object's grid.

    :param orography: Orography on a grid covering the target's.
    :type orography: xarray.DataArray
    :param like: Object whose ``lat`` and ``lon`` coordinates define the target grid.
    :type like: xarray.DataArray | xarray.Dataset
    :returns: Orography on the target grid.
    :rtype: xarray.DataArray
    """
    return orography.sel(lat=like["lat"].values, lon=like["lon"].values, method="nearest", tolerance=1e-3)


def downscale_to_stations(
    forecast: xr.Dataset,
    stations: xr.Dataset,
    variable: str = "t2m",
    orography: xr.DataArray | None = None,
    gamma: float = STD_LAPSE,
) -> xr.Dataset:
    """Read a gridded forecast at station locations with simple baselines.

    :param forecast: Forecast with dimensions ``(time, lead_time[, ensemble], lat, lon)``.
    :type forecast: xarray.Dataset
    :param stations: Station dataset with coordinates and elevations.
    :type stations: xarray.Dataset
    :param variable: Name of the variable to read.
    :type variable: str
    :param orography: Model surface height [m] on the forecast grid; enables the lapse baseline.
    :type orography: xarray.DataArray | None
    :param gamma: Lapse rate [K m^-1].
    :type gamma: float
    :returns: Dataset with ``nearest``, ``bilinear`` and, when orography is given,
        ``bilinear_lapse``, ``h_model`` and ``h_diff``, over ``(init_time, lead_time[, ensemble], station)``.
    :rtype: xarray.Dataset
    """
    fc = forecast[variable].rename({"time": "init_time"})
    bilinear = interpolator_for(fc, stations, "bilinear")
    nearest = interpolator_for(fc, stations, "nearest")
    if not bilinear.inside.all():
        logger.warning(
            "%d station(s) fall outside the forecast grid and become NaN: %s",
            int((~bilinear.inside).sum()),
            list(stations.station.values[~bilinear.inside][:5]),
        )

    out = xr.Dataset({"nearest": nearest.apply(fc), "bilinear": bilinear.apply(fc)})
    if orography is not None:
        h_model = bilinear.apply(orography.astype("f8"))
        out["bilinear_lapse"] = lapse_correct(out["bilinear"], stations["elevation"].values, h_model.values, gamma)
        out["h_model"] = ("station", h_model.values)
        out["h_diff"] = ("station", (stations["elevation"].values - h_model.values).astype("f4"))
    out = out.assign_coords(station=stations["station"].values)
    for coord in ("latitude", "longitude", "elevation"):
        out = out.assign_coords({coord: ("station", stations[coord].values)})
    out = out.assign_coords(
        valid_time=(
            ("init_time", "lead_time"),
            out["init_time"].values[:, None] + out["lead_time"].values[None, :],
        )
    )
    out.attrs.update({"variable": variable, "gamma_K_per_m": gamma, "interpolation": "bilinear / nearest, no wrap"})
    return out


def analysis_baseline(
    source,
    forecast_like: xr.Dataset,
    stations: xr.Dataset,
    variable: str = "t2m",
    orography: xr.DataArray | None = None,
    gamma: float = STD_LAPSE,
) -> xr.Dataset:
    """Read an analysis at station locations for every forecast valid time.

    The result carries no forecast error, so its distance from the observations
    measures the mismatch between a grid cell and a point.

    :param source: Callable taking ``(times, variables)`` and returning a ``(time, variable, lat, lon)`` field.
    :type source: typing.Callable
    :param forecast_like: Forecast defining the initialisation times, lead times and grid.
    :type forecast_like: xarray.Dataset
    :param stations: Station dataset with coordinates and elevations.
    :type stations: xarray.Dataset
    :param variable: Name of the variable to read.
    :type variable: str
    :param orography: Model surface height [m] on the forecast grid; enables the lapse baseline.
    :type orography: xarray.DataArray | None
    :param gamma: Lapse rate [K m^-1].
    :type gamma: float
    :returns: Dataset with the same variable names over ``(init_time, lead_time, station)``.
    :rtype: xarray.Dataset
    """
    init = forecast_like["time"].values
    lead = forecast_like["lead_time"].values
    valid = init[:, None] + lead[None, :]
    unique = pd.DatetimeIndex(np.unique(valid.ravel()))
    logger.info("fetching %d unique analysis time(s) for the baseline", unique.size)

    field = source(list(unique), [variable]).sel(variable=variable, drop=True)
    field = field.sel(
        lat=forecast_like["lat"].values, lon=forecast_like["lon"].values, method="nearest", tolerance=1e-3
    )
    bilinear = interpolator_for(field, stations, "bilinear")
    nearest = interpolator_for(field, stations, "nearest")
    columns = {"nearest": nearest.apply(field), "bilinear": bilinear.apply(field)}
    if orography is not None:
        h_model = bilinear.apply(orography.astype("f8")).values
        columns["bilinear_lapse"] = lapse_correct(columns["bilinear"], stations["elevation"].values, h_model, gamma)

    position = np.searchsorted(unique.to_numpy().astype("datetime64[ns]"), valid.ravel().astype("datetime64[ns]"))
    out = {
        name: (("init_time", "lead_time", "station"), arr.values[position].reshape(init.size, lead.size, -1))
        for name, arr in columns.items()
    }
    return xr.Dataset(
        out,
        coords={
            "init_time": init,
            "lead_time": lead,
            "station": stations["station"].values,
            "valid_time": (("init_time", "lead_time"), valid),
        },
    )


def observations_at_valid_times(
    forecast_at_stations: xr.Dataset, stations: xr.Dataset, variable: str = "t2m"
) -> xr.DataArray:
    """Lay observations out to match a forecast read at stations.

    :param forecast_at_stations: Output of :func:`downscale_to_stations`.
    :type forecast_at_stations: xarray.Dataset
    :param stations: Station dataset holding the observations.
    :type stations: xarray.Dataset
    :param variable: Name of the observed variable.
    :type variable: str
    :returns: Observations over ``(init_time, lead_time, station)``, NaN where none exist.
    :rtype: xarray.DataArray
    """
    obs = stations[variable].sel(station=forecast_at_stations["station"].values)
    valid = forecast_at_stations["valid_time"].values
    flat = obs.reindex(time=valid.ravel().astype("datetime64[ns]")).values
    values = flat.reshape(*valid.shape, obs.sizes["station"])
    return xr.DataArray(
        values,
        dims=("init_time", "lead_time", "station"),
        coords={
            "init_time": forecast_at_stations["init_time"],
            "lead_time": forecast_at_stations["lead_time"],
            "station": forecast_at_stations["station"],
            "valid_time": forecast_at_stations["valid_time"],
        },
        name=f"obs_{variable}",
    )
