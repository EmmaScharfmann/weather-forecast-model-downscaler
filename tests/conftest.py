"""Shared fixtures: synthetic grids, datasets and providers used across the tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def signed(lon):
    """Convert longitudes to -180..180.

    :param lon: Longitudes in any convention.
    :type lon: numpy.ndarray
    :returns: Signed longitudes.
    :rtype: numpy.ndarray
    """
    return ((np.asarray(lon) + 180) % 360) - 180


def terrain(lat, lon):
    """Analytic orography used to give the synthetic data a learnable signal.

    :param lat: Latitudes.
    :type lat: numpy.ndarray
    :param lon: Longitudes.
    :type lon: numpy.ndarray
    :returns: Heights in metres.
    :rtype: numpy.ndarray
    """
    return 100 + 40 * (np.asarray(lat) - 40) + 30 * signed(lon)


def coarse_terrain(lat, lon):
    """Smoothed orography standing in for the coarse model's surface.

    :param lat: Latitudes.
    :type lat: numpy.ndarray
    :param lon: Longitudes.
    :type lon: numpy.ndarray
    :returns: Heights in metres, lower than :func:`terrain` over high ground.
    :rtype: numpy.ndarray
    """
    return 0.6 * terrain(lat, lon)


def diurnal(hour):
    """Analytic daily temperature cycle.

    :param hour: Hour of day.
    :type hour: numpy.ndarray
    :returns: Temperature anomaly in kelvin.
    :rtype: numpy.ndarray
    """
    return 5 * np.sin((np.asarray(hour) - 9) / 24 * 2 * np.pi)


@pytest.fixture(scope="session")
def coarse_grid():
    """Regular grid matching a cropped global forecast, with non-monotonic longitudes.

    :returns: Latitudes and longitudes.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    lat = 53.0 - 0.25 * np.arange(53)
    lon = np.concatenate([353.0 + 0.25 * np.arange(28), 0.25 * np.arange(45)])
    return lat, lon


@pytest.fixture(scope="session")
def global_grid():
    """Global quarter-degree grid, as the real coarse archive uses.

    :returns: Latitudes and longitudes.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    return 90 - 0.25 * np.arange(721), 0.25 * np.arange(1440)


@pytest.fixture(scope="session")
def fine_grid():
    """Curvilinear grid standing in for a regional reanalysis.

    :returns: Two-dimensional latitudes and longitudes.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    ny, nx = 72, 76
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    lat = 41.0 + 0.15 * yy + 0.01 * xx
    lon = -5.5 + 0.2 * xx - 0.015 * yy
    return lat.astype("f4"), lon.astype("f4")


@pytest.fixture(scope="session")
def static_fields(fine_grid):
    """Static fields on the fine grid, with a coarse surface that is smoother.

    :param fine_grid: Fine grid coordinates.
    :type fine_grid: tuple[numpy.ndarray, numpy.ndarray]
    :returns: Fields keyed as the model expects them.
    :rtype: dict
    """
    lat, lon = fine_grid
    fine_orography = terrain(lat, lon)
    coarse_orography = coarse_terrain(lat, lon)
    return {
        "cerra_orography": fine_orography.astype("f4"),
        "era5_orography": coarse_orography.astype("f4"),
        "land_sea_mask": np.ones_like(lat, dtype="f4"),
        "lat": lat,
        "lon": lon,
    }


@pytest.fixture(scope="session")
def stats(static_fields):
    """Normalisation statistics matching the synthetic fields.

    :param static_fields: Static fields on the fine grid.
    :type static_fields: dict
    :returns: Statistics for every normalised channel.
    :rtype: dict
    """
    height_diff = static_fields["cerra_orography"] - static_fields["era5_orography"]
    return {
        "era5_t2m": {"mean": 290.0, "std": 8.0},
        "residual": {"mean": 0.0, "std": 1.5},
        "cerra_orography": {
            "mean": float(static_fields["cerra_orography"].mean()),
            "std": float(static_fields["cerra_orography"].std()),
        },
        "orography_diff": {"mean": float(height_diff.mean()), "std": float(height_diff.std())},
        "lat": {"mean": float(static_fields["lat"].mean()), "std": float(static_fields["lat"].std())},
        "lon": {"mean": float(static_fields["lon"].mean()), "std": float(static_fields["lon"].std())},
    }


@pytest.fixture
def station_set(fine_grid):
    """Synthetic stations spread over the fine grid.

    :param fine_grid: Fine grid coordinates.
    :type fine_grid: tuple[numpy.ndarray, numpy.ndarray]
    :returns: Dataset with coordinates, elevations and hourly observations.
    :rtype: xarray.Dataset
    """
    lat2d, lon2d = fine_grid
    rng = np.random.default_rng(1)
    n = 40
    lat = rng.uniform(float(lat2d.min()) + 1, float(lat2d.max()) - 1, n)
    lon = rng.uniform(float(lon2d.min()) + 1, float(lon2d.max()) - 1, n)
    elevation = terrain(lat, lon) + rng.uniform(-100, 400, n)
    hours = pd.date_range("2023-08-05", "2023-08-20", freq="1h").to_numpy().astype("datetime64[ns]")
    hour_of_day = pd.DatetimeIndex(hours).hour.values
    values = (
        300 - 0.0065 * elevation[None, :] + diurnal(hour_of_day)[:, None] + rng.normal(0, 1.2, (len(hours), n))
    ).astype("f4")
    ds = xr.Dataset(
        {"t2m": (("time", "station"), values)},
        coords={
            "time": hours,
            "station": [f"s{i}" for i in range(n)],
            "latitude": ("station", lat),
            "longitude": ("station", lon),
            "elevation": ("station", elevation),
        },
    )
    ds["holdout"] = ("station", np.arange(n) % 5 == 0)
    return ds


@pytest.fixture
def forecast_store(tmp_path, coarse_grid):
    """Write three synthetic forecast stores with completion markers.

    :param tmp_path: Temporary directory.
    :type tmp_path: pathlib.Path
    :param coarse_grid: Coarse grid coordinates.
    :type coarse_grid: tuple[numpy.ndarray, numpy.ndarray]
    :returns: Directory holding the stores.
    :rtype: pathlib.Path
    """
    lat, lon = coarse_grid
    directory = tmp_path / "forecasts"
    directory.mkdir()
    lead = (np.arange(1, 13) * 6).astype("timedelta64[h]").astype("timedelta64[ns]")
    rng = np.random.default_rng(0)
    for init in np.array(["2023-08-05", "2023-08-07", "2023-08-09"], "datetime64[ns]"):
        valid_hours = ((init + lead) - init.astype("datetime64[D]")) / np.timedelta64(1, "h")
        base = (300 - 0.0065 * terrain(lat[:, None], lon[None, :]))[None] + diurnal(valid_hours)[:, None, None]
        values = (base[None, :, None] + rng.normal(0, 0.5, (1, 12, 3, lat.size, lon.size))).astype("f4")
        path = directory / f"forecast_{str(init)[:10]}.zarr"
        xr.Dataset(
            {"t2m": (("time", "lead_time", "ensemble", "lat", "lon"), values)},
            coords={"time": [init], "lead_time": lead, "ensemble": np.arange(3), "lat": lat, "lon": lon},
        ).to_zarr(path)
        (directory / f"{path.name}.done").write_text("ok")
    return directory


@pytest.fixture
def analysis_source(global_grid):
    """Callable standing in for the coarse analysis archive.

    :param global_grid: Global grid coordinates.
    :type global_grid: tuple[numpy.ndarray, numpy.ndarray]
    :returns: Source with the Earth2Studio call signature and a record of its calls.
    :rtype: typing.Callable
    """
    lat, lon = global_grid
    field = coarse_terrain(lat[:, None], lon[None, :])

    class Source:
        def __init__(self):
            self.calls: list[int] = []

        def __call__(self, times, variables):
            index = pd.DatetimeIndex(times)
            self.calls.append(len(index))
            values = ((300 - 0.0065 * field)[None] + diurnal(index.hour.values)[:, None, None]).astype("f4")
            return xr.DataArray(
                values[:, None],
                dims=("time", "variable", "lat", "lon"),
                coords={
                    "time": index.to_numpy().astype("datetime64[ns]"),
                    "variable": list(variables),
                    "lat": lat,
                    "lon": lon,
                },
            )

    return Source()
