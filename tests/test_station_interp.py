"""Tests for interpolation of gridded fields to station points."""

import numpy as np
import pytest
import xarray as xr
from conftest import signed, terrain

from downscaler.processing.station_interpolation import (
    CurvilinearInterpolator,
    GridInterpolator,
    analysis_baseline,
    downscale_to_stations,
    height_from_geopotential,
    lapse_correct,
    observations_at_valid_times,
)


def linear_field(lat, lon):
    """Field exactly reproducible by bilinear interpolation."""
    return 280 + 0.5 * np.asarray(lat) + 0.2 * signed(lon)


@pytest.fixture
def coarse_field(coarse_grid):
    lat, lon = coarse_grid
    return xr.DataArray(
        linear_field(lat[:, None], lon[None, :]),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )


@pytest.fixture
def points():
    return np.array([45.13, 48.77, 42.0, 50.1, 41.0, 52.9]), np.array([-3.3, 2.21, 8.9, -6.9, 10.9, 0.05])


def test_bilinear_is_exact_on_a_linear_field_across_the_meridian(coarse_field, points):
    lat, lon = points
    out = GridInterpolator.build(coarse_field.lat.values, coarse_field.lon.values, lat, lon).apply(coarse_field)
    assert out.dims == ("station",)
    assert np.allclose(out.values, linear_field(lat, lon), atol=1e-3)


def test_nearest_selects_the_closest_cell(coarse_field):
    out = GridInterpolator.build(
        coarse_field.lat.values, coarse_field.lon.values, [45.10, 45.20], [2.12, 2.20], method="nearest"
    ).apply(coarse_field)
    assert np.allclose(out.values, [linear_field(45.0, 2.0), linear_field(45.25, 2.25)], atol=1e-3)


def test_points_outside_the_grid_become_nan(coarse_field):
    out = GridInterpolator.build(
        coarse_field.lat.values, coarse_field.lon.values, [55.0, 45.0, 45.0], [2.0, 20.0, 2.0]
    ).apply(coarse_field)
    assert np.isnan(out.values[:2]).all() and np.isfinite(out.values[2])


def test_applying_to_a_different_grid_raises(coarse_field, points):
    interp = GridInterpolator.build(coarse_field.lat.values, coarse_field.lon.values, *points)
    with pytest.raises(ValueError, match="not the grid"):
        interp.apply(coarse_field.isel(lat=slice(0, 10)))


def test_unknown_method_raises(coarse_field, points):
    with pytest.raises(ValueError, match="bilinear"):
        GridInterpolator.build(coarse_field.lat.values, coarse_field.lon.values, *points, method="cubic")


def test_curvilinear_reproduces_a_linear_field_and_flags_distant_points(fine_grid):
    lat2d, lon2d = fine_grid
    field = xr.DataArray(linear_field(lat2d, lon2d)[None].repeat(3, 0), dims=("t", "y", "x"))
    lat = np.array([float(lat2d.mean()), float(lat2d.min()) + 2, 70.0])
    lon = np.array([float(lon2d.mean()), float(lon2d.min()) + 2, 40.0])
    out = CurvilinearInterpolator.build(lat2d, lon2d, lat, lon, k=4).apply(field)
    assert out.shape == (3, 3)
    assert np.isnan(out.values[:, 2]).all()
    assert np.abs(out.values[0, :2] - linear_field(lat[:2], lon[:2])).max() < 0.05


def test_lapse_correction_recovers_the_station_temperature(coarse_grid, points):
    lat, lon = coarse_grid
    orography = xr.DataArray(terrain(lat[:, None], lon[None, :]), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    model_temperature = 300 - 0.0065 * orography
    station_lat, station_lon = points
    elevation = terrain(station_lat, station_lon) + 250.0
    stations = xr.Dataset(
        {},
        coords={
            "station": [f"s{i}" for i in range(len(station_lat))],
            "latitude": ("station", station_lat),
            "longitude": ("station", station_lon),
            "elevation": ("station", elevation),
        },
    )
    forecast = xr.Dataset(
        {
            "t2m": model_temperature.expand_dims(
                time=[np.datetime64("2023-08-05")],
                lead_time=np.array([6], "timedelta64[h]").astype("timedelta64[ns]"),
            )
        }
    )
    result = downscale_to_stations(forecast, stations, "t2m", orography=orography)
    truth = 300 - 0.0065 * elevation
    assert np.allclose((result.bilinear - truth).values.ravel(), 0.0065 * 250, atol=2e-3)
    assert np.allclose((result.bilinear_lapse - truth).values.ravel(), 0, atol=2e-3)
    assert np.allclose(result.h_diff.values, 250, atol=1e-2)


def test_height_from_geopotential_divides_by_gravity():
    assert np.isclose(float(height_from_geopotential(xr.DataArray(9.80665 * 1000.0))), 1000.0)


def test_lapse_correct_cools_with_height():
    assert lapse_correct(300.0, 1000.0, 0.0) == pytest.approx(293.5)


def test_observations_are_matched_by_valid_time_and_station_id(coarse_grid, station_set):
    lat, lon = coarse_grid
    lead = (np.arange(1, 13) * 6).astype("timedelta64[h]").astype("timedelta64[ns]")
    values = np.zeros((1, 12, lat.size, lon.size), "f4")
    forecast = xr.Dataset(
        {"t2m": (("time", "lead_time", "lat", "lon"), values)},
        coords={"time": [np.datetime64("2023-08-05T00:00")], "lead_time": lead, "lat": lat, "lon": lon},
    )
    at = downscale_to_stations(forecast, station_set, "t2m")
    reversed_stations = station_set.isel(station=slice(None, None, -1))
    obs = observations_at_valid_times(at, reversed_stations, "t2m")
    assert obs.dims == ("init_time", "lead_time", "station")
    expected = station_set.t2m.sel(time=np.datetime64("2023-08-05T06:00")).values
    assert np.allclose(obs.values[0, 0], expected, equal_nan=True)


def test_analysis_baseline_fetches_each_valid_time_once(coarse_grid, station_set, analysis_source):
    lat, lon = coarse_grid
    lead = (np.arange(1, 13) * 6).astype("timedelta64[h]").astype("timedelta64[ns]")
    forecast = xr.Dataset(
        {"t2m": (("time", "lead_time", "lat", "lon"), np.zeros((2, 12, lat.size, lon.size), "f4"))},
        coords={
            "time": np.array(["2023-08-05T00:00", "2023-08-07T00:00"], "datetime64[ns]"),
            "lead_time": lead,
            "lat": lat,
            "lon": lon,
        },
    )
    baseline = analysis_baseline(analysis_source, forecast, station_set, "t2m")
    valid = forecast.time.values[:, None] + lead[None, :]
    assert baseline.bilinear.dims == ("init_time", "lead_time", "station")
    assert analysis_source.calls[0] == np.unique(valid).size < valid.size
