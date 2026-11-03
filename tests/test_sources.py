"""Reading ERA5 and AIFS-ENS from archives laid out like the Earthmover ones."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from downscale.sources import ERA5, available_inits, fetch_forecast, open_forecasts, save_forecast


def earthmover_archive(inits, ingested=None, members=5, leads=7):
    """Synthetic forecast archive with the Earthmover AIFS-ENS layout.

    :param inits: Initialisation times.
    :type inits: pandas.DatetimeIndex
    :param ingested: Ingested length per initialisation; ``None`` leaves the record empty.
    :type ingested: list | None
    :param members: Ensemble size.
    :type members: int
    :param leads: Number of six-hourly lead times, starting at zero.
    :type leads: int
    :returns: The archive, temperatures in degrees Celsius.
    :rtype: xarray.Dataset
    """
    lat = 90 - 0.25 * np.arange(721)
    lon = -180 + 0.25 * np.arange(1440)
    lat, lon = lat[(lat > 35) & (lat < 58)], lon[(lon > -15) & (lon < 20)]
    lead = pd.to_timedelta(6 * np.arange(leads), "h")
    shape = (len(inits), leads, members, lat.size, lon.size)
    values = (
        15 + 0.1 * np.arange(members)[None, None, :, None, None]
        + 0.01 * lon[None, None, None, None, :] + np.zeros(shape)
    ).astype("f4")
    ingested = (
        np.full(len(inits), np.timedelta64("NaT"), "timedelta64[ns]") if ingested is None
        else np.asarray(ingested, "timedelta64[ns]")
    )
    ds = xr.Dataset(
        {"temperature_2m": (("init_time", "lead_time", "ensemble_member", "latitude", "longitude"), values,
                            {"units": "degree_Celsius"})},
        coords={
            "init_time": inits, "lead_time": lead, "ensemble_member": np.arange(members, dtype="i2"),
            "latitude": lat, "longitude": lon,
            "ingested_forecast_length": ("init_time", ingested),
            "expected_forecast_length": ("init_time", np.full(len(inits), lead[-1].to_numpy())),
            "valid_time": (("init_time", "lead_time"), inits.values[:, None] + lead.values[None, :]),
        },
    )
    return ds


@pytest.fixture
def archive():
    inits = pd.date_range("2025-08-01", periods=4, freq="6h")
    return earthmover_archive(inits)


def test_forecasts_are_cropped_converted_to_kelvin_and_standardised(archive):
    ds = archive.rename({"init_time": "time", "ensemble_member": "ensemble", "latitude": "lat", "longitude": "lon"})
    da = fetch_forecast(ds, "2025-08-01 06:00", (41.0, 51.5, -5.5, 10.0), members=3, max_lead_hours=24)
    assert da.dims == ("time", "lead_time", "ensemble", "lat", "lon")
    assert da.sizes["ensemble"] == 3 and da.sizes["lead_time"] == 5 and da.sizes["time"] == 1
    assert da.lat.min() >= 41 and da.lat.max() <= 51.5 and da.lon.min() >= -5.5 and da.lon.max() <= 10
    assert da.attrs["units"] == "K" and da.name == "t2m"
    assert np.all(np.diff(da.lon.values) > 0)
    expected = 15 + 273.15 + 0.1 * np.arange(3)[:, None] + 0.01 * da.lon.values[None, :]
    np.testing.assert_allclose(da.isel(time=0, lead_time=0, lat=0).values, expected, rtol=1e-6)


def test_an_unknown_initialisation_is_refused(archive):
    ds = archive.rename({"init_time": "time", "ensemble_member": "ensemble", "latitude": "lat", "longitude": "lon"})
    with pytest.raises(KeyError, match="not in the forecast archive"):
        fetch_forecast(ds, "2024-01-01", (41.0, 51.5, -5.5, 10.0))


def test_incomplete_initialisations_are_left_out():
    inits = pd.date_range("2025-08-01", periods=3, freq="6h")
    full, partial = np.timedelta64(36, "h"), np.timedelta64(12, "h")
    ds = earthmover_archive(inits, ingested=[full, partial, np.timedelta64("NaT")])
    ds = ds.rename({"init_time": "time"})
    recent = available_inits(ds, now=pd.Timestamp("2025-08-01 18:00"))
    assert list(recent) == [inits[0]]
    later = available_inits(ds, now=pd.Timestamp("2025-08-03"))
    assert list(later) == [inits[0], inits[2]]


def test_a_local_store_with_the_same_layout_opens_like_the_archive(archive, tmp_path):
    archive.to_zarr(tmp_path / "aifs.zarr")
    ds = open_forecasts(str(tmp_path / "aifs.zarr"))
    assert {"time", "lead_time", "ensemble", "lat", "lon"} <= set(ds.dims)
    da = fetch_forecast(ds, "2025-08-01", (41.0, 51.5, -5.5, 10.0))
    path = save_forecast(da, tmp_path / "forecasts")
    assert path.name == "aifs-ens_20250801T0000.zarr" and (tmp_path / "forecasts" / (path.name + ".done")).exists()
    xr.testing.assert_allclose(xr.open_zarr(path)["t2m"].load(), da)


def test_era5_is_put_on_descending_latitudes_and_positive_longitudes():
    lat, lon = np.arange(40.0, 45.0, 0.25), np.arange(-5.0, 5.0, 0.25)
    times = pd.date_range("2020-08-01", periods=3, freq="6h")
    ds = xr.Dataset(
        {"2m_temperature": (("time", "latitude", "longitude"),
                            290 + np.zeros((3, lat.size, lon.size)) + lon[None, None, :])},
        coords={"time": times, "latitude": lat, "longitude": lon,
                "z_sfc": (("latitude", "longitude"), np.ones((lat.size, lon.size)))},
    )
    era5 = ERA5(dataset=ds)
    out = era5(list(times[:2]), ["t2m"])
    assert out.dims == ("time", "variable", "lat", "lon") and out.sizes["time"] == 2
    assert out.lat.values[0] > out.lat.values[-1]
    assert out.lon.min() >= 0 and np.all(np.diff(out.lon.values) > 0)
    at = out.sel(lon=355.0, variable="t2m").isel(time=0, lat=0)
    assert float(at) == pytest.approx(285.0)
    assert era5.surface_geopotential().dims == ("lat", "lon")


def test_era5_crops_to_a_box_across_the_meridian_before_reading():
    lat, lon = np.arange(50.0, 39.75, -0.25), np.arange(0.0, 360.0, 0.25)
    times = pd.date_range("2020-08-01", periods=2, freq="6h")
    ds = xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), 290 + np.zeros((2, lat.size, lon.size)) + lon[None, None, :])},
        coords={"valid_time": times, "latitude": lat, "longitude": lon},
    )
    era5 = ERA5(dataset=ds, box=(42.0, 44.0, -2.0, 2.0))
    out = era5(list(times), ["t2m"])
    assert out.sizes["lat"] == 9 and out.sizes["lon"] == 17
    assert set(np.round(out.lon.values, 2)) == set(np.round(np.r_[np.arange(358.0, 360.0, 0.25), np.arange(0.0, 2.25, 0.25)], 2))
    assert float(out.sel(lon=359.0, variable="t2m").isel(time=0, lat=0)) == pytest.approx(290 + 359.0)


def test_an_end_date_includes_its_day_but_an_end_time_does_not():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from fetch_forecasts import select_inits

    inits = pd.date_range("2025-08-01", periods=8, freq="6h")
    assert len(select_inits(inits, "2025-08-01", "2025-08-01", [0, 12])) == 2
    assert len(select_inits(inits, "2025-08-01", "2025-08-01 00:00", [0, 12])) == 1
    assert len(select_inits(inits, "2025-08-01", "2025-08-02", None)) == 8


def test_a_folder_of_stores_written_per_initialisation_opens_as_one_archive(archive, tmp_path):
    folder = tmp_path / "aifs-ens-2020"
    for i in range(archive.sizes["init_time"]):
        one = archive.isel(init_time=[i]).rename(
            {"init_time": "time", "ensemble_member": "ensemble", "latitude": "lat", "longitude": "lon",
             "temperature_2m": "t2m"}
        ).drop_vars(["ingested_forecast_length", "expected_forecast_length"])
        one["t2m"] = one["t2m"] + 273.15
        one["t2m"].attrs = {"units": "K"}
        one.to_zarr(folder / f"aifs-ens_{i}.zarr")
    ds = open_forecasts(str(folder))
    assert ds.sizes["time"] == archive.sizes["init_time"]
    assert list(available_inits(ds)) == list(pd.DatetimeIndex(archive.init_time.values))
    da = fetch_forecast(ds, archive.init_time.values[2], (41.0, 51.5, -5.5, 10.0))
    reference = fetch_forecast(
        archive.rename({"init_time": "time", "ensemble_member": "ensemble", "latitude": "lat", "longitude": "lon"}),
        archive.init_time.values[2], (41.0, 51.5, -5.5, 10.0),
    )
    np.testing.assert_allclose(da.values, reference.values, rtol=1e-6)


def test_a_downscaled_store_is_replaced_only_when_complete(tmp_path):
    from downscale.inference import save_downscaled

    path = tmp_path / "run.zarr"
    first = xr.Dataset({"t2m_ds": ("x", np.zeros(3))})
    save_downscaled(first, path)
    save_downscaled(first + 1, path)
    assert float(xr.open_zarr(path).t2m_ds.mean()) == 1
    assert not (tmp_path / "run.zarr.tmp").exists()


def test_the_public_era5_copy_gets_a_filled_surface_geopotential(monkeypatch):
    from downscale import sources

    lat, lon = np.arange(90.0, -90.25, -45.0), np.arange(0.0, 360.0, 90.0)
    times = pd.to_datetime(["1940-01-01", "2000-01-01", "2020-08-01"])
    surface = np.full((3, lat.size, lon.size), np.nan)
    surface[1:] = 9.81 * 100
    ds = xr.Dataset(
        {
            "2m_temperature": (("time", "latitude", "longitude"), np.full((3, lat.size, lon.size), 290.0)),
            "geopotential_at_surface": (("time", "latitude", "longitude"), surface),
            "u_component_of_wind": (("time", "latitude", "longitude"), np.zeros((3, lat.size, lon.size))),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    monkeypatch.setattr(sources.xr, "open_zarr", lambda *args, **kwargs: ds)
    era5 = ERA5(dataset=sources.open_arco_era5())
    assert list(era5.ds.data_vars) == ["2m_temperature"]
    assert float(era5.surface_geopotential().min()) == pytest.approx(981.0)
    assert float(era5(["2020-08-01"], ["t2m"]).mean()) == 290.0
