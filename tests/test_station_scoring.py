"""Tests for scoring forecasts against station observations."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from downscale import station_scoring as ss


def make_case(n_init=6, n_lead=8, n_members=10, n_stations=60, seed=0, bias=0.0, gaps=0.0, clustered=False):
    """Build a synthetic forecast and its truth.

    :returns: Forecast dataset with one method and the matching observations.
    :rtype: tuple[xarray.Dataset, xarray.DataArray]
    """
    rng = np.random.default_rng(seed)
    lat = rng.uniform(42, 51, n_stations)
    lon = rng.uniform(-4.5, 8, n_stations)
    elevation = rng.gamma(2, 300, n_stations).clip(1, 4000)
    if clustered:
        lat[:40] = 45.0 + rng.normal(0, 0.02, 40)
        lon[:40] = 2.0 + rng.normal(0, 0.02, 40)
    init = pd.date_range("2023-08-05", periods=n_init, freq="2D").to_numpy().astype("datetime64[ns]")
    lead = (np.arange(1, n_lead + 1) * 6).astype("timedelta64[h]").astype("timedelta64[ns]")
    coords = {
        "init_time": init,
        "lead_time": lead,
        "station": [f"s{i}" for i in range(n_stations)],
        "latitude": ("station", lat),
        "longitude": ("station", lon),
        "elevation": ("station", elevation),
    }
    truth = 300 + rng.normal(0, 3, (n_init, n_lead, n_stations))
    if gaps:
        truth[rng.random(truth.shape) < gaps] = np.nan
    obs = xr.DataArray(truth, dims=("init_time", "lead_time", "station"), coords=coords)
    error = rng.normal(0, 1.0, (n_init, n_lead, n_members, n_stations)) + bias
    if clustered:
        error[..., :40] += 2.0
    forecast = xr.DataArray(
        truth[:, :, None, :] + error,
        dims=("init_time", "lead_time", "ensemble", "station"),
        coords={**coords, "ensemble": np.arange(n_members)},
    )
    return xr.Dataset({"bilinear": forecast}), obs


def test_scores_match_an_explicit_computation_with_missing_observations():
    forecast, obs = make_case(gaps=0.1, bias=0.3)
    result = ss.combine_cases(
        [ss.score_station_case(forecast, obs, methods=["bilinear"], bin_by=None, density_alpha_deg=None)]
    )["bilinear"]
    values, truth = forecast.bilinear.values, obs.values
    mean = values.mean(2)
    valid = np.isfinite(truth)
    for lead in range(truth.shape[1]):
        chosen = valid[:, lead]
        difference = (mean[:, lead] - truth[:, lead])[chosen]
        assert float(result.rmse.isel(lead_time=lead)) == pytest.approx(np.sqrt((difference**2).mean()), rel=1e-5)
        assert float(result.bias.isel(lead_time=lead)) == pytest.approx(difference.mean(), rel=1e-4, abs=1e-6)
        members = values[:, lead].transpose(0, 2, 1)[chosen]
        observed = truth[:, lead][chosen]
        m = members.shape[1]
        skill = np.abs(members - observed[:, None]).mean(1)
        spread = np.abs(members[:, :, None] - members[:, None, :]).sum((1, 2)) / (2 * m * (m - 1))
        assert float(result.crps.isel(lead_time=lead)) == pytest.approx((skill - spread).mean(), rel=1e-4)


def test_density_weighting_reduces_the_influence_of_a_cluster():
    forecast, obs = make_case(clustered=True, n_stations=50)
    unweighted = ss.combine_cases(
        [ss.score_station_case(forecast, obs, methods=["bilinear"], bin_by=None, density_alpha_deg=None)]
    )["bilinear"]
    weighted = ss.combine_cases([ss.score_station_case(forecast, obs, methods=["bilinear"], bin_by=None)])["bilinear"]
    assert float(unweighted.bias.mean()) > 1.5
    assert float(weighted.bias.mean()) < float(unweighted.bias.mean()) - 0.5


def test_region_and_elevation_bins_match_an_explicit_subset():
    forecast, obs = make_case(n_stations=200, seed=3)
    result = ss.combine_cases([ss.score_station_case(forecast, obs, methods=["bilinear"], density_alpha_deg=None)])[
        "bilinear"
    ]
    assert set(result.rmse.dims) == {"lead_time", "region", "elevation"}
    lat, lon = obs.latitude.values, obs.longitude.values
    elevation = obs.elevation.values
    inside = (lat >= 44) & (lat <= 46.8) & (lon >= 5.8) & (lon <= 7.8) & (elevation >= 200) & (elevation < 500)
    assert inside.sum() >= 1
    difference = (forecast.bilinear.mean("ensemble").values - obs.values)[:, 2][:, inside]
    assert float(result.rmse.sel(region="alps", elevation=200).isel(lead_time=2)) == pytest.approx(
        np.sqrt((difference**2).mean()), rel=1e-5
    )


def test_combining_cases_sums_statistics_rather_than_averaging_scores():
    forecast, obs = make_case(n_init=6, n_stations=80, seed=5)
    forecast["bilinear"] = forecast.bilinear + xr.DataArray(
        [0, 0, 0, 0, 3.0, 3.0], dims="init_time", coords={"init_time": forecast.init_time}
    )
    slices = [slice(0, 1), slice(1, 4), slice(4, 6)]
    cases = [
        ss.score_station_case(
            forecast.isel(init_time=s),
            obs.isel(init_time=s),
            methods=["bilinear"],
            bin_by=None,
            density_alpha_deg=None,
        )
        for s in slices
    ]
    combined = ss.combine_cases(cases)["bilinear"].rmse
    whole = ss.combine_cases(
        [ss.score_station_case(forecast, obs, methods=["bilinear"], bin_by=None, density_alpha_deg=None)]
    )["bilinear"].rmse
    assert np.allclose(combined.values, whole.values, rtol=1e-6)
    averaged = np.mean([ss.combine_cases([c])["bilinear"].rmse.values for c in cases], axis=0)
    assert np.abs(averaged - whole.values).max() > 0.05


def test_bootstrap_is_reproducible_and_brackets_the_point_estimate():
    forecast, obs = make_case(n_init=10, n_stations=60, seed=7, bias=0.5)
    cases = [
        ss.score_station_case(
            forecast.isel(init_time=[i]),
            obs.isel(init_time=[i]),
            methods=["bilinear"],
            bin_by=None,
            density_alpha_deg=None,
        )
        for i in range(10)
    ]
    point = ss.combine_cases(cases)["bilinear"]
    lower, upper = ss.bootstrap_ci(cases, "bilinear", n_boot=100, block=2, seed=1)
    again, _ = ss.bootstrap_ci(cases, "bilinear", n_boot=100, block=2, seed=1)
    assert np.array_equal(lower.bias.values, again.bias.values)
    assert bool(((lower.bias <= point.bias + 1e-9) & (point.bias <= upper.bias + 1e-9)).all())
    assert float((upper.rmse - lower.rmse).min()) > 0


def test_bootstrap_needs_several_cases():
    forecast, obs = make_case(n_init=2, n_stations=20)
    cases = [
        ss.score_station_case(forecast.isel(init_time=[i]), obs.isel(init_time=[i]), methods=["bilinear"], bin_by=None)
        for i in range(2)
    ]
    with pytest.raises(ValueError, match="at least 3 cases"):
        ss.bootstrap_ci(cases, "bilinear")


def test_station_selection_splits_train_and_holdout():
    forecast, obs = make_case(n_stations=100)
    holdout = np.zeros(100, bool)
    holdout[::5] = True
    held, held_obs = ss.select_stations(forecast, obs, which="holdout", holdout=holdout)
    assert held.sizes["station"] == 20 and held_obs.sizes["station"] == 20
    assert ss.select_stations(obs, which="train", holdout=holdout).sizes["station"] == 80
    with pytest.raises(ValueError, match="holdout mask"):
        ss.select_stations(obs, which="holdout")


def test_ensembles_of_different_sizes_are_scored_side_by_side():
    forecast, obs = make_case(n_members=10, n_stations=40)
    small = forecast.bilinear.isel(ensemble=slice(0, 4)).rename({"ensemble": "ensemble_small"})
    combined = xr.Dataset({"big": forecast.bilinear, "small": small})
    result = ss.combine_cases(
        [ss.score_station_case(combined, obs, methods=["big", "small"], bin_by=None, density_alpha_deg=None)]
    )
    assert np.isfinite(result["big"].crps.values).all()
    assert np.isfinite(result["small"].crps.values).all()


def test_to_frame_lays_methods_out_by_lead_time():
    forecast, obs = make_case(n_stations=40)
    result = ss.combine_cases(
        [ss.score_station_case(forecast, obs, methods=["bilinear"], bin_by=None, density_alpha_deg=None)]
    )
    frame = ss.to_frame(result, "rmse")
    assert list(frame.columns) == ["bilinear"] and frame.index[0] == 6
