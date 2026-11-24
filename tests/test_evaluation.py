"""Ensemble scores per lead time."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from downscaler.evaluation.ensemble_scores import at_valid_times, elevation_scores, fair_crps, lead_scores, pooled


def brute_force_crps(ens, obs):
    m = ens.shape[0]
    skill = np.abs(ens - obs).mean(0)
    spread = np.abs(ens[:, None] - ens[None]).sum((0, 1)) / (2 * m * (m - 1))
    return skill - spread


def test_fair_crps_matches_the_pairwise_definition():
    rng = np.random.default_rng(0)
    ens, obs = rng.normal(size=(7, 5, 3)), rng.normal(size=(5, 3))
    np.testing.assert_allclose(fair_crps(ens, obs), brute_force_crps(ens, obs))
    np.testing.assert_allclose(fair_crps(np.moveaxis(ens, 0, 2), obs, axis=2), brute_force_crps(ens, obs))


def test_lead_scores_skip_missing_truth_and_recover_known_errors():
    lead = pd.to_timedelta([0, 6, 12], "h")
    members = np.arange(4)
    truth = xr.DataArray(np.zeros((2, 3, 10)), dims=("time", "lead_time", "station"), coords={"lead_time": lead})
    offset = np.array([1.0, 2.0, 3.0])[None, :, None, None]
    spread = np.array([-1, -1, 1, 1])[None, None, :, None]
    forecast = xr.DataArray(
        offset + spread + np.zeros((2, 3, 4, 10)),
        dims=("time", "lead_time", "member", "station"),
        coords={"lead_time": lead, "member": members},
    )
    truth[:, 2, :5] = np.nan
    table = lead_scores(forecast, truth)
    assert list(table.lead_hours) == [0, 6, 12]
    assert list(table.n) == [20, 20, 10]
    np.testing.assert_allclose(table.rmse_K, [1, 2, 3])
    np.testing.assert_allclose(table.bias_K, [1, 2, 3])
    np.testing.assert_allclose(table.spread_K, np.sqrt(4 / 3))
    total = pooled(table)
    assert total.rmse_K == pytest.approx(np.sqrt((20 * 1 + 20 * 4 + 10 * 9) / 50))


def test_reference_is_read_at_each_forecast_valid_time():
    times = pd.date_range("2020-08-01", periods=10, freq="6h")
    reference = xr.DataArray(
        np.arange(10.0)[:, None] * np.ones((1, 2)), dims=("time", "station"), coords={"time": times}
    )
    inits = times[[0, 4]]
    lead = pd.to_timedelta([0, 6, 36], "h")
    valid = xr.DataArray(
        inits.values[:, None] + lead.values[None, :],
        dims=("time", "lead_time"),
        coords={"time": inits.values, "lead_time": lead.values},
    )
    picked = at_valid_times(reference, valid)
    assert picked.dims == ("time", "lead_time", "station")
    np.testing.assert_array_equal(picked.values[0, :, 0], [0, 1, 6])
    assert np.isnan(picked.values[1, 2]).all()


def test_elevation_bands_split_the_stations():
    observed = np.zeros((3, 4))
    fields = {"a": np.array([1.0, 1.0, 2.0, 2.0]) + observed}
    table = elevation_scores(fields, observed, np.array([10, 100, 600, 900]), [0, 500, 1000])
    assert table.set_index("band").rmse_K.round(3).to_dict() == {
        "all": round(np.sqrt(2.5), 3),
        "0-500 m": 1.0,
        "500-1000 m": 2.0,
    }


def test_combining_initialisations_matches_scoring_them_together():
    rng = np.random.default_rng(3)
    lead = pd.to_timedelta([0, 6], "h")
    forecast = xr.DataArray(
        rng.normal(size=(3, 2, 5, 8)), dims=("time", "lead_time", "member", "station"), coords={"lead_time": lead}
    )
    truth = xr.DataArray(rng.normal(size=(3, 2, 8)), dims=("time", "lead_time", "station"), coords={"lead_time": lead})
    truth[1, 0, :3] = np.nan
    together = lead_scores(forecast, truth)
    from downscaler.evaluation.ensemble_scores import combine_leads

    apart = combine_leads([lead_scores(forecast.isel(time=[i]), truth.isel(time=[i])) for i in range(3)])
    pd.testing.assert_frame_equal(apart[together.columns].reset_index(drop=True), together, check_dtype=False)


def test_a_single_field_is_scored_by_its_error_alone():
    lead = pd.to_timedelta([0, 6], "h")
    truth = xr.DataArray(np.zeros((2, 4)), dims=("lead_time", "station"), coords={"lead_time": lead})
    field = xr.DataArray(
        [[1.0, -1.0, 1.0, -1.0], [2.0, 2.0, 2.0, 2.0]], dims=("lead_time", "station"), coords={"lead_time": lead}
    )
    table = lead_scores(field, truth)
    np.testing.assert_allclose(table.rmse_K, [1, 2])
    np.testing.assert_allclose(table.bias_K, [0, 2])
    np.testing.assert_allclose(table.crps_K, [1, 2])
    np.testing.assert_allclose(table.spread_K, [0, 0])


def test_native_cells_keep_coarse_blocks_and_hide_the_sea():
    from downscaler.visualization.plots import native_cells

    coarse = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]], dims=("lat", "lon"), coords={"lat": [45.25, 45.0], "lon": [0.0, 0.25]}
    )
    lat2d, lon2d = np.meshgrid([45.27, 45.2, 45.05, 44.98], [-0.05, 0.08, 0.2, 0.3], indexing="ij")
    land = np.ones(lat2d.shape, bool)
    land[0, 0] = False
    out = native_cells(coarse, lat2d, lon2d, land)
    assert np.isnan(out[0, 0])
    np.testing.assert_array_equal(out[1], [1, 1, 2, 2])
    np.testing.assert_array_equal(out[3], [3, 3, 4, 4])
