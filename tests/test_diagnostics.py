"""Tests for the field diagnostics: land masking, edge profile and sea report."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from downscale.diagnostics import (
    distance_from_edge,
    edge_figure,
    edge_profile,
    land_mask,
    mask_sea,
    sea_report,
    suggested_margin,
)


def make_store(ny=80, nx=90, n_samples=4, edge_width=0, sea_factor=1.0, seed=0):
    """Build a downscaled store with a controllable edge effect and land/sea contrast.

    :param edge_width: Width in cells of an inflated band along the boundary.
    :type edge_width: int
    :param sea_factor: Multiplier applied to the correction and spread over sea.
    :type sea_factor: float
    :returns: Store in the shape the downscaling stage writes.
    :rtype: xarray.Dataset
    """
    rng = np.random.default_rng(seed)
    coarse = np.full((1, 3, 2, ny, nx), 290.0, "f4")
    scale = np.ones((ny, nx), "f4")
    if edge_width:
        scale[distance_from_edge((ny, nx)) < edge_width] = 4.0
    land = np.ones((ny, nx), bool)
    land[:, : nx // 3] = False
    if sea_factor != 1.0:
        scale = scale * np.where(land, 1.0, sea_factor)
    fine = coarse[..., None, :, :] + rng.normal(0, 1, (1, 3, 2, n_samples, ny, nx)) * scale
    return xr.Dataset(
        {
            "t2m_interp": (("time", "lead_time", "ensemble", "y", "x"), coarse),
            "t2m_ds": (("time", "lead_time", "ensemble", "sample", "y", "x"), fine.astype("f4")),
            "land_sea_mask": (("y", "x"), land.astype("f4")),
        }
    )


def test_distance_from_edge_is_zero_on_the_boundary():
    distance = distance_from_edge((5, 7))
    assert distance[0, 0] == 0 and distance[0, 3] == 0 and distance[4, 6] == 0
    assert distance[2, 3] == 2
    assert distance.max() == 2


def test_land_mask_reads_the_stored_field_and_returns_none_when_absent():
    store = make_store()
    mask = land_mask(store)
    assert mask.dtype == bool and mask.sum() == mask.size - mask.shape[0] * (mask.shape[1] // 3)
    assert land_mask(store.drop_vars("land_sea_mask")) is None


def test_mask_sea_blanks_only_sea_points():
    store = make_store()
    mask = land_mask(store)
    field = np.ones(mask.shape, "f4")
    masked = mask_sea(field, mask)
    assert np.isnan(masked[~mask]).all()
    assert np.isfinite(masked[mask]).all()
    assert np.array_equal(mask_sea(field, None), field)


def test_edge_profile_finds_an_injected_edge_band():
    profile = edge_profile(make_store(edge_width=5), max_distance=20, interior_from=15)
    assert list(profile.distance_cells[:3]) == [0, 1, 2]
    assert profile.loc[profile.distance_cells < 5, "spread_ratio"].min() > 2
    assert abs(profile.loc[profile.distance_cells > 8, "spread_ratio"].mean() - 1) < 0.1


def test_edge_profile_is_flat_without_an_edge_effect():
    profile = edge_profile(make_store(edge_width=0), max_distance=20, interior_from=15)
    assert (profile.spread_ratio - 1).abs().max() < 0.2


def test_suggested_margin_matches_the_injected_width():
    wide = suggested_margin(edge_profile(make_store(edge_width=5), interior_from=15))
    none = suggested_margin(edge_profile(make_store(edge_width=0), interior_from=15), tolerance=0.3)
    assert wide == 5
    assert none == 0


def test_edge_profile_can_restrict_itself_to_land():
    store = make_store(edge_width=4)
    land = edge_profile(store, interior_from=15, land_only=True)
    everywhere = edge_profile(store, interior_from=15, land_only=False)
    assert land.n_points.iloc[0] < everywhere.n_points.iloc[0]


def test_edge_profile_rejects_an_unreachable_interior():
    with pytest.raises(ValueError, match="no points further"):
        edge_profile(make_store(ny=20, nx=20), interior_from=50)


def test_sea_report_detects_a_sea_land_contrast():
    report = sea_report(make_store(sea_factor=3.0))
    assert report["spread_ratio_sea_over_land"] == pytest.approx(3.0, rel=0.1)
    assert report["correction_ratio_sea_over_land"] > 2
    assert 0 < report["land_fraction"] < 1


def test_sea_report_is_empty_without_a_mask():
    assert sea_report(make_store().drop_vars("land_sea_mask")) == {}


def test_edge_figure_is_written(tmp_path):
    profile = edge_profile(make_store(edge_width=5), interior_from=15)
    edge_figure(profile, suggested_margin(profile), tmp_path / "edge.png")
    assert (tmp_path / "edge.png").stat().st_size > 10_000


def smooth_field(ny, nx, scale_cells, rng):
    """White noise low-pass filtered to a given scale, giving a known spectral roll-off."""
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(rng.normal(size=(ny, nx)), scale_cells, mode="wrap")


def test_grid_spacing_matches_a_known_geometry():
    from downscale.diagnostics import grid_spacing_km

    ny, nx = 40, 50
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    lat = 45.0 + 0.05 * yy
    lon = 2.0 + 0.05 * xx / np.cos(np.deg2rad(45.0))
    assert grid_spacing_km(lat, lon) == pytest.approx(5.56, rel=0.02)


def test_white_noise_has_a_flat_spectrum():
    from downscale.diagnostics import power_spectrum

    rng = np.random.default_rng(0)
    spectrum = power_spectrum(rng.normal(size=(8, 96, 96)), spacing_km=5.0)
    middle = spectrum[(spectrum.wavelength_km < 60) & (spectrum.wavelength_km > 15)]
    assert middle.power.std() / middle.power.mean() < 0.25


def test_a_smooth_field_loses_power_at_short_wavelengths():
    from downscale.diagnostics import power_spectrum

    rng = np.random.default_rng(1)
    fields = np.stack([smooth_field(96, 96, 4, rng) for _ in range(6)])
    spectrum = power_spectrum(fields, spacing_km=5.0)
    long_waves = spectrum[spectrum.wavelength_km > 100].power.mean()
    short_waves = spectrum[spectrum.wavelength_km < 20].power.mean()
    assert short_waves < long_waves / 100


def test_comparison_detects_too_little_and_too_much_small_scale_power():
    from downscale.diagnostics import effective_resolution_km, spectrum_comparison

    rng = np.random.default_rng(2)
    target = np.stack([smooth_field(96, 96, 2, rng) for _ in range(6)])
    smoother = np.stack([smooth_field(96, 96, 6, rng) for _ in range(6)])
    noisy = target + rng.normal(0, target.std(), target.shape)
    comparison = spectrum_comparison({"target": target, "smoother": smoother, "noisy": noisy}, spacing_km=5.0)
    short = comparison[comparison.wavelength_km < 25]
    assert short.smoother_over_target.mean() < 0.5
    assert short.noisy_over_target.mean() > 2
    assert effective_resolution_km(comparison, "smoother", "target") > effective_resolution_km(
        comparison, "noisy", "target", tolerance=0.5
    ) or np.isnan(effective_resolution_km(comparison, "noisy", "target"))


def test_spectrum_figure_is_written(tmp_path):
    from downscale.diagnostics import spectrum_comparison, spectrum_figure

    rng = np.random.default_rng(3)
    target = np.stack([smooth_field(64, 64, 2, rng) for _ in range(3)])
    comparison = spectrum_comparison({"target": target, "model": target * 1.1}, spacing_km=5.0)
    spectrum_figure(comparison, ["target", "model"], tmp_path / "spectrum.png")
    assert (tmp_path / "spectrum.png").stat().st_size > 10_000


def station_case(n_stations=60, n_init=3, n_lead=4, n_members=5, seed=0, mountain_gain=True):
    """Forecasts at stations where the downscaled method is better at high elevation."""
    import xarray as xr

    rng = np.random.default_rng(seed)
    elevation = rng.uniform(0, 2500, n_stations)
    coords = {
        "init_time": np.array(["2025-08-01", "2025-08-03", "2025-08-05"], "datetime64[ns]")[:n_init],
        "lead_time": (np.arange(1, n_lead + 1) * 6).astype("timedelta64[h]").astype("timedelta64[ns]"),
        "station": [f"s{i}" for i in range(n_stations)],
        "latitude": ("station", rng.uniform(42, 51, n_stations)),
        "longitude": ("station", rng.uniform(-4, 8, n_stations)),
        "elevation": ("station", elevation),
    }
    truth = 300 + rng.normal(0, 3, (n_init, n_lead, n_stations))
    coarse_error = 0.002 * elevation if mountain_gain else np.zeros(n_stations)

    def build(offset):
        return xr.DataArray(
            truth[:, :, None, :] + rng.normal(0, 1, (n_init, n_lead, n_members, n_stations)) + offset,
            dims=("init_time", "lead_time", "ensemble", "station"),
            coords={**coords, "ensemble": np.arange(n_members)},
        )

    forecast = xr.Dataset({"coarse": build(coarse_error[None, None, None, :]), "downscaled": build(0.0)})
    obs = xr.DataArray(truth, dims=("init_time", "lead_time", "station"), coords=coords)
    return forecast, obs


def test_station_errors_recover_a_known_bias():
    from downscale.diagnostics import StationErrors

    forecast, obs = station_case(mountain_gain=False)
    forecast["coarse"] = forecast["coarse"] + 2.0
    accumulator = StationErrors(["coarse", "downscaled"])
    accumulator.add(forecast, obs)
    table = accumulator.table()
    assert len(table) == obs.sizes["station"]
    assert table.bias_coarse.mean() == pytest.approx(2.0, abs=0.2)
    assert abs(table.bias_downscaled.mean()) < 0.2
    assert set(table.columns) >= {"latitude", "longitude", "elevation", "rmse_coarse", "rmse_downscaled"}


def test_improvement_tracks_elevation_when_it_should(tmp_path):
    from downscale.diagnostics import StationErrors, improvement_figure

    forecast, obs = station_case(mountain_gain=True)
    accumulator = StationErrors(["coarse", "downscaled"])
    accumulator.add(forecast, obs)
    summary = improvement_figure(
        accumulator.table(), "coarse", "downscaled", None, None, None, tmp_path / "improvement.png"
    )
    assert summary["correlation_gain_elevation"] > 0.5
    assert summary["mean_gain_above_1000m_K"] > summary["mean_gain_below_500m_K"]
    assert (tmp_path / "improvement.png").stat().st_size > 10_000


def test_improvement_is_flat_when_the_methods_are_equivalent(tmp_path):
    from downscale.diagnostics import StationErrors, improvement_figure

    forecast, obs = station_case(mountain_gain=False)
    accumulator = StationErrors(["coarse", "downscaled"])
    accumulator.add(forecast, obs)
    summary = improvement_figure(accumulator.table(), "coarse", "downscaled", None, None, None, tmp_path / "flat.png")
    assert abs(summary["correlation_gain_elevation"]) < 0.4


def test_quantiles_show_a_damped_distribution(tmp_path):
    from downscale.diagnostics import DistributionSamples, quantile_figure

    forecast, obs = station_case(n_stations=200, mountain_gain=False)
    forecast["coarse"].mean("station")
    forecast["coarse"] = (forecast["coarse"] - forecast["coarse"].mean()) * 0.5 + forecast["coarse"].mean()
    samples = DistributionSamples(["coarse", "downscaled"])
    samples.add(forecast, obs)
    quantiles = samples.quantiles()
    top = quantiles.level == 0.99
    assert float(quantiles.loc[top, "coarse member"].iloc[0]) < float(quantiles.loc[top, "observed"].iloc[0])
    assert float(quantiles.loc[top, "downscaled member"].iloc[0]) > float(quantiles.loc[top, "coarse member"].iloc[0])
    quantile_figure(quantiles, ["coarse", "downscaled"], {}, tmp_path / "q.png")
    assert (tmp_path / "q.png").stat().st_size > 10_000


def test_station_error_figure_is_written(tmp_path):
    from downscale.diagnostics import StationErrors, station_error_figure

    forecast, obs = station_case()
    accumulator = StationErrors(["coarse", "downscaled"])
    accumulator.add(forecast, obs)
    station_error_figure(accumulator.table(), ["coarse", "downscaled"], {}, tmp_path / "errors.png")
    assert (tmp_path / "errors.png").stat().st_size > 10_000


def test_distance_to_coast_grows_inland():
    from downscale.diagnostics import distance_to_coast_km

    mask = np.ones((40, 60), bool)
    mask[:, :10] = False
    distance = distance_to_coast_km(mask, spacing_km=5.0)
    assert distance[20, 5] == 0
    assert distance[20, 10] == pytest.approx(5.0, abs=0.1)
    assert distance[20, 30] == pytest.approx(105.0, abs=0.1)


def test_coast_profile_separates_coastal_from_inland_gains(tmp_path):
    from downscale.diagnostics import coast_figure, coast_profile

    rng = np.random.default_rng(0)
    n = 300
    distance = rng.uniform(0, 300, n)
    table = pd.DataFrame(
        {
            "station": [f"s{i}" for i in range(n)],
            "latitude": rng.uniform(42, 51, n),
            "longitude": rng.uniform(-4, 8, n),
            "elevation": rng.uniform(0, 2000, n),
            "rmse_coarse": 3.0,
            "rmse_downscaled": np.where(distance < 25, 3.0, 2.0),
        }
    )
    profile = coast_profile(table, distance, "coarse", "downscaled")
    assert profile.loc[profile.band == "0-10 km", "mean_gain_K"].iloc[0] == pytest.approx(0.0)
    assert profile.loc[profile.band == "100-200 km", "mean_gain_K"].iloc[0] == pytest.approx(1.0)
    summary = coast_figure(table, distance, profile, tmp_path / "coast.png")
    assert summary["mean_gain_beyond_25km_K"] > summary["mean_gain_within_25km_K"]
    assert (tmp_path / "coast.png").stat().st_size > 10_000
