"""End-to-end tests of the pipeline scripts against synthetic sources."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_pairs  # noqa: E402
import download_cerra  # noqa: E402
import eval_downscaled  # noqa: E402
import fetch_forecasts  # noqa: E402
from conftest import coarse_terrain, diurnal, terrain  # noqa: E402
from downscale.config import load_config  # noqa: E402


def torch_modules():
    """Import the parts of the pipeline that need torch, skipping the test if it is absent.

    :returns: The ``downscale_forecasts`` module and the downscaler module.
    :rtype: tuple[types.ModuleType, types.ModuleType]
    """
    pytest.importorskip("torch", reason="torch is not installed")
    import downscale_forecasts

    from downscale import downscaler

    return downscale_forecasts, downscaler


@pytest.fixture
def source_grid():
    """Curvilinear source grid wider than the configured region."""
    ny, nx = 140, 150
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    lat = 33.0 + 0.16 * yy + 0.015 * xx
    lon = (-18.0 + 0.22 * xx - 0.02 * yy) % 360
    return lat, lon


@pytest.fixture
def fake_cds(source_grid):
    """Climate Data Store client and GRIB opener backed by synthetic fields."""
    lat, lon = source_grid
    orography = terrain(lat, lon)

    class Client:
        def __init__(self):
            self.requests = []

        def retrieve(self, dataset, request, target):
            self.requests.append(request)
            Path(target).write_text(json.dumps(request))

    def opener(path):
        request = json.loads(Path(path).read_text())
        coords = {
            "latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon),
            "step": np.timedelta64(0, "ns"),
        }
        if "orography" in request["variable"]:
            return xr.Dataset(
                {"orog": (("y", "x"), orography.astype("f4")), "lsm": (("y", "x"), np.ones_like(lat, "f4"))},
                coords=coords,
            )
        times = pd.DatetimeIndex([
            pd.Timestamp(int(request["year"]), int(request["month"]), int(d)) + pd.Timedelta(t + ":00")
            for d in request["day"] for t in request["time"]
        ])
        values = (
            300 - 0.0065 * orography[None] + diurnal(times.hour.values)[:, None, None]
        ).astype("f4")
        return xr.Dataset(
            {"t2m": (("time", "y", "x"), values)},
            coords={**coords, "time": times.to_numpy(), "valid_time": ("time", times.to_numpy())},
        )

    return Client(), opener


@pytest.fixture
def config_path(tmp_path):
    """A configuration file pointing every path into the temporary directory."""
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "region": {"name": None, "box": [41.0, 51.5, -5.5, 10.0]},
        "target": {"years": [2014, 2014], "train_years": [2014, 2014]},
        "paths": {
            "target": str(tmp_path / "target"), "pairs": str(tmp_path / "pairs"),
            "downscaled": str(tmp_path / "downscaled"), "evaluation": str(tmp_path / "eval"),
            "stations_cache": str(tmp_path / "cache"),
        },
    }))
    return path


@pytest.fixture
def config(config_path):
    """Configuration parsed from :func:`config_path`."""
    return load_config(config_path)


def test_download_crops_checks_and_consolidates(config, fake_cds, tmp_path):
    client, opener = fake_cds
    out = Path(config.paths.target)
    out.mkdir(parents=True)
    download_cerra.do_static(config, out, client=client, opener=opener)
    static = xr.open_zarr(out / "static.zarr")
    assert static.orography.attrs["units"] == "m"
    assert float(static.lon.min()) < 0 < float(static.lon.max())
    for month in range(1, 13):
        tag, problems = download_cerra.do_month(config, out, 2014, month, client=client, opener=opener)
        assert problems == []
    january = xr.open_zarr(out / "months" / "t2m_201401.zarr")
    assert january.t2m.dims == ("time", "y", "x") and january.sizes["time"] == 31 * 4
    assert np.array_equal(january.lat.values, static.lat.values)
    assert not (out / "grib" / "t2m_201401.grib").exists()
    requests = len(client.requests)
    download_cerra.do_month(config, out, 2014, 1, client=client, opener=opener)
    assert len(client.requests) == requests
    consolidated = xr.open_zarr(download_cerra.consolidate(config, out))
    assert consolidated.sizes["time"] == 365 * 4


def test_hourly_download_fills_the_hours_between_analyses_from_forecasts(config, fake_cds):
    client, opener = fake_cds
    out = Path(config.paths.target)
    out.mkdir(parents=True)
    download_cerra.do_static(config, out, client=client, opener=opener)

    def forecast_opener(path):
        request = json.loads(Path(path).read_text())
        ds = opener(path)
        if "leadtime_hour" not in request:
            return ds
        steps = pd.to_timedelta([int(h) for h in request["leadtime_hour"]], "h").to_numpy()
        valid = ds.time.values[:, None] + steps[None, :]
        ds = ds.drop_vars(["step", "valid_time"]).expand_dims(step=steps, axis=1)
        return ds.assign_coords(valid_time=(("time", "step"), valid))

    tag, problems = download_cerra.do_month(
        config, out, 2014, 2, client=client, opener=forecast_opener, hourly=True, days=[3, 4]
    )
    assert problems == []
    month = xr.open_zarr(out / "months" / "t2m_201402.zarr")
    assert month.sizes["time"] == 48
    assert pd.DatetimeIndex(month.time.values)[0] == pd.Timestamp("2014-02-03")
    assert sorted(r["product_type"] for r in client.requests[-2:]) == ["analysis", "forecast"]


def test_a_published_crop_is_adopted_and_a_conflicting_one_refused(config, fake_cds, tmp_path):
    client, opener = fake_cds
    published = tmp_path / "published"
    published.mkdir()
    out = Path(config.paths.target)
    out.mkdir(parents=True)
    download_cerra.do_static(config, out, client=client, opener=opener)
    crop = json.loads((out / "crop.json").read_text())
    smaller = {**crop, "y0": crop["y0"] + 1, "x1": crop["x1"] - 1}
    (published / "manifest.json").write_text(json.dumps({"crop": smaller}))
    with pytest.raises(ValueError, match="different crop"):
        download_cerra.adopt_crop(out, str(published))
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    download_cerra.adopt_crop(fresh, str(published))
    download_cerra.do_static(config, fresh, client=client, opener=opener)
    static = xr.open_zarr(fresh / "static.zarr")
    assert (static.sizes["y"], static.sizes["x"]) == (smaller["y1"] - smaller["y0"], smaller["x1"] - smaller["x0"])


def test_download_flags_a_month_with_missing_times(config, fake_cds):
    client, opener = fake_cds
    out = Path(config.paths.target)
    out.mkdir(parents=True)
    download_cerra.do_static(config, out, client=client, opener=opener)

    def gappy_opener(path):
        ds = opener(path)
        return ds.isel(time=slice(4, None)) if "t2m" in ds else ds

    tag, problems = download_cerra.do_month(config, out, 2014, 3, client=client, opener=gappy_opener)
    assert problems and "missing" in problems[0]
    assert not (out / "months" / "t2m_201403.zarr").exists()


def test_apply_crop_rejects_a_different_grid(config, fake_cds):
    client, opener = fake_cds
    out = Path(config.paths.target)
    out.mkdir(parents=True)
    download_cerra.do_static(config, out, client=client, opener=opener)
    crop = json.loads((out / "crop.json").read_text())
    standardised = download_cerra.standardise(opener(out / "grib" / "static.grib"))
    with pytest.raises(ValueError, match="differs"):
        download_cerra.apply_crop(standardised.isel(y=slice(0, 100)), crop)


@pytest.fixture
def downloaded_target(config, fake_cds):
    """A full year of synthetic target data on disk."""
    client, opener = fake_cds
    out = Path(config.paths.target)
    out.mkdir(parents=True)
    download_cerra.do_static(config, out, client=client, opener=opener)
    for month in range(1, 13):
        download_cerra.do_month(config, out, 2014, month, client=client, opener=opener)
    return out


def test_pairs_align_in_space_and_recover_the_lapse_rate(
    config, config_path, downloaded_target, analysis_source, global_grid
):
    lat, lon = global_grid
    geopotential = xr.DataArray(
        coarse_terrain(lat[:, None], lon[None, :]) * 9.80665,
        dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
    )
    status = build_pairs.main(
        ["--config", str(config_path), "--target", str(downloaded_target), "--out", config.paths.pairs],
        source=analysis_source, geopotential=geopotential,
    )
    assert status == 0
    reports = [json.loads(line) for line in open(Path(config.paths.pairs) / "month_reports.jsonl")]
    assert len(reports) == 12 and all(r["corr"] > 0.9 and not r["problems"] for r in reports)
    month = xr.open_zarr(Path(config.paths.pairs) / "months" / "pairs_201407.zarr")
    target_month = xr.open_zarr(downloaded_target / "months" / "t2m_201407.zarr")
    assert np.array_equal(month.fine_t2m.values, target_month.t2m.values)
    assert np.isfinite(month.coarse_t2m.values).all()
    static = xr.open_zarr(Path(config.paths.pairs) / "static.zarr")
    expected = coarse_terrain(static.lat.values, static.lon.values)
    assert np.abs(static.era5_orography.values - expected).max() < 25


def test_pairs_consolidation_computes_statistics(
    config, config_path, downloaded_target, analysis_source, global_grid
):
    lat, lon = global_grid
    geopotential = xr.DataArray(
        coarse_terrain(lat[:, None], lon[None, :]) * 9.80665,
        dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
    )
    build_pairs.main(
        ["--config", str(config_path), "--target", str(downloaded_target), "--out", config.paths.pairs],
        source=analysis_source, geopotential=geopotential,
    )
    stats = build_pairs.consolidate(config, Path(config.paths.pairs))
    assert stats["n_train_steps"] == 365 * 4
    assert -7.5 < stats["checks"]["residual_vs_height_slope_K_per_km"] < -5.5
    assert (Path(config.paths.pairs) / "check.png").stat().st_size > 20_000


@pytest.fixture
def trained_model(tmp_path, static_fields, stats):
    """An untrained network saved as a checkpoint, enough to exercise inference."""
    torch = pytest.importorskip("torch", reason="torch is not installed")
    _, dsc = torch_modules()
    torch.manual_seed(0)
    net = dsc.DownscalerNet(dsc.Config(loss="crps", base=8, depth=3, noise_channels=4))
    with torch.no_grad():
        for parameter in net.unet.out.parameters():
            parameter.normal_(0, 0.05)
    path = tmp_path / "model.pt"
    dsc.Downscaler(net, stats, static_fields).save(path)
    return path


def test_downscaling_produces_an_ensemble_on_the_fine_grid(
    config, config_path, forecast_store, trained_model, static_fields
):
    downscale_forecasts, _ = torch_modules()
    status = downscale_forecasts.main([
        "--config", str(config_path), "--model", str(trained_model),
        "--forecasts", *sorted(str(p) for p in forecast_store.glob("*.zarr")),
        "--out", config.paths.downscaled, "--samples", "3", "--device", "cpu",
    ])
    assert status == 0
    result = xr.open_zarr(Path(config.paths.downscaled) / "forecast_2023-08-05.zarr")
    height, width = static_fields["lat"].shape
    assert result.t2m_ds.dims == ("time", "lead_time", "ensemble", "sample", "y", "x")
    assert result.t2m_ds.shape == (1, 12, 3, 3, height, width)
    assert result.t2m_interp.shape == (1, 12, 3, height, width)
    expected = (300 - 0.0065 * terrain(static_fields["lat"], static_fields["lon"])) + diurnal(6)
    assert np.abs(result.t2m_interp.isel(time=0, lead_time=0, ensemble=0).values - expected).mean() < 0.5
    assert (Path(config.paths.downscaled) / "map_check.png").exists()


def test_fetched_aifs_ens_forecasts_can_be_downscaled(config, config_path, trained_model, static_fields, tmp_path):
    from test_sources import earthmover_archive

    downscale_forecasts, _ = torch_modules()
    inits = pd.date_range("2025-08-01", periods=4, freq="6h")
    archive = earthmover_archive(inits, ingested=[np.timedelta64(36, "h")] * 4).rename(
        {"init_time": "time", "ensemble_member": "ensemble", "latitude": "lat", "longitude": "lon"}
    )
    out = tmp_path / "forecasts"
    args = ["--config", str(config_path), "--start", "2025-08-01", "--end", "2025-08-01", "--out", str(out),
            "--members", "2", "--max-lead-hours", "12"]
    assert fetch_forecasts.main(args, archive=archive) == 0
    stores = sorted(out.glob("*.zarr"))
    assert [p.name for p in stores] == ["aifs-ens_20250801T0000.zarr", "aifs-ens_20250801T1200.zarr"]
    assert fetch_forecasts.main(args, archive=archive) == 0

    status = downscale_forecasts.main([
        "--config", str(config_path), "--model", str(trained_model), "--forecasts", str(stores[0]),
        "--out", config.paths.downscaled, "--samples", "2", "--device", "cpu",
    ])
    assert status == 0
    result = xr.open_zarr(Path(config.paths.downscaled) / stores[0].name)
    assert result.t2m_ds.shape == (1, 3, 2, 2, *static_fields["lat"].shape)
    assert float(result.t2m_interp.mean()) == pytest.approx(273.15 + 15, abs=1)


def test_downscaling_refuses_an_unfinished_forecast(config, config_path, forecast_store, trained_model):
    downscale_forecasts, _ = torch_modules()
    marker = forecast_store / "forecast_2023-08-05.zarr.done"
    marker.unlink()
    with pytest.raises(RuntimeError, match="no .done marker"):
        downscale_forecasts.main([
            "--config", str(config_path), "--model", str(trained_model),
            "--forecasts", str(forecast_store / "forecast_2023-08-05.zarr"),
            "--out", config.paths.downscaled, "--device", "cpu",
        ])


class _StationProvider:
    """Provider generating observations consistent with the synthetic terrain."""

    name, license = "fake", "test licence"

    def __init__(self, n=50):
        rng = np.random.default_rng(1)
        self.lat = rng.uniform(42, 50.5, n)
        self.lon = rng.uniform(-4, 8.5, n)
        self.elevation = (terrain(self.lat, self.lon) + rng.uniform(-100, 500, n)).clip(0)
        self.table = pd.DataFrame({
            "station_id": [f"{i:08d}" for i in range(n)], "name": "station",
            "latitude": self.lat, "longitude": self.lon, "elevation": self.elevation,
            "start_date": pd.Timestamp("2000-01-01"), "end_date": pd.NaT,
        })

    def list_stations(self, region, start, end, variables):
        return self.table.copy()

    def fetch(self, ids, start, end, variables):
        hours = pd.date_range(start, end, freq="1h")
        rows = []
        for station_id in ids:
            index = int(station_id)
            values = (
                300 - 0.0065 * self.elevation[index] + diurnal(hours.hour)
                + np.random.default_rng(index).normal(0, 1.2, len(hours))
            )
            rows.append(pd.DataFrame({
                "station_id": station_id, "variable": "t2m", "time": hours, "value": values,
            }))
        return pd.concat(rows, ignore_index=True)


def test_evaluation_compares_both_forecasts_and_the_reading_paths_agree(
    config, config_path, forecast_store, trained_model
):
    downscale_forecasts, _ = torch_modules()
    downscale_forecasts.main([
        "--config", str(config_path), "--model", str(trained_model),
        "--forecasts", *sorted(str(p) for p in forecast_store.glob("*.zarr")),
        "--out", config.paths.downscaled, "--samples", "3", "--device", "cpu",
    ])
    config.evaluation.n_bootstrap = 20
    summary = eval_downscaled.run(
        config,
        sorted(str(p) for p in forecast_store.glob("*.zarr")),
        sorted(str(p) for p in Path(config.paths.downscaled).glob("forecast_*.zarr")),
        config.paths.evaluation,
        provider=_StationProvider(),
    )
    out = Path(config.paths.evaluation)
    for name in (
        "by_lead.csv", "by_region_elevation.csv", "heat_events.csv",
        "summary.json", "scorecard.png", "heatwave_map.png",
    ):
        assert (out / name).stat().st_size > 0
    by_lead = pd.read_csv(out / "by_lead.csv").pivot(index="lead_h", columns="method", values="rmse")
    assert np.abs(by_lead["coarse_on_fine_grid"] - by_lead["coarse"]).max() < 0.15
    assert summary["members"] == {"coarse": 3, "downscaled": 9}
    assert np.isfinite(summary["rmse_K"]["downscaled"])


def test_evaluation_rejects_a_stale_downscaled_store(config, config_path, forecast_store, trained_model):
    downscale_forecasts, _ = torch_modules()
    downscale_forecasts.main([
        "--config", str(config_path), "--model", str(trained_model),
        "--forecasts", str(forecast_store / "forecast_2023-08-05.zarr"),
        "--out", config.paths.downscaled, "--samples", "2", "--device", "cpu",
    ])
    path = Path(config.paths.downscaled) / "forecast_2023-08-05.zarr"
    truncated = xr.open_zarr(path).isel(ensemble=slice(0, 2)).load()
    import shutil

    shutil.rmtree(path)
    truncated.to_zarr(path)
    stations = xr.Dataset(
        {"t2m": (("time", "station"), np.full((5, 1), 300.0, "f4"))},
        coords={
            "time": pd.date_range("2023-08-05", periods=5, freq="6h").to_numpy().astype("datetime64[ns]"),
            "station": ["a"], "latitude": ("station", [45.0]), "longitude": ("station", [2.0]),
            "elevation": ("station", [300.0]),
        },
    )
    with pytest.raises(ValueError, match="stale"):
        eval_downscaled.at_stations(
            forecast_store / "forecast_2023-08-05.zarr", path, stations, "t2m"
        )


def test_spectrum_script_reports_a_ratio_to_the_target(config_path, tmp_path, trained_model, static_fields):
    """The spectrum script runs on a pairs directory and writes its three outputs."""
    torch_modules()
    import check_spectrum

    lat, lon = static_fields["lat"], static_fields["lon"]
    times = pd.date_range("2014-07-01T12", periods=6, freq="1D").to_numpy().astype("datetime64[ns]")
    rng = np.random.default_rng(0)
    coarse = (290 + rng.normal(0, 1, (len(times), *lat.shape))).astype("f4")
    fine = (coarse + rng.normal(0, 0.5, coarse.shape)).astype("f4")
    pairs = Path(tmp_path) / "pairs"
    (pairs / "months").mkdir(parents=True)
    xr.Dataset(
        {"coarse_t2m": (("time", "y", "x"), coarse), "fine_t2m": (("time", "y", "x"), fine)},
        coords={"time": times, "lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
    ).to_zarr(pairs / "months" / "pairs_201407.zarr")

    out = Path(tmp_path) / "spectrum"
    status = check_spectrum.main([
        "--config", str(config_path), "--model", str(trained_model), "--pairs", str(pairs),
        "--out", str(out), "--n-times", "4", "--hour", "12", "--members", "3", "--device", "cpu",
    ])
    assert status == 0
    for name in ("spectrum.csv", "spectrum.png", "spectrum.json"):
        assert (out / name).stat().st_size > 0
    comparison = pd.read_csv(out / "spectrum.csv")
    assert {"target", "coarse", "model member", "model mean"} <= set(comparison.columns)
    assert (comparison.wavelength_km.diff().dropna() < 0).all()

def test_hourly_forecasts_are_written_for_every_hour_and_scored_against_stations(
    config, config_path, forecast_store, trained_model, static_fields, stats, tmp_path
):
    torch_modules()
    import torch

    import downscale_hourly
    from downscale import hourly

    torch.manual_seed(0)
    net = hourly.HourlyNet(hourly.HourlyConfig(loss="crps", base=8, depth=3, noise_channels=4))
    with torch.no_grad():
        for parameter in net.unet.out.parameters():
            parameter.normal_(0, 0.05)
    hourly_path = tmp_path / "hourly.pt"
    hourly.HourlyDownscaler(net, {**stats, "tendency": {"mean": 0.0, "std": 2.0}}, static_fields).save(hourly_path)
    out = tmp_path / "hourly_out"
    status = downscale_hourly.main([
        "--config", str(config_path), "--hourly", str(hourly_path), "--spatial", str(trained_model),
        "--forecasts", *sorted(str(p) for p in forecast_store.glob("*.zarr")),
        "--out", str(out), "--samples", "2", "--device", "cpu",
    ])
    assert status == 0
    name = "forecast_2023-08-05.zarr"
    coarse = xr.open_zarr(out / "forecasts" / name)
    fine, spatial = xr.open_zarr(out / "hourly" / name), xr.open_zarr(out / "spatial" / name)
    assert coarse.sizes["lead_time"] == 66  # +6 h to +71 h
    assert (out / "forecasts" / f"{name}.done").exists()
    assert np.array_equal(fine.lead_time.values, spatial.lead_time.values)
    assert np.allclose(fine.t2m_interp.values, spatial.t2m_interp.values, atol=1e-3)
    assert fine.t2m_ds.shape == (1, 66, 3, 2, *static_fields["lat"].shape)

    config.evaluation.n_bootstrap = 20
    summary = eval_downscaled.run(
        config, sorted(str(p) for p in (out / "forecasts").glob("*.zarr")),
        sorted(str(p) for p in (out / "hourly").glob("*.zarr")), tmp_path / "eval", provider=_StationProvider(),
    )
    by_lead = pd.read_csv(tmp_path / "eval" / "by_lead.csv")
    assert sorted(by_lead.lead_h.unique())[:3] == [6, 7, 8]
    assert np.isfinite(summary["rmse_K"]["downscaled"])
