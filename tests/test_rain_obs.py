"""Observation readers and observation-aware rainfall metrics."""

import gzip

import numpy as np
import pandas as pd
import pytest

from downscaler.data.precipitation_observations import (
    Aggregator,
    at_stations,
    choose_test_days,
    gauge_matrix,
    read_gauge_file,
    read_odim,
)
from downscaler.evaluation import precipitation_metrics as metrics
from downscaler.evaluation.ensemble_scores import fair_crps


def blob(cx, cy=20, r=3, peak=10.0, size=40):
    y, x = np.mgrid[0:size, 0:size]
    return peak * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r**2))


def test_crps_matches_existing_fair_crps_and_reduces_to_mae():
    rng = np.random.default_rng(0)
    forecast = rng.gamma(0.5, 2, (6, 8, 5, 7))
    reference = rng.gamma(0.5, 2, (6, 5, 7))
    result = metrics.crps(forecast, reference)
    assert result["crps_mm"] == pytest.approx(fair_crps(forecast, reference, axis=1).mean())
    skill = np.abs(forecast - reference[:, None]).mean()
    spread = np.abs(forecast[:, :, None] - forecast[:, None]).mean()
    assert result["ensemble_crps_mm"] == pytest.approx(skill - spread / 2)
    single = metrics.crps(forecast[:, :1], reference)
    assert single["crps_mm"] == pytest.approx(np.abs(forecast[:, 0] - reference).mean())


def test_missing_reference_points_are_ignored_everywhere():
    rng = np.random.default_rng(1)
    forecast = rng.gamma(0.4, 3, (4, 3, 20, 20))
    reference = rng.gamma(0.4, 3, (4, 20, 20))
    gappy = reference.copy()
    gappy[:, :, 10:] = np.nan
    a = metrics.all_scores(forecast, gappy, gridded=False)
    b = metrics.all_scores(forecast[..., :10], reference[..., :10], gridded=False)
    assert a["amount"] == pytest.approx(b["amount"])
    assert a["probability"] == pytest.approx(b["probability"])
    assert a["occurrence"]["csi_1mm"] == pytest.approx(b["occurrence"]["csi_1mm"])


def test_double_penalty_csi_zero_but_fss_recovers_with_scale():
    observed = blob(12)[None]
    shifted = blob(28)[None, None]
    occurrence = metrics.occurrence(shifted, observed, thresholds=(1.0,))
    assert occurrence["csi_1mm"] == 0
    assert occurrence["frequency_bias_1mm"] == pytest.approx(1.0)
    curve = metrics.structure(shifted, observed, thresholds=(1.0,), sizes=(1, 9, 33))["fss_1mm"]
    assert curve[0] == pytest.approx(0) and curve[0] < curve[1] < curve[2]
    perfect = metrics.structure(observed[:, None], observed, thresholds=(1.0,), sizes=(1, 9))
    assert perfect["fss_1mm"] == [1.0, 1.0] and perfect["useful_scale_km_1mm"] == 10.0


def test_blurring_lowers_small_scale_power_and_extremes():
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(2)
    sharp = rng.gamma(0.3, 4, (5, 64, 64))
    blurred = gaussian_filter(sharp, (0, 3, 3))
    a, b = metrics.spectrum(sharp), metrics.spectrum(blurred)
    assert b["power"][-1] < 0.01 * a["power"][-1]
    assert (
        metrics.intensity(blurred[:, None], sharp)["q99.9_mm"] < metrics.intensity(sharp[:, None], sharp)["q99.9_mm"]
    )


def test_complete_box_finds_the_fully_observed_square():
    reference = np.full((2, 50, 60), 1.0)
    reference[0, :10] = np.nan
    reference[1, :, 45:] = np.nan
    row, col, size, hours = metrics.complete_box(reference, min_size=8)
    assert size == 40 and row >= 10 and col + size <= 45 and hours.all()


def test_complete_box_skips_empty_and_patchy_hours():
    reference = np.full((3, 40, 40), 1.0)
    reference[1] = np.nan  # empty hour, like EURADCLIM on 2023-10-30
    reference[2, ::2, ::2] = np.nan  # patchy hour: no large square is always observed
    row, col, size, hours = metrics.complete_box(reference, min_size=8)
    assert size == 40 and hours.tolist() == [True, False, False]


def test_aggregator_averages_pixels_and_drops_poorly_observed_cells():
    lat, lon = np.array([45.0, 45.1]), np.array([2.0, 2.1, 2.2])
    src_lat, src_lon = np.meshgrid(np.linspace(44.96, 45.14, 10), np.linspace(1.96, 2.24, 15), indexing="ij")
    aggregate = Aggregator(src_lat, src_lon, lat, lon)
    values = np.ones_like(src_lat) * 3
    assert np.allclose(aggregate(values), 3)
    values[:5, :5] = np.nan  # most of the south-west cell
    out = aggregate(values)
    assert np.isnan(out[0, 0]) and np.isfinite(out[1, 2])


def test_read_odim_values_nodata_and_coordinates(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "RAD_202210150100.h5"
    projdef = "+proj=laea +lat_0=55 +lon_0=10 +x_0=1950000 +y_0=-2100000 +units=m +ellps=WGS84"
    with h5py.File(path, "w") as h5:
        data = np.array([[0.0, 1.5], [-9999000.0, 4.0]])
        h5.create_dataset("dataset1/data1/data", data=data)
        h5["dataset1/data1"].create_group("what").attrs.update({"nodata": -9999000.0})
        h5.create_group("where").attrs.update(
            {
                "projdef": projdef.encode(),
                "UL_lon": 2.0,
                "UL_lat": 47.0,
                "xscale": 2000.0,
                "yscale": 2000.0,
                "xsize": 2,
                "ysize": 2,
            }
        )
    values, lat, lon = read_odim(path)
    assert values[0, 1] == 1.5 and np.isnan(values[1, 0])
    assert lat[0, 0] < 47.0 and lon[0, 0] > 2.0 and lat[1, 0] < lat[0, 0]
    assert abs(lat[0, 0] - 47.0) < 0.03 and abs(lon[0, 0] - 2.0) < 0.03


def test_gauge_file_keeps_good_quality_and_right_labelled_hours(tmp_path):
    text = (
        "NUM_POSTE;NOM_USUEL;LAT;LON;ALTI;AAAAMMJJHH;RR1;QRR1;T;QT\n"
        "13001009;AIX;43.53;5.42;173;2022101501;1.2;1;10;1\n"
        "13001009;AIX;43.53;5.42;173;2022101502;7.0;2;10;1\n"  # doubtful
        "13001009;AIX;43.53;5.42;173;2022101503;;1;10;1\n"  # missing
        "13001009;AIX;43.53;5.42;173;2022101601;3.0;1;10;1\n"
    )  # not requested
    source = tmp_path / "H_13_previous-2020-2024.csv.gz"
    source.write_bytes(gzip.compress(text.encode()))
    times = pd.date_range("2022-10-15T01:00", periods=3, freq="h")
    frame = read_gauge_file(source, times)
    assert frame.rain_mm.tolist() == pytest.approx([1.2]) and frame.time[0] == pd.Timestamp("2022-10-15T01:00")
    assert frame.station_id[0] == "13001009"


def test_gauge_matrix_places_stations_in_their_cell_and_filters_gaps():
    times = pd.date_range("2022-10-15T01:00", periods=10, freq="h")
    rows = [{"station_id": "A", "lat": 45.04, "lon": 2.16, "time": t, "rain_mm": 1.0} for t in times]
    rows += [{"station_id": "B", "lat": 45.0, "lon": 2.0, "time": t, "rain_mm": 0.0} for t in times[:5]]
    values, stations = gauge_matrix(pd.DataFrame(rows), times, np.array([45.0, 45.1]), np.array([2.0, 2.1, 2.2]))
    assert list(stations.station_id) == ["A"] and (stations.row[0], stations.col[0]) == (0, 2)
    fields = np.arange(2 * 1 * 2 * 3, dtype=float).reshape(2, 1, 2, 3)
    assert at_stations(fields, stations)[:, 0, 0].tolist() == [2.0, 8.0]


def test_test_days_are_reproducible_and_wet():
    days = pd.date_range("2022-01-01", "2022-12-31", freq="D")
    rng = np.random.default_rng(3)
    daily = pd.DataFrame({"wet_fraction": rng.random(len(days))}, index=days)
    a, b = choose_test_days(daily, 2, 0.75, seed=0), choose_test_days(daily, 2, 0.75, seed=0)
    assert a == b and len(a) == 8
    assert daily.loc[a].wet_fraction.min() >= daily.wet_fraction.quantile(0.75) - 0.2
