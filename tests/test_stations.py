"""Tests for station loading, quality control and caching."""

import numpy as np
import pandas as pd
import pytest

from downscale.stations import QCConfig, holdout_mask, load_stations


START, END = pd.Timestamp("2023-08-05"), pd.Timestamp("2023-08-07 23:00")


class FakeProvider:
    """Network with closed, gappy, unknown-start and sentinel-valued stations."""

    name, license = "fake", "test licence"

    def __init__(self):
        self.fetch_calls: list[list[str]] = []
        self.table = pd.DataFrame(
            [
                ("A", "lowland", 45.0, 4.0, 200.0, "2000-01-01", None),
                ("B", "mountain", 45.5, 6.5, 2000.0, "2005-01-01", None),
                ("C", "closed", 45.1, 4.1, 300.0, "1990-01-01", "2010-12-31"),
                ("D", "gappy", 45.2, 4.2, 250.0, "2000-01-01", None),
                ("E", "sentinels", 45.3, 4.3, 260.0, "2000-01-01", None),
                ("F", "unknown start", 45.4, 4.4, 270.0, None, None),
                ("G", "late start", 45.6, 4.6, 280.0, "2023-08-06", None),
            ],
            columns=["station_id", "name", "latitude", "longitude", "elevation", "start_date", "end_date"],
        )
        for column in ("start_date", "end_date"):
            self.table[column] = pd.to_datetime(self.table[column])

    def list_stations(self, region, start, end, variables):
        lat_min, lat_max, lon_min, lon_max = region
        table = self.table
        inside = (
            (table.latitude >= lat_min)
            & (table.latitude <= lat_max)
            & (table.longitude >= lon_min)
            & (table.longitude <= lon_max)
        )
        return table[inside].copy()

    def fetch(self, ids, start, end, variables):
        self.fetch_calls.append(list(ids))
        rows = []
        for station_id in ids:
            if station_id == "C":
                continue
            hours = pd.date_range(start, end, freq="1h")
            series = pd.Series(290.0 + np.arange(len(hours)) % 5, index=hours)
            if station_id == "D":
                series = series[::3]
            if station_id == "E":
                series.iloc[5] = -9999.0
                series.iloc[9] = 999.0
            for variable in variables:
                rows.append(
                    pd.DataFrame(
                        {
                            "station_id": station_id,
                            "variable": variable,
                            "time": series.index,
                            "value": series.values,
                        }
                    )
                )
        return (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(columns=["station_id", "variable", "time", "value"])
        )


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def loaded(provider, tmp_path):
    return load_stations(provider, (40, 50, 0, 10), START, END, cache_dir=tmp_path / "cache")


def test_only_active_and_complete_stations_are_kept(loaded):
    ds, report = loaded
    assert list(ds.station.values) == ["A", "B", "E"]
    reasons = report["reason"].to_dict()
    assert reasons["C"] == "not active over the window"
    assert reasons["G"] == "not active over the window"
    assert reasons["F"] == "unknown start date"
    assert reasons["D"].startswith("completeness")


def test_dataset_shape_units_and_metadata(loaded):
    ds, _ = loaded
    assert ds.t2m.dims == ("time", "station") and ds.t2m.shape == (72, 3)
    assert ds.t2m.attrs["units"] == "K"
    assert ds.time.dtype == np.dtype("datetime64[ns]")
    assert ds.time.values[0] == np.datetime64("2023-08-05T00:00")
    assert float(ds.elevation.sel(station="B")) == 2000.0
    assert ds.attrs["license"] == "test licence"


def test_sentinel_values_become_missing(loaded):
    ds, report = loaded
    values = ds.t2m.sel(station="E").values
    assert np.isnan(values).sum() == 2
    assert report.loc["E", "n_out_of_range"] == 2
    assert report.loc["E", "completeness"] == pytest.approx(70 / 72)


def test_repeated_calls_use_the_cache(provider, tmp_path):
    first, _ = load_stations(provider, (40, 50, 0, 10), START, END, cache_dir=tmp_path / "cache")
    calls = len(provider.fetch_calls)
    second, _ = load_stations(provider, (40, 50, 0, 10), START, END, cache_dir=tmp_path / "cache")
    assert len(provider.fetch_calls) == calls
    assert np.allclose(first.t2m.values, second.t2m.values, equal_nan=True)


def test_a_new_station_is_fetched_alone(provider, tmp_path):
    load_stations(provider, (40, 50, 0, 10), START, END, cache_dir=tmp_path / "cache")
    provider.table.loc[len(provider.table)] = ["H", "new", 45.7, 4.7, 310.0, pd.Timestamp("2001-01-01"), pd.NaT]
    ds, _ = load_stations(provider, (40, 50, 0, 10), START, END, cache_dir=tmp_path / "cache")
    assert provider.fetch_calls[-1] == ["H"]
    assert "H" in ds.station.values


def test_a_narrower_window_is_served_from_a_wider_cache(provider, tmp_path):
    load_stations(provider, (40, 50, 0, 10), START, END, cache_dir=tmp_path / "cache")
    calls = len(provider.fetch_calls)
    ds, _ = load_stations(
        provider,
        (40, 50, 0, 10),
        pd.Timestamp("2023-08-06"),
        pd.Timestamp("2023-08-06 23:00"),
        cache_dir=tmp_path / "cache",
    )
    assert list(provider.fetch_calls[calls:]) == [["G"]]
    assert ds.sizes["time"] == 24


def test_a_wider_window_triggers_a_fetch(provider, tmp_path):
    load_stations(provider, (40, 50, 0, 10), START, END, cache_dir=tmp_path / "cache")
    calls = len(provider.fetch_calls)
    load_stations(provider, (40, 50, 0, 10), START, pd.Timestamp("2023-08-09 23:00"), cache_dir=tmp_path / "cache")
    assert len(provider.fetch_calls) == calls + 1


def test_unknown_start_and_overlap_options(provider, tmp_path):
    ds, report = load_stations(
        provider, (40, 50, 0, 10), START, END, cache_dir=tmp_path / "cache", include_unknown_start=True
    )
    assert "F" in ds.station.values
    overlap, _ = load_stations(provider, (40, 50, 0, 10), START, END, cache_dir=tmp_path / "cache", active="overlap")
    assert "G" in overlap.station.values


def test_region_filter_restricts_the_station_list(provider, tmp_path):
    ds, _ = load_stations(provider, (45.4, 45.6, 6, 7), START, END, cache_dir=tmp_path / "cache")
    assert list(ds.station.values) == ["B"]


def test_holdout_is_deterministic_and_stable_under_subsetting():
    ids = [f"{i:08d}" for i in range(5000)]
    first, second = holdout_mask(ids, 0.2), holdout_mask(ids, 0.2)
    assert np.array_equal(first, second)
    assert abs(first.mean() - 0.2) < 0.02
    assert np.array_equal(holdout_mask(ids[::2], 0.2), first[::2])


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"variables": ["nonexistent"]}, KeyError),
        ({"variables": ["tx1h"]}, ValueError),
        ({"active": "sometimes"}, ValueError),
    ],
)
def test_invalid_arguments_raise(provider, tmp_path, kwargs, error):
    with pytest.raises(error):
        load_stations(provider, (40, 50, 0, 10), START, END, cache_dir=tmp_path / "cache", **kwargs)


def test_timezone_aware_timestamps_raise(provider, tmp_path):
    with pytest.raises(ValueError, match="naive UTC"):
        load_stations(
            provider, (40, 50, 0, 10), pd.Timestamp("2023-08-05", tz="UTC"), END, cache_dir=tmp_path / "cache"
        )


def test_custom_quality_thresholds_are_applied(provider, tmp_path):
    ds, _ = load_stations(
        provider,
        (40, 50, 0, 10),
        START,
        END,
        cache_dir=tmp_path / "cache",
        qc=QCConfig(min_completeness=0.3),
    )
    assert "D" in ds.station.values


def published_two_periods(folder):
    """Write a published station set covering two separate months."""
    import json

    meta = pd.DataFrame(
        {
            "station_id": ["a", "b", "c"],
            "name": ["A", "B", "C"],
            "latitude": [45.0, 46.0, 47.0],
            "longitude": [1.0, 2.0, 3.0],
            "elevation_m": [10.0, 200.0, 900.0],
            "holdout": [False, True, False],
        }
    )
    first = pd.date_range("2020-08-01", "2020-08-31", freq="1h")
    second = pd.date_range("2025-08-01", "2025-08-31", freq="1h")
    rows = [
        pd.DataFrame({"station_id": s, "time": t, "t2m": 290.0})
        for s, t in (("a", first), ("b", first), ("b", second), ("c", second))
    ]
    folder.mkdir()
    meta.to_parquet(folder / "metadata.parquet", index=False)
    pd.concat(rows).to_parquet(folder / "observations.parquet", index=False)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "start": "2020-08-01 00:00:00",
                "end": "2025-08-31 00:00:00",
                "periods": [
                    {"start": "2020-08-01 00:00:00", "end": "2020-08-31 00:00:00"},
                    {"start": "2025-08-01 00:00:00", "end": "2025-08-31 00:00:00"},
                ],
            }
        )
    )
    return folder


def test_one_period_of_a_merged_set_holds_only_its_hours_and_stations(tmp_path):
    from downscale.stations import load_published_stations

    folder = published_two_periods(tmp_path / "stations")
    august_2025 = load_published_stations(folder, start="2025-08-01", end="2025-08-31")
    assert list(august_2025.station.values) == ["b", "c"]
    assert august_2025.sizes["time"] == 721
    everything = load_published_stations(folder)
    assert everything.sizes["time"] == 2 * 721 and everything.sizes["station"] == 3
    with pytest.raises(ValueError, match="no observation"):
        load_published_stations(folder, start="2022-01-01", end="2022-01-31")


def test_a_merged_set_covers_each_period_but_not_the_gap(tmp_path):
    from downscale.fetch import _covers, _manifest

    manifest = _manifest(published_two_periods(tmp_path / "stations"))
    assert _covers(manifest, pd.Timestamp("2025-08-03"), pd.Timestamp("2025-08-20"))
    assert not _covers(manifest, pd.Timestamp("2022-08-03"), pd.Timestamp("2022-08-20"))
    assert not _covers(manifest, pd.Timestamp("2020-08-30"), pd.Timestamp("2025-08-02"))


def test_loading_retries_a_timeout_and_then_gives_up():
    from downscale import fetch

    class Flaky:
        calls = 0

        def load(self):
            Flaky.calls += 1
            if Flaky.calls < 2:
                raise TimeoutError("read timed out")
            return "loaded"

    import time

    sleep, time.sleep = time.sleep, lambda s: None
    try:
        assert fetch.load_with_retries(Flaky(), 3) == "loaded"

        class Broken:
            def load(self):
                raise TimeoutError("down")

        with pytest.raises(TimeoutError):
            fetch.load_with_retries(Broken(), 2)

        class Wrong:
            def load(self):
                raise KeyError("not a network error")

        with pytest.raises(KeyError):
            fetch.load_with_retries(Wrong(), 3)
    finally:
        time.sleep = sleep
