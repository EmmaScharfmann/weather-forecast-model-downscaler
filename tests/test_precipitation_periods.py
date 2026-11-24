"""Training years and gauge-chosen validation/test days in the annual runner."""

import json

import pandas as pd
import pytest

from scripts.precipitation_year import periods


def config(tmp_path, **annual):
    return {"annual": {"year": 2019, "validation_year": 2020, "test_year": 2022, **annual}}


def test_single_year_keeps_previous_behaviour(tmp_path):
    result = periods(config(tmp_path))
    assert sum(role == "train" for role, *_ in result) == 12
    assert [s for role, s, _ in result if role == "test"][0] == pd.Timestamp("2022-01-15")


def test_several_years_and_listed_days(tmp_path):
    (tmp_path / "val.json").write_text(json.dumps({"days": ["2020-02-03", "2020-07-09"]}))
    (tmp_path / "test.json").write_text(json.dumps({"days": ["2023-10-30"]}))
    result = periods(
        config(
            tmp_path,
            years=[2016, 2017, 2018, 2019],
            validation_days=str(tmp_path / "val.json"),
            test_days=str(tmp_path / "test.json"),
        )
    )
    train = [s for role, s, _ in result if role == "train"]
    assert len(train) == 48 and train[0] == pd.Timestamp("2016-01-01")
    assert [(s, e) for role, s, e in result if role == "validation"] == [
        (pd.Timestamp("2020-02-03"), pd.Timestamp("2020-02-04")),
        (pd.Timestamp("2020-07-09"), pd.Timestamp("2020-07-10")),
    ]
    assert [s for role, s, _ in result if role == "test"] == [pd.Timestamp("2023-10-30")]


def test_listed_days_cannot_overlap_training(tmp_path):
    (tmp_path / "val.json").write_text(json.dumps({"days": ["2019-06-01"]}))
    with pytest.raises(ValueError, match="overlap"):
        periods(config(tmp_path, validation_days=str(tmp_path / "val.json")))


def test_feature_stack_matches_features_and_saves_memory():
    import numpy as np
    import xarray as xr

    from downscale.precipitation import FeatureStack, features

    rng = np.random.default_rng(0)
    n, h, w = 30, 6, 7
    predictors = rng.normal(0, 1, (n, 3, h, w)).astype("float32")
    predictors[:, 0] = rng.gamma(0.5, 1, (n, h, w))
    ds = xr.Dataset(
        {"predictors": (("time", "variable", "lat", "lon"), predictors)},
        coords={
            "time": pd.date_range("2019-01-01T01:00", periods=n, freq="h"),
            "variable": ["tp", "t2m", "msl"],
            "lat": np.linspace(44, 45, h),
            "lon": np.linspace(2, 3, w),
        },
    )
    static = rng.normal(0, 1, (8, h, w)).astype("float32")
    reference, mean, std = features(ds, static=static)
    stats = FeatureStack.statistics(ds, block=7)
    assert np.allclose(stats[0], mean, atol=1e-5) and np.allclose(stats[1], std, atol=1e-4)
    stack = FeatureStack(ds, mean, std, static, block=7)
    index = np.array([3, 4, 5, 20])
    assert stack.shape == reference.shape and len(stack) == n
    assert np.allclose(stack[index], reference[index], atol=1e-6)
    assert np.allclose(stack[2:9], reference[2:9], atol=1e-6)
    half = FeatureStack(ds, mean, std, static, "float16")
    assert half.data.nbytes * 2 == stack.data.nbytes
    assert np.allclose(half[index], reference[index], atol=5e-3)


def test_blocked_hourly_baseline_matches_single_pass():
    import numpy as np
    import xarray as xr

    from downscale.rain_temporal import hourly_baseline

    rng = np.random.default_rng(1)
    times = pd.DatetimeIndex(
        list(pd.date_range("2019-01-01T01:00", periods=6 * 23, freq="h"))
        + list(pd.date_range("2019-03-01T01:00", periods=6 * 5, freq="h"))
    )
    rain = (rng.gamma(0.4, 2, (len(times), 3, 4)) * (rng.random((len(times), 1, 1)) > 0.4)).astype("float32")
    ds = xr.Dataset(
        {"predictors": (("time", "variable", "lat", "lon"), rain[:, None])},
        coords={"time": times, "variable": ["tp"]},
        attrs={"forecast_interval_hours": 6},
    )
    whole = hourly_baseline(ds, block_windows=10_000)
    for block in (1, 2, 3, 7):
        assert np.allclose(hourly_baseline(ds, block_windows=block), whole, atol=1e-5)
    totals = whole[: 6 * 23].reshape(23, 6, 3, 4).sum(1)
    assert np.allclose(totals, rain[: 6 * 23].reshape(23, 6, 3, 4).sum(1), atol=1e-4)
