"""Tests for the hourly downscaler: the fields in a 6-hour window, the inputs it is given and its checkpoint."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is installed with requirements/downscaler.txt")

from downscale import downscaler as dsc  # noqa: E402
from downscale import hourly  # noqa: E402


@pytest.fixture
def hourly_stats(stats):
    """The spatial statistics plus the change over a window, which the hourly inputs need."""
    return {**stats, "tendency": {"mean": 0.0, "std": 2.0}}


def spatial_model(static_fields, stats, loss="mse"):
    """A small spatial model whose last layer is not zero, so its output depends on the input."""
    net = dsc.DownscalerNet(dsc.Config(loss=loss, base=8, depth=2, noise_channels=4 if loss == "crps" else 0))
    with torch.no_grad():
        for parameter in net.unet.out.parameters():
            parameter.normal_(0, 0.1)
    return dsc.Downscaler(net, stats, static_fields)


def hourly_model(static_fields, stats, loss="mse"):
    """A small hourly model with the same settings as :func:`spatial_model`."""
    cfg = hourly.HourlyConfig(loss=loss, base=8, depth=2, noise_channels=4 if loss == "crps" else 0)
    return hourly.HourlyDownscaler(hourly.HourlyNet(cfg), stats, static_fields)


def test_straight_line_runs_between_the_two_ends_of_the_window():
    c0, c1 = np.full((3, 4, 5), 280.0, "f4"), np.full((3, 4, 5), 292.0, "f4")
    line = hourly.temporal_interpolation(c0, c1, np.array([0, 3, 6]))
    assert np.allclose(line[:, 0, 0], [280.0, 286.0, 292.0])


def test_inputs_have_one_channel_per_name_and_the_baseline_is_the_line(static_fields, hourly_stats):
    height, width = static_fields["lat"].shape
    c0, c1 = np.full((2, height, width), 285.0, "f4"), np.full((2, height, width), 291.0, "f4")
    times = np.array(["2020-07-01T03", "2020-07-01T09"], "datetime64[ns]")
    static_norm = dsc.static_stack(static_fields, hourly_stats)
    inputs, baseline = hourly.make_hourly_inputs(c0, c1, np.array([3, 3]), times, static_norm, hourly_stats)
    assert inputs.shape == (2, len(hourly.HourlyConfig().channels), height, width)
    assert np.allclose(baseline, 288.0)
    assert np.allclose(inputs[:, -1], 0.0)  # position in the window: the middle


def test_it_starts_as_a_copy_of_the_spatial_model(tmp_path, static_fields, hourly_stats):
    height, width = static_fields["lat"].shape
    spatial = spatial_model(static_fields, hourly_stats)
    spatial.save(tmp_path / "spatial.pt")
    model = hourly_model(static_fields, hourly_stats)
    hourly.init_from_spatial(model.net, str(tmp_path / "spatial.pt"))
    rng = np.random.default_rng(0)
    c0 = (290 + rng.normal(size=(3, height, width))).astype("f4")
    c1 = c0 + 4.0  # the window changes, but the first hour of it is the coarse field at its start
    times = np.array(["2020-07-01T00"] * 3, "datetime64[ns]")
    expected = spatial.predict(c0, times, members=1)
    got = model.predict_hours(c0, c1, np.zeros(3), times, members=1)
    assert np.allclose(got, expected, atol=1e-4)


def test_hours_in_the_window_differ_once_the_new_channels_are_trained(static_fields, hourly_stats):
    height, width = static_fields["lat"].shape
    model = hourly_model(static_fields, hourly_stats)
    with torch.no_grad():
        model.net.unet.inc.net[0].weight.normal_(0, 0.1)
    c0, c1 = np.full((2, height, width), 285.0, "f4"), np.full((2, height, width), 295.0, "f4")
    times = np.array(["2020-07-01T00", "2020-07-01T03"], "datetime64[ns]")
    first, middle = model.predict_hours(c0, c1, np.array([0, 3]), times, members=1)[:, 0]
    assert not np.allclose(first, middle)


def test_a_member_keeps_its_noise_from_hour_to_hour(static_fields, hourly_stats):
    height, width = static_fields["lat"].shape
    model = hourly_model(static_fields, hourly_stats, loss="crps")
    with torch.no_grad():
        for parameter in model.net.unet.out.parameters():
            parameter.normal_(0, 0.1)
    c0, c1 = np.full((3, height, width), 288.0, "f4"), np.full((3, height, width), 288.0, "f4")
    times = np.array(["2020-07-01T00"] * 3, "datetime64[ns]")
    fine = model.predict_hours(c0, c1, np.array([2, 2, 2]), times, members=2, seed=1)
    assert fine.shape == (3, 2, height, width)
    assert np.allclose(fine[0], fine[1]) and np.allclose(fine[1], fine[2])  # same inputs, same noise per member
    assert not np.allclose(fine[0, 0], fine[0, 1])  # members differ from each other


def test_checkpoint_round_trip_and_the_spatial_checkpoint_is_refused(tmp_path, static_fields, hourly_stats):
    height, width = static_fields["lat"].shape
    model = hourly_model(static_fields, hourly_stats)
    model.save(tmp_path / "hourly.pt")
    restored = hourly.HourlyDownscaler.load(tmp_path / "hourly.pt", device="cpu")
    c0, c1 = np.full((2, height, width), 285.0, "f4"), np.full((2, height, width), 291.0, "f4")
    times = np.array(["2020-07-01T00", "2020-07-01T03"], "datetime64[ns]")
    offsets = np.array([0, 3])
    assert np.allclose(restored.predict_hours(c0, c1, offsets, times, members=1),
                       model.predict_hours(c0, c1, offsets, times, members=1), atol=1e-5)
    spatial_model(static_fields, hourly_stats).save(tmp_path / "spatial.pt")
    with pytest.raises(ValueError, match="channels"):
        hourly.HourlyDownscaler.load(tmp_path / "spatial.pt", device="cpu")
