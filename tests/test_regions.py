"""Tests for region parsing and grid cropping."""

import numpy as np
import pytest

from downscaler.data.regions import crop_indices, parse_region, to_signed_lon


def test_parse_region_accepts_presets_strings_and_sequences():
    assert parse_region("france") == (40.0, 53.0, -7.0, 11.0)
    assert parse_region("40,50,0,10") == (40.0, 50.0, 0.0, 10.0)
    assert parse_region([40, 50, 0, 10]) == (40.0, 50.0, 0.0, 10.0)


def test_parse_region_rejects_unknown_names():
    with pytest.raises(ValueError, match="region must be"):
        parse_region("atlantis")


def test_crop_indices_orders_longitudes_west_to_east_across_the_meridian(coarse_grid):
    full_lat = 90 - 0.25 * np.arange(721)
    full_lon = 0.25 * np.arange(1440)
    lat_idx, lon_idx = crop_indices(parse_region("france"), full_lat, full_lon)
    lons = full_lon[lon_idx]
    assert lat_idx.size == 53 and lon_idx.size == 73
    assert lons[0] == 353.0 and lons[27] == 359.75 and lons[28] == 0.0 and lons[-1] == 11.0
    assert np.all(np.diff(full_lat[lat_idx]) < 0)


def test_crop_indices_keeps_every_point_inside_the_box():
    lat = np.linspace(50, 40, 60)
    lon = np.linspace(0, 20, 80)
    lat_idx, lon_idx = crop_indices((42, 48, 5, 15), lat, lon)
    assert lat[lat_idx].min() >= 42 and lat[lat_idx].max() <= 48
    assert ((lat >= 42) & (lat <= 48)).sum() == lat_idx.size


def test_crop_indices_rejects_an_empty_box():
    with pytest.raises(ValueError, match="selects no grid point"):
        crop_indices((80, 85, 0, 1), np.linspace(50, 40, 10), np.linspace(0, 20, 10))


def test_to_signed_lon_wraps_at_180():
    assert np.allclose(to_signed_lon([0, 179, 181, 359.75]), [0, 179, -179, -0.25])
