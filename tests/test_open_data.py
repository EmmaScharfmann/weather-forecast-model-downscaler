"""Tests for reading AIFS-ENS from ECMWF open data, against GRIB messages built in memory."""

import json

import numpy as np
import pandas as pd
import pytest

eccodes = pytest.importorskip("eccodes")

from downscale import open_data  # noqa: E402


def message(number: int, value: float) -> bytes:
    """A 2 m temperature field on a 1-degree global grid, warmer to the east.

    :param number: Ensemble member number.
    :type number: int
    :param value: Temperature at longitude zero [K].
    :type value: float
    :returns: The encoded GRIB2 message.
    :rtype: bytes
    """
    handle = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    for key, v in {
        "Ni": 360, "Nj": 181, "latitudeOfFirstGridPointInDegrees": 90.0, "latitudeOfLastGridPointInDegrees": -90.0,
        "longitudeOfFirstGridPointInDegrees": 0.0, "longitudeOfLastGridPointInDegrees": 359.0,
        "iDirectionIncrementInDegrees": 1.0, "jDirectionIncrementInDegrees": 1.0,
    }.items():
        eccodes.codes_set(handle, key, v)
    if number:
        eccodes.codes_set(handle, "productDefinitionTemplateNumber", 1)
        eccodes.codes_set(handle, "perturbationNumber", number)
    lon = np.arange(360.0)
    eccodes.codes_set_values(handle, (value + np.where(lon < 180, lon, lon - 360) / 10 + np.zeros((181, 1))).ravel())
    out = eccodes.codes_get_message(handle)
    eccodes.codes_release(handle)
    return out


class Bucket:
    """Serves one file per lead time and kind, with its index, from memory."""

    def __init__(self, leads, n_perturbed, model="aifs-ens"):
        self.files, self.reads = {}, []
        perturbed = list(range(n_perturbed, 0, -1))
        layout = {"aifs-ens": (("cf", [0]), ("pf", perturbed)), "ifs-ens": (("ef", perturbed + [0]),)}[model]
        for lead in leads:
            for kind, numbers in layout:
                blob, index = b"", []
                for number in numbers:
                    for param, value in (("2t", 280.0 + number + lead / 6), ("10u", 0.0)):
                        encoded = message(number, value)
                        record = {"param": param, "_offset": len(blob), "_length": len(encoded)}
                        index.append({**record, "number": str(number)} if number else record)
                        blob += encoded
                url = open_data.file_url("2025-08-01T00", lead, kind, "mem://", model)
                self.files[url + ".grib2"] = blob
                self.files[url + ".index"] = "\n".join(json.dumps(r) for r in index).encode()

    def __call__(self, url, byte_range=None):
        self.reads.append(url)
        data = self.files[url]
        return data if byte_range is None else data[byte_range[0]:byte_range[0] + byte_range[1]]


@pytest.mark.parametrize("model", ["aifs-ens", "ifs-ens"])
def test_members_and_lead_times_are_read_and_cropped_to_the_region(model):
    bucket = Bucket([6, 12], n_perturbed=4, model=model)
    da = open_data.fetch_open_data(
        "2025-08-01T00", (40.0, 50.0, -5.0, 10.0), members=3, max_lead_hours=12, base="mem://", get=bucket,
        model=model,
    )
    assert da.dims == ("time", "lead_time", "ensemble", "lat", "lon") and da.shape == (1, 2, 3, 11, 16)
    assert list(da.lead_time.values / np.timedelta64(1, "h")) == [6, 12]
    assert da.time.values[0] == np.datetime64(pd.Timestamp("2025-08-01"), "ns")
    assert np.all(np.diff(da.lon.values) > 0) and da.lon.values[0] == -5.0
    at_zero = da.sel(lon=0.0).isel(lat=0, time=0)
    assert np.allclose(at_zero.values, [[281, 282, 283], [282, 283, 284]])  # control, then members 1 and 2
    assert float(da.sel(lon=-5.0).isel(lat=0, time=0, lead_time=0, ensemble=0)) == pytest.approx(280.5)
    grib_reads = [u for u in bucket.reads if u.endswith(".grib2")]
    assert len(grib_reads) == 2 * 3  # only the 2 m temperature of the members asked for
