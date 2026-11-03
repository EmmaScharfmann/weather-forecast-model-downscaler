"""AIFS-ENS forecasts from ECMWF's open data, readable without an account.

Every run since mid-2025 is mirrored on Google Cloud (``gs://ecmwf-open-data``, also on AWS as
``s3://ecmwf-forecasts``, which often answers 503) as one GRIB2 file per lead time and kind
(``cf`` for the control member, ``pf`` for the 50 perturbed ones), each with a JSON-lines index giving
the byte range of every field. Only the 2 m temperature messages of the members asked for are read, and
each is cropped to the region as soon as it is decoded, so one ~600 kB global message is read per field.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr

from .regions import Region, crop_indices, to_signed_lon

logger = logging.getLogger(__name__)

OPEN_DATA_URL = "https://storage.googleapis.com/ecmwf-open-data"
PARAMS = {"t2m": "2t"}


def _get(url: str, byte_range: tuple[int, int] | None = None, retries: int = 6) -> bytes:
    """Read a URL, retrying the 503 answers S3 gives under load.

    :param url: URL to read.
    :type url: str
    :param byte_range: ``(offset, length)`` to read, ``None`` for the whole object.
    :type byte_range: tuple[int, int] | None
    :param retries: Attempts before giving up.
    :type retries: int
    :returns: The bytes.
    :rtype: bytes
    """
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[0] + byte_range[1] - 1}"} if byte_range else {}
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == retries:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise AssertionError("unreachable")


def file_url(init, lead_hours: int, kind: str, base: str = OPEN_DATA_URL) -> str:
    """URL of one GRIB2 file, without its extension.

    :param init: Initialisation time.
    :type init: str | pandas.Timestamp
    :param lead_hours: Lead time [h].
    :type lead_hours: int
    :param kind: ``"cf"`` or ``"pf"``.
    :type kind: str
    :param base: Bucket URL.
    :type base: str
    :returns: The URL, to which ``.grib2`` or ``.index`` is appended.
    :rtype: str
    """
    t = pd.Timestamp(init)
    return f"{base}/{t:%Y%m%d}/{t:%H}z/aifs-ens/0p25/enfo/{t:%Y%m%d%H}0000-{lead_hours}h-enfo-{kind}"


def decode(message: bytes) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Decode one GRIB message on a regular latitude-longitude grid.

    :param message: The message.
    :type message: bytes
    :returns: The member number, latitudes, longitudes and values over ``(lat, lon)``.
    :rtype: tuple[int, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    import eccodes

    handle = eccodes.codes_new_from_message(message)
    try:
        number = int(eccodes.codes_get(handle, "number")) if eccodes.codes_is_defined(handle, "number") else 0
        ni, nj = eccodes.codes_get(handle, "Ni"), eccodes.codes_get(handle, "Nj")
        lat = np.linspace(
            eccodes.codes_get(handle, "latitudeOfFirstGridPointInDegrees"),
            eccodes.codes_get(handle, "latitudeOfLastGridPointInDegrees"), nj,
        )
        lon0 = eccodes.codes_get(handle, "longitudeOfFirstGridPointInDegrees")
        lon = lon0 + np.arange(ni) * eccodes.codes_get(handle, "iDirectionIncrementInDegrees")
        values = eccodes.codes_get_values(handle).reshape(nj, ni)
    finally:
        eccodes.codes_release(handle)
    return number, lat, np.mod(lon, 360.0), values


def fetch_open_data(
    init, region: Region, variable: str = "t2m", members: int | None = 10, max_lead_hours: int = 240,
    step_hours: int = 6, base: str = OPEN_DATA_URL, get=_get,
) -> xr.DataArray:
    """Read one initialisation over a region, laid out as :func:`downscale.sources.fetch_forecast` does.

    Member 0 is the control forecast and members 1 onwards the perturbed ones, in order of their number.

    :param init: Initialisation time.
    :type init: str | pandas.Timestamp
    :param region: ``(lat_min, lat_max, lon_min, lon_max)``, longitudes in -180..180.
    :type region: tuple[float, float, float, float]
    :param variable: Short name, a key of :data:`PARAMS`.
    :type variable: str
    :param members: Members to keep, control included; ``None`` keeps all 51.
    :type members: int | None
    :param max_lead_hours: Last lead time [h].
    :type max_lead_hours: int
    :param step_hours: Spacing of the lead times kept [h]; the first is one step after initialisation.
    :type step_hours: int
    :param base: Bucket URL.
    :type base: str
    :param get: Callable reading a URL and an optional byte range, replaceable for testing.
    :type get: typing.Callable
    :returns: Values over ``(time, lead_time, ensemble, lat, lon)`` in kelvin, longitudes in -180..180.
    :rtype: xarray.DataArray
    :raises KeyError: If a member's field is missing from a file.
    """
    param = PARAMS[variable]
    perturbed = None if members is None else members - 1
    leads = list(range(step_hours, max_lead_hours + 1, step_hours))
    crop = coords = None
    fields = []
    for lead in leads:
        by_number = {}
        for kind in ("cf", "pf"):
            if kind == "pf" and perturbed == 0:
                continue
            url = file_url(init, lead, kind, base)
            index = [json.loads(line) for line in get(url + ".index").decode().splitlines() if line.strip()]
            records = sorted(
                (r for r in index if r.get("param") == param), key=lambda r: int(r.get("number", 0))
            )
            if kind == "pf" and perturbed is not None:
                records = records[:perturbed]
            for record in records:
                number, lat, lon, values = decode(get(url + ".grib2", (record["_offset"], record["_length"])))
                if crop is None:
                    crop = crop_indices(region, lat, lon)
                    coords = lat[crop[0]], to_signed_lon(lon[crop[1]])
                by_number[number] = values[np.ix_(*crop)].astype("f4")
        fields.append(np.stack([by_number[n] for n in sorted(by_number)]))
        logger.info("%s +%dh: %d members", pd.Timestamp(init), lead, len(by_number))
    lead_time = (np.array(leads) * np.timedelta64(1, "h")).astype("timedelta64[ns]")
    return xr.DataArray(
        np.stack(fields)[None],
        dims=("time", "lead_time", "ensemble", "lat", "lon"),
        coords={
            "time": [np.datetime64(pd.Timestamp(init), "ns")], "lead_time": lead_time,
            "ensemble": np.arange(fields[0].shape[0]), "lat": coords[0], "lon": coords[1],
        },
        name=variable,
        attrs={"units": "K", "source": "ECMWF open data, AIFS-ENS"},
    )
