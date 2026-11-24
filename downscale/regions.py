from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .constants import REGION_PRESETS


Region = tuple[float, float, float, float]


def parse_region(spec: str | Sequence[float]) -> Region:
    """Resolve a region specification to a bounding box.

    :param spec: A preset name, ``"lat_min,lat_max,lon_min,lon_max"``, or a four-element sequence.
    :type spec: str | collections.abc.Sequence[float]
    :returns: ``(lat_min, lat_max, lon_min, lon_max)`` with longitudes in -180..180.
    :rtype: tuple[float, float, float, float]
    :raises ValueError: If the string is neither a known preset nor four numbers.
    """
    if isinstance(spec, str):
        if spec in REGION_PRESETS:
            return REGION_PRESETS[spec]
        try:
            lat0, lat1, lon0, lon1 = (float(x) for x in spec.split(","))
        except ValueError as exc:
            raise ValueError(
                f"region must be one of {sorted(REGION_PRESETS)} or four comma-separated numbers "
                f"'lat_min,lat_max,lon_min,lon_max'; got {spec!r}"
            ) from exc
        return lat0, lat1, lon0, lon1
    lat0, lat1, lon0, lon1 = (float(x) for x in spec)
    return lat0, lat1, lon0, lon1


def crop_indices(region: Region, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays selecting a region from a regular latitude/longitude grid.

    :param region: Bounding box as returned by :func:`parse_region`.
    :type region: tuple[float, float, float, float]
    :param lat: Grid latitudes, in the grid's own order.
    :type lat: numpy.ndarray
    :param lon: Grid longitudes, 0..360 or signed.
    :type lon: numpy.ndarray
    :returns: Latitude indices in grid order and longitude indices ordered west to east.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    :raises ValueError: If the region selects no grid point.
    """
    lat_min, lat_max, lon_min, lon_max = region
    lat = np.asarray(lat)
    lat_idx = np.flatnonzero((lat >= min(lat_min, lat_max)) & (lat <= max(lat_min, lat_max)))

    lon360 = np.asarray(lon) % 360
    west, east = lon_min % 360, lon_max % 360
    if np.isclose(west, east):
        lon_idx = np.arange(lon360.size)
    elif west <= east:
        lon_idx = np.flatnonzero((lon360 >= west) & (lon360 <= east))
        lon_idx = lon_idx[np.argsort(lon360[lon_idx], kind="stable")]
    else:
        left = np.flatnonzero(lon360 >= west)
        right = np.flatnonzero(lon360 <= east)
        lon_idx = np.concatenate(
            [
                left[np.argsort(lon360[left], kind="stable")],
                right[np.argsort(lon360[right], kind="stable")],
            ]
        )

    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError(f"region {region} selects no grid point")
    return lat_idx, lon_idx


def to_signed_lon(lon: np.ndarray | Sequence[float]) -> np.ndarray:
    """Convert longitudes from 0..360 to -180..180.

    :param lon: Longitudes in any convention.
    :type lon: numpy.ndarray | collections.abc.Sequence[float]
    :returns: Longitudes in -180..180.
    :rtype: numpy.ndarray
    """
    return ((np.asarray(lon, dtype="f8") + 180.0) % 360.0) - 180.0
