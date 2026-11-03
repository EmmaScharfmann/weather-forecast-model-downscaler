"""Coarse fields read from Earthmover: ERA5 to train on, AIFS-ENS forecasts to downscale.

Both are opened lazily, so nothing is read until a time and a region are selected.
The forecasts can also come from a Zarr store with the same layout, local or on the
Hugging Face Hub, which is how forecasts older than the Earthmover archive are served.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .constants import ZERO_C
from .fetch import load_with_retries
from .regions import Region, crop_indices, to_signed_lon

logger = logging.getLogger(__name__)

ERA5_REPO = "earthmover-public/era5"
#: Google's public copy of ERA5 (ARCO), hourly from 1940, readable without an account.
ERA5_ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
#: Earthmover marketplace listing of AIFS-ENS 1.0; subscribing creates a repository under your organisation.
AIFS_ENS_LISTING = "https://app.earthmover.io/marketplace/6a10729e8a69e6bcaeae0f90"
#: Forecasts this long after initialisation without an ingestion record are assumed complete.
INGEST_MARGIN = pd.Timedelta(hours=12)

ERA5_ALIASES = {"t2m": ["t2m", "2m_temperature", "2t"]}
GEOPOTENTIAL_ALIASES = ["z_sfc", "z", "geopotential", "geopotential_at_surface"]

#: Forecast store names, in the order they are tried, mapped onto the names used here.
FORECAST_DIMS = {
    "init_time": "time", "ensemble_member": "ensemble", "member": "ensemble",
    "latitude": "lat", "longitude": "lon",
}
FORECAST_VARIABLES = {"t2m": ["t2m", "temperature_2m", "2t"]}


def _first(names: Sequence[str], candidates: Sequence[str]) -> str:
    """Return the first candidate present in a list of names.

    :param names: Names available.
    :type names: collections.abc.Sequence[str]
    :param candidates: Names to try, in order.
    :type candidates: collections.abc.Sequence[str]
    :returns: The first match.
    :rtype: str
    :raises KeyError: If no candidate is available.
    """
    for name in candidates:
        if name in names:
            return name
    raise KeyError(f"none of {list(candidates)} found among {sorted(names)}")


def open_arraylake(repo: str, group: str | None = None) -> xr.Dataset:
    """Open an Arraylake repository read-only.

    Needs an Earthmover account: run ``arraylake auth login`` or set ``ARRAYLAKE_TOKEN``.

    :param repo: Repository name, ``organisation/name``.
    :type repo: str
    :param group: Zarr group inside the repository.
    :type group: str | None
    :returns: The dataset, not loaded.
    :rtype: xarray.Dataset
    """
    from arraylake import Client

    session = Client().get_repo(repo).readonly_session(branch="main")
    logger.info("opening Arraylake repository %s%s", repo, f" ({group})" if group else "")
    return xr.open_zarr(session.store, group=group, consolidated=False, chunks=None)


def open_arco_era5(url: str = ERA5_ARCO, variables: Sequence[str] = ("2m_temperature",)) -> xr.Dataset:
    """Open Google's public ERA5 store, keeping a few single-level fields and the surface geopotential.

    :param url: Store URL.
    :type url: str
    :param variables: Fields to keep.
    :type variables: collections.abc.Sequence[str]
    :returns: The dataset, not loaded.
    :rtype: xarray.Dataset
    """
    logger.info("opening ERA5 from %s", url)
    # Through gcsfs, which backs off when the bucket answers 429 to the many one-hour chunks of a month.
    options = {"token": "anon"} if url.startswith("gs://") else None
    ds = xr.open_zarr(url, consolidated=True, chunks=None, storage_options=options)
    # The surface geopotential is stored at every time but empty at early ones; any filled step will do.
    surface = ds["geopotential_at_surface"].sel(time="2000-01-01T00:00", drop=True)
    return ds[list(variables)].assign_coords(geopotential_at_surface=surface)


class ERA5:
    """ERA5 single-level fields from Earthmover, the coarse input during training.

    Calling it with ``(times, variables)`` returns ``(time, variable, lat, lon)`` with
    descending latitudes and longitudes in 0..360.
    """

    def __init__(
        self, dataset: xr.Dataset | None = None, repo: str = ERA5_REPO, group: str = "single/spatial",
        box: Region | None = None,
    ):
        """
        :param dataset: Already open single-level dataset; ``None`` opens ``repo``.
        :type dataset: xarray.Dataset | None
        :param repo: Arraylake repository holding ERA5.
        :type repo: str
        :param group: Group holding the single-level fields. ``"single/spatial"`` stores one global
            field per chunk, suited to many places at few times; ``"single/temporal"`` stores long
            series over small tiles, suited to one region over months or years.
        :type group: str
        :param box: Region to keep, ``(lat_min, lat_max, lon_min, lon_max)``; ``None`` keeps the globe.
            Cropping before reading means only the tiles of the region are downloaded.
        :type box: tuple[float, float, float, float] | None
        """
        ds = dataset if dataset is not None else open_arraylake(repo, group)
        ds = ds.rename({k: v for k, v in {"valid_time": "time", "latitude": "lat", "longitude": "lon"}.items() if k in ds.dims})
        if box is not None:
            lat_idx, lon_idx = crop_indices(box, ds["lat"].values, ds["lon"].values)
            ds = ds.isel(lat=lat_idx, lon=lon_idx)
        self.ds = ds
        self._flip_lat = bool(self.ds["lat"].values[0] < self.ds["lat"].values[-1])
        self._shift_lon = bool(self.ds["lon"].values.min() < 0)

    def _fix_grid(self, da: xr.DataArray) -> xr.DataArray:
        """Put an already small selection on descending latitudes and 0..360 longitudes.

        :param da: Selection with ``lat`` and ``lon``.
        :type da: xarray.DataArray
        :returns: The reoriented selection.
        :rtype: xarray.DataArray
        """
        if self._flip_lat:
            da = da.isel(lat=slice(None, None, -1))
        if self._shift_lon:
            da = da.assign_coords(lon=da["lon"] % 360).sortby("lon")
        return da

    def __call__(self, times, variables) -> xr.DataArray:
        """Read fields at some times.

        :param times: Times to read.
        :type times: collections.abc.Sequence
        :param variables: Short names, e.g. ``["t2m"]``.
        :type variables: collections.abc.Sequence[str]
        :returns: Values over ``(time, variable, lat, lon)``.
        :rtype: xarray.DataArray
        """
        times = [pd.Timestamp(t) for t in ([times] if isinstance(times, (str, pd.Timestamp)) else times)]
        variables = [variables] if isinstance(variables, str) else list(variables)
        names = list(self.ds.data_vars) + list(self.ds.coords)
        arrays = []
        for v in variables:
            da = self.ds[_first(names, ERA5_ALIASES.get(v, [v]))].sel(time=times).load()
            da = self._fix_grid(da).drop_vars([c for c in da.coords if c not in ("time", "lat", "lon")])
            arrays.append(da.expand_dims(variable=[v]))
        return xr.concat(arrays, dim="variable").transpose("time", "variable", "lat", "lon")

    def surface_geopotential(self) -> xr.DataArray:
        """The model's surface geopotential, stored as a coordinate of the single-level group.

        :returns: Geopotential [m^2 s^-2] over ``(lat, lon)``.
        :rtype: xarray.DataArray
        """
        da = self.ds[_first(list(self.ds.data_vars) + list(self.ds.coords), GEOPOTENTIAL_ALIASES)]
        for dim in ("time", "valid_time", "step", "number"):
            if dim in da.dims:
                da = da.isel({dim: 0}, drop=True)
        return self._fix_grid(da).load()


def _is_store(path: str, fs) -> bool:
    """Report whether a path is a Zarr store rather than a folder of stores.

    :param path: Path on the filesystem.
    :type path: str
    :param fs: fsspec filesystem.
    :type fs: fsspec.AbstractFileSystem
    :returns: ``True`` when the path holds Zarr metadata itself.
    :rtype: bool
    """
    return any(fs.exists(f"{path}/{name}") for name in ("zarr.json", ".zgroup", ".zmetadata"))


def _stores(source: str) -> list[str]:
    """List the Zarr stores a source refers to.

    :param source: A store, or a folder or Hub dataset holding one store per initialisation.
    :type source: str
    :returns: URLs or paths of the stores, sorted.
    :rtype: list[str]
    :raises FileNotFoundError: If no store is found.
    """
    import fsspec

    if source.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        fs, path, prefix = HfFileSystem(), source.removeprefix("hf://").rstrip("/"), "hf://"
    else:
        fs, path, prefix = fsspec.filesystem("file"), str(Path(source).resolve()), ""
    if _is_store(path, fs):
        return [prefix + path]
    found = sorted(p for p in fs.ls(path, detail=False) if p.rstrip("/").endswith(".zarr"))
    if not found:
        raise FileNotFoundError(f"no .zarr store found in {source}")
    return [prefix + p.rstrip("/") for p in found]


def open_forecasts(source: str) -> xr.Dataset:
    """Open an AIFS-ENS archive.

    :param source: An Arraylake repository (``organisation/name``), or a Zarr store with
        the same layout: local, or on the Hub as ``hf://datasets/owner/name[/path]``. A folder
        or dataset holding one store per initialisation, as Earth2Studio writes them, is
        opened as a single archive.
    :type source: str
    :returns: The archive with dimensions renamed to ``time`` (initialisation), ``lead_time``,
        ``ensemble``, ``lat`` and ``lon``, not loaded.
    :rtype: xarray.Dataset
    """
    if source.startswith("hf://") or Path(source).exists():
        stores = _stores(source)
        logger.info("opening %d forecast store(s) from %s", len(stores), source)
        if len(stores) == 1:
            ds = xr.open_zarr(stores[0], chunks=None)
        else:
            ds = xr.concat([xr.open_zarr(s, chunks={}) for s in stores], dim="time")
    else:
        ds = open_arraylake(source)
    return ds.rename({k: v for k, v in FORECAST_DIMS.items() if k in ds.dims})


def available_inits(ds: xr.Dataset, now: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """Initialisations whose forecast is complete.

    The Earthmover store records how much of each forecast has been ingested, but
    sometimes leaves it empty; an initialisation older than :data:`INGEST_MARGIN`
    then counts as complete. Stores without that record are complete by construction.

    :param ds: Archive from :func:`open_forecasts`.
    :type ds: xarray.Dataset
    :param now: Current time, for testing; ``None`` reads the clock.
    :type now: pandas.Timestamp | None
    :returns: Complete initialisations, oldest first.
    :rtype: pandas.DatetimeIndex
    """
    inits = pd.DatetimeIndex(ds["time"].values)
    if "ingested_forecast_length" not in ds.coords and "ingested_forecast_length" not in ds:
        return inits
    now = now if now is not None else pd.Timestamp.now("UTC").tz_localize(None)
    ingested = ds["ingested_forecast_length"].values
    complete = np.where(
        np.isnat(ingested), inits <= now - INGEST_MARGIN, ingested >= ds["expected_forecast_length"].values
    )
    return inits[complete]


def forecast_box(region: Region, margin_deg: float) -> Region:
    """Widen a region so interpolation onto a grid inside it never runs off the edge.

    :param region: ``(lat_min, lat_max, lon_min, lon_max)``.
    :type region: tuple[float, float, float, float]
    :param margin_deg: Margin added on every side [degrees].
    :type margin_deg: float
    :returns: The widened box.
    :rtype: tuple[float, float, float, float]
    """
    lat0, lat1, lon0, lon1 = region
    return lat0 - margin_deg, lat1 + margin_deg, lon0 - margin_deg, lon1 + margin_deg


def fetch_forecast(
    ds: xr.Dataset, init, region: Region, variable: str = "t2m", members: int | None = None,
    max_lead_hours: int | None = None,
) -> xr.DataArray:
    """Read one initialisation over a region.

    :param ds: Archive from :func:`open_forecasts`.
    :type ds: xarray.Dataset
    :param init: Initialisation time.
    :type init: str | pandas.Timestamp
    :param region: ``(lat_min, lat_max, lon_min, lon_max)`` to read, longitudes in -180..180.
    :type region: tuple[float, float, float, float]
    :param variable: Short name; only ``"t2m"`` is mapped onto the archive's own name.
    :type variable: str
    :param members: First members to keep; ``None`` keeps all.
    :type members: int | None
    :param max_lead_hours: Last lead time to keep; ``None`` keeps all.
    :type max_lead_hours: int | None
    :returns: Values over ``(time, lead_time, ensemble, lat, lon)`` with one initialisation,
        in kelvin for temperatures, with longitudes in -180..180.
    :rtype: xarray.DataArray
    :raises KeyError: If the initialisation is not in the archive.
    """
    init = pd.Timestamp(init)
    if np.datetime64(init, "ns") not in ds["time"].values.astype("datetime64[ns]"):
        raise KeyError(f"{init} is not in the forecast archive")
    name = _first(list(ds.data_vars), FORECAST_VARIABLES.get(variable, [variable]))
    da = ds[name].sel(time=[np.datetime64(init, "ns")])
    if members is not None:
        da = da.isel(ensemble=slice(0, members))
    if max_lead_hours is not None:
        da = da.sel(lead_time=da["lead_time"] <= np.timedelta64(max_lead_hours, "h"))
    lat_idx, lon_idx = crop_indices(region, da["lat"].values, da["lon"].values)
    da = da.isel(lat=lat_idx, lon=lon_idx)
    # Cropped west to east, so signed longitudes increase even across the prime meridian.
    da = da.assign_coords(lon=to_signed_lon(da["lon"].values))
    logger.info("reading %s %s: %s", name, init, dict(da.sizes))
    da = load_with_retries(da).astype("f4")
    units = da.attrs.get("units", "")
    if units in ("degree_Celsius", "degC", "°C", "C"):
        da = da + np.float32(ZERO_C)
        units = "K"
    da = da.drop_vars([c for c in da.coords if c not in ("time", "lead_time", "ensemble", "lat", "lon")])
    da = da.assign_coords(lead_time=da["lead_time"].values.astype("timedelta64[ns]"))
    da.attrs = {"units": units or "K"}
    return da.rename(variable).transpose("time", "lead_time", "ensemble", "lat", "lon")


def store_name(init) -> str:
    """File name of the store holding one initialisation.

    :param init: Initialisation time.
    :type init: str | pandas.Timestamp
    :returns: Name such as ``aifs-ens_20250801T0000.zarr``.
    :rtype: str
    """
    return f"aifs-ens_{pd.Timestamp(init):%Y%m%dT%H%M}.zarr"


def save_forecast(forecast: xr.DataArray, out_dir: str | Path) -> Path:
    """Write one initialisation where the downscaling and evaluation scripts look for it.

    A ``.done`` marker is written last, so an interrupted write is never mistaken for a
    complete forecast.

    :param forecast: Output of :func:`fetch_forecast`.
    :type forecast: xarray.DataArray
    :param out_dir: Directory of forecast stores.
    :type out_dir: str | pathlib.Path
    :returns: Path of the store.
    :rtype: pathlib.Path
    """
    out = Path(out_dir) / store_name(forecast["time"].values[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_dataset().chunk({"lead_time": 10, "ensemble": -1, "lat": -1, "lon": -1}).to_zarr(out, mode="w")
    Path(str(out) + ".done").touch()
    return out
