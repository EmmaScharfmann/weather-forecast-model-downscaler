"""Fetch observations and reanalysis for any period, from the Hub or the source.

A published copy covers a fixed window. These helpers read that window from the
copy's manifest, use it when it contains the dates asked for, and otherwise fall
back to the original service. The caller names a period, not a source.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import xarray as xr


logger = logging.getLogger(__name__)

AUTO, HUB, SOURCE = "auto", "hub", "source"


def _manifest(path_or_repo: str | Path, revision: str | None = None) -> dict | None:
    """Read the manifest of a published dataset, local or on the Hub.

    :param path_or_repo: Local folder or Hub dataset repository id.
    :type path_or_repo: str | pathlib.Path
    :param revision: Revision to read when the dataset is on the Hub.
    :type revision: str | None
    :returns: The manifest, or ``None`` when it cannot be read.
    :rtype: dict | None
    """
    local = Path(path_or_repo) / "manifest.json"
    if local.exists():
        return json.loads(local.read_text())
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=str(path_or_repo), filename="manifest.json", repo_type="dataset", revision=revision
        )
        return json.loads(Path(downloaded).read_text())
    except Exception as exc:  # noqa: BLE001
        logger.info("could not read the manifest of %s (%s); treating it as unavailable", path_or_repo, exc)
        return None


def _covers(manifest: dict | None, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """Report whether a published dataset spans a period.

    :param manifest: Manifest of the published dataset.
    :type manifest: dict | None
    :param start: First time needed.
    :type start: pandas.Timestamp
    :param end: Last time needed.
    :type end: pandas.Timestamp
    :returns: ``True`` when the dataset covers the whole period.
    :rtype: bool
    """
    if not manifest:
        return False
    if "periods" in manifest:
        return any(pd.Timestamp(p["start"]) <= start and pd.Timestamp(p["end"]) >= end for p in manifest["periods"])
    if "start" in manifest and "end" in manifest:
        first, last = pd.Timestamp(manifest["start"]), pd.Timestamp(manifest["end"])
    elif "years" in manifest:
        first_year, last_year = str(manifest["years"]).split("-")
        first, last = pd.Timestamp(f"{first_year}-01-01"), pd.Timestamp(f"{last_year}-12-31T23:59")
    else:
        return False
    return bool(first <= start and last >= end)


def get_stations(
    start,
    end,
    region: str = "france",
    repo: str | Path | None = None,
    source: str = AUTO,
    cache_dir: str | Path = "outputs/cache/stations",
    provider=None,
    **kwargs,
) -> xr.Dataset:
    """Station observations for a period, from a published copy or the network.

    :param start: First time, naive UTC.
    :type start: str | pandas.Timestamp
    :param end: Last time, naive UTC.
    :type end: str | pandas.Timestamp
    :param region: Region name or bounding box, used when collecting from the network.
    :type region: str
    :param repo: Published dataset to try first; ``None`` goes straight to the network.
    :type repo: str | pathlib.Path | None
    :param source: ``"auto"`` prefers the published copy when it covers the period,
        ``"hub"`` requires it, ``"source"`` always collects from the network.
    :type source: str
    :param cache_dir: Where collected observations are cached.
    :type cache_dir: str | pathlib.Path
    :param provider: Station provider; ``None`` uses Météo-France.
    :type provider: downscale.stations.StationProvider | None
    :param kwargs: Passed through to :func:`~downscale.stations.load_stations`.
    :returns: Observations over ``(time, station)``, restricted to the period.
    :rtype: xarray.Dataset
    :raises ValueError: If ``source="hub"`` but the published copy cannot serve the period.
    """
    from .stations import MeteoFranceHourly, load_published_stations, load_stations

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if source not in (AUTO, HUB, SOURCE):
        raise ValueError(f"source must be one of {AUTO}, {HUB}, {SOURCE}")

    if repo is not None and source in (AUTO, HUB):
        if _covers(_manifest(repo), start, end):
            logger.info("reading stations for %s .. %s from %s", start, end, repo)
            return load_published_stations(repo, start=start, end=end)
        if source == HUB:
            raise ValueError(f"{repo} does not cover {start} .. {end}; use source='auto' to collect them")
        logger.info("%s does not cover %s .. %s; collecting from the network instead", repo, start, end)

    observations, _ = load_stations(provider or MeteoFranceHourly(), region, start, end, cache_dir=cache_dir, **kwargs)
    return observations


def get_target(
    start,
    end,
    repo: str | Path | None = None,
    source: str = AUTO,
    local: str | Path = "data/cerra",
    stream: bool = True,
    load: bool = False,
    retries: int = 3,
    **kwargs,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Reanalysis fields for a period, from a published copy or a local download.

    :param start: First time.
    :type start: str | pandas.Timestamp
    :param end: Last time.
    :type end: str | pandas.Timestamp
    :param repo: Published dataset to try first.
    :type repo: str | pathlib.Path | None
    :param source: ``"auto"`` prefers the published copy, ``"hub"`` requires it,
        ``"source"`` reads the local download.
    :type source: str
    :param local: Folder written by the download stage, used when the published copy will not serve.
    :type local: str | pathlib.Path
    :param stream: Read the published copy lazily rather than downloading it whole.
    :type stream: bool
    :param load: Read the period into memory before returning. Streaming reads chunks on demand,
        so a long computation makes many requests and any one can time out; loading makes them
        all at once, retried on network errors.
    :type load: bool
    :param retries: Attempts at loading before giving up, when ``load`` is set.
    :type retries: int
    :param kwargs: Passed through to :func:`~downscale.publish.load_published_target`.
    :returns: The time-varying field restricted to the period, and the static fields.
    :rtype: tuple[xarray.Dataset, xarray.Dataset]
    :raises ValueError: If ``source="hub"`` but the published copy cannot serve the period.
    :raises FileNotFoundError: If the local download is needed but absent.
    """
    from .publish import load_published_target

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if repo is not None and source in (AUTO, HUB):
        if _covers(_manifest(repo), start, end):
            logger.info("reading the reanalysis for %s .. %s from %s", start, end, repo)
            target, static = load_published_target(repo, stream=stream, **kwargs)
            target = target.sel(time=slice(start, end))
            return (
                (load_with_retries(target, retries), load_with_retries(static, retries)) if load else (target, static)
            )
        if source == HUB:
            raise ValueError(f"{repo} does not cover {start} .. {end}; download those dates instead")
        logger.info("%s does not cover %s .. %s; reading the local download instead", repo, start, end)

    folder = Path(local)
    if not (folder / "target.zarr").exists():
        raise FileNotFoundError(
            f"{folder} holds no downloaded reanalysis. Run scripts/download_cerra.py for these dates, "
            "which needs a Copernicus account and acceptance of the dataset licence."
        )
    target, static = load_published_target(folder)
    return target.sel(time=slice(start, end)), static


def load_with_retries(ds, retries: int = 5):
    """Read a lazily opened dataset into memory, retrying network errors.

    :param ds: Dataset or data array to load.
    :type ds: xarray.Dataset | xarray.DataArray
    :param retries: Attempts before giving up.
    :type retries: int
    :returns: The loaded object.
    :rtype: xarray.Dataset | xarray.DataArray
    """
    import time

    transient = ("Timeout", "Connect", "Proxy", "RemoteProtocol", "Network", "ReadError", "StorageError")
    for attempt in range(1, retries + 1):
        try:
            return ds.load()
        except Exception as exc:  # noqa: BLE001
            # Network errors come from several libraries (httpx, aiohttp, icechunk), so match them by name.
            names = [type(e).__name__ for e in (exc, exc.__cause__, exc.__context__) if e is not None]
            if not isinstance(exc, (OSError, TimeoutError)) and not any(t in n for t in transient for n in names):
                raise
            error = exc
        logger.warning("loading failed (%s), attempt %d of %d", type(error).__name__, attempt, retries)
        time.sleep(10 * attempt)
    raise error
