"""Station observations used as independent verification truth.

A provider supplies a station list and long-format values for one network;
:func:`load_stations` is provider independent and handles caching, quality
control and the train/holdout split. Supporting a new country means writing one
class implementing :class:`StationProvider`.

Times are naive UTC, temperatures are in kelvin and station longitudes are
signed, whereas gridded forecasts use 0..360.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import xarray as xr

from downscaler.constants import STATION_VARIABLES
from downscaler.data.regions import Region, parse_region


logger = logging.getLogger(__name__)

_EMPTY = pd.DataFrame(columns=["station_id", "variable", "time", "value"])


class StationUnavailable(Exception):
    """Raised when a station holds no data for the requested period."""


def _period_arguments(request_class, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, str]:
    """Name the period arguments the installed client expects.

    ``wetterdienst`` renamed ``start_date`` and ``end_date`` to ``start`` and
    ``end``; the installed version decides which pair is accepted.

    :param request_class: Request class whose signature is inspected.
    :type request_class: type
    :param start: First time.
    :type start: pandas.Timestamp
    :param end: Last time.
    :type end: pandas.Timestamp
    :returns: Keyword arguments naming the period.
    :rtype: dict[str, str]
    :raises TypeError: If neither pair of names is accepted.
    """
    import inspect

    accepted = set(inspect.signature(request_class.__init__).parameters)
    for first, last in (("start", "end"), ("start_date", "end_date")):
        if first in accepted and last in accepted:
            return {first: start.isoformat(), last: end.isoformat()}
    raise TypeError(
        f"{request_class.__name__} accepts neither start/end nor start_date/end_date; "
        f"its arguments are {sorted(accepted)}"
    )


class StationProvider(Protocol):
    """Interface a national observation network must implement."""

    name: str
    license: str

    def list_stations(
        self, region: Region, start: pd.Timestamp, end: pd.Timestamp, variables: Sequence[str]
    ) -> pd.DataFrame:
        """List candidate stations within a region.

        :param region: Bounding box.
        :type region: tuple[float, float, float, float]
        :param start: Window start, naive UTC.
        :type start: pandas.Timestamp
        :param end: Window end, naive UTC.
        :type end: pandas.Timestamp
        :param variables: Canonical variable names.
        :type variables: collections.abc.Sequence[str]
        :returns: Frame with ``station_id``, ``name``, ``latitude``, ``longitude``,
            ``elevation``, ``start_date`` and ``end_date``.
        :rtype: pandas.DataFrame
        """

    def fetch(
        self, station_ids: Sequence[str], start: pd.Timestamp, end: pd.Timestamp, variables: Sequence[str]
    ) -> pd.DataFrame:
        """Fetch observations for a set of stations.

        :param station_ids: Station identifiers.
        :type station_ids: collections.abc.Sequence[str]
        :param start: Window start, naive UTC.
        :type start: pandas.Timestamp
        :param end: Window end, naive UTC.
        :type end: pandas.Timestamp
        :param variables: Canonical variable names.
        :type variables: collections.abc.Sequence[str]
        :returns: Long frame with ``station_id``, ``variable``, ``time`` and ``value``.
        :rtype: pandas.DataFrame
        """


class MeteoFranceHourly:
    """Météo-France hourly observations, accessed through ``wetterdienst``.

    Published under the Etalab Open Licence 2.0.
    """

    name = "meteofrance_hourly"
    display_name = "Météo-France"
    license = "Etalab Open Licence 2.0"
    source_url = "https://meteo.data.gouv.fr/"
    licence_url = "https://www.etalab.gouv.fr/licence-ouverte-open-licence/"
    attribution = "Source: Météo-France, données ouvertes, Licence Ouverte / Open Licence 2.0"
    access_tool = "wetterdienst"
    access_note = (
        "Météo-France publishes its observations as per-department CSV archives on "
        "meteo.data.gouv.fr. They are read here through the wetterdienst library, which "
        "resolves the station list, downloads the relevant archives and returns tidy values; "
        "no API key is needed."
    )
    dataset_note = (
        "hourly 'core' observations: instantaneous 2 m air temperature on the hour, parameter temperature_air_mean_2m"
    )

    _PARAMS = {
        "t2m": ("hourly", "core", "temperature_air_mean_2m", 273.15),
        "tx1h": ("hourly", "core", "temperature_air_max_2m", 273.15),
        "tn1h": ("hourly", "core", "temperature_air_min_2m", 273.15),
    }

    def __init__(self, chunk_size: int = 25, cache_disable: bool = False):
        """
        :param chunk_size: Stations requested per call.
        :type chunk_size: int
        :param cache_disable: Bypass the ``wetterdienst`` file cache entirely.
        :type cache_disable: bool
        """
        self.chunk_size = chunk_size
        self.cache_disable = cache_disable

    def _request(
        self, variables: Sequence[str], start: pd.Timestamp, end: pd.Timestamp, cache_disable: bool | None = None
    ):
        """Build a ``wetterdienst`` request.

        :param variables: Canonical variable names.
        :type variables: collections.abc.Sequence[str]
        :param start: Window start.
        :type start: pandas.Timestamp
        :param end: Window end.
        :type end: pandas.Timestamp
        :param cache_disable: Override the instance's cache setting.
        :type cache_disable: bool | None
        :returns: Configured request object.
        :rtype: wetterdienst.provider.meteofrance.observation.MeteoFranceObservationRequest
        :raises KeyError: If a variable is not supported by this provider.
        """
        from wetterdienst import Settings
        from wetterdienst.provider.meteofrance.observation import MeteoFranceObservationRequest

        unknown = [v for v in variables if v not in self._PARAMS]
        if unknown:
            raise KeyError(f"{unknown} not supported by {self.name}; available: {sorted(self._PARAMS)}")
        disable = self.cache_disable if cache_disable is None else cache_disable
        return MeteoFranceObservationRequest(
            parameters=[self._PARAMS[v][:3] for v in variables],
            **_period_arguments(MeteoFranceObservationRequest, start, end),
            settings=Settings(cache_disable=disable),
        )

    @staticmethod
    def _naive_utc(series: pd.Series) -> pd.Series:
        """Convert a timestamp column to naive UTC.

        :param series: Timestamps, possibly timezone aware.
        :type series: pandas.Series
        :returns: Naive UTC timestamps.
        :rtype: pandas.Series
        """
        return pd.to_datetime(series, utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

    def list_stations(self, region, start, end, variables) -> pd.DataFrame:
        """List stations inside a bounding box. See :class:`StationProvider`."""
        lat_min, lat_max, lon_min, lon_max = region
        request = self._request(variables, start, end)
        df = request.filter_by_bbox(
            left=lon_min, bottom=min(lat_min, lat_max), right=lon_max, top=max(lat_min, lat_max)
        ).df.to_pandas()
        df = df.drop_duplicates("station_id").copy()
        for column in ("start_date", "end_date"):
            df[column] = self._naive_utc(df[column])
        df["station_id"] = df["station_id"].astype(str)
        return df[["station_id", "name", "latitude", "longitude", "elevation", "start_date", "end_date"]].reset_index(
            drop=True
        )

    @staticmethod
    def _missing_file(exc: BaseException) -> bool:
        """Report whether an exception chain contains a missing-file error.

        :param exc: Exception to inspect.
        :type exc: BaseException
        :returns: ``True`` if a :class:`FileNotFoundError` appears in the chain.
        :rtype: bool
        """
        while exc is not None:
            if isinstance(exc, FileNotFoundError):
                return True
            exc = exc.__cause__ or exc.__context__
        return False

    @staticmethod
    def _broken_cache(exc: BaseException) -> bool:
        """Report whether an exception indicates a corrupted ``wetterdienst`` cache.

        :param exc: Exception to inspect.
        :type exc: BaseException
        :returns: ``True`` for the ``TypeError`` raised by a half-written cache entry.
        :rtype: bool
        """
        return isinstance(exc, TypeError) and "non-iterable bool" in str(exc)

    def _fetch_chunk(
        self,
        ids: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        variables: Sequence[str],
        cache_disable: bool | None = None,
    ) -> pd.DataFrame:
        """Fetch one chunk of stations, retrying without the file cache if it is corrupted.

        :param ids: Station identifiers.
        :type ids: collections.abc.Sequence[str]
        :param start: Window start.
        :type start: pandas.Timestamp
        :param end: Window end.
        :type end: pandas.Timestamp
        :param variables: Canonical variable names.
        :type variables: collections.abc.Sequence[str]
        :param cache_disable: Override the instance's cache setting.
        :type cache_disable: bool | None
        :returns: Long frame of observations, possibly empty.
        :rtype: pandas.DataFrame
        """
        request = self._request(variables, start, end, cache_disable).filter_by_station_id(list(ids))
        try:
            df = request.values.all().df.to_pandas()
        except TypeError as exc:
            if cache_disable or not self._broken_cache(exc):
                raise
            logger.warning("wetterdienst file cache looks corrupted (%s); retrying this chunk without it", exc)
            return self._fetch_chunk(ids, start, end, variables, cache_disable=True)
        if df.empty:
            return _EMPTY.copy()
        canonical = {self._PARAMS[v][2]: v for v in variables}
        offset = {v: self._PARAMS[v][3] for v in variables}
        out = pd.DataFrame(
            {
                "station_id": df["station_id"].astype(str),
                "variable": df["parameter"].astype(str).map(canonical),
                "time": self._naive_utc(df["timestamp"]),
                "value": df["value"].astype("f8"),
            }
        ).dropna(subset=["variable"])
        out["value"] = out["value"] + out["variable"].map(offset)
        return out[(out["time"] >= start) & (out["time"] <= end)]

    def fetch(self, station_ids, start, end, variables) -> pd.DataFrame:
        """Fetch observations in chunks. See :class:`StationProvider`."""
        ids = [str(i) for i in station_ids]
        frames = []
        for i in range(0, len(ids), self.chunk_size):
            chunk = ids[i : i + self.chunk_size]
            try:
                frames.append(self._fetch_chunk(chunk, start, end, variables))
            except Exception as exc:  # noqa: BLE001
                if not self._missing_file(exc):
                    raise
                for station_id in chunk:
                    try:
                        frames.append(self._fetch_chunk([station_id], start, end, variables))
                    except Exception as inner:  # noqa: BLE001
                        if not self._missing_file(inner):
                            raise
                        logger.info("station %s has no data file for the window", station_id)
        return pd.concat(frames, ignore_index=True) if frames else _EMPTY.copy()


@dataclass
class QCConfig:
    """Quality-control thresholds."""

    min_completeness: float = 0.9
    primary: str = "t2m"
    ranges: dict | None = None


def holdout_mask(station_ids: Sequence[str], frac: float = 0.2, salt: str = "dkit-v1") -> np.ndarray:
    """Select a reproducible subset of stations for testing.

    Membership is decided by hashing each identifier, so the split is stable
    when stations are added or removed.

    :param station_ids: Station identifiers.
    :type station_ids: collections.abc.Sequence[str]
    :param frac: Approximate fraction to hold out.
    :type frac: float
    :param salt: String mixed into the hash to allow alternative splits.
    :type salt: str
    :returns: Boolean mask, ``True`` for held-out stations.
    :rtype: numpy.ndarray
    """
    out = np.empty(len(station_ids), dtype=bool)
    for i, station_id in enumerate(station_ids):
        digest = int(hashlib.md5(f"{salt}:{station_id}".encode()).hexdigest()[:8], 16)
        out[i] = (digest % 10_000) < frac * 10_000
    return out


def _cache_file(cache_dir: str | Path, provider: str, variable: str, start, end, station_id: str) -> Path:
    """Build the cache path for one station, variable and window.

    :param cache_dir: Root cache directory.
    :type cache_dir: str | pathlib.Path
    :param provider: Provider name.
    :type provider: str
    :param variable: Canonical variable name.
    :type variable: str
    :param start: Window start.
    :type start: pandas.Timestamp
    :param end: Window end.
    :type end: pandas.Timestamp
    :param station_id: Station identifier.
    :type station_id: str
    :returns: Path to the gzipped CSV holding that station's values.
    :rtype: pathlib.Path
    """
    tag = f"{start:%Y%m%dT%H}_{end:%Y%m%dT%H}"
    return Path(cache_dir) / provider / variable / tag / f"{station_id}.csv.gz"


def _covering_dirs(cache_dir, provider: str, variable: str, start, end) -> list[Path]:
    """Find cached windows that fully contain the requested one.

    :param cache_dir: Root cache directory.
    :type cache_dir: str | pathlib.Path
    :param provider: Provider name.
    :type provider: str
    :param variable: Canonical variable name.
    :type variable: str
    :param start: Window start.
    :type start: pandas.Timestamp
    :param end: Window end.
    :type end: pandas.Timestamp
    :returns: Matching cache directories, narrowest first.
    :rtype: list[pathlib.Path]
    """
    root = Path(cache_dir) / provider / variable
    found = []
    if root.is_dir():
        for directory in root.iterdir():
            try:
                first, last = (pd.to_datetime(part, format="%Y%m%dT%H") for part in directory.name.split("_"))
            except (ValueError, TypeError):
                continue
            if first <= start and last >= end.floor("h"):
                found.append((last - first, directory))
    return [directory for _, directory in sorted(found)]


def _cached_file(cache_dir, provider: str, variable: str, start, end, station_id: str) -> Path | None:
    """Locate a cached file for a station, allowing a wider cached window.

    :param cache_dir: Root cache directory.
    :type cache_dir: str | pathlib.Path
    :param provider: Provider name.
    :type provider: str
    :param variable: Canonical variable name.
    :type variable: str
    :param start: Window start.
    :type start: pandas.Timestamp
    :param end: Window end.
    :type end: pandas.Timestamp
    :param station_id: Station identifier.
    :type station_id: str
    :returns: Path to a usable cache file, or ``None``.
    :rtype: pathlib.Path | None
    """
    exact = _cache_file(cache_dir, provider, variable, start, end, station_id)
    if exact.exists():
        return exact
    for directory in _covering_dirs(cache_dir, provider, variable, start, end):
        candidate = directory / f"{station_id}.csv.gz"
        if candidate.exists():
            return candidate
    return None


def _fetch_with_cache(provider, ids, start, end, variables, cache_dir) -> pd.DataFrame:
    """Fetch observations, reading from and writing to the per-station cache.

    An empty cache file records the fact that a station has no data; files are
    only written after a successful fetch, so a network error is never cached.

    :param provider: Station provider.
    :type provider: StationProvider
    :param ids: Station identifiers.
    :type ids: collections.abc.Sequence[str]
    :param start: Window start.
    :type start: pandas.Timestamp
    :param end: Window end.
    :type end: pandas.Timestamp
    :param variables: Canonical variable names.
    :type variables: collections.abc.Sequence[str]
    :param cache_dir: Root cache directory.
    :type cache_dir: str | pathlib.Path
    :returns: Long frame of observations within the window.
    :rtype: pandas.DataFrame
    """
    need, frames = [], []
    for station_id in ids:
        if any(_cached_file(cache_dir, provider.name, v, start, end, station_id) is None for v in variables):
            need.append(station_id)
    logger.info("station cache: %d/%d station(s) complete, fetching %d", len(ids) - len(need), len(ids), len(need))
    if need and len(need) == len(ids):
        root = Path(cache_dir) / provider.name / variables[0]
        if root.is_dir():
            logger.info(
                "cached windows for %s: %s (none covers %s .. %s)",
                variables[0],
                sorted(d.name for d in root.iterdir()),
                start,
                end,
            )

    if need:
        fetched = provider.fetch(need, start, end, variables)
        for station_id in need:
            station_rows = fetched[fetched["station_id"] == station_id] if len(fetched) else _EMPTY
            for variable in variables:
                path = _cache_file(cache_dir, provider.name, variable, start, end, station_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = station_rows[station_rows["variable"] == variable] if len(station_rows) else _EMPTY
                rows.to_csv(path, index=False, compression="gzip")

    for station_id in ids:
        for variable in variables:
            path = _cached_file(cache_dir, provider.name, variable, start, end, station_id)
            df = pd.read_csv(path, parse_dates=["time"], dtype={"station_id": str})
            if len(df):
                frames.append(df[(df["time"] >= start) & (df["time"] <= end)])
    return pd.concat(frames, ignore_index=True) if frames else _EMPTY.copy()


def load_stations(
    provider: StationProvider,
    region: str | Region,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    variables: Sequence[str] = ("t2m",),
    cache_dir: str | Path = "outputs/cache/stations",
    qc: QCConfig | None = None,
    holdout_frac: float = 0.2,
    active: str = "full",
    include_unknown_start: bool = False,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Load quality-controlled observations for every usable station in a region.

    :param provider: Station provider.
    :type provider: StationProvider
    :param region: Region name, bounding-box string or four-element sequence.
    :type region: str | tuple[float, float, float, float]
    :param start: Window start, naive UTC.
    :type start: str | pandas.Timestamp
    :param end: Window end, naive UTC.
    :type end: str | pandas.Timestamp
    :param variables: Canonical variable names to load.
    :type variables: collections.abc.Sequence[str]
    :param cache_dir: Root cache directory.
    :type cache_dir: str | pathlib.Path
    :param qc: Quality-control thresholds; ``None`` uses the defaults.
    :type qc: QCConfig | None
    :param holdout_frac: Fraction of stations flagged as held out.
    :type holdout_frac: float
    :param active: ``"full"`` requires activity over the whole window, ``"overlap"`` any overlap.
    :type active: str
    :param include_unknown_start: Keep stations whose start date is unknown.
    :type include_unknown_start: bool
    :returns: Observations over ``(time, station)`` with station metadata and a ``holdout``
        flag, and a per-station report with ``kept`` and ``reason``.
    :rtype: tuple[xarray.Dataset, pandas.DataFrame]
    :raises ValueError: If the timestamps are timezone aware, the primary variable is not
        requested, or ``active`` is not one of the two allowed values.
    :raises KeyError: If a variable is not in :data:`~downscaler.constants.STATION_VARIABLES`.
    """
    qc = qc or QCConfig()
    region = parse_region(region)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is not None or end.tzinfo is not None:
        raise ValueError("start and end must be naive UTC timestamps")
    variables = list(variables)
    unknown_variables = [v for v in variables if v not in STATION_VARIABLES]
    if unknown_variables:
        raise KeyError(f"unknown variable(s) {unknown_variables}; known: {sorted(STATION_VARIABLES)}")
    if qc.primary not in variables:
        raise ValueError(f"QC primary variable {qc.primary!r} must be among the requested variables")
    if active not in ("full", "overlap"):
        raise ValueError("active must be 'full' or 'overlap'")

    stations = provider.list_stations(region, start, end, variables).copy()
    stations["station_id"] = stations["station_id"].astype(str)
    stations = stations.drop_duplicates("station_id").set_index("station_id")

    first, last = stations["start_date"], stations["end_date"]
    unknown = first.isna()
    if active == "full":
        usable = (first <= start) & (last.isna() | (last >= end))
    else:
        usable = (first <= end) & (last.isna() | (last >= start))
    if include_unknown_start:
        cutoff = end if active == "full" else start
        usable = usable | (unknown & (last.isna() | (last >= cutoff)))

    reason = pd.Series("", index=stations.index, dtype=object)
    reason[~usable & unknown] = "unknown start date"
    reason[~usable & ~unknown] = "not active over the window"
    logger.info(
        "%d station(s) listed in region; %d active over the window (%d with unknown start date %s)",
        len(stations),
        int(usable.sum()),
        int(unknown.sum()),
        "included" if include_unknown_start else "skipped",
    )
    ids = list(stations.index[usable])

    long = _fetch_with_cache(provider, ids, start, end, variables, cache_dir) if ids else _EMPTY.copy()

    hours = pd.date_range(start.floor("h"), end.floor("h"), freq="1h")
    cube = {}
    for variable in variables:
        subset = long[long["variable"] == variable]
        wide = (
            subset.pivot_table(index="time", columns="station_id", values="value", aggfunc="first")
            if len(subset)
            else pd.DataFrame(index=pd.DatetimeIndex([], name="time"))
        )
        wide = wide.reindex(index=hours, columns=ids)
        low, high = (qc.ranges or {}).get(variable, STATION_VARIABLES[variable]["range"])
        out_of_range = ((wide < low) | (wide > high)).sum()
        cube[variable] = (wide.where((wide >= low) & (wide <= high)), out_of_range)

    primary, primary_out_of_range = cube[qc.primary]
    completeness = primary.notna().mean() if len(ids) else pd.Series(dtype=float)
    report = pd.DataFrame({"candidate": stations.index.isin(ids)}, index=stations.index)
    report["completeness"] = completeness.reindex(stations.index)
    report["n_out_of_range"] = primary_out_of_range.reindex(stations.index)
    kept_mask = completeness >= qc.min_completeness if len(ids) else pd.Series(dtype=bool)
    report["kept"] = kept_mask.reindex(stations.index).fillna(False).astype(bool)
    reason[report["candidate"] & ~report["kept"]] = f"completeness < {qc.min_completeness:.2f}"
    report["reason"] = reason
    report = report.join(stations[["name", "latitude", "longitude", "elevation"]])
    kept = list(report.index[report["kept"]])
    logger.info("QC kept %d/%d candidate station(s)", len(kept), len(ids))

    meta = stations.loc[kept]
    ds = xr.Dataset(
        {v: (("time", "station"), cube[v][0][kept].to_numpy(dtype="f4")) for v in variables},
        coords={
            "time": hours.to_numpy().astype("datetime64[ns]"),
            "station": np.array(kept, dtype=object),
            "latitude": ("station", meta["latitude"].to_numpy("f8")),
            "longitude": ("station", meta["longitude"].to_numpy("f8")),
            "elevation": ("station", meta["elevation"].to_numpy("f8")),
            "name": ("station", meta["name"].astype(str).to_numpy(dtype=object)),
        },
    )
    ds["holdout"] = ("station", holdout_mask(kept, holdout_frac))
    for variable in variables:
        ds[variable].attrs["units"] = STATION_VARIABLES[variable]["unit"]
    ds.attrs.update(
        {
            "provider": provider.name,
            "license": provider.license,
            "window": f"{start} .. {end} (UTC, hourly, inclusive)",
            "region": list(region),
            "longitude_convention": "signed -180..180",
            "qc_min_completeness": qc.min_completeness,
            "holdout_frac": holdout_frac,
        }
    )
    return ds, report


def load_published_stations(
    path_or_repo: str | Path,
    revision: str | None = None,
    start=None,
    end=None,
) -> xr.Dataset:
    """Read a published station dataset, local or from the Hugging Face Hub.

    Returns the same object as :func:`load_stations`, so the two are
    interchangeable: observations over ``(time, station)`` with station
    coordinates and the holdout flag.

    :param path_or_repo: Local folder holding the Parquet files, or a Hub dataset repository id.
    :type path_or_repo: str | pathlib.Path
    :param revision: Revision to download when reading from the Hub.
    :type revision: str | None
    :param start: First time to keep; ``None`` keeps everything from the start. A dataset can cover
        several separate periods, so selecting one keeps the result small.
    :type start: str | pandas.Timestamp | None
    :param end: Last time to keep; a date alone includes that whole day.
    :type end: str | pandas.Timestamp | None
    :returns: Observations over the hours present in the window, for the stations that report in it,
        with station metadata.
    :rtype: xarray.Dataset
    :raises FileNotFoundError: If the folder lacks the expected Parquet files.
    """
    folder = Path(path_or_repo)
    if not folder.exists():
        from huggingface_hub import snapshot_download

        folder = Path(snapshot_download(repo_id=str(path_or_repo), repo_type="dataset", revision=revision))
    missing = [name for name in ("metadata.parquet", "observations.parquet") if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(f"{folder} does not contain {missing}")

    meta = pd.read_parquet(folder / "metadata.parquet")
    obs = pd.read_parquet(folder / "observations.parquet")
    obs["station_id"] = obs["station_id"].astype(str)
    if start is not None:
        obs = obs[obs["time"] >= pd.Timestamp(start)]
    if end is not None:
        last = pd.Timestamp(end)
        if isinstance(end, str) and len(end.strip()) <= len("YYYY-MM-DD"):
            last = last + pd.Timedelta(hours=23, minutes=59)
        obs = obs[obs["time"] <= last]
    if obs.empty:
        raise ValueError(f"{path_or_repo} has no observation between {start} and {end}")
    meta = meta[meta["station_id"].astype(str).isin(set(obs["station_id"]))]
    ids = meta["station_id"].astype(str).to_list()
    variables = [c for c in obs.columns if c not in ("station_id", "time")]
    # Only the hours present: a merged dataset spans separate periods, and the gaps hold nothing.
    hours = pd.DatetimeIndex(np.sort(obs["time"].unique()))
    data = {}
    for variable in variables:
        wide = obs.pivot_table(index="time", columns="station_id", values=variable, aggfunc="first")
        data[variable] = (("time", "station"), wide.reindex(index=hours, columns=ids).to_numpy("f4"))
    ds = xr.Dataset(
        data,
        coords={
            "time": hours.to_numpy().astype("datetime64[ns]"),
            "station": np.array(ids, dtype=object),
            "latitude": ("station", meta["latitude"].to_numpy("f8")),
            "longitude": ("station", meta["longitude"].to_numpy("f8")),
            "elevation": ("station", meta["elevation_m"].to_numpy("f8")),
            "name": ("station", meta["name"].astype(str).to_numpy(dtype=object)),
        },
    )
    ds["holdout"] = ("station", meta["holdout"].to_numpy(bool))
    for variable in variables:
        ds[variable].attrs["units"] = STATION_VARIABLES.get(variable, {}).get("unit", "K")
    logger.info("loaded %d published station(s) over %s .. %s", len(ids), hours[0], hours[-1])
    return ds


def at_times(ds: xr.Dataset, times: Sequence) -> xr.Dataset:
    """Select observations at given times, filling absent ones with NaN.

    :param ds: Station dataset.
    :type ds: xarray.Dataset
    :param times: Times to select.
    :type times: collections.abc.Sequence
    :returns: Dataset reindexed onto those times.
    :rtype: xarray.Dataset
    """
    return ds.reindex(time=pd.DatetimeIndex(times).to_numpy().astype("datetime64[ns]"))
