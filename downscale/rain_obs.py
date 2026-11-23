"""Independent rainfall observations on the downscaler's grid and hours.

Two references, both independent of PARDIG and of the downscaler:

* Météo-France hourly rain gauges (``RR1``), open data on data.gouv.fr, one
  compressed CSV per department and period. Points: no structure metrics.
* EURADCLIM v3 (KNMI), OPERA radar adjusted with ECA&D daily gauges, 1-h
  accumulations on a 2 km equal-area grid, 2013-2023. Maps: every metric.

Hours are right-labelled UTC, like the rainfall packs: the value at ``t``
is the depth over ``(t-1h, t]``. ``gauge_lag_check`` verifies the gauge
convention against the radar instead of assuming it.
"""
from __future__ import annotations

import io
import json
import os
import re
import time as clock
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_GOUV = "https://www.data.gouv.fr/api/1/datasets/donnees-climatologiques-de-base-horaires/"
KNMI_API = "https://api.dataplatform.knmi.nl/open-data/v1/datasets"
EURADCLIM = ("RAD_OPERA_HOURLY_RAINFALL_ACCUMULATION_EURADCLIM", "3.0")
# KNMI publishes this shared anonymous key; it is heavily rate limited, so set
# KNMI_API_KEY to a free personal key from developer.dataplatform.knmi.nl.
KNMI_ANONYMOUS_KEY = ("eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjUzYTg1ZDBhMmQ5YzRkYzJiYWNlNzQ4"
                      "NTQ2Zjk4ODExIiwiaCI6Im11cm11cjEyOCJ9")
# Quality codes of RR1: 0 protected, 1 validated, 9 filtered, 2 doubtful.
GOOD_QUALITY = {0, 1, 9}


def _download(url, path, headers=None, attempts=6):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        response = requests.get(url, headers=headers, stream=True, timeout=120)
        if response.status_code == 429 and attempt < attempts - 1:
            clock.sleep(30 * (attempt + 1))
            continue
        response.raise_for_status()
        temporary = path.with_suffix(path.suffix + ".part")
        with temporary.open("wb") as stream:
            for block in response.iter_content(1 << 20):
                stream.write(block)
        temporary.replace(path)
        return path
    raise RuntimeError(f"could not download {url}")


# --------------------------------------------------------------------------- gauges

def gauge_catalogue(cache):
    """Department files of the hourly dataset: department, first and last year, URL."""
    path = Path(cache) / "gauge_catalogue.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resources = requests.get(DATA_GOUV, timeout=60).json()["resources"]
        rows = []
        for item in resources:
            match = re.match(r"HOR_departement_(\w+?)_periode_(\d{4})-(\d{4})$", item["title"])
            if match:
                rows.append({"department": match[1], "first": int(match[2]), "last": int(match[3]), "url": item["url"]})
        path.write_text(json.dumps(rows))
    return pd.DataFrame(json.loads(path.read_text()))


def mainland(department):
    """Metropolitan France: 01-95 and Corsica; overseas departments are 97x."""
    return department in ("2A", "2B") or (department.isdigit() and 1 <= int(department) <= 95)


def read_gauge_file(source, times):
    """Good-quality hourly rain of one department file at the requested hour ends."""
    wanted = set(pd.DatetimeIndex(times).strftime("%Y%m%d%H").astype(int))
    columns = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "AAAAMMJJHH", "RR1", "QRR1"]
    parts = []
    for chunk in pd.read_csv(source, sep=";", usecols=columns, chunksize=500_000,
                             dtype={"NUM_POSTE": str, "NOM_USUEL": str}):
        chunk = chunk[chunk.AAAAMMJJHH.isin(wanted) & chunk.RR1.notna()]
        chunk = chunk[chunk.QRR1.fillna(9).astype(int).isin(GOOD_QUALITY)]
        parts.append(chunk)
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    return pd.DataFrame({
        "station_id": frame.NUM_POSTE.str.zfill(8), "name": frame.NOM_USUEL,
        "lat": frame.LAT.astype(float), "lon": frame.LON.astype(float), "elevation_m": frame.ALTI.astype(float),
        "time": pd.to_datetime(frame.AAAAMMJJHH.astype(str), format="%Y%m%d%H"),
        "rain_mm": frame.RR1.astype("float32")})


def load_gauges(times, cache, bbox=None, departments=None):
    """Long table of hourly gauge rain (mm) at the requested right-labelled UTC hours.

    Filtered rows are cached per department, so the large files are read once.
    """
    times = pd.DatetimeIndex(times)
    catalogue = gauge_catalogue(cache)
    catalogue = catalogue[catalogue.department.map(mainland)]
    if departments is not None:
        catalogue = catalogue[catalogue.department.isin(departments)]
    key = f"{times.min():%Y%m%d%H}_{times.max():%Y%m%d%H}_{len(times)}"
    frames = []
    for department, rows in catalogue.groupby("department"):
        out = Path(cache) / "gauges" / f"{department}_{key}.csv.gz"
        if not out.exists():
            parts = []
            for year in sorted(set(times.year) | set((times - pd.Timedelta(hours=1)).year)):
                match = rows[(rows["first"] <= year) & (rows["last"] >= year)]
                for url in match.url:
                    raw = _download(url, Path(cache) / "gauges" / "raw" / Path(url).name)
                    parts.append(read_gauge_file(raw, times))
                    raw.unlink()  # department files are tens of MB; keep only the filtered rows
            frame = pd.concat(parts, ignore_index=True).drop_duplicates(["station_id", "time"])
            out.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(out, index=False)
        frames.append(pd.read_csv(out, dtype={"station_id": str}, parse_dates=["time"]))
    gauges = pd.concat(frames, ignore_index=True)
    if bbox is not None:
        south, north, west, east = bbox
        gauges = gauges[gauges.lat.between(south, north) & gauges.lon.between(west, east)]
    return gauges.reset_index(drop=True)


def wet_days(start, end, cache, threshold=1.0, departments=None):
    """Daily share of gauge-hours with at least ``threshold`` mm, over mainland France.

    Used to choose test days by observed rain only (never by model scores), so
    a test set is not mostly dry hours that every model gets right.
    """
    catalogue = gauge_catalogue(cache)
    catalogue = catalogue[catalogue.department.map(mainland)]
    if departments is not None:
        catalogue = catalogue[catalogue.department.isin(departments)]
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    lo, hi = int(f"{start:%Y%m%d}00"), int(f"{end:%Y%m%d}23")
    totals = []
    for department, rows in catalogue.groupby("department"):
        out = Path(cache) / "wet_days" / f"{department}_{lo}_{hi}_{threshold:g}.csv"
        if not out.exists():
            parts = []
            for url in rows[(rows["first"] <= end.year) & (rows["last"] >= start.year)].url:
                raw = _download(url, Path(cache) / "gauges" / "raw" / Path(url).name)
                for chunk in pd.read_csv(raw, sep=";", usecols=["AAAAMMJJHH", "RR1", "QRR1"], chunksize=500_000):
                    chunk = chunk[chunk.AAAAMMJJHH.between(lo, hi) & chunk.RR1.notna()
                                  & chunk.QRR1.fillna(9).astype(int).isin(GOOD_QUALITY)]
                    # Right-labelled hours: 00 UTC closes the previous day.
                    stamp = pd.to_datetime(chunk.AAAAMMJJHH.astype(str), format="%Y%m%d%H") - pd.Timedelta(hours=1)
                    parts.append(pd.DataFrame({"day": stamp.dt.floor("D"), "wet": chunk.RR1 >= threshold, "rain": chunk.RR1}))
                raw.unlink()
            frame = pd.concat(parts) if parts else pd.DataFrame(columns=["day", "wet", "rain"])
            daily = frame.groupby("day").agg(hours=("wet", "size"), wet=("wet", "sum"), rain=("rain", "sum"))
            out.parent.mkdir(parents=True, exist_ok=True)
            daily.to_csv(out)
        totals.append(pd.read_csv(out, parse_dates=["day"]))
    daily = pd.concat(totals).groupby("day")[["hours", "wet", "rain"]].sum()
    daily = daily.loc[start.floor("D"):end.floor("D")]  # edge hours of partial days
    daily["wet_fraction"] = daily.wet / daily.hours
    daily["mean_rain_mm"] = daily.rain / daily.hours * 24
    return daily


def choose_test_days(daily, per_season=3, quantile=0.75, seed=0):
    """Random wet days per season and year, from gauges only.

    Days above the ``quantile`` of daily wet fraction are drawn with a fixed
    seed, so the choice is reproducible and blind to every model.
    """
    daily = daily.copy()
    daily["season"] = [("DJF", "MAM", "JJA", "SON")[(m % 12) // 3] for m in daily.index.month]
    daily["year"] = daily.index.year
    chosen = []
    for _, group in daily.groupby(["year", "season"]):
        wet = group[group.wet_fraction >= group.wet_fraction.quantile(quantile)]
        chosen += list(wet.sample(min(per_season, len(wet)), random_state=seed).index)
    return sorted(chosen)


def gauge_matrix(gauges, times, lat, lon, min_fraction=0.9):
    """(time, station) rain with NaN gaps, plus each station's grid cell.

    Stations need ``min_fraction`` of the hours. A station is compared with the
    0.1° cell containing it: a point sees sharper rain than a cell average, so
    gauge scores include this representativeness error for every model alike.
    """
    times = pd.DatetimeIndex(times)
    table = gauges.pivot_table(index="time", columns="station_id", values="rain_mm", aggfunc="first")
    table = table.reindex(times)
    table = table.loc[:, table.notna().mean() >= min_fraction]
    meta = gauges.drop_duplicates("station_id").set_index("station_id").loc[table.columns]
    row = np.round((meta.lat.values - lat[0]) / (lat[1] - lat[0])).astype(int)
    col = np.round((meta.lon.values - lon[0]) / (lon[1] - lon[0])).astype(int)
    inside = (row >= 0) & (row < len(lat)) & (col >= 0) & (col < len(lon))
    meta = meta[inside].assign(row=row[inside], col=col[inside])
    return table.values[:, inside].astype("float32"), meta.reset_index()


def at_stations(fields, stations):
    """Sample (time, member, lat, lon) fields at station cells -> (time, member, station)."""
    return fields[..., stations.row.values, stations.col.values]


# --------------------------------------------------------------------------- EURADCLIM

def _knmi_headers():
    return {"Authorization": os.environ.get("KNMI_API_KEY", KNMI_ANONYMOUS_KEY)}


def _knmi_get(url, params=None, attempts=8):
    for attempt in range(attempts):
        response = requests.get(url, params=params, headers=_knmi_headers(), timeout=60)
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()
        clock.sleep(30 * (attempt + 1))
    raise RuntimeError("KNMI rate limit: set KNMI_API_KEY to a personal key (free at developer.dataplatform.knmi.nl)")


def euradclim_files(cache):
    path = Path(cache) / "euradclim_files.json"
    if not path.exists():
        name, version = EURADCLIM
        files, token = [], None
        while True:
            params = {"maxKeys": 500, **({"nextPageToken": token} if token else {})}
            page = _knmi_get(f"{KNMI_API}/{name}/versions/{version}/files", params)
            files += [item["filename"] for item in page.get("files", [])]
            token = page.get("nextPageToken")
            if not token:
                break
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(files))
    return json.loads(path.read_text())


def euradclim_month(month, cache):
    """Download (once) the monthly archive covering ``month`` (a Timestamp)."""
    stamp = f"{month:%Y%m}"
    candidates = [f for f in euradclim_files(cache) if stamp in f]
    if len(candidates) != 1:
        raise ValueError(f"expected one EURADCLIM archive for {stamp}, found {candidates}")
    target = Path(cache) / "euradclim" / candidates[0]
    if not target.exists():
        name, version = EURADCLIM
        url = _knmi_get(f"{KNMI_API}/{name}/versions/{version}/files/{candidates[0]}/url")["temporaryDownloadUrl"]
        _download(url, target)
    return target


def _stamp(name):
    match = re.findall(r"(20\d{10})", Path(name).name)
    return pd.Timestamp(f"{match[-1][:8]}T{match[-1][8:]}") if match else None


def read_odim(source):
    """Values (mm, NaN for no data) and pixel-centre lat/lon of an ODIM HDF5 composite."""
    import h5py
    from pyproj import CRS, Transformer
    with h5py.File(source, "r") as h5:
        data = h5["dataset1/data1/data"][()].astype("float64")
        what = {}
        for group in ("dataset1/what", "dataset1/data1/what"):
            if group in h5:
                what.update(h5[group].attrs)
        where = dict(h5["where"].attrs)
    nodata = float(what.get("nodata", -9999000.0))
    undetect = what.get("undetect")
    gain, offset = float(what.get("gain", 1.0)), float(what.get("offset", 0.0))
    missing = data == nodata
    dry = (data == float(undetect)) if undetect is not None else np.zeros_like(missing)
    values = data * gain + offset
    values[dry] = 0.0
    values[missing] = np.nan
    text = lambda v: v.decode() if isinstance(v, bytes) else str(v)
    crs = CRS.from_proj4(text(where["projdef"]))
    forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    back = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x0, y0 = forward.transform(float(where["UL_lon"]), float(where["UL_lat"]))
    ny, nx = values.shape
    x = x0 + (np.arange(nx) + 0.5) * float(where["xscale"])
    y = y0 - (np.arange(ny) + 0.5) * float(where["yscale"])
    lon, lat = back.transform(*np.meshgrid(x, y))
    return values, lat, lon


class Aggregator:
    """Average equal-area radar pixels inside each target cell (conservative here).

    A cell is kept when at least ``min_coverage`` of its pixels hold data, so
    partly observed cells, offshore and at the radar range limit, become NaN.
    """

    def __init__(self, src_lat, src_lon, lat, lon, min_coverage=0.9):
        dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
        row = np.floor((src_lat - lat[0]) / dlat + 0.5).astype(int)
        col = np.floor((src_lon - lon[0]) / dlon + 0.5).astype(int)
        inside = (row >= 0) & (row < len(lat)) & (col >= 0) & (col < len(lon))
        self.pixels = np.flatnonzero(inside.ravel())
        self.cell = (row * len(lon) + col).ravel()[self.pixels]
        self.size, self.shape = len(lat) * len(lon), (len(lat), len(lon))
        self.total = np.bincount(self.cell, minlength=self.size)
        self.min_coverage = min_coverage

    def __call__(self, values):
        v = values.ravel()[self.pixels]
        ok = np.isfinite(v)
        sums = np.bincount(self.cell[ok], weights=v[ok], minlength=self.size)
        counts = np.bincount(self.cell[ok], minlength=self.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sums / counts
        mean[(counts < self.min_coverage * self.total) | (self.total == 0)] = np.nan
        return mean.reshape(self.shape).astype("float32")


def load_euradclim(times, lat, lon, cache, min_coverage=0.9):
    """(time, lat, lon) EURADCLIM 1-h depth on the rainfall grid; NaN where unobserved."""
    times = pd.DatetimeIndex(times)
    out = Path(cache) / "euradclim" / f"grid_{times.min():%Y%m%d%H}_{times.max():%Y%m%d%H}_{len(times)}.npy"
    if out.exists():
        return np.load(out)
    result = np.full((len(times), len(lat), len(lon)), np.nan, dtype="float32")
    position = {t: i for i, t in enumerate(times)}
    aggregate = None
    for month in sorted({t.to_period("M") for t in times}):
        archive = euradclim_month(month.to_timestamp(), cache)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                stamp = _stamp(member)
                if stamp is None or stamp not in position:
                    continue
                values, src_lat, src_lon = read_odim(io.BytesIO(bundle.read(member)))
                if aggregate is None:
                    aggregate = Aggregator(src_lat, src_lon, lat, lon, min_coverage)
                result[position[stamp]] = aggregate(values)
    missing = np.isnan(result).all((1, 2))
    if missing.any():
        raise ValueError(f"EURADCLIM has no field for {list(times[missing][:3])}...")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, result)
    return result


def gauge_lag_check(gauge_values, stations, radar, lags=(-1, 0, 1)):
    """Correlation of gauges with the radar cell when gauges are shifted by each lag (hours).

    The largest correlation should be at lag 0 if both are right-labelled.
    Only valid when ``radar`` hours are consecutive.
    """
    radar_points = radar[:, stations.row.values, stations.col.values]
    result = {}
    for lag in lags:
        g = np.roll(gauge_values, lag, axis=0)[2:-2]
        r = radar_points[2:-2]
        ok = np.isfinite(g) & np.isfinite(r)
        result[lag] = float(np.corrcoef(g[ok], r[ok])[0, 1])
    return result
