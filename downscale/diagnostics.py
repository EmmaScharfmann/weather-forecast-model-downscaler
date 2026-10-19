"""Diagnostics of the downscaled fields themselves, separate from station scores.

Station scores say how good a forecast is; these say whether the model behaves
sensibly across the domain. Two concerns are handled here: the sea, where the
training target is produced differently from the land and the model should not
be trusted, and the domain boundary, where the network sees less context than
in the interior.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

LAND_FRACTION = 0.5


def land_mask(source: xr.Dataset | dict, threshold: float = LAND_FRACTION) -> np.ndarray | None:
    """Read the land-sea mask from a store or a static-field mapping.

    :param source: Object that may carry a ``land_sea_mask`` field.
    :type source: xarray.Dataset | dict
    :param threshold: Land fraction above which a cell counts as land.
    :type threshold: float
    :returns: Boolean array, ``True`` over land, or ``None`` when no mask is present.
    :rtype: numpy.ndarray | None
    """
    if "land_sea_mask" not in source:
        return None
    values = source["land_sea_mask"]
    return np.asarray(values.values if hasattr(values, "values") else values) > threshold


def mask_sea(field: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Replace sea points with missing values for plotting.

    :param field: Field on the fine grid.
    :type field: numpy.ndarray
    :param mask: Land mask from :func:`land_mask`; ``None`` leaves the field unchanged.
    :type mask: numpy.ndarray | None
    :returns: Field with sea points set to NaN.
    :rtype: numpy.ndarray
    """
    if mask is None:
        return field
    return np.where(mask, field, np.nan)


def distance_from_edge(shape: tuple[int, int]) -> np.ndarray:
    """Number of grid cells between each point and the nearest domain boundary.

    :param shape: Grid shape ``(y, x)``.
    :type shape: tuple[int, int]
    :returns: Integer array of the same shape, zero on the boundary.
    :rtype: numpy.ndarray
    """
    ny, nx = shape
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return np.minimum.reduce([y, x, ny - 1 - y, nx - 1 - x])


def edge_profile(
    downscaled: xr.Dataset, max_distance: int = 40, interior_from: int = 30, land_only: bool = True
) -> pd.DataFrame:
    """Describe how the correction and the sample spread vary with distance from the boundary.

    The network has less context near the edge of the domain, which inflates the
    spread there. Comparing each ring with the interior shows how far in the
    effect reaches.

    :param downscaled: Store written by the downscaling stage.
    :type downscaled: xarray.Dataset
    :param max_distance: Largest distance, in grid cells, to profile.
    :type max_distance: int
    :param interior_from: Distance beyond which points count as interior reference.
    :type interior_from: int
    :param land_only: Restrict the profile to land points when a mask is available.
    :type land_only: bool
    :returns: One row per distance with the mean absolute correction, the mean sample
        spread, and each as a ratio to the interior value.
    :rtype: pandas.DataFrame
    """
    coarse = downscaled["t2m_interp"].values
    fine = downscaled["t2m_ds"].values
    correction = np.abs(fine.mean(axis=-3) - coarse)
    spread = fine.std(axis=-3, ddof=1) if fine.shape[-3] > 1 else np.zeros_like(correction)
    correction = correction.reshape(-1, *correction.shape[-2:])
    spread = spread.reshape(-1, *spread.shape[-2:])

    distance = distance_from_edge(correction.shape[-2:])
    mask = land_mask(downscaled) if land_only else None
    usable = np.ones(distance.shape, bool) if mask is None else mask

    interior = usable & (distance >= interior_from)
    if not interior.any():
        raise ValueError(f"no points further than {interior_from} cells from the boundary")
    reference_correction = float(np.nanmean(correction[:, interior]))
    reference_spread = float(np.nanmean(spread[:, interior]))

    rows = []
    for d in range(min(max_distance, int(distance.max())) + 1):
        ring = usable & (distance == d)
        if not ring.any():
            continue
        mean_correction = float(np.nanmean(correction[:, ring]))
        mean_spread = float(np.nanmean(spread[:, ring]))
        rows.append({
            "distance_cells": d,
            "n_points": int(ring.sum()),
            "mean_abs_correction_K": mean_correction,
            "mean_spread_K": mean_spread,
            "correction_ratio": mean_correction / reference_correction if reference_correction else np.nan,
            "spread_ratio": mean_spread / reference_spread if reference_spread else np.nan,
        })
    profile = pd.DataFrame(rows)
    profile.attrs["interior_correction_K"] = reference_correction
    profile.attrs["interior_spread_K"] = reference_spread
    return profile


def suggested_margin(profile: pd.DataFrame, tolerance: float = 0.1) -> int:
    """Width of the band along the boundary where the model departs from its interior behaviour.

    The band is taken as the contiguous run of distances from the boundary
    inwards whose ratios exceed the tolerance, so isolated departures further
    in, where few grid points contribute, do not inflate the result.

    :param profile: Output of :func:`edge_profile`.
    :type profile: pandas.DataFrame
    :param tolerance: Allowed relative departure from the interior values.
    :type tolerance: float
    :returns: Number of grid cells to discard from each side of the domain.
    :rtype: int
    """
    ordered = profile.sort_values("distance_cells")
    within = (
        (ordered.spread_ratio - 1).abs().le(tolerance) & (ordered.correction_ratio - 1).abs().le(tolerance)
    )
    settled = ordered.distance_cells[within.values]
    return int(settled.iloc[0]) if len(settled) else int(ordered.distance_cells.max()) + 1


def edge_figure(profile: pd.DataFrame, margin: int, path) -> None:
    """Plot the edge profile with the suggested margin marked.

    :param profile: Output of :func:`edge_profile`.
    :type profile: pandas.DataFrame
    :param margin: Suggested margin from :func:`suggested_margin`.
    :type margin: int
    :param path: Destination image.
    :type path: str | pathlib.Path
    :returns: ``None``
    :rtype: None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(profile.distance_cells, profile.spread_ratio, marker="o", ms=3, label="sample spread")
    ax.plot(profile.distance_cells, profile.correction_ratio, marker="s", ms=3, label="absolute correction")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    if margin:
        ax.axvline(margin, color="tab:red", lw=1, ls=":")
        ax.text(margin + 0.5, ax.get_ylim()[1] * 0.95, f"margin {margin} cells", fontsize=8, color="tab:red")
    ax.set_xlabel("distance from the domain boundary (grid cells)")
    ax.set_ylabel("ratio to the interior value")
    ax.set_title("edge effect in the downscaled field")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def sea_report(downscaled: xr.Dataset) -> dict:
    """Compare the model's behaviour over sea and over land.

    The training target is produced differently over water, so a much larger
    correction or spread there is a sign the model should not be trusted at sea.

    :param downscaled: Store written by the downscaling stage.
    :type downscaled: xarray.Dataset
    :returns: Mean absolute correction and spread for land and sea, with their ratios;
        empty when the store carries no land-sea mask.
    :rtype: dict
    """
    mask = land_mask(downscaled)
    if mask is None:
        return {}
    coarse = downscaled["t2m_interp"].values
    fine = downscaled["t2m_ds"].values
    correction = np.abs(fine.mean(axis=-3) - coarse).reshape(-1, *mask.shape)
    spread = (
        fine.std(axis=-3, ddof=1).reshape(-1, *mask.shape)
        if fine.shape[-3] > 1 else np.zeros_like(correction)
    )
    out = {
        "land_fraction": float(mask.mean()),
        "land_correction_K": float(np.nanmean(correction[:, mask])),
        "sea_correction_K": float(np.nanmean(correction[:, ~mask])) if (~mask).any() else float("nan"),
        "land_spread_K": float(np.nanmean(spread[:, mask])),
        "sea_spread_K": float(np.nanmean(spread[:, ~mask])) if (~mask).any() else float("nan"),
    }
    out["correction_ratio_sea_over_land"] = out["sea_correction_K"] / out["land_correction_K"]
    out["spread_ratio_sea_over_land"] = (
        out["sea_spread_K"] / out["land_spread_K"] if out["land_spread_K"] else float("nan")
    )
    return out


def grid_spacing_km(lat2d: np.ndarray, lon2d: np.ndarray) -> float:
    """Mean distance between neighbouring grid cells.

    :param lat2d: Grid latitudes with shape ``(y, x)``.
    :type lat2d: numpy.ndarray
    :param lon2d: Grid longitudes with shape ``(y, x)``.
    :type lon2d: numpy.ndarray
    :returns: Mean spacing in kilometres, averaged over both grid directions.
    :rtype: float
    """
    from .constants import EARTH_RADIUS_KM

    def separation(lat_a, lon_a, lat_b, lon_b):
        phi_a, phi_b = np.deg2rad(lat_a), np.deg2rad(lat_b)
        d_phi, d_lambda = phi_b - phi_a, np.deg2rad(lon_b - lon_a)
        inner = np.sin(d_phi / 2) ** 2 + np.cos(phi_a) * np.cos(phi_b) * np.sin(d_lambda / 2) ** 2
        return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0, 1)))

    along_y = separation(lat2d[:-1, :], lon2d[:-1, :], lat2d[1:, :], lon2d[1:, :])
    along_x = separation(lat2d[:, :-1], lon2d[:, :-1], lat2d[:, 1:], lon2d[:, 1:])
    return float((np.mean(along_y) + np.mean(along_x)) / 2)


def power_spectrum(fields: np.ndarray, spacing_km: float, n_bins: int = 40) -> pd.DataFrame:
    """Radially averaged spatial power spectrum of a stack of fields.

    Each field has its plane removed and a Hann window applied before the
    transform, so that the domain edges and any large-scale gradient do not
    leak energy into the small scales.

    :param fields: Stack of shape ``(n, y, x)``; a single field may be passed as ``(y, x)``.
    :type fields: numpy.ndarray
    :param spacing_km: Grid spacing in kilometres.
    :type spacing_km: float
    :param n_bins: Number of logarithmically spaced wavelength bins.
    :type n_bins: int
    :returns: One row per bin with the central wavelength in kilometres and the mean power.
    :rtype: pandas.DataFrame
    """
    fields = np.asarray(fields, dtype="f8")
    if fields.ndim == 2:
        fields = fields[None]
    n, ny, nx = fields.shape
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    window = np.outer(np.hanning(ny), np.hanning(nx))
    correction = np.mean(window ** 2)

    design = np.stack([np.ones(ny * nx), y.ravel(), x.ravel()], 1)
    pseudo_inverse = np.linalg.pinv(design)
    power_sum = np.zeros((ny, nx))
    for field in fields:
        flat = field.ravel()
        plane = (design @ (pseudo_inverse @ flat)).reshape(ny, nx)
        spectrum = np.fft.fft2((field - plane) * window)
        power_sum += np.abs(spectrum) ** 2 / (correction * ny * nx)
    power = power_sum / n

    ky = np.fft.fftfreq(ny, d=spacing_km)[:, None]
    kx = np.fft.fftfreq(nx, d=spacing_km)[None, :]
    wavenumber = np.sqrt(ky ** 2 + kx ** 2)
    nyquist = 1.0 / (2 * spacing_km)
    smallest = 1.0 / max(ny, nx) / spacing_km
    edges = np.logspace(np.log10(smallest), np.log10(nyquist), n_bins + 1)
    index = np.digitize(wavenumber.ravel(), edges) - 1
    flat_power = power.ravel()
    rows = []
    for b in range(n_bins):
        chosen = index == b
        if not chosen.any():
            continue
        centre = np.sqrt(edges[b] * edges[b + 1])
        rows.append({
            "wavenumber_per_km": centre,
            "wavelength_km": 1.0 / centre,
            "power": float(flat_power[chosen].mean()),
            "n_modes": int(chosen.sum()),
        })
    return pd.DataFrame(rows).sort_values("wavelength_km", ascending=False).reset_index(drop=True)


def spectrum_comparison(fields: dict[str, np.ndarray], spacing_km: float, n_bins: int = 40) -> pd.DataFrame:
    """Power spectra of several field stacks on one set of wavelength bins.

    :param fields: Labels mapped to stacks of shape ``(n, y, x)``.
    :type fields: dict[str, numpy.ndarray]
    :param spacing_km: Grid spacing in kilometres.
    :type spacing_km: float
    :param n_bins: Number of wavelength bins.
    :type n_bins: int
    :returns: Frame indexed by wavelength with one column per label, plus the ratio of
        each field's power to the first label's.
    :rtype: pandas.DataFrame
    """
    spectra = {name: power_spectrum(stack, spacing_km, n_bins) for name, stack in fields.items()}
    reference = next(iter(spectra))
    out = spectra[reference][["wavelength_km", "n_modes"]].copy()
    for name, spectrum in spectra.items():
        out[name] = spectrum["power"].values
    for name in spectra:
        if name != reference:
            out[f"{name}_over_{reference}"] = out[name] / out[reference]
    return out


def effective_resolution_km(comparison: pd.DataFrame, field: str, reference: str, tolerance: float = 0.5) -> float:
    """Smallest wavelength at which a field still carries a comparable share of the reference power.

    :param comparison: Output of :func:`spectrum_comparison`.
    :type comparison: pandas.DataFrame
    :param field: Column to assess.
    :type field: str
    :param reference: Column treated as the truth.
    :type reference: str
    :param tolerance: Minimum ratio to the reference power that still counts as resolved.
    :type tolerance: float
    :returns: Wavelength in kilometres, or NaN when the field never falls below the tolerance.
    :rtype: float
    """
    ratio = comparison[field] / comparison[reference]
    ordered = comparison.sort_values("wavelength_km", ascending=False)
    below = ordered[ratio.loc[ordered.index] < tolerance]
    return float(below.wavelength_km.iloc[0]) if len(below) else float("nan")


def spectrum_figure(comparison: pd.DataFrame, labels: list[str], path) -> None:
    """Plot the power spectra and their ratio to the first label.

    :param comparison: Output of :func:`spectrum_comparison`.
    :type comparison: pandas.DataFrame
    :param labels: Column names to plot, the first being the reference.
    :type labels: list[str]
    :param path: Destination image.
    :type path: str | pathlib.Path
    :returns: ``None``
    :rtype: None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = labels[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name in labels:
        ax[0].loglog(comparison.wavelength_km, comparison[name], marker="o", ms=2.5, label=name)
    ax[0].set_xlabel("wavelength (km)")
    ax[0].set_ylabel("power")
    ax[0].invert_xaxis()
    ax[0].set_title("spatial power spectrum")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8)

    for name in labels[1:]:
        ax[1].semilogx(comparison.wavelength_km, comparison[name] / comparison[reference], marker="o", ms=2.5, label=name)
    ax[1].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[1].axhline(0.5, color="gray", lw=0.8, ls=":")
    ax[1].set_xlabel("wavelength (km)")
    ax[1].set_ylabel(f"power relative to {reference}")
    ax[1].invert_xaxis()
    ax[1].set_title("too little detail below one, invented detail above")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=8)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def diverging_white_cmap():
    """Colour map running red through white to green, with white at the midpoint.

    :returns: Colour map suitable for a quantity whose sign matters.
    :rtype: matplotlib.colors.LinearSegmentedColormap
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "gain", ["#8c2d04", "#d7604d", "#f7f7f7", "#5aae61", "#1b5e20"]
    )


def add_map_background(axis, extent: tuple[float, float, float, float] | None = None) -> bool:
    """Draw coastlines, borders and shaded land behind a station map.

    Falls back silently when cartopy is unavailable, so the figure is still
    produced without geography.

    :param axis: Axes to draw on.
    :type axis: matplotlib.axes.Axes
    :param extent: ``(lon_min, lon_max, lat_min, lat_max)``; ``None`` leaves the limits alone.
    :type extent: tuple[float, float, float, float] | None
    :returns: ``True`` when the background was drawn.
    :rtype: bool
    """
    try:
        import cartopy.feature as cfeature
    except ImportError:
        logger.info("cartopy is not installed: drawing the station map without geography")
        return False
    axis.add_feature(cfeature.OCEAN, facecolor="#dce9f2", zorder=0)
    axis.add_feature(cfeature.LAND, facecolor="#f5f2ec", zorder=0)
    axis.add_feature(cfeature.LAKES, facecolor="#dce9f2", edgecolor="none", zorder=1)
    axis.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#555555", zorder=2)
    axis.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#999999", linestyle="-", zorder=2)
    if extent is not None:
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
    return True


def _map_axes(fig, position, extent):
    """Create axes carrying a geographic projection when cartopy is available.

    :param fig: Figure to add the axes to.
    :type fig: matplotlib.figure.Figure
    :param position: Subplot specification passed to ``add_subplot``.
    :type position: typing.Any
    :param extent: ``(lon_min, lon_max, lat_min, lat_max)``.
    :type extent: tuple[float, float, float, float]
    :returns: The axes and whether a projection was used.
    :rtype: tuple[matplotlib.axes.Axes, bool]
    """
    try:
        import cartopy.crs as ccrs
    except ImportError:
        axis = fig.add_subplot(*position)
        axis.set_aspect(1 / np.cos(np.deg2rad((extent[2] + extent[3]) / 2)))
        return axis, False
    axis = fig.add_subplot(*position, projection=ccrs.PlateCarree())
    axis.set_extent(extent, crs=ccrs.PlateCarree())
    # Plate carrée alone would draw a degree of longitude as long as one of latitude.
    axis.set_aspect(1 / np.cos(np.deg2rad((extent[2] + extent[3]) / 2)))
    return axis, True


def gain_profile(
    table: pd.DataFrame, by: str, edges: Sequence[float], baseline: str = "coarse",
    improved: str = "downscaled", unit: str = "m",
) -> pd.DataFrame:
    """Absolute and relative gain from downscaling, grouped into bands.

    The relative figure matters because the coarse error is itself much larger
    in the mountains, so an absolute gain of the same size means less there.

    :param table: Output of :meth:`StationErrors.table`.
    :type table: pandas.DataFrame
    :param by: Column to band, for example ``"elevation"``.
    :type by: str
    :param edges: Band edges.
    :type edges: collections.abc.Sequence[float]
    :param baseline: Method treated as the reference.
    :type baseline: str
    :param improved: Method compared against it.
    :type improved: str
    :param unit: Unit shown in the band labels.
    :type unit: str
    :returns: One row per band with station counts, both errors, the gain and the gain as a percentage.
    :rtype: pandas.DataFrame
    """
    values = table[by].values
    reference = table[f"rmse_{baseline}"].values
    gain = reference - table[f"rmse_{improved}"].values
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        chosen = (values >= low) & (values < high) & np.isfinite(gain)
        if not chosen.any():
            continue
        label = f"{low:.0f}-{high:.0f} {unit}" if high < 1e5 else f"above {low:.0f} {unit}"
        mean_reference = float(reference[chosen].mean())
        mean_gain = float(gain[chosen].mean())
        rows.append({
            "band": label,
            "n_stations": int(chosen.sum()),
            f"rmse_{baseline}_K": round(mean_reference, 3),
            f"rmse_{improved}_K": round(float(table[f"rmse_{improved}"].values[chosen].mean()), 3),
            "mean_gain_K": round(mean_gain, 3),
            "relative_gain_pct": round(100 * mean_gain / mean_reference, 1) if mean_reference else np.nan,
            "improved_fraction": round(float((gain[chosen] > 0).mean()), 3),
        })
    return pd.DataFrame(rows)


class StationErrors:
    """Accumulates per-station error statistics across initialisations."""

    def __init__(self, methods: Sequence[str]):
        """
        :param methods: Method names to track.
        :type methods: collections.abc.Sequence[str]
        """
        self.methods = list(methods)
        self.sums: dict[str, np.ndarray] = {}
        self.station_ids: np.ndarray | None = None
        self.metadata: dict[str, np.ndarray] = {}

    def add(self, forecast_at_stations: xr.Dataset, obs: xr.DataArray) -> None:
        """Accumulate one initialisation.

        :param forecast_at_stations: Forecasts read at stations.
        :type forecast_at_stations: xarray.Dataset
        :param obs: Observations over ``(init_time, lead_time, station)``.
        :type obs: xarray.DataArray
        :returns: ``None``
        :rtype: None
        """
        if self.station_ids is None:
            self.station_ids = np.asarray(forecast_at_stations["station"].values)
            for name in ("latitude", "longitude", "elevation"):
                self.metadata[name] = np.asarray(forecast_at_stations[name].values)
        truth = obs.values
        valid = np.isfinite(truth)
        for method in self.methods:
            values = forecast_at_stations[method].values
            mean = values.mean(2)
            difference = np.where(valid, mean - truth, np.nan)
            entry = self.sums.setdefault(method, np.zeros((3, truth.shape[-1])))
            entry[0] += np.nansum(difference, axis=(0, 1))
            entry[1] += np.nansum(difference ** 2, axis=(0, 1))
            entry[2] += valid.sum(axis=(0, 1))

    def table(self) -> pd.DataFrame:
        """Per-station bias and root mean squared error for every method.

        :returns: One row per station, with ``bias_<method>`` and ``rmse_<method>`` columns.
        :rtype: pandas.DataFrame
        """
        out = pd.DataFrame({"station": self.station_ids, **self.metadata})
        for method, (total, squared, count) in self.sums.items():
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"bias_{method}"] = total / count
                out[f"rmse_{method}"] = np.sqrt(squared / count)
        out["n"] = next(iter(self.sums.values()))[2].astype(int)
        return out


def station_error_figure(table: pd.DataFrame, methods: Sequence[str], labels: dict, path) -> None:
    """Map the per-station error of two methods on a shared diverging scale.

    :param table: Output of :meth:`StationErrors.table`.
    :type table: pandas.DataFrame
    :param methods: The two method names to compare.
    :type methods: collections.abc.Sequence[str]
    :param labels: Display names keyed by method.
    :type labels: dict
    :param path: Destination image.
    :type path: str | pathlib.Path
    :returns: ``None``
    :rtype: None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limit = float(np.nanpercentile(np.abs(np.concatenate([table[f"bias_{m}"] for m in methods])), 98))
    extent = (
        float(table.longitude.min()) - 0.6, float(table.longitude.max()) + 0.6,
        float(table.latitude.min()) - 0.4, float(table.latitude.max()) + 0.4,
    )
    fig = plt.figure(figsize=(7 * len(methods), 6.5), constrained_layout=True)
    for position, method in enumerate(methods, start=1):
        axis, _ = _map_axes(fig, (1, len(methods), position), extent)
        add_map_background(axis, extent)
        points = axis.scatter(
            table.longitude, table.latitude, c=table[f"bias_{method}"], cmap="RdBu_r",
            vmin=-limit, vmax=limit, s=14, edgecolors="k", linewidths=0.2, zorder=3,
        )
        mean_bias = float(np.nanmean(table[f"bias_{method}"]))
        axis.set_title(f"{labels.get(method, method)}\nmean station bias {mean_bias:+.2f} K")
        fig.colorbar(points, ax=axis, label="forecast minus observation (K)")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def improvement_figure(
    table: pd.DataFrame, baseline: str, improved: str, orography: np.ndarray | None,
    lat2d: np.ndarray | None, lon2d: np.ndarray | None, path,
) -> dict:
    """Map where downscaling helps, in absolute and relative terms.

    The absolute map shows that the gains concentrate over high ground; the
    relative panel asks whether that simply reflects the larger errors there.

    :param table: Output of :meth:`StationErrors.table`.
    :type table: pandas.DataFrame
    :param baseline: Method treated as the reference.
    :type baseline: str
    :param improved: Method compared against it.
    :type improved: str
    :param orography: Unused, kept so callers need not know whether a background is drawn.
    :type orography: numpy.ndarray | None
    :param lat2d: Unused.
    :type lat2d: numpy.ndarray | None
    :param lon2d: Unused.
    :type lon2d: numpy.ndarray | None
    :param path: Destination image.
    :type path: str | pathlib.Path
    :returns: Correlations with elevation, the mean gain below 500 m and above 1000 m,
        the same in relative terms, and the fraction of stations improved.
    :rtype: dict
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = table[f"rmse_{baseline}"]
    gain = reference - table[f"rmse_{improved}"]
    relative = 100 * gain / reference
    usable = np.isfinite(gain) & np.isfinite(table.elevation) & (reference > 0)
    low_ground = usable & (table.elevation < 500)
    high_ground = usable & (table.elevation >= 1000)
    summary = {
        "correlation_gain_elevation": round(float(np.corrcoef(table.elevation[usable], gain[usable])[0, 1]), 3),
        "correlation_relative_gain_elevation": round(
            float(np.corrcoef(table.elevation[usable], relative[usable])[0, 1]), 3
        ),
        "mean_gain_below_500m_K": round(float(gain[low_ground].mean()), 3),
        "mean_gain_above_1000m_K": round(float(gain[high_ground].mean()), 3),
        "mean_relative_gain_below_500m_pct": round(float(relative[low_ground].mean()), 1),
        "mean_relative_gain_above_1000m_pct": round(float(relative[high_ground].mean()), 1),
        "stations_improved_fraction": round(float((gain[usable] > 0).mean()), 3),
    }

    extent = (
        float(table.longitude.min()) - 0.6, float(table.longitude.max()) + 0.6,
        float(table.latitude.min()) - 0.4, float(table.latitude.max()) + 0.4,
    )
    cmap = diverging_white_cmap()
    fig = plt.figure(figsize=(19, 6.5), constrained_layout=True)
    for position, (values, title, label, scale) in enumerate(
        [
            (gain, "absolute reduction in station RMSE", "RMSE reduction (K)", None),
            (relative, "the same as a share of the coarse error", "RMSE reduction (%)", 40.0),
        ],
        start=1,
    ):
        axis, _ = _map_axes(fig, (1, 3, position), extent)
        add_map_background(axis, extent)
        limit = scale or float(np.nanpercentile(np.abs(values[usable]), 98))
        points = axis.scatter(
            table.longitude, table.latitude, c=values, cmap=cmap, vmin=-limit, vmax=limit,
            s=16, edgecolors="k", linewidths=0.2, zorder=3,
        )
        axis.set_title(f"{title}\n{100 * summary['stations_improved_fraction']:.0f}% of stations improve")
        fig.colorbar(points, ax=axis, label=label)

    axis = fig.add_subplot(1, 3, 3)
    axis.scatter(table.elevation[usable], gain[usable], s=10, alpha=0.35, color="tab:blue")
    order = np.argsort(table.elevation[usable].values)
    elevation_sorted = table.elevation[usable].values[order]
    window = max(10, int(usable.sum()) // 25)
    for values, colour, name, twin in (
        (gain, "tab:red", "absolute (K)", False),
        (relative, "tab:purple", "relative (%)", True),
    ):
        smoothed = pd.Series(values[usable].values[order]).rolling(
            window, center=True, min_periods=window // 2
        ).mean()
        target_axis = axis.twinx() if twin else axis
        target_axis.plot(elevation_sorted, smoothed, color=colour, lw=2, label=name)
        if twin:
            target_axis.set_ylabel("relative reduction (%)", color=colour)
            target_axis.tick_params(axis="y", labelcolor=colour)
    axis.axhline(0, color="k", lw=0.8)
    axis.set_xlabel("station elevation (m)")
    axis.set_ylabel("absolute reduction (K)", color="tab:red")
    axis.tick_params(axis="y", labelcolor="tab:red")
    axis.set_title(
        f"running mean over {window} stations\n"
        f"absolute correlation {summary['correlation_gain_elevation']:.2f}, "
        f"relative {summary['correlation_relative_gain_elevation']:.2f}"
    )
    axis.grid(alpha=0.3)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return summary


class DistributionSamples:
    """Collects forecast and observed values so their distributions can be compared."""

    def __init__(self, methods: Sequence[str], seed: int = 0):
        """
        :param methods: Method names to track.
        :type methods: collections.abc.Sequence[str]
        :param seed: Seed for drawing one ensemble member per case.
        :type seed: int
        """
        self.methods = list(methods)
        self.rng = np.random.default_rng(seed)
        self.observed: list[np.ndarray] = []
        self.mean: dict[str, list[np.ndarray]] = {m: [] for m in self.methods}
        self.member: dict[str, list[np.ndarray]] = {m: [] for m in self.methods}

    def add(self, forecast_at_stations: xr.Dataset, obs: xr.DataArray) -> None:
        """Accumulate one initialisation.

        :param forecast_at_stations: Forecasts read at stations.
        :type forecast_at_stations: xarray.Dataset
        :param obs: Observations over ``(init_time, lead_time, station)``.
        :type obs: xarray.DataArray
        :returns: ``None``
        :rtype: None
        """
        truth = obs.values
        valid = np.isfinite(truth)
        self.observed.append(truth[valid])
        for method in self.methods:
            values = forecast_at_stations[method].values
            self.mean[method].append(values.mean(2)[valid])
            drawn = self.rng.integers(0, values.shape[2], size=valid.sum())
            flattened = values.transpose(0, 1, 3, 2)[valid]
            self.member[method].append(flattened[np.arange(flattened.shape[0]), drawn])

    def quantiles(self, levels: Sequence[float] | None = None) -> pd.DataFrame:
        """Quantiles of the observations and of each method.

        :param levels: Quantile levels; ``None`` uses a set weighted towards the tails.
        :type levels: collections.abc.Sequence[float] | None
        :returns: One row per level with the observed quantile and one column per method.
        :rtype: pandas.DataFrame
        """
        levels = list(levels) if levels is not None else [
            0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999
        ]
        observed = np.concatenate(self.observed)
        out = pd.DataFrame({"level": levels, "observed": np.quantile(observed, levels)})
        for method in self.methods:
            out[f"{method} mean"] = np.quantile(np.concatenate(self.mean[method]), levels)
            out[f"{method} member"] = np.quantile(np.concatenate(self.member[method]), levels)
        return out


def quantile_figure(quantiles: pd.DataFrame, methods: Sequence[str], labels: dict, path) -> None:
    """Plot forecast quantiles against observed quantiles.

    :param quantiles: Output of :meth:`DistributionSamples.quantiles`.
    :type quantiles: pandas.DataFrame
    :param methods: Method names to plot.
    :type methods: collections.abc.Sequence[str]
    :param labels: Display names keyed by method.
    :type labels: dict
    :param path: Destination image.
    :type path: str | pathlib.Path
    :returns: ``None``
    :rtype: None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"coarse": "tab:blue", "downscaled": "tab:red"}
    observed = quantiles["observed"] - 273.15
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    low, high = float(observed.min()) - 1, float(observed.max()) + 1
    for axis, kind in zip(ax, ("member", "mean")):
        axis.plot([low, high], [low, high], "k--", lw=1)
        for method in methods:
            axis.plot(
                observed, quantiles[f"{method} {kind}"] - 273.15, marker="o", ms=4,
                color=colours.get(method), label=labels.get(method, method),
            )
        for level, value in zip(quantiles.level, observed):
            if level in (0.001, 0.5, 0.999):
                axis.annotate(f"{level:g}", (value, value), fontsize=7, xytext=(3, -9), textcoords="offset points")
        axis.set_xlabel("observed quantile (degC)")
        axis.set_ylabel("forecast quantile (degC)")
        axis.set_title(f"distribution of a single {kind}" if kind == "member" else "distribution of the ensemble mean")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def distance_to_coast_km(mask: np.ndarray, spacing_km: float) -> np.ndarray:
    """Distance from each land cell to the nearest sea cell.

    :param mask: Boolean land mask on the fine grid.
    :type mask: numpy.ndarray
    :param spacing_km: Grid spacing in kilometres.
    :type spacing_km: float
    :returns: Distance in kilometres, zero over sea.
    :rtype: numpy.ndarray
    """
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(mask, sampling=(spacing_km, spacing_km)).astype("f4")


def maritime_influence(distance_km: np.ndarray, decay_km: float | None = None) -> np.ndarray:
    """Turn distance from the coast into a bounded index of maritime influence.

    Distance itself is a poor model input: most of a continental domain lies
    hundreds of kilometres inland, so after normalisation the first fifty
    kilometres, where the coastal effect actually varies, occupy a fraction of
    the channel's range. An exponential decay spends the range where the signal
    is, giving one at the shoreline and approaching zero inland.

    :param distance_km: Distance to the nearest sea cell [km].
    :type distance_km: numpy.ndarray
    :param decay_km: Decay scale; ``None`` uses
        :data:`~downscale.constants.COAST_DECAY_KM`.
    :type decay_km: float | None
    :returns: Index between zero and one.
    :rtype: numpy.ndarray
    """
    from .constants import COAST_DECAY_KM

    return np.exp(-np.asarray(distance_km, dtype="f8") / (decay_km or COAST_DECAY_KM)).astype("f4")


def station_distance_to_coast(
    table: pd.DataFrame, mask: np.ndarray, lat2d: np.ndarray, lon2d: np.ndarray, spacing_km: float
) -> np.ndarray:
    """Distance from each station to the nearest sea cell.

    :param table: Station table carrying ``latitude`` and ``longitude``.
    :type table: pandas.DataFrame
    :param mask: Boolean land mask on the fine grid.
    :type mask: numpy.ndarray
    :param lat2d: Fine-grid latitudes.
    :type lat2d: numpy.ndarray
    :param lon2d: Fine-grid longitudes.
    :type lon2d: numpy.ndarray
    :param spacing_km: Grid spacing in kilometres.
    :type spacing_km: float
    :returns: Distance in kilometres, one value per station.
    :rtype: numpy.ndarray
    """
    from .station_interp import CurvilinearInterpolator

    field = distance_to_coast_km(mask, spacing_km)
    interpolator = CurvilinearInterpolator.build(
        lat2d, lon2d, table.latitude.values, table.longitude.values, k=1, max_km=1e6
    )
    flat = field.ravel()[interpolator.flat_idx[:, 0]]
    return flat


def coast_profile(
    table: pd.DataFrame, distance: np.ndarray, baseline: str, improved: str,
    edges: Sequence[float] = (0, 10, 25, 50, 100, 200, 1e6),
) -> pd.DataFrame:
    """Summarise the gain from downscaling by distance from the coast.

    :param table: Output of :meth:`StationErrors.table`.
    :type table: pandas.DataFrame
    :param distance: Distance to the coast for each station [km].
    :type distance: numpy.ndarray
    :param baseline: Method treated as the reference.
    :type baseline: str
    :param improved: Method compared against it.
    :type improved: str
    :param edges: Band edges in kilometres.
    :type edges: collections.abc.Sequence[float]
    :returns: One row per band, including the mean station elevation, since distance
        from the coast and elevation are correlated.
    :rtype: pandas.DataFrame
    """
    with_distance = table.assign(distance_to_coast_km=distance)
    profile = gain_profile(with_distance, "distance_to_coast_km", edges, baseline, improved, unit="km")
    elevations = []
    for low, high in zip(edges[:-1], edges[1:]):
        chosen = (distance >= low) & (distance < high)
        if chosen.any():
            elevations.append(round(float(table.elevation.values[chosen].mean()), 0))
    profile.insert(2, "mean_elevation_m", elevations)
    return profile


def coast_figure(table: pd.DataFrame, distance: np.ndarray, profile: pd.DataFrame, path) -> dict:
    """Plot the gain from downscaling against distance from the coast.

    :param table: Output of :meth:`StationErrors.table`.
    :type table: pandas.DataFrame
    :param distance: Distance to the coast for each station [km].
    :type distance: numpy.ndarray
    :param profile: Output of :func:`coast_profile`.
    :type profile: pandas.DataFrame
    :param path: Destination image.
    :type path: str | pathlib.Path
    :returns: The correlation between gain and distance, and the gain within and beyond 25 km.
    :rtype: dict
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gain = (table["rmse_coarse"] - table["rmse_downscaled"]).values
    usable = np.isfinite(gain)
    inland = usable & (distance >= 25)
    coastal = usable & (distance < 25)
    summary = {
        "correlation_gain_distance": round(float(np.corrcoef(distance[usable], gain[usable])[0, 1]), 3),
        "mean_gain_within_25km_K": round(float(gain[coastal].mean()), 3) if coastal.any() else None,
        "mean_gain_beyond_25km_K": round(float(gain[inland].mean()), 3),
        "n_coastal_stations": int(coastal.sum()),
    }

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    scatter = ax[0].scatter(
        np.maximum(distance[usable], 1), gain[usable], c=table.elevation.values[usable],
        cmap="terrain", s=12, alpha=0.7,
    )
    ax[0].set_xscale("log")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].axvline(25, color="tab:red", lw=1, ls=":")
    ax[0].set_xlabel("distance from the coast (km)")
    ax[0].set_ylabel("RMSE reduction (K)")
    ax[0].set_title(f"gain against distance from the coast, correlation {summary['correlation_gain_distance']:.2f}")
    ax[0].grid(alpha=0.3)
    fig.colorbar(scatter, ax=ax[0], label="station elevation (m)")

    positions = np.arange(len(profile))
    ax[1].bar(positions, profile.mean_gain_K, color="tab:green", alpha=0.8)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(positions)
    ax[1].set_xticklabels(
        [f"{band}\nn={n}" for band, n in zip(profile.band, profile.n_stations)], fontsize=7
    )
    ax[1].set_ylabel("mean RMSE reduction (K)")
    ax[1].set_title("gain by distance band")
    ax[1].grid(alpha=0.3, axis="y")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return summary


def coast_transect(
    fields: dict[str, np.ndarray], distance_km: np.ndarray, mask: np.ndarray,
    edges: Sequence[float] = (0, 5, 10, 15, 20, 30, 40, 60, 80, 120, 200),
) -> pd.DataFrame:
    """Average each field across bands of distance from the coast, over time and day by day.

    A model that predicts the conditional mean reproduces the time-averaged
    profile but not any single day's, because the strength of the land-sea
    contrast varies. Comparing the two separates a model that has failed to
    learn the gradient from one that has learned all of it that is predictable.

    :param fields: Labels mapped to arrays of shape ``(time, y, x)``.
    :type fields: dict[str, numpy.ndarray]
    :param distance_km: Distance to the nearest sea cell [km].
    :type distance_km: numpy.ndarray
    :param mask: Land points to include.
    :type mask: numpy.ndarray
    :param edges: Band edges in kilometres.
    :type edges: collections.abc.Sequence[float]
    :returns: One row per band with, for each field, the time-mean value and the
        standard deviation across times of the band average.
    :rtype: pandas.DataFrame
    """
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        chosen = mask & (distance_km >= low) & (distance_km < high)
        if not chosen.any():
            continue
        row = {"band": f"{low:.0f}-{high:.0f} km", "distance_km": (low + high) / 2,
               "n_points": int(chosen.sum())}
        for name, values in fields.items():
            per_time = values[:, chosen].mean(axis=1)
            row[f"{name}_mean"] = round(float(per_time.mean()), 3)
            row[f"{name}_sd_over_time"] = round(float(per_time.std(ddof=1)), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def transect_figure(transect: pd.DataFrame, labels: Sequence[str], path) -> None:
    """Plot the cross-coast profiles and their day-to-day variability.

    :param transect: Output of :func:`coast_transect`.
    :type transect: pandas.DataFrame
    :param labels: Field names to plot.
    :type labels: collections.abc.Sequence[str]
    :param path: Destination image.
    :type path: str | pathlib.Path
    :returns: ``None``
    :rtype: None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name in labels:
        ax[0].plot(transect.distance_km, transect[f"{name}_mean"], marker="o", ms=4, label=name)
        ax[1].plot(transect.distance_km, transect[f"{name}_sd_over_time"], marker="o", ms=4, label=name)
    ax[0].set_title("time-mean correction across the coast (K)")
    ax[1].set_title("day-to-day variability of that correction (K)")
    for axis in ax:
        axis.set_xlabel("distance from the coast (km)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    ax[0].axhline(0, color="k", lw=0.8)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def comparison_figure(
    coarse: np.ndarray, downscaled: np.ndarray, target: np.ndarray,
    lat2d: np.ndarray, lon2d: np.ndarray, path, mask: np.ndarray | None = None,
    time_label: str = "",
) -> dict:
    """Show the coarse, downscaled and target fields with the two error maps.

    The three fields share one colour scale so they can be read against each
    other; the two error maps share a second, diverging one so the reduction is
    visible rather than inferred.

    :param coarse: Coarse field on the fine grid [K], shape ``(y, x)``.
    :type coarse: numpy.ndarray
    :param downscaled: Downscaled field [K], same shape.
    :type downscaled: numpy.ndarray
    :param target: Target field [K], same shape.
    :type target: numpy.ndarray
    :param lat2d: Fine-grid latitudes.
    :type lat2d: numpy.ndarray
    :param lon2d: Fine-grid longitudes.
    :type lon2d: numpy.ndarray
    :param path: Destination image.
    :type path: str | pathlib.Path
    :param mask: Land mask; sea points are left blank when given.
    :type mask: numpy.ndarray | None
    :param time_label: Text placed above the figure, usually the valid time.
    :type time_label: str
    :returns: The root mean squared error of each field against the target.
    :rtype: dict
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = [mask_sea(f, mask) for f in (coarse, downscaled, target)]
    errors = [fields[0] - fields[2], fields[1] - fields[2]]
    scores = {
        "coarse": float(np.sqrt(np.nanmean(errors[0] ** 2))),
        "downscaled": float(np.sqrt(np.nanmean(errors[1] ** 2))),
    }
    low, high = np.nanpercentile(fields[2], [1, 99])
    limit = float(np.nanpercentile(np.abs(errors[0]), 99))

    fig, ax = plt.subplots(1, 5, figsize=(24, 4.8), constrained_layout=True)
    titles = ["coarse field", "downscaled", "target"]
    for axis, field, title in zip(ax[:3], fields, titles):
        mesh = axis.pcolormesh(lon2d, lat2d, field - 273.15, cmap="RdYlBu_r", vmin=low - 273.15,
                               vmax=high - 273.15, shading="auto")
        axis.set_title(title)
        fig.colorbar(mesh, ax=axis, label="degC")
    for axis, error, name in zip(ax[3:], errors, ("coarse", "downscaled")):
        mesh = axis.pcolormesh(lon2d, lat2d, error, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
        axis.set_title(f"{name} minus target, RMSE {scores[name]:.2f} K")
        fig.colorbar(mesh, ax=axis, label="K")
    mid_lat = float(np.mean(lat2d))
    for axis in ax:
        axis.set_aspect(1 / np.cos(np.deg2rad(mid_lat)))
    if time_label:
        fig.suptitle(time_label)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return scores


def score_against_stations(
    values: dict[str, np.ndarray], observed: np.ndarray, elevation: np.ndarray,
    hours: np.ndarray, edges: Sequence[float],
) -> pd.DataFrame:
    """Score several fields against station observations, overall and by band.

    :param values: Field names mapped to arrays of shape ``(time, station)``.
    :type values: dict[str, numpy.ndarray]
    :param observed: Observations of shape ``(time, station)``.
    :type observed: numpy.ndarray
    :param elevation: Station elevations [m].
    :type elevation: numpy.ndarray
    :param hours: UTC hour of each time.
    :type hours: numpy.ndarray
    :param edges: Elevation band edges [m].
    :type edges: collections.abc.Sequence[float]
    :returns: One row per field and grouping with bias, RMSE and the sample count.
    :rtype: pandas.DataFrame
    """
    rows = []

    def add(name: str, grouping: str, band: str, chosen: np.ndarray) -> None:
        difference = (values[name] - observed)[chosen]
        difference = difference[np.isfinite(difference)]
        if difference.size == 0:
            return
        rows.append({
            "field": name, "grouping": grouping, "band": band, "n": int(difference.size),
            "bias_K": round(float(difference.mean()), 3),
            "rmse_K": round(float(np.sqrt((difference ** 2).mean())), 3),
        })

    everything = np.ones_like(observed, bool)
    for name in values:
        add(name, "all", "all stations", everything)
        for low, high in zip(edges[:-1], edges[1:]):
            band = (elevation >= low) & (elevation < high)
            add(name, "elevation", f"{int(low)}-{int(high)} m", np.broadcast_to(band, observed.shape))
        for hour in np.unique(hours):
            add(name, "hour", f"{int(hour):02d} UTC", (hours == hour)[:, None] & everything)
    return pd.DataFrame(rows)


def floor_figure(scores: pd.DataFrame, path, fields: Sequence[str] = ("coarse", "downscaled", "target"),
                 labels: dict | None = None) -> None:
    """Plot each field's error against stations, with the target shaded as the floor.

    :param scores: Output of :func:`score_against_stations`.
    :type scores: pandas.DataFrame
    :param path: Destination image.
    :type path: str | pathlib.Path
    :param fields: Field names to plot, the last being the floor.
    :type fields: collections.abc.Sequence[str]
    :param labels: Display names keyed by field.
    :type labels: dict | None
    :returns: ``None``
    :rtype: None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = labels or {
        "coarse": "coarse field", "downscaled": "coarse field + downscaler",
        "target": "target reanalysis (the floor)",
    }
    colours = {"coarse": "tab:blue", "downscaled": "tab:red", "target": "tab:green"}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for axis, grouping, title in (
        (ax[0], "elevation", "RMSE against stations by elevation (K)"),
        (ax[1], "hour", "RMSE against stations by time of day (K)"),
    ):
        subset = scores[scores.grouping == grouping]
        bands = list(dict.fromkeys(subset.band))
        positions = np.arange(len(bands))
        for name in fields:
            chosen = subset[subset.field == name].set_index("band").reindex(bands)
            axis.plot(positions, chosen.rmse_K.values, marker="o",
                      color=colours.get(name), label=labels.get(name, name))
            if name == fields[-1]:
                axis.fill_between(positions, 0, chosen.rmse_K.values,
                                  color=colours.get(name, "tab:green"), alpha=0.12)
        axis.set_xticks(positions)
        axis.set_xticklabels(bands, fontsize=7, rotation=20)
        axis.set_title(title)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.3)
    ax[0].set_ylabel("RMSE (K)")
    ax[0].legend(fontsize=8)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def validation_figure(
    fields: dict[str, np.ndarray], observed: np.ndarray, elevation: np.ndarray,
    hours: np.ndarray, path, edges: Sequence[float] | None = None, labels: dict | None = None,
) -> pd.DataFrame:
    """Four views of how each field performs against station observations.

    The panels answer different questions: where the error sits in terrain, when
    it sits in the day, how the errors are distributed, and whether the forecast
    reaches the observed extremes.

    :param fields: Field names mapped to values at stations, shape ``(time, station)``.
    :type fields: dict[str, numpy.ndarray]
    :param observed: Observations of the same shape.
    :type observed: numpy.ndarray
    :param elevation: Station elevations [m].
    :type elevation: numpy.ndarray
    :param hours: UTC hour of each time.
    :type hours: numpy.ndarray
    :param path: Destination image.
    :type path: str | pathlib.Path
    :param edges: Elevation band edges [m]; ``None`` uses the defaults.
    :type edges: collections.abc.Sequence[float] | None
    :param labels: Display names keyed by field.
    :type labels: dict | None
    :returns: The scores behind the first two panels.
    :rtype: pandas.DataFrame
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .constants import elevation_edges

    names = list(fields)
    labels = labels or {}
    colours = {"coarse": "tab:blue", "downscaled": "tab:red", "target": "tab:green"}
    band_edges = elevation_edges(list(edges) if edges is not None else None)
    scores = score_against_stations(fields, observed, elevation, hours, band_edges)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    by_elevation = scores[scores.grouping == "elevation"]
    bands = list(dict.fromkeys(by_elevation.band))
    positions = np.arange(len(bands))
    for name in names:
        chosen = by_elevation[by_elevation.field == name].set_index("band").reindex(bands)
        ax[0, 0].plot(positions, chosen.rmse_K.values, marker="o",
                      color=colours.get(name), label=labels.get(name, name))
        if name == names[-1] and len(names) > 2:
            ax[0, 0].fill_between(positions, 0, chosen.rmse_K.values,
                                  color=colours.get(name, "tab:green"), alpha=0.12)
    ax[0, 0].set_xticks(positions)
    ax[0, 0].set_xticklabels(bands, fontsize=7, rotation=20)
    ax[0, 0].set_ylabel("RMSE (K)")
    ax[0, 0].set_ylim(bottom=0)
    ax[0, 0].set_title("error by station elevation")
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)

    by_hour = scores[scores.grouping == "hour"]
    hour_bands = list(dict.fromkeys(by_hour.band))
    positions = np.arange(len(hour_bands))
    for name in names:
        chosen = by_hour[by_hour.field == name].set_index("band").reindex(hour_bands)
        ax[0, 1].plot(positions, chosen.rmse_K.values, marker="o", color=colours.get(name),
                      label=labels.get(name, name))
        ax[0, 1].plot(positions, chosen.bias_K.values, marker="s", ls="--", color=colours.get(name))
    ax[0, 1].axhline(0, color="k", lw=0.8)
    ax[0, 1].set_xticks(positions)
    ax[0, 1].set_xticklabels(hour_bands, fontsize=8)
    ax[0, 1].set_ylabel("K")
    ax[0, 1].set_title("error by time of day (solid: RMSE, dashed: bias)")
    ax[0, 1].grid(alpha=0.3)

    valid = np.isfinite(observed)
    for name in names:
        difference = (fields[name] - observed)[valid & np.isfinite(fields[name])]
        ax[1, 0].hist(difference, bins=80, range=(-8, 8), histtype="step", linewidth=1.6,
                      density=True, color=colours.get(name), label=labels.get(name, name))
    ax[1, 0].axvline(0, color="k", lw=0.8)
    ax[1, 0].set_xlabel("forecast minus observation (K)")
    ax[1, 0].set_ylabel("density")
    ax[1, 0].set_title("distribution of the error")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].legend(fontsize=8)

    levels = np.array([0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
    observed_quantiles = np.quantile(observed[valid], levels) - 273.15
    low, high = observed_quantiles.min() - 1, observed_quantiles.max() + 1
    ax[1, 1].plot([low, high], [low, high], "k--", lw=1)
    for name in names:
        usable = valid & np.isfinite(fields[name])
        ax[1, 1].plot(observed_quantiles, np.quantile(fields[name][usable], levels) - 273.15,
                      marker="o", ms=4, color=colours.get(name), label=labels.get(name, name))
    ax[1, 1].set_xlabel("observed quantile (degC)")
    ax[1, 1].set_ylabel("forecast quantile (degC)")
    ax[1, 1].set_title("does it reach the observed extremes?")
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].legend(fontsize=8)

    fig.savefig(path, dpi=120)
    plt.close(fig)
    return scores