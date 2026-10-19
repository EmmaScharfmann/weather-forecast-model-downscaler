"""Figures for exploring a downscaled forecast interactively.

Unlike :mod:`downscale.diagnostics`, which writes report images from scripts, these
return the figure so it displays inline in a notebook.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

#: One colour per forecast throughout the figures.
COLOURS = {"AIFS-ENS": "tab:blue", "AIFS-ENS + downscaler": "tab:red", "CERRA": "tab:green",
           "CERRA (the floor)": "tab:green"}
FRANCE_EXTENT = (-5.5, 10.0, 41.2, 51.4)
#: Elevation colours, as on the Space: green lowlands, tan hills, white peaks; meant for 0 to 3000 m.
TERRAIN_CMAP = LinearSegmentedColormap.from_list("terrain", [
    (0, "#3b7d3b"), (0.1, "#8fbf62"), (0.25, "#e6dc9a"), (0.5, "#b8925e"), (0.75, "#8a6a52"), (1, "#f5f5f5")])


def native_cells(field, lat2d: np.ndarray, lon2d: np.ndarray, land: np.ndarray | None = None) -> np.ndarray:
    """Show a regular-grid field on a finer grid without smoothing it.

    Each fine point takes the value of the coarse cell it falls in, so the coarse cells stay
    visible as blocks, while a fine land-sea mask can hide the sea along the real coastline.

    :param field: Field over ``(lat, lon)`` with 1-D coordinates.
    :type field: xarray.DataArray
    :param lat2d: Fine grid latitudes, ``(y, x)``.
    :type lat2d: numpy.ndarray
    :param lon2d: Fine grid longitudes, ``(y, x)``, in the field's longitude convention.
    :type lon2d: numpy.ndarray
    :param land: Boolean land mask on the fine grid; sea points become NaN. ``None`` keeps them.
    :type land: numpy.ndarray | None
    :returns: Values over ``(y, x)``.
    :rtype: numpy.ndarray
    """
    lat, lon = np.asarray(field["lat"].values), np.asarray(field["lon"].values)
    i = np.abs(lat2d.ravel()[:, None] - lat[None]).argmin(1)
    j = np.abs(lon2d.ravel()[:, None] - lon[None]).argmin(1)
    out = np.asarray(field.transpose("lat", "lon").values)[i, j].reshape(lat2d.shape).astype("f8")
    return out if land is None else np.where(land, out, np.nan)


def _shrink(extent: Sequence[float]) -> float:
    """How much shorter a degree of longitude is than one of latitude at the centre of a map.

    :param extent: ``(lon_min, lon_max, lat_min, lat_max)``.
    :type extent: collections.abc.Sequence[float]
    :returns: The cosine of the central latitude.
    :rtype: float
    """
    return float(np.cos(np.deg2rad((extent[2] + extent[3]) / 2)))


def _geo_axes(fig, n: int, extent: Sequence[float], ncols: int | None = None):
    """Lay out ``n`` map axes in a grid, with coastlines when cartopy is available.

    ``n`` may also be a list of subplot specs, to place the maps in a custom layout.

    :param fig: Figure to draw on.
    :type fig: matplotlib.figure.Figure
    :param n: Number of maps, or the subplot specs to draw them in.
    :type n: int | list
    :param extent: ``(lon_min, lon_max, lat_min, lat_max)``.
    :type extent: collections.abc.Sequence[float]
    :param ncols: Maps per row; ``None`` puts them all on one row.
    :type ncols: int | None
    :returns: The axes and the keyword arguments to pass to ``pcolormesh``.
    :rtype: tuple[list, dict]
    """
    if isinstance(n, int):
        ncols = ncols or n
        slots = [(-(-n // ncols), ncols, i + 1) for i in range(n)]
    else:
        slots = [(spec,) for spec in n]
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        axes = [fig.add_subplot(*slot) for slot in slots]
        for axis in axes:
            axis.set_xlim(extent[0], extent[1])
            axis.set_ylim(extent[2], extent[3])
            axis.set_aspect(1 / _shrink(extent))
        return axes, {}
    axes = [fig.add_subplot(*slot, projection=ccrs.PlateCarree()) for slot in slots]
    for axis in axes:
        axis.set_extent(extent, crs=ccrs.PlateCarree())
        # Plate carrée draws a degree of longitude as long as one of latitude, which widens France
        # by half; shortening it to its true length at the centre of the map undoes that.
        axis.set_aspect(1 / _shrink(extent))
        axis.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="#333333")
        axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, edgecolor="#555555")
    return axes, {"transform": ccrs.PlateCarree()}


def maps(
    panels: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray]], *, cmap: str = "RdYlBu_r",
    vmin: float | None = None, vmax: float | None = None, label: str = "°C",
    extent: Sequence[float] = FRANCE_EXTENT, title: str | None = None, width: float = 5.2,
):
    """Draw fields side by side on one colour scale.

    :param panels: ``(title, lon, lat, values)`` per map. Coordinates are 1-D for a regular
        grid, drawn as its native cells, or 2-D for the fine grid.
    :type panels: collections.abc.Sequence[tuple]
    :param cmap: Colour map.
    :type cmap: str
    :param vmin: Lower end of the colour scale; ``None`` takes the 1st percentile.
    :type vmin: float | None
    :param vmax: Upper end; ``None`` takes the 99th percentile.
    :type vmax: float | None
    :param label: Colour bar label.
    :type label: str
    :param extent: ``(lon_min, lon_max, lat_min, lat_max)``.
    :type extent: collections.abc.Sequence[float]
    :param title: Figure title.
    :type title: str | None
    :param width: Width of one map [inches].
    :type width: float
    :returns: The figure.
    :rtype: matplotlib.figure.Figure
    """
    values = np.concatenate([np.ravel(p[3]) for p in panels])
    values = values[np.isfinite(values)]
    vmin = float(np.percentile(values, 1)) if vmin is None else vmin
    vmax = float(np.percentile(values, 99)) if vmax is None else vmax
    fig = plt.figure(layout="constrained")
    axes, kwargs = _geo_axes(fig, len(panels), extent)
    aspect = (extent[3] - extent[2]) / ((extent[1] - extent[0]) * _shrink(extent))
    fig.set_size_inches(width * len(panels) + 0.9, width * aspect + (0.6 if title else 0.3))
    for axis, (name, lon, lat, field) in zip(axes, panels):
        mesh = axis.pcolormesh(lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", **kwargs)
        axis.set_title(name, fontsize=11)
    fig.colorbar(mesh, ax=axes, label=label, shrink=0.85, pad=0.02)
    if title:
        fig.suptitle(title, fontsize=12)
    return fig


def map_grid(
    panels: Sequence[dict], *, ncols: int = 2, extent: Sequence[float] = FRANCE_EXTENT,
    title: str | None = None, width: float = 6.0,
):
    """Draw maps in a grid, each with its own colour scale and colour bar.

    :param panels: One dictionary per map with ``title``, ``lon``, ``lat`` and ``values``, and
        optionally ``cmap``, ``vmin``, ``vmax`` and ``label`` (the colour bar's). Coordinates are
        1-D for a regular grid, drawn as its native cells, or 2-D for the fine grid.
    :type panels: collections.abc.Sequence[dict]
    :param ncols: Maps per row.
    :type ncols: int
    :param extent: ``(lon_min, lon_max, lat_min, lat_max)``.
    :type extent: collections.abc.Sequence[float]
    :param title: Figure title.
    :type title: str | None
    :param width: Width of one map [inches].
    :type width: float
    :returns: The figure.
    :rtype: matplotlib.figure.Figure
    """
    fig = plt.figure(layout="constrained")
    axes, kwargs = _geo_axes(fig, len(panels), extent, ncols)
    aspect = (extent[3] - extent[2]) / ((extent[1] - extent[0]) * _shrink(extent))
    nrows = -(-len(panels) // ncols)
    fig.set_size_inches((width + 1.0) * ncols, (width * aspect + 0.4) * nrows + (0.4 if title else 0))
    for axis, panel in zip(axes, panels):
        values = np.asarray(panel["values"], "f8")
        finite = values[np.isfinite(values)]
        vmin = panel.get("vmin", float(np.percentile(finite, 1)))
        vmax = panel.get("vmax", float(np.percentile(finite, 99)))
        mesh = axis.pcolormesh(panel["lon"], panel["lat"], values, cmap=panel.get("cmap", "RdYlBu_r"),
                               vmin=vmin, vmax=vmax, shading="auto", **kwargs)
        axis.set_title(panel["title"], fontsize=12)
        fig.colorbar(mesh, ax=axis, label=panel.get("label", ""), shrink=0.9, pad=0.02)
    if title:
        fig.suptitle(title, fontsize=13)
    return fig


def lead_figure(
    tables: dict[str, pd.DataFrame], metrics: Sequence[str] = ("rmse_K", "crps_K"),
    references: Sequence[str] = (), title: str | None = None,
):
    """Plot scores against lead time.

    :param tables: Forecast names mapped to outputs of :func:`downscale.evaluation.lead_scores`.
    :type tables: dict[str, pandas.DataFrame]
    :param metrics: Columns to plot, one panel each.
    :type metrics: collections.abc.Sequence[str]
    :param references: Names in ``tables`` that are single fields, such as a reanalysis read at the
        same valid times: drawn dashed, and only on the RMSE and bias panels.
    :type references: collections.abc.Sequence[str]
    :param title: Figure title.
    :type title: str | None
    :returns: The figure.
    :rtype: matplotlib.figure.Figure
    """
    names = {"rmse_K": "RMSE of the ensemble mean (K)", "crps_K": "fair CRPS (K)", "bias_K": "bias (K)",
             "spread_skill": "spread / RMSE", "spread_K": "ensemble spread (K)"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.2), layout="constrained", squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        for name, table in tables.items():
            reference = name in references
            if reference and metric not in ("rmse_K", "bias_K"):
                continue
            axis.plot(table.lead_hours / 24, table[metric], marker="o", ms=3, color=COLOURS.get(name, "tab:green"),
                      ls="--" if reference else "-", label=name)
        if metric in ("bias_K",):
            axis.axhline(0, color="k", lw=0.8)
        if metric == "spread_skill":
            axis.axhline(1, color="k", lw=0.8)
        axis.set_xlabel("lead time (days)")
        axis.set_ylabel(names.get(metric, metric))
        axis.grid(alpha=0.3)
        if metric in ("rmse_K", "crps_K", "spread_K"):
            axis.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=9)
    if title:
        fig.suptitle(title, fontsize=12)
    return fig


def station_maps(
    lon: np.ndarray, lat: np.ndarray, rmse: dict[str, np.ndarray], *, baseline: str = "AIFS-ENS",
    improved: str = "AIFS-ENS + downscaler", extent: Sequence[float] = FRANCE_EXTENT, title: str | None = None,
    width: float = 5.2,
):
    """Map the RMSE of two forecasts at each station, and how much the second lowers it.

    The two RMSE maps sit side by side on one colour scale; the map below them shows the
    reduction, positive where ``improved`` beats ``baseline``. Larger values are drawn on top.

    :param lon: Station longitudes.
    :type lon: numpy.ndarray
    :param lat: Station latitudes.
    :type lat: numpy.ndarray
    :param rmse: RMSE per station [K], keyed by forecast name, as from
        :func:`downscale.evaluation.station_rmse`.
    :type rmse: dict[str, numpy.ndarray]
    :param baseline: Forecast the reduction is measured from.
    :type baseline: str
    :param improved: Forecast compared against it.
    :type improved: str
    :param extent: ``(lon_min, lon_max, lat_min, lat_max)``.
    :type extent: collections.abc.Sequence[float]
    :param title: Figure title.
    :type title: str | None
    :param width: Width of one map [inches].
    :type width: float
    :returns: The figure.
    :rtype: matplotlib.figure.Figure
    """
    lon, lat = np.asarray(lon), np.asarray(lat)
    gain = rmse[baseline] - rmse[improved]
    both = np.concatenate([rmse[baseline], rmse[improved]])
    low, high = np.nanpercentile(both, [2, 98])
    gain_cmap = LinearSegmentedColormap.from_list(
        "gain", ["#8c2d04", "#e6705a", "#f7f7f7", "#5aae61", "#00441b"])
    improved_share = 100 * np.mean(gain[np.isfinite(gain)] > 0)
    panels = [
        (rmse[baseline], f"{baseline}\nmedian RMSE {np.nanmedian(rmse[baseline]):.2f} K", "YlOrRd", low, high,
         "RMSE (K)"),
        (rmse[improved], f"{improved}\nmedian RMSE {np.nanmedian(rmse[improved]):.2f} K", "YlOrRd", low, high,
         "RMSE (K)"),
        (gain, f"RMSE reduction\n{improved_share:.0f}% of stations improve", gain_cmap, -1.5, 1.5,
         "RMSE reduction (K)"),
    ]
    fig = plt.figure(layout="constrained")
    grid = fig.add_gridspec(2, 4)
    axes, kwargs = _geo_axes(fig, [grid[0, :2], grid[0, 2:], grid[1, 1:3]], extent)
    aspect = (extent[3] - extent[2]) / ((extent[1] - extent[0]) * _shrink(extent))
    fig.set_size_inches((width + 1.0) * 2, (width * aspect + 0.6) * 2 + (0.3 if title else 0))
    for axis, (values, name, cmap, vmin, vmax, label) in zip(axes, panels):
        ok = np.isfinite(values)
        order = np.argsort(np.abs(values[ok]))
        points = axis.scatter(lon[ok][order], lat[ok][order], c=values[ok][order], cmap=cmap, vmin=vmin, vmax=vmax,
                              s=14, edgecolors="#444444", linewidths=0.3, zorder=3, **kwargs)
        axis.set_title(name, fontsize=11)
        fig.colorbar(points, ax=axis, label=label, shrink=0.85, pad=0.02)
    if title:
        fig.suptitle(title, fontsize=12)
    return fig


def elevation_figure(table: pd.DataFrame, title: str | None = None):
    """Bar chart of RMSE per elevation band, one bar per field.

    :param table: Output of :func:`downscale.evaluation.elevation_scores`.
    :type table: pandas.DataFrame
    :param title: Figure title.
    :type title: str | None
    :returns: The figure.
    :rtype: matplotlib.figure.Figure
    """
    bands = list(dict.fromkeys(table.band))
    fields = list(dict.fromkeys(table.field))
    width = 0.8 / len(fields)
    counts = table.groupby("band", sort=False).n.max()
    fig, axis = plt.subplots(figsize=(1.4 * len(bands) + 3, 4.2), layout="constrained")
    for i, name in enumerate(fields):
        chosen = table[table.field == name].set_index("band").reindex(bands)
        axis.bar(np.arange(len(bands)) + (i - (len(fields) - 1) / 2) * width, chosen.rmse_K, width,
                 color=COLOURS.get(name), label=name)
    axis.set_xticks(np.arange(len(bands)))
    axis.set_xticklabels([f"{b}\n(n={counts[b]:,})" for b in bands], fontsize=8)
    axis.set_ylabel("RMSE against stations (K)")
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=9)
    if title:
        axis.set_title(title, fontsize=12)
    return fig


def plume_figure(
    valid: np.ndarray, series: dict[str, np.ndarray], observed: np.ndarray | None = None, title: str | None = None,
):
    """Every member of several ensembles at one point, against time.

    :param valid: Valid times.
    :type valid: numpy.ndarray
    :param series: Forecast names mapped to values over ``(member, time)`` in °C.
    :type series: dict[str, numpy.ndarray]
    :param observed: Observed values over ``time`` in °C, drawn when given.
    :type observed: numpy.ndarray | None
    :param title: Figure title.
    :type title: str | None
    :returns: The figure.
    :rtype: matplotlib.figure.Figure
    """
    fig, axis = plt.subplots(figsize=(11, 4), layout="constrained")
    for name, values in series.items():
        colour = COLOURS.get(name)
        axis.plot(valid, values.T, color=colour, lw=0.5, alpha=0.25)
        axis.plot(valid, values.mean(0), color=colour, lw=2.2, label=f"{name} (mean of {values.shape[0]} members)")
    if observed is not None:
        axis.plot(valid, observed, "k.", ms=6, label="observed")
    axis.set_ylabel("2 m temperature (°C)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=9)
    if title:
        axis.set_title(title, fontsize=12)
    fig.autofmt_xdate()
    return fig
