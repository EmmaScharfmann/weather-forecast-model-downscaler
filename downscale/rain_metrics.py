"""Rainfall metrics grouped by the question they answer, with missing references.

Every function takes forecasts shaped ``(time, member, *space)`` and a reference
shaped ``(time, *space)`` where NaN marks unobserved points. Deterministic
forecasts have one member. Which output each metric sees matters:

* amount (bias, MAE, RMSE) uses the ensemble mean, the usual point forecast;
* CRPS uses all members;
* occurrence, structure and intensity use individual members, because an
  ensemble mean is smooth by construction and would be judged on its blur.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


THRESHOLDS = (0.1, 1.0, 5.0, 10.0)  # mm/h
FSS_SIZES = (1, 3, 5, 9, 17, 33)  # cells; 0.1° ≈ 8-11 km over France
QUANTILES = (0.9, 0.99, 0.999)


def _valid(reference):
    return np.isfinite(reference)


def amount(forecast, reference):
    """Bias, MAE and RMSE of the ensemble mean, in mm."""
    mean = forecast.mean(1)
    ok = _valid(reference)
    error = mean[ok] - reference[ok]
    return {
        "bias_mm": float(error.mean()),
        "mae_mm": float(np.abs(error).mean()),
        "rmse_mm": float(np.sqrt((error**2).mean())),
    }


def crps(forecast, reference):
    """Fair CRPS (estimates the generator) and finite-ensemble CRPS, in mm.

    One member: both reduce to the MAE of that member.
    """
    ok = _valid(reference)
    members = forecast.shape[1]
    skill = np.abs(forecast - reference[:, None]).mean(1)[ok].mean()
    if members == 1:
        return {"crps_mm": float(skill), "ensemble_crps_mm": float(skill)}
    ordered = np.sort(forecast, axis=1)
    weights = (2 * np.arange(members) - members + 1).reshape(1, -1, *[1] * (forecast.ndim - 2))
    spread = (weights * ordered).sum(1) / (members * members)  # = E|X-X'| / 2
    ensemble = skill - spread[ok].mean()
    fair = skill - spread[ok].mean() * members / (members - 1)
    return {"crps_mm": float(fair), "ensemble_crps_mm": float(ensemble)}


def occurrence(forecast, reference, thresholds=THRESHOLDS):
    """CSI and frequency bias above each threshold, counted over all members."""
    ok = np.broadcast_to(_valid(reference)[:, None], forecast.shape)
    result = {}
    for t in thresholds:
        observed = np.broadcast_to((reference >= t)[:, None], forecast.shape)
        predicted = forecast >= t
        hits = np.sum(predicted & observed & ok)
        misses = np.sum(~predicted & observed & ok)
        false = np.sum(predicted & ~observed & ok)
        union = hits + misses + false
        result[f"csi_{t:g}mm"] = float(hits / union) if union else None
        result[f"frequency_bias_{t:g}mm"] = float((hits + false) / (hits + misses)) if hits + misses else None
        result[f"observed_fraction_{t:g}mm"] = float(np.mean(reference[_valid(reference)] >= t))
    return result


def _fractions(binary, valid, size):
    """Fraction of rainy observed cells in a size x size window, per hour."""
    # Smooth the two spatial axes only, never across hours or members.
    size_per_axis = [1] * binary.ndim
    for axis in range(binary.ndim - 2, binary.ndim):
        size_per_axis[axis] = size
    count = uniform_filter(valid.astype("float64"), size_per_axis, mode="constant")
    rain = uniform_filter((binary & valid).astype("float64"), size_per_axis, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, rain / count, np.nan)


def structure(forecast, reference, thresholds=THRESHOLDS, sizes=FSS_SIZES, cell_km=10.0):
    """Fractions skill score against neighbourhood size, and the smallest useful scale.

    Fractions only count observed cells, so unobserved areas neither help nor hurt.
    "Useful" follows Roberts and Lean (2008): FSS >= 0.5 + f/2, f the observed
    rainy fraction.
    """
    valid = np.broadcast_to(_valid(reference)[:, None], forecast.shape)
    observed = np.broadcast_to(np.nan_to_num(reference, nan=0.0)[:, None], forecast.shape)
    result = {"sizes_cells": list(sizes), "sizes_km": [s * cell_km for s in sizes]}
    for t in thresholds:
        f = float(np.mean(reference[_valid(reference)] >= t))
        curve = []
        for size in sizes:
            a = _fractions(forecast >= t, valid, size)
            b = _fractions(observed >= t, valid, size)
            ok = valid & np.isfinite(a)
            denominator = np.mean(a[ok] ** 2 + b[ok] ** 2)
            curve.append(float(1 - np.mean((a[ok] - b[ok]) ** 2) / denominator) if denominator else None)
        useful = 0.5 + f / 2
        first = next((s for s, v in zip(sizes, curve) if v is not None and v >= useful), None)
        result[f"fss_{t:g}mm"] = curve
        result[f"useful_fss_{t:g}mm"] = useful
        result[f"useful_scale_km_{t:g}mm"] = None if first is None else first * cell_km
    return result


def intensity(forecast, reference, quantiles=QUANTILES):
    """Wet-hour frequency and high quantiles of all values, wherever they fall."""
    ok = _valid(reference)
    values = forecast[np.broadcast_to(ok[:, None], forecast.shape)]
    truth = reference[ok]
    result = {
        "wet_fraction_0.1mm": float(np.mean(values >= 0.1)),
        "reference_wet_fraction_0.1mm": float(np.mean(truth >= 0.1)),
    }
    for q in quantiles:
        result[f"q{q * 100:g}_mm"] = float(np.quantile(values, q))
        result[f"reference_q{q * 100:g}_mm"] = float(np.quantile(truth, q))
    return result


def qq(forecast, reference, quantiles=np.linspace(0.9, 0.9999, 60)):
    """Quantile pairs for a Q-Q plot of the wet tail."""
    ok = _valid(reference)
    values = forecast[np.broadcast_to(ok[:, None], forecast.shape)]
    return {
        "quantiles": quantiles.tolist(),
        "forecast": np.quantile(values, quantiles).tolist(),
        "reference": np.quantile(reference[ok], quantiles).tolist(),
    }


def complete_box(reference, min_size=32):
    """Largest square observed at every kept hour, for spectra: (row, col, size, hours).

    Empty hours are always dropped. If partly observed hours leave no square of
    ``min_size``, only the best-observed hours are kept. Returns None if none fits.
    """
    observed = _valid(reference).mean((1, 2))
    for hours in (observed > 0, observed >= observed.max() - 0.01):
        ok = _valid(reference[hours]).all(0)
        h, w = ok.shape
        best = np.zeros((h + 1, w + 1), dtype=int)
        top = (0, 0, 0)
        for i in range(h):
            for j in range(w):
                if ok[i, j]:
                    best[i + 1, j + 1] = 1 + min(best[i, j], best[i + 1, j], best[i, j + 1])
                    if best[i + 1, j + 1] > top[2]:
                        top = (i - best[i + 1, j + 1] + 1, j - best[i + 1, j + 1] + 1, best[i + 1, j + 1])
        if top[2] >= min_size:
            return (*top, hours)
    return None


def spectrum(fields, cell_km=10.0):
    """Radially averaged spatial power spectrum of (n, size, size) fields.

    A Hann window limits edge leakage; each field's mean is removed. Returns
    wavelengths in km and the mean power, comparable between sources on the
    same box and hours.
    """
    n, size, _ = fields.shape
    window = np.outer(np.hanning(size), np.hanning(size))
    centred = (fields - fields.mean((1, 2), keepdims=True)) * window
    power = (np.abs(np.fft.fft2(centred)) ** 2).mean(0)
    k = np.fft.fftfreq(size)
    radius = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    bins = np.arange(1, size // 2 + 1) / size
    index = np.digitize(radius, bins - 0.5 / size)
    values = np.array([power[index == i].mean() for i in range(1, len(bins) + 1)])
    return {"wavelength_km": (cell_km / bins).tolist(), "power": values.tolist()}


def all_scores(forecast, reference, gridded=True, cell_km=10.0):
    """Every metric of the blog's table for one forecast against one reference."""
    result = {
        "amount": amount(forecast, reference),
        "probability": crps(forecast, reference),
        "occurrence": occurrence(forecast, reference),
        "intensity": intensity(forecast, reference),
    }
    if gridded:
        result["structure"] = structure(forecast, reference, cell_km=cell_km)
    return result
