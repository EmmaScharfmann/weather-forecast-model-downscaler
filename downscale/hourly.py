"""The hourly downscaler: a fine field at any hour from the two 6-hourly coarse fields around it.

The spatial model in :mod:`downscale.downscaler` turns a coarse field at one time into a fine field at that
time. A 6-hourly forecast leaves five hours unanswered in every window, and filling them by straight-line
interpolation smooths the daily cycle and misses its warmest hour. This model is given the coarse field at
both ends of a window, how far through the window the hour is, and the same fixed maps, and predicts the
correction to the straight-line value, so it can learn how the true hourly curve bends away from the line.

It is a separate network from the spatial one. Two extra input channels, the change over the window and the
position in it, mean it cannot reuse that checkpoint unchanged; :func:`init_from_spatial` starts it from the
spatial model's weights with the new channels switched off, so at the start of the window (position zero) it
reproduces the spatial model exactly and training only has to add the hours in between.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

import numpy as np
import torch

from .constants import TIME_CHANNELS, channels
from .downscaler import Config, Downscaler, DownscalerNet, time_features, to_kelvin

if TYPE_CHECKING:
    import xarray as xr

#: Length of the coarse forecast step the model fills in [hours].
WINDOW_HOURS = 6
EXTRA_CHANNELS = ["tendency", "offset"]


@dataclass
class HourlyConfig(Config):
    """Architecture settings of the hourly model: the spatial settings plus two input channels."""

    @property
    def channels(self) -> list[str]:
        """Input channel names, excluding noise.

        :returns: The spatial channels followed by the change over the window and the position in it.
        :rtype: list[str]
        """
        return [*channels(self.coast_distance), *EXTRA_CHANNELS]


class HourlyNet(DownscalerNet):
    """A :class:`~downscale.downscaler.DownscalerNet` whose noise can be supplied, so that a member keeps its
    detail from one hour to the next."""

    def forward(self, x: torch.Tensor, members: int = 1, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Predict normalised corrections.

        :param x: Input of shape ``(batch, channels, height, width)``.
        :type x: torch.Tensor
        :param members: Maps to produce per input.
        :type members: int
        :param noise: Noise of shape ``(batch * members, noise_channels, height, width)``, ordered like
            ``x.repeat_interleave(members, 0)``; ``None`` draws fresh noise.
        :type noise: torch.Tensor | None
        :returns: Corrections of shape ``(batch, members, height, width)``.
        :rtype: torch.Tensor
        """
        import torch.nn.functional as F

        batch, _, height, width = x.shape
        m = members if self.cfg.stochastic else 1
        x = x.repeat_interleave(m, 0)
        if self.cfg.stochastic:
            noise = self._noise(batch * m, height, width, x.device) if noise is None else noise
            x = torch.cat([x, noise], 1)
        multiple = 2 ** self.cfg.depth
        pad_h, pad_w = (-height) % multiple, (-width) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        y = self.unet(x)[..., :height, :width]
        return y.reshape(batch, m, height, width)


def temporal_interpolation(c0: np.ndarray, c1: np.ndarray, offset_hours: np.ndarray) -> np.ndarray:
    """The straight-line value between the two coarse fields.

    :param c0: Coarse field at the start of the window [K], ``(n, y, x)``.
    :type c0: numpy.ndarray
    :param c1: Coarse field at the end of the window [K], ``(n, y, x)``.
    :type c1: numpy.ndarray
    :param offset_hours: Hours after the start, ``(n,)``, from 0 to the window length.
    :type offset_hours: numpy.ndarray
    :returns: The interpolated field, ``(n, y, x)``.
    :rtype: numpy.ndarray
    """
    w = (np.asarray(offset_hours, "f4") / WINDOW_HOURS)[:, None, None]
    return ((1.0 - w) * c0 + w * c1).astype("f4")


def make_hourly_inputs(
    c0: np.ndarray, c1: np.ndarray, offset_hours: np.ndarray, valid_times, static_norm: np.ndarray, stats: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the normalised input and the baseline it corrects.

    :param c0: Coarse field at the start of the window [K], ``(n, y, x)``.
    :type c0: numpy.ndarray
    :param c1: Coarse field at the end of the window [K], ``(n, y, x)``.
    :type c1: numpy.ndarray
    :param offset_hours: Hours after the start of the window, ``(n,)``.
    :type offset_hours: numpy.ndarray
    :param valid_times: The times being predicted, one per sample.
    :type valid_times: numpy.ndarray | collections.abc.Sequence
    :param static_norm: Normalised static channels from :func:`~downscale.downscaler.static_stack`.
    :type static_norm: numpy.ndarray
    :param stats: Normalisation statistics, with ``era5_t2m``, ``tendency`` and ``residual``.
    :type stats: dict
    :returns: The input ``(n, channels, y, x)`` and the straight-line baseline [K] ``(n, y, x)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    n, height, width = c0.shape
    baseline = temporal_interpolation(c0, c1, offset_hours)
    main = ((baseline - stats["era5_t2m"]["mean"]) / stats["era5_t2m"]["std"]).astype("f4")[:, None]
    tendency = ((c1 - c0 - stats["tendency"]["mean"]) / stats["tendency"]["std"]).astype("f4")[:, None]
    position = (2.0 * np.asarray(offset_hours, "f4") / WINDOW_HOURS - 1.0)[:, None, None, None]
    position = np.broadcast_to(position, (n, 1, height, width))
    static = np.broadcast_to(static_norm[None], (n, *static_norm.shape))
    calendar = np.broadcast_to(time_features(valid_times)[:, :, None, None], (n, len(TIME_CHANNELS), height, width))
    return np.concatenate([main, static, calendar, tendency, position], 1).astype("f4"), baseline


def hourly_residual(baseline: np.ndarray, fine_k: np.ndarray, stats: dict) -> np.ndarray:
    """The normalised training target: the fine field minus the straight-line baseline.

    :param baseline: Straight-line coarse value [K].
    :type baseline: numpy.ndarray
    :param fine_k: The analysis at the hour [K].
    :type fine_k: numpy.ndarray
    :param stats: Normalisation statistics holding ``residual``.
    :type stats: dict
    :returns: Normalised residual.
    :rtype: numpy.ndarray
    """
    residual = stats["residual"]
    return ((fine_k - baseline - residual["mean"]) / residual["std"]).astype("f4")


def default_device() -> str:
    """CUDA, then Apple's MPS, then the CPU.

    :returns: Torch device name.
    :rtype: str
    """
    mps = getattr(torch.backends, "mps", None)
    return "cuda" if torch.cuda.is_available() else "mps" if mps and mps.is_available() else "cpu"


def safe_load(path) -> dict:
    """Read a checkpoint without unpickling arbitrary objects.

    Checkpoints hold tensors, plain containers and the numpy arrays of the static fields, so only the
    numpy types needed to rebuild those arrays are allowed; anything else fails instead of running.

    :param path: Checkpoint file.
    :type path: str | pathlib.Path
    :returns: The checkpoint.
    :rtype: dict
    """
    core = getattr(np, "_core", None) or np.core
    reconstruct = core.multiarray._reconstruct
    dtypes = {type(np.dtype(code)) for code in ("f2", "f4", "f8", "i1", "i2", "i4", "i8", "u1", "?")}
    allowed = [
        (reconstruct, "numpy.core.multiarray._reconstruct"),
        (reconstruct, "numpy._core.multiarray._reconstruct"),
        np.ndarray, np.dtype, *dtypes,
    ]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_spatial(path, device: str | None = None) -> Downscaler:
    """Read a spatial model's checkpoint with :func:`safe_load`.

    :param path: Checkpoint file.
    :type path: str | pathlib.Path
    :param device: Torch device; ``None`` picks :func:`default_device`.
    :type device: str | None
    :returns: Model ready for inference.
    :rtype: downscale.downscaler.Downscaler
    :raises ValueError: If the checkpoint's channels differ from this code's.
    """
    checkpoint = safe_load(path)
    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in checkpoint["config"].items() if k in known})
    if checkpoint["channels"] != cfg.channels:
        raise ValueError(f"checkpoint channels {checkpoint['channels']} differ from this code's {cfg.channels}")
    net = DownscalerNet(cfg)
    net.load_state_dict(checkpoint["state_dict"])
    return Downscaler(net, checkpoint["stats"], checkpoint["static"], device or default_device())


def window_samples(times) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find, for every hour, the two coarse steps of the window around it.

    Windows start at 00, 06, 12 and 18 UTC, the steps of the coarse forecast. An hour is kept only when
    both ends of its window are present.

    :param times: Hourly times, sorted.
    :type times: numpy.ndarray | collections.abc.Sequence
    :returns: The index of each kept hour, of the start and of the end of its window.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    t = np.asarray(times, "datetime64[h]")
    position = {value: i for i, value in enumerate(t.tolist())}
    hours, starts, ends = [], [], []
    for i, value in enumerate(t):
        offset = int((value - value.astype("datetime64[D]")) / np.timedelta64(1, "h")) % WINDOW_HOURS
        start = value - np.timedelta64(offset, "h")
        j, k = position.get(start.tolist()), position.get((start + np.timedelta64(WINDOW_HOURS, "h")).tolist())
        if j is not None and k is not None:
            hours.append(i), starts.append(j), ends.append(k)
    return np.array(hours, int), np.array(starts, int), np.array(ends, int)


def offsets_of(times) -> np.ndarray:
    """Hours since the start of the 6-hour window, from 0 to 5.

    :param times: Times.
    :type times: numpy.ndarray | collections.abc.Sequence
    :returns: The offsets.
    :rtype: numpy.ndarray
    """
    t = np.asarray(times, "datetime64[h]")
    return ((t - t.astype("datetime64[D]")) / np.timedelta64(1, "h")).astype(int) % WINDOW_HOURS


def hourly_leads(forecast: "xr.DataArray", step_hours: int = 1) -> "xr.DataArray":
    """Interpolate a forecast in time onto every hour between its first and last lead times.

    :param forecast: Forecast with a ``lead_time`` dimension.
    :type forecast: xarray.DataArray
    :param step_hours: Spacing of the new lead times.
    :type step_hours: int
    :returns: The straight-line forecast at every hour.
    :rtype: xarray.DataArray
    """
    leads = forecast["lead_time"].values
    hourly = np.arange(leads[0], leads[-1] + np.timedelta64(1, "h"), np.timedelta64(step_hours, "h"))
    return forecast.interp(lead_time=hourly.astype(leads.dtype))


def init_from_spatial(net: HourlyNet, spatial_checkpoint: str) -> None:
    """Start an hourly network from the spatial model's weights, with the new channels switched off.

    Every weight that has the same shape is copied. The first convolution is rebuilt: the spatial channels
    keep their weights, the two new channels start at zero, and the noise channels keep theirs.

    :param net: Freshly built hourly network, modified in place.
    :type net: HourlyNet
    :param spatial_checkpoint: Path of the spatial model's checkpoint.
    :type spatial_checkpoint: str
    :returns: ``None``
    :rtype: None
    """
    state = safe_load(spatial_checkpoint)["state_dict"]
    target = net.state_dict()
    n_spatial = len(channels(net.cfg.coast_distance))
    for name, weight in state.items():
        if weight.shape == target[name].shape:
            target[name] = weight.clone()
        elif name == "unet.inc.net.0.weight":
            rebuilt = torch.zeros_like(target[name])
            rebuilt[:, :n_spatial] = weight[:, :n_spatial]
            rebuilt[:, n_spatial + len(EXTRA_CHANNELS):] = weight[:, n_spatial:]
            target[name] = rebuilt
        else:
            raise ValueError(f"cannot initialise {name}: {tuple(weight.shape)} against {tuple(target[name].shape)}")
    net.load_state_dict(target)


class HourlyDownscaler(Downscaler):
    """A trained hourly model together with everything inference needs."""

    @classmethod
    def load(cls, path, device: str | None = None) -> "HourlyDownscaler":
        """Read a checkpoint written by :meth:`~downscale.downscaler.Downscaler.save`.

        :param path: Checkpoint file.
        :type path: str | pathlib.Path
        :param device: Torch device; ``None`` picks :func:`default_device`.
        :type device: str | None
        :returns: Model ready for inference.
        :rtype: HourlyDownscaler
        :raises ValueError: If the checkpoint's channels differ from this code's.
        """
        device = device or default_device()
        checkpoint = safe_load(path)
        known = {f.name for f in fields(HourlyConfig)}
        cfg = HourlyConfig(**{k: v for k, v in checkpoint["config"].items() if k in known})
        if checkpoint["channels"] != cfg.channels:
            raise ValueError(f"checkpoint channels {checkpoint['channels']} differ from this code's {cfg.channels}")
        net = HourlyNet(cfg)
        net.load_state_dict(checkpoint["state_dict"])
        return cls(net, checkpoint["stats"], checkpoint["static"], device)

    @torch.no_grad()
    def predict_hours(
        self, c0: np.ndarray, c1: np.ndarray, offset_hours: np.ndarray, valid_times, members: int = 4,
        batch: int = 8, noise: np.ndarray | None = None, seed: int = 0,
    ) -> np.ndarray:
        """Fine fields at some hours inside coarse windows.

        :param c0: Coarse field at the start of each window [K], ``(n, y, x)``, already on the fine grid.
        :type c0: numpy.ndarray
        :param c1: Coarse field at the end of each window [K], ``(n, y, x)``.
        :type c1: numpy.ndarray
        :param offset_hours: Hours after the start of each window, ``(n,)``.
        :type offset_hours: numpy.ndarray
        :param valid_times: The times predicted, one per sample.
        :type valid_times: numpy.ndarray | collections.abc.Sequence
        :param members: Maps per input.
        :type members: int
        :param batch: Samples processed at once.
        :type batch: int
        :param noise: Noise of shape ``(members, noise_channels, y, x)`` shared by every sample, so that member
            ``j`` keeps its detail from hour to hour; ``None`` draws it once from ``seed``.
        :type noise: numpy.ndarray | None
        :param seed: Seed for the shared noise when ``noise`` is not given.
        :type seed: int
        :returns: Fine fields [K], ``(n, members, y, x)``.
        :rtype: numpy.ndarray
        """
        n_members = members if self.net.cfg.stochastic else 1
        height, width = c0.shape[1:]
        shared = None
        if self.net.cfg.stochastic:
            if noise is None:
                torch.manual_seed(seed)
                shared = self.net._noise(n_members, height, width, self.device)
            else:
                shared = torch.from_numpy(noise).to(self.device)
        out = []
        offsets = np.asarray(offset_hours)
        for i in range(0, len(c0), batch):
            sl = slice(i, i + batch)
            x, baseline = make_hourly_inputs(c0[sl], c1[sl], offsets[sl], np.asarray(valid_times)[sl],
                                             self.static_norm, self.stats)
            x = torch.from_numpy(x).to(self.device)
            batch_noise = None if shared is None else shared[None].expand(x.shape[0], *shared.shape).reshape(
                x.shape[0] * n_members, *shared.shape[1:])
            pred = self.net(x, n_members, noise=batch_noise).cpu().numpy()
            out.append(to_kelvin(baseline, pred, self.stats).astype("f4"))
        return np.concatenate(out, 0)


def hourly_forecast(forecast: xr.DataArray) -> xr.DataArray:
    """The coarse forecast at every hour, from its first lead time to the hour before its last.

    The last lead time has no window after it, so the hours stop just before it.

    :param forecast: Forecast over ``(time, lead_time, ensemble, lat, lon)``.
    :type forecast: xarray.DataArray
    :returns: The interpolated forecast.
    :rtype: xarray.DataArray
    """
    return hourly_leads(forecast).isel(lead_time=slice(None, -1))


def downscale_hours(
    model: HourlyDownscaler, forecast: xr.DataArray, leads: np.ndarray, samples: int, batch: int, seed: int = 0
) -> xr.Dataset:
    """Run the hourly model over every member of a 6-hourly forecast.

    :param model: Hourly model.
    :type model: downscale.hourly.HourlyDownscaler
    :param forecast: 6-hourly forecast over ``(time, lead_time, ensemble, lat, lon)`` in kelvin.
    :type forecast: xarray.DataArray
    :param leads: Hourly lead times to produce.
    :type leads: numpy.ndarray
    :param samples: Fine maps per coarse member.
    :type samples: int
    :param batch: Hours processed at once.
    :type batch: int
    :param seed: Seed of the noise; member ``m`` uses ``seed + m`` at every hour.
    :type seed: int
    :returns: ``t2m_interp`` and ``t2m_ds`` laid out as :func:`downscale.inference.downscale_forecast` does.
    :rtype: xarray.Dataset
    :raises ValueError: If a lead time's window does not start on a forecast step.
    """
    from .inference import to_fine_grid

    coarse = to_fine_grid(forecast, model.lat, model.lon).astype("f4")
    steps = forecast["lead_time"].values
    init = forecast["time"].values
    n_init, _, n_ensemble = coarse.shape[:3]
    n_samples = samples if model.net.cfg.stochastic else 1
    valid = init[:, None] + leads[None, :]
    offsets = offsets_of(valid[0])
    if not all(offsets_of(v).tolist() == offsets.tolist() for v in valid):
        raise ValueError("initialisations at different hours of the 6-hour cycle")
    start_leads = leads - offsets.astype("timedelta64[h]").astype(leads.dtype)
    position = {lead: i for i, lead in enumerate(steps.tolist())}
    missing = [s for s in start_leads.tolist() if s not in position]
    if missing:
        raise ValueError(f"windows start between forecast steps, e.g. at lead {missing[0]}")
    window = np.timedelta64(WINDOW_HOURS, "h").astype(leads.dtype)
    starts = np.array([position[s] for s in start_leads.tolist()])
    ends = np.array([position[s] for s in (start_leads + window).tolist()])

    line = np.empty((n_init, len(leads), n_ensemble, *model.lat.shape), "f4")
    fine = np.empty((n_init, len(leads), n_ensemble, n_samples, *model.lat.shape), "f4")
    for i in range(n_init):
        for member in range(n_ensemble):
            c0, c1 = coarse[i, starts, member], coarse[i, ends, member]
            line[i, :, member] = temporal_interpolation(c0, c1, offsets)
            fine[i, :, member] = model.predict_hours(
                c0, c1, offsets, valid[i], members=samples, batch=batch, seed=seed + member
            )
    import xarray as xr

    return xr.Dataset(
        {
            "t2m_interp": (("time", "lead_time", "ensemble", "y", "x"), line, {"units": "K"}),
            "t2m_ds": (("time", "lead_time", "ensemble", "sample", "y", "x"), fine, {"units": "K"}),
            "land_sea_mask": (("y", "x"), model.static["land_sea_mask"].astype("f4")),
            "orography": (("y", "x"), model.static["cerra_orography"].astype("f4"), {"units": "m"}),
        },
        coords={
            "time": init,
            "lead_time": leads,
            "ensemble": forecast["ensemble"].values,
            "sample": np.arange(n_samples),
            "lat": (("y", "x"), model.lat),
            "lon": (("y", "x"), model.lon),
            "valid_time": (("time", "lead_time"), valid),
        },
    )


def config_dict(cfg: HourlyConfig) -> dict:
    """The settings of an hourly model as a plain dict, for reports.

    :param cfg: Architecture settings.
    :type cfg: HourlyConfig
    :returns: The fields of the configuration.
    :rtype: dict
    """
    return asdict(cfg)
