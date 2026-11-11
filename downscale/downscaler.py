"""The downscaling model: coarse field plus fine terrain to fine field.

The network predicts the correction to add to the interpolated coarse field,
and takes noise channels so that repeated passes give different plausible maps.
With the fair CRPS loss those maps form an ensemble whose spread is trained to
be realistic; with the mean squared error loss the network is deterministic and
produces the conditional mean, which is smoother than reality.

A saved checkpoint carries the weights, the normalisation statistics and the
fine-grid terrain, so inference needs no other file.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import COAST_CHANNEL, STATIC_CHANNELS, TIME_CHANNELS, channels


class _Block(nn.Module):
    """Two convolutions with group normalisation and a smooth activation."""

    def __init__(self, channels_in: int, channels_out: int):
        """
        :param channels_in: Input channels.
        :type channels_in: int
        :param channels_out: Output channels.
        :type channels_out: int
        """
        super().__init__()
        groups = min(8, channels_out)
        self.net = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1), nn.GroupNorm(groups, channels_out), nn.SiLU(),
            nn.Conv2d(channels_out, channels_out, 3, padding=1), nn.GroupNorm(groups, channels_out), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input of shape ``(batch, channels_in, height, width)``.
        :type x: torch.Tensor
        :returns: Output of shape ``(batch, channels_out, height, width)``.
        :rtype: torch.Tensor
        """
        return self.net(x)


class UNet(nn.Module):
    """Encoder-decoder with skip connections, zero-initialised output layer."""

    def __init__(
        self, channels_in: int, channels_out: int = 1, base: int = 32, depth: int = 3,
        upsample: str = "transpose",
    ):
        """
        :param channels_in: Input channels.
        :type channels_in: int
        :param channels_out: Output channels.
        :type channels_out: int
        :param base: Channel count at full resolution.
        :type base: int
        :param depth: Number of downsampling steps.
        :type depth: int
        :param upsample: ``"transpose"`` for strided transposed convolutions, or ``"bilinear"``
            for interpolation followed by a convolution, which avoids grid-scale artefacts.
        :type upsample: str
        :raises ValueError: If the upsampling method is unknown.
        """
        super().__init__()
        if upsample not in ("transpose", "bilinear"):
            raise ValueError("upsample must be 'transpose' or 'bilinear'")
        widths = [base * 2 ** i for i in range(depth + 1)]
        self.inc = _Block(channels_in, widths[0])
        self.down = nn.ModuleList([_Block(widths[i], widths[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) if upsample == "transpose"
            else nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(widths[i + 1], widths[i], 3, padding=1),
            )
            for i in reversed(range(depth))
        ])
        self.dec = nn.ModuleList([_Block(2 * widths[i], widths[i]) for i in reversed(range(depth))])
        self.out = nn.Conv2d(widths[0], channels_out, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input of shape ``(batch, channels_in, height, width)``.
        :type x: torch.Tensor
        :returns: Output of shape ``(batch, channels_out, height, width)``.
        :rtype: torch.Tensor
        """
        skips = [self.inc(x)]
        for block in self.down:
            skips.append(block(F.avg_pool2d(skips[-1], 2)))
        h = skips.pop()
        for up, dec in zip(self.up, self.dec):
            h = dec(torch.cat([up(h), skips.pop()], 1))
        return self.out(h)


@dataclass
class Config:
    """Model architecture settings.

    The defaults reproduce the original architecture so that existing
    checkpoints load unchanged.
    """

    loss: str = "crps"
    base: int = 32
    depth: int = 3
    noise_channels: int = 4
    upsample: str = "transpose"
    noise_fine_fraction: float = 0.5
    noise_coarse_factor: int | list[int] = 8
    coast_distance: bool = False

    @property
    def channels(self) -> list[str]:
        """Input channel names this configuration implies.

        :returns: Channel names in input order, excluding noise.
        :rtype: list[str]
        """
        return channels(self.coast_distance)

    @property
    def stochastic(self) -> bool:
        """Whether repeated passes give different maps.

        :returns: ``True`` for a CRPS-trained model with noise channels.
        :rtype: bool
        """
        return self.loss == "crps" and self.noise_channels > 0


class DownscalerNet(nn.Module):
    """A :class:`UNet` with noise channels, producing several maps per input."""

    def __init__(self, cfg: Config):
        """
        :param cfg: Architecture settings.
        :type cfg: Config
        """
        super().__init__()
        self.cfg = cfg
        n_noise = cfg.noise_channels if cfg.stochastic else 0
        self.unet = UNet(len(cfg.channels) + n_noise, 1, cfg.base, cfg.depth, cfg.upsample)

    def _noise(self, n: int, height: int, width: int, device) -> torch.Tensor:
        """Draw noise channels across one or several spatial scales.

        Channels drawn at reduced resolution and upsampled make members differ
        coherently over tens of kilometres; channels drawn at full resolution
        let them differ pixel by pixel. Giving ``noise_coarse_factor`` a list
        spreads the ensemble's variability over several scales instead of
        concentrating it at one, which otherwise shows up as a peak in the
        power spectrum at that scale.

        :param n: Batch size.
        :type n: int
        :param height: Grid height.
        :type height: int
        :param width: Grid width.
        :type width: int
        :param device: Torch device.
        :type device: torch.device | str
        :returns: Noise of shape ``(n, noise_channels, height, width)``.
        :rtype: torch.Tensor
        """
        total = self.cfg.noise_channels
        n_fine = int(round(total * self.cfg.noise_fine_fraction))
        n_coarse = total - n_fine
        parts = []
        if n_fine:
            parts.append(torch.randn(n, n_fine, height, width, device=device))
        if n_coarse:
            factors = self.cfg.noise_coarse_factor
            factors = [factors] if isinstance(factors, int) else list(factors)
            per_factor = [n_coarse // len(factors)] * len(factors)
            for i in range(n_coarse - sum(per_factor)):
                per_factor[i] += 1
            for factor, count in zip(factors, per_factor):
                if not count:
                    continue
                factor = max(1, int(factor))
                coarse = torch.randn(
                    n, count, math.ceil(height / factor), math.ceil(width / factor), device=device
                )
                parts.append(
                    F.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)
                )
        return torch.cat(parts, 1)

    def forward(self, x: torch.Tensor, members: int = 1) -> torch.Tensor:
        """Predict normalised corrections.

        :param x: Input of shape ``(batch, channels, height, width)``.
        :type x: torch.Tensor
        :param members: Maps to produce per input; ignored by a deterministic model.
        :type members: int
        :returns: Corrections of shape ``(batch, members, height, width)``.
        :rtype: torch.Tensor
        """
        batch, _, height, width = x.shape
        m = members if self.cfg.stochastic else 1
        x = x.repeat_interleave(m, 0)
        if self.cfg.stochastic:
            x = torch.cat([x, self._noise(batch * m, height, width, x.device)], 1)
        multiple = 2 ** self.cfg.depth
        pad_h, pad_w = (-height) % multiple, (-width) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        y = self.unet(x)[..., :height, :width]
        return y.reshape(batch, m, height, width)


def fair_crps(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Fair continuous ranked probability score, averaged over the retained pixels.

    :param pred: Ensemble of shape ``(batch, members, height, width)``.
    :type pred: torch.Tensor
    :param target: Truth of shape ``(batch, height, width)``.
    :type target: torch.Tensor
    :param mask: Boolean grid of shape ``(height, width)``; ``None`` keeps every pixel.
    :type mask: torch.Tensor | None
    :returns: Scalar loss.
    :rtype: torch.Tensor
    :raises ValueError: If fewer than two members are given.
    """
    m = pred.shape[1]
    if m < 2:
        raise ValueError("fair CRPS needs at least 2 members")
    skill = (pred - target[:, None]).abs().mean(1)
    spread = (pred[:, :, None] - pred[:, None]).abs().sum((1, 2)) / (2 * m * (m - 1))
    score = skill - spread
    return score.mean() if mask is None else score[:, mask].mean()


def multiscale_fair_crps(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, factors=(2, 4, 8, 16)
) -> torch.Tensor:
    """Fair CRPS at full resolution and on fields averaged over blocks of several sizes.

    Per pixel, CRPS only sees how far members spread at each point, not over which distance they differ, so
    noise texture at the wrong scale costs nothing as long as the spread is right. Scoring block averages
    too penalises members whose differences sit at the wrong scale. Averages are taken over the retained
    pixels only, and a block counts when most of it is retained.

    :param pred: Ensemble of shape ``(batch, members, height, width)``.
    :type pred: torch.Tensor
    :param target: Truth of shape ``(batch, height, width)``.
    :type target: torch.Tensor
    :param mask: Boolean grid of shape ``(height, width)``; ``None`` keeps every pixel.
    :type mask: torch.Tensor | None
    :param factors: Block sizes in pixels.
    :type factors: collections.abc.Sequence[int]
    :returns: Scalar loss, the mean of the score at every scale.
    :rtype: torch.Tensor
    """
    weight = (torch.ones_like(target[0]) if mask is None else mask.to(pred.dtype))[None, None]
    scores = [fair_crps(pred, target, mask)]
    for factor in factors:
        covered = F.avg_pool2d(weight, factor, ceil_mode=True)
        blocks = covered.clamp_min(1e-6)
        pooled_pred = F.avg_pool2d(pred * weight, factor, ceil_mode=True) / blocks
        pooled_target = F.avg_pool2d(target[:, None] * weight, factor, ceil_mode=True)[:, 0] / blocks[:, 0]
        scores.append(fair_crps(pooled_pred, pooled_target, covered[0, 0] > 0.5))
    return torch.stack(scores).mean()


def mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean squared error of the first member, averaged over the retained pixels.

    :param pred: Prediction of shape ``(batch, members, height, width)``.
    :type pred: torch.Tensor
    :param target: Truth of shape ``(batch, height, width)``.
    :type target: torch.Tensor
    :param mask: Boolean grid of shape ``(height, width)``; ``None`` keeps every pixel.
    :type mask: torch.Tensor | None
    :returns: Scalar loss.
    :rtype: torch.Tensor
    """
    squared_error = (pred[:, 0] - target) ** 2
    return squared_error.mean() if mask is None else squared_error[:, mask].mean()


def static_stack(static: dict, stats: dict, coast_distance: bool = False) -> np.ndarray:
    """Normalise the static channels.

    :param static: Fields ``cerra_orography``, ``era5_orography``, ``land_sea_mask``, ``lat``,
        ``lon`` and, when requested, ``coast_distance``.
    :type static: dict
    :param stats: Normalisation statistics with ``mean`` and ``std`` per field.
    :type stats: dict
    :param coast_distance: Include the distance-to-coast channel.
    :type coast_distance: bool
    :returns: Array of shape ``(n_static, height, width)`` in channel order.
    :rtype: numpy.ndarray
    :raises KeyError: If the distance channel is requested but absent.
    """
    height_diff = static["cerra_orography"] - static["era5_orography"]
    normalise = lambda values, key: (values - stats[key]["mean"]) / stats[key]["std"]
    stack = [
        normalise(static["cerra_orography"], "cerra_orography"),
        (height_diff - stats["orography_diff"]["mean"]) / stats["orography_diff"]["std"],
        static["land_sea_mask"],
        normalise(static["lat"], "lat"),
        normalise(static["lon"], "lon"),
    ]
    if coast_distance:
        if COAST_CHANNEL not in static:
            raise KeyError(f"{COAST_CHANNEL} requested but not present in the static fields")
        stack.append(normalise(static[COAST_CHANNEL], COAST_CHANNEL))
    return np.stack(stack).astype("f4")


def time_features(times) -> np.ndarray:
    """Encode the time of day and day of year.

    :param times: Valid times.
    :type times: numpy.ndarray | collections.abc.Sequence
    :returns: Array of shape ``(n_times, 4)`` holding the sine and cosine of each cycle.
    :rtype: numpy.ndarray
    """
    t = np.asarray(times, dtype="datetime64[ns]")
    hour = (t - t.astype("datetime64[D]")) / np.timedelta64(1, "h")
    day_of_year = (t - t.astype("datetime64[Y]")) / np.timedelta64(1, "D")
    daily, yearly = 2 * np.pi * hour / 24, 2 * np.pi * day_of_year / 365.25
    return np.stack([np.sin(daily), np.cos(daily), np.sin(yearly), np.cos(yearly)], 1).astype("f4")


def make_inputs(coarse_k: np.ndarray, times, static_norm: np.ndarray, stats: dict) -> np.ndarray:
    """Assemble the normalised model input.

    :param coarse_k: Coarse field on the fine grid [K], shape ``(time, height, width)``.
    :type coarse_k: numpy.ndarray
    :param times: Valid times, one per leading index.
    :type times: numpy.ndarray | collections.abc.Sequence
    :param static_norm: Normalised static channels from :func:`static_stack`.
    :type static_norm: numpy.ndarray
    :param stats: Normalisation statistics.
    :type stats: dict
    :returns: Array of shape ``(time, channels, height, width)``.
    :rtype: numpy.ndarray
    """
    n_times, height, width = coarse_k.shape
    coarse = ((coarse_k - stats["era5_t2m"]["mean"]) / stats["era5_t2m"]["std"]).astype("f4")[:, None]
    static = np.broadcast_to(static_norm[None], (n_times, *static_norm.shape))
    calendar = np.broadcast_to(
        time_features(times)[:, :, None, None], (n_times, len(TIME_CHANNELS), height, width)
    )
    return np.concatenate([coarse, static, calendar], 1)


def residual_norm(coarse_k: np.ndarray, fine_k: np.ndarray, stats: dict) -> np.ndarray:
    """Normalise the training target, which is the fine minus coarse difference.

    :param coarse_k: Coarse field on the fine grid [K].
    :type coarse_k: numpy.ndarray
    :param fine_k: Fine field [K].
    :type fine_k: numpy.ndarray
    :param stats: Normalisation statistics holding ``residual``.
    :type stats: dict
    :returns: Normalised residual, same shape as the inputs.
    :rtype: numpy.ndarray
    """
    residual = stats["residual"]
    return ((fine_k - coarse_k - residual["mean"]) / residual["std"]).astype("f4")


def to_kelvin(coarse_k: np.ndarray, pred_norm: np.ndarray, stats: dict) -> np.ndarray:
    """Turn normalised corrections back into temperatures.

    :param coarse_k: Coarse field [K], shape ``(batch, height, width)``.
    :type coarse_k: numpy.ndarray
    :param pred_norm: Normalised corrections, shape ``(batch, members, height, width)``.
    :type pred_norm: numpy.ndarray
    :param stats: Normalisation statistics holding ``residual``.
    :type stats: dict
    :returns: Fine field [K], shape ``(batch, members, height, width)``.
    :rtype: numpy.ndarray
    """
    residual = stats["residual"]
    return coarse_k[:, None] + pred_norm * residual["std"] + residual["mean"]


class Downscaler:
    """A trained model together with everything inference needs."""

    def __init__(self, net: DownscalerNet, stats: dict, static: dict, device: str = "cpu"):
        """
        :param net: Trained network.
        :type net: DownscalerNet
        :param stats: Normalisation statistics.
        :type stats: dict
        :param static: Fine-grid static fields.
        :type static: dict
        :param device: Torch device.
        :type device: str
        """
        self.net = net.to(device).eval()
        self.stats = stats
        self.static = static
        self.device = device
        self.static_norm = static_stack(static, stats, net.cfg.coast_distance)
        self.lat, self.lon = static["lat"], static["lon"]

    def save(self, path, extra: dict | None = None) -> None:
        """Write the checkpoint.

        :param path: Destination file.
        :type path: str | pathlib.Path
        :param extra: Additional metadata to store alongside the weights.
        :type extra: dict | None
        :returns: ``None``
        :rtype: None
        """
        torch.save({
            "state_dict": self.net.state_dict(),
            "config": asdict(self.net.cfg),
            "stats": self.stats,
            "static": {k: np.asarray(v) for k, v in self.static.items()},
            "channels": self.net.cfg.channels,
            "extra": extra or {},
        }, path)

    @classmethod
    def load(cls, path, device: str | None = None) -> "Downscaler":
        """Read a checkpoint.

        :param path: Checkpoint file.
        :type path: str | pathlib.Path
        :param device: Torch device; ``None`` selects CUDA, then Apple's MPS, when available.
        :type device: str | None
        :returns: Model ready for inference.
        :rtype: Downscaler
        :raises ValueError: If the checkpoint's channel order differs from this code's.
        """
        if device is None:
            mps = getattr(torch.backends, "mps", None)
            device = "cuda" if torch.cuda.is_available() else "mps" if mps and mps.is_available() else "cpu"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        known = {f.name for f in fields(Config)}
        cfg = Config(**{k: v for k, v in checkpoint["config"].items() if k in known})
        if checkpoint["channels"] != cfg.channels:
            raise ValueError(
                f"checkpoint channels {checkpoint['channels']} differ from this code's {cfg.channels}"
            )
        net = DownscalerNet(cfg)
        net.load_state_dict(checkpoint["state_dict"])
        return cls(net, checkpoint["stats"], checkpoint["static"], device)

    @torch.no_grad()
    def predict(self, coarse_k: np.ndarray, times, members: int = 4, batch: int = 8) -> np.ndarray:
        """Downscale a coarse field already on the fine grid.

        :param coarse_k: Coarse field [K], shape ``(time, height, width)``.
        :type coarse_k: numpy.ndarray
        :param times: Valid times, one per leading index.
        :type times: numpy.ndarray | collections.abc.Sequence
        :param members: Maps per input; a deterministic model always returns one.
        :type members: int
        :param batch: Time steps processed at once.
        :type batch: int
        :returns: Fine field [K], shape ``(time, members, height, width)``.
        :rtype: numpy.ndarray
        """
        out = []
        for i in range(0, len(coarse_k), batch):
            coarse = coarse_k[i:i + batch]
            x = torch.from_numpy(
                make_inputs(coarse, times[i:i + batch], self.static_norm, self.stats)
            ).to(self.device)
            pred = self.net(x, members).cpu().numpy()
            out.append(to_kelvin(coarse, pred, self.stats).astype("f4"))
        return np.concatenate(out, 0)