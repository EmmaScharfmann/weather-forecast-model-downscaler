"""Typed access to the YAML configuration.

``config_spatial.yaml`` configures the spatial downscaler and everything shared. ``config_temporal.yaml``
starts with ``extends: config_spatial.yaml`` and lists only what the hourly (temporal) downscaler changes,
plus its own ``temporal`` section, so a diff of the two files is exactly what is specific to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .constants import REGION_PRESETS

DEFAULT_CONFIG_PATH = Path("config_spatial.yaml")
#: Times of CERRA's three-hourly analyses.
CERRA_ANALYSIS_TIMES = [f"{h:02d}:00" for h in range(0, 24, 3)]
Region = tuple[float, float, float, float]
T = TypeVar("T")


@dataclass
class RegionConfig:
    """Area of interest."""

    name: str | None = "france"
    box: list[float] | None = None

    def resolve(self) -> Region:
        """Return the bounding box this configuration refers to.

        :returns: ``(lat_min, lat_max, lon_min, lon_max)`` with longitudes in -180..180.
        :rtype: tuple[float, float, float, float]
        :raises ValueError: If neither a known preset name nor a four-element box is given.
        """
        if self.box is not None:
            if len(self.box) != 4:
                raise ValueError(f"region.box must have 4 values, got {self.box}")
            return tuple(float(v) for v in self.box)  # type: ignore[return-value]
        if self.name in REGION_PRESETS:
            return REGION_PRESETS[self.name]
        raise ValueError(f"region.name {self.name!r} is not one of {sorted(REGION_PRESETS)}; set region.box instead")


@dataclass
class PathsConfig:
    """Filesystem locations for each pipeline stage."""

    target: str = "data/cerra"
    pairs: str = "data/pairs"
    runs: str = "runs"
    forecasts: str = "outputs/forecasts"
    downscaled: str = "outputs/downscaled"
    evaluation: str = "outputs/eval"
    stations_cache: str = "outputs/cache/stations"


@dataclass
class TargetConfig:
    """High-resolution reanalysis used as the training target."""

    dataset: str = "reanalysis-cerra-single-levels"
    variable: str = "2m_temperature"
    static_variables: list[str] = field(default_factory=lambda: ["orography", "land_sea_mask"])
    years: list[int] = field(default_factory=lambda: [2014, 2020])
    train_years: list[int] = field(default_factory=lambda: [2014, 2019])
    times: list[str] = field(default_factory=lambda: ["00:00", "06:00", "12:00", "18:00"])
    cds_workers: int = 2
    #: Fetch every hour, from the analyses and the forecasts between them, instead of ``times``.
    hourly: bool = False
    #: ``[start, end]`` day ranges to download instead of whole ``years``; ``None`` uses ``years``.
    periods: list[list[str]] | None = None
    #: Published dataset whose crop to adopt, so the grid matches models trained on it.
    crop_like: str | None = None


@dataclass
class CoarseConfig:
    """Coarse source providing the model input."""

    source: str = "era5"
    variable: str = "t2m"
    fetch_batch: int = 28


@dataclass
class ForecastsConfig:
    """AIFS-ENS forecasts to downscale."""

    source: str = "emmahf/ecmwf-aifs-ens-forecast-subscription"
    variable: str = "t2m"
    members: int | None = 10
    max_lead_hours: int | None = 240
    init_hours: list[int] = field(default_factory=lambda: [0, 12])


@dataclass
class StationsConfig:
    """Observation network used as independent truth."""

    provider: str = "meteofrance_hourly"
    variables: list[str] = field(default_factory=lambda: ["t2m"])
    min_completeness: float = 0.9
    holdout_frac: float = 0.2
    active: str = "full"
    include_unknown_start: bool = False
    window: list[str] | None = None


@dataclass
class TrainingConfig:
    """Model architecture and optimisation settings."""

    loss: str = "crps"
    members: int = 4
    eval_members: int = 8
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    base_channels: int = 32
    depth: int = 3
    noise_channels: int = 4
    upsample: str = "bilinear"
    noise_fine_fraction: float = 0.0
    noise_coarse_factor: int | list[int] = 8
    coast_distance: bool = True
    coast_decay_km: float = 25.0
    seed: int = 0
    max_steps_per_epoch: int | None = None
    land_only: bool = True


@dataclass
class InferenceConfig:
    """Settings for applying a trained model to forecasts."""

    samples: int = 4
    batch_size: int = 8


@dataclass
class EvaluationConfig:
    """Scoring settings."""

    heat_thresholds_c: list[float] = field(default_factory=lambda: [30.0, 35.0])
    elevation_edges: list[float] = field(default_factory=lambda: [0, 200, 500, 1000, 1500, 2500, 5000])
    n_bootstrap: int = 200
    bootstrap_block: int = 2
    density_alpha_deg: float | None = 0.75
    lead_hours_shown: list[int] = field(default_factory=lambda: [6, 24, 48, 96, 240])


@dataclass
class TemporalConfig:
    """The hourly (temporal) downscaler, which fills the hours between 6-hourly coarse steps."""

    #: Spatial model the hourly one starts from and is compared with.
    spatial_model: str = "runs/spatial/model.pt"
    #: Hourly checkpoint to continue from, keeping its statistics; ``None`` starts from ``spatial_model``.
    init_hourly: str | None = None
    #: Start from random weights instead of the spatial model's.
    from_scratch: bool = False
    #: Block sizes in pixels whose averages are also scored by the loss; empty for the plain CRPS.
    multiscale: list[int] = field(default_factory=list)
    #: First and last validation time; training uses the hours before ``val_start``.
    val_start: str | None = None
    val_end: str | None = None
    #: CERRA analysis times, and the forecast lead times [h] that fill the hours between them.
    analysis_times: list[str] = field(default_factory=lambda: list(CERRA_ANALYSIS_TIMES))
    forecast_leads: list[str] = field(default_factory=lambda: ["1", "2"])
    #: Correlation time [h] of each member's noise at inference; ``None`` keeps it fixed for the whole forecast.
    noise_hours: float | None = None


@dataclass
class Config:
    """Full pipeline configuration."""

    region: RegionConfig = field(default_factory=RegionConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    coarse: CoarseConfig = field(default_factory=CoarseConfig)
    forecasts: ForecastsConfig = field(default_factory=ForecastsConfig)
    stations: StationsConfig = field(default_factory=StationsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)

    @property
    def region_box(self) -> Region:
        """Shorthand for ``config.region.resolve()``.

        :returns: The configured bounding box.
        :rtype: tuple[float, float, float, float]
        """
        return self.region.resolve()

    @property
    def val_years(self) -> list[int]:
        """Years kept out of training.

        :returns: Years within ``target.years`` that follow ``target.train_years``.
        :rtype: list[int]
        """
        first, last = self.target.years
        return [y for y in range(int(first), int(last) + 1) if y > int(self.target.train_years[1])]


def _build(cls: type[T], values: dict[str, Any] | None) -> T:
    """Instantiate a dataclass from a mapping, rejecting unknown keys.

    :param cls: Target dataclass.
    :type cls: type
    :param values: Mapping of field names to values; ``None`` yields defaults.
    :type values: dict[str, typing.Any] | None
    :returns: Instance of ``cls``.
    :rtype: object
    :raises ValueError: If ``values`` contains keys that are not fields of ``cls``.
    """
    values = values or {}
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)} for {cls.__name__}; expected {sorted(known)}")
    return cls(**values)


def _read(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    """Read a YAML file, merging it over the file it ``extends``.

    :param path: File to read.
    :type path: pathlib.Path
    :param seen: Files already being read, to reject a cycle.
    :type seen: tuple[pathlib.Path, ...]
    :returns: Sections mapped to their values.
    :rtype: dict
    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If files extend each other in a cycle.
    """
    if not path.exists():
        raise FileNotFoundError(f"configuration file not found: {path}")
    if path.resolve() in seen:
        raise ValueError(f"configuration files extend each other in a cycle: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    base = _read(path.parent / parent, (*seen, path.resolve()))
    return _merge(base, raw)


def _merge(base: dict, raw: dict) -> dict:
    """Replace a base configuration's values, section by section.

    :param base: Sections mapped to their values.
    :type base: dict
    :param raw: Sections mapped to the values replacing them.
    :type raw: dict
    :returns: The merged sections.
    :rtype: dict
    """
    out = {k: dict(v or {}) for k, v in base.items()}
    for section, values in raw.items():
        out[section] = {**out.get(section, {}), **(values or {})}
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load the pipeline configuration.

    :param path: Path to a YAML file; ``None`` uses ``config_spatial.yaml`` if present, else defaults. A file
        may name another in ``extends``, relative to itself, and then holds only what it changes.
    :type path: str | pathlib.Path | None
    :param overrides: Section names mapped to dictionaries of values replacing the file's.
    :type overrides: typing.Any
    :returns: Parsed configuration.
    :rtype: Config
    :raises FileNotFoundError: If ``path`` is given but does not exist.
    """
    if path is None:
        raw = _read(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else {}
    else:
        raw = _read(Path(path))
    raw = _merge(raw, overrides)
    sections = {f.name: f.type for f in fields(Config)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"unknown configuration section(s) {sorted(unknown)}")
    return Config(**{
        name: _build(cls, raw.get(name))
        for name, cls in [(f.name, f.default_factory().__class__) for f in fields(Config) if is_dataclass(f.default_factory())]
    })


def apply_cli(config_section: Any, args: Any, names: list[str]) -> None:
    """Overwrite configuration fields with command-line values that were given.

    :param config_section: Dataclass instance to update in place.
    :type config_section: typing.Any
    :param args: Parsed arguments; attributes set to ``None`` are ignored.
    :type args: typing.Any
    :param names: Field names to copy across.
    :type names: list[str]
    :returns: ``None``
    :rtype: None
    """
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            setattr(config_section, name, value)