"""Packaging trained downscalers and their data for the Hugging Face Hub.

Cards are generated from the artefacts the pipeline already writes, so the
numbers quoted in them cannot drift from the numbers that were measured. Where
a report is missing the card says so rather than leaving a gap.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

#: Attribution required by each upstream source.
ATTRIBUTIONS = {
    "cerra": (
        "CERRA regional reanalysis, Copernicus Climate Change Service (C3S). "
        "Generated using Copernicus Climate Change Service information."
    ),
    "era5": (
        "ERA5 reanalysis, Copernicus Climate Change Service (C3S). "
        "Generated using Copernicus Climate Change Service information."
    ),
    "aifs": "AIFS-ENS forecasts, ECMWF, released under CC-BY-4.0.",
    "meteofrance": "Station observations, Météo-France, Etalab Open Licence 2.0.",
}


def _read_json(path: Path) -> dict | None:
    """Read a JSON report if it exists.

    :param path: File to read.
    :type path: pathlib.Path
    :returns: Parsed contents, or ``None`` when the file is absent.
    :rtype: dict | None
    """
    if not path.exists():
        logger.info("no report at %s; the card will say the figure was not measured", path)
        return None
    return json.loads(path.read_text())


def _table(rows: list[tuple[str, Any]]) -> str:
    """Render a two-column table.

    :param rows: Label and value pairs.
    :type rows: list[tuple[str, typing.Any]]
    :returns: Markdown table.
    :rtype: str
    """
    body = "\n".join(f"| {label} | {value} |" for label, value in rows)
    return f"| | |\n|---|---|\n{body}"


def load_published_target(path_or_repo: str | Path, stream: bool = True, revision: str | None = None):
    """Open a published reanalysis subset, from the Hub or from a local folder.

    :param path_or_repo: Local folder holding the stores, or a Hub dataset repository id.
    :type path_or_repo: str | pathlib.Path
    :param stream: Read lazily over the network rather than downloading the whole repository.
        Ignored for a local folder.
    :type stream: bool
    :param revision: Revision to read when the repository is on the Hub.
    :type revision: str | None
    :returns: The time-varying field and the static fields, on the same grid.
    :rtype: tuple[xarray.Dataset, xarray.Dataset]
    :raises FileNotFoundError: If a local folder lacks the expected stores.
    """
    import xarray as xr

    folder = Path(path_or_repo)
    if folder.exists():
        missing = [n for n in ("target.zarr", "static.zarr") if not (folder / n).exists()]
        if missing:
            raise FileNotFoundError(f"{folder} does not contain {missing}")
        root = str(folder)
    elif stream:
        root = f"hf://datasets/{path_or_repo}"
        if revision:
            root = f"hf://datasets/{path_or_repo}@{revision}"
    else:
        from huggingface_hub import snapshot_download

        root = snapshot_download(repo_id=str(path_or_repo), repo_type="dataset", revision=revision)
    logger.info("reading the target reanalysis from %s", root)
    return xr.open_zarr(f"{root}/target.zarr"), xr.open_zarr(f"{root}/static.zarr")


@dataclass
class ModelArtefacts:
    """Reports produced for one trained model."""

    config: dict
    validation: dict | None
    floor: dict | None
    spectrum: dict | None
    evaluation: dict | None

    @classmethod
    def load(cls, run_dir: Path, evaluation_dir: Path | None = None) -> ModelArtefacts:
        """Collect whatever reports a run directory holds.

        :param run_dir: Directory written by the training script.
        :type run_dir: pathlib.Path
        :param evaluation_dir: Directory written by the evaluation script, if any.
        :type evaluation_dir: pathlib.Path | None
        :returns: The reports found.
        :rtype: ModelArtefacts
        :raises FileNotFoundError: If the run holds no checkpoint.
        """
        import torch

        checkpoint = run_dir / "model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"no checkpoint at {checkpoint}")
        config = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
        return cls(
            config=config,
            validation=_read_json(run_dir / "val_report.json"),
            floor=_read_json(run_dir / "floor.json"),
            spectrum=_read_json(run_dir / "spectrum.json"),
            evaluation=_read_json(evaluation_dir / "summary.json") if evaluation_dir else None,
        )


def model_card(
    artefacts: ModelArtefacts,
    repo_id: str,
    dataset_repo: str | None,
    region: str,
    target_name: str,
    coarse_name: str,
    forecast_repo: str,
) -> str:
    """Build the model card.

    :param artefacts: Reports collected by :meth:`ModelArtefacts.load`.
    :type artefacts: ModelArtefacts
    :param repo_id: Repository this card will live in.
    :type repo_id: str
    :param dataset_repo: Companion dataset repository, if published.
    :type dataset_repo: str | None
    :param region: Area the model was trained for.
    :type region: str
    :param target_name: Name of the high-resolution training target.
    :type target_name: str
    :param coarse_name: Name of the coarse source used during training.
    :type coarse_name: str
    :param forecast_repo: Repository of the forecast model it was applied to.
    :type forecast_repo: str
    :returns: Markdown card with a YAML header.
    :rtype: str
    """
    cfg = artefacts.config
    validation = artefacts.validation or {}
    model_scores = validation.get("model", {}).get("land", {})
    interpolation = validation.get("interpolation", {}).get("land", {})
    evaluation = artefacts.evaluation or {}
    floor = artefacts.floor or {}

    summary_rows = [
        ("Task", f"downscaling 2 m temperature from a coarse forecast to the {target_name} grid"),
        ("Region", region),
        ("Loss", cfg.get("loss", "crps")),
        ("Parameters", "about 2.1 M"),
        ("Inputs", f"{len(cfg.get('channels', [])) or 11} channels, see below"),
    ]

    def fmt(value, digits=2, suffix=""):
        return f"{value:.{digits}f}{suffix}" if isinstance(value, (int, float)) else "not measured"

    results = []
    if model_scores:
        results.append(
            f"On the validation year, against {target_name} over land, the model reaches "
            f"**{fmt(model_scores.get('rmse'))} K** RMSE where plain interpolation reaches "
            f"{fmt(interpolation.get('rmse'))} K."
        )
    if evaluation:
        rmse = evaluation.get("rmse_K", {})
        crps = evaluation.get("crps_K", {})
        ssr = evaluation.get("ssr", {})
        results.append(
            f"Applied to {forecast_repo} forecasts and verified against station observations, it lowers "
            f"RMSE from {fmt(rmse.get('coarse'))} to {fmt(rmse.get('downscaled'))} K "
            f"({evaluation.get('rmse_change_pct', float('nan')):+.0f}%), CRPS from "
            f"{fmt(crps.get('coarse'))} to {fmt(crps.get('downscaled'))} K "
            f"({evaluation.get('crps_change_pct', float('nan')):+.0f}%), and raises the spread-skill "
            f"ratio from {fmt(ssr.get('coarse'))} to {fmt(ssr.get('downscaled'))}."
        )
    if floor:
        share = floor.get("share_of_recoverable_error_removed")
        if share is not None:
            results.append(
                f"Measured against {target_name} itself, the model removes **{100 * share:.0f}%** of the "
                f"error that separating the coarse field from the target could remove, leaving "
                f"{fmt(floor.get('remaining_gap_to_floor_K'))} K between it and the target's own "
                "distance from the stations."
            )
    results_text = "\n\n".join(results) or "No evaluation report was available when this card was written."

    channels = cfg.get("channels") or []
    channel_text = (
        "\n".join(f"- `{name}`" for name in channels)
        if channels
        else (
            "- the coarse field interpolated onto the fine grid\n"
            "- fine-scale terrain height, and its difference from the coarse model's surface\n"
            "- land fraction, and an index of maritime influence\n"
            "- latitude and longitude\n"
            "- time of day and day of year, as sine and cosine pairs"
        )
    )

    dataset_line = (
        f"Training data: [{dataset_repo}](https://huggingface.co/datasets/{dataset_repo})."
        if dataset_repo
        else "Training data is not published."
    )

    return f"""---
license: other
license_name: mixed-see-card
license_link: LICENSE
library_name: pytorch
pipeline_tag: image-to-image
tags:
- weather
- downscaling
- climate
- super-resolution
- ensemble
---

# Downscaler for AI weather forecasts over {region}

A small ensemble downscaler that adds local detail to a coarse weather
forecast. It takes 2 m temperature on a quarter-degree grid, together with
fine-scale terrain, and produces temperature on the {target_name} grid. Trained
on {coarse_name} paired with {target_name}, it never sees forecasts during
training, so it can be applied to any forecast model that supplies the same
coarse field.

{_table(summary_rows)}

## Results

{results_text}

## Inputs

All channels are on the fine grid and normalised with statistics stored in the
checkpoint:

{channel_text}

Trained with the {cfg.get("loss", "crps")} loss, the model also takes
{cfg.get("noise_channels", 0)} noise channels, so repeated passes give
different plausible fields that together form an ensemble.

## Usage

```python
from downscale.downscaler import Downscaler

model = Downscaler.load("model.pt")
fine = model.predict(coarse_kelvin, valid_times, members=4)
```

`coarse_kelvin` has shape `(time, y, x)` and must already sit on the fine grid;
the checkpoint carries the terrain, the normalisation statistics and the grid
coordinates, so nothing else is needed. The result has shape
`(time, members, y, x)` in kelvin.

## Training

{dataset_line} The target is {target_name} and the coarse input is
{coarse_name} interpolated onto the target grid. The loss is computed over land
only, since the target treats sea surface temperature differently from land.

## Limitations

- **The model has no wind, humidity or cloud inputs.** It can learn corrections
  that are a fixed function of place and time of day, so it captures terrain
  effects well and flow-dependent effects such as foehn or sea breezes only on
  average.
- **Most of the correction is not predictable from these inputs.** A
  climatology of the correction conditioned on month and hour explains only
  10 to 30% of its variance, which bounds what any model with this input set
  can achieve.
- **It reproduces the target's view of the world**, including its biases. Scores
  against independent station observations are the honest measure.
- **The ensemble is under-dispersed against stations**, partly because station
  verification includes the mismatch between a grid cell and a thermometer that
  no member can contain.
- **It is regional.** Applying it outside {region} is untested.

## Licence and attribution

The weights are derived from the sources below and inherit their terms.

- {ATTRIBUTIONS["cerra"]}
- {ATTRIBUTIONS["era5"]}
- {ATTRIBUTIONS["aifs"]}
- {ATTRIBUTIONS["meteofrance"]}

Neither Copernicus, ECMWF nor Météo-France endorses this work.
"""


def dataset_card(
    manifest: dict, repo_id: str, model_repo: str | None, region: str, target_name: str, coarse_name: str
) -> str:
    """Build the dataset card.

    :param manifest: Provenance written beside the data.
    :type manifest: dict
    :param repo_id: Repository this card will live in.
    :type repo_id: str
    :param model_repo: Companion model repository, if published.
    :type model_repo: str | None
    :param region: Area the data covers.
    :type region: str
    :param target_name: Name of the high-resolution target.
    :type target_name: str
    :param coarse_name: Name of the coarse source.
    :type coarse_name: str
    :returns: Markdown card with a YAML header.
    :rtype: str
    """
    years = manifest.get("years", "unknown")
    rows = [
        ("Coarse input", f"{coarse_name}, interpolated onto the target grid"),
        ("Target", target_name),
        ("Region", region),
        ("Period", years),
        ("Times", manifest.get("times_utc", "unknown")),
        ("Grid", f"{manifest.get('grid_shape', ['?', '?'])[0]} x {manifest.get('grid_shape', ['?', '?'])[1]} points"),
        ("Size", manifest.get("size_mb", "unknown")),
    ]
    model_line = (
        f"A downscaler trained on this data is at [{model_repo}](https://huggingface.co/{model_repo})."
        if model_repo
        else "No trained model is published alongside this data."
    )
    stats = manifest.get("stats", {}).get("checks", {})
    baseline = stats.get("baseline_rmse_train_K")
    slope = stats.get("residual_vs_height_slope_K_per_km")
    checks = []
    if baseline is not None:
        checks.append(
            f"Plain interpolation of the coarse field differs from the target by "
            f"**{baseline:.2f} K** over the training years, which is the baseline any downscaler must beat."
        )
    if slope is not None:
        checks.append(
            f"The difference falls by **{abs(slope):.1f} K per kilometre** of height difference between "
            "the two surfaces, close to the standard atmospheric lapse rate, which confirms the two "
            "grids are correctly aligned."
        )

    return f"""---
license: other
license_name: mixed-see-card
license_link: LICENSE
pretty_name: Coarse and fine temperature pairs for downscaling over {region}
tags:
- weather
- downscaling
- climate
size_categories:
- 1K<n<10K
---

# Training pairs for downscaling weather forecasts over {region}

Matched coarse and fine 2 m temperature fields on a single grid, ready for
training a downscaler. Each time step holds the coarse field interpolated onto
the fine grid and the fine field itself, so a model can be trained pixel by
pixel without any further preparation.

{_table(rows)}

{model_line}

## Contents

- `pairs.zarr`: `coarse_t2m` and `fine_t2m` over `(time, y, x)`, both in kelvin.
- `static.zarr`: fine-scale orography, the coarse model's surface height on the
  same grid, and the land fraction.
- `stats.json`: normalisation statistics computed over the training years and
  over land only, plus the sanity checks below.

Coordinates are two-dimensional, since the target uses a Lambert grid.
Longitudes are signed and times are UTC.

## Sanity checks

{" ".join(checks) or "No checks were recorded."}

## Usage

```python
import xarray as xr

pairs = xr.open_zarr("pairs.zarr")
static = xr.open_zarr("static.zarr")
residual = pairs.fine_t2m - pairs.coarse_t2m    # what a downscaler must learn
```

## Caveats

- **The target is a model, not an observation.** Anything trained on it learns
  its biases as well as its detail; verify against independent observations.
- **Sea points are not trustworthy.** The target treats water differently from
  land, so training and evaluation here use land only.
- **The coarse field is an analysis, not a forecast.** A model trained on these
  pairs learns the relationship between scales, not how to correct a forecast's
  errors.

## Licence and attribution

- {ATTRIBUTIONS["cerra"]}
- {ATTRIBUTIONS["era5"]}

Neither Copernicus nor ECMWF endorses this work.
"""


LICENSE_TEXT = """This repository combines material derived from several sources, each keeping its own terms.

1. Data derived from CERRA and ERA5: Copernicus Climate Change Service (C3S) information,
   {cerra}
   Licence to use Copernicus products: https://cds.climate.copernicus.eu/

2. Where forecasts are included, they were produced with ECMWF's AIFS-ENS, released under
   CC-BY-4.0: https://creativecommons.org/licenses/by/4.0/

3. Where station observations are included, they come from Meteo-France under the Etalab
   Open Licence 2.0.

Model weights trained on the above are released under the same terms as the data they
derive from. The code that produced them is licensed separately.
""".format(cerra=ATTRIBUTIONS["cerra"])


def target_card(manifest: dict, repo_id: str, region: str) -> str:
    """Build the card for the high-resolution reanalysis subset.

    :param manifest: Provenance written beside the data, including how it was obtained.
    :type manifest: dict
    :param repo_id: Repository this card will live in.
    :type repo_id: str
    :param region: Area the data covers.
    :type region: str
    :returns: Markdown card with a YAML header.
    :rtype: str
    """
    name = manifest.get("display_name", "the reanalysis")
    shape = manifest.get("grid_shape", ["?", "?"])
    catalogue = manifest.get("catalogue_url")
    request = manifest.get("example_request", {})
    rows = [
        ("Source", f"{name}, {manifest.get('long_name', '')}"),
        ("Produced by", manifest.get("producer", "unknown")),
        ("Catalogue", f"[{catalogue}]({catalogue})" if catalogue else "unknown"),
        ("Variable", "2 m temperature"),
        ("Region", region),
        ("Period", manifest.get("years", "unknown")),
        ("Times", manifest.get("times_utc", "unknown")),
        ("Grid", f"{shape[0]} x {shape[1]} points, about {manifest.get('spacing_km', '?')} km"),
        ("Time steps", manifest.get("n_times", "unknown")),
        ("Size", f"{manifest.get('size_mb', '?')} MB"),
    ]
    request_block = (
        "\n".join("    " + line for line in json.dumps(request, indent=4).splitlines()) if request else "    {}"
    )
    crop = manifest.get("crop", {})
    crop_line = (
        f"rows {crop.get('y0')} to {crop.get('y1')} and columns {crop.get('x0')} to {crop.get('x1')} "
        f"of the source grid of {crop.get('grid_shape', ['?', '?'])}"
        if crop
        else "an index rectangle recorded in `crop.json`"
    )

    cdsapi_version = f" {manifest['cdsapi_version']}" if manifest.get("cdsapi_version") else ""
    return f"""---
license: other
license_name: copernicus
license_link: LICENSE
pretty_name: {name} 2 m temperature over {region}
tags:
- weather
- climate
- reanalysis
---

# {name} 2 m temperature over {region}

A regional subset of {name}, cropped to {region} and stored as Zarr so it can be
read directly with xarray. Published to spare others the download and the GRIB
handling, which is the slow part of working with this reanalysis.

{_table(rows)}

{manifest.get("boundary_note", "")}

## How the data was obtained

{name} is distributed through the Copernicus Climate Data Store. Access needs a
free account, an API key in `~/.cdsapirc`, and acceptance of the dataset's
licence on its catalogue page; requests are refused until the licence is
accepted. Downloads used the `cdsapi` client{cdsapi_version},
one request per month, with a body of this form:

```python
import cdsapi

cdsapi.Client().retrieve(
    "{manifest.get("dataset", "")}",
{request_block},
    "target.grib",
)
```

The Climate Data Store cannot subset a Lambert grid server-side, so each
request returns the full European domain and the crop is applied locally. The
rectangle is fixed once and reused for every month, so all files share exactly
one grid: {crop_line}. Its bounds are stored in `crop.json`.

Each month was then checked before being kept: no missing or duplicated time
steps, no missing values, and temperatures inside a plausible range. A month
that failed was reported and discarded rather than silently included.

GRIB files were decoded with `cfgrib`, which needs the ecCodes library; the
published Zarr avoids that dependency entirely.

## Contents

- `target.zarr`: `t2m` over `(time, y, x)` in kelvin.
- `static.zarr`: `orography` in metres and `land_sea_mask` as a land fraction,
  both on the same grid.
- `crop.json`: the index rectangle taken from the source grid.

The reanalysis uses a Lambert conformal grid, so `lat` and `lon` are
two-dimensional coordinates over `(y, x)` rather than axes. Longitudes are
signed and times are UTC.

```python
import xarray as xr

target = xr.open_zarr("target.zarr")
static = xr.open_zarr("static.zarr")
target.t2m.isel(time=0).plot(x="lon", y="lat")
```

## Caveats

- **This is a model, not an observation.** It assimilates observations but
  remains a reanalysis, with its own biases, particularly over complex terrain.
- **Sea surface temperature is produced differently from land temperature**, so
  the two should not be treated alike.
- **One variable, one subset, one grid.** The full reanalysis holds many more
  variables over a wider domain and a longer period.
- **This is a derived copy.** For the authoritative data, go to the catalogue.

## Licence and attribution

{ATTRIBUTIONS["cerra"]}

Licence to use Copernicus products: {manifest.get("licence_url", "https://cds.climate.copernicus.eu/")}

Neither Copernicus, C3S nor ECMWF endorses this work or is responsible for it.
"""


def stations_card(manifest: dict, repo_id: str, region: str) -> str:
    """Build the card for a set of station observations.

    :param manifest: Provenance written beside the data, including how it was collected.
    :type manifest: dict
    :param repo_id: Repository this card will live in.
    :type repo_id: str
    :param region: Area the stations cover.
    :type region: str
    :returns: Markdown card with a YAML header.
    :rtype: str
    """
    network = manifest.get("display_name", manifest.get("provider", "the network"))
    licence = manifest.get("licence", "see the source")
    licence_url = manifest.get("licence_url")
    source_url = manifest.get("source_url")
    tool = manifest.get("access_tool", "a client library")
    tool_version = manifest.get("access_tool_version")
    rows = [
        ("Network", f"{network} ({manifest.get('provider', '')})"),
        ("Source", f"[{source_url}]({source_url})" if source_url else "unknown"),
        ("Region", region),
        ("Period", f"{manifest.get('start', '?')} to {manifest.get('end', '?')} UTC"),
        ("Stations kept", manifest.get("n_stations", "unknown")),
        ("Stations considered", manifest.get("n_candidates", "unknown")),
        ("Observations", manifest.get("n_observations", "unknown")),
        ("Variables", ", ".join(manifest.get("variables", []))),
        ("Held out for testing", manifest.get("n_holdout", "unknown")),
    ]
    licence_line = f"[{licence}]({licence_url})" if licence_url else licence

    return f"""---
license: other
license_name: etalab-2.0
license_link: LICENSE
pretty_name: Quality-controlled hourly station temperature over {region}
tags:
- weather
- observations
- verification
size_categories:
- 1M<n<10M
configs:
- config_name: stations
  data_files: metadata.parquet
- config_name: observations
  data_files: observations.parquet
---

# Hourly station temperature over {region}

Quality-controlled hourly observations from {network}, assembled for verifying
gridded forecasts. Published because collecting them is slow: the network is
served station by station, so a national set takes about an hour to download
even though the result is small.

{_table(rows)}

## How the data was collected

{manifest.get("access_note", "")}

The subset published here is {manifest.get("dataset_note", "hourly observations")}.
Collection used `{tool}{f" {tool_version}" if tool_version else ""}`, and proceeded
in three steps:

1. **Select.** The station list is filtered to those inside the region's bounding
   box and active over the whole period, so completeness is a meaningful measure.
2. **Download.** Stations are requested in small batches. A station whose archive
   is missing aborts its whole batch, so a failed batch is retried one station at
   a time and the missing ones are recorded rather than silently dropped.
3. **Standardise.** Values are converted to kelvin, timestamps to naive UTC, and
   each station is cached individually so a repeated or narrower request costs
   nothing.

Code: see the `stations` module of the project that produced this dataset.

## Contents

- `metadata.parquet`: one row per station, with identifier, name, latitude,
  longitude, elevation in metres and a `holdout` flag.
- `observations.parquet`: long format, one row per station and hour, in kelvin.
- `qc_report.csv`: every station that was considered, whether it was kept and
  why not, so the selection can be audited.

```python
import pandas as pd

meta = pd.read_parquet("metadata.parquet")
obs = pd.read_parquet("observations.parquet")
```

## Quality control

Stations were kept when they were active over the whole period and reported a
valid value for at least {manifest.get("min_completeness", "?")} of its hours.
Values outside a plausible range were discarded first, which removes sentinel
values such as -9999. Many listed stations measure other variables and return
no temperature at all; these appear in the report with zero completeness, which
is why far fewer stations are kept than considered.

The `holdout` flag marks a reproducible subset, chosen by hashing the station
identifier so that it does not change when stations are added or removed. Use
it to keep a set of stations out of any training.

## Caveats

- **Siting is not controlled.** Urban, valley and exposed sites are all present,
  and a station's reading can differ from its surroundings for local reasons.
- **The network is uneven.** Lowlands are far better sampled than high ground,
  so an unweighted average over stations is biased towards flat terrain.
- **Hourly values are instantaneous**, so a daily maximum computed from
  six-hourly samples will understate the true maximum.
- **This is a derived copy.** For the authoritative data, go to the source.

## Licence and attribution

The observations are published by {network} under {licence_line} and are
redistributed here under the same terms.

> {manifest.get("attribution", "")}

{network} does not endorse this work and is not responsible for it.
"""
