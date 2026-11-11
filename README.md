# weather-forecast-model-downscaler

Train a downscaler that turns a 0.25° ensemble temperature forecast into 5.5 km maps over France, and
apply it to AIFS Ensemble 1.0 forecasts read from Earthmover.

The downscaler is a small U-Net trained on pairs of ERA5 (coarse) and CERRA (fine). It learns the
correction from the interpolated coarse field to the fine one, from the terrain, the coast, the time of
day and of year, and a draw of noise, so every coarse member yields several plausible fine maps.
Running the forecast models themselves lives in a separate project,
[earth2studio-pipeline](https://github.com/EmmaScharfmann/earth2studio-pipeline).

## Setup

```bash
uv sync
uv run arraylake auth login      # Earthmover account: ERA5 and AIFS-ENS
uv run hf auth login             # model and datasets on the Hugging Face Hub
```

AIFS-ENS comes from the [Earthmover marketplace](https://app.earthmover.io/marketplace/6a10729e8a69e6bcaeae0f90).
Subscribing creates a repository under your organisation; set `forecasts.source` in `config_spatial.yaml` to it.

## Tutorial

`tutorial_inference.ipynb` fetches an AIFS-ENS run, applies the published downscaler
([EmmaScharfmann/france-temperature-downscaler](https://huggingface.co/EmmaScharfmann/france-temperature-downscaler)),
maps both forecasts, and evaluates them against CERRA, CERRA's own floor and Météo-France stations.

`training_tutorial.ipynb` trains the downscaler from scratch: it pairs ERA5 from Earthmover with CERRA
from the Hub, trains on 2014 to 2019, scores the model on 2020 against interpolation, and saves a
checkpoint that `tutorial_inference.ipynb` can load.

`tutorial_temporal.ipynb` fills the hours between AIFS-ENS's 6-hourly steps with the hourly model
([kashif/france-temperature-hourly-downscaler](https://huggingface.co/kashif/france-temperature-hourly-downscaler)),
reading the forecast from ECMWF's open data, and scores it hour by hour against the stations.

## Configuration

`config_spatial.yaml` configures the spatial downscaler and everything shared (region, paths, stations,
evaluation). `config_temporal.yaml` starts with `extends: config_spatial.yaml` and lists only what the
hourly model changes, plus its own `temporal` section, so the two files show what is common and what is
specific. Every script takes `--config`; command-line flags override it.

## Pipeline

| Step | Script |
|---|---|
| Download the CERRA target (Copernicus account) | `scripts/download_cerra.py` |
| Pair it with ERA5 from Earthmover | `scripts/build_pairs.py`, then `--consolidate` |
| Train | `scripts/train_downscaler.py --out runs/crps` |
| Check the model against its target's floor | `scripts/check_target_floor.py --model runs/crps/model.pt --start ... --end ...` |
| Check the power spectrum | `scripts/check_spectrum.py --model runs/crps/model.pt` |
| Fetch AIFS-ENS forecasts | `scripts/fetch_forecasts.py --start 2025-08-01 --end 2025-08-31` |
| Downscale them | `scripts/downscale_forecasts.py --model runs/crps/model.pt` |
| Score both against stations | `scripts/eval_downscaled.py` |
| Publish the model or datasets to the Hub | `scripts/publish.py` |

For hourly output between the 6-hourly steps, every script takes `--config config_temporal.yaml`:

| Step | Script |
|---|---|
| Download hourly CERRA (analyses plus their +1 h/+2 h forecasts) | `scripts/download_cerra.py` |
| Pair it with ERA5, from Google's public copy | `scripts/build_pairs.py` |
| Train, starting from the spatial model | `scripts/train_downscaler.py --temporal --out runs/hourly` |
| Fine-tune it with CRPS on block averages too, against grain from the noise | `scripts/train_downscaler.py --temporal --init-hourly runs/hourly/model.pt --multiscale 2 4 8 16 --epochs 6 --learning-rate 3e-4 --out runs/hourly_ms` |
| Fetch AIFS-ENS from ECMWF open data, no account needed | `scripts/fetch_forecasts.py --start ... --end ...` |
| Downscale every hour, with the spatial model as baseline | `scripts/downscale_hourly.py --hourly runs/hourly_ms/model.pt --spatial runs/spatial/model.pt` |
| Score each against stations | `scripts/eval_downscaled.py --forecasts outputs/hourly/forecasts/*.zarr --downscaled outputs/hourly/hourly/*.zarr` |
| Animate one forecast hour by hour, next to the stations | `scripts/movie_hourly.py --run outputs/hourly --stations` |
| Power spectra in space and time against CERRA | `scripts/check_spectrum.py --model runs/spatial/model.pt --hourly runs/hourly_ms/model.pt` |

The hourly pairs must be on the same grid as the spatial model, which `target.crop_like` takes care of for the
published one; training refuses them otherwise. `scripts/movie_hourly.py --forecast ... --downscaled ...`
animates the spatial model alone, at the forecast's own steps. `fetch_forecasts.py --source open-data:ifs-ens`
reads IFS ENS (or `ifs`, `aifs-single`) from the same open data.

## Layout

- `downscale/`: the library. `sources.py` (ERA5 and AIFS-ENS access), `open_data.py` (ECMWF open data),
  `downscaler.py` (network and checkpoint), `hourly.py` (the hourly model), `inference.py` (apply to a forecast), `evaluation.py` and `plots.py` (scores and figures
  for the tutorial), `stations.py` and `station_scoring.py` (observations and scoring), `diagnostics.py`
  (report figures), `fetch.py` and `publish.py` (Hub datasets).
- `scripts/`: the pipeline steps above.
- `tests/`: `uv run pytest`, on synthetic data, no network needed.
- `findings.md`: behaviour of the data and libraries worth knowing.
