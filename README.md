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
Subscribing creates a repository under your organisation; set `forecasts.source` in `config.yaml` to it.

## Tutorial

`tutorial_inference.ipynb` fetches an AIFS-ENS run, applies the published downscaler
([EmmaScharfmann/france-temperature-downscaler](https://huggingface.co/EmmaScharfmann/france-temperature-downscaler)),
maps both forecasts, and evaluates them against CERRA, CERRA's own floor and Météo-France stations.

`training_tutorial.ipynb` trains the downscaler from scratch: it pairs ERA5 from Earthmover with CERRA
from the Hub, trains on 2014 to 2019, scores the model on 2020 against interpolation, and saves a
checkpoint that `tutorial_inference.ipynb` can load.

## Pipeline

Every script takes `--config config.yaml`; command-line flags override it.

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

For hourly output between the 6-hourly steps:

| Step | Script |
|---|---|
| Download hourly CERRA (analyses plus their +1 h/+2 h forecasts) | `scripts/download_cerra.py --hourly --start ... --end ...` |
| Pair it with ERA5, here from Google's public copy | `scripts/build_pairs.py --coarse-source era5-arco` |
| Train, starting from a spatial model | `scripts/train_hourly.py --spatial runs/crps/model.pt --out runs/hourly` |
| Fetch AIFS-ENS from ECMWF open data, no account needed | `scripts/fetch_forecasts.py --source open-data --start ... --end ...` |
| Downscale every hour, with the spatial model as baseline | `scripts/downscale_hourly.py --hourly runs/hourly/model.pt --spatial runs/crps/model.pt` |
| Score each against stations | `scripts/eval_downscaled.py --forecasts outputs/hourly/forecasts/*.zarr --downscaled outputs/hourly/hourly/*.zarr` |

The hourly pairs must be on the same grid as the spatial model; `train_hourly.py` refuses them otherwise.

## Layout

- `downscale/`: the library. `sources.py` (ERA5 and AIFS-ENS access), `downscaler.py` (network and
  checkpoint), `inference.py` (apply to a forecast), `evaluation.py` and `plots.py` (scores and figures
  for the tutorial), `stations.py` and `station_scoring.py` (observations and scoring), `diagnostics.py`
  (report figures), `fetch.py` and `publish.py` (Hub datasets).
- `scripts/`: the pipeline steps above.
- `tests/`: `uv run pytest`, on synthetic data, no network needed.
- `findings.md`: behaviour of the data and libraries worth knowing.
