# Do we really need bigger AI weather forecasting models to improve resolution?

![Screenshot 2026-10-15 at 15.40.25](https://cdn-uploads.huggingface.co/production/uploads/6a2f0f8a8aec62a63aae7447/TJNpUDaleSOH9htX_eUpQ.png)


## TL;DR

In this blog, I want to share something NLP has taught me: **you don’t always need a bigger model; often you need to adapt the model to your task. The same idea applies to AI weather forecasting.** If you need forecasts at higher resolution, you don’t necessarily need a bigger model at that resolution. You can take an existing global model and train a downscaler on top of it to learn the local details, like the effects of orography or coastlines. For many applications, this gives you a more locally accurate forecast that is much cheaper to train and run than a high-resolution global model

## Table of Contents

 - [Intro](#intro)
- [What is a downscaler & demo](#what-is-a-downscaler)
- [Downscaler's components](#downscalers-components)
- [Running the downscaler over France](#running-the-downscaler-over-france)
- [Training the downscaler over France on temperature](#training-the-downscaler-over-france-on-temperature)
- [Conclusion and some random notes](#conclusion-and-some-random-notes)


## Intro

AI weather models are very good at forecasting large-scale weather patterns. But they see the world as a grid, often at 0.25° resolution (roughly equivalent to a grid of 28 × 28 km in France), where each cell holds a single value for temperature, wind, and every other variable. Anything smaller than a cell gets averaged out, so local effects from valleys, mountains, islands, or coastlines are lost.

Big labs are tackling this by building models that work at finer resolution. WeatherNext 3, for example, forecasts surface temperature and humidity at around 5 km. But building these models requires the data, compute, and engineering that only a few organizations have.

If NLP has taught me anything, it’s that you don’t always need a bigger model; often you need to adapt an existing model to your task. The same idea applies to AI weather forecasting: instead of retraining global models at higher resolution, we can adapt existing ones to the region, variables and resolution we care about.

One way to do this is to train a downscaler on top of an existing model to learn the local details. It’s cheap: I trained one in around an hour on a single GPU to predict temperature over France, and it improved AIFS Ensemble 1.0 forecasts by X% against observations (temperature) from weather stations.

## What is a downscaler? 

A weather forecasting model works at a specific resolution: it only knows the weather on the cells of a grid and is blind inside each cell. A downscaler is an ML model that takes the low resolution forecast and fills in the local details. In NLP terms, it’s like adding a task-specific head on top of a frozen embedding model: the big model provides a good general representation of the atmosphere, and the small head adapts it to what you need.

To train a downscaler, you need pairs of the same weather at two resolutions: a coarse field (low resolution), like the output of a global model, and a fine field, like a high-resolution regional reanalysis. The downscaler learns the difference between the two, using static information such as orography and land–sea boundaries to learn how the weather varies within a coarse cell: a mountaintop is colder than the valley below, and the coast is milder than inland. Once trained, you can plug it on top of a coarse forecast model and get a high-resolution forecast at low cost.


![Screenshot 2026-10-15 at 15.36.54](https://cdn-uploads.huggingface.co/production/uploads/6a2f0f8a8aec62a63aae7447/b7Cl6HVQLZ7EP7RIL99Rf.png)

However, like all ML models, downscalers are unfortunately not magical. They perform well on some tasks and poorly on others. So before training one, we should make sure that what we are trying to improve can be improved by a downscaler. Below is my summary of what a downscaler can and cannot improve.
| what a downscaler can improve | what a downscaler can't improve |
|---|---|
| Add the patterns due to the terrain (valleys, mountains, coastlines) | Fix global errors in the coarse model: if the coarse forecast puts the heat in the wrong region, the downscaler will not fix it |
| Produce a single or an ensemble of fine fields from one coarse field | Add details that the inputs do not contain (it cannot add the effect of mountains if you do not input the elevation) |
| Work behind any coarse model (as it never sees a forecast during training) | Reduce the uncertainty of the large-scale forecast |
| Run in seconds, once trained | Exceed the quality of the fine field used during training |

<iframe
	src="https://emmascharfmann-downscaling-demo.hf.space"
	frameborder="0"
	width="850"
	height="850"
></iframe>


## Downscaler's components

### Training 


![downscaler_training_diagram](https://cdn-uploads.huggingface.co/production/uploads/6a2f0f8a8aec62a63aae7447/ThaGcN9zWdbnWuSLAs4Is.png)

During training, we show the model the same weather at two resolutions with some additional details on the field (static data). It learns what the coarse version systematically misses by predicting the error between the coarse and fine weather. 

**Two datasets**: a coarse dataset and a fine dataset, covering the same place and the same times.
  - In our example, for the coarse dataset, we used ERA5 at 0.25°.
  - In our example, for the fine dataset, CERRA at 5.5 km was used.
Note that we chose those two datasets because CERRA takes its boundary conditions from ERA5, so they describe the same weather at two resolutions. For every time step, we build a pair of the coarse field interpolated onto the fine grid and the fine field itself. The interpolated field is used as the model’s main input and as the baseline it has to beat. The model is trained to learn to predict the difference between the coarse field and the fine field.

**Static data**: In addition to the field to predict, the downscaler needs additional information to infer what is happening within the cell. These fields never change, which is exactly why the model learns easily from them: the correction they imply is the same on every day. We add the season and the time of the day, so the model can also learn how that correction varies through the day and through the year. In our example, we’re using the elevation of the fine-scale field, the height difference between the real surface and the coarse model’s surface, the land fraction and the distance to the sea of the cells, the longitude/latitude and the season and time of the day as static data.

**Loss function**: We made the choice in our example of building a downscaler for an ensemble model. So, we use fair CRPS, which scores an ensemble rather than a single field. It rewards being close to the truth and having realistic spread. Mean squared error could have been used as a loss function if building a downscaler for a deterministic model. Note that the loss in our example is computed over land only, since the target reanalysis produces sea surface temperature by a different route.

**Noise**: while the correction is correlated with the inputs, it is not fully determined by them. So instead of asking for one prediction, we train the downscaler to predict multiple outputs. Each forward pass draws fresh noise and produces a different fine field. Therefore, the downscaler produces an ensemble of forecasts from one coarse field.

**Downscaler**: it is a deep learning model which learns to predict the difference between the coarse field and the fine field from the coarse field and the static data. In our example, we trained a U-Net with about 2M weights, which took around an hour to train on a GPU.


### Inference

Once trained, the downscaler can be added on top of any forecast model (AI-based or physics-based) which outputs the same variable at the same resolution and covers the same region as the training data. In our example, that means that our downscaler can be used with any model producing 2m temperature at 0.25° over France. 

![downscaler_inference](https://cdn-uploads.huggingface.co/production/uploads/6a2f0f8a8aec62a63aae7447/nrYkMy6CfIv91mUhlJx5W.png)

**Input data**: the downscaler takes the coarse ensemble forecast as input, as well as the same static data used during training, which is stored inside the model. Each coarse member is interpolated onto the fine grid before the forward pass. In our example, ten members from AIFS-ENS at 0.25° are used as input.

**Downscaler**: it takes the interpolated coarse field, the static data and a draw of noise, then predicts the correction to add to the coarse field. Since the noise is drawn  on each pass, running the same member several times gives multiple fine fields. In our example we draw four samples per member.

**Output**: the model outputs multiple members at fine resolution. In our example, each of the ten coarse members yields four samples, giving forty members at 5.5 km. Note that in our example, running the downscaler takes only a few seconds per forecast.


## Running the downscaler over France 

This section walks you through how to use the downscaler we trained to improve the resolution of the temperature predicted by a model (here AIFS ENS 0.25°) on France. 

0. Set up
Clone the repo and install dependencies. 
  ```bash
git clone https://github.com/EmmaScharfmann/weather-forecast-model-downscaler
cd weather-forecast-model-downscaler
uv sync
```

The forecasts are fetched from [the Earthmover Marketplace](https://app.earthmover.io/marketplace/6a10729e8a69e6bcaeae0f90),
which needs a [free account](https://app.earthmover.io/auth/login) and a subscription to the AIFS-ENS dataset. Subscribing creates a repository under your organisation. Put the repo name in `FORECASTS` below, then authenticate once. 
I also recommend having a free Hugging Face account ([sign up here](https://huggingface.co/join)) to download the model's weights and data.

```bash
uv run arraylake auth login # or export ARRAYLAKE_TOKEN=ema_xxxxxxxxxxxx
uv run hf auth login # or export HF_TOKEN=hf_xxxxxxxxxxxx
```

Then, import the packages and set the config. 

<details>

<summary>See packages and config </summary>

```python
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from huggingface_hub import hf_hub_download

from downscaler.constants import COARSE_CROP_MARGIN_DEG, ZERO_C
from downscaler.models.temperature import Downscaler
from downscaler.evaluation.ensemble_scores import at_valid_times, lead_scores, stack_members, station_rmse
from downscaler.data.fetch import get_target
from downscaler.models.temperature_inference import downscale_forecast, save_downscaled
from downscaler.visualization.plots import map_grid, maps, native_cells, station_maps
from downscaler.data.regions import parse_region
from downscaler.data.coarse_sources import (
    available_inits,
    fetch_forecast,
    forecast_box,
    open_forecasts,
    save_forecast,
    store_name,
)
from downscaler.processing.station_interpolation import curvilinear_for
from downscaler.data.stations import load_published_stations

MODEL_REPO = "EmmaScharfmann/france-temperature-downscaler"
FORECASTS = "<earthmover-org>/ecmwf-aifs-ens-forecast-subscription"  # your Earthmover subscription
FORECASTS_2020 = "hf://datasets/EmmaScharfmann/aifs-ens-france-aug2020"
CERRA_REPO = "EmmaScharfmann/cerra-france-t2m-2016-2020"
STATIONS = "EmmaScharfmann/france-stations"  # Météo-France, Aug 2020, Aug 2025, Jan-Jun-Jul 2026

MEMBERS = 10  # AIFS-ENS members to downscale  (out of 51)
MAX_LEAD_HOURS = 240
SAMPLES = 4  # members out of the downscaler (total 4 x 10  = 40)
EVAL_SAMPLES = 1  # both ensembles keep 10 members, and downscaling is 4x faster
BOX = forecast_box(parse_region("france"), COARSE_CROP_MARGIN_DEG)
OUT = Path("outputs/tutorial")  # every forecast is fetched and downscaled once, then read from here
```
</details>


1. Fetch the coarse forecast 

```python
archive = open_forecasts(FORECASTS)
inits = available_inits(archive)

INIT = pd.Timestamp("2025-08-11 00:00")
coarse = fetch_forecast(archive, INIT, BOX, members=MEMBERS, max_lead_hours=MAX_LEAD_HOURS)
```

2. Run the downscaler

The model needs a fine grid as input, so the coarse field has to be interpolated first. The static data are part of the model's checkpoint on the Hub. Here, we choose to predict 4 fine members from the coarse field. 

```python
model = Downscaler.load(hf_hub_download(MODEL_REPO, "model.pt"))
LAND = xr.DataArray(model.static["land_sea_mask"] > 0.5, dims=("y", "x"))

downscaled = downscale_forecast(model, coarse, samples=SAMPLES)
fine = stack_members(downscaled)

save_forecast(coarse, OUT / "forecasts")
save_downscaled(downscaled, OUT / "downscaled" / store_name(INIT))
```
`fine` is the output of the downscaler: its shape is `(time, lead_time, member, y, x)`, with 40 members (10 coarse members × 4 samples), and it holds the temperature in kelvin over France at a resolution of 5.5 km. On a GPU, the downscaler takes seconds to run. 

We can plot the coarse forecast and the downscaled forecast: 

```python
LEAD = 60  # hours: 12 UTC on 13 August
k = int(np.flatnonzero(coarse.lead_time.values == np.timedelta64(LEAD, "h"))[0])
valid = pd.Timestamp(downscaled.valid_time.values[0, k])

raw_mean = native_cells(coarse.isel(time=0, lead_time=k).mean("ensemble") - ZERO_C, model.lat, model.lon, LAND.values)
fine_mean = fine.isel(time=0, lead_time=k).mean("member").where(LAND).values - ZERO_C
interp_mean = downscaled.t2m_interp.isel(time=0, lead_time=k).mean("ensemble").where(LAND).values - ZERO_C
elevation = np.where(LAND, model.static["cerra_orography"], np.nan)

LABELS = {
    "title": f"Ensemble mean 2 m temperature, valid {valid:%d %B %Y %H} UTC (+{LEAD} h)",
    "coarse": "(a) AIFS-ENS, 0.25° (~28 km)",
    "downscaled": "(b) AIFS-ENS + downscaler, ~5.5 km (~0.05°)",
    "added": "(c) Correction added by the downscaler: (b) minus (a)",
    "elevation": "(d) Elevation of the 5.5 km grid",
    "temperature_unit": "2 m temperature (°C)",
    "correction_unit": "correction (K)",
    "elevation_unit": "elevation (m)",
}
low, high = np.nanpercentile(np.concatenate([raw_mean.ravel(), fine_mean.ravel()]), [1, 99])
fig = map_grid(
    [
        dict(
            title=LABELS["coarse"],
            lon=model.lon,
            lat=model.lat,
            values=raw_mean,
            vmin=low,
            vmax=high,
            label=LABELS["temperature_unit"],
        ),
        dict(
            title=LABELS["downscaled"],
            lon=model.lon,
            lat=model.lat,
            values=fine_mean,
            vmin=low,
            vmax=high,
            label=LABELS["temperature_unit"],
        ),
        dict(
            title=LABELS["added"],
            lon=model.lon,
            lat=model.lat,
            values=fine_mean - interp_mean,
            cmap="RdBu_r",
            vmin=-6,
            vmax=6,
            label=LABELS["correction_unit"],
        ),
        dict(
            title=LABELS["elevation"],
            lon=model.lon,
            lat=model.lat,
            values=elevation,
            cmap="terrain",
            vmin=0,
            vmax=3000,
            label=LABELS["elevation_unit"],
        ),
    ],
    title=LABELS["title"],
)
```

![Screenshot 2026-10-15 at 15.36.54](https://cdn-uploads.huggingface.co/production/uploads/6a2f0f8a8aec62a63aae7447/b7Cl6HVQLZ7EP7RIL99Rf.png)


We see on panel b) that the resolution is improved everywhere but it is especially obvious in the mountains. On panel c) and d), we see that main corrections are added where the orography is (Massif Central, Alps, Pyrenees). The downscaler is also able to correct the temperature on the coast (especially in Brittany, panel c) from the distance to coast, used as static data. 

3. Compare forecasts against CERRA

CERRA is what the model has learned to reproduce during the training. Comparing against it shows how much the model actually learned.

<details>

<summary> Expand to see the details of the evaluation code</summary>

```python
def run(archive, init):
    """The coarse and downscaled forecasts of one initialisation, computed once and saved."""
    name = store_name(init)
    path = OUT / "downscaled" / name
    if not path.exists():
        coarse = fetch_forecast(archive, init, BOX, members=MEMBERS, max_lead_hours=MAX_LEAD_HOURS)
        save_forecast(coarse, OUT / "forecasts")
        save_downscaled(downscale_forecast(model, coarse, samples=EVAL_SAMPLES), path)
    return xr.open_zarr(OUT / "forecasts" / name)["t2m"], xr.open_zarr(path)


def score(downscaled, stations, cerra=None, short_leads=48):
    """Score one downscaled run against the stations and, when given, against CERRA."""
    downscaled = downscaled.isel(sample=slice(0, EVAL_SAMPLES))  # the run from section c has more
    forecasts = {
        "AIFS-ENS": downscaled.t2m_interp.rename(ensemble="member").load(),
        "AIFS-ENS + downscaler": stack_members(downscaled).load(),
    }
    read = curvilinear_for(model.lat, model.lon, stations).apply
    observed = at_valid_times(stations.t2m, downscaled.valid_time)  # (time, lead_time, station)
    out = {"stations": {}, "means": {}}
    short = downscaled.lead_time <= np.timedelta64(short_leads, "h")
    for name, forecast in forecasts.items():
        at = read(forecast)
        out["stations"][name] = lead_scores(at, observed)
        out["means"][name] = at.mean("member").sel(lead_time=short).values
    out["observed"] = observed.sel(lead_time=short).values
    if cerra is not None:
        truth = at_valid_times(cerra.t2m, downscaled.valid_time)  # (time, lead_time, y, x)
        out["cerra"] = {name: lead_scores(f, truth.where(LAND)) for name, f in forecasts.items()}
        at = read(truth)  # CERRA at the stations
        out["stations"]["CERRA"] = lead_scores(at, observed)  # same valid times: the floor
        out["means"]["CERRA"] = at.sel(lead_time=short).values
    return out
```

```python
archive_2020 = open_forecasts(FORECASTS_2020)
inits_2020 = available_inits(archive_2020)
cerra, _ = get_target(
    inits_2020[0], inits_2020[-1] + pd.Timedelta(hours=MAX_LEAD_HOURS), repo=CERRA_REPO, load=True
)  # about 30 MB, read once rather than streamed
stations_2020 = load_published_stations(STATIONS, start="2020-08-01", end="2020-08-31")
print(
    f"{len(inits_2020)} initialisations, {inits_2020[0]:%d %b} to {inits_2020[-1]:%d %b %Y}; "
    f"{stations_2020.sizes['station']} stations"
)

runs_2020 = {init: run(archive_2020, init) for init in inits_2020}
results_2020 = [score(downscaled_2020, stations_2020, cerra) for _, downscaled_2020 in runs_2020.values()]
```

</details>

```python
coarse_first, first = runs_2020[inits_2020[0]]
k = int(np.flatnonzero(first.lead_time.values == np.timedelta64(36, "h"))[0])
valid_2020 = pd.Timestamp(first.valid_time.values[0, k])

raw = native_cells(
    coarse_first.isel(time=0, lead_time=k).mean("ensemble").load() - ZERO_C, model.lat, model.lon, LAND.values
)
interp = first.t2m_interp.isel(time=0, lead_time=k).mean("ensemble").where(LAND).values - ZERO_C
down = stack_members(first).isel(time=0, lead_time=k).mean("member").where(LAND).values - ZERO_C
truth = cerra.t2m.sel(time=valid_2020).where(LAND).values - ZERO_C

when = f"valid {valid_2020:%d %b %Y %H} UTC (+36 h)"
```

```python
fig = maps(
    [
        ("AIFS-ENS - coarse dataset, resolution 0.25° (~28 km)", model.lon, model.lat, raw),
        ("CERRA - fine dataset, resolution: 5.5 km (~0.05°)", model.lon, model.lat, truth),
        ("AIFS-ENS + downscaler - resolution: 5.5 km (~0.05°)", model.lon, model.lat, down),
    ],
    width=4.4,
    title=f"Ensemble mean 2 m temperature, {when}",
)
```

![Screenshot 2026-10-15 at 15.40.25](https://cdn-uploads.huggingface.co/production/uploads/6a2f0f8a8aec62a63aae7447/TJNpUDaleSOH9htX_eUpQ.png)

```python
fig = maps(
    [
        ("AIFS-ENS minus CERRA", model.lon, model.lat, interp - truth),
        ("AIFS-ENS + downscaler minus CERRA", model.lon, model.lat, down - truth),
    ],
    cmap="RdBu_r",
    vmin=-5,
    vmax=5,
    label="K",
    title=f"Error against CERRA, {when}",
)
for name, error in [("AIFS-ENS", interp - truth), ("AIFS-ENS + downscaler", down - truth)]:
    print(f"{name:24s} RMSE against CERRA over land {np.sqrt(np.nanmean(error**2)):.2f} K")
```

![Screenshot 2026-10-15 at 15.41.01](https://cdn-uploads.huggingface.co/production/uploads/6a2f0f8a8aec62a63aae7447/9fNUMPfupWgiW4qAGiah0.png)


We plot the coarse forecast (AIFS ENS), the fine field (CERRA) and AIFS ENS + downscaler on summer 2020 (which was not used for training). We see that the error of AIFS-ENS against CERRA is very high in the mountains, which is pretty well corrected by our downscaler.


4. Compare forecasts against observations

Observations from Météo-France weather stations are used to evaluate the model against an independent ground truth. Downloading the station data takes around 1 hour for 1 month, so we cached the stations' data on the Hub ([EmmaScharfmann/france-stations](https://huggingface.co/datasets/EmmaScharfmann/france-stations)) to save some time: `load_published_stations` in the evaluation code above reads them from there.

```python
# Ensemble means and observations at every station, first 48 h of every run
names = ("AIFS-ENS", "AIFS-ENS + downscaler")
fields = {
    name: np.concatenate([r["means"][name].reshape(-1, r["means"][name].shape[-1]) for r in results_2020])
    for name in names
}
observed = np.concatenate([r["observed"].reshape(-1, r["observed"].shape[-1]) for r in results_2020])
rmse_2020 = {name: station_rmse(fields[name], observed) for name in names}
fig = station_maps(
    stations_2020.longitude.values,
    stations_2020.latitude.values,
    rmse_2020,
    title="RMSE against each station, first 48 h, August 2020",
)
```

![image](https://cdn-uploads.huggingface.co/production/uploads/6a2f0f8a8aec62a63aae7447/0JxjUTCKmMY8RbNE6ZmGS.png)

Again, we see that the downscaler corrects pretty well the error in the mountains (Alps, Massif Central, Pyrenees) and on the coast (Brittany, Normandy). 

You can run more forecast and evaluation against observations on our [demo space](https://emmascharfmann-downscaling-demo.hf.space). 

## Training the downscaler over France on temperature

You can train your own downscaler (you can change the data, resolution, variable, region...) following what we did to train our own downscaler. I will walk you through the main steps in this blog, but you can find a complete tutorial [here](https://github.com/EmmaScharfmann/weather-forecast-model-downscaler/blob/main/training_tutorial.ipynb). 

0. Set up

You need to clone the [github repo](https://github.com/EmmaScharfmann/weather-forecast-model-downscaler) and log into your Earthmover free account to fetch the coarse field (ERA5 in the tutorial). Then, you need to adapt the configuration for the training: number of epochs, steps per epochs, starting and ending dates of the data for training and validation. 

1. Building the training pair

You need to build pairs of coarse field - fine field. In our tutorial, we're using ERA5 for the coarse field and CERRA for the fine field from 2014 to 2019 for the training and 2020 for the evaluation. ERA5 is fetched from Earthmover while CERRA is downloaded from the Hub. Then, the static data (at the resolution of the fine field) are added to the pairs: orography, longitude/latitude, land-sea mask, time of the year/day and distance to the coast. The model has to learn to predict the difference between the two maps below: 

![Screenshot 2026-10-15 at 14.43.37](https://cdn-uploads.huggingface.co/production/uploads/6a2f0f8a8aec62a63aae7447/dFnAAASv4Z2-NDiuVAdL6.png)

2. Training the model

We train a small U-net to learn to predict the difference between the coarse and the fine field. 
Random noise is added to predict a probabilistic correction. A CRPS loss is used to train the ensemble downscaler. 2014-2019 is used for training.

3. Evaluating the model 

We finally evaluate the model against CERRA (year 2020). We train the model here for 20 epochs. 

![loss](https://cdn-uploads.huggingface.co/production/uploads/6a2f0f8a8aec62a63aae7447/X8ueIzdRrA3Th91yZRPCh.png)

We plot the difference between the coarse model (ERA5) and ERA5 + downscaler. As expected, we see that local errors in the mountains or the coast are corrected by the downscaler while global errors of the coarse model are not corrected by the downscaler (error in Normandy).

![against cerra training](https://cdn-uploads.huggingface.co/production/uploads/6a2f0f8a8aec62a63aae7447/0HQQ3Q4d9AeoK4MmzlALc.png)


## Conclusion and some random notes

Overall, training a downscaler is doable if you have a GPU and a couple of hours to spend. It will not make a forecast better globally, but it will make it (way) better locally. As often, the most interesting part is the evaluation. We made the choice to evaluate the downscaler on the fine prediction as well as on observations as CERRA helps evaluate if the downscaler learned something and the observations shows if the final model is good. But scoring CERRA against the observations also gave us a ceiling (an upper bound on the downscaler’s performance). And the downscaler is surprisingly close to that ceiling (see the plots on the [demo space](https://emmascharfmann-downscaling-demo.hf.space))!

As always at Hugging Face, we’d love to get your feedback about what was useful (or not) and what we could do next! Please reach out to us via the blog post comments or by [email](mailto:emma.scharfmann@huggingface.co).

**Some notes about why we made these choices:**
- We are not doing a robust evaluation of how downscalers perform. The goal is to show what a downscaler is, and how to train it, run it and evaluate it on a given region and variable.
- We focused on temperature as it is a smooth variable. It would be very interesting to apply the same method to precipitation or extreme events.
- We used AIFS-ENS because the forecasts were available on Earthmover and it was fast to run (0.5 s per step on a GPU).
- Our focus on summer is arbitrary. I initially wanted to see how well AI forecasts predicted heatwaves, and drifted towards training a downscaler.

**Some technical details worth sharing:**

- We generate noise at several spatial scales, otherwise the ensemble’s variability piles up at one wavelength.
- The power spectrum (see the demo space) was our most useful diagnostic. It showed how much detail the model adds, and at which scales.
- All errors have a strong daily cycle, because of effects that CERRA, AIFS-ENS and the downscaler all struggle to predict, such as inversions at night and surface heating in the afternoon.
- CERRA’s own performance is the upper bound of what our downscaler can achieve.
- Around 20 to 30% of the difference between the coarse and the fine field is predictable from our inputs. The rest would need more inputs, such as wind.