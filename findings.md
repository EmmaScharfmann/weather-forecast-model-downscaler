# Findings

Behaviour of the data, the libraries and the models that is not visible from
the code, recorded while building this pipeline. Each entry names the file it
affects. Dates refer to when the behaviour was observed.

## Grids and coordinates

**Cropped regions are not contiguous in storage order.**
`src/downscaler/data/regions.py`, `src/downscaler/processing/station_interpolation.py`

Gridded archives store longitudes 0..360 with descending latitudes. A region
spanning the prime meridian, such as France at 7°W to 11°E, becomes two blocks:
353..359.75 and 0..11. `crop_indices` returns longitude indices ordered west to
east, so the resulting coordinate reads 353, …, 359.75, 0, …, 11 and is **not
monotonic**. This keeps the array compatible with the archives' own convention,
but anything that interpolates in longitude must convert with `to_signed_lon`
first.

**The regional reanalysis uses a Lambert grid.**
`src/downscaler/processing/station_interpolation.py`, `scripts/download_cerra.py`

CERRA's latitude and longitude are two-dimensional fields over `(y, x)`, so the
regular-grid interpolator does not apply. `CurvilinearInterpolator` uses a
k-d tree on three-dimensional unit vectors, which gives true great-circle
distances and needs no special handling at the meridian.

**A rotated fine grid pokes outside a rectangular coarse domain.**
`scripts/downscale_forecasts.py`

The corners of the cropped CERRA grid fall just outside the France box the
forecasts were cut to, about 0.2% of points. These take edge values with a
warning; more than `MAX_FRACTION_OUTSIDE` is treated as a misconfigured region.

**The Climate Data Store cannot crop a Lambert grid.**
`scripts/download_cerra.py`

Each request downloads the full European domain and the crop happens locally.
The index rectangle is computed once and saved to `crop.json`, so every month
shares exactly the same grid; a month whose source grid differs is rejected
rather than silently misaligned.

## Model output

**The forecast store starts at the first lead step, not at zero.**

A ten-day forecast at six-hourly steps holds 40 entries beginning at +6 h.
Label results using `lead_time` values rather than integer positions.

**Six-hourly output cannot resolve daily extremes.**

The warmest hour of the day usually falls between two steps, so daily maxima
computed from four samples are systematically too low. Score the six-hourly
values, or define events at the steps nearest the afternoon peak.

**Ensemble members from one initial state understate uncertainty.**

When every member starts from the same analysis, spread comes only from the
model's own sampling. Expect a spread-skill ratio well below one at short lead
times. Station verification lowers it further, because the roughly 2 K mismatch
between a grid cell and a point appears in the error but in no member.

## Station observations

**Station metadata is unreliable about history.**
`src/downscaler/data/stations.py`

Of about 13,800 Météo-France stations listed inside the France box, roughly 680
have no start date. A sample of twelve returned no data at all for August 2023;
their names end in `-SPC`, suggesting a separate flood-forecasting network. They
are excluded by default, and `include_unknown_start` costs only download time
because the completeness check removes them anyway.

**Many listed stations do not measure temperature.**

For 5–30 August 2023, 2,229 stations were active over the whole window but 429
of them returned no temperature at all. They appear in the quality-control
report with zero completeness. The network of about 1,800 temperature stations
is what remains.

**Requesting a closed station fails the whole chunk.**
`src/downscaler/data/stations.py`

The underlying service raises a missing-file error, which aborts every station
requested alongside it. The provider retries the chunk one station at a time.

**A corrupted download cache surfaces as an unrelated type error.**
`src/downscaler/data/stations.py`

An interrupted `wetterdienst` download leaves a half-written entry in
`~/.cache/wetterdienst`, after which reads fail with
`TypeError: cannot unpack non-iterable bool object` inside `fsspec`. The
provider retries that chunk with the cache disabled; clearing the directory
fixes it permanently.

**The cache key is the window, not the period covered.**
`src/downscaler/data/stations.py`

Caching per station and window means a request for a different window refetches
everything. `_covering_dirs` therefore reuses any cached window that fully
contains the requested one. A narrower request is free; a wider one must fetch.

**Spatial patterns in this domain are usually elevation in disguise.**
`src/downscaler/evaluation/diagnostics.py`

Two diagnostics were built on predictions that the measurements then
contradicted, in both cases because the spatial variable was confounded with
terrain. The domain boundary was expected to inflate the spread and does not;
distance from the boundary simply tracks the difference between the flat,
partly maritime corners of the crop and the mountainous interior. Coastal
stations were expected to gain little from downscaling, after they appeared
washed out on an improvement map whose colour scale was set by two-kelvin
mountain gains; measured by distance band they gain 0.21 K within ten
kilometres of the sea and 0.39 K between ten and twenty-five, against 0.10 K
more than two hundred kilometres inland.

The pattern that does hold is elevation: the gain rises from about 0.1 K below
five hundred metres to roughly 2 K above three thousand. Any spatial
diagnostic here should therefore report mean elevation per band, and gains
should be shown in relative as well as absolute terms, since the coarse error
is itself three times larger in the mountains than in the lowlands.

## Verification

**A gridded reanalysis is a circular reference for temperature.**

ERA5-Land applies its own lapse-rate correction to ERA5, which also provides
the forecasts' initial conditions, so a lapse-rate baseline would nearly
reproduce it by construction. Station observations are the only independent
truth here, and the only one available after CERRA ends in mid-2021.

**Scores must be summed, not averaged, across initialisations.**
`src/downscaler/evaluation/station_scoring.py`

Combining must sum the per-case statistics and take the square root afterwards.
Averaging per-case root mean squared errors differed by 0.56 K in one test.

**Ensembles of different sizes cannot share a dimension name.**
`src/downscaler/evaluation/station_scoring.py`, `scripts/eval_downscaled.py`

Putting a 10-member and a 40-member forecast into one dataset under the name
`ensemble` makes xarray align them, padding the smaller with missing values and
silently producing missing ensemble scores. The downscaled ensemble therefore
uses `ensemble_ds`, and the scorer accepts any dimension whose name begins with
`ensemble`.

**A dense cluster of stations distorts the mean.**
`src/downscaler/evaluation/station_scoring.py`

With 40 of 50 synthetic stations crammed into one spot and biased by 2 K, the
unweighted mean bias was 1.61 K and the density-weighted one 0.24 K.

**Bootstrap intervals over a dozen initialisations are illustrative.**
`src/downscaler/evaluation/station_scoring.py`

Neighbouring initialisations share weather, which is why blocks are resampled
together. With few cases the intervals describe sampling noise within the
period studied, not uncertainty about the method in general.

## Training

**A constant lapse rate helps by day and harms at night.**

Measured against about 1,800 stations over an August heatwave, the correction
reduced the error by 0.4–0.55 K at 12 and 18 UTC and slightly increased it at
00 and 06 UTC, when cold air pools in valleys. By elevation, it moved the bias
above 2,500 m from +4.91 K to −0.47 K while worsening the lowlands. The overall
average looked unfavourable only because most stations are low.

**The coarse grid damps the daily cycle.**

Interpolated fields are warm at night and cold at midday relative to stations,
for the analysis as much as the forecast. No height correction can fix this,
since it reflects the difference between a cell average and a point.

**Mean squared error and fair CRPS fail differently.**
`src/downscaler/models/temperature.py`

On synthetic data with a known noise floor, the mean squared error model
reached the best possible root mean squared error with zero spread, while the
CRPS model reached the theoretical best score of 0.564 K with a spread-skill
ratio approaching one. The first gives one smooth map; the second keeps
realistic variability.

**Masking the sea in the loss changes the gradient but not always the result.**
`src/downscaler/models/temperature.py`, `scripts/train_downscaler.py`

With `training.land_only`, the sea is still given to the network as input, since
coastal gradients carry information, but it is excluded from the loss and from
the normalisation statistics. The unit tests confirm the excluded region
receives no gradient. On synthetic data whose sea target was pure noise, the
land scores were identical either way (0.301 K both), because zero-mean noise
does not pull the shared weights in any particular direction. The setting
should matter when the real sea target carries learnable structure that
competes with the land signal, which is worth testing by training both ways and
comparing the land scores.

**The output layer starts at zero.**
`src/downscaler/models/temperature.py`

Zero-initialising the final convolution makes the untrained model reproduce the
interpolated field exactly, so training begins from the baseline rather than
from noise.

**Excess power at the grid scale comes from the decoder and the noise.**
`src/downscaler/models/temperature.py`

Measured on real data, a trained member carried 0.80 to 0.97 of the target's
power between 25 and 400 km, where the interpolated coarse field fell to 0.014,
but 6.4 times the target's power below 25 km, rising to about 22 times at the
two-pixel scale. The ensemble mean sat at 1.3 there, so most of the excess was
spread rather than bias.

Two causes, both now configurable. Strided transposed convolutions produce
checkerboard artefacts at exactly the two-pixel scale, so ``upsample:
bilinear`` interpolates and then convolves instead. Noise channels drawn at
full resolution give the model an easy way to satisfy the spread term of the
CRPS at the grid scale, so ``noise_fine_fraction: 0`` draws every channel at
reduced resolution and upsamples it, making members differ coherently over tens
of kilometres.

On a synthetic target with no learnable structure at the grid scale, the two
changes reduced the small-scale power ratio from 7785 to 1807 (upsampling
alone) and then to 67 (both), while the station-scale scores were unchanged to
three decimal places. The fix is about realism, not accuracy.

**The domain boundary turned out not to inflate the spread.**
`src/downscaler/evaluation/diagnostics.py`

The expectation was that the network, having less context at the boundary,
would produce more spread there. Measured over France the opposite holds: the
sample spread near the edge is 0.91 of the interior value and the absolute
correction 0.93, while both rise with distance inland, reaching 1.06 and 1.45
at forty cells. The suggested margin is zero.

The trend is geography rather than an artefact. Distance from the boundary is
confounded with terrain: the corners of the rotated crop fall on sea and
lowland, and the deep interior holds the Alps and the Massif Central, where the
corrections are genuinely large. Any future edge diagnostic should therefore
compare like terrain with like, for example by profiling within an elevation
band, rather than treating distance from the boundary as a controlled variable.

**A convolutional network can infer distance from the coast on its own.**
`src/downscaler/models/temperature.py`, `scripts/train_downscaler.py`

The land-sea mask is almost binary at five kilometres, so the expectation was
that a correction decaying inland would be hard to learn from it and that an
explicit distance-to-coast channel would help. On a synthetic coastline with a
wiggly shore, a four-kelvin correction decaying over forty kilometres and a
noise floor of five hundredths of a kelvin, training with and without the
channel gave 0.0505 against 0.0506 K. Both reached the noise floor, so the
network recovers the decay from the mask's edge without being told.

The channel is kept, controlled by ``training.coast_distance``, since it costs
one input and may still help where the real coastline is more complex than the
test, but it should not be expected to sharpen a blurred coastline on its own.
A first attempt at the same test was uninformative because the injected noise
matched the achievable error exactly, and both runs reported 0.3999 K; a
synthetic experiment is only conclusive when the irreducible floor sits well
below the effect being measured.

## Dependencies

**Reading GRIB needs the ecCodes binary library.**

`pip install cfgrib` alone is not enough; the Python `eccodes` package must be
recent enough to pull in `eccodeslib`, which ships the library as a wheel. Pin
`eccodes>=2.38` and `eccodeslib` explicitly, and verify with
`python -m eccodes selfcheck`. The wheel needs glibc 2.28 or newer; on older
systems install ecCodes from conda-forge and set `ECCODES_DIR`.

**Pin the scoring library.**

WeatherBench-X is installed from its git repository, so pin a commit for
reproducibility.