"""Choose wet test days from gauges, then build their ERA5/PARDIG test pack.

The year runner's four test days (15 January, April, July, October 2022) are
dry over France on three of them: gauges record no hour with ≥1 mm, so they
test almost nothing about rain. This script picks days by observed gauge rain
only, never by model scores, and builds the pack on the checkpoint's own grid.

    python scripts/precipitation_test_pack.py choose --start 2022-01-01 --end 2023-12-31 \
        --out data/precipitation/obs_test/days.json
    python scripts/precipitation_test_pack.py build --days data/precipitation/obs_test/days.json \
        --cache data/precipitation/obs_test

Dates are kept within EURADCLIM v3 (2013-2023), after the checkpoint's training (2019) and validation (2020).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml


def choose(args):
    from downscale.rain_obs import choose_test_days, wet_days
    daily = wet_days(args.start, args.end, args.gauge_cache)
    days = choose_test_days(daily, args.per_season, args.quantile, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    record = {"days": [str(d.date()) for d in days], "rule": f"{args.per_season} random days per season and year "
              f"above the {args.quantile:.0%} quantile of gauge wet-hour fraction (≥1 mm), seed {args.seed}",
              "wet_fraction": {str(d.date()): float(daily.loc[d, "wet_fraction"]) for d in days}}
    Path(args.out).write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


def build(args):
    from downscale.precip_data import ERA5_NAMES, crop, open_store
    from downscale.sources import ERA5_ARCO
    from scripts.precipitation import write_pack
    from scripts.precipitation_year import day_pack, hour_ends, targets
    from scripts.precipitation_obs_eval import resolve_model
    checkpoint = torch.load(resolve_model(args.model), map_location="cpu", weights_only=True)
    grid = xr.Dataset(coords={"lat": checkpoint["lat"].numpy(), "lon": checkpoint["lon"].numpy()})
    cfg = yaml.safe_load(Path(args.config).read_text())
    days = [pd.Timestamp(d) for d in json.loads(Path(args.days).read_text())["days"]]
    if any(d.year in (2019, 2020) for d in days):
        raise ValueError("test days overlap the 2019 training or 2020 validation years")
    times = pd.DatetimeIndex(np.concatenate([hour_ends(d, d + pd.Timedelta(days=1)).values for d in days]))
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    truth = targets(times, grid, cache, workers=args.workers)
    box = cfg["region"]
    source = crop(open_store(ERA5_ARCO)[list(ERA5_NAMES.values())], (box[0] - 1, box[1] + 1, box[2] - 1, box[3] + 1))
    paths = []
    for day in days:
        stamps = hour_ends(day, day + pd.Timedelta(days=1))
        path = cache / "days" / f"{day:%Y%m%d}.zarr"
        if not path.exists():
            ds = day_pack(source, stamps, np.asarray(truth[times.get_indexer(stamps)]), grid, cfg)
            write_pack(ds, path)
        paths.append(path)
        print(f"built {day:%Y-%m-%d}", flush=True)
    out = cache / "test.zarr"
    if out.exists():
        shutil.rmtree(out)
    ds = xr.open_mfdataset([str(p) for p in paths], engine="zarr", combine="nested", concat_dim="time")
    write_pack(ds.chunk({"time": 24, "variable": -1, "lat": -1, "lon": -1}), out)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("choose")
    c.add_argument("--start", default="2022-01-01")
    c.add_argument("--end", default="2023-12-31")
    c.add_argument("--per-season", type=int, default=3)
    c.add_argument("--quantile", type=float, default=0.75)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--gauge-cache", default="data/observations")
    c.add_argument("--out", default="data/precipitation/obs_test/days.json")
    b = sub.add_parser("build")
    b.add_argument("--days", default="data/precipitation/obs_test/days.json")
    b.add_argument("--model", default="hf://EmmaScharfmann/france-precipitation-downscaler/model.pt",
                   help="checkpoint whose grid the pack uses (path or hf://)")
    b.add_argument("--config", default="config_precipitation_year.yaml")
    b.add_argument("--cache", default="data/precipitation/obs_test")
    b.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    choose(args) if args.command == "choose" else build(args)


if __name__ == "__main__":
    main()
