"""Fetch AIFS-ENS forecasts over the region, ready to be downscaled.

Usage::

    python scripts/fetch_forecasts.py --config config.yaml --start 2025-08-01 --end 2025-08-31

Every initialisation between ``--start`` and ``--end`` at the configured hours is
read from the archive in ``forecasts.source`` and written to ``paths.forecasts`` as
one store per initialisation, which ``downscale_forecasts.py`` and
``eval_downscaled.py`` then pick up. Initialisations already fetched are skipped.

Reading from Earthmover needs an account: run ``arraylake auth login`` or set
``ARRAYLAKE_TOKEN``. ``--source open-data`` reads ECMWF's open data instead,
which needs no account and holds every run since mid-2025.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from downscale.config import load_config
from downscale.constants import COARSE_CROP_MARGIN_DEG
from downscale.open_data import fetch_open_data
from downscale.sources import available_inits, fetch_forecast, forecast_box, open_forecasts, save_forecast, store_name

logger = logging.getLogger("fetch")
OPEN_DATA = "open-data"


def select_inits(available: pd.DatetimeIndex, start, end, hours: Sequence[int] | None) -> pd.DatetimeIndex:
    """Initialisations to fetch.

    :param available: Complete initialisations in the archive.
    :type available: pandas.DatetimeIndex
    :param start: First initialisation.
    :type start: str | pandas.Timestamp
    :param end: Last initialisation; a date alone includes that whole day.
    :type end: str | pandas.Timestamp
    :param hours: Initialisation hours to keep; ``None`` or empty keeps all.
    :type hours: collections.abc.Sequence[int] | None
    :returns: The selected initialisations.
    :rtype: pandas.DatetimeIndex
    """
    date_only = isinstance(end, str) and len(end.strip()) <= len("YYYY-MM-DD")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if date_only:
        end = end + pd.Timedelta(hours=23, minutes=59)
    chosen = available[(available >= start) & (available <= end)]
    if hours:
        chosen = chosen[chosen.hour.isin(list(hours))]
    return chosen


def main(argv: Sequence[str] | None = None, archive=None) -> int:
    """Command-line entry point.

    :param argv: Argument list; ``None`` reads ``sys.argv``.
    :type argv: collections.abc.Sequence[str] | None
    :param archive: Already open archive, for testing; ``None`` opens ``forecasts.source``.
    :type archive: xarray.Dataset | None
    :returns: Exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--start", required=True, help="first initialisation")
    parser.add_argument("--end", required=True, help="last initialisation")
    parser.add_argument("--source", default=None, help=f"overrides forecasts.source; {OPEN_DATA!r} reads ECMWF open data")
    parser.add_argument("--members", type=int, default=None, help="overrides forecasts.members")
    parser.add_argument("--max-lead-hours", type=int, default=None, help="overrides forecasts.max_lead_hours")
    parser.add_argument("--out", default=None, help="overrides paths.forecasts")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    out = Path(args.out or cfg.paths.forecasts)
    source = args.source or cfg.forecasts.source
    open_data = source == OPEN_DATA and archive is None
    if open_data:
        first, last = pd.Timestamp(args.start).floor("D"), pd.Timestamp(args.end).floor("D") + pd.Timedelta(days=1)
        available = pd.date_range(first, last, freq="6h", inclusive="left")
        ds = None
    else:
        ds = archive if archive is not None else open_forecasts(source)
        available = available_inits(ds)
    inits = select_inits(available, args.start, args.end, cfg.forecasts.init_hours)
    if inits.empty:
        parser.error(f"no complete initialisation between {args.start} and {args.end} in the archive")
    box = forecast_box(cfg.region_box, COARSE_CROP_MARGIN_DEG)
    members = args.members or cfg.forecasts.members
    max_lead = args.max_lead_hours or cfg.forecasts.max_lead_hours

    written = 0
    for init in inits:
        if Path(str(out / store_name(init)) + ".done").exists():
            logger.info("skipping %s, already fetched", init)
            continue
        if open_data:
            forecast = fetch_open_data(init, box, cfg.forecasts.variable, members, max_lead or 360)
        else:
            forecast = fetch_forecast(ds, init, box, cfg.forecasts.variable, members, max_lead)
        logger.info("wrote %s", save_forecast(forecast, out))
        written += 1
    print(f"\n{len(inits)} initialisation(s) in {out}/, {written} fetched now")
    return 0


if __name__ == "__main__":
    sys.exit(main())
