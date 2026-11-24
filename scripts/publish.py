"""Package the downscaler and its training data for the Hugging Face Hub.

Usage::

    # build the folders and read the generated cards before anything is uploaded
    python scripts/publish.py model --run runs/crps_multiscale --eval outputs/eval_multiscale \
        --repo you/france-temperature-downscaler --out publish/model
    python scripts/publish.py dataset --repo you/france-downscaling-pairs --out publish/dataset

    # then upload, privately unless --public is given
    python scripts/publish.py model ... --push

Nothing is uploaded without ``--push``, so the cards and the licence file can
be reviewed first.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import xarray as xr

from downscaler.config import load_config
from downscaler.publishing import (
    LICENSE_TEXT,
    ModelArtefacts,
    dataset_card,
    model_card,
    stations_card,
    target_card,
)


logger = logging.getLogger("publish")
MODEL_EXTRAS = [
    "val_report.json",
    "history.json",
    "curves.png",
    "val_maps.png",
    "spectrum.png",
    "spectrum.json",
    "floor.png",
    "floor.json",
    "coast_transect.png",
]
EVAL_EXTRAS = [
    "summary.json",
    "scorecard.png",
    "improvement.png",
    "station_errors.png",
    "quantiles.png",
    "heatwave_map.png",
    "by_lead.csv",
    "gain_by_elevation.csv",
]


def folder_size_mb(folder: Path) -> float:
    """Total size of a folder.

    :param folder: Directory to measure.
    :type folder: pathlib.Path
    :returns: Size in megabytes.
    :rtype: float
    """
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6


def build_model(args, cfg) -> Path:
    """Assemble the model repository folder.

    :param args: Parsed arguments.
    :type args: argparse.Namespace
    :param cfg: Pipeline configuration.
    :type cfg: downscaler.config.Config
    :returns: The folder written.
    :rtype: pathlib.Path
    """
    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "model.pt", out / "model.pt")
    reports = out / "reports"
    reports.mkdir(exist_ok=True)
    for name in MODEL_EXTRAS:
        if (run / name).exists():
            shutil.copy(run / name, reports / name)
    evaluation = Path(args.eval) if args.eval else None
    if evaluation:
        for name in EVAL_EXTRAS:
            if (evaluation / name).exists():
                shutil.copy(evaluation / name, reports / name)

    artefacts = ModelArtefacts.load(run, evaluation)
    card = model_card(
        artefacts,
        args.repo,
        args.dataset_repo,
        args.region,
        args.target_name,
        args.coarse_name,
        args.forecast_repo,
    )
    (out / "README.md").write_text(card)
    (out / "LICENSE").write_text(LICENSE_TEXT)
    logger.info(
        "model folder: %s (%.1f MB, %d report file(s))", out, folder_size_mb(out), len(list(reports.iterdir()))
    )
    return out


def build_dataset(args, cfg) -> Path:
    """Assemble the dataset repository folder.

    :param args: Parsed arguments.
    :type args: argparse.Namespace
    :param cfg: Pipeline configuration.
    :type cfg: downscaler.config.Config
    :returns: The folder written.
    :rtype: pathlib.Path
    :raises FileNotFoundError: If the consolidated pairs are missing.
    """
    pairs_dir = Path(args.pairs or cfg.paths.pairs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = pairs_dir / "pairs.zarr"
    if not source.exists():
        raise FileNotFoundError(f"{source} not found; run build_pairs.py --consolidate first")
    for name in ("pairs.zarr", "static.zarr"):
        destination = out / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(pairs_dir / name, destination)
    for name in ("stats.json", "check.png", "predictability.png", "predictability.csv"):
        if (pairs_dir / name).exists():
            shutil.copy(pairs_dir / name, out / name)

    pairs = xr.open_zarr(out / "pairs.zarr")
    stats = json.loads((pairs_dir / "stats.json").read_text()) if (pairs_dir / "stats.json").exists() else {}
    manifest = {
        "years": f"{cfg.target.years[0]}-{cfg.target.years[1]}",
        "train_years": f"{cfg.target.train_years[0]}-{cfg.target.train_years[1]}",
        "times_utc": ", ".join(cfg.target.times),
        "n_times": int(pairs.sizes["time"]),
        "grid_shape": [int(pairs.sizes["y"]), int(pairs.sizes["x"])],
        "region": list(cfg.region_box),
        "stats": stats,
        "size_mb": round(folder_size_mb(out), 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "README.md").write_text(
        dataset_card(manifest, args.repo, args.model_repo, args.region, args.target_name, args.coarse_name)
    )
    (out / "LICENSE").write_text(LICENSE_TEXT)
    logger.info("dataset folder: %s (%.1f MB, %d time steps)", out, manifest["size_mb"], manifest["n_times"])
    return out


def build_target(args, cfg) -> Path:
    """Assemble the repository folder for the high-resolution reanalysis subset.

    :param args: Parsed arguments.
    :type args: argparse.Namespace
    :param cfg: Pipeline configuration.
    :type cfg: downscaler.config.Config
    :returns: The folder written.
    :rtype: pathlib.Path
    :raises FileNotFoundError: If the consolidated target is missing.
    """
    from downscaler.evaluation.diagnostics import grid_spacing_km

    target_dir = Path(args.target or cfg.paths.target)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = target_dir / "target.zarr"
    if not source.exists():
        raise FileNotFoundError(f"{source} not found; run download_cerra.py --consolidate first")
    for name in ("target.zarr", "static.zarr"):
        destination = out / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(target_dir / name, destination)
    for name in ("crop.json", "sanity.png"):
        if (target_dir / name).exists():
            shutil.copy(target_dir / name, out / name)

    from downscaler.constants import TARGET_SOURCES

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from download_cerra import build_request

    data = xr.open_zarr(out / "target.zarr")
    static = xr.open_zarr(out / "static.zarr")
    source = TARGET_SOURCES.get(cfg.target.dataset, {})
    try:
        from importlib import metadata as importlib_metadata

        cdsapi_version = importlib_metadata.version("cdsapi")
    except Exception:  # noqa: BLE001
        cdsapi_version = None
    example = build_request([cfg.target.variable], int(cfg.target.years[0]), 1, cfg.target.times)
    example["day"] = ["01", "...", "31"]
    manifest = {
        **source,
        "dataset": cfg.target.dataset,
        "example_request": example,
        "cdsapi_version": cdsapi_version,
        "crop": json.loads((out / "crop.json").read_text()) if (out / "crop.json").exists() else {},
        "years": f"{cfg.target.years[0]}-{cfg.target.years[1]}",
        "times_utc": ", ".join(cfg.target.times),
        "n_times": int(data.sizes["time"]),
        "grid_shape": [int(data.sizes["y"]), int(data.sizes["x"])],
        "spacing_km": round(grid_spacing_km(static.lat.values, static.lon.values), 2),
        "region": list(cfg.region_box),
        "size_mb": round(folder_size_mb(out), 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "README.md").write_text(target_card(manifest, args.repo, args.region))
    (out / "LICENSE").write_text(LICENSE_TEXT)
    logger.info("target folder: %s (%.1f MB, %d time steps)", out, manifest["size_mb"], manifest["n_times"])
    return out


def build_stations(args, cfg, provider=None) -> Path:
    """Assemble the repository folder for a set of station observations.

    :param args: Parsed arguments.
    :type args: argparse.Namespace
    :param cfg: Pipeline configuration.
    :type cfg: downscaler.config.Config
    :param provider: Station provider; ``None`` builds the configured one.
    :type provider: downscaler.data.stations.StationProvider | None
    :returns: The folder written.
    :rtype: pathlib.Path
    """
    import pandas as pd

    from downscaler.data.stations import MeteoFranceHourly, QCConfig, load_stations

    provider = provider or MeteoFranceHourly()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stations, report = load_stations(
        provider,
        cfg.region_box,
        args.start,
        args.end,
        variables=cfg.stations.variables,
        cache_dir=args.cache or cfg.paths.stations_cache,
        qc=QCConfig(min_completeness=cfg.stations.min_completeness, primary=cfg.stations.variables[0]),
        holdout_frac=cfg.stations.holdout_frac,
    )
    metadata = pd.DataFrame(
        {
            "station_id": stations.station.values.astype(str),
            "name": stations["name"].values.astype(str),
            "latitude": stations.latitude.values,
            "longitude": stations.longitude.values,
            "elevation_m": stations.elevation.values,
            "holdout": stations.holdout.values.astype(bool),
        }
    )
    metadata.to_parquet(out / "metadata.parquet", index=False)
    variables = [v for v in stations.data_vars if "time" in stations[v].dims]
    observations = (
        stations[variables]
        .to_dataframe()
        .reset_index()
        .rename(columns={"station": "station_id"})
        .dropna(subset=variables, how="all")
    )
    observations["station_id"] = observations["station_id"].astype(str)
    observations[["station_id", "time", *variables]].to_parquet(out / "observations.parquet", index=False)
    report.to_csv(out / "qc_report.csv")

    tool = getattr(provider, "access_tool", None)
    try:
        from importlib import metadata as importlib_metadata

        tool_version = importlib_metadata.version(tool) if tool else None
    except Exception:  # noqa: BLE001
        tool_version = None
    manifest = {
        "provider": provider.name,
        "display_name": getattr(provider, "display_name", provider.name),
        "licence": provider.license,
        "licence_url": getattr(provider, "licence_url", None),
        "source_url": getattr(provider, "source_url", None),
        "attribution": getattr(provider, "attribution", None),
        "access_note": getattr(provider, "access_note", None),
        "dataset_note": getattr(provider, "dataset_note", None),
        "access_tool": tool,
        "access_tool_version": tool_version,
        "region": list(cfg.region_box),
        "start": str(pd.Timestamp(args.start)),
        "end": str(pd.Timestamp(args.end)),
        "variables": list(variables),
        "n_stations": int(len(metadata)),
        "n_candidates": int(report.candidate.sum()),
        "n_holdout": int(metadata.holdout.sum()),
        "n_observations": int(len(observations)),
        "min_completeness": cfg.stations.min_completeness,
        "size_mb": round(folder_size_mb(out), 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "README.md").write_text(stations_card(manifest, args.repo, args.region))
    (out / "LICENSE").write_text(LICENSE_TEXT)
    logger.info(
        "stations folder: %s (%.1f MB, %d stations, %d observations)",
        out,
        manifest["size_mb"],
        manifest["n_stations"],
        manifest["n_observations"],
    )
    return out


def push(folder: Path, repo_id: str, repo_type: str, public: bool) -> None:
    """Upload a folder to the Hub.

    :param folder: Folder to upload.
    :type folder: pathlib.Path
    :param repo_id: Destination repository.
    :type repo_id: str
    :param repo_type: ``"model"`` or ``"dataset"``.
    :type repo_type: str
    :param public: Create the repository public rather than private.
    :type public: bool
    :returns: ``None``
    :rtype: None
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, private=not public, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type=repo_type)
    kind = "datasets/" if repo_type == "dataset" else ""
    print(f"pushed to https://huggingface.co/{kind}{repo_id} ({'public' if public else 'private'})")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    :param argv: Argument list; ``None`` reads ``sys.argv``.
    :type argv: collections.abc.Sequence[str] | None
    :returns: Exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["model", "dataset", "target", "stations"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--repo", required=True, help="destination repository, for example you/name")
    parser.add_argument("--out", required=True, help="folder to assemble before uploading")
    parser.add_argument("--run", default=None, help="model: the training run directory")
    parser.add_argument("--eval", default=None, help="model: the evaluation directory")
    parser.add_argument("--pairs", default=None, help="dataset: overrides paths.pairs")
    parser.add_argument("--target", default=None, help="target: overrides paths.target")
    parser.add_argument("--cache", default=None, help="stations: overrides paths.stations_cache")
    parser.add_argument("--start", default=None, help="stations: first time, UTC")
    parser.add_argument("--end", default=None, help="stations: last time, UTC")
    parser.add_argument("--model-repo", default=None, help="dataset: the companion model repository")
    parser.add_argument("--dataset-repo", default=None, help="model: the companion dataset repository")
    parser.add_argument("--region", default="France")
    parser.add_argument("--target-name", default="CERRA (5.5 km)")
    parser.add_argument("--coarse-name", default="ERA5 (0.25 degrees)")
    parser.add_argument("--forecast-repo", default="ecmwf/aifs-ens-1.0")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.kind == "model":
        if not args.run:
            parser.error("--run is required when publishing a model")
        folder = build_model(args, cfg)
    elif args.kind == "dataset":
        folder = build_dataset(args, cfg)
    elif args.kind == "target":
        folder = build_target(args, cfg)
    else:
        if not (args.start and args.end):
            parser.error("--start and --end are required when publishing stations")
        folder = build_stations(args, cfg)

    print(f"\nread {folder}/README.md and {folder}/LICENSE before uploading")
    if args.push:
        push(folder, args.repo, "model" if args.kind == "model" else "dataset", args.public)
    else:
        print("nothing uploaded; rerun with --push when the card looks right")
    return 0


if __name__ == "__main__":
    sys.exit(main())
