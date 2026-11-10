"""Tests for the YAML configuration loader."""

import pytest
import yaml

from downscale.config import Config, load_config


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "region": {"name": "france"},
        "paths": {"pairs": "data/custom"},
        "target": {"years": [2015, 2017], "train_years": [2015, 2016]},
        "training": {"loss": "mse", "epochs": 3},
    }))
    return path


def test_defaults_apply_to_absent_sections(config_file):
    cfg = load_config(config_file)
    assert cfg.paths.pairs == "data/custom"
    assert cfg.paths.runs == "runs"
    assert cfg.evaluation.n_bootstrap == 200


def test_region_resolves_from_a_preset_or_an_explicit_box(config_file):
    assert load_config(config_file).region_box == (40.0, 53.0, -7.0, 11.0)
    cfg = load_config(config_file, region={"name": None, "box": [40, 50, 0, 10]})
    assert cfg.region_box == (40.0, 50.0, 0.0, 10.0)


def test_validation_years_are_those_after_the_training_years(config_file):
    assert load_config(config_file).val_years == [2017]


def test_overrides_replace_file_values(config_file):
    assert load_config(config_file, training={"epochs": 9}).training.epochs == 9
    assert load_config(config_file).training.loss == "mse"


def test_unknown_keys_and_sections_raise(config_file):
    with pytest.raises(ValueError, match="unknown key"):
        load_config(config_file, training={"nonexistent": 1})
    with pytest.raises(ValueError, match="unknown configuration section"):
        load_config(config_file, nonexistent={"a": 1})


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_unknown_region_name_raises(config_file):
    with pytest.raises(ValueError, match="not one of"):
        load_config(config_file, region={"name": "atlantis", "box": None}).region_box


def test_shipped_configs_parse_and_the_temporal_one_changes_only_what_it_lists():
    spatial, temporal = load_config("config_spatial.yaml"), load_config("config_temporal.yaml")
    assert isinstance(spatial, Config) and spatial.training.loss in ("crps", "mse")
    listed = yaml.safe_load(open("config_temporal.yaml"))
    assert listed.pop("extends") == "config_spatial.yaml"
    for section in ("region", "paths", "target", "coarse", "training", "stations", "evaluation"):
        for key, value in vars(getattr(spatial, section)).items():
            expected = listed.get(section, {}).get(key, value)
            assert getattr(getattr(temporal, section), key) == expected, f"{section}.{key}"
    assert temporal.target.hourly and not spatial.target.hourly


def test_a_file_extends_another_relative_to_itself_and_cycles_are_refused(tmp_path):
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "a.yaml").write_text(yaml.safe_dump({"training": {"epochs": 3, "loss": "mse"}}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"extends": "base/a.yaml", "training": {"epochs": 5}}))
    cfg = load_config(tmp_path / "b.yaml", training={"members": 2})
    assert (cfg.training.epochs, cfg.training.loss, cfg.training.members) == (5, "mse", 2)
    (tmp_path / "base" / "a.yaml").write_text(yaml.safe_dump({"extends": "../b.yaml"}))
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "b.yaml")
    assert len(cfg.target.times) == 4