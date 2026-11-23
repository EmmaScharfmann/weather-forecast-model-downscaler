"""Training years and gauge-chosen validation/test days in the annual runner."""
import json

import pandas as pd
import pytest

from scripts.precipitation_year import periods


def config(tmp_path, **annual):
    return {"annual": {"year": 2019, "validation_year": 2020, "test_year": 2022, **annual}}


def test_single_year_keeps_previous_behaviour(tmp_path):
    result = periods(config(tmp_path))
    assert sum(role == "train" for role, *_ in result) == 12
    assert [s for role, s, _ in result if role == "test"][0] == pd.Timestamp("2022-01-15")


def test_several_years_and_listed_days(tmp_path):
    (tmp_path / "val.json").write_text(json.dumps({"days": ["2020-02-03", "2020-07-09"]}))
    (tmp_path / "test.json").write_text(json.dumps({"days": ["2023-10-30"]}))
    result = periods(config(tmp_path, years=[2016, 2017, 2018, 2019], validation_days=str(tmp_path / "val.json"),
                            test_days=str(tmp_path / "test.json")))
    train = [s for role, s, _ in result if role == "train"]
    assert len(train) == 48 and train[0] == pd.Timestamp("2016-01-01")
    assert [(s, e) for role, s, e in result if role == "validation"] == [
        (pd.Timestamp("2020-02-03"), pd.Timestamp("2020-02-04")), (pd.Timestamp("2020-07-09"), pd.Timestamp("2020-07-10"))]
    assert [s for role, s, _ in result if role == "test"] == [pd.Timestamp("2023-10-30")]


def test_listed_days_cannot_overlap_training(tmp_path):
    (tmp_path / "val.json").write_text(json.dumps({"days": ["2019-06-01"]}))
    with pytest.raises(ValueError, match="overlap"):
        periods(config(tmp_path, validation_days=str(tmp_path / "val.json")))
