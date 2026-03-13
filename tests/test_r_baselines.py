import json
from pathlib import Path

import numpy as np
import pandas as pd

from rdacca_hp.core import rdacca_hp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmark" / "data"
EXPECTED = ROOT / "benchmark" / "expected"


def load_json(name):
    with open(EXPECTED / name, "r", encoding="utf-8") as f:
        return json.load(f)


def load_clean_csv(path):
    return pd.read_csv(path, keep_default_na=False)


def hellinger_transform(df: pd.DataFrame) -> pd.DataFrame:
    row_sums = df.sum(axis=1)
    rel = df.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)
    return np.sqrt(rel)


def assert_close_array(actual, expected, atol=1e-4, rtol=1e-4):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"\nactual={actual}\nexpected={expected}"
    )


def strip_list(xs):
    return [str(x).strip() for x in xs]


def normalize_var_names(xs):
    return [str(x).strip().replace(", and ", " and ").replace(" ,", ",") for x in xs]


def test_rda_numeric_2vars_against_r():
    mite = load_clean_csv(DATA / "mite.csv")
    mite_env = load_clean_csv(DATA / "mite_env.csv")
    dv = hellinger_transform(mite)
    iv = mite_env[["SubsDens", "WatrCont"]].copy()

    expected = load_json("rda_numeric_2vars.json")

    result = rdacca_hp(
        dv=dv,
        iv=iv,
        method="RDA",
        type="adjR2",
        scale=False,
        var_part=True,
    )

    assert abs(result.total_explained_variation - expected["total_explained_variation"]) < 5e-4

    actual_var_rows = normalize_var_names(result.var_part.index)
    expected_var_rows = normalize_var_names(expected["var_part"]["rownames"])
    assert actual_var_rows == expected_var_rows

    actual_hier_rows = strip_list(result.hier_part.index)
    expected_hier_rows = strip_list(expected["hier_part"]["rownames"])
    assert actual_hier_rows == expected_hier_rows

    assert_close_array(
        result.var_part["Fractions"].values,
        expected["var_part"]["values"],
        atol=5e-4,
    )
    assert_close_array(
        result.var_part["% Total"].values,
        expected["var_part"]["perc"],
        atol=1e-2,
    )

    assert_close_array(
        result.hier_part["Unique"].values,
        expected["hier_part"]["unique"],
        atol=5e-4,
    )
    assert_close_array(
        result.hier_part["Average.share"].values,
        expected["hier_part"]["average_share"],
        atol=5e-4,
    )
    assert_close_array(
        result.hier_part["Individual"].values,
        expected["hier_part"]["individual"],
        atol=5e-4,
    )
    assert_close_array(
        result.hier_part["I.perc(%)"].values,
        expected["hier_part"]["perc"],
        atol=1e-2,
    )


def test_rda_unordered_factor_against_r():
    mite = load_clean_csv(DATA / "mite.csv")
    mite_env = load_clean_csv(DATA / "mite_env.csv")
    dv = hellinger_transform(mite)
    iv = mite_env[["WatrCont", "Topo", "Substrate"]].copy()

    expected = load_json("rda_unordered_factor.json")

    result = rdacca_hp(
        dv=dv,
        iv=iv,
        method="RDA",
        type="adjR2",
        scale=False,
        var_part=True,
        categorical_factors=["Topo", "Substrate"],
    )

    assert abs(result.total_explained_variation - expected["total_explained_variation"]) < 5e-4

    actual_var_rows = normalize_var_names(result.var_part.index)
    expected_var_rows = normalize_var_names(expected["var_part"]["rownames"])
    assert actual_var_rows == expected_var_rows

    actual_hier_rows = strip_list(result.hier_part.index)
    expected_hier_rows = strip_list(expected["hier_part"]["rownames"])
    assert actual_hier_rows == expected_hier_rows

    assert_close_array(
        result.var_part["Fractions"].values,
        expected["var_part"]["values"],
        atol=5e-4,
    )
    assert_close_array(
        result.var_part["% Total"].values,
        expected["var_part"]["perc"],
        atol=1e-2,
    )

    assert_close_array(
        result.hier_part["Unique"].values,
        expected["hier_part"]["unique"],
        atol=5e-4,
    )
    assert_close_array(
        result.hier_part["Average.share"].values,
        expected["hier_part"]["average_share"],
        atol=5e-4,
    )
    assert_close_array(
        result.hier_part["Individual"].values,
        expected["hier_part"]["individual"],
        atol=5e-4,
    )
    assert_close_array(
        result.hier_part["I.perc(%)"].values,
        expected["hier_part"]["perc"],
        atol=1e-2,
    )


def test_rda_mite_full_mixed_against_r():
    mite = load_clean_csv(DATA / "mite.csv")
    mite_env = load_clean_csv(DATA / "mite_env.csv")
    dv = hellinger_transform(mite)

    expected = load_json("rda_mite_full_mixed.json")

    result = rdacca_hp(
        dv=dv,
        iv=mite_env,
        method="RDA",
        type="adjR2",
        scale=False,
        var_part=True,
        categorical_factors=["Substrate", "Topo", "Shrub"],
    )

    assert abs(result.total_explained_variation - expected["total_explained_variation"]) < 5e-4

    actual_var_rows = normalize_var_names(result.var_part.index)
    expected_var_rows = normalize_var_names(expected["var_part"]["rownames"])
    assert actual_var_rows == expected_var_rows

    actual_hier_rows = strip_list(result.hier_part.index)
    expected_hier_rows = strip_list(expected["hier_part"]["rownames"])
    assert actual_hier_rows == expected_hier_rows

    assert_close_array(
        result.var_part["Fractions"].values,
        expected["var_part"]["values"],
        atol=5e-4,
    )
    assert_close_array(
        result.var_part["% Total"].values,
        expected["var_part"]["perc"],
        atol=1e-2,
    )

    assert_close_array(
        result.hier_part["Unique"].values,
        expected["hier_part"]["unique"],
        atol=5e-4,
    )
    assert_close_array(
        result.hier_part["Average.share"].values,
        expected["hier_part"]["average_share"],
        atol=5e-4,
    )
    assert_close_array(
        result.hier_part["Individual"].values,
        expected["hier_part"]["individual"],
        atol=5e-4,
    )
    assert_close_array(
        result.hier_part["I.perc(%)"].values,
        expected["hier_part"]["perc"],
        atol=1e-2,
    )


def test_rda_ordered_factor_mixed_against_r():
    mite = load_clean_csv(DATA / "mite.csv")
    mite_env = load_clean_csv(DATA / "mite_env.csv")
    dv = hellinger_transform(mite)
    iv = mite_env[["WatrCont", "Substrate", "Shrub"]].copy()

    expected = load_json("rda_ordered_factor_mixed.json")

    result = rdacca_hp(
        dv=dv,
        iv=iv,
        method="RDA",
        type="adjR2",
        scale=False,
        var_part=True,
        categorical_factors=["Substrate"],
        ordered_factors={"Shrub": ["None", "Few", "Many"]},
    )

    assert abs(result.total_explained_variation - expected["total_explained_variation"]) < 5e-4

    actual_var_rows = normalize_var_names(result.var_part.index)
    expected_var_rows = normalize_var_names(expected["var_part"]["rownames"])
    assert actual_var_rows == expected_var_rows

    actual_hier_rows = strip_list(result.hier_part.index)
    expected_hier_rows = strip_list(expected["hier_part"]["rownames"])
    assert actual_hier_rows == expected_hier_rows

    assert_close_array(
        result.var_part["Fractions"].values,
        expected["var_part"]["values"],
        atol=5e-4,
    )
    assert_close_array(
        result.var_part["% Total"].values,
        expected["var_part"]["perc"],
        atol=1e-2,
    )

    assert_close_array(
        result.hier_part["Unique"].values,
        expected["hier_part"]["unique"],
        atol=5e-4,
    )
    assert_close_array(
        result.hier_part["Average.share"].values,
        expected["hier_part"]["average_share"],
        atol=5e-4,
    )
    assert_close_array(
        result.hier_part["Individual"].values,
        expected["hier_part"]["individual"],
        atol=5e-4,
    )
    assert_close_array(
        result.hier_part["I.perc(%)"].values,
        expected["hier_part"]["perc"],
        atol=1e-2,
    )