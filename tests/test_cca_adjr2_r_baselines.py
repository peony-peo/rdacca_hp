from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rdacca_hp import rdacca_hp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "benchmark" / "dbrda_reference"
REFERENCE_DIR = PROJECT_ROOT / "benchmark" / "cca_adjr2_reference"

FACTOR_KWARGS = {
    "categorical_factors": ["Substrate", "Topo"],
    "ordered_factors": {"Shrub": ["None", "Few", "Many"]},
}


def _read_data(name: str, *, preserve_none: bool = False) -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / name,
        index_col=0,
        keep_default_na=not preserve_none,
    )


def _read_reference(name: str) -> pd.DataFrame:
    return pd.read_csv(REFERENCE_DIR / name, index_col=0)


def _normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.astype(str).str.strip()
    out.columns = out.columns.astype(str).str.strip()
    return out


def _assert_result_matches_r(result, prefix: str) -> None:
    expected_hier = _normalize_table(
        _read_reference(f"{prefix}_hier_part.csv")
    )
    expected_var = _normalize_table(
        _read_reference(f"{prefix}_var_part.csv")
    )
    expected_total = float(
        pd.read_csv(REFERENCE_DIR / f"{prefix}_total.csv")["total"].iloc[0]
    )
    actual_hier = _normalize_table(result.hier_part)
    actual_var = _normalize_table(result.var_part)

    assert list(actual_hier.index) == list(expected_hier.index)
    assert list(actual_hier.columns) == list(expected_hier.columns)
    assert list(actual_var.index) == list(expected_var.index)
    assert list(actual_var.columns) == list(expected_var.columns)

    # R and NumPy use different random-number generators. With 1000
    # permutations, fraction estimates agree within 0.005 and percentages
    # within one percentage point for these fixed public-data references.
    np.testing.assert_allclose(
        actual_hier.iloc[:, :3].to_numpy(dtype=float),
        expected_hier.iloc[:, :3].to_numpy(dtype=float),
        rtol=0,
        atol=0.005,
    )
    np.testing.assert_allclose(
        actual_hier.iloc[:, 3].to_numpy(dtype=float),
        expected_hier.iloc[:, 3].to_numpy(dtype=float),
        rtol=0,
        atol=1.0,
    )
    np.testing.assert_allclose(
        actual_var.iloc[:, 0].to_numpy(dtype=float),
        expected_var.iloc[:, 0].to_numpy(dtype=float),
        rtol=0,
        atol=0.005,
    )
    np.testing.assert_allclose(
        actual_var.iloc[:, 1].to_numpy(dtype=float),
        expected_var.iloc[:, 1].to_numpy(dtype=float),
        rtol=0,
        atol=1.0,
    )
    assert result.total_explained_variation == pytest.approx(
        expected_total,
        abs=0.005,
    )


@pytest.fixture(scope="module")
def mite_data():
    mite = _read_data("mite.csv")
    env = _read_data("mite_env.csv", preserve_none=True)
    xy = _read_data("mite_xy.csv")
    pcnm = _read_data("mite_pcnm.csv")
    return mite, env, xy, pcnm


def test_cca_xy_adjr2_matches_r(mite_data):
    mite, _, xy, _ = mite_data
    result = rdacca_hp(
        dv=mite,
        iv=xy,
        method="CCA",
        type="adjR2",
        var_part=True,
        n_perm=1000,
        random_state=202601,
    )
    _assert_result_matches_r(result, "cca_xy_adjr2")


def test_cca_environment_factors_adjr2_match_r(mite_data):
    mite, env, _, _ = mite_data
    result = rdacca_hp(
        dv=mite,
        iv=env,
        method="CCA",
        type="adjR2",
        var_part=True,
        n_perm=1000,
        random_state=202602,
        **FACTOR_KWARGS,
    )
    _assert_result_matches_r(result, "cca_env_adjr2")


def test_cca_grouped_predictors_adjr2_match_r(mite_data):
    mite, env, xy, pcnm = mite_data
    groups = {
        "Environment": env,
        "Spatial": xy,
        "PCNM": pcnm.iloc[:, :3],
    }
    result = rdacca_hp(
        dv=mite,
        iv=groups,
        method="CCA",
        type="adjR2",
        var_part=True,
        n_perm=1000,
        random_state=202603,
        **FACTOR_KWARGS,
    )
    _assert_result_matches_r(result, "cca_groups_adjr2")
