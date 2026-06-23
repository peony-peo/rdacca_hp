from pathlib import Path

import pandas as pd
import pytest

from rdacca_hp import rdacca_hp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "benchmark" / "dbrda_reference"
REFERENCE_DIR = PROJECT_ROOT / "benchmark" / "cca_r2_reference"

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

    pd.testing.assert_frame_equal(
        _normalize_table(result.hier_part),
        expected_hier,
        check_exact=False,
        rtol=0,
        atol=1e-4,
    )
    pd.testing.assert_frame_equal(
        _normalize_table(result.var_part),
        expected_var,
        check_exact=False,
        rtol=0,
        atol=1e-4,
    )
    assert result.total_explained_variation == pytest.approx(
        expected_total,
        abs=5e-4,
    )


@pytest.fixture(scope="module")
def mite_data():
    mite = _read_data("mite.csv")
    env = _read_data("mite_env.csv", preserve_none=True)
    xy = _read_data("mite_xy.csv")
    pcnm = _read_data("mite_pcnm.csv")
    return mite, env, xy, pcnm


def test_cca_xy_r2_matches_r(mite_data):
    mite, _, xy, _ = mite_data
    result = rdacca_hp(
        dv=mite,
        iv=xy,
        method="CCA",
        type="R2",
        var_part=True,
    )
    _assert_result_matches_r(result, "cca_xy_r2")


def test_cca_environment_factors_r2_match_r(mite_data):
    mite, env, _, _ = mite_data
    result = rdacca_hp(
        dv=mite,
        iv=env,
        method="CCA",
        type="R2",
        var_part=True,
        **FACTOR_KWARGS,
    )
    _assert_result_matches_r(result, "cca_env_r2")


def test_cca_grouped_predictors_r2_match_r(mite_data):
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
        type="R2",
        var_part=True,
        **FACTOR_KWARGS,
    )
    _assert_result_matches_r(result, "cca_groups_r2")
