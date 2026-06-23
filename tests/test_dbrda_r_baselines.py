from pathlib import Path
import importlib

import numpy as np
import pandas as pd
import pytest

import rdacca_hp
from rdacca_hp import (
    VEGAN_DISTANCE_METHODS,
    calculate_distance_matrix,
    permu_hp,
    rdacca_hp as run_rdacca_hp,
)


REFERENCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "dbrda_reference"
)

DISTANCE_METHODS = (
    "bray",
    "euclidean",
    "manhattan",
    "canberra",
    "jaccard",
    "kulczynski",
    "gower",
    "hellinger",
    "chord",
)


def _read_csv(name: str, *, preserve_none: bool = False) -> pd.DataFrame:
    return pd.read_csv(
        REFERENCE_DIR / name,
        index_col=0,
        keep_default_na=not preserve_none,
    )


@pytest.fixture(scope="module")
def mite_data():
    mite = _read_csv("mite.csv")
    env = _read_csv("mite_env.csv", preserve_none=True)
    xy = _read_csv("mite_xy.csv")
    pcnm = _read_csv("mite_pcnm.csv")
    bray = _read_csv("distance_bray.csv").to_numpy(dtype=float)
    groups = {
        "Environment": env,
        "Spatial": xy,
        "PCNM": pcnm.iloc[:, :3],
    }
    factor_kwargs = {
        "categorical_factors": ["Substrate", "Topo"],
        "ordered_factors": {"Shrub": ["None", "Few", "Many"]},
    }
    return mite, env, xy, pcnm, bray, groups, factor_kwargs


def _normalize_r_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.astype(str).str.strip()
    out.columns = out.columns.astype(str).str.strip()
    return out


def _assert_result_matches_r(result, prefix: str):
    expected_hier = _normalize_r_table(_read_csv(f"{prefix}_hier_part.csv"))
    expected_var = _normalize_r_table(_read_csv(f"{prefix}_var_part.csv"))
    expected_total = float(
        pd.read_csv(REFERENCE_DIR / f"{prefix}_total.csv")["total"].iloc[0]
    )

    actual_hier = _normalize_r_table(result.hier_part)
    actual_var = _normalize_r_table(result.var_part)

    pd.testing.assert_frame_equal(
        actual_hier,
        expected_hier,
        check_exact=False,
        rtol=0,
        atol=1e-4,
    )
    pd.testing.assert_frame_equal(
        actual_var,
        expected_var,
        check_exact=False,
        rtol=0,
        atol=1e-4,
    )
    assert result.total_explained_variation == pytest.approx(
        expected_total, abs=5e-4
    )


def test_new_distance_api_is_public():
    assert hasattr(rdacca_hp, "calculate_distance_matrix")
    assert hasattr(rdacca_hp, "VEGAN_DISTANCE_METHODS")
    assert set(VEGAN_DISTANCE_METHODS) == set(DISTANCE_METHODS)
    assert "calculate_distance_matrix" in rdacca_hp.__all__
    assert "VEGAN_DISTANCE_METHODS" in rdacca_hp.__all__


@pytest.mark.parametrize("method", DISTANCE_METHODS)
def test_distance_matrix_matches_vegan(method, mite_data):
    mite = mite_data[0]
    expected = _read_csv(f"distance_{method}.csv").to_numpy(dtype=float)
    actual = calculate_distance_matrix(mite, method=method)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(actual, actual.T, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.diag(actual), 0.0, rtol=0, atol=1e-12)


def test_raw_bray_and_precomputed_bray_are_equivalent(mite_data):
    mite, _, xy, _, bray, _, _ = mite_data

    raw_result = run_rdacca_hp(
        dv=mite,
        iv=xy,
        method="dbRDA",
        type="adjR2",
        distance="bray",
        var_part=True,
    )
    precomputed_result = run_rdacca_hp(
        dv=bray,
        iv=xy,
        method="dbRDA",
        type="adjR2",
        var_part=True,
    )

    pd.testing.assert_frame_equal(raw_result.hier_part, precomputed_result.hier_part)
    pd.testing.assert_frame_equal(raw_result.var_part, precomputed_result.var_part)
    assert raw_result.total_explained_variation == precomputed_result.total_explained_variation
    _assert_result_matches_r(raw_result, "dbrda_bray_xy")


def test_grouped_raw_dbrda_matches_r(mite_data):
    mite, _, _, _, _, groups, factor_kwargs = mite_data

    result = run_rdacca_hp(
        dv=mite,
        iv=groups,
        method="dbRDA",
        type="adjR2",
        distance="bray",
        var_part=True,
        **factor_kwargs,
    )

    assert list(result.hier_part.index) == ["Environment", "Spatial", "PCNM"]
    _assert_result_matches_r(result, "dbrda_bray_groups")


def test_grouped_permutation_raw_and_precomputed_are_equivalent(mite_data):
    mite, _, _, _, bray, groups, factor_kwargs = mite_data
    common = dict(
        iv=groups,
        method="dbRDA",
        type="adjR2",
        permutations=9,
        random_state=2026,
        verbose=False,
        **factor_kwargs,
    )

    raw_result = permu_hp(dv=mite, distance="bray", **common)
    precomputed_result = permu_hp(dv=bray, **common)

    pd.testing.assert_frame_equal(raw_result, precomputed_result)
    assert list(raw_result.index) == ["Environment", "Spatial", "PCNM"]
    p_values = raw_result["Pr(>I)"].str.split().str[0].astype(float)
    assert p_values.between(0, 1).all()


def test_unnamed_group_list_uses_r_style_names(mite_data):
    _, _, xy, pcnm, bray, _, _ = mite_data
    result = run_rdacca_hp(
        dv=bray,
        iv=[xy, pcnm.iloc[:, :3]],
        method="dbRDA",
        type="R2",
    )
    assert list(result.hier_part.index) == ["X1", "X2"]


def test_grouped_dbrda_forwards_all_dbrda_options(monkeypatch, mite_data):
    core_module = importlib.import_module("rdacca_hp.core")
    _, _, xy, pcnm, bray, _, _ = mite_data
    seen = []

    def fake_calculate_dbrda(
        dv_dist,
        iv,
        type="adjR2",
        add=False,
        sqrt_dist=False,
        n_axes=None,
        dbrdatype="dbrda",
    ):
        seen.append((add, sqrt_dist, n_axes, dbrdatype))
        return 0.1

    monkeypatch.setattr(core_module, "calculate_dbrda", fake_calculate_dbrda)
    run_rdacca_hp(
        dv=bray,
        iv={"XY": xy, "PCNM": pcnm.iloc[:, :3]},
        method="dbRDA",
        type="R2",
        add="cailliez",
        sqrt_dist=True,
        n_axes=4,
        dbrdatype="capscale",
    )

    assert seen
    assert all(item == ("cailliez", True, 4, "capscale") for item in seen)


def test_distance_and_group_validation_errors(mite_data):
    mite, _, xy, _, bray, _, _ = mite_data

    with pytest.raises(ValueError, match="Unsupported distance method"):
        calculate_distance_matrix(mite, method="not-a-distance")

    negative = mite.copy()
    negative.iloc[0, 0] = -1
    with pytest.raises(ValueError, match="requires non-negative"):
        calculate_distance_matrix(negative, method="bray")

    with pytest.raises(ValueError, match="produced NaN or Inf"):
        calculate_distance_matrix(np.zeros((2, 3)), method="bray")

    with pytest.raises(ValueError, match="Insufficient number of predictor groups"):
        run_rdacca_hp(
            dv=bray,
            iv={"Only": xy},
            method="dbRDA",
            type="R2",
        )

    with pytest.raises(ValueError, match="same number of rows"):
        run_rdacca_hp(
            dv=bray,
            iv={"A": xy, "B": xy.iloc[:-1]},
            method="dbRDA",
            type="R2",
        )
