import numpy as np
import pandas as pd
import pytest

from rdacca_hp.core import (
    _cca_projection,
    calculate_cca,
    chi_square_transform,
    rdacca_hp,
)
from rdacca_hp.utils import create_cca_test_data


def test_calculate_cca_returns_finite_r2_and_adj_r2():
    dv, iv = create_cca_test_data(n_samples=50, n_species=8, n_predictors=3, seed=101)

    r2 = calculate_cca(dv, iv, type="R2", n_perm=49)
    adj_r2 = calculate_cca(
        dv,
        iv,
        type="adjR2",
        n_perm=49,
        random_state=101,
    )

    assert np.isfinite(r2)
    assert np.isfinite(adj_r2)


def test_rdacca_hp_cca_with_dataframe_predictors_runs():
    dv, iv = create_cca_test_data(n_samples=50, n_species=8, n_predictors=3, seed=202)
    iv_df = pd.DataFrame(iv, columns=["Env1", "Env2", "Env3"])

    result = rdacca_hp(
        dv=dv,
        iv=iv_df,
        method="CCA",
        type="adjR2",
        var_part=True,
        n_perm=49,
        random_state=2026,
    )

    assert result.method_type == ["CCA", "adjR2"]
    assert result.hier_part.shape == (3, 4)
    assert result.var_part is not None
    assert np.isfinite(result.total_explained_variation)


def test_rdacca_hp_cca_with_grouped_predictors_runs():
    dv, iv = create_cca_test_data(n_samples=40, n_species=6, n_predictors=4, seed=303)

    groups = {
        "Climate": pd.DataFrame(iv[:, :2], columns=["T", "P"]),
        "Soil": pd.DataFrame(iv[:, 2:], columns=["N", "C"]),
    }

    result = rdacca_hp(
        dv=dv,
        iv=groups,
        method="CCA",
        type="R2",
        var_part=False,
        n_perm=29,
    )

    assert result.method_type == ["CCA", "R2"]
    assert list(result.hier_part.index) == ["Climate", "Soil"]
    assert result.var_part is None


def test_cca_rejects_zero_sum_samples():
    dv, iv = create_cca_test_data(n_samples=30, n_species=5, n_predictors=2, seed=404)
    dv[0, :] = 0
    iv_df = pd.DataFrame(iv, columns=["E1", "E2"])

    with pytest.raises(ValueError, match="All row sums must be positive"):
        rdacca_hp(
            dv=dv,
            iv=iv_df,
            method="CCA",
            type="R2",
            var_part=False,
            n_perm=19,
        )


def test_cca_adjusted_r2_is_reproducible_and_can_be_negative():
    rng = np.random.default_rng(2)
    dv = rng.poisson(3.0, size=(30, 6)).astype(float)
    iv = rng.normal(size=(30, 2))

    first = calculate_cca(
        dv,
        iv,
        type="adjR2",
        n_perm=199,
        random_state=2026,
    )
    second = calculate_cca(
        dv,
        iv,
        type="adjR2",
        n_perm=199,
        random_state=2026,
    )

    assert first == second
    assert first < 0

    partition = rdacca_hp(
        dv=dv,
        iv=pd.DataFrame(iv, columns=["E1", "E2"]),
        method="CCA",
        type="adjR2",
        var_part=True,
        n_perm=199,
        random_state=2026,
    )
    assert partition.total_explained_variation < 0
    assert not (partition.hier_part["I.perc(%)"] == 0).all()


def test_cca_adjusted_r2_uses_vegan_formula():
    rng = np.random.default_rng(606)
    dv = rng.poisson(2.5, size=(24, 5)).astype(float)
    iv = rng.normal(size=(24, 2))
    y_chi, row_weights, _ = chi_square_transform(dv)
    r2, _, _ = _cca_projection(y_chi, iv, row_weights)

    permutation_rng = np.random.default_rng(707)
    null_r2 = []
    for _ in range(31):
        permutation = permutation_rng.permutation(y_chi.shape[0])
        permuted = y_chi[permutation, :]
        permuted_weights = row_weights[permutation]
        permuted_r2, _, _ = _cca_projection(
            permuted,
            iv,
            permuted_weights,
        )
        null_r2.append(permuted_r2)

    expected = 1.0 - (1.0 - r2) / (1.0 - np.mean(null_r2))
    actual = calculate_cca(
        dv,
        iv,
        type="adjR2",
        n_perm=31,
        random_state=707,
    )

    assert actual == expected


def test_cca_adjusted_r2_honors_n_perm():
    rng = np.random.default_rng(808)
    dv = rng.poisson(2.5, size=(25, 5)).astype(float)
    iv = rng.normal(size=(25, 2))

    with_9 = calculate_cca(
        dv,
        iv,
        type="adjR2",
        n_perm=9,
        random_state=909,
    )
    with_99 = calculate_cca(
        dv,
        iv,
        type="adjR2",
        n_perm=99,
        random_state=909,
    )

    assert with_9 != with_99


def test_rdacca_hp_cca_adjusted_r2_is_reproducible():
    dv, iv = create_cca_test_data(
        n_samples=35,
        n_species=6,
        n_predictors=3,
        seed=505,
    )
    iv_df = pd.DataFrame(iv, columns=["E1", "E2", "E3"])
    kwargs = dict(
        dv=dv,
        iv=iv_df,
        method="CCA",
        type="adjR2",
        var_part=True,
        n_perm=49,
        random_state=505,
    )

    first = rdacca_hp(**kwargs)
    second = rdacca_hp(**kwargs)

    assert first.total_explained_variation == second.total_explained_variation
    pd.testing.assert_frame_equal(first.hier_part, second.hier_part)
    pd.testing.assert_frame_equal(first.var_part, second.var_part)
