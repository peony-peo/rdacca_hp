import numpy as np
import pandas as pd

from rdacca_hp.core import calculate_cca, rdacca_hp
from rdacca_hp.utils import create_cca_test_data


def test_calculate_cca_returns_finite_r2_and_adj_r2():
    dv, iv = create_cca_test_data(n_samples=50, n_species=8, n_predictors=3, seed=101)

    r2 = calculate_cca(dv, iv, type="R2", n_perm=49)
    adj_r2 = calculate_cca(dv, iv, type="adjR2", n_perm=49)

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


def test_cca_handles_nonnegative_species_data_with_zeros():
    dv, iv = create_cca_test_data(n_samples=30, n_species=5, n_predictors=2, seed=404)
    dv[0, :] = 0
    iv_df = pd.DataFrame(iv, columns=["E1", "E2"])

    result = rdacca_hp(
        dv=dv,
        iv=iv_df,
        method="CCA",
        type="R2",
        var_part=False,
        n_perm=19,
    )

    assert np.isfinite(result.total_explained_variation)