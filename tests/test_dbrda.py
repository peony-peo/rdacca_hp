import numpy as np
import pandas as pd
import pytest

from scipy.spatial.distance import squareform

from rdacca_hp.core import calculate_dbrda, rdacca_hp
from rdacca_hp.utils import create_distance_test_data, create_test_data


def test_calculate_dbrda_returns_finite_values():
    dv_dist = create_distance_test_data(n_samples=30, n_species=10, seed=123)
    _, iv = create_test_data(n_samples=30, n_predictors=3, n_responses=1, seed=123)

    r2 = calculate_dbrda(dv_dist, iv, type="R2")
    adj_r2 = calculate_dbrda(dv_dist, iv, type="adjR2")

    assert np.isfinite(r2)
    assert np.isfinite(adj_r2)


def test_rdacca_hp_dbrda_runs_with_dataframe_predictors():
    dv_dist = create_distance_test_data(n_samples=30, n_species=10, seed=321)
    _, iv = create_test_data(n_samples=30, n_predictors=3, n_responses=1, seed=321)
    iv_df = pd.DataFrame(iv, columns=["A", "B", "C"])

    result = rdacca_hp(
        dv=dv_dist,
        iv=iv_df,
        method="dbRDA",
        type="adjR2",
        var_part=True,
        add=True,
        sqrt_dist=True,
        n_axes=5,
    )

    assert result.method_type == ["DBRDA", "adjR2"]
    assert result.var_part is not None
    assert result.hier_part.shape == (3, 4)
    assert np.isfinite(result.total_explained_variation)


def test_rdacca_hp_dbrda_accepts_grouped_predictors():
    dv_dist = create_distance_test_data(n_samples=24, n_species=8, seed=99)
    _, iv = create_test_data(n_samples=24, n_predictors=4, n_responses=1, seed=99)

    groups = {
        "Climate": pd.DataFrame(iv[:, :2], columns=["T", "P"]),
        "Soil": pd.DataFrame(iv[:, 2:], columns=["N", "C"]),
    }

    result = rdacca_hp(
        dv=dv_dist,
        iv=groups,
        method="dbRDA",
        type="R2",
        var_part=False,
        add=True,
    )

    assert list(result.hier_part.index) == ["Climate", "Soil"]
    assert result.var_part is None


def test_rdacca_hp_dbrda_rejects_invalid_distance_matrix():
    bad = np.array([[0.0, 1.0], [2.0, 0.0]])  # not symmetric
    iv = np.array([[1.0], [2.0]])

    with pytest.raises(ValueError, match="square symmetric distance matrix"):
        rdacca_hp(dv=bad, iv=iv, method="dbRDA", type="R2")

def test_calculate_dbrda_accepts_condensed_distance_vector():
    dv_dist = create_distance_test_data(n_samples=20, n_species=8, seed=202)
    dv_condensed = squareform(dv_dist)
    _, iv = create_test_data(n_samples=20, n_predictors=3, n_responses=1, seed=202)

    r2 = calculate_dbrda(dv_condensed, iv, type="R2")
    adj_r2 = calculate_dbrda(dv_condensed, iv, type="adjR2")

    assert np.isfinite(r2)
    assert np.isfinite(adj_r2)


def test_rdacca_hp_dbrda_accepts_condensed_distance_vector():
    dv_dist = create_distance_test_data(n_samples=20, n_species=8, seed=303)
    dv_condensed = squareform(dv_dist)
    _, iv = create_test_data(n_samples=20, n_predictors=3, n_responses=1, seed=303)

    result = rdacca_hp(
        dv=dv_condensed,
        iv=iv,
        method="dbRDA",
        type="adjR2",
        var_part=True,
        add=True,
        sqrt_dist=False,
    )

    assert result.method_type == ["DBRDA", "adjR2"]
    assert result.var_part is not None
    assert np.isfinite(result.total_explained_variation)