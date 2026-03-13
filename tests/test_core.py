import io

import numpy as np
import pandas as pd
import pytest

from rdacca_hp.core import RdaccaHpResult, rdacca_hp
from rdacca_hp.utils import create_test_data


def test_rda_with_numeric_dataframe_returns_result_object():
    dv, iv = create_test_data(n_samples=60, n_predictors=3, n_responses=2, seed=123)
    iv_df = pd.DataFrame(iv, columns=["A", "B", "C"])

    result = rdacca_hp(
        dv=dv,
        iv=iv_df,
        method="RDA",
        type="adjR2",
        var_part=True,
    )

    assert isinstance(result, RdaccaHpResult)
    assert result.method_type == ["RDA", "adjR2"]
    assert np.isfinite(result.total_explained_variation)
    assert result.hier_part.shape == (3, 4)
    assert result.var_part is not None
    assert "Total" in result.var_part.index


def test_rda_accepts_dataframe_with_categorical_and_ordered_factors():
    rng = np.random.RandomState(7)
    n = 50

    iv = pd.DataFrame({
        "X1": rng.normal(size=n),
        "Substrate": rng.choice(["A", "B", "C"], size=n),
        "Shrub": pd.Categorical(
            rng.choice(["None", "Few", "Many"], size=n),
            categories=["None", "Few", "Many"],
            ordered=True,
        ),
    })

    y = (
        0.7 * iv["X1"].to_numpy()
        + (iv["Substrate"] == "B").astype(float).to_numpy() * 0.3
        + (iv["Shrub"].astype(str) == "Many").astype(float).to_numpy() * 0.2
        + rng.normal(scale=0.1, size=n)
    )
    dv = y.reshape(-1, 1)

    result = rdacca_hp(
        dv=dv,
        iv=iv,
        method="RDA",
        type="R2",
        var_part=True,
        categorical_factors=["Substrate"],
        ordered_factors={"Shrub": ["None", "Few", "Many"]},
    )

    assert list(result.hier_part.index) == ["X1", "Substrate", "Shrub"]
    assert result.var_part is not None
    assert np.isfinite(result.total_explained_variation)


def test_rda_accepts_grouped_dict_predictors():
    dv, iv = create_test_data(n_samples=50, n_predictors=4, n_responses=2, seed=11)

    groups = {
        "Climate": pd.DataFrame(iv[:, :2], columns=["T", "P"]),
        "Soil": pd.DataFrame(iv[:, 2:], columns=["N", "C"]),
    }

    result = rdacca_hp(
        dv=dv,
        iv=groups,
        method="RDA",
        type="R2",
        var_part=True,
    )

    assert list(result.hier_part.index) == ["Climate", "Soil"]
    assert result.var_part is not None
    assert result.var_part.index[-1] == "Total"


def test_rda_accepts_grouped_list_predictors():
    dv, iv = create_test_data(n_samples=50, n_predictors=4, n_responses=2, seed=12)

    groups = [
        pd.DataFrame(iv[:, :2], columns=["T", "P"]),
        pd.DataFrame(iv[:, 2:], columns=["N", "C"]),
    ]

    result = rdacca_hp(
        dv=dv,
        iv=groups,
        method="RDA",
        type="R2",
        var_part=False,
    )

    assert result.var_part is None
    assert result.hier_part.shape[0] == 2


def test_rdacca_hp_invalid_method_raises():
    dv, iv = create_test_data(n_samples=30, n_predictors=2, n_responses=1, seed=3)

    with pytest.raises(ValueError, match="method must be"):
        rdacca_hp(dv=dv, iv=iv, method="BAD", type="R2")


def test_rdacca_hp_invalid_type_raises():
    dv, iv = create_test_data(n_samples=30, n_predictors=2, n_responses=1, seed=3)

    with pytest.raises(ValueError, match="type must be"):
        rdacca_hp(dv=dv, iv=iv, method="RDA", type="BAD")


def test_result_repr_and_summary_work(capsys):
    dv, iv = create_test_data(n_samples=40, n_predictors=3, n_responses=2, seed=21)
    iv_df = pd.DataFrame(iv, columns=["A", "B", "C"])

    result = rdacca_hp(dv=dv, iv=iv_df, method="RDA", type="R2", var_part=True)

    text = repr(result)
    assert "RdaccaHpResult" in text
    assert "RDA" in text

    result.summary()
    captured = capsys.readouterr().out
    assert "Method:" in captured
    assert "Hierarchical Partitioning:" in captured
    assert "Variation Partitioning:" in captured