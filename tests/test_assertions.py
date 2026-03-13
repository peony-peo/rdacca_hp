import numpy as np
import pandas as pd
import pytest

from rdacca_hp.core import rdacca_hp
from rdacca_hp.utils import (
    check_data_quality,
    create_binary_matrix,
    genList,
    preprocess_predictor_dataframe,
    remove_index_like_columns,
    sanitize_tabular_input,
)


def test_binary_matrix_matches_expected_for_three_predictors():
    expected = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ])
    observed = create_binary_matrix(3)
    np.testing.assert_array_equal(observed, expected)


def test_genlist_sign_logic_matches_expected_example():
    assert genList([1, -2, 4], -8) == [-9, 10, -12]
    assert genList([1, -2, 4], 8) == [9, -10, 12]


def test_remove_index_like_columns_drops_only_true_index_columns():
    df = pd.DataFrame({
        "Unnamed: 0": [0, 1, 2],
        "value": [10, 20, 30],
        "index_like_but_not_really": [10, 11, 12],
    })

    cleaned = remove_index_like_columns(df, warn=False)

    assert "Unnamed: 0" not in cleaned.columns
    assert "value" in cleaned.columns
    assert "index_like_but_not_really" in cleaned.columns


def test_sanitize_tabular_input_recurses_into_dict_and_list():
    df1 = pd.DataFrame({"Unnamed: 0": [0, 1], "A": [1.0, 2.0]})
    df2 = pd.DataFrame({"row.names": [1, 2], "B": [3.0, 4.0]})

    out_dict = sanitize_tabular_input({"g1": df1}, warn=False)
    out_list = sanitize_tabular_input([df2], warn=False)

    assert list(out_dict["g1"].columns) == ["A"]
    assert list(out_list[0].columns) == ["B"]


def test_preprocess_predictor_dataframe_encodes_numeric_unordered_and_ordered():
    iv = pd.DataFrame({
        "WatrCont": [1.0, 2.0, 3.0, 4.0],
        "Substrate": ["A", "B", "A", "C"],
        "Shrub": ["None", "Few", "Many", "Few"],
    })

    combined, encoded_groups, group_columns = preprocess_predictor_dataframe(
        iv,
        categorical_factors=["Substrate"],
        ordered_factors={"Shrub": ["None", "Few", "Many"]},
        warn=False,
    )

    assert "WatrCont" in encoded_groups
    assert encoded_groups["WatrCont"].shape[1] == 1

    assert "Substrate" in encoded_groups
    assert encoded_groups["Substrate"].shape[1] == 2  # drop_first dummy coding

    assert "Shrub" in encoded_groups
    assert encoded_groups["Shrub"].shape[1] == 2  # polynomial contrasts for 3 levels

    assert combined.shape[0] == 4
    assert set(group_columns.keys()) == {"WatrCont", "Substrate", "Shrub"}


def test_preprocess_predictor_dataframe_rejects_unknown_ordered_levels():
    iv = pd.DataFrame({
        "Shrub": ["None", "Few", "Unknown", "Many"]
    })

    with pytest.raises(ValueError, match="outside declared levels"):
        preprocess_predictor_dataframe(
            iv,
            ordered_factors={"Shrub": ["None", "Few", "Many"]},
            warn=False,
        )


def test_check_data_quality_rejects_nan_and_inf():
    dv = np.array([[1.0], [2.0], [np.nan]])
    iv = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        check_data_quality(dv, iv)

    dv2 = np.array([[1.0], [2.0], [3.0]])
    iv2 = np.array([[1.0, 0.0], [0.0, np.inf], [1.0, 1.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        check_data_quality(dv2, iv2)


def test_check_data_quality_rejects_row_mismatch_for_dataframe_iv():
    dv = np.array([[1.0], [2.0], [3.0]])
    iv = pd.DataFrame({"A": [1.0, 2.0]})

    with pytest.raises(ValueError, match="same number of rows"):
        check_data_quality(dv, iv)


def test_hierarchical_partitioning_internal_identities_hold():
    rng = np.random.RandomState(42)
    X = rng.randn(80, 3)
    y = (0.8 * X[:, [0]] - 0.3 * X[:, [1]] + 0.2 * X[:, [2]] + 0.05 * rng.randn(80, 1))

    result = rdacca_hp(
        dv=y,
        iv=pd.DataFrame(X, columns=["A", "B", "C"]),
        method="RDA",
        type="R2",
        var_part=True,
    )

    hier = result.hier_part

    np.testing.assert_allclose(
        hier["Unique"].values + hier["Average.share"].values,
        hier["Individual"].values,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        hier["Individual"].sum(),
        result.total_explained_variation,
        atol=2e-4,
    )

    np.testing.assert_allclose(
        result.var_part.loc["Total", "Fractions"],
        result.total_explained_variation,
        atol=5e-4,
    )