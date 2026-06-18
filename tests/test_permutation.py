import numpy as np
import pandas as pd

from rdacca_hp.permutation import (
    _calculate_p_values,
    _permute_variables,
    permu_hp,
)
from rdacca_hp.utils import create_test_data


def test_permute_variables_preserves_dataframe_structure():
    iv = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "cat": ["A", "B", "A", "C"],
    })

    out = _permute_variables(iv, n_samples=len(iv))

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["num", "cat"]
    assert out.shape == iv.shape
    assert sorted(out["cat"].tolist()) == sorted(iv["cat"].tolist())


def test_permute_variables_preserves_grouped_dict_structure():
    iv = {
        "g1": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
        "g2": np.array([[10, 20], [30, 40], [50, 60]]),
    }

    out = _permute_variables(iv, n_samples=3)

    assert set(out.keys()) == {"g1", "g2"}
    assert isinstance(out["g1"], pd.DataFrame)
    assert out["g1"].shape == (3, 2)
    assert out["g2"].shape == (3, 2)


def test_calculate_p_values_matches_r_style_ecdf_formula():
    obs = np.array([0.5, 0.2])
    perm = np.array([
        [0.4, 0.1],
        [0.6, 0.2],
        [0.5, 0.3],
    ])

    p = _calculate_p_values(obs, perm, n_perm=3)

    expected = []
    for j in range(len(obs)):
        x = np.concatenate(([obs[j]], perm[:, j]))
        ecdf_at_obs = np.mean(x <= obs[j])
        expected.append(1 - ecdf_at_obs + 1 / (perm.shape[0] + 1))

    np.testing.assert_allclose(p, np.array(expected))

def test_permu_hp_returns_expected_columns_for_rda():
    dv, iv = create_test_data(n_samples=40, n_predictors=3, n_responses=2, seed=10)
    iv_df = pd.DataFrame(iv, columns=["A", "B", "C"])

    result = permu_hp(
        dv=dv,
        iv=iv_df,
        method="RDA",
        type="R2",
        permutations=19,
        verbose=False,
        random_state=123,
    )

    expected_cols = ["Unique", "Average.share", "Individual", "I.perc(%)", "Pr(>I)", "Significance"]
    assert list(result.columns) == expected_cols
    assert result.shape[0] == 3
    assert ((result["Pr(>I)"] >= 0) & (result["Pr(>I)"] <= 1)).all()


def test_permu_hp_is_reproducible_given_random_state():
    dv, iv = create_test_data(n_samples=36, n_predictors=3, n_responses=2, seed=22)
    iv_df = pd.DataFrame(iv, columns=["A", "B", "C"])

    r1 = permu_hp(
        dv=dv,
        iv=iv_df,
        method="RDA",
        type="R2",
        permutations=15,
        verbose=False,
        random_state=999,
    )

    r2 = permu_hp(
        dv=dv,
        iv=iv_df,
        method="RDA",
        type="R2",
        permutations=15,
        verbose=False,
        random_state=999,
    )

    pd.testing.assert_frame_equal(r1, r2)


def test_permu_hp_runs_with_categorical_and_ordered_predictors():
    rng = np.random.RandomState(1234)
    n = 36

    iv = pd.DataFrame({
        "X1": rng.normal(size=n),
        "Substrate": rng.choice(["A", "B", "C"], size=n),
        "Shrub": rng.choice(["None", "Few", "Many"], size=n),
    })

    y = (
        0.6 * iv["X1"].to_numpy()
        + (iv["Substrate"] == "B").astype(float).to_numpy() * 0.2
        + (iv["Shrub"] == "Many").astype(float).to_numpy() * 0.2
        + rng.normal(scale=0.15, size=n)
    )
    dv = y.reshape(-1, 1)

    result = permu_hp(
        dv=dv,
        iv=iv,
        method="RDA",
        type="R2",
        permutations=9,
        verbose=False,
        random_state=7,
        categorical_factors=["Substrate"],
        ordered_factors={"Shrub": ["None", "Few", "Many"]},
    )

    assert list(result.index) == ["X1", "Substrate", "Shrub"]
    assert "Pr(>I)" in result.columns