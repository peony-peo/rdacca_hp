import numpy as np
import pandas as pd
import pytest

import rdacca_hp.permutation as pm
from rdacca_hp import permu_hp, rdacca_hp
from rdacca_hp.core import (
    _fill_commonality_values,
    _get_hp_cached_structures,
    _get_hp_individual_r2_weights,
)


def _legacy_permu_rda(iv, dv, *, type="adjR2", scale=False, permutations=13,
                      random_state=123, ordered_factors=None,
                      categorical_factors=None):
    """Reconstruct the pre-batch RDA permutation loop using current reference helpers."""
    ordered_factors = ordered_factors or {}
    categorical_factors = categorical_factors or []
    iv_is_dataframe = isinstance(iv, pd.DataFrame)

    obs_result = rdacca_hp(
        dv=dv,
        iv=iv,
        method="RDA",
        type=type,
        scale=scale,
        var_part=False,
        ordered_factors=ordered_factors,
        categorical_factors=categorical_factors,
    )
    obs_individual = obs_result.hier_part["Individual"].to_numpy(dtype=float)

    perm_base_iv, perm_engine = pm._build_permutation_engine(
        iv=iv,
        method="RDA",
        type=type,
        scale=scale,
        n_perm=1000,
        add=False,
        sqrt_dist=False,
        n_axes=None,
        dbrdatype="dbrda",
        ordered_factors=ordered_factors,
        categorical_factors=categorical_factors,
        kwargs={},
        cca_rng=None,
    )

    rng = np.random.default_rng(random_state)
    n_samples = pm._infer_n_samples(dv, perm_base_iv)
    values = np.full((permutations - 1, len(obs_individual)), np.nan, dtype=float)

    for i in range(permutations - 1):
        if iv_is_dataframe and isinstance(perm_base_iv, dict):
            permuted_iv = pm._permute_grouped_dict_independently(
                perm_base_iv, n_samples, rng
            )
        else:
            permuted_iv = pm._permute_variables(perm_base_iv, n_samples, rng)

        result = perm_engine(dv, permuted_iv)
        values[i, :] = result.hier_part["Individual"].to_numpy(dtype=float)

    p_values = pm._calculate_p_values(
        obs_values=obs_individual,
        perm_values=values,
        n_perm=permutations,
    )
    return pm._create_result_dataframe(obs_result, p_values)


def test_hp_direct_r2_to_individual_matches_full_commonality():
    rng = np.random.default_rng(20261005)

    for n_items in range(2, 8):
        binary_matrix, total_combinations, bit_counts, _, _, commonlist = (
            _get_hp_cached_structures(n_items)
        )
        weights = _get_hp_individual_r2_weights(n_items)

        for _ in range(10):
            r2_values = rng.normal(size=total_combinations)
            common_m = np.zeros((total_combinations, 3), dtype=float)
            common_m[:, 1] = r2_values
            _fill_commonality_values(common_m, commonlist)

            legacy = np.empty(n_items, dtype=float)
            for j in range(n_items):
                legacy[j] = np.sum(
                    binary_matrix[j, :] * (common_m[:, 2] / (bit_counts + 1e-10))
                )

            direct = weights @ r2_values
            np.testing.assert_allclose(direct, legacy, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "type,scale",
    [("R2", False), ("adjR2", False), ("adjR2", True)],
)
def test_rda_batched_numeric_dataframe_matches_legacy(type, scale):
    rng = np.random.default_rng(101)
    n = 24
    iv = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("ABCD"))
    beta = rng.normal(size=(4, 5))
    dv = iv.to_numpy() @ beta + rng.normal(scale=0.35, size=(n, 5))

    kwargs = dict(
        dv=dv,
        iv=iv,
        method="RDA",
        type=type,
        scale=scale,
        permutations=13,
        verbose=False,
        random_state=771,
    )
    fast = permu_hp(**kwargs)
    legacy = _legacy_permu_rda(
        iv=iv,
        dv=dv,
        type=type,
        scale=scale,
        permutations=13,
        random_state=771,
    )
    pd.testing.assert_frame_equal(fast, legacy)


def test_rda_batched_grouped_predictors_matches_legacy():
    rng = np.random.default_rng(202)
    n = 22
    g1 = pd.DataFrame(rng.normal(size=(n, 2)), columns=["A1", "A2"])
    g2 = pd.DataFrame(rng.normal(size=(n, 1)), columns=["B"])
    g3 = pd.DataFrame(rng.normal(size=(n, 1)), columns=["C"])
    iv = {"Environment": g1, "Space": g2, "Other": g3}
    x = np.column_stack([g1.to_numpy(), g2.to_numpy(), g3.to_numpy()])
    dv = x @ rng.normal(size=(4, 3)) + rng.normal(scale=0.3, size=(n, 3))

    fast = permu_hp(
        dv=dv, iv=iv, method="RDA", type="adjR2",
        permutations=11, verbose=False, random_state=99,
    )
    legacy = _legacy_permu_rda(
        iv=iv, dv=dv, type="adjR2", permutations=11, random_state=99,
    )
    pd.testing.assert_frame_equal(fast, legacy)


def test_rda_batched_factor_groups_match_legacy():
    rng = np.random.default_rng(303)
    n = 28
    iv = pd.DataFrame({
        "X1": rng.normal(size=n),
        "Substrate": rng.choice(["A", "B", "C"], size=n),
        "Shrub": rng.choice(["None", "Few", "Many"], size=n),
    })
    y = (
        0.7 * iv["X1"].to_numpy()
        + 0.3 * (iv["Substrate"] == "B").to_numpy(dtype=float)
        + 0.2 * (iv["Shrub"] == "Many").to_numpy(dtype=float)
        + rng.normal(scale=0.2, size=n)
    )[:, None]

    ordered = {"Shrub": ["None", "Few", "Many"]}
    categorical = ["Substrate"]
    fast = permu_hp(
        dv=y,
        iv=iv,
        method="RDA",
        type="R2",
        permutations=9,
        verbose=False,
        random_state=18,
        ordered_factors=ordered,
        categorical_factors=categorical,
    )
    legacy = _legacy_permu_rda(
        iv=iv,
        dv=y,
        type="R2",
        permutations=9,
        random_state=18,
        ordered_factors=ordered,
        categorical_factors=categorical,
    )
    pd.testing.assert_frame_equal(fast, legacy)


def test_rda_rank_deficient_fallback_matches_legacy():
    rng = np.random.default_rng(404)
    n = 24
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    iv = pd.DataFrame({"A": a, "B": b, "A_copy": a})
    dv = np.column_stack([
        0.6 * a - 0.2 * b + rng.normal(scale=0.3, size=n),
        -0.1 * a + 0.5 * b + rng.normal(scale=0.3, size=n),
    ])

    fast = permu_hp(
        dv=dv, iv=iv, method="RDA", type="adjR2",
        permutations=9, verbose=False, random_state=51,
    )
    legacy = _legacy_permu_rda(
        iv=iv, dv=dv, type="adjR2", permutations=9, random_state=51,
    )
    pd.testing.assert_frame_equal(fast, legacy)
