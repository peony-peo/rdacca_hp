import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

import rdacca_hp.permutation as pm
from rdacca_hp import permu_hp, rdacca_hp
from rdacca_hp.core import (
    _dbrda_subset_r2_values_fast,
    _get_hp_cached_structures,
    calculate_dbrda,
)
from rdacca_hp.utils import prepare_dbrda_response


def _make_data(seed=2026, n=20, p=4, q=7):
    rng = np.random.default_rng(seed)
    y = rng.gamma(2.0, 1.0, size=(n, q))
    y[rng.random(y.shape) < 0.18] = 0.0
    x = rng.normal(size=(n, p))
    iv = pd.DataFrame(x, columns=[f"X{i+1}" for i in range(p)])
    return y, iv


def _legacy_permu_dbrda(iv, dv, *, type="adjR2", permutations=11,
                         random_state=123, add=False, sqrt_dist=False,
                         distance=None, ordered_factors=None,
                         categorical_factors=None):
    """Reconstruct the pre-dbRDA-Round-2 permutation loop."""
    ordered_factors = ordered_factors or {}
    categorical_factors = categorical_factors or []
    iv_is_dataframe = isinstance(iv, pd.DataFrame)
    dv_prepared = prepare_dbrda_response(dv, distance=distance)

    obs_result = rdacca_hp(
        dv=dv_prepared,
        iv=iv,
        method="dbRDA",
        type=type,
        var_part=False,
        add=add,
        sqrt_dist=sqrt_dist,
        dbrdatype="dbrda",
        ordered_factors=ordered_factors,
        categorical_factors=categorical_factors,
        distance=None,
    )
    obs_individual = obs_result.hier_part["Individual"].to_numpy(dtype=float)

    perm_base_iv, perm_engine = pm._build_permutation_engine(
        iv=iv,
        method="DBRDA",
        type=type,
        scale=False,
        n_perm=1000,
        add=add,
        sqrt_dist=sqrt_dist,
        n_axes=None,
        dbrdatype="dbrda",
        ordered_factors=ordered_factors,
        categorical_factors=categorical_factors,
        kwargs={},
        cca_rng=None,
    )

    rng = np.random.default_rng(random_state)
    n_samples = pm._infer_n_samples(dv_prepared, perm_base_iv)
    values = np.full((permutations - 1, len(obs_individual)), np.nan, dtype=float)

    for i in range(permutations - 1):
        if iv_is_dataframe and isinstance(perm_base_iv, dict):
            permuted_iv = pm._permute_grouped_dict_independently(
                perm_base_iv, n_samples, rng
            )
        else:
            permuted_iv = pm._permute_variables(perm_base_iv, n_samples, rng)

        result = perm_engine(dv_prepared, permuted_iv)
        values[i, :] = result.hier_part["Individual"].to_numpy(dtype=float)

    p_values = pm._calculate_p_values(
        obs_values=obs_individual,
        perm_values=values,
        n_perm=permutations,
    )
    return pm._create_result_dataframe(obs_result, p_values)


@pytest.mark.parametrize("type", ["R2", "adjR2"])
@pytest.mark.parametrize(
    "add,sqrt_dist",
    [(False, False), (False, True), ("lingoes", False), ("cailliez", False)],
)
def test_dbrda_subset_fast_matches_legacy(type, add, sqrt_dist):
    y, iv = _make_data(seed=11, n=16, p=3, q=6)
    dv = squareform(pdist(y, metric="braycurtis"))
    x = iv.to_numpy(dtype=float)
    groups = [x[:, [j]] for j in range(x.shape[1])]
    _, _, _, _, combo_indices, _ = _get_hp_cached_structures(x.shape[1])

    fast = _dbrda_subset_r2_values_fast(
        dv, groups, combo_indices, type=type, add=add, sqrt_dist=sqrt_dist
    )

    legacy = np.empty_like(fast)
    for i, subset in enumerate(combo_indices):
        legacy[i] = calculate_dbrda(
            dv,
            x[:, subset],
            type=type,
            add=add,
            sqrt_dist=sqrt_dist,
            dbrdatype="dbrda",
        )

    np.testing.assert_allclose(fast, legacy, rtol=0.0, atol=1e-10)


def test_dbrda_batched_condensed_matches_legacy_fixed_seed():
    y, iv = _make_data(seed=22, n=21, p=4, q=7)
    dv = pdist(y, metric="braycurtis")

    fast = permu_hp(
        dv=dv, iv=iv, method="dbRDA", type="adjR2",
        permutations=13, verbose=False, random_state=707,
    )
    legacy = _legacy_permu_dbrda(
        iv=iv, dv=dv, type="adjR2", permutations=13, random_state=707,
    )
    pd.testing.assert_frame_equal(fast, legacy)


def test_dbrda_batched_raw_bray_matches_legacy_fixed_seed():
    y, iv = _make_data(seed=33, n=19, p=3, q=6)

    fast = permu_hp(
        dv=y,
        iv=iv,
        method="dbRDA",
        type="R2",
        distance="bray",
        permutations=11,
        verbose=False,
        random_state=808,
    )
    legacy = _legacy_permu_dbrda(
        iv=iv,
        dv=y,
        type="R2",
        distance="bray",
        permutations=11,
        random_state=808,
    )
    pd.testing.assert_frame_equal(fast, legacy)


@pytest.mark.parametrize(
    "add,sqrt_dist",
    [(False, True), ("lingoes", False)],
)
def test_dbrda_batched_options_match_legacy_fixed_seed(add, sqrt_dist):
    y, iv = _make_data(seed=44, n=18, p=3, q=5)
    dv = pdist(y, metric="braycurtis")

    fast = permu_hp(
        dv=dv,
        iv=iv,
        method="dbRDA",
        type="adjR2",
        add=add,
        sqrt_dist=sqrt_dist,
        permutations=9,
        verbose=False,
        random_state=919,
    )
    legacy = _legacy_permu_dbrda(
        iv=iv,
        dv=dv,
        type="adjR2",
        add=add,
        sqrt_dist=sqrt_dist,
        permutations=9,
        random_state=919,
    )
    pd.testing.assert_frame_equal(fast, legacy)


def test_dbrda_batched_grouped_predictors_matches_legacy():
    y, base = _make_data(seed=55, n=20, p=4, q=6)
    dv = pdist(y, metric="braycurtis")
    iv = {
        "Environment": base[["X1", "X2"]],
        "Space": base[["X3"]],
        "Other": base[["X4"]],
    }

    fast = permu_hp(
        dv=dv, iv=iv, method="dbRDA", type="adjR2",
        permutations=9, verbose=False, random_state=1001,
    )
    legacy = _legacy_permu_dbrda(
        iv=iv, dv=dv, type="adjR2", permutations=9, random_state=1001,
    )
    pd.testing.assert_frame_equal(fast, legacy)


def test_dbrda_rank_deficient_fallback_matches_legacy():
    y, base = _make_data(seed=66, n=20, p=2, q=6)
    a = base["X1"].to_numpy()
    b = base["X2"].to_numpy()
    iv = pd.DataFrame({"A": a, "B": b, "A_copy": a})
    dv = pdist(y, metric="braycurtis")

    fast = permu_hp(
        dv=dv, iv=iv, method="dbRDA", type="adjR2",
        permutations=9, verbose=False, random_state=111,
    )
    legacy = _legacy_permu_dbrda(
        iv=iv, dv=dv, type="adjR2", permutations=9, random_state=111,
    )
    pd.testing.assert_frame_equal(fast, legacy)


def test_dbrda_fastpath_failure_propagates_by_default(monkeypatch):
    y, iv = _make_data(seed=77, n=18, p=3, q=5)
    dv = pdist(y, metric="braycurtis")

    def fail_batched_dbrda(**kwargs):
        raise ValueError("forced dbRDA permutation failure")

    monkeypatch.setattr(
        pm,
        "_dbrda_permutation_individuals_batched",
        fail_batched_dbrda,
    )

    with pytest.raises(ValueError, match="forced dbRDA permutation failure"):
        permu_hp(
            dv=dv,
            iv=iv,
            method="dbRDA",
            type="R2",
            permutations=5,
            verbose=False,
            random_state=12,
        )
