import numpy as np
import pandas as pd
from typing import Union, List, Dict
import time

from .core import rdacca_hp, _rdacca_hp_multi, _rdacca_hp_single
from .utils import (
    sanitize_tabular_input,
    preprocess_predictor_dataframe,
    preprocess_grouped_predictors,
    prepare_dbrda_response,
)


def _build_permutation_engine(iv, method, type, scale, n_perm, add, sqrt_dist, n_axes, dbrdatype,
                              ordered_factors, categorical_factors, kwargs,
                              cca_rng=None):
    """
    Build a lightweight evaluator for permutation runs without changing the public API
    or output structure.
    """
    core_kwargs = dict(kwargs)
    core_kwargs["n_perm"] = n_perm
    core_kwargs["add"] = add
    core_kwargs["sqrt_dist"] = sqrt_dist
    core_kwargs["n_axes"] = n_axes
    core_kwargs["dbrdatype"] = dbrdatype
    core_kwargs["ordered_factors"] = ordered_factors
    core_kwargs["categorical_factors"] = categorical_factors
    if method == "CCA" and type == "adjR2":
        core_kwargs["_cca_rng"] = (
            cca_rng if cca_rng is not None else np.random.default_rng()
        )

    # Single DataFrame input in rdacca_hp() is internally converted into logical
    # predictor groups (including categorical / ordered factor expansions).
    # Do that once here, then only permute the already-prepared groups.
    if isinstance(iv, pd.DataFrame):
        _, encoded_groups, _ = preprocess_predictor_dataframe(
            iv,
            ordered_factors=ordered_factors,
            categorical_factors=categorical_factors,
            warn=False,
        )

        def evaluate(dv_current, iv_current):
            return _rdacca_hp_multi(
                dv=dv_current,
                iv=iv_current,
                method=method,
                type=type,
                scale=scale,
                var_part=False,
                **core_kwargs,
            )

        return encoded_groups, evaluate

    if isinstance(iv, (dict, list)):
        grouped_iv = preprocess_grouped_predictors(
            iv,
            ordered_factors=ordered_factors,
            categorical_factors=categorical_factors,
            warn=False,
        )

        def evaluate(dv_current, iv_current):
            return _rdacca_hp_multi(
                dv=dv_current,
                iv=iv_current,
                method=method,
                type=type,
                scale=scale,
                var_part=False,
                **core_kwargs,
            )

        return grouped_iv, evaluate

    def evaluate(dv_current, iv_current):
        return _rdacca_hp_single(
            dv=dv_current,
            iv=iv_current,
            method=method,
            type=type,
            scale=scale,
            var_part=False,
            **core_kwargs,
        )

    return iv, evaluate


def permu_hp(dv: Union[np.ndarray, pd.DataFrame],
             iv: Union[np.ndarray, pd.DataFrame, List, Dict],
             method: str = "RDA",
             type: str = "adjR2",
             permutations: int = 1000,
             scale: bool = False,
             n_perm: int = 1000,          # for CCA adjR2
             add: bool | str = False,     # False, True/"lingoes", or "cailliez"
             sqrt_dist: bool = False,     # for dbRDA
             n_axes: int = None,          # for dbRDA
             ordered_factors: dict | None = None,
             categorical_factors: list | None = None,
             verbose: bool = True,
             random_state: int | None = None,
             dbrdatype: str = "dbrda",
             distance: str | None = None,
             skip_failed: bool = False,
             **kwargs) -> pd.DataFrame:
    """
    Permutation test for hierarchical partitioning.

    This follows rdacca.hp::permu.hp semantics:
    - ``permutations`` is the total number of values used for the empirical distribution.
    - The observed value is included once, so only ``permutations - 1`` randomized runs
      are performed.
    - For DataFrame input, each original predictor is permuted independently.
    - For dict/list grouped input, all groups are permuted using the same row order.
    - Supplying ``distance`` treats ``dv`` as raw response data and computes the
      db-RDA distance matrix once before the permutation loop.
    - The result matches R ``permu.hp()`` with ``Individual`` and ``Pr(>I)``
      columns. Significance stars are included in the character ``Pr(>I)``
      values using the same spacing rules as R.
    - By default a failed permutation stops the analysis, as in R. Set
      ``skip_failed=True`` to omit failed permutations.
    """

    method = method.upper()
    if method not in {"RDA", "CCA", "DBRDA"}:
        raise ValueError("method must be 'RDA', 'CCA', or 'dbRDA'")

    if type not in {"R2", "adjR2"}:
        raise ValueError("type must be 'R2' or 'adjR2'")

    permutations = int(permutations)
    if permutations < 2:
        raise ValueError("permutations must be at least 2")

    ordered_factors = ordered_factors or {}
    categorical_factors = categorical_factors or []
    rng = np.random.default_rng(random_state)
    cca_rng = rng if method == "CCA" and type == "adjR2" else None

    dv = sanitize_tabular_input(dv, warn=verbose)
    iv = sanitize_tabular_input(iv, warn=verbose)
    iv_is_dataframe_input = isinstance(iv, pd.DataFrame)

    # Prepare distances only once. Recomputing them in every randomized run
    # would be unnecessary because permu.hp permutes iv, not dv.
    dv_for_permutation = (
        prepare_dbrda_response(dv, distance=distance)
        if method == "DBRDA"
        else dv
    )

    n_random = permutations - 1

    if verbose:
        print(f"Please wait: running {n_random} permutations")
        start_time = time.perf_counter()

    # ---- observed result ----
    obs_result = rdacca_hp(
        dv=dv_for_permutation,
        iv=iv,
        method=method,
        type=type,
        scale=scale,
        var_part=False,
        n_perm=n_perm,
        add=add,
        sqrt_dist=sqrt_dist,
        n_axes=n_axes,
        dbrdatype=dbrdatype,
        distance=None,
        ordered_factors=ordered_factors,
        categorical_factors=categorical_factors,
        random_state=cca_rng,
        **kwargs
    )

    obs_individual = obs_result.hier_part["Individual"].to_numpy(dtype=float)
    n_vars = len(obs_individual)

    # ---- build a lightweight internal evaluator for permutations ----
    perm_base_iv, perm_engine = _build_permutation_engine(
        iv=iv,
        method=method,
        type=type,
        scale=scale,
        n_perm=n_perm,
        add=add,
        sqrt_dist=sqrt_dist,
        n_axes=n_axes,
        dbrdatype=dbrdatype,
        ordered_factors=ordered_factors,
        categorical_factors=categorical_factors,
        kwargs=kwargs,
        cca_rng=cca_rng,
    )

    # ---- initialize ----
    perm_individual = np.full((n_random, n_vars), np.nan, dtype=float)
    failed_perms = 0

    # Always infer n from the explanatory variables, not from dv.
    # This is essential for dbRDA when dv is a condensed distance vector.
    n_samples = _infer_n_samples(dv_for_permutation, perm_base_iv)

    # ---- permutation loop ----
    for i in range(n_random):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_random} permutations")

        if iv_is_dataframe_input and isinstance(perm_base_iv, dict):
            permuted_iv = _permute_grouped_dict_independently(perm_base_iv, n_samples, rng)
        else:
            permuted_iv = _permute_variables(perm_base_iv, n_samples, rng)

        try:
            perm_result = perm_engine(dv_for_permutation, permuted_iv)
            perm_individual[i, :] = perm_result.hier_part["Individual"].to_numpy(dtype=float)

        except Exception as e:
            if not skip_failed:
                raise
            failed_perms += 1
            if verbose:
                print(f"Warning: permutation {i + 1} failed: {e}")

    # ---- summarize failures ----
    if verbose:
        print(f"Failed permutations: {failed_perms}/{n_random}")

    valid_mask = ~np.isnan(perm_individual).any(axis=1)
    valid_perm = perm_individual[valid_mask]

    if valid_perm.shape[0] == 0:
        raise RuntimeError(
            "All permutations failed. "
            "Please inspect categorical/ordered factor preprocessing and permutation handling."
        )

    # ---- p-values ----
    p_values = _calculate_p_values(
        obs_values=obs_individual,
        perm_values=valid_perm,
        n_perm=valid_perm.shape[0] + 1,
    )

    result_df = _create_result_dataframe(obs_result, p_values)

    if verbose:
        elapsed_time = time.perf_counter() - start_time
        print(f"Permutation test completed in {elapsed_time:.2f} seconds")

    return result_df


def _infer_n_samples(dv, iv) -> int:
    """
    Infer sample size from iv first, then dv.

    Permutation is applied to explanatory variables, so iv is the reliable source
    of sample size. This also prevents a dbRDA condensed distance vector of length
    n * (n - 1) / 2 from being mistaken for n samples.
    """
    if isinstance(iv, pd.DataFrame):
        return iv.shape[0]

    if isinstance(iv, dict):
        first = next(iter(iv.values()))
        return first.shape[0] if hasattr(first, "shape") else len(first)

    if isinstance(iv, list):
        first = iv[0]
        return first.shape[0] if hasattr(first, "shape") else len(first)

    if iv is not None:
        arr = np.asarray(iv)
        if arr.ndim >= 1:
            return arr.shape[0]

    if isinstance(dv, pd.DataFrame):
        return dv.shape[0]

    if isinstance(dv, np.ndarray):
        arr = np.asarray(dv)
        if arr.ndim == 2:
            return arr.shape[0]

    raise ValueError("Cannot infer sample size from inputs.")


def _permute_dataframe_rows(df: pd.DataFrame, perms: np.ndarray) -> pd.DataFrame:
    return df.iloc[perms].reset_index(drop=True)


def _permute_array_rows(value, perms: np.ndarray):
    arr = np.asarray(value)
    if arr.ndim == 1:
        return arr[perms]
    return arr[perms, :]


def _permute_grouped_dict_independently(iv: Dict, n_samples: int, rng) -> Dict:
    """
    Permute each logical predictor group independently.

    Used for original DataFrame input after one-time factor encoding. A categorical
    variable expanded to multiple dummy columns stays together within its group,
    but different original variables receive different permutations, matching the
    data.frame branch of rdacca.hp::permu.hp.
    """
    out = {}
    for k, v in iv.items():
        perms = rng.permutation(n_samples)
        if isinstance(v, pd.DataFrame):
            out[k] = _permute_dataframe_rows(v, perms)
        else:
            out[k] = _permute_array_rows(v, perms)
    return out


def _permute_dataframe_columns(df: pd.DataFrame, n_samples: int, rng) -> pd.DataFrame:
    """
    Permute each column of a DataFrame independently while preserving
    DataFrame structure, column names, and dtypes as much as possible.
    """
    out = df.copy().reset_index(drop=True)

    for col in out.columns:
        perms = rng.permutation(n_samples)
        out[col] = out[col].iloc[perms].to_numpy()

    return out


def _permute_ndarray_columns(arr: np.ndarray, n_samples: int, rng) -> np.ndarray:
    """
    Permute each column of an ndarray independently.
    """
    arr = np.asarray(arr).copy()

    if arr.ndim == 1:
        perms = rng.permutation(n_samples)
        return arr[perms]

    for j in range(arr.shape[1]):
        perms = rng.permutation(n_samples)
        arr[:, j] = arr[perms, j]

    return arr


def _permute_variables(
    iv: Union[np.ndarray, pd.DataFrame, List, Dict],
    n_samples: int,
    rng=None,
):
    """
    Permute predictor variables while preserving structure.

    Backward-compatible:
    - existing calls like _permute_variables(iv, n_samples) still work
    - optimized internal calls may pass rng explicitly
    """
    if rng is None:
        rng = np.random.default_rng()

    # ---- grouped dict input: synchronized row permutation across groups ----
    if isinstance(iv, dict):
        perms = rng.permutation(n_samples)
        out = {}

        for k, v in iv.items():
            if isinstance(v, pd.DataFrame):
                out[k] = _permute_dataframe_rows(v, perms)
            else:
                out[k] = _permute_array_rows(v, perms)
        return out

    # ---- grouped list input: synchronized row permutation across groups ----
    if isinstance(iv, list):
        perms = rng.permutation(n_samples)
        out = []

        for v in iv:
            if isinstance(v, pd.DataFrame):
                out.append(_permute_dataframe_rows(v, perms))
            else:
                out.append(_permute_array_rows(v, perms))
        return out

    # ---- single DataFrame input ----
    if isinstance(iv, pd.DataFrame):
        return _permute_dataframe_columns(iv, n_samples, rng)

    # ---- ndarray input ----
    return _permute_ndarray_columns(np.asarray(iv), n_samples, rng)


def _calculate_p_values(obs_values: np.ndarray,
                        perm_values: np.ndarray,
                        n_perm: int | None = None) -> np.ndarray:
    """
    Calculate p-values using the rdacca.hp::permu.hp ECDF formula.

    ``n_perm`` is the total permutations argument, including the observed value.
    If omitted, it is inferred as one observed value plus the number of valid
    randomized runs.
    """
    obs_values = np.asarray(obs_values, dtype=float)
    perm_values = np.asarray(perm_values, dtype=float)

    total_permutations = int(n_perm) if n_perm is not None else perm_values.shape[0] + 1
    decimals = len(str(total_permutations))

    p_values = np.zeros(len(obs_values), dtype=float)

    for i in range(len(obs_values)):
        x = np.concatenate(([obs_values[i]], perm_values[:, i]))
        ecdf_at_obs = np.mean(x <= obs_values[i])
        p = 1.0 - ecdf_at_obs + 1.0 / (total_permutations + 1.0)
        # R's round() uses IEC 60559 ties-to-even rounding. NumPy follows
        # that rule, while Python's built-in round can differ for decimal
        # halfway values because of their binary representation.
        p_values[i] = float(np.round(p, decimals))

    return p_values


def _create_result_dataframe(obs_result, p_values: np.ndarray) -> pd.DataFrame:
    """
    Format permutation results like R permu.hp or as the expanded Python table.
    """
    hier_part = obs_result.hier_part
    return pd.DataFrame(
        {
            "Individual": hier_part["Individual"].to_numpy(dtype=float),
            "Pr(>I)": [_format_r_p_value(value) for value in p_values],
        },
        index=hier_part.index,
    )


def _format_r_p_value(p_value: float) -> str:
    """Combine the rounded p-value and stars as R permu.hp prints them."""
    if p_value <= 0.001:
        suffix = " ***"
    elif p_value <= 0.01:
        suffix = "  **"
    elif p_value <= 0.05:
        suffix = "   *"
    else:
        suffix = "    "
    return f"{p_value:g}{suffix}"


def _format_significance(p_value: float) -> str:
    """
    Convert p-value to significance stars, following rdacca.hp::permu.hp.
    """
    if p_value <= 0.001:
        return "***"
    elif p_value <= 0.01:
        return "**"
    elif p_value <= 0.05:
        return "*"
    else:
        return ""
