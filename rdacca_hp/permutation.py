import numpy as np
import pandas as pd
from typing import Union, List, Dict
import time

from .core import rdacca_hp, _rdacca_hp_multi, _rdacca_hp_single
from .utils import sanitize_tabular_input, preprocess_predictor_dataframe


def _build_permutation_engine(iv, method, type, scale, n_perm, add, sqrt_dist, n_axes,
                              ordered_factors, categorical_factors, kwargs):
    """
    Build a lightweight evaluator for permutation runs without changing the public API
    or output structure.
    """
    core_kwargs = dict(kwargs)
    core_kwargs["n_perm"] = n_perm
    core_kwargs["add"] = add
    core_kwargs["sqrt_dist"] = sqrt_dist
    core_kwargs["n_axes"] = n_axes
    core_kwargs["ordered_factors"] = ordered_factors
    core_kwargs["categorical_factors"] = categorical_factors

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

        return iv, evaluate

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
             permutations: int = 999,
             scale: bool = False,
             n_perm: int = 1000,          # for CCA adjR2
             add: bool = False,           # for dbRDA
             sqrt_dist: bool = False,     # for dbRDA
             n_axes: int = None,          # for dbRDA
             ordered_factors: dict | None = None,
             categorical_factors: list | None = None,
             verbose: bool = True,
             random_state: int | None = None,
             **kwargs) -> pd.DataFrame:
    """
    Permutation test for hierarchical partitioning.

    Notes
    -----
    - 对 DataFrame 输入：逐列独立置换（与 R 单表分支意图一致）
    - 对 dict/list 分组输入：每个组内部按同一个行置换同步打乱
    - 会保留 DataFrame 结构，避免列名/类型丢失
    """

    method = method.upper()
    if method not in {"RDA", "CCA", "DBRDA"}:
        raise ValueError("method must be 'RDA', 'CCA', or 'dbRDA'")

    if type not in {"R2", "adjR2"}:
        raise ValueError("type must be 'R2' or 'adjR2'")

    ordered_factors = ordered_factors or {}
    categorical_factors = categorical_factors or []
    rng = np.random.default_rng(random_state)

    dv = sanitize_tabular_input(dv, warn=verbose)
    iv = sanitize_tabular_input(iv, warn=verbose)

    if verbose:
        print(f"Running permutation test with {permutations} permutations...")
        start_time = time.perf_counter()

    # ---- observed result ----
    obs_result = rdacca_hp(
        dv=dv,
        iv=iv,
        method=method,
        type=type,
        scale=scale,
        var_part=False,
        n_perm=n_perm,
        add=add,
        sqrt_dist=sqrt_dist,
        n_axes=n_axes,
        ordered_factors=ordered_factors,
        categorical_factors=categorical_factors,
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
        ordered_factors=ordered_factors,
        categorical_factors=categorical_factors,
        kwargs=kwargs,
    )

    # ---- initialize ----
    perm_individual = np.full((permutations, n_vars), np.nan, dtype=float)
    failed_perms = 0

    n_samples = _infer_n_samples(dv, perm_base_iv)

    # ---- permutation loop ----
    for i in range(permutations):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{permutations} permutations")

        permuted_iv = _permute_variables(perm_base_iv, n_samples, rng)

        try:
            perm_result = perm_engine(dv, permuted_iv)
            perm_individual[i, :] = perm_result.hier_part["Individual"].to_numpy(dtype=float)

        except Exception as e:
            failed_perms += 1
            if verbose:
                print(f"Warning: permutation {i + 1} failed: {e}")

    # ---- summarize failures ----
    if verbose:
        print(f"Failed permutations: {failed_perms}/{permutations}")

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
        n_perm=valid_perm.shape[0]
    )

    result_df = _create_result_dataframe(obs_result, p_values)

    if verbose:
        elapsed_time = time.perf_counter() - start_time
        print(f"Permutation test completed in {elapsed_time:.2f} seconds")

    return result_df


def _infer_n_samples(dv, iv) -> int:
    """
    Infer sample size from dv first, then iv.
    """
    if isinstance(dv, pd.DataFrame):
        return dv.shape[0]

    if isinstance(dv, np.ndarray):
        return dv.shape[0]

    if isinstance(iv, pd.DataFrame):
        return iv.shape[0]

    if isinstance(iv, dict):
        first = next(iter(iv.values()))
        return first.shape[0] if hasattr(first, "shape") else len(first)

    if isinstance(iv, list):
        first = iv[0]
        return first.shape[0] if hasattr(first, "shape") else len(first)

    arr = np.asarray(iv)
    return arr.shape[0]


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

    # ---- grouped dict input ----
    if isinstance(iv, dict):
        perms = rng.permutation(n_samples)
        out = {}

        for k, v in iv.items():
            if isinstance(v, pd.DataFrame):
                out[k] = v.iloc[perms].reset_index(drop=True)
            else:
                arr = np.asarray(v)
                if arr.ndim == 1:
                    out[k] = arr[perms]
                else:
                    out[k] = arr[perms, :]
        return out

    # ---- grouped list input ----
    if isinstance(iv, list):
        perms = rng.permutation(n_samples)
        out = []

        for v in iv:
            if isinstance(v, pd.DataFrame):
                out.append(v.iloc[perms].reset_index(drop=True))
            else:
                arr = np.asarray(v)
                if arr.ndim == 1:
                    out.append(arr[perms])
                else:
                    out.append(arr[perms, :])
        return out

    # ---- single DataFrame input ----
    if isinstance(iv, pd.DataFrame):
        return _permute_dataframe_columns(iv, n_samples, rng)

    # ---- ndarray input ----
    return _permute_ndarray_columns(np.asarray(iv), n_samples, rng)


def _calculate_p_values(obs_values: np.ndarray,
                        perm_values: np.ndarray,
                        n_perm: int) -> np.ndarray:
    """
    Calculate p-values from valid permutation results only.
    """
    n_vars = len(obs_values)
    p_values = np.zeros(n_vars, dtype=float)

    for i in range(n_vars):
        count = np.sum(perm_values[:, i] >= obs_values[i])
        p_values[i] = (count + 1) / (n_perm + 1)

    return p_values


def _create_result_dataframe(obs_result, p_values: np.ndarray) -> pd.DataFrame:
    """
    Merge observed hierarchical partitioning result with p-values.
    """
    hier_part = obs_result.hier_part.copy()
    hier_part["Pr(>I)"] = p_values
    hier_part["Significance"] = hier_part["Pr(>I)"].apply(_format_significance)
    return hier_part


def _format_significance(p_value: float) -> str:
    """
    Convert p-value to significance stars.
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.1:
        return "."
    else:
        return ""
