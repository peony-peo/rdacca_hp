import numpy as np
import pandas as pd
from typing import Union, List, Dict
import time

from .core import rdacca_hp
from .utils import sanitize_tabular_input


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

    if random_state is not None:
        np.random.seed(random_state)

    dv = sanitize_tabular_input(dv, warn=verbose)
    iv = sanitize_tabular_input(iv, warn=verbose)

    if verbose:
        print(f"Running permutation test with {permutations} permutations...")
        start_time = time.time()

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

    # ---- initialize ----
    perm_individual = np.full((permutations, n_vars), np.nan, dtype=float)
    failed_perms = 0

    n_samples = _infer_n_samples(dv, iv)

    # ---- permutation loop ----
    for i in range(permutations):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{permutations} permutations")

        permuted_iv = _permute_variables(iv, n_samples)

        try:
            perm_result = rdacca_hp(
                dv=dv,
                iv=permuted_iv,
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
        elapsed_time = time.time() - start_time
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


def _permute_variables(iv: Union[np.ndarray, pd.DataFrame, List, Dict], n_samples: int):
    """
    Permute predictor variables while preserving structure.

    Rules
    -----
    1) DataFrame:
       每一列独立置换，保留 DataFrame 结构、列名和 dtype。
    2) dict/list:
       每个组内部按同一个行置换同步打乱，保留组结构。
    3) ndarray:
       与旧逻辑一致，逐列独立置换。
    """
    # ---- grouped dict input ----
    if isinstance(iv, dict):
        perms = np.random.permutation(n_samples)
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
        perms = np.random.permutation(n_samples)
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
        out = iv.copy().reset_index(drop=True)

        for col in out.columns:
            perms = np.random.permutation(n_samples)
            # 保留 Series 结构，避免 object/category 被转坏
            out[col] = out[col].iloc[perms].to_numpy()

        return out

    # ---- ndarray input ----
    arr = np.asarray(iv).copy()

    if arr.ndim == 1:
        perms = np.random.permutation(n_samples)
        return arr[perms]

    for j in range(arr.shape[1]):
        perms = np.random.permutation(n_samples)
        arr[:, j] = arr[perms, j]

    return arr


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