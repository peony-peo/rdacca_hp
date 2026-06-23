# core.py
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Union, List, Dict, Any, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.linalg import eigh
from sklearn.metrics import pairwise_distances
from .utils import (
    check_data_quality,
    create_binary_matrix,
    genList,
    calculate_adjusted_r2,
    generate_combination_names,
    get_combination_names,
    preprocess_predictor_dataframe,
    preprocess_grouped_predictors,
    coerce_distance_input,
    prepare_dbrda_response,
)


class RdaccaHpResult:
    """Container for rdacca_hp results"""

    def __init__(self, method_type, total_explained_variation, hier_part, var_part=None):
        self.method_type = method_type  # [method, type]
        self.total_explained_variation = total_explained_variation
        self.hier_part = hier_part  # DataFrame with individual contributions
        self.var_part = var_part  # DataFrame with variation partitioning
        self._var_part_flag = var_part is not None

    def __repr__(self):
        return f"RdaccaHpResult(method={self.method_type[0]}, type={self.method_type[1]}, total_R2={self.total_explained_variation:.4f})"

    def summary(self):
        """Print summary of results"""
        print(f"Method: {self.method_type[0]}, Type: {self.method_type[1]}")
        print(f"Total explained variation: {self.total_explained_variation:.4f}")
        print("\nHierarchical Partitioning:")
        print(self.hier_part.to_string(float_format="%.4f"))

        if self._var_part_flag:
            print("\nVariation Partitioning:")
            print(self.var_part.to_string(float_format="%.4f"))


def safe_divide(numerator, denominator, default=0.0):
    """安全的除法运算，避免除零错误"""
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def ensure_non_negative(array, default=0.0):
    """确保数组非负"""
    return np.maximum(array, default)




@lru_cache(maxsize=None)
def _get_hp_cached_structures(n_items: int):
    """Cache combination structures reused across hierarchical partitioning calls."""
    binary_matrix = create_binary_matrix(n_items).astype(np.int8, copy=False)
    total_combinations = binary_matrix.shape[1]
    bit_counts = np.sum(binary_matrix, axis=0).astype(float)

    order_indices = tuple(
        j
        for size in range(1, n_items + 1)
        for j in range(total_combinations)
        if bit_counts[j] == size
    )

    combo_indices = tuple(
        tuple(np.flatnonzero(binary_matrix[:, i]))
        for i in range(total_combinations)
    )

    commonlist = []
    seqID = [2 ** i for i in range(n_items)]
    for i in range(total_combinations):
        bit = binary_matrix[0, i]
        if bit == 1:
            ivname = [0, -seqID[0]]
        else:
            ivname = [seqID[0]]

        for j in range(1, n_items):
            bit = binary_matrix[j, i]
            if bit == 1:
                alist = ivname.copy()
                blist = genList(ivname, -seqID[j])
                ivname = alist + blist
            else:
                ivname = genList(ivname, seqID[j])

        commonlist.append(tuple(-x for x in ivname))

    return binary_matrix, total_combinations, bit_counts, order_indices, combo_indices, tuple(commonlist)


def _fill_commonality_values(commonM: np.ndarray, commonlist) -> None:
    """Fill commonM[:, 2] from a precomputed inclusion-exclusion structure."""
    for i, r2list in enumerate(commonlist):
        ccsum = 0.0
        for indexs in r2list:
            indexu = abs(indexs)
            if indexu != 0:
                ccvalue = commonM[indexu - 1, 1]
                if indexs < 0:
                    ccvalue = -ccvalue
                ccsum += ccvalue
        commonM[i, 2] = ccsum

def _as_2d_float_array(x: np.ndarray) -> np.ndarray:
    """Convert input to a 2D float ndarray without changing external behavior."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _r2_variance_weighted_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute variance-weighted multioutput R² using pure NumPy.

    This mirrors sklearn.metrics.r2_score(..., multioutput='variance_weighted')
    for the non-degenerate cases relevant to the package, while avoiding the
    heavy validation overhead that dominates permutation-time benchmarks.
    """
    y_true = _as_2d_float_array(y_true)
    y_pred = _as_2d_float_array(y_pred)

    diff = y_true - y_pred
    y_mean = np.mean(y_true, axis=0, keepdims=True)

    sse = np.sum(diff * diff, axis=0)
    sst = np.sum((y_true - y_mean) ** 2, axis=0)

    total_sse = float(np.sum(sse))
    total_sst = float(np.sum(sst))

    if total_sst <= 0:
        return 1.0 if total_sse <= 0 else 0.0

    r2 = 1.0 - total_sse / total_sst

    if r2 > 1.0 and r2 < 1.0 + 1e-12:
        return 1.0
    if r2 < 0.0 and r2 > -1e-12:
        return 0.0
    return float(r2)


def _fit_predict_lstsq_with_intercept(iv: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """
    Fit the same linear model shape as sklearn LinearRegression(fit_intercept=True)
    using NumPy least squares, and return fitted values.
    """
    X = _as_2d_float_array(iv)
    Y = _as_2d_float_array(dv)

    n_samples = X.shape[0]
    intercept = np.ones((n_samples, 1), dtype=float)
    X_design = np.concatenate((intercept, X), axis=1)

    coef, _, _, _ = np.linalg.lstsq(X_design, Y, rcond=None)
    return X_design @ coef


def calculate_rda(dv: np.ndarray, iv: np.ndarray, type: str = "adjR2") -> float:
    """
    Calculate R-squared for RDA analysis.

    Public signature and return semantics are unchanged; the implementation is
    optimized to avoid repeated sklearn/pandas validation overhead inside the
    hierarchical partitioning and permutation loops.
    """
    X = _as_2d_float_array(iv)
    Y = _as_2d_float_array(dv)

    n_samples, n_predictors = X.shape
    y_pred = _fit_predict_lstsq_with_intercept(X, Y)
    r_squared = _r2_variance_weighted_numpy(Y, y_pred)

    if type == "R2":
        return r_squared
    return calculate_adjusted_r2(r_squared, n_samples, n_predictors)

def chi_square_transform(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the same initial chi-square transformation as vegan::initCA."""
    Y = _as_2d_float_array(Y)
    if not np.all(np.isfinite(Y)):
        raise ValueError("NA/NaN/Inf is not allowed in CCA response data")
    if np.any(Y < 0):
        raise ValueError("Species abundance data should be non-negative for CCA")

    row_totals = np.sum(Y, axis=1)
    if np.any(row_totals <= 0):
        raise ValueError("All row sums must be positive in the CCA response matrix")

    # vegan::cca excludes species columns that have zero marginal totals.
    Y = Y[:, np.sum(Y, axis=0) > 0]
    if Y.shape[1] == 0:
        raise ValueError("CCA response data contain no positive species columns")

    proportions = Y / float(np.sum(Y))
    row_weights = np.sum(proportions, axis=1)
    col_weights = np.sum(proportions, axis=0)
    expected = np.outer(row_weights, col_weights)
    Y_chi = (proportions - expected) / np.sqrt(expected)
    return Y_chi, row_weights, col_weights


def _cca_projection(Y_chi: np.ndarray, X: np.ndarray,
                    row_weights: np.ndarray):
    """Fit the weighted CCA constraint space used by vegan."""
    X = _as_2d_float_array(X)
    if X.shape[0] != Y_chi.shape[0]:
        raise ValueError(
            "Dependent and independent variables must have the same number of rows."
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("Independent variables contain NaN or Inf values.")

    weighted_mean = np.sum(X * row_weights[:, None], axis=0)
    X_weighted = (X - weighted_mean) * np.sqrt(row_weights)[:, None]

    U, singular_values, _ = np.linalg.svd(X_weighted, full_matrices=False)
    if singular_values.size == 0:
        basis = np.empty((X.shape[0], 0), dtype=float)
    else:
        tolerance = 1e-7 * singular_values[0]
        rank = int(np.sum(singular_values > tolerance))
        basis = U[:, :rank]

    fitted = basis @ (basis.T @ Y_chi)
    total_inertia = float(np.sum(Y_chi ** 2))
    if total_inertia <= 0:
        raise ValueError("CCA response data have zero total inertia")
    constrained_inertia = float(np.sum(fitted ** 2))
    return constrained_inertia / total_inertia, basis, total_inertia


def _permutation_cca_adjusted(Y_chi: np.ndarray, X: np.ndarray,
                              row_weights: np.ndarray, basis: np.ndarray,
                              observed_r2: float, n_perm: int,
                              random_state=None) -> float:
    """Apply vegan::RsquareAdj.cca's permutation adjustment."""
    n_perm = int(n_perm)
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1 for CCA adjusted R2")
    if basis.shape[1] == 0:
        raise ValueError("CCA adjusted R2 is undefined for a rank-zero model")
    if basis.shape[1] >= Y_chi.shape[0] - 1:
        raise ValueError("CCA adjusted R2 is undefined without residual degrees of freedom")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    permuted_r2 = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        permutation = rng.permutation(Y_chi.shape[0])
        permuted = Y_chi[permutation, :]
        permuted_weights = row_weights[permutation]

        # vegan::permutest.cca permutes the CCA row weights with the
        # response and rebuilds the weighted constraint QR for every
        # permutation. A fixed observed-model basis overestimates adjR2.
        permuted_r2[i], _, _ = _cca_projection(
            permuted,
            X,
            permuted_weights,
        )

    mean_permuted_r2 = float(np.mean(permuted_r2))
    denominator = 1.0 - mean_permuted_r2
    if abs(denominator) < 1e-12:
        raise ValueError("CCA adjusted R2 is undefined for this permutation distribution")
    return 1.0 - (1.0 - observed_r2) / denominator


def calculate_cca_r2_adj(Y: np.ndarray, X: np.ndarray, n_perm: int = 1000,
                         random_state=None) -> Tuple[float, float]:
    Y_chi, row_weights, _ = chi_square_transform(Y)
    r_squared, basis, total_inertia = _cca_projection(Y_chi, X, row_weights)
    adj_r_squared = _permutation_cca_adjusted(
        Y_chi,
        X,
        row_weights,
        basis,
        r_squared,
        n_perm,
        random_state=random_state,
    )
    return r_squared, adj_r_squared


def calculate_cca(dv: np.ndarray, iv: np.ndarray, type: str = "adjR2",
                  n_perm: int = 1000, random_state=None) -> float:
    """
    Calculate R-squared for CCA (Canonical Correspondence Analysis)

    Parameters
    ----------
    dv : ndarray
        Species abundance data (samples x species)
    iv : ndarray
        Environmental variables (samples x predictors)
    type : str
        Type of R-squared: "R2" or "adjR2"
    n_perm : int
        Number of permutations for adjusted R-squared calculation

    Returns
    -------
    float
        R-squared value
    """
    if type not in {"R2", "adjR2"}:
        raise ValueError("type must be 'R2' or 'adjR2'")

    Y_chi, row_weights, _ = chi_square_transform(dv)
    r_squared, basis, total_inertia = _cca_projection(Y_chi, iv, row_weights)
    if type == "R2":
        return r_squared

    return _permutation_cca_adjusted(
        Y_chi,
        iv,
        row_weights,
        basis,
        r_squared,
        n_perm,
        random_state=random_state,
    )


def calculate_pcoa(distance_matrix: np.ndarray, n_axes: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Principal Coordinates Analysis (PCoA) on a distance matrix

    Parameters
    ----------
    distance_matrix : ndarray
        Square distance matrix (n_samples x n_samples)
    n_axes : int, optional
        Number of principal coordinates to return

    Returns
    -------
    tuple
        (pcoa_scores, eigenvalues) - PCoA scores and eigenvalues
    """
    n_samples = distance_matrix.shape[0]

    # Center the distance matrix using double centering
    # Gower's double centering: B = -0.5 * (I - 1/n) * D^2 * (I - 1/n)

    # Step 1: Square the distance matrix
    D_sq = distance_matrix ** 2

    # Step 2: Create centering matrix
    I = np.eye(n_samples)
    ones = np.ones((n_samples, n_samples)) / n_samples
    centering_matrix = I - ones

    # Step 3: Apply double centering
    B = -0.5 * centering_matrix @ D_sq @ centering_matrix

    # Step 4: Eigen decomposition
    eigenvalues, eigenvectors = eigh(B)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Only keep positive eigenvalues and corresponding eigenvectors
    positive_mask = eigenvalues > 1e-10
    eigenvalues = eigenvalues[positive_mask]
    eigenvectors = eigenvectors[:, positive_mask]

    # Calculate PCoA scores
    pcoa_scores = eigenvectors * np.sqrt(eigenvalues)

    # Select number of axes if specified
    if n_axes is not None and n_axes < pcoa_scores.shape[1]:
        pcoa_scores = pcoa_scores[:, :n_axes]
        eigenvalues = eigenvalues[:n_axes]

    return pcoa_scores, eigenvalues


def euclidify_distance_matrix(distance_matrix: np.ndarray, method: str = "lingoes") -> np.ndarray:
    """
    Make a distance matrix Euclidean by adding a constant

    Parameters
    ----------
    distance_matrix : ndarray
        Input distance matrix
    method : str
        Method to use: "lingoes" or "cailliez"

    Returns
    -------
    ndarray
        Euclidean distance matrix
    """
    if method == "lingoes":
        # Lingoes method: add constant to non-diagonal elements
        n = distance_matrix.shape[0]

        # Create a matrix with squared distances
        D_sq = distance_matrix ** 2

        # Calculate the constant
        A = -0.5 * D_sq
        row_sums = A.sum(axis=1, keepdims=True)
        col_sums = A.sum(axis=0, keepdims=True)
        total_sum = A.sum()

        # Double centering
        B = A - row_sums / n - col_sums / n + total_sum / (n ** 2)

        # Find the smallest eigenvalue
        eigenvalues = np.linalg.eigvalsh(B)
        min_eigenvalue = np.min(eigenvalues)

        if min_eigenvalue >= 0:
            # Already Euclidean
            return distance_matrix
        else:
            # Add constant to make Euclidean
            constant = -2 * min_eigenvalue
            new_D_sq = D_sq + constant * (1 - np.eye(n))
            return np.sqrt(new_D_sq)

    elif method == "cailliez":
        # Match vegan::addCailliez: use the largest real eigenvalue of
        # the 2n x 2n block matrix built from Gower-centred distances.
        n = distance_matrix.shape[0]

        def gower_double_center(matrix):
            return (
                matrix
                - np.mean(matrix, axis=0, keepdims=True)
                - np.mean(matrix, axis=1, keepdims=True)
                + np.mean(matrix)
            )

        z = np.zeros((2 * n, 2 * n), dtype=float)
        z[n:, :n] = -np.eye(n)
        z[:n, n:] = -gower_double_center(distance_matrix ** 2)
        z[n:, n:] = gower_double_center(2.0 * distance_matrix)

        eigenvalues = np.linalg.eigvals(z)
        constant = max(float(np.max(eigenvalues.real)), 0.0)
        return distance_matrix + constant * (1 - np.eye(n))

    else:
        raise ValueError("Method must be 'lingoes' or 'cailliez'")


def check_distance_matrix(distance_matrix: np.ndarray) -> bool:
    """
    Check if a matrix is a valid distance matrix

    Parameters
    ----------
    distance_matrix : ndarray
        Matrix to check

    Returns
    -------
    bool
        True if valid distance matrix
    """
    # Check square
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        return False

    # Check symmetric
    if not np.allclose(distance_matrix, distance_matrix.T):
        return False

    # Check non-negative
    if np.any(distance_matrix < 0):
        return False

    # Check zero diagonal
    if not np.allclose(np.diag(distance_matrix), 0):
        return False

    return True


def calculate_rda_r2_adj(dv: np.ndarray, iv: np.ndarray, type: str = "adjR2") -> Tuple[float, float]:
    """
    Calculate R-squared and adjusted R-squared for RDA (separate function for db-RDA).

    Function name, parameters, and returned tuple are unchanged.
    """
    X = _as_2d_float_array(iv)
    Y = _as_2d_float_array(dv)

    n_samples, n_predictors = X.shape
    y_pred = _fit_predict_lstsq_with_intercept(X, Y)
    r_squared = _r2_variance_weighted_numpy(Y, y_pred)
    adj_r_squared = calculate_adjusted_r2(r_squared, n_samples, n_predictors)
    return r_squared, adj_r_squared

def _apply_distance_correction(distance_matrix: np.ndarray, add=False) -> np.ndarray:
    """Apply vegan-compatible ``add`` choices to a dissimilarity matrix."""
    if add is False or add is None:
        return distance_matrix
    if add is True:
        method = "lingoes"
    elif isinstance(add, str) and add.lower() in {"lingoes", "cailliez"}:
        method = add.lower()
    else:
        raise ValueError("add must be False, True, 'lingoes', or 'cailliez'")
    return euclidify_distance_matrix(distance_matrix, method=method)


def _calculate_dbrda_vegan_dbrda(dv_dist: np.ndarray, iv: np.ndarray, type: str = "adjR2",
                                 add=False, sqrt_dist: bool = False) -> float:
    """
    Vegan::dbrda-style db-RDA R2/adjusted R2.

    This uses the centered Gower matrix directly:
        B = -0.5 * J D^2 J

    R2 is computed as:
        trace(H B) / trace(B)

    where H is the projection matrix of centered predictors.
    """
    distance_matrix = coerce_distance_input(dv_dist)

    if sqrt_dist:
        distance_matrix = np.sqrt(distance_matrix)

    distance_matrix = _apply_distance_correction(distance_matrix, add=add)

    n_samples = distance_matrix.shape[0]

    X = _as_2d_float_array(iv)
    if X.shape[0] != n_samples:
        raise ValueError("Dependent and independent variables must have the same number of rows.")

    # Gower-centered matrix
    J = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    B = -0.5 * J @ (distance_matrix ** 2) @ J

    total_inertia = float(np.trace(B))
    if abs(total_inertia) < 1e-12:
        r_squared = 0.0
        n_predictors = X.shape[1]
    else:
        # Center predictors; intercept is handled by centering.
        Xc = X - np.mean(X, axis=0, keepdims=True)

        # Use rank for adjusted R2, closer to vegan behavior under collinearity.
        n_predictors = int(np.linalg.matrix_rank(Xc))

        if n_predictors == 0:
            r_squared = 0.0
        else:
            # Projection matrix H = X (X'X)^- X'
            H = Xc @ np.linalg.pinv(Xc.T @ Xc) @ Xc.T

            constrained_inertia = float(np.trace(H @ B))
            r_squared = constrained_inertia / total_inertia

    if type == "R2":
        return r_squared

    return calculate_adjusted_r2(r_squared, n_samples, n_predictors)


def calculate_dbrda(dv_dist: np.ndarray, iv: np.ndarray, type: str = "adjR2",
                    add=False, sqrt_dist: bool = False, n_axes: int = None,
                    dbrdatype: str = "dbrda") -> float:
    """
    Calculate R-squared for db-RDA.

    Parameters
    ----------
    dv_dist : ndarray
        Distance matrix or condensed distance vector.
    iv : ndarray
        Environmental variables.
    type : str
        "R2" or "adjR2".
    add : bool
        Whether to add a constant to euclidify dissimilarities.
    sqrt_dist : bool
        Whether to take square root of distances.
    n_axes : int, optional
        Number of PCoA axes to use. Only used when dbrdatype="capscale".
    dbrdatype : str
        "dbrda" or "capscale".

        "dbrda" matches the default behavior of R rdacca.hp >= 1.1.3.
        "capscale" keeps the previous Python behavior.
    """
    dbrdatype = str(dbrdatype).lower()

    if dbrdatype not in ["dbrda", "capscale"]:
        raise ValueError("dbrdatype must be 'dbrda' or 'capscale'")

    # R rdacca.hp 1.1.3 default: dbrdatype = "dbrda"
    if dbrdatype == "dbrda":
        return _calculate_dbrda_vegan_dbrda(
            dv_dist=dv_dist,
            iv=iv,
            type=type,
            add=add,
            sqrt_dist=sqrt_dist,
        )

    # Previous Python behavior: capscale-like PCoA positive-axis route
    distance_matrix = coerce_distance_input(dv_dist)

    if sqrt_dist:
        distance_matrix = np.sqrt(distance_matrix)

    distance_matrix = _apply_distance_correction(distance_matrix, add=add)

    pcoa_scores, eigenvalues = calculate_pcoa(distance_matrix, n_axes)

    if pcoa_scores.shape[1] == 0:
        raise ValueError("PCoA produced no positive eigenvalues")

    r_squared, adj_r_squared = calculate_rda_r2_adj(pcoa_scores, iv, type)

    if type == "R2":
        return r_squared
    return adj_r_squared

def _prepare_multi_group_iv(iv: Union[List, Dict]) -> Tuple[List[np.ndarray], List[str]]:
    """
    Prepare multiple predictor groups for analysis

    Parameters
    ----------
    iv : list or dict
        List or dictionary of predictor groups

    Returns
    -------
    tuple
        (iv_arrays, group_names) - List of predictor arrays and group names
    """
    if isinstance(iv, dict):
        group_names = list(iv.keys())
        iv_arrays = []
        for name in group_names:
            group_data = iv[name]
            if hasattr(group_data, 'values'):
                group_data = group_data.values
            iv_arrays.append(group_data)
    elif isinstance(iv, list):
        group_names = [f"X{i + 1}" for i in range(len(iv))]
        iv_arrays = []
        for i, group_data in enumerate(iv):
            if hasattr(group_data, 'values'):
                group_data = group_data.values
            iv_arrays.append(group_data)
    else:
        raise ValueError("For multiple groups, iv should be a list or dict")

    return iv_arrays, group_names


def _combine_groups(iv_arrays: List[np.ndarray], selected_indices: List[int]) -> np.ndarray:
    """
    Combine selected groups into a single predictor matrix

    Parameters
    ----------
    iv_arrays : list
        List of predictor arrays
    selected_indices : list
        Indices of groups to combine

    Returns
    -------
    ndarray
        Combined predictor matrix
    """
    if not selected_indices:
        raise ValueError("At least one group must be selected")

    combined = []
    for idx in selected_indices:
        if idx < 0 or idx >= len(iv_arrays):
            raise ValueError(f"Invalid group index: {idx}")
        combined.append(iv_arrays[idx])

    return np.hstack(combined)


def _rdacca_hp_multi(dv, iv, method, type, scale, var_part, **kwargs):
    """
    Hierarchical partitioning for multiple predictor groups
    """
    # Prepare multiple groups
    iv_arrays, group_names = _prepare_multi_group_iv(iv)
    n_groups = len(iv_arrays)

    if n_groups < 2:
        raise ValueError("Analysis not conducted. Insufficient number of predictor groups.")

    # Check data quality for each group
    for i, group_data in enumerate(iv_arrays):
        _, group_clean = check_data_quality(dv, group_data)
        iv_arrays[i] = group_clean

    # Standardize if requested (for RDA)
    if method.upper() == "RDA" and scale:
        dv = (dv - np.mean(dv, axis=0)) / np.std(dv, axis=0)

    # Create binary matrix and other reusable combination structures
    binary_matrix, total_combinations, bit_counts, order_indices, combo_indices, commonlist = _get_hp_cached_structures(n_groups)

    # Calculate R-squared for all group combinations
    commonM = np.zeros((total_combinations, 3))

    for i, selected_indices in enumerate(combo_indices):
        if not selected_indices:
            continue

        # Combine selected groups
        combined_iv = _combine_groups(iv_arrays, list(selected_indices))

        # Calculate R-squared based on method
        if method.upper() in ["RDA"]:
            r2_value = calculate_rda(dv, combined_iv, type)
            commonM[i, 1] = r2_value
        elif method.upper() in ["CCA"]:
            r2_value = calculate_cca(
                dv,
                combined_iv,
                type,
                n_perm=kwargs.get('n_perm', 1000),
                random_state=kwargs.get('_cca_rng'),
            )
            commonM[i, 1] = r2_value
        elif method.upper() in ["DBRDA"]:
            r2_value = calculate_dbrda(
                dv, combined_iv, type,
                add=kwargs.get('add', False),
                sqrt_dist=kwargs.get('sqrt_dist', False),
                n_axes=kwargs.get('n_axes', None),
                dbrdatype=kwargs.get('dbrdatype', 'dbrda')
            )
            commonM[i, 1] = r2_value
        else:
            raise ValueError(f"Unknown method: {method}")

    # Calculate commonality metrics from cached inclusion-exclusion structure
    _fill_commonality_values(commonM, commonlist)

    # Fill the first column with ordered indices
    for i, idx in enumerate(order_indices):
        commonM[i, 0] = idx

    # Create output matrix for variation partitioning
    outputcommonM = np.zeros((total_combinations + 1, 2))
    totalRSquare = np.sum(commonM[:, 2])

    for i in range(total_combinations):
        idx = int(commonM[i, 0])
        outputcommonM[i, 0] = round(commonM[idx, 2], 4)

        # Safe division for percentages
        percentage = safe_divide(commonM[idx, 2], totalRSquare) * 100
        outputcommonM[i, 1] = round(percentage, 2)

    # Total row
    outputcommonM[total_combinations, 0] = round(totalRSquare, 4)
    outputcommonM[total_combinations, 1] = 100.0

    # Create descriptive row names for variation partitioning
    ordered_binary_matrix = binary_matrix[:, order_indices]
    rowNames = get_combination_names(ordered_binary_matrix, group_names)

    # Ensure row names match the data shape
    if len(rowNames) != outputcommonM.shape[0]:
        rowNames = [f"Combination_{i + 1}" for i in range(outputcommonM.shape[0] - 1)] + ["Total"]

    # Calculate variable importance (hierarchical partitioning)
    VariableImportance = np.zeros((n_groups, 4))

    for i in range(n_groups):
        # Calculate individual contribution (I)
        weights = binary_matrix[i, :] * (commonM[:, 2] / (bit_counts + 1e-10))
        individual_value = np.sum(weights)

        # Ensure individual contribution is non-negative
        VariableImportance[i, 2] = round(individual_value, 4)

    # Unique contributions are the single-group combinations
    VariableImportance[:, 0] = outputcommonM[:n_groups, 0]  # Unique

    # Average shared contribution = Individual - Unique
    VariableImportance[:, 1] = VariableImportance[:, 2] - VariableImportance[:, 0]

    # Calculate percentages
    total_individual = round(np.sum(VariableImportance[:, 2]), 3)

    if total_individual == 0:
        VariableImportance[:, 3] = 0.0
    else:
        percentages = 100 * VariableImportance[:, 2] / total_individual
        VariableImportance[:, 3] = np.round(percentages, 2)

    # Ensure no NaN or infinite values
    VariableImportance = np.where(np.isfinite(VariableImportance), VariableImportance, np.nan)

    # Create pandas DataFrames for results
    hier_part_df = pd.DataFrame(
        VariableImportance,
        columns=["Unique", "Average.share", "Individual", "I.perc(%)"],
        index=group_names
    )

    # Create results object
    if var_part:
        var_part_df = pd.DataFrame(
            outputcommonM,
            columns=["Fractions", "% Total"],
            index=rowNames
        )
        return RdaccaHpResult(
            method_type=[method, type],
            total_explained_variation=total_individual,
            hier_part=hier_part_df,
            var_part=var_part_df
        )
    else:
        return RdaccaHpResult(
            method_type=[method, type],
            total_explained_variation=total_individual,
            hier_part=hier_part_df
        )


def _rdacca_hp_single(dv, iv, method, type, scale, var_part, **kwargs):
    """
    Core hierarchical partitioning for single group of predictors
    """
    # Data preprocessing
    dv, iv = check_data_quality(dv, iv)

    ordered_factors = kwargs.get("ordered_factors", None) or {}
    categorical_factors = kwargs.get("categorical_factors", None) or []

    # ---------------------------------------------------------
    # IMPORTANT:
    # Keep purely numeric DataFrame inputs on the single-table path,
    # to match the R rdacca.hp data.frame branch.
    #
    # Only route to the multi-group branch when factor encoding is
    # actually needed:
    #   1) user explicitly declares ordered/categorical factors, or
    #   2) the DataFrame contains non-numeric columns.
    # ---------------------------------------------------------
    if isinstance(iv, pd.DataFrame):
        has_declared_factors = bool(ordered_factors) or bool(categorical_factors)
        has_non_numeric = not all(pd.api.types.is_numeric_dtype(iv[col]) for col in iv.columns)

        if has_declared_factors or has_non_numeric:
            _, encoded_groups, _ = preprocess_predictor_dataframe(
                iv,
                ordered_factors=ordered_factors,
                categorical_factors=categorical_factors,
                warn=False,
            )
            return _rdacca_hp_multi(
                dv=dv,
                iv=encoded_groups,
                method=method,
                type=type,
                scale=scale,
                var_part=var_part,
                **kwargs,
            )

        # Pure numeric DataFrame: stay on single-table branch
        var_names = iv.columns.tolist()
        iv = iv.to_numpy(dtype=float)

    else:
        # Non-DataFrame input
        if hasattr(iv, "columns"):
            var_names = iv.columns.tolist()
        else:
            iv = np.asarray(iv, dtype=float)
            if iv.ndim == 1:
                iv = iv.reshape(-1, 1)
            var_names = [f"X{i + 1}" for i in range(iv.shape[1])]

    n_samples, n_vars = iv.shape

    if n_samples <= n_vars:
        raise ValueError("sample size (row) is less than the number of predictors")

    if n_vars < 2:
        raise ValueError("Analysis not conducted. Insufficient number of predictors.")

    # Standardize if requested (for RDA)
    if method.upper() == "RDA" and scale:
        dv = (dv - np.mean(dv, axis=0)) / np.std(dv, axis=0)

    # Create binary matrix and other reusable combination structures
    binary_matrix, total_combinations, bit_counts, order_indices, combo_indices, commonlist = _get_hp_cached_structures(n_vars)

    # Calculate R-squared for all combinations
    commonM = np.zeros((total_combinations, 3))

    for i, selected_indices in enumerate(combo_indices):
        if not selected_indices:
            continue

        subset_iv = iv[:, selected_indices]

        # Calculate R-squared based on method
        if method.upper() in ["RDA"]:
            r2_value = calculate_rda(dv, subset_iv, type)
            commonM[i, 1] = r2_value
        elif method.upper() in ["CCA"]:
            r2_value = calculate_cca(
                dv,
                subset_iv,
                type,
                n_perm=kwargs.get('n_perm', 1000),
                random_state=kwargs.get('_cca_rng'),
            )
            commonM[i, 1] = r2_value
        elif method.upper() in ["DBRDA"]:
            r2_value = calculate_dbrda(
                dv, subset_iv, type,
                add=kwargs.get('add', False),
                sqrt_dist=kwargs.get('sqrt_dist', False),
                n_axes=kwargs.get('n_axes', None),
                dbrdatype=kwargs.get('dbrdatype', 'dbrda')
            )
            commonM[i, 1] = r2_value
        else:
            raise ValueError(f"Unknown method: {method}")

    # Calculate commonality metrics from cached inclusion-exclusion structure
    _fill_commonality_values(commonM, commonlist)

    # Fill the first column with ordered indices
    for i, idx in enumerate(order_indices):
        commonM[i, 0] = idx

    # Create output matrix for variation partitioning
    outputcommonM = np.zeros((total_combinations + 1, 2))
    totalRSquare = np.sum(commonM[:, 2])

    for i in range(total_combinations):
        idx = int(commonM[i, 0])
        outputcommonM[i, 0] = round(commonM[idx, 2], 4)

        percentage = safe_divide(commonM[idx, 2], totalRSquare) * 100
        outputcommonM[i, 1] = round(percentage, 2)

    # Total row
    outputcommonM[total_combinations, 0] = round(totalRSquare, 4)
    outputcommonM[total_combinations, 1] = 100.0

    # Create descriptive row names for variation partitioning
    ordered_binary_matrix = binary_matrix[:, order_indices]
    rowNames = get_combination_names(ordered_binary_matrix, var_names)

    if len(rowNames) != outputcommonM.shape[0]:
        rowNames = [f"Combination_{i + 1}" for i in range(outputcommonM.shape[0] - 1)] + ["Total"]

    # Calculate variable importance (hierarchical partitioning)
    VariableImportance = np.zeros((n_vars, 4))

    for i in range(n_vars):
        weights = binary_matrix[i, :] * (commonM[:, 2] / (bit_counts + 1e-10))
        individual_value = np.sum(weights)
        VariableImportance[i, 2] = round(individual_value, 4)

    # Unique contributions are the single-variable combinations
    VariableImportance[:, 0] = outputcommonM[:n_vars, 0]

    # Average shared contribution = Individual - Unique
    VariableImportance[:, 1] = VariableImportance[:, 2] - VariableImportance[:, 0]

    # Percentages
    total_individual = round(np.sum(VariableImportance[:, 2]), 3)

    if total_individual == 0:
        VariableImportance[:, 3] = 0.0
    else:
        percentages = 100 * VariableImportance[:, 2] / total_individual
        VariableImportance[:, 3] = np.round(percentages, 2)

    VariableImportance = np.where(np.isfinite(VariableImportance), VariableImportance, np.nan)

    hier_part_df = pd.DataFrame(
        VariableImportance,
        columns=["Unique", "Average.share", "Individual", "I.perc(%)"],
        index=var_names
    )

    if var_part:
        var_part_df = pd.DataFrame(
            outputcommonM,
            columns=["Fractions", "% Total"],
            index=rowNames
        )
        return RdaccaHpResult(
            method_type=[method, type],
            total_explained_variation=total_individual,
            hier_part=hier_part_df,
            var_part=var_part_df
        )
    else:
        return RdaccaHpResult(
            method_type=[method, type],
            total_explained_variation=total_individual,
            hier_part=hier_part_df
        )


def rdacca_hp(dv: Union[np.ndarray, pd.DataFrame],
              iv: Union[np.ndarray, pd.DataFrame, List, Dict],
              method: str = "RDA",
              type: str = "adjR2",
              scale: bool = False,
              var_part: bool = False,
              n_perm: int = 1000,  # For CCA
              add: bool | str = False,  # False, True/"lingoes", or "cailliez"
              sqrt_dist: bool = False,  # For db-RDA
              n_axes: int = None,  # For db-RDA
              dbrdatype: str = "dbrda",  # For db-RDA: "dbrda" or "capscale"
              ordered_factors: dict | None = None,
              categorical_factors: list | None = None,
              distance: str | None = None,  # For raw db-RDA response matrices
              random_state: int | None = None,  # For reproducible CCA adjusted R2
              **kwargs) -> RdaccaHpResult:
    """
    Hierarchical and Variation Partitioning for Canonical Analysis

    Parameters
    ----------
    dv : array-like
        Response variable, either a numeric vector, matrix, data frame, or distance matrix
    iv : array-like, list or dict
        Predictors as either a data frame or a list/dict of data frames
    method : str
        Type of canonical analysis: "RDA", "dbRDA" or "CCA"
    type : str
        Type of total explained variation: "R2" or "adjR2"
    scale : bool
        Whether to standardize response variables (for RDA)
    var_part : bool
        Whether to show variation partitioning results
    n_perm : int
        Number of permutations for computing adjusted R-square for CCA
    random_state : int, optional
        Seed for reproducible CCA adjusted R-square permutations.
    add : bool or str
        Euclidification correction for db-RDA: False, True/"lingoes", or
        "cailliez".
    sqrt_dist : bool
        Whether to take square root of distances (for db-RDA)
    n_axes : int, optional
        Number of PCoA axes to use (for db-RDA)
    distance : str, optional
        Distance method used when db-RDA receives a raw response matrix. If
        None, dv must already be a square distance matrix or condensed vector.

    Returns
    -------
    RdaccaHpResult : Object containing partitioning results
    """

    # Parameter validation
    ordered_factors = ordered_factors or {}
    categorical_factors = categorical_factors or []

    method = method.upper()
    if method not in ["RDA", "CCA", "DBRDA"]:
        raise ValueError("method must be 'RDA', 'CCA', or 'dbRDA'")

    if type not in ["R2", "adjR2"]:
        raise ValueError("type must be 'R2' or 'adjR2'")

    # Existing calls with precomputed distances remain unchanged when
    # distance=None. A named method means dv is raw response data.
    if method == "DBRDA":
        dv = prepare_dbrda_response(dv, distance=distance)

    # These parameters apply to both the data.frame-style and grouped paths.
    kwargs['n_perm'] = n_perm
    kwargs['add'] = add
    kwargs['sqrt_dist'] = sqrt_dist
    kwargs['n_axes'] = n_axes
    kwargs['dbrdatype'] = dbrdatype
    kwargs['ordered_factors'] = ordered_factors
    kwargs['categorical_factors'] = categorical_factors
    if method == "CCA" and type == "adjR2":
        kwargs['_cca_rng'] = np.random.default_rng(random_state)

    # Handle different types of iv input
    if isinstance(iv, (list, dict)):
        # Match the R list-of-data.frames branch: each element is one logical
        # predictor group, including any encoded factor columns it contains.
        iv = preprocess_grouped_predictors(
            iv,
            ordered_factors=ordered_factors,
            categorical_factors=categorical_factors,
            warn=False,
        )
        return _rdacca_hp_multi(dv, iv, method, type, scale, var_part, **kwargs)

    return _rdacca_hp_single(dv, iv, method, type, scale, var_part, **kwargs)
