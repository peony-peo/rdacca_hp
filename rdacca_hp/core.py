# core.py
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Union, List, Dict, Any, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils import resample
from scipy.linalg import eigh
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from .utils import (
    check_data_quality,
    create_binary_matrix,
    genList,
    calculate_adjusted_r2,
    generate_combination_names,
    get_combination_names,
    preprocess_predictor_dataframe,
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
    """
    Perform chi-square transformation for CCA

    Parameters
    ----------
    Y : ndarray
        Species abundance matrix (samples x species)

    Returns
    -------
    tuple
        (Y_chi, row_weights, col_weights) - Chi-square transformed matrix and weights
    """
    # Ensure non-negative values for species data
    if np.any(Y < 0):
        raise ValueError("Species abundance data should be non-negative for CCA")

    # Calculate totals
    total = np.sum(Y)
    row_totals = np.sum(Y, axis=1, keepdims=True)
    col_totals = np.sum(Y, axis=0, keepdims=True)

    # Calculate expected values under independence
    expected = np.dot(row_totals, col_totals) / total

    # Avoid division by zero
    expected_safe = np.where(expected > 0, expected, 1)

    # Chi-square transformation
    Y_chi = (Y - expected) / np.sqrt(expected_safe)

    # Calculate weights
    row_weights = row_totals.flatten() / total
    col_weights = col_totals.flatten() / total

    return Y_chi, row_weights, col_weights


def _permutation_cca_adjusted(Y: np.ndarray, X: np.ndarray, observed_r2: float,
                              n_perm: int, row_weights: np.ndarray) -> float:
    """
    Calculate adjusted R-squared for CCA using permutation test

    Parameters
    ----------
    Y : ndarray
        Species data
    X : ndarray
        Environmental data
    observed_r2 : float
        Observed R-squared value
    n_perm : int
        Number of permutations
    row_weights : ndarray
        Row weights from chi-square transformation

    Returns
    -------
    float
        Adjusted R-squared
    """
    n_samples = Y.shape[0]
    permuted_r2 = []

    for i in range(n_perm):
        # Permute species data while preserving site structure
        perm_indices = resample(np.arange(n_samples), replace=False, n_samples=n_samples)
        Y_perm = Y[perm_indices, :]

        try:
            # Recalculate R-squared with permuted data
            Y_chi_perm, _, _ = chi_square_transform(Y_perm)
            Y_weighted_perm = Y_chi_perm * np.sqrt(row_weights[:, np.newaxis])
            X_weighted_perm = X * np.sqrt(row_weights[:, np.newaxis])

            Y_centered_perm = Y_weighted_perm - np.mean(Y_weighted_perm, axis=0)
            X_centered_perm = X_weighted_perm - np.mean(X_weighted_perm, axis=0)

            covariance_perm = Y_centered_perm.T @ X_centered_perm
            U_perm, s_perm, Vt_perm = np.linalg.svd(covariance_perm, full_matrices=False)

            total_variance_perm = np.sum(Y_centered_perm ** 2)
            explained_variance_perm = np.sum(s_perm ** 2)
            r2_perm = safe_divide(explained_variance_perm, total_variance_perm)

            permuted_r2.append(r2_perm)
        except (np.linalg.LinAlgError, ValueError):
            permuted_r2.append(0)

    # Calculate adjusted R-squared using permutation distribution
    if permuted_r2:
        expected_r2 = np.mean(permuted_r2)
        adj_r2 = observed_r2 - expected_r2
        return max(0, adj_r2)  # Ensure non-negative
    else:
        return observed_r2


def calculate_cca_r2_adj(Y: np.ndarray, X: np.ndarray, n_perm: int = 1000) -> Tuple[float, float]:
    """
    Calculate R-squared and adjusted R-squared for CCA using permutation

    Parameters
    ----------
    Y : ndarray
        Species abundance matrix
    X : ndarray
        Environmental variables matrix
    n_perm : int
        Number of permutations for adjusted R-squared calculation

    Returns
    -------
    tuple
        (r_squared, adj_r_squared)
    """
    n_samples, n_species = Y.shape
    n_predictors = X.shape[1]

    # Chi-square transformation
    Y_chi, row_weights, _ = chi_square_transform(Y)

    # Weighted correlation/covariance calculation
    Y_weighted = Y_chi * np.sqrt(row_weights[:, np.newaxis])
    X_weighted = X * np.sqrt(row_weights[:, np.newaxis])

    # Calculate R-squared using SVD
    try:
        # Center the weighted matrices
        Y_centered = Y_weighted - np.mean(Y_weighted, axis=0)
        X_centered = X_weighted - np.mean(X_weighted, axis=0)

        # SVD approach for CCA
        covariance = Y_centered.T @ X_centered
        U, s, Vt = np.linalg.svd(covariance, full_matrices=False)

        # R-squared as proportion of variance explained
        total_variance = np.sum(Y_centered ** 2)
        explained_variance = np.sum(s ** 2)
        r_squared = safe_divide(explained_variance, total_variance)

    except (np.linalg.LinAlgError, ValueError):
        # Fallback to simple correlation if SVD fails
        correlation_matrix = np.corrcoef(np.column_stack([Y_weighted, X_weighted]).T)
        if correlation_matrix.shape[0] > 1:
            cross_corr = correlation_matrix[:n_species, n_species:]
            r_squared = np.mean(cross_corr ** 2) if cross_corr.size > 0 else 0
        else:
            r_squared = 0

    # Permutation test for adjusted R-squared
    adj_r_squared = _permutation_cca_adjusted(Y, X, r_squared, n_perm, row_weights)

    return r_squared, adj_r_squared


def calculate_cca(dv: np.ndarray, iv: np.ndarray, type: str = "adjR2", n_perm: int = 1000) -> float:
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
    # Check if species data is appropriate for CCA
    if np.any(dv < 0):
        print("Warning: Negative values in species data. CCA typically requires non-negative abundance data.")

    try:
        r_squared, adj_r_squared = calculate_cca_r2_adj(dv, iv, n_perm)

        if type == "R2":
            return r_squared
        else:
            return adj_r_squared

    except Exception as e:
        print(f"CCA calculation failed: {e}. Falling back to RDA calculation.")
        # Fallback to RDA if CCA fails
        return calculate_rda(dv, iv, type)


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
        # Cailliez method: add constant to all elements
        n = distance_matrix.shape[0]

        # Create matrices for the Cailliez method
        A = -0.5 * distance_matrix ** 2
        I = np.eye(n)
        ones = np.ones((n, n))

        # Construct the matrix for the eigenvalue problem
        M = np.block([
            [A @ (I - ones / n), -I],
            [-(I - ones / n) @ A @ (I - ones / n), (I - ones / n) @ A]
        ])

        # Find eigenvalues
        eigenvalues = np.linalg.eigvals(M)
        real_eigenvalues = eigenvalues[np.isreal(eigenvalues)].real

        if len(real_eigenvalues) == 0:
            # Fallback to Lingoes method
            return euclidify_distance_matrix(distance_matrix, method="lingoes")

        constant = np.max(real_eigenvalues)

        if constant <= 0:
            # Already Euclidean
            return distance_matrix
        else:
            # Add constant
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

def _calculate_dbrda_fallback(dv_dist: np.ndarray, iv: np.ndarray, type: str = "adjR2") -> float:
    """
    Fallback method for db-RDA when PCoA fails

    Parameters
    ----------
    dv_dist : ndarray
        Distance matrix
    iv : ndarray
        Environmental variables
    type : str

    Returns
    -------
    float
        R-squared value
    """
    # Simple approach: use distance matrix directly with some transformation
    n_samples = dv_dist.shape[0]

    # Try to convert distance matrix to Euclidean space using simple method
    try:
        # Use classical multidimensional scaling
        mds = MDS(n_components=min(10, n_samples - 1), dissimilarity='precomputed', random_state=42)
        mds_scores = mds.fit_transform(dv_dist)

        # Use MDS scores as response variables
        r_squared, adj_r_squared = calculate_rda_r2_adj(mds_scores, iv, type)

        if type == "R2":
            return r_squared
        else:
            return adj_r_squared

    except Exception as e:
        print(f"db-RDA fallback also failed: {e}")
        # Last resort: return a reasonable default
        return 0.0


def calculate_dbrda(dv_dist: np.ndarray, iv: np.ndarray, type: str = "adjR2",
                    add: bool = False, sqrt_dist: bool = False, n_axes: int = None) -> float:
    """
    Calculate R-squared for db-RDA (Distance-based Redundancy Analysis)

    Parameters
    ----------
    dv_dist : ndarray
        Distance matrix (n_samples x n_samples)
    iv : ndarray
        Environmental variables (n_samples x predictors)
    type : str
        Type of R-squared: "R2" or "adjR2"
    add : bool
        Whether to add constant to make distance matrix Euclidean
    sqrt_dist : bool
        Whether to take square root of distances
    n_axes : int, optional
        Number of PCoA axes to use

    Returns
    -------
    float
        R-squared value
    """
    # Check if dv_dist is a valid distance matrix
    if not check_distance_matrix(dv_dist):
        raise ValueError("dv should be a square symmetric distance matrix for db-RDA")

    n_samples = dv_dist.shape[0]

    # Preprocess distance matrix
    distance_matrix = dv_dist.copy()

    # Take square root if requested
    if sqrt_dist:
        distance_matrix = np.sqrt(distance_matrix)

    # Make Euclidean if requested
    if add:
        distance_matrix = euclidify_distance_matrix(distance_matrix, method="lingoes")

    # Perform PCoA on the distance matrix
    try:
        pcoa_scores, eigenvalues = calculate_pcoa(distance_matrix, n_axes)

        # Use PCoA scores as response variables in RDA
        if pcoa_scores.shape[1] == 0:
            raise ValueError("PCoA produced no positive eigenvalues")

        # Calculate R-squared using RDA on PCoA scores
        r_squared, adj_r_squared = calculate_rda_r2_adj(pcoa_scores, iv, type)

        if type == "R2":
            return r_squared
        else:
            return adj_r_squared

    except Exception as e:
        print(f"db-RDA calculation failed: {e}")
        # Fallback to simple approach
        return _calculate_dbrda_fallback(dv_dist, iv, type)


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
        group_names = [f"Group_{i + 1}" for i in range(len(iv))]
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
            r2_value = calculate_cca(dv, combined_iv, type, n_perm=kwargs.get('n_perm', 1000))
            commonM[i, 1] = r2_value
        elif method.upper() in ["DBRDA"]:
            r2_value = calculate_dbrda(
                dv, combined_iv, type,
                add=kwargs.get('add', False),
                sqrt_dist=kwargs.get('sqrt_dist', False),
                n_axes=kwargs.get('n_axes', None)
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

    if total_individual <= 0:
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

    # If predictors are a mixed-type DataFrame, preprocess into encoded groups
    # and reuse the multi-group implementation so that categorical / ordered
    # factors remain one logical predictor group.
    ordered_factors = kwargs.get("ordered_factors", None)
    categorical_factors = kwargs.get("categorical_factors", None)

    if isinstance(iv, pd.DataFrame):
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

    n_samples, n_vars = iv.shape

    if n_samples <= n_vars:
        raise ValueError("sample size (row) is less than the number of predictors")

    if n_vars < 2:
        raise ValueError("Analysis not conducted. Insufficient number of predictors.")

    # Standardize if requested (for RDA)
    if method.upper() == "RDA" and scale:
        dv = (dv - np.mean(dv, axis=0)) / np.std(dv, axis=0)

    # Create variable names if not provided
    if hasattr(iv, 'columns'):
        var_names = iv.columns.tolist()
    else:
        var_names = [f"X{i + 1}" for i in range(n_vars)]

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
            # Use CCA calculation
            r2_value = calculate_cca(dv, subset_iv, type, n_perm=kwargs.get('n_perm', 1000))
            commonM[i, 1] = r2_value
        elif method.upper() in ["DBRDA"]:
            # Use db-RDA calculation
            r2_value = calculate_dbrda(
                dv, subset_iv, type,
                add=kwargs.get('add', False),
                sqrt_dist=kwargs.get('sqrt_dist', False),
                n_axes=kwargs.get('n_axes', None)
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

        # 修复除零错误
        percentage = safe_divide(commonM[idx, 2], totalRSquare) * 100
        outputcommonM[i, 1] = round(percentage, 2)  # 限制在0-100之间

    # 修复Total行
    outputcommonM[total_combinations, 0] = round(totalRSquare, 4)
    outputcommonM[total_combinations, 1] = 100.0

    # Create descriptive row names for variation partitioning
    ordered_binary_matrix = binary_matrix[:, order_indices]
    rowNames = get_combination_names(ordered_binary_matrix, var_names)

    # Ensure row names match the data shape
    if len(rowNames) != outputcommonM.shape[0]:
        # Create simple sequential names if there's a mismatch
        rowNames = [f"Combination_{i + 1}" for i in range(outputcommonM.shape[0] - 1)] + ["Total"]

    # Calculate variable importance (hierarchical partitioning)
    VariableImportance = np.zeros((n_vars, 4))

    for i in range(n_vars):
        # Calculate individual contribution (I)
        weights = binary_matrix[i, :] * (commonM[:, 2] / (bit_counts + 1e-10))
        individual_value = np.sum(weights)

        # 确保个体贡献非负
        VariableImportance[i, 2] = round(individual_value, 4)

    # Unique contributions are the single-variable combinations
    VariableImportance[:, 0] = outputcommonM[:n_vars, 0]  # Unique

    # Average shared contribution = Individual - Unique
    VariableImportance[:, 1] = VariableImportance[:, 2] - VariableImportance[:, 0]

    # 修复百分比计算
    total_individual = round(np.sum(VariableImportance[:, 2]), 3)

    if total_individual <= 0:
        # 如果总个体贡献为0或负，将所有百分比设为0
        VariableImportance[:, 3] = 0.0
    else:
        # 确保个体贡献非负，然后计算百分比
        percentages = 100 * VariableImportance[:, 2] / total_individual
        VariableImportance[:, 3] = np.round(percentages, 2)

    # 确保没有NaN或无限值
    VariableImportance = np.where(np.isfinite(VariableImportance), VariableImportance, np.nan)

    # Create pandas DataFrames for results
    hier_part_df = pd.DataFrame(
        VariableImportance,
        columns=["Unique", "Average.share", "Individual", "I.perc(%)"],
        index=var_names
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


def rdacca_hp(dv: Union[np.ndarray, pd.DataFrame],
              iv: Union[np.ndarray, pd.DataFrame, List, Dict],
              method: str = "RDA",
              type: str = "adjR2",
              scale: bool = False,
              var_part: bool = False,
              n_perm: int = 1000,  # For CCA
              add: bool = False,  # For db-RDA
              sqrt_dist: bool = False,  # For db-RDA
              n_axes: int = None,  # For db-RDA
              ordered_factors: dict | None = None,
              categorical_factors: list | None = None,
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
    add : bool
        Whether to add constant to make distance matrix Euclidean (for db-RDA)
    sqrt_dist : bool
        Whether to take square root of distances (for db-RDA)
    n_axes : int, optional
        Number of PCoA axes to use (for db-RDA)

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

    # Special check for db-RDA
    if method == "DBRDA":
        if not check_distance_matrix(dv):
            raise ValueError("For db-RDA, dv should be a square symmetric distance matrix")

    # Handle different types of iv input
    if isinstance(iv, (list, dict)):
        # Multiple predictor groups
        kwargs['ordered_factors'] = ordered_factors
        kwargs['categorical_factors'] = categorical_factors
        return _rdacca_hp_multi(dv, iv, method, type, scale, var_part, **kwargs)
    else:
        # Single predictor group
        # Pass additional parameters to the single group function
        kwargs['n_perm'] = n_perm
        kwargs['add'] = add
        kwargs['sqrt_dist'] = sqrt_dist
        kwargs['n_axes'] = n_axes
        kwargs['ordered_factors'] = ordered_factors
        kwargs['categorical_factors'] = categorical_factors

        return _rdacca_hp_single(dv, iv, method, type, scale, var_part, **kwargs)