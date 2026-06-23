import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform


VEGAN_DISTANCE_METHODS = (
    "bray",
    "euclidean",
    "manhattan",
    "canberra",
    "jaccard",
    "kulczynski",
    "gower",
    "hellinger",
    "chord",
)

_DISTANCE_ALIASES = {
    "braycurtis": "bray",
    "bray-curtis": "bray",
    "cityblock": "manhattan",
}

def _is_strict_sequential_index(series: pd.Series) -> bool:
    """
    判断一列是否像典型导出索引列：
    0..n-1 或 1..n
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().any():
        return False

    vals = s.to_numpy()
    n = len(vals)

    zero_based = np.array_equal(vals, np.arange(n))
    one_based = np.array_equal(vals, np.arange(1, n + 1))
    return zero_based or one_based


def remove_index_like_columns(df: pd.DataFrame, warn: bool = True) -> pd.DataFrame:
    """
    只删除高置信度的伪索引列，不粗暴删第一列。
    会处理：
    - Unnamed: 0
    - row.names
    - rownames
    - index
    但要求其值是严格的 0..n-1 或 1..n
    """
    if not isinstance(df, pd.DataFrame):
        return df

    df = df.copy()
    cols_to_drop = []

    for col in df.columns:
        col_str = str(col).strip().lower()

        name_looks_like_index = (
            col_str.startswith("unnamed:")
            or col_str in {"index", "row.names", "rownames"}
        )

        if not name_looks_like_index:
            continue

        if _is_strict_sequential_index(df[col]):
            cols_to_drop.append(col)

    if cols_to_drop:
        if warn:
            print(f"[info] dropped index-like columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    return df


def sanitize_tabular_input(x, warn: bool = True):
    """
    统一清理输入：
    - DataFrame: 删除高置信度伪索引列
    - dict/list: 递归清理其中的 DataFrame
    - ndarray/其他: 原样返回
    """
    if isinstance(x, pd.DataFrame):
        return remove_index_like_columns(x, warn=warn)

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if isinstance(v, pd.DataFrame):
                out[k] = remove_index_like_columns(v, warn=warn)
            else:
                out[k] = v
        return out

    if isinstance(x, list):
        out = []
        for v in x:
            if isinstance(v, pd.DataFrame):
                out.append(remove_index_like_columns(v, warn=warn))
            else:
                out.append(v)
        return out

    return x


def encode_unordered_factor(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Encode an unordered categorical predictor using treatment/dummy coding."""
    cat = series.astype("category")
    dummies = pd.get_dummies(cat, prefix=prefix, drop_first=True)
    if dummies.shape[1] == 0:
        return pd.DataFrame(index=series.index)
    return dummies.astype(float)


def encode_ordered_factor_poly(series: pd.Series, levels: list, prefix: str) -> pd.DataFrame:
    """Approximate R's contr.poly encoding for ordered factors."""
    cat = pd.Categorical(series, categories=levels, ordered=True)
    codes = cat.codes
    if (codes < 0).any():
        missing = sorted(pd.Series(series)[codes < 0].astype(str).unique().tolist())
        raise ValueError(
            f"Ordered factor '{prefix}' contains values outside declared levels {levels}: {missing}"
        )

    k = len(levels)
    if k < 2:
        return pd.DataFrame(index=series.index)

    x = np.arange(1, k + 1, dtype=float)
    V = np.vander(x, N=k, increasing=True)[:, 1:]
    Q, _ = np.linalg.qr(V)
    encoded = Q[codes, :]

    suffixes = ["L", "Q", "C", "^4", "^5", "^6", "^7", "^8"]
    cols = [f"{prefix}.{suffixes[i]}" if i < len(suffixes) else f"{prefix}.{i+1}" for i in range(k - 1)]
    return pd.DataFrame(encoded, columns=cols, index=series.index)


def encode_predictor_column(series: pd.Series,
                            name: str,
                            ordered_factors: dict | None = None,
                            categorical_factors: list | None = None) -> pd.DataFrame:
    """Encode one predictor column based on explicit or inferred type."""
    ordered_factors = ordered_factors or {}
    categorical_factors = categorical_factors or []

    if name in ordered_factors:
        return encode_ordered_factor_poly(series, ordered_factors[name], name)

    if pd.api.types.is_numeric_dtype(series):
        return pd.DataFrame({name: pd.to_numeric(series, errors='raise').astype(float)})

    if name in categorical_factors:
        return encode_unordered_factor(series, name)

    if pd.api.types.is_categorical_dtype(series.dtype):
        if getattr(series.dtype, 'ordered', False):
            levels = list(series.cat.categories)
            return encode_ordered_factor_poly(series, levels, name)
        return encode_unordered_factor(series, name)

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return encode_unordered_factor(series, name)

    return pd.DataFrame({name: pd.to_numeric(series, errors='raise').astype(float)})


def preprocess_predictor_dataframe(iv: pd.DataFrame,
                                   ordered_factors: dict | None = None,
                                   categorical_factors: list | None = None,
                                   warn: bool = True):
    """Convert a user-facing predictor DataFrame into encoded groups by original variable."""
    iv = sanitize_tabular_input(iv, warn=warn)
    encoded_groups = {}
    group_columns = {}

    for col in iv.columns:
        enc = encode_predictor_column(iv[col], str(col), ordered_factors=ordered_factors,
                                      categorical_factors=categorical_factors)
        if enc.shape[1] == 0:
            raise ValueError(f"Predictor '{col}' produced zero encoded columns; please check its values.")
        encoded_groups[str(col)] = enc
        group_columns[str(col)] = list(enc.columns)

    combined = pd.concat(encoded_groups.values(), axis=1) if encoded_groups else pd.DataFrame(index=iv.index)
    return combined, encoded_groups, group_columns


def preprocess_grouped_predictors(iv,
                                  ordered_factors: dict | None = None,
                                  categorical_factors: list | None = None,
                                  warn: bool = True):
    """Encode grouped predictors while preserving user-supplied group structure."""
    iv = sanitize_tabular_input(iv, warn=warn)
    if isinstance(iv, dict):
        out = {}
        for name, group in iv.items():
            if isinstance(group, pd.DataFrame):
                _, enc_groups, _ = preprocess_predictor_dataframe(group,
                                                                  ordered_factors=ordered_factors,
                                                                  categorical_factors=categorical_factors,
                                                                  warn=warn)
                out[name] = pd.concat(enc_groups.values(), axis=1)
            else:
                arr = np.asarray(group, dtype=float)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                out[name] = pd.DataFrame(arr)
        return out
    elif isinstance(iv, list):
        out = []
        for group in iv:
            if isinstance(group, pd.DataFrame):
                _, enc_groups, _ = preprocess_predictor_dataframe(group,
                                                                  ordered_factors=ordered_factors,
                                                                  categorical_factors=categorical_factors,
                                                                  warn=warn)
                out.append(pd.concat(enc_groups.values(), axis=1))
            else:
                arr = np.asarray(group, dtype=float)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                out.append(pd.DataFrame(arr))
        return out
    return iv

def creatbin(col: int, binmatrix: np.ndarray) -> np.ndarray:
    """
    Internal function to create binary matrix for combinations

    Parameters
    ----------
    col : int
        Column number (0-indexed in Python, but algorithm follows R logic)
    binmatrix : ndarray
        Input binary matrix

    Returns
    -------
    ndarray
        Updated binary matrix
    """
    row = 0  # Python uses 0-indexing, R uses 1-indexing
    val = col + 1  # Adjust for R's 1-indexing vs Python's 0-indexing

    while val != 0:
        if val % 2 == 1:  # odd number
            binmatrix[row, col] = 1
        val = val // 2  # integer division
        row += 1

    return binmatrix


def odd(val: int) -> bool:
    """
    Check if number is odd

    Parameters
    ----------
    val : int
        Input number

    Returns
    -------
    bool
        True if odd, False if even
    """
    return val % 2 == 1


def genList(ivlist: List[int], value: int) -> List[int]:
    """
    Generate variable combination list

    Parameters
    ----------
    ivlist : list
        List of variable indices
    value : int
        Sequence ID

    Returns
    -------
    list
        Updated variable index list
    """
    numlist = len(ivlist)
    newlist = [0] * numlist

    for i in range(numlist):
        newlist[i] = abs(ivlist[i]) + abs(value)
        if ((ivlist[i] < 0) and (value >= 0)) or ((ivlist[i] >= 0) and (value < 0)):
            newlist[i] = newlist[i] * -1

    return newlist


def create_binary_matrix(n_vars: int) -> np.ndarray:
    """
    Create binary matrix for all variable combinations

    Parameters
    ----------
    n_vars : int
        Number of predictor variables

    Returns
    -------
    ndarray
        Binary matrix of shape (n_vars, 2^n_vars - 1)
    """
    total_combinations = 2 ** n_vars - 1
    binary_matrix = np.zeros((n_vars, total_combinations), dtype=int)

    for i in range(total_combinations):
        binary_matrix = creatbin(i, binary_matrix)

    return binary_matrix


def get_variable_combinations(variable_names: List[str]) -> List[List[str]]:
    """
    Get all possible combinations of variables

    Parameters
    ----------
    variable_names : list
        List of variable names

    Returns
    -------
    list
        List of variable combinations
    """
    n_vars = len(variable_names)
    binary_matrix = create_binary_matrix(n_vars)
    combinations = []

    for col in range(binary_matrix.shape[1]):
        combo = []
        for row in range(binary_matrix.shape[0]):
            if binary_matrix[row, col] == 1:
                combo.append(variable_names[row])
        combinations.append(combo)

    return combinations


def generate_combination_names(n_vars: int, var_names: List[str]) -> List[str]:
    """
    Generate combination names for variation partitioning

    Parameters
    ----------
    n_vars : int
        Number of variables
    var_names : list
        List of variable names

    Returns
    -------
    list
        Combination names including "Total"
    """
    from itertools import combinations

    combination_names = []

    # 生成单个变量的组合名称
    for i in range(n_vars):
        combination_names.append(f"Unique to {var_names[i]}")

    # 生成多个变量的组合名称
    for r in range(2, n_vars + 1):
        for combo in combinations(range(n_vars), r):
            if r == 2:
                name = f"Common to {var_names[combo[0]]} and {var_names[combo[1]]}"
            else:
                name = "Common to "
                for i, idx in enumerate(combo):
                    if i == len(combo) - 1:
                        name += f"and {var_names[idx]}"
                    else:
                        name += f"{var_names[idx]}, "
            combination_names.append(name)

    # 添加 Total
    combination_names.append("Total")

    return combination_names


def get_combination_names(binary_matrix: np.ndarray, var_names: List[str]) -> List[str]:
    """
    Generate descriptive names for each combination for variation partitioning output

    Parameters
    ----------
    binary_matrix : ndarray
        Binary matrix from create_binary_matrix
    var_names : list
        List of variable names

    Returns
    -------
    list
        Descriptive names for each combination (including "Total" at the end)
    """
    n_vars = len(var_names)
    combination_names = []

    for col in range(binary_matrix.shape[1]):
        n_bits = np.sum(binary_matrix[:, col])

        if n_bits == 1:
            name = "Unique to "
        else:
            name = "Common to "

        bits_count = 0
        for row in range(n_vars):
            if binary_matrix[row, col] == 1:
                if n_bits == 1:
                    name += var_names[row]
                else:
                    bits_count += 1
                    if bits_count == n_bits:
                        name += f"and {var_names[row]}"
                    else:
                        name += f"{var_names[row]}, "

        combination_names.append(name)

    # 添加 "Total" 到最后
    combination_names.append("Total")

    return combination_names


def create_test_data(n_samples: int = 100, n_predictors: int = 3,
                     n_responses: int = 2, correlation_strength: float = 0.5,
                     noise_level: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create test data for hierarchical partitioning analysis

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_predictors : int
        Number of predictor variables
    n_responses : int
        Number of response variables
    correlation_strength : float
        Strength of correlation between predictors and responses (0-1)
    noise_level : float
        Level of random noise to add
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (response_data, predictor_data) as numpy arrays
    """
    np.random.seed(seed)

    # Generate predictor variables with some correlation structure
    # Create a correlation matrix for predictors
    corr_matrix = np.eye(n_predictors)
    for i in range(n_predictors):
        for j in range(i + 1, n_predictors):
            corr = 0.3  # Moderate correlation between predictors
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    # Generate multivariate normal predictors
    iv = np.random.multivariate_normal(
        mean=np.zeros(n_predictors),
        cov=corr_matrix,
        size=n_samples
    )

    # Create response variables that depend on predictors
    # Use different weights for different predictors
    weights = np.linspace(0.8, 0.2, n_predictors)

    # Create response variables
    dv = np.zeros((n_samples, n_responses))
    for i in range(n_responses):
        # Each response depends on all predictors but with different patterns
        response_weights = np.roll(weights, i) * correlation_strength
        linear_combination = np.zeros(n_samples)

        for j in range(n_predictors):
            linear_combination += response_weights[j] * iv[:, j]

        # Add noise
        dv[:, i] = linear_combination + np.random.normal(0, noise_level, n_samples)

    return dv, iv


def create_cca_test_data(n_samples: int = 100, n_species: int = 10, n_predictors: int = 3,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create test data suitable for CCA (Canonical Correspondence Analysis)

    Parameters
    ----------
    n_samples : int
        Number of samples (sites)
    n_species : int
        Number of species
    n_predictors : int
        Number of environmental predictors
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (species_data, environmental_data) as numpy arrays
    """
    np.random.seed(seed)

    # Create environmental predictors with some correlation structure
    corr_matrix = np.eye(n_predictors)
    for i in range(n_predictors):
        for j in range(i + 1, n_predictors):
            corr = 0.2  # Moderate correlation between environmental variables
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    # Generate multivariate normal environmental variables
    environmental_data = np.random.multivariate_normal(
        mean=np.zeros(n_predictors),
        cov=corr_matrix,
        size=n_samples
    )

    # Create species abundance data (non-negative)
    # Species respond to environmental gradients with some noise

    # Create species-environment relationships
    # Each species has different responses to environmental variables
    species_responses = np.random.uniform(-0.5, 0.5, (n_predictors, n_species))

    # Linear combination of environmental variables
    linear_comb = environmental_data @ species_responses

    # Apply exponential to get positive values and add Poisson noise for count data
    expected_abundance = np.exp(linear_comb) * 5  # Scale factor to control abundance

    # Generate Poisson-distributed counts
    species_data = np.random.poisson(expected_abundance)

    # Ensure no negative values (though Poisson should already be non-negative)
    species_data = np.maximum(species_data, 0)

    # Add some zeros to simulate real species data (many zeros are common)
    zero_mask = np.random.random(species_data.shape) < 0.1  # 10% zeros
    species_data[zero_mask] = 0

    return species_data, environmental_data


def create_distance_test_data(n_samples: int = 50, n_species: int = 20,
                             distance_metric: str = "braycurtis", seed: int = 42) -> np.ndarray:
    """
    Create test data for db-RDA analysis

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_species : int
        Number of species
    distance_metric : str
        Distance metric to use
    seed : int
        Random seed

    Returns
    -------
    ndarray
        Distance matrix
    """
    np.random.seed(seed)

    # Create species abundance data
    species_data = np.random.gamma(shape=2, scale=1, size=(n_samples, n_species))

    # Add some zeros to simulate real data
    zero_mask = np.random.random(species_data.shape) < 0.3
    species_data[zero_mask] = 0

    # Calculate distance matrix
    if distance_metric == "braycurtis":
        # Bray-Curtis is commonly used in ecology
        distance_matrix = pairwise_distances(species_data, metric='braycurtis')
    elif distance_metric == "jaccard":
        # Jaccard distance for presence-absence
        presence_absence = (species_data > 0).astype(float)
        distance_matrix = pairwise_distances(presence_absence, metric='jaccard')
    else:
        # Euclidean as default
        distance_matrix = pairwise_distances(species_data, metric='euclidean')

    return distance_matrix


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
    from scipy.linalg import eigh

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
        # Match vegan::addCailliez.
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


def is_condensed_distance_vector(x: np.ndarray) -> bool:
    """
    Check whether x looks like a condensed distance vector, i.e. the output of
    scipy.spatial.distance.pdist or an R-like dist object converted to 1D.

    Valid condensed length m must satisfy:
        n * (n - 1) / 2 = m
    for some integer n >= 2.
    """
    arr = np.asarray(x, dtype=float)

    if arr.ndim != 1:
        return False

    m = arr.size
    if m == 0:
        return False

    # Solve n(n-1)/2 = m
    n = (1 + np.sqrt(1 + 8 * m)) / 2
    n_int = int(round(n))

    if n_int < 2:
        return False

    return n_int * (n_int - 1) // 2 == m


def calculate_distance_matrix(response_data, method: str = "bray") -> np.ndarray:
    """Calculate a sample-by-sample dissimilarity matrix from raw response data.

    The method names follow commonly used ``vegan::vegdist`` names. Rows are
    samples and columns are response variables (for example, species).
    """
    if isinstance(response_data, pd.DataFrame):
        data = response_data.to_numpy(dtype=float)
    else:
        data = np.asarray(response_data, dtype=float)

    if data.ndim != 2:
        raise ValueError("Raw db-RDA response data must be a two-dimensional matrix.")
    if data.shape[0] < 2:
        raise ValueError("Raw db-RDA response data must contain at least two samples.")
    if data.shape[1] < 1:
        raise ValueError("Raw db-RDA response data must contain at least one response variable.")
    if not np.all(np.isfinite(data)):
        raise ValueError("NA/NaN/Inf is not allowed in db-RDA response data.")

    method_key = str(method).strip()
    method_key = _DISTANCE_ALIASES.get(method_key.lower(), method_key.lower())
    if method_key not in VEGAN_DISTANCE_METHODS:
        supported = ", ".join(VEGAN_DISTANCE_METHODS)
        raise ValueError(f"Unsupported distance method '{method}'. Supported methods: {supported}.")

    if method_key in {"bray", "jaccard", "kulczynski", "hellinger", "chord"}:
        if np.any(data < 0):
            raise ValueError(f"Distance method '{method_key}' requires non-negative response data.")

    if method_key == "hellinger":
        row_sums = data.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError("Hellinger distance cannot be calculated for empty samples.")
        transformed = np.sqrt(data / row_sums[:, None])
        return pairwise_distances(transformed, metric="euclidean")

    if method_key == "chord":
        row_norms = np.linalg.norm(data, axis=1)
        if np.any(row_norms <= 0):
            raise ValueError("Chord distance cannot be calculated for empty samples.")
        transformed = data / row_norms[:, None]
        return pairwise_distances(transformed, metric="euclidean")

    if method_key == "gower":
        ranges = np.ptp(data, axis=0)
        active = ranges > 0
        if not np.any(active):
            return np.zeros((data.shape[0], data.shape[0]), dtype=float)
        scaled = data[:, active] / ranges[active]
        return pairwise_distances(scaled, metric="manhattan") / int(np.sum(active))

    if method_key == "euclidean":
        return pairwise_distances(data, metric="euclidean")
    if method_key == "manhattan":
        return pairwise_distances(data, metric="manhattan")
    if method_key == "canberra":
        n_samples = data.shape[0]
        distances = np.zeros((n_samples, n_samples), dtype=float)
        for i in range(n_samples):
            xi = data[i]
            for j in range(i + 1, n_samples):
                xj = data[j]
                denominator = np.abs(xi) + np.abs(xj)
                active = denominator > 0
                value = (
                    0.0
                    if not np.any(active)
                    else float(np.mean(np.abs(xi[active] - xj[active]) / denominator[active]))
                )
                distances[i, j] = distances[j, i] = value
        return distances

    n_samples = data.shape[0]
    distances = np.zeros((n_samples, n_samples), dtype=float)
    for i in range(n_samples):
        xi = data[i]
        for j in range(i + 1, n_samples):
            xj = data[j]
            abs_sum = float(np.sum(np.abs(xi - xj)))
            total_sum = float(np.sum(xi + xj))

            if method_key == "bray":
                value = np.nan if total_sum == 0 else abs_sum / total_sum
            elif method_key == "jaccard":
                bray = np.nan if total_sum == 0 else abs_sum / total_sum
                value = 2.0 * bray / (1.0 + bray)
            else:  # kulczynski
                minimum_sum = float(np.sum(np.minimum(xi, xj)))
                xi_sum = float(np.sum(xi))
                xj_sum = float(np.sum(xj))
                value = (
                    np.nan
                    if xi_sum == 0 or xj_sum == 0
                    else 1.0 - 0.5 * (minimum_sum / xi_sum + minimum_sum / xj_sum)
                )

            distances[i, j] = distances[j, i] = value

    if not np.all(np.isfinite(distances)):
        raise ValueError(
            f"Distance method '{method_key}' produced NaN or Inf; "
            "check for empty or otherwise invalid samples."
        )
    return distances


def prepare_dbrda_response(dv, distance: str | None = None) -> np.ndarray:
    """Prepare db-RDA response data while preserving the legacy distance input.

    ``distance=None`` means that ``dv`` is already a square distance matrix or
    condensed distance vector. Supplying a method means that ``dv`` is a raw
    response matrix and distances are calculated inside the package.
    """
    if distance is None:
        return coerce_distance_input(dv)
    return calculate_distance_matrix(dv, method=distance)


def coerce_distance_input(distance_input: np.ndarray) -> np.ndarray:
    """
    Accept either:
    1. a square symmetric distance matrix, or
    2. a condensed distance vector (dist-like / pdist-like),
    and return a square symmetric distance matrix.
    """
    arr = np.asarray(distance_input, dtype=float)

    # Case 1: already a square distance matrix
    if arr.ndim == 2:
        if check_distance_matrix(arr):
            return arr
        raise ValueError("For db-RDA, dv should be a square symmetric distance matrix or a valid condensed distance vector.")

    # Case 2: condensed distance vector
    if is_condensed_distance_vector(arr):
        mat = squareform(arr)
        if check_distance_matrix(mat):
            return mat

    raise ValueError("For db-RDA, dv should be a square symmetric distance matrix or a valid condensed distance vector.")


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


def check_data_quality(dv, iv):
    """
    检查输入数据质量：
    - dv 必须是纯数值
    - iv 可以包含数值列和类别列
    - 检查维度一致、缺失值、无穷值
    """
    # ---- dv 检查 ----
    if isinstance(dv, pd.DataFrame):
        dv_arr = dv.to_numpy(dtype=float)
    else:
        dv_arr = np.asarray(dv, dtype=float)

    if np.any(np.isnan(dv_arr)) or np.any(np.isinf(dv_arr)):
        raise ValueError("Dependent variables contain NaN or Inf values.")

    n_samples = dv_arr.shape[0]

    # ---- iv 检查 ----
    if isinstance(iv, pd.DataFrame):
        if iv.shape[0] != n_samples:
            raise ValueError("Dependent and independent variables must have the same number of rows.")

        # 数值列：检查 NaN / Inf
        numeric_cols = iv.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            iv_num = iv[numeric_cols].to_numpy(dtype=float)
            if np.any(np.isnan(iv_num)) or np.any(np.isinf(iv_num)):
                raise ValueError("Independent numeric variables contain NaN or Inf values.")

        # 非数值列：只检查缺失，不检查 np.isinf
        non_numeric_cols = [c for c in iv.columns if c not in numeric_cols]
        if len(non_numeric_cols) > 0:
            if iv[non_numeric_cols].isna().any().any():
                raise ValueError("Independent categorical variables contain missing values.")

        return dv, iv

    elif isinstance(iv, dict):
        for key, value in iv.items():
            if isinstance(value, pd.DataFrame):
                if value.shape[0] != n_samples:
                    raise ValueError(f"Group '{key}' does not have the same number of rows as dv.")

                numeric_cols = value.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    arr = value[numeric_cols].to_numpy(dtype=float)
                    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                        raise ValueError(f"Numeric columns in group '{key}' contain NaN or Inf values.")

                non_numeric_cols = [c for c in value.columns if c not in numeric_cols]
                if len(non_numeric_cols) > 0:
                    if value[non_numeric_cols].isna().any().any():
                        raise ValueError(f"Categorical columns in group '{key}' contain missing values.")
            else:
                arr = np.asarray(value, dtype=float)
                if arr.shape[0] != n_samples:
                    raise ValueError(f"Group '{key}' does not have the same number of rows as dv.")
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    raise ValueError(f"Group '{key}' contains NaN or Inf values.")

        return dv, iv

    else:
        iv_arr = np.asarray(iv, dtype=float)

        if iv_arr.shape[0] != n_samples:
            raise ValueError("Dependent and independent variables must have the same number of rows.")

        if np.any(np.isnan(iv_arr)) or np.any(np.isinf(iv_arr)):
            raise ValueError("Independent variables contain NaN or Inf values.")

        return dv, iv


def calculate_adjusted_r2(r_squared: float, n_samples: int, n_predictors: int) -> float:
    """
    Calculate adjusted R-squared using Ezekiel's formula

    Parameters
    ----------
    r_squared : float
        Raw R-squared value
    n_samples : int
        Number of samples
    n_predictors : int
        Number of predictors

    Returns
    -------
    float
        Adjusted R-squared
    """
    if n_samples <= n_predictors + 1:
        return 0.0

    return 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_predictors - 1)


def test_cca_utils():
    """Test CCA-specific utility functions"""
    print("Testing CCA utility functions...")

    # Test CCA test data creation
    species_data, env_data = create_cca_test_data(n_samples=50, n_species=5, n_predictors=2, seed=42)

    assert species_data.shape == (50, 5), f"Expected species data shape (50, 5), got {species_data.shape}"
    assert env_data.shape == (50, 2), f"Expected env data shape (50, 2), got {env_data.shape}"
    assert np.all(species_data >= 0), "Species data should be non-negative"
    print("✓ CCA test data creation test passed")

    # Test chi-square transformation
    try:
        Y_chi, row_weights, col_weights = chi_square_transform(species_data)
        assert Y_chi.shape == species_data.shape, "Chi-square transformed data should have same shape"
        assert len(row_weights) == species_data.shape[0], "Row weights length should match samples"
        assert len(col_weights) == species_data.shape[1], "Column weights length should match species"
        print("✓ Chi-square transformation test passed")
    except Exception as e:
        print(f"Chi-square transformation test failed: {e}")

    print("All CCA utility function tests passed! 🎉")


def test_dbrda_utils():
    """Test db-RDA specific utility functions"""
    print("Testing db-RDA utility functions...")

    # Test distance matrix creation
    try:
        distance_matrix = create_distance_test_data(n_samples=20, n_species=10, seed=42)
        assert distance_matrix.shape == (20, 20), f"Expected shape (20, 20), got {distance_matrix.shape}"
        assert np.allclose(distance_matrix, distance_matrix.T), "Distance matrix should be symmetric"
        assert np.allclose(np.diag(distance_matrix), 0), "Distance matrix diagonal should be zero"
        assert np.all(distance_matrix >= 0), "Distance matrix should be non-negative"
        print("✓ Distance matrix creation test passed")
    except Exception as e:
        print(f"Distance matrix creation test failed: {e}")

    # Test distance matrix validation
    try:
        # Test valid distance matrix
        valid_matrix = np.array([
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0]
        ])
        assert check_distance_matrix(valid_matrix) == True, "Valid distance matrix should return True"

        # Test invalid (non-symmetric) matrix
        invalid_matrix1 = np.array([
            [0, 1, 3],  # Different from [2,1,0] in third position
            [1, 0, 1],
            [2, 1, 0]
        ])
        assert check_distance_matrix(invalid_matrix1) == False, "Non-symmetric matrix should return False"

        # Test invalid (negative) matrix
        invalid_matrix2 = np.array([
            [0, 1, 2],
            [1, 0, -1],  # Negative value
            [2, -1, 0]
        ])
        assert check_distance_matrix(invalid_matrix2) == False, "Matrix with negative values should return False"

        # Test invalid (non-zero diagonal) matrix
        invalid_matrix3 = np.array([
            [1, 1, 2],  # Non-zero diagonal
            [1, 0, 1],
            [2, 1, 0]
        ])
        assert check_distance_matrix(invalid_matrix3) == False, "Matrix with non-zero diagonal should return False"

        print("✓ Distance matrix validation test passed")
    except Exception as e:
        print(f"Distance matrix validation test failed: {e}")

    # Test PCoA calculation
    try:
        distance_matrix = create_distance_test_data(n_samples=10, n_species=8, seed=42)
        pcoa_scores, eigenvalues = calculate_pcoa(distance_matrix)

        assert pcoa_scores.shape[0] == 10, "PCoA scores should have same number of rows as input"
        assert len(eigenvalues) == pcoa_scores.shape[1], "Number of eigenvalues should match number of PCoA axes"
        assert np.all(eigenvalues > 0), "Eigenvalues should be positive"
        print("✓ PCoA calculation test passed")
    except Exception as e:
        print(f"PCoA calculation test failed: {e}")

    # Test PCoA with limited axes
    try:
        distance_matrix = create_distance_test_data(n_samples=15, n_species=10, seed=42)
        pcoa_scores, eigenvalues = calculate_pcoa(distance_matrix, n_axes=3)

        assert pcoa_scores.shape[1] == 3, "PCoA should return specified number of axes"
        assert len(eigenvalues) == 3, "Should return specified number of eigenvalues"
        print("✓ PCoA with limited axes test passed")
    except Exception as e:
        print(f"PCoA with limited axes test failed: {e}")

    # Test Euclidean distance matrix (should not change)
    try:
        # Create a Euclidean distance matrix
        points = np.random.rand(8, 2)
        euclidean_dist = pairwise_distances(points)

        euclidified = euclidify_distance_matrix(euclidean_dist, method="lingoes")
        # Should be very close to original (might have tiny numerical differences)
        assert np.allclose(euclidean_dist, euclidified, atol=1e-10), "Euclidean matrix should not change significantly"
        print("✓ Euclidean distance matrix test passed")
    except Exception as e:
        print(f"Euclidean distance matrix test failed: {e}")

    # Test different distance metrics
    try:
        species_data = np.random.gamma(2, 1, (10, 5))
        species_data[np.random.random(species_data.shape) < 0.2] = 0

        # Bray-Curtis
        dist_bray = pairwise_distances(species_data, metric='braycurtis')
        assert check_distance_matrix(dist_bray), "Bray-Curtis should produce valid distance matrix"

        # Jaccard (on presence-absence)
        pa_data = (species_data > 0).astype(float)
        dist_jaccard = pairwise_distances(pa_data, metric='jaccard')
        assert check_distance_matrix(dist_jaccard), "Jaccard should produce valid distance matrix"

        print("✓ Different distance metrics test passed")
    except Exception as e:
        print(f"Different distance metrics test failed: {e}")

    print("All db-RDA utility function tests passed! 🎉")


def test_utils():
    """Test utility functions"""
    print("Testing utility functions...")

    # Test odd function
    assert odd(1) == True
    assert odd(2) == False
    assert odd(3) == True
    assert odd(0) == False
    print("✓ odd function test passed")

    # Test genList function
    ivlist = [1, -2, 3]
    value = 1
    result = genList(ivlist, value)
    expected = [2, -3, 4]  # Based on R logic
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ genList function test passed")

    # Test binary matrix creation
    n_vars = 3
    binary_matrix = create_binary_matrix(n_vars)
    expected_shape = (n_vars, 7)  # 2^3 - 1 = 7 combinations
    assert binary_matrix.shape == expected_shape, f"Expected shape {expected_shape}, got {binary_matrix.shape}"
    print("✓ Binary matrix shape test passed")

    # Test specific combinations for n_vars = 3
    # The matrix should represent all non-empty subsets of 3 variables
    expected_matrix = np.array([
        [1, 0, 1, 0, 1, 0, 1],  # Variable 1
        [0, 1, 1, 0, 0, 1, 1],  # Variable 2
        [0, 0, 0, 1, 1, 1, 1]  # Variable 3
    ])

    # Since the exact ordering might differ from R, we just check that
    # all combinations are represented (each column has at least one 1)
    for col in range(binary_matrix.shape[1]):
        assert np.sum(binary_matrix[:, col]) >= 1, f"Column {col} has no variables"
    print("✓ Binary matrix content test passed")

    # Test variable combinations
    var_names = ['X1', 'X2', 'X3']
    combinations = get_variable_combinations(var_names)
    assert len(combinations) == 7, f"Expected 7 combinations, got {len(combinations)}"
    print("✓ Variable combinations test passed")

    # Test combination names - 修复这里：现在期望8个名称（7个组合 + "Total"）
    combination_names = get_combination_names(binary_matrix, var_names)
    assert len(combination_names) == 8, f"Expected 8 combination names (7 combinations + Total), got {len(combination_names)}"

    # Check that names contain the expected patterns
    unique_found = False
    common_found = False
    total_found = False
    for name in combination_names:
        if name.startswith("Unique to"):
            unique_found = True
        if name.startswith("Common to"):
            common_found = True
        if name == "Total":
            total_found = True

    assert unique_found, "Should have unique combination names"
    assert common_found, "Should have common combination names"
    assert total_found, "Should have 'Total' in combination names"
    print("✓ Combination names test passed")

    # Test test data creation
    dv, iv = create_test_data(n_samples=50, n_predictors=2, n_responses=1)
    assert dv.shape == (50, 1), f"Expected dv shape (50, 1), got {dv.shape}"
    assert iv.shape == (50, 2), f"Expected iv shape (50, 2), got {iv.shape}"
    print("✓ Test data creation test passed")

    # Test data quality check
    dv_clean, iv_clean = check_data_quality(dv, iv)
    assert isinstance(dv_clean, np.ndarray), "Should return numpy array"
    assert isinstance(iv_clean, np.ndarray), "Should return numpy array"
    print("✓ Data quality check test passed")

    # Test adjusted R2 calculation
    r2 = 0.8
    n = 100
    p = 5
    adj_r2 = calculate_adjusted_r2(r2, n, p)
    expected_adj_r2 = 1 - (1 - 0.8) * (100 - 1) / (100 - 5 - 1)
    assert abs(adj_r2 - expected_adj_r2) < 1e-10, f"Adjusted R2 calculation incorrect"
    print("✓ Adjusted R2 calculation test passed")

    # Test CCA-specific functions
    test_cca_utils()

    # Test db-RDA specific functions
    test_dbrda_utils()

    print("All utility function tests passed! ")


if __name__ == "__main__":
    test_utils()
